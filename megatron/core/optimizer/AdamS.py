# coding=utf-8
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AdamS optimizer with optional fused Triton kernel.

AdamS uses previous momentum for variance:
    v_t = beta2 * m_{t-1}^2 + (1 - beta2) * g_t^2

This implementation intentionally does not persist `exp_avg_sq` in optimizer
state. The variance term is computed on-the-fly each step.
"""

import math
import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


_TRITON_AVAILABLE = triton is not None
_TRITON_SUPPORTED_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
)
_TRITON_LIBCUDA_FIX_ATTEMPTED = False
_TRITON_LIBCUDA_FIX_DIR = None
_TRITON_CACHE_FIX_ATTEMPTED = False
_TRITON_CACHE_DIR = None


def _prepend_env_path(var_name, value):
    if not value:
        return
    cur = os.environ.get(var_name, "")
    if not cur:
        os.environ[var_name] = value
        return
    parts = [p for p in cur.split(":") if p]
    if value in parts:
        return
    os.environ[var_name] = f"{value}:{cur}"


def _prepare_triton_cache_dir():
    global _TRITON_CACHE_FIX_ATTEMPTED
    global _TRITON_CACHE_DIR

    if _TRITON_CACHE_FIX_ATTEMPTED:
        return _TRITON_CACHE_DIR
    _TRITON_CACHE_FIX_ATTEMPTED = True

    existing = os.environ.get("TRITON_CACHE_DIR")
    if existing:
        os.makedirs(existing, exist_ok=True)
        _TRITON_CACHE_DIR = existing
        return existing

    uid = "nouid"
    if hasattr(os, "getuid"):
        try:
            uid = str(os.getuid())
        except OSError:
            uid = "nouid"
    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    rank = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))
    cache_dir = f"/tmp/triton_cache_u{uid}_j{job_id}_r{rank}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    _TRITON_CACHE_DIR = cache_dir
    return cache_dir


def _prepare_triton_libcuda_symlink():
    _prepare_triton_cache_dir()
    global _TRITON_LIBCUDA_FIX_ATTEMPTED
    global _TRITON_LIBCUDA_FIX_DIR

    if _TRITON_LIBCUDA_FIX_ATTEMPTED:
        return _TRITON_LIBCUDA_FIX_DIR
    _TRITON_LIBCUDA_FIX_ATTEMPTED = True

    candidates = [
        "/usr/local/cuda-12.8/compat/lib.real/libcuda.so.1",
        "/usr/local/cuda/compat/lib/libcuda.so.1",
        "/usr/local/cuda-13.1/compat/lib.real/libcuda.so.1",
        "/usr/local/cuda/compat/lib.real/libcuda.so.1",
    ]
    source = None
    for path in candidates:
        if os.path.exists(path):
            source = path
            break

    if source is None:
        _TRITON_LIBCUDA_FIX_DIR = None
        return None

    fix_dir = "/tmp/triton_libcuda_fix"
    os.makedirs(fix_dir, exist_ok=True)

    link_so1 = os.path.join(fix_dir, "libcuda.so.1")
    link_so = os.path.join(fix_dir, "libcuda.so")

    for link_path, target in ((link_so1, source), (link_so, source)):
        if os.path.islink(link_path) or os.path.exists(link_path):
            try:
                os.unlink(link_path)
            except OSError:
                pass
        if not os.path.exists(link_path):
            os.symlink(target, link_path)

    source_dir = os.path.dirname(source)
    _prepend_env_path("LD_LIBRARY_PATH", fix_dir)
    _prepend_env_path("LIBRARY_PATH", fix_dir)
    _prepend_env_path("LD_LIBRARY_PATH", source_dir)
    _prepend_env_path("LIBRARY_PATH", source_dir)
    os.environ["TRITON_LIBCUDA_PATH"] = fix_dir
    _TRITON_LIBCUDA_FIX_DIR = fix_dir
    return fix_dir


def _as_float(name: str, value):
    """Return a scalar float from Python number or 1-element tensor."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f'{name} must be a scalar tensor, got shape {tuple(value.shape)}')
        return float(value.item())
    return float(value)


if _TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        ],
        key=['numel'],
    )
    @triton.jit
    def _adams_fused_step_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        numel,
        beta1,
        one_minus_beta1,
        beta2,
        one_minus_beta2,
        lr,
        step_size,
        inv_sqrt_bias2,
        eps,
        weight_decay,
        APPLY_WEIGHT_DECAY: tl.constexpr,
        MAXIMIZE: tl.constexpr,
        ADAM_W_MODE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        param = tl.load(param_ptr + offsets, mask=mask, other=0)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0)
        exp_avg_prev = tl.load(exp_avg_ptr + offsets, mask=mask, other=0)

        param_f32 = param.to(tl.float32)
        grad_f32 = grad.to(tl.float32)
        exp_avg_prev_f32 = exp_avg_prev.to(tl.float32)

        if MAXIMIZE:
            grad_f32 = -grad_f32

        if APPLY_WEIGHT_DECAY and not ADAM_W_MODE:
            grad_f32 = grad_f32 + weight_decay * param_f32

        variance = exp_avg_prev_f32 * exp_avg_prev_f32
        variance = variance * beta2 + (grad_f32 * grad_f32) * one_minus_beta2

        exp_avg_new = exp_avg_prev_f32 * beta1 + grad_f32 * one_minus_beta1

        if APPLY_WEIGHT_DECAY and ADAM_W_MODE:
            param_f32 = param_f32 * (1.0 - lr * weight_decay)

        denom = tl.sqrt(variance)
        denom = denom * inv_sqrt_bias2 + eps
        param_new = param_f32 - step_size * (exp_avg_new / denom)

        tl.store(exp_avg_ptr + offsets, exp_avg_new.to(exp_avg_prev.dtype), mask=mask)
        tl.store(param_ptr + offsets, param_new.to(param.dtype), mask=mask)


def _adams_step_triton(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    step_size: float,
    inv_sqrt_bias2: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    adam_w_mode: bool,
) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    if (param.device.type != 'cuda' or
            grad.device != param.device or
            exp_avg.device != param.device):
        return False

    _prepare_triton_libcuda_symlink()

    if (param.dtype not in _TRITON_SUPPORTED_DTYPES or
            grad.dtype not in _TRITON_SUPPORTED_DTYPES or
            exp_avg.dtype not in _TRITON_SUPPORTED_DTYPES):
        return False
    if (not param.is_contiguous() or
            not grad.is_contiguous() or
            not exp_avg.is_contiguous()):
        return False

    numel = int(param.numel())
    if numel == 0:
        return True
    if grad.numel() != numel or exp_avg.numel() != numel:
        return False

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    _adams_fused_step_kernel[grid](
        param,
        grad,
        exp_avg,
        numel,
        float(beta1),
        1.0 - float(beta1),
        float(beta2),
        1.0 - float(beta2),
        float(lr),
        float(step_size),
        float(inv_sqrt_bias2),
        float(eps),
        float(weight_decay),
        APPLY_WEIGHT_DECAY=bool(weight_decay != 0.0),
        MAXIMIZE=bool(maximize),
        ADAM_W_MODE=bool(adam_w_mode),
    )
    return True


def _adams_step_torch(
    param: torch.Tensor,
    grad_data: torch.Tensor,
    exp_avg: torch.Tensor,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    step_size: float,
    inv_sqrt_bias2: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    adam_w_mode: bool,
):
    if maximize:
        grad_data = -grad_data

    if weight_decay != 0.0:
        if adam_w_mode:
            param.add_(param, alpha=-lr * weight_decay)
        else:
            grad_data = grad_data.add(param, alpha=weight_decay)

    variance = exp_avg.mul(exp_avg)
    variance.mul_(beta2)
    variance.addcmul_(grad_data, grad_data, value=1.0 - beta2)

    exp_avg.mul_(beta1).add_(grad_data, alpha=1.0 - beta1)

    denom = variance.sqrt()
    if inv_sqrt_bias2 != 1.0:
        denom.mul_(inv_sqrt_bias2)
    denom.add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)


class AdamS(torch.optim.Optimizer):
    """AdamS optimizer with fused Triton path and no persistent exp_avg_sq."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        maximize: bool = False,
        adam_w_mode: bool = True,
        bias_correction: bool = True,
        set_grad_none: bool = True,
    ):
        lr_value = _as_float('lr', lr)
        if lr_value < 0.0:
            raise ValueError(f"Invalid learning rate: {lr_value}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            adam_w_mode=adam_w_mode,
            bias_correction=bias_correction,
        )
        super().__init__(params, defaults)
        self.set_grad_none = set_grad_none

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('adam_w_mode', True)
            group.setdefault('bias_correction', True)
        for param_state in self.state.values():
            if isinstance(param_state, dict):
                param_state.pop('exp_avg_sq', None)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for param_state in self.state.values():
            if isinstance(param_state, dict):
                param_state.pop('exp_avg_sq', None)

    def zero_grad(self, set_to_none: Optional[bool] = None):
        if set_to_none is None:
            set_to_none = self.set_grad_none
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = _as_float('lr', group['lr'])
            eps = _as_float('eps', group['eps'])
            weight_decay = _as_float('weight_decay', group['weight_decay'])
            maximize = bool(group.get('maximize', False))
            adam_w_mode = bool(group.get('adam_w_mode', True))
            use_bias_correction = bool(group.get('bias_correction', True))

            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError('AdamS does not support sparse gradients')

                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                elif ('exp_avg' not in state or
                      state['exp_avg'].shape != param.shape or
                      state['exp_avg'].device != param.device):
                    state['exp_avg'] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                state.pop('exp_avg_sq', None)

                prev_step = state.get('step', 0)
                if torch.is_tensor(prev_step):
                    prev_step = int(prev_step.item())
                step = int(prev_step) + 1
                state['step'] = step
                exp_avg = state['exp_avg']

                if use_bias_correction:
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    step_size = lr / bias_correction1
                    inv_sqrt_bias2 = 1.0 / math.sqrt(bias_correction2)
                else:
                    step_size = lr
                    inv_sqrt_bias2 = 1.0

                grad_data = grad.detach()
                triton_applied = _adams_step_triton(
                    param,
                    grad_data,
                    exp_avg,
                    beta1=beta1,
                    beta2=beta2,
                    lr=lr,
                    step_size=step_size,
                    inv_sqrt_bias2=inv_sqrt_bias2,
                    eps=eps,
                    weight_decay=weight_decay,
                    maximize=maximize,
                    adam_w_mode=adam_w_mode,
                )
                if not triton_applied:
                    _adams_step_torch(
                        param,
                        grad_data,
                        exp_avg,
                        beta1=beta1,
                        beta2=beta2,
                        lr=lr,
                        step_size=step_size,
                        inv_sqrt_bias2=inv_sqrt_bias2,
                        eps=eps,
                        weight_decay=weight_decay,
                        maximize=maximize,
                        adam_w_mode=adam_w_mode,
                    )

        return loss


__all__ = ['AdamS']
