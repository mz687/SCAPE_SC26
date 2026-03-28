# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
import math
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

from .. import tensor_parallel
from ..transformer.module import param_is_not_shared
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBuffer

logger = logging.getLogger(__name__)

_TRITON_AVAILABLE = triton is not None and tl is not None
_TRITON_MASK_PACK_BLOCK = 512

if _TRITON_AVAILABLE:

    @triton.jit
    def _mask_pack_1bit_kernel(
        mask_u8_ptr,
        packed_ptr,
        nbits,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        byte_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        nbytes = (nbits + 7) // 8
        byte_mask = byte_offsets < nbytes

        bit_offsets = byte_offsets[:, None] * 8 + tl.arange(0, 8)[None, :]
        bit_mask = bit_offsets < nbits
        bits = tl.load(mask_u8_ptr + bit_offsets, mask=bit_mask, other=0).to(tl.int32)
        shifts = tl.arange(0, 8).to(tl.int32)[None, :]
        packed_vals = tl.sum(bits << shifts, axis=1)
        tl.store(packed_ptr + byte_offsets, packed_vals.to(tl.uint8), mask=byte_mask)

    @triton.jit
    def _mask_unpack_1bit_kernel(
        packed_ptr,
        mask_u8_ptr,
        nbits,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        byte_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        nbytes = (nbits + 7) // 8
        byte_mask = byte_offsets < nbytes
        packed_vals = tl.load(packed_ptr + byte_offsets, mask=byte_mask, other=0).to(tl.int32)
        shifts = tl.arange(0, 8).to(tl.int32)[None, :]
        bits = (packed_vals[:, None] >> shifts) & 1

        bit_offsets = byte_offsets[:, None] * 8 + tl.arange(0, 8)[None, :]
        bit_mask = bit_offsets < nbits
        tl.store(mask_u8_ptr + bit_offsets, bits.to(tl.uint8), mask=bit_mask)


def _launch_mask_pack_triton(mask_u8: torch.Tensor, packed_u8: torch.Tensor) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable for mask pack")
    nbits = int(mask_u8.numel())
    nbytes = (nbits + 7) // 8
    if int(packed_u8.numel()) != nbytes:
        raise RuntimeError(
            f"Mask pack size mismatch: expected {nbytes} bytes, got {int(packed_u8.numel())}."
        )
    if nbytes <= 0:
        return
    grid = (triton.cdiv(nbytes, _TRITON_MASK_PACK_BLOCK),)
    _mask_pack_1bit_kernel[grid](
        mask_u8,
        packed_u8,
        nbits,
        BLOCK_SIZE=_TRITON_MASK_PACK_BLOCK,
    )


def _launch_mask_unpack_triton(
    packed_u8: torch.Tensor,
    mask_u8: torch.Tensor,
    nbits: int,
) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable for mask unpack")
    nbits = int(nbits)
    nbytes = (nbits + 7) // 8
    if int(packed_u8.numel()) != nbytes:
        raise RuntimeError(
            f"Mask unpack size mismatch: expected {nbytes} bytes, got {int(packed_u8.numel())}."
        )
    if nbytes <= 0:
        return
    grid = (triton.cdiv(nbytes, _TRITON_MASK_PACK_BLOCK),)
    _mask_unpack_1bit_kernel[grid](
        packed_u8,
        mask_u8,
        nbits,
        BLOCK_SIZE=_TRITON_MASK_PACK_BLOCK,
    )


@dataclass
class _ParamSlice:
    start: int
    end: int
    param: torch.nn.Parameter
    owner: int
    exclude_from_topk: bool = False


@dataclass
class _BufferState:
    residual: torch.Tensor
    current_mask_u8: torch.Tensor
    next_mask_u8: torch.Tensor
    mask_pack_segments: Tuple[Tuple[int, int, int, int, int], ...] = tuple()
    current_mask_packed_u8: Optional[torch.Tensor] = None
    next_mask_packed_u8: Optional[torch.Tensor] = None
    has_current_mask: bool = False
    has_next_mask: bool = False
    next_mask_allreduce_handle: Optional[Any] = None
    next_mask_allreduce_uses_packed: bool = False


@dataclass
class _PendingUpdate:
    buffer_idx: int
    start: int
    end: int
    param: torch.nn.Parameter
    beta1: float
    beta2: float
    eps: float
    bias_correction: bool
    weight_decay: float


@dataclass
class _SparseSlicePlan:
    start: int
    end: int
    param: torch.nn.Parameter
    beta1: float
    beta2: float
    eps: float
    bias_correction: bool
    weight_decay: float
    payload_offset: int
    payload_length: int
    layer_profile: Dict[str, Any]


class TopKPerLayerSyncMomentumAdamSFP8ReducerV2:
    """MCore variant of TopKPerLayerSyncMomentumAdamSFP8ReducerV2.

    This reducer performs one-step delayed per-parameter top-k synchronization on
    AdamS momentum (exp_avg), optionally quantizing sparse payloads to FP8.

    It writes the final AdamS update metric into grad buffers, so the wrapped
    optimizer is expected to be SGD(momentum=0, weight_decay=0).
    """

    def __init__(
        self,
        buffers: List[_ParamAndGradBuffer],
        ddp_config: DistributedDataParallelConfig,
    ) -> None:
        self._buffers = list(buffers)
        self._ddp_config = ddp_config

        self._target_density = float(ddp_config.topk_adams_density)
        self._density_start = float(ddp_config.topk_adams_density_start)
        self._warmup_steps = max(0, int(ddp_config.topk_adams_density_warmup_steps))
        self._cooldown_steps = max(0, int(ddp_config.topk_adams_density_cooldown_steps))
        self._cooldown_start_step = int(ddp_config.topk_adams_density_cooldown_start_step)
        self._start_iter = max(0, int(ddp_config.topk_adams_start_iter))
        self._use_fp8_topk_quant = bool(ddp_config.use_fp8_topk_quant)
        self._move_clip_grad_to_reducer = bool(ddp_config.move_clip_grad_to_reducer)
        self._use_exclude_from_topk = bool(
            getattr(ddp_config, 'topk_adams_use_exclude_from_topk', True)
        )

        self._fp8_dtype = torch.float8_e5m2
        self._fp8_scale_eps = 1e-6
        self._fp8_scale_interval = 100
        self._fp8_allreduce_warned = False
        self._layer_fp8_scales: Dict[Tuple[int, int, int], float] = {}
        self._packed_fp8_payload_scale: float = 1.0
        self._last_fp8_scale_update_iter = -1

        self._prepared_iteration: Optional[int] = None
        self._async_mask_stream: Optional[torch.cuda.Stream] = None

        self._profile_enabled = self._env_flag('MEGATRON_TOPK_REDUCER_PROFILE', default=False)
        self._profile_sync_cuda = self._env_flag(
            'MEGATRON_TOPK_REDUCER_PROFILE_SYNC_CUDA',
            default=True,
        )
        self._profile_rank0_only = self._env_flag(
            'MEGATRON_TOPK_REDUCER_PROFILE_RANK0_ONLY',
            default=True,
        )
        self._profile_log_interval = max(
            1, int(os.getenv('MEGATRON_TOPK_REDUCER_PROFILE_LOG_INTERVAL', '1'))
        )
        self._profile_top_layers = max(
            0, int(os.getenv('MEGATRON_TOPK_REDUCER_PROFILE_TOP_LAYERS', '8'))
        )
        self._profile_step_ms: Dict[str, float] = {}
        self._profile_step_calls: Dict[str, int] = {}
        self._profile_step_values: Dict[str, float] = {}
        self._profile_total_ms: Dict[str, float] = {}
        self._profile_total_calls: Dict[str, int] = {}
        self._profile_total_values: Dict[str, float] = {}
        self._profile_steps = 0
        self._profile_layer_records: List[Dict[str, Any]] = []

        self._buffer_param_slices: List[List[_ParamSlice]] = []
        self._buffer_states: List[_BufferState] = []
        self._buffer_group_ranks: List[int] = []

        if torch.cuda.is_available():
            try:
                self._async_mask_stream = torch.cuda.Stream(
                    device=torch.cuda.current_device()
                )
            except Exception:
                self._async_mask_stream = torch.cuda.Stream()

        for buffer in self._buffers:
            grad_data = buffer.grad_data
            numel = int(grad_data.numel())
            group = buffer.data_parallel_group
            group_size = max(1, self._group_size(group))
            group_rank = self._group_rank(group)
            self._buffer_group_ranks.append(group_rank)

            sorted_slices: List[Tuple[int, int, torch.nn.Parameter]] = []
            param_to_name = getattr(buffer, 'param_to_name', {})
            for param in buffer.params:
                start, end, _ = buffer.param_index_map[param]
                sorted_slices.append((int(start), int(end), param))
            sorted_slices.sort(key=lambda item: item[0])
            param_slices: List[_ParamSlice] = []
            total_slices = len(sorted_slices)
            if total_slices > 0:
                base = total_slices // group_size
                rem = total_slices % group_size
                cursor = 0
                for owner in range(group_size):
                    count = base + (1 if owner < rem else 0)
                    for _ in range(count):
                        if cursor >= total_slices:
                            break
                        start, end, param = sorted_slices[cursor]
                        param_name = param_to_name.get(param, '')
                        exclude_from_topk = (
                            self._use_exclude_from_topk
                            and self._should_exclude_name_from_topk(param_name)
                        )
                        param_slices.append(
                            _ParamSlice(
                                start=start,
                                end=end,
                                param=param,
                                owner=int(owner),
                                exclude_from_topk=exclude_from_topk,
                            )
                        )
                        cursor += 1
            self._buffer_param_slices.append(param_slices)

            mask_pack_segments, packed_mask_numel = self._build_mask_pack_layout(
                numel=numel,
                param_slices=param_slices,
            )
            state = _BufferState(
                residual=torch.zeros(numel, dtype=torch.float32, device=grad_data.device),
                current_mask_u8=torch.zeros(numel, dtype=torch.uint8, device=grad_data.device),
                next_mask_u8=torch.zeros(numel, dtype=torch.uint8, device=grad_data.device),
                mask_pack_segments=mask_pack_segments,
                current_mask_packed_u8=torch.zeros(
                    packed_mask_numel,
                    dtype=torch.uint8,
                    device=grad_data.device,
                ),
                next_mask_packed_u8=torch.zeros(
                    packed_mask_numel,
                    dtype=torch.uint8,
                    device=grad_data.device,
                ),
            )
            self._buffer_states.append(state)

        self._param_to_group: Dict[torch.nn.Parameter, Dict[str, Any]] = {}
        self._param_to_state_dict: Dict[
            torch.nn.Parameter, Dict[torch.nn.Parameter, Dict[str, Any]]
        ] = {}
        self._param_to_optim_param: Dict[torch.nn.Parameter, torch.Tensor] = {}
        self._warned_missing_mapping = False
        self._last_synced_grad_norm: Optional[float] = None
        self._last_update_metric_norm: Optional[float] = None
        self._last_norm_step: Optional[int] = None
        self._clip_grad_max_norm: float = 0.0
        self._grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None

    @staticmethod
    def _iter_optimizer_wrappers(optimizer: Any) -> List[Any]:
        wrappers = getattr(optimizer, 'chained_optimizers', None)
        if wrappers is None:
            return [optimizer]
        return list(wrappers)

    @staticmethod
    def _group_size(group: torch.distributed.ProcessGroup) -> int:
        try:
            return int(group.size())
        except Exception:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return int(torch.distributed.get_world_size(group=group))
            return 1

    @staticmethod
    def _group_rank(group: torch.distributed.ProcessGroup) -> int:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return 0
        try:
            return int(torch.distributed.get_rank(group=group))
        except Exception:
            return 0

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name, None)
        if raw is None:
            return bool(default)
        return raw.strip().lower() in ('1', 'true', 'yes', 'on', 'y', 't')

    def _profile_should_log(self) -> bool:
        if not self._profile_enabled:
            return False
        if (
            self._profile_rank0_only
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        ):
            return False
        return True

    def _profile_reset_step(self) -> None:
        if not self._profile_enabled:
            return
        self._profile_step_ms.clear()
        self._profile_step_calls.clear()
        self._profile_step_values.clear()
        self._profile_layer_records.clear()

    def _profile_add_ms(self, key: str, elapsed_ms: float, calls: int = 1) -> None:
        if not self._profile_enabled:
            return
        self._profile_step_ms[key] = self._profile_step_ms.get(key, 0.0) + float(elapsed_ms)
        self._profile_step_calls[key] = self._profile_step_calls.get(key, 0) + int(calls)
        self._profile_total_ms[key] = self._profile_total_ms.get(key, 0.0) + float(elapsed_ms)
        self._profile_total_calls[key] = self._profile_total_calls.get(key, 0) + int(calls)

    def _profile_add_value(self, key: str, value: float) -> None:
        if not self._profile_enabled:
            return
        self._profile_step_values[key] = self._profile_step_values.get(key, 0.0) + float(value)
        self._profile_total_values[key] = self._profile_total_values.get(key, 0.0) + float(value)

    def _profile_add_layer_record(self, record: Dict[str, Any]) -> None:
        if not self._profile_enabled:
            return
        self._profile_layer_records.append(record)

    def _profile_tic(self, tensor: Optional[torch.Tensor] = None) -> Optional[float]:
        if not self._profile_enabled:
            return None
        if self._profile_sync_cuda and tensor is not None and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        return time.perf_counter()

    def _profile_toc(
        self,
        key: str,
        t0: Optional[float],
        tensor: Optional[torch.Tensor] = None,
        calls: int = 1,
    ) -> float:
        if t0 is None or not self._profile_enabled:
            return 0.0
        if self._profile_sync_cuda and tensor is not None and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._profile_add_ms(key, elapsed_ms, calls=calls)
        return elapsed_ms

    def _log_profile_step(
        self,
        train_iter: int,
        current_density: float,
        next_density: float,
        dense_mode: bool,
        use_fp8_quantized_payload: bool,
        total_selected: int,
        total_numel: int,
    ) -> None:
        if not self._profile_enabled:
            return
        self._profile_steps += 1
        if not self._profile_should_log():
            return
        if int(train_iter) % self._profile_log_interval != 0:
            return

        effective_density = (
            float(total_selected) / float(total_numel) if total_numel > 0 else 0.0
        )

        sorted_step_ms = sorted(
            self._profile_step_ms.items(), key=lambda item: item[1], reverse=True
        )
        timing_parts: List[str] = []
        for key, total_ms in sorted_step_ms:
            calls = max(1, int(self._profile_step_calls.get(key, 0)))
            timing_parts.append(
                f"{key}={total_ms:.3f}ms(calls={calls},avg={total_ms / calls:.3f}ms)"
            )

        value_parts: List[str] = []
        for key, value in sorted(self._profile_step_values.items()):
            value_parts.append(f"{key}={value:.3f}")

        msg = (
            f"{self.__class__.__name__} profile step={int(train_iter)} "
            f"mode={'dense' if dense_mode else 'sparse'} "
            f"current_density={current_density:.6f} "
            f"next_density={next_density:.6f} "
            f"effective_density={effective_density:.6f} "
            f"use_fp8_payload={bool(use_fp8_quantized_payload)}"
        )
        if timing_parts:
            msg += " timings: " + " | ".join(timing_parts)
        if value_parts:
            msg += " values: " + " | ".join(value_parts)
        logger.info(msg)

        if self._profile_top_layers <= 0 or not self._profile_layer_records:
            return

        ranked_layers = sorted(
            self._profile_layer_records,
            key=lambda rec: (
                float(rec.get('payload_allreduce_ms', 0.0))
                + float(rec.get('mask_topk_ms', 0.0))
                + float(rec.get('grad_copyback_ms', 0.0))
            ),
            reverse=True,
        )
        details: List[str] = []
        for rec in ranked_layers[: self._profile_top_layers]:
            details.append(
                "b{buffer}[{start}:{end}] sel={selected}/{numel} "
                "mask_topk={mask_topk_ms:.3f}ms "
                "payload_ar={payload_allreduce_ms:.3f}ms "
                "decompress_scatter={decompress_scatter_ms:.3f}ms "
                "grad_copyback={grad_copyback_ms:.3f}ms "
                "residual={residual_update_ms:.3f}ms "
                "update={update_compute_ms:.3f}ms".format(
                    buffer=int(rec.get('buffer_idx', -1)),
                    start=int(rec.get('start', -1)),
                    end=int(rec.get('end', -1)),
                    selected=int(rec.get('selected', 0)),
                    numel=int(rec.get('numel', 0)),
                    mask_topk_ms=float(rec.get('mask_topk_ms', 0.0)),
                    payload_allreduce_ms=float(rec.get('payload_allreduce_ms', 0.0)),
                    decompress_scatter_ms=float(rec.get('decompress_scatter_ms', 0.0)),
                    grad_copyback_ms=float(rec.get('grad_copyback_ms', 0.0)),
                    residual_update_ms=float(rec.get('residual_update_ms', 0.0)),
                    update_compute_ms=float(rec.get('update_compute_ms', 0.0)),
                )
            )
        logger.info(
            "%s profile step=%d top_layers: %s",
            self.__class__.__name__,
            int(train_iter),
            " || ".join(details),
        )

    def set_optimizer(self, optimizer: Any) -> None:
        """Attach optimizer state views used by the reducer."""
        self._param_to_group.clear()
        self._param_to_state_dict.clear()
        self._param_to_optim_param.clear()
        self._clip_grad_max_norm = 0.0
        self._grad_stats_parallel_group = None

        optim_param_to_group: Dict[torch.Tensor, Dict[str, Any]] = {}
        optim_param_to_state_dict: Dict[torch.Tensor, Dict[torch.Tensor, Dict[str, Any]]] = {}

        for wrapper in self._iter_optimizer_wrappers(optimizer):
            if getattr(wrapper, 'is_stub_optimizer', False):
                continue

            wrapper_config = getattr(wrapper, 'config', None)
            if wrapper_config is not None and hasattr(wrapper_config, 'clip_grad'):
                self._clip_grad_max_norm = max(
                    self._clip_grad_max_norm, float(getattr(wrapper_config, 'clip_grad', 0.0))
                )

            if self._grad_stats_parallel_group is None:
                get_group_fn = getattr(wrapper, 'get_grad_stats_parallel_group', None)
                if callable(get_group_fn):
                    try:
                        self._grad_stats_parallel_group = get_group_fn()
                    except Exception:
                        self._grad_stats_parallel_group = None

            inner_optimizer = getattr(wrapper, 'optimizer', wrapper)
            param_groups = getattr(inner_optimizer, 'param_groups', None)
            state = getattr(inner_optimizer, 'state', None)
            if param_groups is None or state is None:
                continue

            for group in param_groups:
                for param in group.get('params', []):
                    optim_param_to_group[param] = group
                    optim_param_to_state_dict[param] = state

        for buffer_param_slices in self._buffer_param_slices:
            for param_slice in buffer_param_slices:
                model_param = param_slice.param
                optim_param = getattr(model_param, 'main_param', model_param)

                group = optim_param_to_group.get(optim_param, None)
                state_dict = optim_param_to_state_dict.get(optim_param, None)
                if group is None or state_dict is None:
                    continue

                self._param_to_group[model_param] = group
                self._param_to_state_dict[model_param] = state_dict
                self._param_to_optim_param[model_param] = optim_param

                state = state_dict.setdefault(optim_param, {})
                exp_avg = state.get('exp_avg', None)
                if (
                    exp_avg is None
                    or exp_avg.shape != optim_param.shape
                    or exp_avg.device != optim_param.device
                    or exp_avg.dtype != torch.float32
                ):
                    exp_avg = torch.zeros_like(optim_param, dtype=torch.float32)
                    state['exp_avg'] = exp_avg
                state['momentum_buffer'] = exp_avg
                state.pop('exp_avg_sq', None)

        total_model_params = sum(len(buffer_param_slices) for buffer_param_slices in self._buffer_param_slices)
        mapped_model_params = len(self._param_to_optim_param)
        if mapped_model_params < total_model_params:
            logger.warning(
                "Top-k AdamS reducer optimizer mapping incomplete: mapped %d / %d model parameters.",
                mapped_model_params,
                total_model_params,
            )

    def _optimizer_state_key(self, model_param: torch.nn.Parameter) -> torch.Tensor:
        return self._param_to_optim_param.get(model_param, model_param)

    def _optimizer_state_entry(self, model_param: torch.nn.Parameter) -> Optional[Dict[str, Any]]:
        state_dict = self._param_to_state_dict.get(model_param, None)
        if state_dict is None:
            return None
        return state_dict.get(self._optimizer_state_key(model_param), None)

    def _ensure_optimizer_state_entry(self, model_param: torch.nn.Parameter) -> Dict[str, Any]:
        state_dict = self._param_to_state_dict[model_param]
        return state_dict.setdefault(self._optimizer_state_key(model_param), {})

    def _model_or_optim_param_flat(self, model_param: torch.nn.Parameter) -> torch.Tensor:
        return self._optimizer_state_key(model_param).view(-1)

    def prepare_pre_forward(self, train_iter: int) -> None:
        """Move precomputed next-step masks into current-step masks once per iteration."""
        if self._prepared_iteration == int(train_iter):
            return
        for state in self._buffer_states:
            self._wait_next_mask_allreduce(state)
            if state.has_next_mask:
                state.current_mask_u8.copy_(state.next_mask_u8)
                state.has_current_mask = True
        self._prepared_iteration = int(train_iter)

    def _wait_next_mask_allreduce(self, state: _BufferState) -> None:
        handle = state.next_mask_allreduce_handle
        if handle is None:
            state.next_mask_allreduce_uses_packed = False
            return
        t_wait = self._profile_tic(state.next_mask_u8)
        handle.wait()
        self._profile_toc('mask.next_async_wait_ms', t_wait, state.next_mask_u8)

        if state.next_mask_allreduce_uses_packed:
            packed = state.next_mask_packed_u8
            if packed is None:
                raise RuntimeError("Internal error: next packed mask buffer is missing.")
            t_unpack = self._profile_tic(state.next_mask_u8)
            self._unpack_packed_mask_to_local_u8(
                packed,
                state.next_mask_u8,
                state.mask_pack_segments,
            )
            self._profile_toc('mask.next_async_unpack_ms', t_unpack, state.next_mask_u8)

        state.next_mask_allreduce_handle = None
        state.next_mask_allreduce_uses_packed = False
        if self._async_mask_stream is not None and state.next_mask_u8.is_cuda:
            t_stream_wait = self._profile_tic(state.next_mask_u8)
            torch.cuda.current_stream(state.next_mask_u8.device).wait_stream(
                self._async_mask_stream
            )
            self._profile_toc(
                'mask.next_async_stream_wait_ms', t_stream_wait, state.next_mask_u8
            )
        state.has_next_mask = True

    def _launch_next_mask_allreduce_async(
        self,
        state: _BufferState,
        group: torch.distributed.ProcessGroup,
    ) -> None:
        # Re-launching is illegal while a previous async mask all-reduce is inflight.
        self._wait_next_mask_allreduce(state)

        world_size = self._group_size(group)
        if world_size <= 1 or state.next_mask_u8.numel() == 0:
            state.next_mask_allreduce_handle = None
            state.next_mask_allreduce_uses_packed = False
            state.has_next_mask = True
            return

        state.has_next_mask = False

        if self._async_mask_stream is not None and state.next_mask_u8.is_cuda:
            current_stream = torch.cuda.current_stream(state.next_mask_u8.device)
            t_stream_wait = self._profile_tic(state.next_mask_u8)
            self._async_mask_stream.wait_stream(current_stream)
            self._profile_toc(
                'mask.next_async_launch_wait_stream_ms',
                t_stream_wait,
                state.next_mask_u8,
            )
            with torch.cuda.stream(self._async_mask_stream):
                state.next_mask_allreduce_handle = self._allreduce_max_mask_async_launch(
                    state,
                    group,
                    profile_key='mask.next_async_launch_ms',
                )
            return

        state.next_mask_allreduce_handle = self._allreduce_max_mask_async_launch(
            state,
            group,
            profile_key='mask.next_async_launch_ms',
        )

    def _flush_pending_next_mask_allreduces(self) -> None:
        for state in self._buffer_states:
            self._wait_next_mask_allreduce(state)

    def _scheduled_density(self, train_iter: int) -> float:
        train_iter = int(train_iter)
        target = min(max(float(self._target_density), 0.0), 1.0)
        start = min(max(float(self._density_start), 0.0), 1.0)

        # Optional cooldown schedule: geometrically increase density from target to 1.0.
        if self._cooldown_steps > 0:
            start_density = min(max(target, 1.0e-12), 1.0)
            if start_density >= 1.0:
                return 1.0

            if self._cooldown_start_step >= 0:
                cooldown_start_step = int(self._cooldown_start_step)
                # User-requested behavior: backfill start when configured step is too early.
                if cooldown_start_step < self._cooldown_steps:
                    cooldown_start_step = train_iter - int(self._cooldown_steps)
            else:
                cooldown_start_step = int(self._start_iter)

            if train_iter < cooldown_start_step:
                return start_density

            t = train_iter - cooldown_start_step
            if t >= self._cooldown_steps:
                return 1.0

            frac = float(t) / float(self._cooldown_steps)
            scheduled = start_density * ((1.0 / start_density) ** frac)
            return min(scheduled, 1.0)

        # Geometric warmup from density_start to target density.
        if self._warmup_steps <= 0 or abs(start - target) <= 1.0e-12:
            return target
        if start <= 0.0:
            return target

        t = max(0, train_iter - int(self._start_iter))
        if t >= self._warmup_steps:
            return target

        frac = float(t) / float(self._warmup_steps)
        return start * ((target / start) ** frac)

    def _in_density_warmup_stage(self, train_iter: int) -> bool:
        scheduled_density = float(self._scheduled_density(int(train_iter)))
        return abs(scheduled_density - float(self._target_density)) > 1.0e-12

    def _use_fp8_quantized_payload(self, train_iter: int) -> bool:
        return bool(self._use_fp8_topk_quant and not self._in_density_warmup_stage(train_iter))

    @staticmethod
    def _topk_k(density: float, numel: int) -> int:
        if numel <= 0:
            return 0
        if density <= 0.0:
            return 0
        if density >= 1.0:
            return numel
        return max(1, int(density * numel))

    @staticmethod
    def _should_exclude_name_from_topk(name: str) -> bool:
        if not name:
            return False
        lname = str(name).lower()
        return (
            'layernorm' in lname
            or 'layer_norm' in lname
            or 'rmsnorm' in lname
            or 'rms_norm' in lname
            or 'bias' in lname
        )

    @staticmethod
    def _mask_pack_numel(mask_numel: int) -> int:
        return (int(mask_numel) + 7) // 8

    def _build_mask_pack_layout(
        self,
        numel: int,
        param_slices: List[_ParamSlice],
    ) -> tuple[Tuple[Tuple[int, int, int, int, int], ...], int]:
        total_numel = int(numel)
        if total_numel <= 0:
            return tuple(), 0

        raw_segments: List[Tuple[int, int, int]] = []
        for param_slice in param_slices:
            start = max(0, min(int(param_slice.start), total_numel))
            end = max(0, min(int(param_slice.end), total_numel))
            if end <= start:
                continue
            raw_segments.append((int(start), int(end), int(param_slice.owner)))

        if not raw_segments:
            packed_numel = self._mask_pack_numel(total_numel)
            return ((0, total_numel, 0, packed_numel, -1),), packed_numel

        # Coalesce adjacent ranges with same owner so pack/unpack launches fewer kernels.
        merged_ranges: List[List[int]] = []
        for start, end, owner in raw_segments:
            if not merged_ranges:
                merged_ranges.append([start, end, owner])
                continue
            prev_start, prev_end, prev_owner = merged_ranges[-1]
            if owner == prev_owner and start == prev_end:
                merged_ranges[-1][1] = end
            else:
                merged_ranges.append([start, end, owner])

        segments: List[Tuple[int, int, int, int, int]] = []
        packed_cursor = 0
        for start, end, owner in merged_ranges:
            nbytes = self._mask_pack_numel(end - start)
            segments.append((int(start), int(end), int(packed_cursor), int(nbytes), int(owner)))
            packed_cursor += int(nbytes)

        if packed_cursor <= 0:
            packed_numel = self._mask_pack_numel(total_numel)
            return ((0, total_numel, 0, packed_numel, -1),), packed_numel

        return tuple(segments), int(packed_cursor)

    def _pack_mask_segments_to_packed(
        self,
        mask_u8: torch.Tensor,
        packed_u8: torch.Tensor,
        segments: Tuple[Tuple[int, int, int, int, int], ...],
    ) -> None:
        packed_u8.zero_()
        if packed_u8.numel() == 0 or not segments:
            return

        use_triton = (
            _TRITON_AVAILABLE
            and mask_u8.is_cuda
            and packed_u8.is_cuda
            and mask_u8.is_contiguous()
            and packed_u8.is_contiguous()
        )
        if use_triton:
            for start, end, packed_offset, byte_len, _owner in segments:
                if byte_len <= 0:
                    continue
                _launch_mask_pack_triton(
                    mask_u8[start:end],
                    packed_u8[packed_offset : packed_offset + byte_len],
                )
            return

        if not hasattr(self, '_mask_pack_fallback_warned'):
            self._mask_pack_fallback_warned = False
        if not self._mask_pack_fallback_warned:
            logger.warning(
                "Top-k reducer Triton pack kernel unavailable; using torch fallback for mask pack."
            )
            self._mask_pack_fallback_warned = True

        shifts = torch.arange(8, device=mask_u8.device, dtype=torch.int16)
        weights = (1 << shifts).to(torch.int16)
        for start, end, packed_offset, byte_len, _owner in segments:
            if byte_len <= 0:
                continue
            src = mask_u8[start:end].to(torch.int16)
            rem = int(src.numel()) % 8
            if rem != 0:
                pad = torch.zeros(8 - rem, dtype=src.dtype, device=src.device)
                src = torch.cat((src, pad), dim=0)
            if src.numel() == 0:
                continue
            packed_vals = (src.view(-1, 8) * weights).sum(dim=1).to(torch.uint8)
            packed_u8[packed_offset : packed_offset + byte_len].copy_(packed_vals)

    def _unpack_packed_mask_to_local_u8(
        self,
        packed_u8: torch.Tensor,
        mask_u8: torch.Tensor,
        segments: Tuple[Tuple[int, int, int, int, int], ...],
    ) -> None:
        mask_u8.zero_()
        if packed_u8.numel() == 0 or not segments:
            return

        use_triton = (
            _TRITON_AVAILABLE
            and packed_u8.is_cuda
            and mask_u8.is_cuda
            and packed_u8.is_contiguous()
            and mask_u8.is_contiguous()
        )
        if use_triton:
            for start, end, packed_offset, byte_len, _owner in segments:
                if byte_len <= 0:
                    continue
                _launch_mask_unpack_triton(
                    packed_u8[packed_offset : packed_offset + byte_len],
                    mask_u8[start:end],
                    int(end - start),
                )
            return

        if not hasattr(self, '_mask_unpack_fallback_warned'):
            self._mask_unpack_fallback_warned = False
        if not self._mask_unpack_fallback_warned:
            logger.warning(
                "Top-k reducer Triton unpack kernel unavailable; using torch fallback for mask unpack."
            )
            self._mask_unpack_fallback_warned = True

        shifts = torch.arange(8, device=mask_u8.device, dtype=torch.int16)
        for start, end, packed_offset, byte_len, _owner in segments:
            if byte_len <= 0:
                continue
            packed_vals = packed_u8[packed_offset : packed_offset + byte_len].to(torch.int16)
            bits = ((packed_vals.unsqueeze(1) >> shifts) & 1).to(torch.uint8).reshape(-1)
            mask_u8[start:end].copy_(bits[: int(end - start)])

    def _allreduce_max_mask_sync(
        self,
        state: _BufferState,
        group: torch.distributed.ProcessGroup,
        profile_key: str,
    ) -> None:
        mask_u8 = state.current_mask_u8
        world_size = self._group_size(group)
        if world_size <= 1 or mask_u8.numel() == 0:
            return

        packed = state.current_mask_packed_u8
        segments = state.mask_pack_segments
        can_use_packed = (
            packed is not None
            and packed.device == mask_u8.device
            and packed.numel() > 0
            and bool(segments)
        )
        if not can_use_packed:
            self._allreduce_max_mask_(mask_u8, group, async_op=False, profile_key=profile_key)
            return

        t_pack = self._profile_tic(mask_u8)
        self._pack_mask_segments_to_packed(mask_u8, packed, segments)
        self._profile_toc(f'{profile_key}.pack_ms', t_pack, packed)

        t_allreduce = self._profile_tic(packed)
        torch.distributed.all_reduce(
            packed,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
            async_op=False,
        )
        self._profile_toc(profile_key, t_allreduce, packed)

        t_unpack = self._profile_tic(mask_u8)
        self._unpack_packed_mask_to_local_u8(packed, mask_u8, segments)
        self._profile_toc(f'{profile_key}.unpack_ms', t_unpack, mask_u8)

    def _allreduce_max_mask_async_launch(
        self,
        state: _BufferState,
        group: torch.distributed.ProcessGroup,
        profile_key: str,
    ) -> Optional[Any]:
        mask_u8 = state.next_mask_u8
        world_size = self._group_size(group)
        if world_size <= 1 or mask_u8.numel() == 0:
            state.next_mask_allreduce_uses_packed = False
            return None

        packed = state.next_mask_packed_u8
        segments = state.mask_pack_segments
        can_use_packed = (
            packed is not None
            and packed.device == mask_u8.device
            and packed.numel() > 0
            and bool(segments)
        )
        if not can_use_packed:
            state.next_mask_allreduce_uses_packed = False
            return self._allreduce_max_mask_(
                mask_u8,
                group,
                async_op=True,
                profile_key=profile_key,
            )

        t_pack = self._profile_tic(mask_u8)
        self._pack_mask_segments_to_packed(mask_u8, packed, segments)
        self._profile_toc('mask.next_async_pack_ms', t_pack, packed)

        t_allreduce = self._profile_tic(packed)
        handle = torch.distributed.all_reduce(
            packed,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
            async_op=True,
        )
        self._profile_toc(profile_key, t_allreduce, packed)
        state.next_mask_allreduce_uses_packed = True
        return handle

    def _build_mask_from_corrected(
        self,
        grad_data: torch.Tensor,
        residual: torch.Tensor,
        param_slices: List[_ParamSlice],
        density: float,
        dense_mode: bool,
        out_mask_u8: torch.Tensor,
        group_rank: int,
        profile_tag: str = 'mask.build',
    ) -> None:
        t_zero = self._profile_tic(out_mask_u8)
        out_mask_u8.zero_()
        self._profile_toc(f'{profile_tag}.zero_ms', t_zero, out_mask_u8)

        for param_slice in param_slices:
            start = param_slice.start
            end = param_slice.end
            param = param_slice.param
            numel = end - start
            if numel <= 0:
                continue
            if int(param_slice.owner) != int(group_rank):
                continue

            self._profile_add_value(f'{profile_tag}.candidate_layers', 1.0)
            self._profile_add_value(f'{profile_tag}.candidate_numel', float(numel))

            if param_slice.exclude_from_topk:
                t_fill = self._profile_tic(out_mask_u8)
                out_mask_u8[start:end] = 1
                self._profile_toc(f'{profile_tag}.excluded_fill_ms', t_fill, out_mask_u8)
                self._profile_add_value(f'{profile_tag}.excluded_layers', 1.0)
                self._profile_add_value(f'{profile_tag}.excluded_numel', float(numel))
                self._profile_add_value(f'{profile_tag}.selected_elements', float(numel))
                continue

            if dense_mode or density >= 1.0:
                t_fill = self._profile_tic(out_mask_u8)
                out_mask_u8[start:end] = 1
                self._profile_toc(f'{profile_tag}.dense_fill_ms', t_fill, out_mask_u8)
                self._profile_add_value(f'{profile_tag}.selected_elements', float(numel))
                continue

            k = self._topk_k(density, numel)
            if k <= 0:
                continue
            if k >= numel:
                t_fill = self._profile_tic(out_mask_u8)
                out_mask_u8[start:end] = 1
                self._profile_toc(f'{profile_tag}.full_fill_ms', t_fill, out_mask_u8)
                self._profile_add_value(f'{profile_tag}.selected_elements', float(numel))
                continue

            state = self._optimizer_state_entry(param)
            if state is None or 'exp_avg' not in state:
                t_fill = self._profile_tic(out_mask_u8)
                out_mask_u8[start:end] = 1
                self._profile_toc(f'{profile_tag}.missing_state_fill_ms', t_fill, out_mask_u8)
                self._profile_add_value(f'{profile_tag}.selected_elements', float(numel))
                continue

            t_corrected = self._profile_tic(grad_data)
            exp_avg_prev = state['exp_avg'].view(-1).to(torch.float32)
            group = self._param_to_group.get(param, {})
            beta1 = float(group.get('betas', (0.9, 0.999))[0])

            grad_slice_fp32 = grad_data[start:end].to(torch.float32)
            corrected = exp_avg_prev * beta1
            corrected.add_(grad_slice_fp32, alpha=1.0 - beta1)
            corrected.add_(residual[start:end])
            self._profile_toc(f'{profile_tag}.corrected_ms', t_corrected, grad_data)

            t_topk = self._profile_tic(corrected)
            _, topk_idx = torch.topk(corrected.abs(), k=k, sorted=False)
            self._profile_toc(f'{profile_tag}.topk_ms', t_topk, corrected)

            t_index_fill = self._profile_tic(out_mask_u8)
            out_mask_u8[start:end].index_fill_(0, topk_idx, 1)
            self._profile_toc(f'{profile_tag}.index_fill_ms', t_index_fill, out_mask_u8)
            self._profile_add_value(f'{profile_tag}.selected_elements', float(k))
            self._profile_add_value(f'{profile_tag}.topk_calls', 1.0)

    def _allreduce_max_mask_(
        self,
        mask_u8: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        async_op: bool = False,
        profile_key: Optional[str] = None,
    ) -> Optional[Any]:
        world_size = self._group_size(group)
        if world_size <= 1 or mask_u8.numel() == 0:
            return None
        t_allreduce = self._profile_tic(mask_u8)
        handle = torch.distributed.all_reduce(
            mask_u8,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
            async_op=async_op,
        )
        key = profile_key
        if key is None:
            key = 'mask.allreduce_async_launch_ms' if async_op else 'mask.allreduce_sync_ms'
        self._profile_toc(key, t_allreduce, mask_u8)
        return handle if async_op else None

    def _build_synced_mask_from_corrected(
        self,
        grad_data: torch.Tensor,
        residual: torch.Tensor,
        param_slices: List[_ParamSlice],
        density: float,
        dense_mode: bool,
        out_mask_u8: torch.Tensor,
        state: _BufferState,
        group: torch.distributed.ProcessGroup,
        group_rank: int,
        profile_tag: str = 'mask.current_sync',
    ) -> None:
        self._build_mask_from_corrected(
            grad_data=grad_data,
            residual=residual,
            param_slices=param_slices,
            density=density,
            dense_mode=dense_mode,
            out_mask_u8=out_mask_u8,
            group_rank=group_rank,
            profile_tag=f'{profile_tag}.build',
        )
        if out_mask_u8.data_ptr() == state.current_mask_u8.data_ptr():
            self._allreduce_max_mask_sync(
                state,
                group,
                profile_key=f'{profile_tag}.allreduce_sync_ms',
            )
        else:
            # Defensive fallback for unexpected mask tensors.
            self._allreduce_max_mask_(
                out_mask_u8,
                group,
                profile_key=f'{profile_tag}.allreduce_sync_ms',
            )

    def _allreduce_average_(
        self,
        tensor: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        profile_key: Optional[str] = None,
    ) -> None:
        world_size = self._group_size(group)
        if world_size <= 1 or tensor.numel() == 0:
            return
        key_base = profile_key or 'allreduce.average'
        t_allreduce = self._profile_tic(tensor)
        torch.distributed.all_reduce(tensor, group=group)
        self._profile_toc(f'{key_base}.allreduce_ms', t_allreduce, tensor)
        t_div = self._profile_tic(tensor)
        tensor.div_(world_size)
        self._profile_toc(f'{key_base}.div_ms', t_div, tensor)

    def _fp8_allreduce_(
        self,
        tensor: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        profile_key: str = 'sparse.layer.fp8.payload',
    ) -> float:
        total_ms = 0.0
        try:
            t_allreduce = self._profile_tic(tensor)
            torch.distributed.all_reduce(tensor, group=group)
            total_ms += self._profile_toc(f'{profile_key}.allreduce_ms', t_allreduce, tensor)
        except Exception as exc:
            if not self._fp8_allreduce_warned:
                logger.warning(
                    "FP8 all-reduce failed in top-k reducer; falling back to FP32 all-reduce. "
                    "Exception: %s",
                    exc,
                )
                self._fp8_allreduce_warned = True
            t_cast = self._profile_tic(tensor)
            fallback = tensor.to(torch.float32)
            total_ms += self._profile_toc(f'{profile_key}.fallback_cast_ms', t_cast, tensor)
            t_fallback_ar = self._profile_tic(fallback)
            torch.distributed.all_reduce(fallback, group=group)
            total_ms += self._profile_toc(
                f'{profile_key}.fallback_allreduce_ms', t_fallback_ar, fallback
            )
            t_copy_back = self._profile_tic(tensor)
            tensor.copy_(fallback.to(tensor.dtype))
            total_ms += self._profile_toc(
                f'{profile_key}.fallback_copyback_ms', t_copy_back, tensor
            )
        return total_ms

    def _should_refresh_layer_scales(self, train_iter: int) -> bool:
        if self._last_fp8_scale_update_iter < 0:
            return True
        return (int(train_iter) - int(self._last_fp8_scale_update_iter)) >= int(
            self._fp8_scale_interval
        )

    @staticmethod
    def _max_exp_to_scale(exponent: float, eps: float) -> float:
        if not math.isfinite(exponent):
            return 1.0
        scale = math.ldexp(1.0, int(exponent))
        return scale if scale >= eps else eps

    def _global_layer_max_exponent(
        self,
        selected_values: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        profile_key: str = 'sparse.layer.fp8.scale_refresh',
    ) -> float:
        t_local = self._profile_tic(selected_values)
        abs_selected = selected_values.abs()
        nonzero = abs_selected > 0
        if bool(nonzero.any().item()):
            exponent = torch.floor(torch.log2(abs_selected[nonzero])).max()
        else:
            exponent = torch.tensor(
                float("-inf"), dtype=torch.float32, device=selected_values.device
            )
        self._profile_toc(f'{profile_key}.local_maxexp_ms', t_local, selected_values)
        if self._group_size(group) > 1:
            t_allreduce = self._profile_tic(exponent)
            torch.distributed.all_reduce(exponent, op=torch.distributed.ReduceOp.MAX, group=group)
            self._profile_toc(f'{profile_key}.allreduce_max_ms', t_allreduce, exponent)
        return float(exponent.item())

    def _fp8_quantized_allreduce(
        self,
        selected_values: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        layer_key: Tuple[int, int, int],
        train_iter: int,
        profile_key_prefix: str = 'sparse.layer.fp8',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize selected values to FP8 with shared layer-wise scale before all-reduce."""
        if selected_values.numel() == 0:
            empty = selected_values.to(torch.float32)
            return empty, empty

        t_cast = self._profile_tic(selected_values)
        selected_fp32 = selected_values.to(torch.float32)
        self._profile_toc(f'{profile_key_prefix}.cast_fp32_ms', t_cast, selected_values)

        world_size = self._group_size(group)
        should_refresh = self._should_refresh_layer_scales(train_iter)
        refreshed_scale = False
        if should_refresh or layer_key not in self._layer_fp8_scales:
            t_refresh = self._profile_tic(selected_fp32)
            exponent = self._global_layer_max_exponent(
                selected_fp32,
                group,
                profile_key=f'{profile_key_prefix}.scale_refresh',
            )
            self._layer_fp8_scales[layer_key] = self._max_exp_to_scale(
                exponent, self._fp8_scale_eps
            )
            refreshed_scale = True
            self._profile_toc(
                f'{profile_key_prefix}.scale_refresh_total_ms', t_refresh, selected_fp32
            )
            self._profile_add_value(f'{profile_key_prefix}.scale_refresh_count', 1.0)
        layer_scale = float(self._layer_fp8_scales.get(layer_key, 1.0))

        t_normalize = self._profile_tic(selected_fp32)
        n_workers_f = float(max(1, world_size))
        inv_n_workers = 1.0 / n_workers_f
        normalized = selected_fp32 / layer_scale
        if inv_n_workers != 1.0:
            normalized.mul_(inv_n_workers)

        fp8_info = torch.finfo(self._fp8_dtype)
        fp8_max = float(fp8_info.max)
        fp8_tiny = float(fp8_info.tiny) if fp8_info.tiny > 0 else 0.0

        abs_normalized = normalized.abs()
        finite = torch.isfinite(abs_normalized)
        nonzero = torch.logical_and(finite, abs_normalized > 0)
        if fp8_tiny > 0.0:
            underflow_mask = torch.logical_and(nonzero, abs_normalized < fp8_tiny)
        else:
            underflow_mask = torch.zeros_like(nonzero)
        if bool(underflow_mask.any().item()):
            normalized.masked_fill_(underflow_mask, 0.0)
        overflow_mask = torch.logical_or(~finite, abs_normalized > fp8_max)
        if bool(overflow_mask.any().item()):
            raise RuntimeError(
                "FP8 overflow detected in top-k reducer payload. "
                "Try disabling --use-fp8-topk-quant or reducing LR."
            )
        self._profile_toc(
            f'{profile_key_prefix}.normalize_clip_ms', t_normalize, selected_fp32
        )

        t_quantize = self._profile_tic(normalized)
        quant_fp8 = normalized.to(self._fp8_dtype)
        local_compressed_fp32 = quant_fp8.to(torch.float32)
        local_compressed_fp32.mul_(layer_scale * n_workers_f)
        quant_error = selected_fp32 - local_compressed_fp32
        self._profile_toc(
            f'{profile_key_prefix}.quantize_error_ms', t_quantize, quant_fp8
        )

        if world_size > 1:
            self._fp8_allreduce_(
                quant_fp8,
                group,
                profile_key=f'{profile_key_prefix}.payload',
            )

        t_dequant = self._profile_tic(quant_fp8)
        synced_selected = quant_fp8.to(torch.float32)
        synced_selected.mul_(layer_scale)
        self._profile_toc(f'{profile_key_prefix}.dequantize_ms', t_dequant, quant_fp8)

        if refreshed_scale:
            self._last_fp8_scale_update_iter = int(train_iter)
        return synced_selected, quant_error


    def _fp8_quantized_allreduce_packed(
        self,
        selected_values: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        train_iter: int,
        profile_key_prefix: str = 'sparse.buffer.fp8',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize packed sparse payloads to FP8 and all-reduce once."""
        if selected_values.numel() == 0:
            empty = selected_values.to(torch.float32)
            return empty, empty

        t_cast = self._profile_tic(selected_values)
        selected_fp32 = selected_values.to(torch.float32)
        self._profile_toc(f'{profile_key_prefix}.cast_fp32_ms', t_cast, selected_values)

        world_size = self._group_size(group)
        should_refresh = self._should_refresh_layer_scales(train_iter)
        refreshed_scale = False
        if (
            should_refresh
            or not math.isfinite(self._packed_fp8_payload_scale)
            or self._packed_fp8_payload_scale <= 0.0
        ):
            t_refresh = self._profile_tic(selected_fp32)
            exponent = self._global_layer_max_exponent(
                selected_fp32,
                group,
                profile_key=f'{profile_key_prefix}.scale_refresh',
            )
            self._packed_fp8_payload_scale = self._max_exp_to_scale(
                exponent, self._fp8_scale_eps
            )
            refreshed_scale = True
            self._profile_toc(
                f'{profile_key_prefix}.scale_refresh_total_ms', t_refresh, selected_fp32
            )
            self._profile_add_value(f'{profile_key_prefix}.scale_refresh_count', 1.0)
        payload_scale = float(self._packed_fp8_payload_scale)

        t_normalize = self._profile_tic(selected_fp32)
        n_workers_f = float(max(1, world_size))
        inv_n_workers = 1.0 / n_workers_f
        normalized = selected_fp32 / payload_scale
        if inv_n_workers != 1.0:
            normalized.mul_(inv_n_workers)

        fp8_info = torch.finfo(self._fp8_dtype)
        fp8_max = float(fp8_info.max)
        fp8_tiny = float(fp8_info.tiny) if fp8_info.tiny > 0 else 0.0

        abs_normalized = normalized.abs()
        finite = torch.isfinite(abs_normalized)
        nonzero = torch.logical_and(finite, abs_normalized > 0)
        if fp8_tiny > 0.0:
            underflow_mask = torch.logical_and(nonzero, abs_normalized < fp8_tiny)
        else:
            underflow_mask = torch.zeros_like(nonzero)
        if bool(underflow_mask.any().item()):
            normalized.masked_fill_(underflow_mask, 0.0)
        overflow_mask = torch.logical_or(~finite, abs_normalized > fp8_max)
        if bool(overflow_mask.any().item()):
            raise RuntimeError(
                "FP8 overflow detected in packed top-k reducer payload. "
                "Try disabling --use-fp8-topk-quant or reducing LR."
            )
        self._profile_toc(
            f'{profile_key_prefix}.normalize_clip_ms', t_normalize, selected_fp32
        )

        t_quantize = self._profile_tic(normalized)
        quant_fp8 = normalized.to(self._fp8_dtype)
        local_compressed_fp32 = quant_fp8.to(torch.float32)
        local_compressed_fp32.mul_(payload_scale * n_workers_f)
        quant_error = selected_fp32 - local_compressed_fp32
        self._profile_toc(
            f'{profile_key_prefix}.quantize_error_ms', t_quantize, quant_fp8
        )

        if world_size > 1:
            self._fp8_allreduce_(
                quant_fp8,
                group,
                profile_key=f'{profile_key_prefix}.payload',
            )

        t_dequant = self._profile_tic(quant_fp8)
        synced_selected = quant_fp8.to(torch.float32)
        synced_selected.mul_(payload_scale)
        self._profile_toc(f'{profile_key_prefix}.dequantize_ms', t_dequant, quant_fp8)

        if refreshed_scale:
            self._last_fp8_scale_update_iter = int(train_iter)
        return synced_selected, quant_error

    @staticmethod
    def _group_weight_decay(group: Dict[str, Any]) -> float:
        if 'weight_decay_reducer' in group:
            return float(group['weight_decay_reducer'])
        return float(group.get('weight_decay', 0.0))

    def _reduce_dense_fallback(
        self,
        grad_data: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        state: _BufferState,
        profile_tag: str = 'dense.buffer',
    ) -> None:
        self._allreduce_average_(
            grad_data,
            group,
            profile_key=f'{profile_tag}.grad_allreduce',
        )
        t_wait = self._profile_tic(state.next_mask_u8)
        self._wait_next_mask_allreduce(state)
        self._profile_toc(f'{profile_tag}.wait_next_mask_ms', t_wait, state.next_mask_u8)

        t_reset = self._profile_tic(state.residual)
        state.residual.zero_()
        state.current_mask_u8.fill_(1)
        state.next_mask_u8.fill_(1)
        state.next_mask_allreduce_handle = None
        state.has_current_mask = True
        state.has_next_mask = True
        self._profile_toc(f'{profile_tag}.state_reset_ms', t_reset, state.residual)

    @torch.no_grad()
    def reduce(self, train_iter: int, force_all_reduce: bool = False) -> None:
        self._prepared_iteration = None
        self._profile_reset_step()
        reduce_t0 = self._profile_tic()

        current_density = self._scheduled_density(int(train_iter))
        next_density = self._scheduled_density(int(train_iter) + 1)
        # Use iteration-derived Adam step so resume is stable even when checkpoint
        # formats do not persist per-parameter optimizer "step" entries.
        adam_step = int(train_iter) + 1

        dense_mode = bool(force_all_reduce or int(train_iter) < self._start_iter)
        if force_all_reduce:
            current_density = 1.0
            next_density = 1.0
        use_fp8_quantized_payload = self._use_fp8_quantized_payload(int(train_iter))

        total_selected = 0
        total_numel = 0
        synced_grad_sq_sum: Optional[torch.Tensor] = None
        update_metric_sq_sum: Optional[torch.Tensor] = None
        move_clip_grad_to_reducer = bool(self._move_clip_grad_to_reducer)
        pending_updates: List[_PendingUpdate] = []
        synced_grad_buffers: List[Optional[torch.Tensor]] = [None] * len(self._buffers)

        for buffer_idx, buffer in enumerate(self._buffers):
            grad_data = buffer.grad_data
            if grad_data is None or grad_data.numel() == 0:
                continue

            state = self._buffer_states[buffer_idx]
            param_slices = self._buffer_param_slices[buffer_idx]
            group = buffer.data_parallel_group
            group_rank = self._buffer_group_ranks[buffer_idx]

            if not param_slices:
                self._reduce_dense_fallback(
                    grad_data,
                    group,
                    state,
                    profile_tag='dense.buffer',
                )
                continue

            missing_mapping = False
            for param_slice in param_slices:
                if (
                    param_slice.param not in self._param_to_group
                    or param_slice.param not in self._param_to_state_dict
                ):
                    missing_mapping = True
                    break
            if missing_mapping:
                if not self._warned_missing_mapping:
                    logger.warning(
                        "Top-k AdamS reducer could not map at least one model parameter "
                        "to optimizer state. Falling back to dense sync for affected buffers."
                    )
                    self._warned_missing_mapping = True
                self._reduce_dense_fallback(
                    grad_data,
                    group,
                    state,
                    profile_tag='dense.buffer',
                )
                continue

            if move_clip_grad_to_reducer and synced_grad_buffers[buffer_idx] is None:
                if grad_data.dtype == torch.float32:
                    synced_grad_buffers[buffer_idx] = grad_data
                else:
                    synced_grad_buffers[buffer_idx] = torch.empty_like(
                        grad_data, dtype=torch.float32
                    )

            if dense_mode:
                # In scheduled dense mode, mirror non-topk communication behavior:
                # one dense all-reduce on gradients and no mask collective.
                self._reduce_dense_fallback(
                    grad_data,
                    group,
                    state,
                    profile_tag='dense.buffer',
                )

                # Preserve one-step-ahead mask preparation at the dense->sparse boundary.
                next_dense_mode = bool(force_all_reduce or (int(train_iter) + 1) < self._start_iter)
                if not next_dense_mode:
                    self._build_mask_from_corrected(
                        grad_data=grad_data,
                        residual=state.residual,
                        param_slices=param_slices,
                        density=next_density,
                        dense_mode=False,
                        out_mask_u8=state.next_mask_u8,
                        group_rank=group_rank,
                        profile_tag='mask.next_from_dense',
                    )
                    self._launch_next_mask_allreduce_async(state, group)

                total_selected += int(grad_data.numel())
                total_numel += int(grad_data.numel())

                for param_slice in param_slices:
                    start = param_slice.start
                    end = param_slice.end
                    param = param_slice.param
                    if end <= start:
                        continue

                    param_state = self._ensure_optimizer_state_entry(param)
                    exp_avg = param_state['exp_avg'].view(-1)
                    exp_avg_prev = exp_avg.to(torch.float32).clone()

                    group_cfg = self._param_to_group[param]
                    beta1, beta2 = group_cfg.get('betas', (0.9, 0.999))
                    beta1 = float(beta1)
                    beta2 = float(beta2)
                    eps = float(group_cfg.get('eps', 1.0e-8))
                    bias_correction = bool(group_cfg.get('bias_correction', True))
                    weight_decay = self._group_weight_decay(group_cfg)

                    grad_slice = grad_data[start:end]
                    synced_grad = grad_slice.to(torch.float32)

                    if synced_grad_sq_sum is None:
                        synced_grad_sq_sum = torch.zeros(
                            (), dtype=torch.float64, device=synced_grad.device
                        )
                    if param_is_not_shared(
                        param
                    ) and tensor_parallel.param_is_not_tensor_parallel_duplicate(param):
                        synced_grad_sq_sum.add_(synced_grad.to(torch.float64).pow(2).sum())

                    if move_clip_grad_to_reducer:
                        synced_grad_buffer = synced_grad_buffers[buffer_idx]
                        if synced_grad_buffer is None:
                            raise RuntimeError("Internal error: synced_grad_buffer is not initialized.")
                        t_syncbuf_copy = self._profile_tic(synced_grad_buffer)
                        synced_grad_buffer[start:end].copy_(synced_grad)
                        self._profile_toc(
                            'dense.layer.synced_grad_buffer_copy_ms',
                            t_syncbuf_copy,
                            synced_grad_buffer,
                        )
                        pending_updates.append(
                            _PendingUpdate(
                                buffer_idx=buffer_idx,
                                start=start,
                                end=end,
                                param=param,
                                beta1=beta1,
                                beta2=beta2,
                                eps=eps,
                                bias_correction=bias_correction,
                                weight_decay=weight_decay,
                            )
                        )
                        continue

                    synced_momentum = exp_avg_prev * beta1
                    synced_momentum.add_(synced_grad, alpha=1.0 - beta1)
                    exp_avg.copy_(synced_momentum.to(exp_avg.dtype))
                    param_state['momentum_buffer'] = param_state['exp_avg']

                    variance = exp_avg_prev * exp_avg_prev
                    variance.mul_(beta2)
                    variance.addcmul_(synced_grad, synced_grad, value=1.0 - beta2)

                    if bias_correction:
                        bias_correction1 = 1.0 - beta1**adam_step
                        bias_correction2 = 1.0 - beta2**adam_step
                        inv_bias1 = 1.0 / bias_correction1 if bias_correction1 != 0.0 else 1.0
                        inv_sqrt_bias2 = (
                            1.0 / math.sqrt(bias_correction2) if bias_correction2 > 0.0 else 1.0
                        )
                    else:
                        inv_bias1 = 1.0
                        inv_sqrt_bias2 = 1.0

                    denom = variance.sqrt()
                    denom.mul_(inv_sqrt_bias2)
                    denom.add_(eps)

                    update = synced_momentum * inv_bias1
                    update.div_(denom)
                    if weight_decay != 0.0:
                        update.add_(
                            self._model_or_optim_param_flat(param).to(torch.float32), alpha=weight_decay
                        )
                    if update_metric_sq_sum is None:
                        update_metric_sq_sum = torch.zeros((), dtype=torch.float64, device=update.device)
                    update_metric_sq_sum.add_(update.to(torch.float64).pow(2).sum())

                    t_copyback = self._profile_tic(grad_slice)
                    grad_slice.copy_(update.to(grad_slice.dtype))
                    self._profile_toc('dense.layer.grad_copyback_ms', t_copyback, grad_slice)

                continue

            if not state.has_current_mask:
                if state.next_mask_allreduce_handle is not None:
                    self._wait_next_mask_allreduce(state)
                if state.has_next_mask:
                    state.current_mask_u8.copy_(state.next_mask_u8)
                    state.has_current_mask = True
                else:
                    self._build_synced_mask_from_corrected(
                        grad_data=grad_data,
                        residual=state.residual,
                        param_slices=param_slices,
                        density=current_density,
                        dense_mode=dense_mode,
                        out_mask_u8=state.current_mask_u8,
                        state=state,
                        group=group,
                        group_rank=group_rank,
                        profile_tag='mask.current_sync',
                    )
                    state.has_current_mask = True

            self._build_mask_from_corrected(
                grad_data=grad_data,
                residual=state.residual,
                param_slices=param_slices,
                density=next_density,
                dense_mode=(force_all_reduce or (int(train_iter) + 1) < self._start_iter),
                out_mask_u8=state.next_mask_u8,
                group_rank=group_rank,
                profile_tag='mask.next_async',
            )
            self._launch_next_mask_allreduce_async(state, group)

            current_mask = state.current_mask_u8.bool()
            total_selected += int(current_mask.sum().item())
            total_numel += int(current_mask.numel())

            sparse_slice_plans: List[_SparseSlicePlan] = []
            payload_chunks: List[torch.Tensor] = []
            payload_offset = 0

            for param_slice in param_slices:
                start = param_slice.start
                end = param_slice.end
                param = param_slice.param
                if end <= start:
                    continue

                param_state = self._ensure_optimizer_state_entry(param)
                exp_avg = param_state['exp_avg'].view(-1)
                exp_avg_prev = exp_avg.to(torch.float32).clone()

                group_cfg = self._param_to_group[param]
                beta1, beta2 = group_cfg.get('betas', (0.9, 0.999))
                beta1 = float(beta1)
                beta2 = float(beta2)
                eps = float(group_cfg.get('eps', 1.0e-8))
                bias_correction = bool(group_cfg.get('bias_correction', True))
                weight_decay = self._group_weight_decay(group_cfg)

                grad_slice = grad_data[start:end]
                grad_fp32 = grad_slice.to(torch.float32)
                residual_slice = state.residual[start:end]
                mask_slice = current_mask[start:end]
                selected_count = int(mask_slice.sum().item())

                t_corrected = self._profile_tic(grad_slice)
                corrected = exp_avg_prev * beta1
                corrected.add_(grad_fp32, alpha=1.0 - beta1)
                corrected.add_(residual_slice)
                self._profile_toc('sparse.layer.corrected_ms', t_corrected, grad_slice)

                layer_profile: Dict[str, Any] = {
                    'buffer_idx': int(buffer_idx),
                    'start': int(start),
                    'end': int(end),
                    'numel': int(end - start),
                    'selected': int(selected_count),
                    'mask_topk_ms': 0.0,
                    'payload_allreduce_ms': 0.0,
                    'decompress_scatter_ms': 0.0,
                    'grad_copyback_ms': 0.0,
                    'residual_update_ms': 0.0,
                    'update_compute_ms': 0.0,
                }

                t_residual = self._profile_tic(residual_slice)
                residual_slice.copy_(corrected)
                if selected_count > 0:
                    residual_slice.masked_fill_(mask_slice, 0.0)
                    layer_profile['residual_update_ms'] += self._profile_toc(
                        'sparse.layer.residual_update_ms',
                        t_residual,
                        residual_slice,
                    )
                else:
                    layer_profile['residual_update_ms'] += self._profile_toc(
                        'sparse.layer.residual_copy_ms',
                        t_residual,
                        residual_slice,
                    )

                if selected_count > 0:
                    t_select_idx = self._profile_tic(mask_slice)
                    selected_idx = torch.nonzero(mask_slice, as_tuple=False).flatten()
                    selected = corrected.index_select(0, selected_idx)
                    self._profile_toc('sparse.layer.select_index_ms', t_select_idx, mask_slice)
                    payload_chunks.append(selected)

                sparse_slice_plans.append(
                    _SparseSlicePlan(
                        start=start,
                        end=end,
                        param=param,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        bias_correction=bias_correction,
                        weight_decay=weight_decay,
                        payload_offset=payload_offset,
                        payload_length=selected_count,
                        layer_profile=layer_profile,
                    )
                )
                payload_offset += selected_count

            total_payload_numel = int(payload_offset)
            synced_payload_flat: Optional[torch.Tensor] = None
            quant_error_flat: Optional[torch.Tensor] = None
            payload_total_ms = 0.0
            if total_payload_numel > 0:
                if len(payload_chunks) == 1:
                    packed_payload = payload_chunks[0]
                else:
                    t_pack = self._profile_tic(payload_chunks[0])
                    packed_payload = torch.cat(payload_chunks, dim=0)
                    self._profile_toc('sparse.buffer.payload_pack_ms', t_pack, packed_payload)

                t_payload = self._profile_tic(packed_payload)
                if use_fp8_quantized_payload:
                    synced_payload_flat, quant_error_flat = self._fp8_quantized_allreduce_packed(
                        packed_payload,
                        group,
                        train_iter=int(train_iter),
                        profile_key_prefix='sparse.buffer.fp8',
                    )
                else:
                    synced_payload_flat = packed_payload
                    self._allreduce_average_(
                        synced_payload_flat,
                        group,
                        profile_key='sparse.buffer.payload_fp32',
                    )
                payload_total_ms = self._profile_toc(
                    'sparse.buffer.payload_total_ms',
                    t_payload,
                    packed_payload,
                )

                inv_total_payload = 1.0 / float(total_payload_numel)
                for slice_plan in sparse_slice_plans:
                    if slice_plan.payload_length <= 0:
                        continue
                    slice_plan.layer_profile['payload_allreduce_ms'] += (
                        payload_total_ms
                        * float(slice_plan.payload_length)
                        * inv_total_payload
                    )

            for slice_plan in sparse_slice_plans:
                start = slice_plan.start
                end = slice_plan.end
                param = slice_plan.param
                if end <= start:
                    continue

                param_state = self._ensure_optimizer_state_entry(param)
                exp_avg = param_state['exp_avg'].view(-1)
                exp_avg_prev = exp_avg.to(torch.float32).clone()

                grad_slice = grad_data[start:end]
                residual_slice = state.residual[start:end]
                mask_slice = current_mask[start:end]

                synced_momentum = torch.zeros_like(exp_avg_prev)
                if slice_plan.payload_length > 0:
                    selected_idx = torch.nonzero(mask_slice, as_tuple=False).flatten()
                    if int(selected_idx.numel()) != int(slice_plan.payload_length):
                        raise RuntimeError(
                            "Top-k payload packing mismatch: selected count changed "
                            "between pack and unpack passes."
                        )
                    if synced_payload_flat is None:
                        raise RuntimeError("Internal error: packed sparse payload is missing.")

                    payload_start = int(slice_plan.payload_offset)
                    payload_end = payload_start + int(slice_plan.payload_length)
                    synced_selected = synced_payload_flat[payload_start:payload_end]

                    t_scatter = self._profile_tic(synced_momentum)
                    synced_momentum.index_copy_(0, selected_idx, synced_selected)
                    slice_plan.layer_profile['decompress_scatter_ms'] += self._profile_toc(
                        'sparse.layer.decompress_scatter_ms',
                        t_scatter,
                        synced_momentum,
                    )

                    if quant_error_flat is not None:
                        residual_selected = residual_slice.index_select(0, selected_idx)
                        residual_selected.add_(quant_error_flat[payload_start:payload_end])
                        residual_slice.index_copy_(0, selected_idx, residual_selected)

                if abs(1.0 - slice_plan.beta1) < 1e-12:
                    synced_grad = torch.zeros_like(synced_momentum)
                else:
                    synced_grad = (
                        synced_momentum - exp_avg_prev * slice_plan.beta1
                    ) / (1.0 - slice_plan.beta1)
                synced_grad.mul_(mask_slice.to(dtype=synced_grad.dtype))
                if synced_grad_sq_sum is None:
                    synced_grad_sq_sum = torch.zeros((), dtype=torch.float64, device=synced_grad.device)
                if param_is_not_shared(
                    param
                ) and tensor_parallel.param_is_not_tensor_parallel_duplicate(param):
                    synced_grad_sq_sum.add_(synced_grad.to(torch.float64).pow(2).sum())

                if move_clip_grad_to_reducer:
                    synced_grad_buffer = synced_grad_buffers[buffer_idx]
                    if synced_grad_buffer is None:
                        raise RuntimeError("Internal error: synced_grad_buffer is not initialized.")
                    t_syncbuf_copy = self._profile_tic(synced_grad_buffer)
                    synced_grad_buffer[start:end].copy_(synced_grad)
                    self._profile_toc(
                        'sparse.layer.synced_grad_buffer_copy_ms',
                        t_syncbuf_copy,
                        synced_grad_buffer,
                    )
                    self._profile_add_layer_record(slice_plan.layer_profile)
                    pending_updates.append(
                        _PendingUpdate(
                            buffer_idx=buffer_idx,
                            start=start,
                            end=end,
                            param=param,
                            beta1=slice_plan.beta1,
                            beta2=slice_plan.beta2,
                            eps=slice_plan.eps,
                            bias_correction=slice_plan.bias_correction,
                            weight_decay=slice_plan.weight_decay,
                        )
                    )
                    continue

                exp_avg.copy_(synced_momentum.to(exp_avg.dtype))
                param_state['momentum_buffer'] = param_state['exp_avg']

                variance = exp_avg_prev * exp_avg_prev
                variance.mul_(slice_plan.beta2)
                variance.addcmul_(synced_grad, synced_grad, value=1.0 - slice_plan.beta2)

                if slice_plan.bias_correction:
                    bias_correction1 = 1.0 - slice_plan.beta1**adam_step
                    bias_correction2 = 1.0 - slice_plan.beta2**adam_step
                    inv_bias1 = 1.0 / bias_correction1 if bias_correction1 != 0.0 else 1.0
                    inv_sqrt_bias2 = (
                        1.0 / math.sqrt(bias_correction2) if bias_correction2 > 0.0 else 1.0
                    )
                else:
                    inv_bias1 = 1.0
                    inv_sqrt_bias2 = 1.0

                denom = variance.sqrt()
                denom.mul_(inv_sqrt_bias2)
                denom.add_(slice_plan.eps)

                t_update = self._profile_tic(synced_momentum)
                update = synced_momentum * inv_bias1
                update.div_(denom)
                if slice_plan.weight_decay != 0.0:
                    update.add_(
                        self._model_or_optim_param_flat(param).to(torch.float32),
                        alpha=slice_plan.weight_decay,
                    )
                slice_plan.layer_profile['update_compute_ms'] += self._profile_toc(
                    'sparse.layer.update_compute_ms',
                    t_update,
                    synced_momentum,
                )
                if update_metric_sq_sum is None:
                    update_metric_sq_sum = torch.zeros((), dtype=torch.float64, device=update.device)
                update_metric_sq_sum.add_(update.to(torch.float64).pow(2).sum())

                t_copyback = self._profile_tic(grad_slice)
                grad_slice.copy_(update.to(grad_slice.dtype))
                slice_plan.layer_profile['grad_copyback_ms'] += self._profile_toc(
                    'sparse.layer.grad_copyback_ms',
                    t_copyback,
                    grad_slice,
                )
                self._profile_add_layer_record(slice_plan.layer_profile)

        reduced_synced_grad_sq_sum: Optional[torch.Tensor] = None
        if synced_grad_sq_sum is not None:
            reduced_synced_grad_sq_sum = synced_grad_sq_sum.clone()
            if (
                self._grad_stats_parallel_group is not None
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self._group_size(self._grad_stats_parallel_group) > 1
            ):
                t_gradnorm_ar = self._profile_tic(reduced_synced_grad_sq_sum)
                torch.distributed.all_reduce(
                    reduced_synced_grad_sq_sum,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self._grad_stats_parallel_group,
                )
                self._profile_toc(
                    'clip.grad_stats_allreduce_ms',
                    t_gradnorm_ar,
                    reduced_synced_grad_sq_sum,
                )

        if move_clip_grad_to_reducer and pending_updates:
            clip_coeff = 1.0
            if (
                self._clip_grad_max_norm > 0.0
                and reduced_synced_grad_sq_sum is not None
                and float(reduced_synced_grad_sq_sum.item()) > 0.0
            ):
                synced_grad_norm = float(
                    torch.sqrt(torch.clamp_min(reduced_synced_grad_sq_sum, 0.0)).item()
                )
                clip_coeff = self._clip_grad_max_norm / (synced_grad_norm + 1.0e-6)

            if clip_coeff < 1.0:
                for pending in pending_updates:
                    synced_grad_buffer = synced_grad_buffers[pending.buffer_idx]
                    if synced_grad_buffer is None:
                        raise RuntimeError(
                            "Internal error: synced_grad_buffer is not initialized."
                        )
                    t_clip = self._profile_tic(synced_grad_buffer)
                    synced_grad_buffer[pending.start : pending.end].mul_(clip_coeff)
                    self._profile_toc('pending.clip_scale_ms', t_clip, synced_grad_buffer)

            for pending in pending_updates:
                synced_grad_buffer = synced_grad_buffers[pending.buffer_idx]
                if synced_grad_buffer is None:
                    raise RuntimeError("Internal error: synced_grad_buffer is not initialized.")

                grad_data = self._buffers[pending.buffer_idx].grad_data
                grad_slice = grad_data[pending.start : pending.end]
                t_fetch_grad = self._profile_tic(synced_grad_buffer)
                synced_grad = synced_grad_buffer[pending.start : pending.end].to(torch.float32)
                self._profile_toc('pending.fetch_synced_grad_ms', t_fetch_grad, synced_grad_buffer)

                param_state = self._ensure_optimizer_state_entry(pending.param)
                exp_avg = param_state['exp_avg'].view(-1)
                exp_avg_prev = exp_avg.to(torch.float32).clone()

                mask_slice = (
                    self._buffer_states[pending.buffer_idx]
                    .current_mask_u8[pending.start : pending.end]
                    .to(torch.float32)
                )
                synced_momentum = exp_avg_prev * pending.beta1
                synced_momentum.add_(synced_grad, alpha=1.0 - pending.beta1)
                synced_momentum.mul_(mask_slice)
                exp_avg.copy_(synced_momentum.to(exp_avg.dtype))
                param_state['momentum_buffer'] = param_state['exp_avg']

                variance = exp_avg_prev * exp_avg_prev
                variance.mul_(pending.beta2)
                variance.addcmul_(synced_grad, synced_grad, value=1.0 - pending.beta2)

                if pending.bias_correction:
                    bias_correction1 = 1.0 - pending.beta1**adam_step
                    bias_correction2 = 1.0 - pending.beta2**adam_step
                    inv_bias1 = 1.0 / bias_correction1 if bias_correction1 != 0.0 else 1.0
                    inv_sqrt_bias2 = (
                        1.0 / math.sqrt(bias_correction2) if bias_correction2 > 0.0 else 1.0
                    )
                else:
                    inv_bias1 = 1.0
                    inv_sqrt_bias2 = 1.0

                denom = variance.sqrt()
                denom.mul_(inv_sqrt_bias2)
                denom.add_(pending.eps)

                t_update = self._profile_tic(synced_momentum)
                update = synced_momentum * inv_bias1
                update.div_(denom)
                if pending.weight_decay != 0.0:
                    update.add_(
                        self._model_or_optim_param_flat(pending.param).to(torch.float32),
                        alpha=pending.weight_decay,
                    )
                self._profile_toc('pending.update_compute_ms', t_update, synced_momentum)
                if update_metric_sq_sum is None:
                    update_metric_sq_sum = torch.zeros((), dtype=torch.float64, device=update.device)
                update_metric_sq_sum.add_(update.to(torch.float64).pow(2).sum())

                t_copyback = self._profile_tic(grad_slice)
                grad_slice.copy_(update.to(grad_slice.dtype))
                self._profile_toc('pending.grad_copyback_ms', t_copyback, grad_slice)

        if synced_grad_sq_sum is None:
            self._last_synced_grad_norm = 0.0
        elif reduced_synced_grad_sq_sum is not None:
            self._last_synced_grad_norm = float(
                torch.sqrt(torch.clamp_min(reduced_synced_grad_sq_sum, 0.0)).item()
            )
        else:
            self._last_synced_grad_norm = float(
                torch.sqrt(torch.clamp_min(synced_grad_sq_sum, 0.0)).item()
            )

        if update_metric_sq_sum is None:
            self._last_update_metric_norm = 0.0
        else:
            self._last_update_metric_norm = float(
                torch.sqrt(torch.clamp_min(update_metric_sq_sum, 0.0)).item()
            )
        self._last_norm_step = int(train_iter)

        is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        if total_numel > 0 and is_rank0:
            logger.info(
                "%s step=%d density=%f selected=%d total=%d",
                self.__class__.__name__,
                int(train_iter),
                float(total_selected) / float(total_numel),
                total_selected,
                total_numel,
            )

        self._profile_toc('reduce.total_ms', reduce_t0)
        self._log_profile_step(
            train_iter=int(train_iter),
            current_density=float(current_density),
            next_density=float(next_density),
            dense_mode=bool(dense_mode),
            use_fp8_quantized_payload=bool(use_fp8_quantized_payload),
            total_selected=int(total_selected),
            total_numel=int(total_numel),
        )

    def get_last_synced_grad_norm(self) -> Optional[float]:
        return self._last_synced_grad_norm

    def get_last_update_metric_norm(self) -> Optional[float]:
        return self._last_update_metric_norm

    def state_dict(self) -> Dict[str, Any]:
        self._flush_pending_next_mask_allreduces()
        buffer_states: List[Dict[str, Any]] = []
        for state in self._buffer_states:
            buffer_states.append(
                {
                    'residual': state.residual.detach().cpu().clone(),
                    'current_mask_u8': state.current_mask_u8.detach().to(torch.uint8).cpu().clone(),
                    'next_mask_u8': state.next_mask_u8.detach().to(torch.uint8).cpu().clone(),
                    'has_current_mask': bool(state.has_current_mask),
                    'has_next_mask': bool(state.has_next_mask),
                }
            )
        fp8_layer_scales = [
            (int(key[0]), int(key[1]), int(key[2]), float(value))
            for key, value in self._layer_fp8_scales.items()
        ]
        return {
            'version': 2,
            'buffer_states': buffer_states,
            'prepared_iteration': self._prepared_iteration,
            'fp8_layer_scales': fp8_layer_scales,
            'last_fp8_scale_update_iter': int(self._last_fp8_scale_update_iter),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return
        loaded_buffer_states = state_dict.get('buffer_states', [])
        if not isinstance(loaded_buffer_states, list):
            return
        self._flush_pending_next_mask_allreduces()

        loaded_buffers = 0
        loaded_current_masks = 0
        loaded_next_masks = 0

        for idx, loaded in enumerate(loaded_buffer_states):
            if idx >= len(self._buffer_states):
                break
            if not isinstance(loaded, dict):
                continue

            state = self._buffer_states[idx]
            loaded_buffers += 1

            residual = loaded.get('residual', None)
            residual_ok = (
                isinstance(residual, torch.Tensor)
                and residual.shape == state.residual.shape
                and residual.numel() == state.residual.numel()
            )
            if residual_ok:
                state.residual.copy_(residual.to(device=state.residual.device, dtype=state.residual.dtype))
            else:
                state.residual.zero_()

            current_mask = loaded.get('current_mask_u8', None)
            current_mask_loaded = (
                isinstance(current_mask, torch.Tensor)
                and current_mask.shape == state.current_mask_u8.shape
                and current_mask.numel() == state.current_mask_u8.numel()
            )
            if current_mask_loaded:
                state.current_mask_u8.copy_(
                    current_mask.to(device=state.current_mask_u8.device, dtype=torch.uint8)
                )
            else:
                state.current_mask_u8.zero_()

            next_mask = loaded.get('next_mask_u8', None)
            next_mask_loaded = (
                isinstance(next_mask, torch.Tensor)
                and next_mask.shape == state.next_mask_u8.shape
                and next_mask.numel() == state.next_mask_u8.numel()
            )
            if next_mask_loaded:
                state.next_mask_u8.copy_(
                    next_mask.to(device=state.next_mask_u8.device, dtype=torch.uint8)
                )
            else:
                state.next_mask_u8.zero_()

            # Backward-compatibility: if explicit mask-valid flags are absent in older
            # checkpoints, infer validity from whether masks were loaded successfully.
            current_mask_flag = loaded.get('has_current_mask', current_mask_loaded)
            next_mask_flag = loaded.get('has_next_mask', next_mask_loaded)

            state.has_current_mask = bool(current_mask_flag) and bool(current_mask_loaded)
            state.next_mask_allreduce_handle = None
            state.next_mask_allreduce_uses_packed = False
            state.has_next_mask = bool(next_mask_flag) and bool(next_mask_loaded)

            if state.has_current_mask:
                loaded_current_masks += 1
            if state.has_next_mask:
                loaded_next_masks += 1

        prepared = state_dict.get('prepared_iteration', None)
        try:
            self._prepared_iteration = int(prepared) if prepared is not None else None
        except (TypeError, ValueError):
            self._prepared_iteration = None

        loaded_scales = state_dict.get('fp8_layer_scales', [])
        restored_scales: Dict[Tuple[int, int, int], float] = {}
        if isinstance(loaded_scales, (list, tuple)):
            for entry in loaded_scales:
                if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                    continue
                try:
                    buffer_idx = int(entry[0])
                    start = int(entry[1])
                    end = int(entry[2])
                    scale = float(entry[3])
                except (TypeError, ValueError):
                    continue
                if scale <= 0.0 or not math.isfinite(scale):
                    continue
                restored_scales[(buffer_idx, start, end)] = scale
        self._layer_fp8_scales = restored_scales

        last_scale_update = state_dict.get('last_fp8_scale_update_iter', -1)
        try:
            self._last_fp8_scale_update_iter = int(last_scale_update)
        except (TypeError, ValueError):
            self._last_fp8_scale_update_iter = -1

        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            logger.info(
                "%s restored reducer state: buffers=%d current_masks=%d next_masks=%d prepared_iteration=%s",
                self.__class__.__name__,
                int(loaded_buffers),
                int(loaded_current_masks),
                int(loaded_next_masks),
                str(self._prepared_iteration),
            )


class TopKPerLayerSyncMomentumAdamSReducerV2(TopKPerLayerSyncMomentumAdamSFP8ReducerV2):
    """MCore variant of TopKPerLayerSyncMomentumAdamSReducerV2 (non-FP8 payload)."""

    def __init__(
        self,
        buffers: List[_ParamAndGradBuffer],
        ddp_config: DistributedDataParallelConfig,
    ) -> None:
        # Share the exact same top-k AdamS schedule/config knobs as FP8 V2, while
        # explicitly disabling FP8 payload quantization for this reducer variant.
        super().__init__(buffers=buffers, ddp_config=replace(ddp_config, use_fp8_topk_quant=False))
