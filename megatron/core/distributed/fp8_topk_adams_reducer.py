# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import torch

from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBuffer

logger = logging.getLogger(__name__)


@dataclass
class _ParamSlice:
    start: int
    end: int
    param: torch.nn.Parameter
    owner: int


@dataclass
class _BufferState:
    residual: torch.Tensor
    current_mask_u8: torch.Tensor
    next_mask_u8: torch.Tensor
    has_current_mask: bool = False
    has_next_mask: bool = False


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
        self._start_iter = max(0, int(ddp_config.topk_adams_start_iter))
        self._use_fp8_topk_quant = bool(ddp_config.use_fp8_topk_quant)

        self._fp8_dtype = torch.float8_e5m2
        self._fp8_scale_eps = 1e-6
        self._fp8_scale_interval = 100
        self._fp8_allreduce_warned = False
        self._layer_fp8_scales: Dict[Tuple[int, int, int], float] = {}
        self._last_fp8_scale_update_iter = -1

        self._prepared_iteration: Optional[int] = None

        self._buffer_param_slices: List[List[_ParamSlice]] = []
        self._buffer_states: List[_BufferState] = []
        self._buffer_group_ranks: List[int] = []

        for buffer in self._buffers:
            grad_data = buffer.grad_data
            numel = int(grad_data.numel())
            group = buffer.data_parallel_group
            group_size = max(1, self._group_size(group))
            group_rank = self._group_rank(group)
            self._buffer_group_ranks.append(group_rank)

            state = _BufferState(
                residual=torch.zeros(numel, dtype=torch.float32, device=grad_data.device),
                current_mask_u8=torch.zeros(numel, dtype=torch.uint8, device=grad_data.device),
                next_mask_u8=torch.zeros(numel, dtype=torch.uint8, device=grad_data.device),
            )
            self._buffer_states.append(state)

            sorted_slices: List[Tuple[int, int, torch.nn.Parameter]] = []
            for param in buffer.params:
                start, end, _ = buffer.param_index_map[param]
                sorted_slices.append((int(start), int(end), param))
            sorted_slices.sort(key=lambda item: item[0])
            param_slices: List[_ParamSlice] = []
            for idx, (start, end, param) in enumerate(sorted_slices):
                owner = int(idx % group_size)
                param_slices.append(
                    _ParamSlice(start=start, end=end, param=param, owner=owner)
                )
            self._buffer_param_slices.append(param_slices)

        self._param_to_group: Dict[torch.nn.Parameter, Dict[str, Any]] = {}
        self._param_to_state_dict: Dict[
            torch.nn.Parameter, Dict[torch.nn.Parameter, Dict[str, Any]]
        ] = {}
        self._param_to_optim_param: Dict[torch.nn.Parameter, torch.Tensor] = {}
        self._warned_missing_mapping = False
        self._last_synced_grad_norm: Optional[float] = None
        self._last_update_metric_norm: Optional[float] = None
        self._last_norm_step: Optional[int] = None

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

    def set_optimizer(self, optimizer: Any) -> None:
        """Attach optimizer state views used by the reducer."""
        self._param_to_group.clear()
        self._param_to_state_dict.clear()
        self._param_to_optim_param.clear()

        optim_param_to_group: Dict[torch.Tensor, Dict[str, Any]] = {}
        optim_param_to_state_dict: Dict[torch.Tensor, Dict[torch.Tensor, Dict[str, Any]]] = {}

        for wrapper in self._iter_optimizer_wrappers(optimizer):
            if getattr(wrapper, 'is_stub_optimizer', False):
                continue

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
            if state.has_next_mask:
                state.current_mask_u8.copy_(state.next_mask_u8)
                state.has_current_mask = True
        self._prepared_iteration = int(train_iter)

    def _scheduled_density(self, train_iter: int) -> float:
        if train_iter < self._start_iter:
            return 1.0

        target = min(max(self._target_density, 0.0), 1.0)
        start = min(max(self._density_start, 0.0), 1.0)
        if self._warmup_steps <= 0:
            return target

        progress = min(max(train_iter - self._start_iter, 0), self._warmup_steps)
        alpha = float(progress) / float(self._warmup_steps)
        return start + (target - start) * alpha

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
        return max(1, int(math.ceil(density * numel)))

    def _build_mask_from_corrected(
        self,
        grad_data: torch.Tensor,
        residual: torch.Tensor,
        param_slices: List[_ParamSlice],
        density: float,
        dense_mode: bool,
        out_mask_u8: torch.Tensor,
        group_rank: int,
    ) -> None:
        out_mask_u8.zero_()

        for param_slice in param_slices:
            start = param_slice.start
            end = param_slice.end
            param = param_slice.param
            numel = end - start
            if numel <= 0:
                continue
            if int(param_slice.owner) != int(group_rank):
                continue

            if dense_mode or density >= 1.0:
                out_mask_u8[start:end] = 1
                continue

            k = self._topk_k(density, numel)
            if k <= 0:
                continue
            if k >= numel:
                out_mask_u8[start:end] = 1
                continue

            state = self._optimizer_state_entry(param)
            if state is None or 'exp_avg' not in state:
                out_mask_u8[start:end] = 1
                continue

            exp_avg_prev = state['exp_avg'].view(-1).to(torch.float32)
            group = self._param_to_group.get(param, {})
            beta1 = float(group.get('betas', (0.9, 0.999))[0])

            grad_slice_fp32 = grad_data[start:end].to(torch.float32)
            corrected = exp_avg_prev * beta1
            corrected.add_(grad_slice_fp32, alpha=1.0 - beta1)
            corrected.add_(residual[start:end])

            _, topk_idx = torch.topk(corrected.abs(), k=k, sorted=False)
            out_mask_u8[start:end].index_fill_(0, topk_idx, 1)

    def _allreduce_max_mask_(
        self, mask_u8: torch.Tensor, group: torch.distributed.ProcessGroup
    ) -> None:
        world_size = self._group_size(group)
        if world_size <= 1 or mask_u8.numel() == 0:
            return
        torch.distributed.all_reduce(mask_u8, op=torch.distributed.ReduceOp.MAX, group=group)

    def _build_synced_mask_from_corrected(
        self,
        grad_data: torch.Tensor,
        residual: torch.Tensor,
        param_slices: List[_ParamSlice],
        density: float,
        dense_mode: bool,
        out_mask_u8: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        group_rank: int,
    ) -> None:
        self._build_mask_from_corrected(
            grad_data=grad_data,
            residual=residual,
            param_slices=param_slices,
            density=density,
            dense_mode=dense_mode,
            out_mask_u8=out_mask_u8,
            group_rank=group_rank,
        )
        self._allreduce_max_mask_(out_mask_u8, group)

    def _allreduce_average_(self, tensor: torch.Tensor, group: torch.distributed.ProcessGroup) -> None:
        world_size = self._group_size(group)
        if world_size <= 1 or tensor.numel() == 0:
            return
        torch.distributed.all_reduce(tensor, group=group)
        tensor.div_(world_size)

    def _fp8_allreduce_(
        self, tensor: torch.Tensor, group: torch.distributed.ProcessGroup
    ) -> None:
        try:
            torch.distributed.all_reduce(tensor, group=group)
        except Exception as exc:
            if not self._fp8_allreduce_warned:
                logger.warning(
                    "FP8 all-reduce failed in top-k reducer; falling back to FP32 all-reduce. "
                    "Exception: %s",
                    exc,
                )
                self._fp8_allreduce_warned = True
            fallback = tensor.to(torch.float32)
            torch.distributed.all_reduce(fallback, group=group)
            tensor.copy_(fallback.to(tensor.dtype))

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
        self, selected_values: torch.Tensor, group: torch.distributed.ProcessGroup
    ) -> float:
        abs_selected = selected_values.abs()
        nonzero = abs_selected > 0
        if bool(nonzero.any().item()):
            exponent = torch.floor(torch.log2(abs_selected[nonzero])).max()
        else:
            exponent = torch.tensor(
                float("-inf"), dtype=torch.float32, device=selected_values.device
            )
        if self._group_size(group) > 1:
            torch.distributed.all_reduce(exponent, op=torch.distributed.ReduceOp.MAX, group=group)
        return float(exponent.item())

    def _fp8_quantized_allreduce(
        self,
        selected_values: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        layer_key: Tuple[int, int, int],
        train_iter: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize selected values to FP8 with shared layer-wise scale before all-reduce."""
        if selected_values.numel() == 0:
            empty = selected_values.to(torch.float32)
            return empty, empty

        selected_fp32 = selected_values.to(torch.float32)
        world_size = self._group_size(group)
        should_refresh = self._should_refresh_layer_scales(train_iter)
        refreshed_scale = False
        if should_refresh or layer_key not in self._layer_fp8_scales:
            exponent = self._global_layer_max_exponent(selected_fp32, group)
            self._layer_fp8_scales[layer_key] = self._max_exp_to_scale(
                exponent, self._fp8_scale_eps
            )
            refreshed_scale = True
        layer_scale = float(self._layer_fp8_scales.get(layer_key, 1.0))

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

        quant_fp8 = normalized.to(self._fp8_dtype)
        local_compressed_fp32 = quant_fp8.to(torch.float32)
        local_compressed_fp32.mul_(layer_scale * n_workers_f)
        quant_error = selected_fp32 - local_compressed_fp32

        if world_size > 1:
            self._fp8_allreduce_(quant_fp8, group)

        synced_selected = quant_fp8.to(torch.float32)
        synced_selected.mul_(layer_scale)

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
    ) -> None:
        self._allreduce_average_(grad_data, group)
        state.residual.zero_()
        state.current_mask_u8.fill_(1)
        state.next_mask_u8.fill_(1)
        state.has_current_mask = True
        state.has_next_mask = True

    @torch.no_grad()
    def reduce(self, train_iter: int, force_all_reduce: bool = False) -> None:
        self._prepared_iteration = None

        current_density = self._scheduled_density(int(train_iter))
        next_density = self._scheduled_density(int(train_iter) + 1)

        dense_mode = bool(force_all_reduce or int(train_iter) < self._start_iter)
        if force_all_reduce:
            current_density = 1.0
            next_density = 1.0
        use_fp8_quantized_payload = self._use_fp8_quantized_payload(int(train_iter))

        total_selected = 0
        total_numel = 0
        synced_grad_sq_sum: Optional[torch.Tensor] = None
        update_metric_sq_sum: Optional[torch.Tensor] = None

        for buffer_idx, buffer in enumerate(self._buffers):
            grad_data = buffer.grad_data
            if grad_data is None or grad_data.numel() == 0:
                continue

            state = self._buffer_states[buffer_idx]
            param_slices = self._buffer_param_slices[buffer_idx]
            group = buffer.data_parallel_group
            group_rank = self._buffer_group_ranks[buffer_idx]

            if not param_slices:
                self._reduce_dense_fallback(grad_data, group, state)
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
                self._reduce_dense_fallback(grad_data, group, state)
                continue

            if not state.has_current_mask:
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
                        group=group,
                        group_rank=group_rank,
                    )
                    state.has_current_mask = True

            self._build_synced_mask_from_corrected(
                grad_data=grad_data,
                residual=state.residual,
                param_slices=param_slices,
                density=next_density,
                dense_mode=(force_all_reduce or (int(train_iter) + 1) < self._start_iter),
                out_mask_u8=state.next_mask_u8,
                group=group,
                group_rank=group_rank,
            )
            state.has_next_mask = True

            current_mask = state.current_mask_u8.bool()
            total_selected += int(current_mask.sum().item())
            total_numel += int(current_mask.numel())

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

                corrected = exp_avg_prev * beta1
                corrected.add_(grad_fp32, alpha=1.0 - beta1)
                corrected.add_(residual_slice)

                synced_momentum = torch.zeros_like(exp_avg_prev)
                if bool(mask_slice.any().item()):
                    selected_idx = torch.nonzero(mask_slice, as_tuple=False).flatten()
                    selected = corrected.index_select(0, selected_idx)
                    if use_fp8_quantized_payload:
                        layer_key = (int(buffer_idx), int(start), int(end))
                        synced_selected, quant_error = self._fp8_quantized_allreduce(
                            selected, group, layer_key=layer_key, train_iter=int(train_iter)
                        )
                    else:
                        synced_selected = selected.clone()
                        self._allreduce_average_(synced_selected, group)
                        quant_error = torch.zeros_like(synced_selected)
                    synced_momentum.index_copy_(0, selected_idx, synced_selected)

                    residual_slice.copy_(corrected)
                    residual_slice.masked_fill_(mask_slice, 0.0)
                    if use_fp8_quantized_payload:
                        residual_selected = residual_slice.index_select(0, selected_idx)
                        residual_selected.add_(quant_error)
                        residual_slice.index_copy_(0, selected_idx, residual_selected)
                else:
                    residual_slice.copy_(corrected)

                exp_avg.copy_(synced_momentum.to(exp_avg.dtype))
                param_state['momentum_buffer'] = param_state['exp_avg']

                step = int(param_state.get('step', 0)) + 1
                param_state['step'] = step

                if abs(1.0 - beta1) < 1e-12:
                    synced_grad = torch.zeros_like(synced_momentum)
                else:
                    synced_grad = (synced_momentum - exp_avg_prev * beta1) / (1.0 - beta1)
                synced_grad.mul_(mask_slice.to(dtype=synced_grad.dtype))
                if synced_grad_sq_sum is None:
                    synced_grad_sq_sum = torch.zeros((), dtype=torch.float64, device=synced_grad.device)
                synced_grad_sq_sum.add_(synced_grad.to(torch.float64).pow(2).sum())

                variance = exp_avg_prev * exp_avg_prev
                variance.mul_(beta2)
                variance.addcmul_(synced_grad, synced_grad, value=1.0 - beta2)

                if bias_correction:
                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
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
                    update.add_(self._model_or_optim_param_flat(param).to(torch.float32), alpha=weight_decay)
                if update_metric_sq_sum is None:
                    update_metric_sq_sum = torch.zeros((), dtype=torch.float64, device=update.device)
                update_metric_sq_sum.add_(update.to(torch.float64).pow(2).sum())

                grad_slice.copy_(update.to(grad_slice.dtype))

        if synced_grad_sq_sum is None:
            self._last_synced_grad_norm = 0.0
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

    def get_last_synced_grad_norm(self) -> Optional[float]:
        return self._last_synced_grad_norm

    def get_last_update_metric_norm(self) -> Optional[float]:
        return self._last_update_metric_norm

    def state_dict(self) -> Dict[str, Any]:
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

        for idx, loaded in enumerate(loaded_buffer_states):
            if idx >= len(self._buffer_states):
                break
            if not isinstance(loaded, dict):
                continue

            state = self._buffer_states[idx]

            residual = loaded.get('residual', None)
            if isinstance(residual, torch.Tensor) and residual.numel() == state.residual.numel():
                state.residual.copy_(residual.to(device=state.residual.device, dtype=state.residual.dtype))
            else:
                state.residual.zero_()

            current_mask = loaded.get('current_mask_u8', None)
            if (
                isinstance(current_mask, torch.Tensor)
                and current_mask.numel() == state.current_mask_u8.numel()
            ):
                state.current_mask_u8.copy_(
                    current_mask.to(device=state.current_mask_u8.device, dtype=torch.uint8)
                )
            else:
                state.current_mask_u8.zero_()

            next_mask = loaded.get('next_mask_u8', None)
            if isinstance(next_mask, torch.Tensor) and next_mask.numel() == state.next_mask_u8.numel():
                state.next_mask_u8.copy_(next_mask.to(device=state.next_mask_u8.device, dtype=torch.uint8))
            else:
                state.next_mask_u8.zero_()

            state.has_current_mask = bool(loaded.get('has_current_mask', False))
            state.has_next_mask = bool(loaded.get('has_next_mask', False))

        prepared = state_dict.get('prepared_iteration', None)
        self._prepared_iteration = int(prepared) if prepared is not None else None

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
