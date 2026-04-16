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
_TRITON_SPARSE_UPDATE_BLOCK = 1024

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

    @triton.jit
    def _non_topk_decay_kernel(
        full_param_ptr,
        mask_u8_ptr,
        decay_factor,
        numel,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < numel

        values = tl.load(full_param_ptr + offsets, mask=valid, other=0.0)
        mask_vals = tl.load(mask_u8_ptr + offsets, mask=valid, other=0).to(tl.int32)
        is_non_topk = mask_vals == 0
        out = tl.where(is_non_topk, values * decay_factor, values)
        tl.store(full_param_ptr + offsets, out, mask=valid)

    @triton.jit
    def _topk_scatter_writeback_kernel(
        full_param_ptr,
        selected_idx_ptr,
        selected_vals_ptr,
        num_selected,
        full_numel,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < num_selected

        selected_idx = tl.load(selected_idx_ptr + offsets, mask=valid, other=0).to(tl.int64)
        selected_vals = tl.load(selected_vals_ptr + offsets, mask=valid, other=0.0)
        store_mask = valid & (selected_idx >= 0) & (selected_idx < full_numel)
        tl.store(full_param_ptr + selected_idx, selected_vals, mask=store_mask)


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


def _launch_non_topk_decay_triton(
    full_param_fp32: torch.Tensor,
    mask_u8: torch.Tensor,
    decay_factor: float,
) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable for non-topk decay")
    if full_param_fp32.dtype != torch.float32:
        raise RuntimeError(
            f"Expected FP32 full-param tensor for Triton decay, got {full_param_fp32.dtype}."
        )
    if mask_u8.dtype != torch.uint8:
        raise RuntimeError(f"Expected uint8 mask tensor for Triton decay, got {mask_u8.dtype}.")
    if int(full_param_fp32.numel()) != int(mask_u8.numel()):
        raise RuntimeError(
            "Non-topk decay tensor size mismatch: "
            f"param={int(full_param_fp32.numel())}, mask={int(mask_u8.numel())}."
        )
    numel = int(full_param_fp32.numel())
    if numel <= 0:
        return
    grid = (triton.cdiv(numel, _TRITON_SPARSE_UPDATE_BLOCK),)
    _non_topk_decay_kernel[grid](
        full_param_fp32,
        mask_u8,
        float(decay_factor),
        numel,
        BLOCK_SIZE=_TRITON_SPARSE_UPDATE_BLOCK,
    )


def _launch_topk_scatter_writeback_triton(
    full_param_fp32: torch.Tensor,
    selected_idx: torch.Tensor,
    selected_vals_fp32: torch.Tensor,
) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable for top-k scatter writeback")
    if full_param_fp32.dtype != torch.float32:
        raise RuntimeError(
            f"Expected FP32 full-param tensor for Triton top-k writeback, got {full_param_fp32.dtype}."
        )
    if selected_vals_fp32.dtype != torch.float32:
        raise RuntimeError(
            "Expected FP32 selected-value tensor for Triton top-k writeback, "
            f"got {selected_vals_fp32.dtype}."
        )
    if selected_idx.dtype not in (torch.int64, torch.int32):
        raise RuntimeError(
            "Expected int32/int64 selected-index tensor for Triton top-k writeback, "
            f"got {selected_idx.dtype}."
        )
    if int(selected_idx.numel()) != int(selected_vals_fp32.numel()):
        raise RuntimeError(
            "Top-k scatter tensor size mismatch: "
            f"idx={int(selected_idx.numel())}, vals={int(selected_vals_fp32.numel())}."
        )
    num_selected = int(selected_idx.numel())
    if num_selected <= 0:
        return
    grid = (triton.cdiv(num_selected, _TRITON_SPARSE_UPDATE_BLOCK),)
    _topk_scatter_writeback_kernel[grid](
        full_param_fp32,
        selected_idx,
        selected_vals_fp32,
        num_selected,
        int(full_param_fp32.numel()),
        BLOCK_SIZE=_TRITON_SPARSE_UPDATE_BLOCK,
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
    residual_cpu: Optional[torch.Tensor]
    current_mask_u8: Optional[torch.Tensor]
    next_mask_u8: Optional[torch.Tensor]
    mask_pack_segments: Tuple[Tuple[int, int, int, int, int], ...] = tuple()
    current_mask_packed_u8: Optional[torch.Tensor] = None
    next_mask_packed_u8: Optional[torch.Tensor] = None
    mask_numel: int = 0
    use_packed_mask_only: bool = False
    has_current_mask: bool = False
    has_next_mask: bool = False
    next_mask_allreduce_handle: Optional[Any] = None
    next_mask_allreduce_event: Optional[Any] = None
    next_mask_allreduce_uses_packed: bool = False
    next_mask_allreduce_unpacked_u8: Optional[torch.Tensor] = None
    mask_allgather_segments_by_rank: Tuple[Tuple[Tuple[int, int, int, int], ...], ...] = tuple()
    mask_allgather_rank_payload_bytes: Tuple[int, ...] = tuple()
    mask_allgather_send_bytes: int = 0
    next_mask_allgather_send_u8: Optional[torch.Tensor] = None
    next_mask_allgather_recv_u8: Optional[torch.Tensor] = None


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
        self._offload_full_param_to_cpu = bool(
            getattr(ddp_config, 'topk_adams_full_param_cpu_offload', False)
        )
        self._offload_residual_to_cpu = bool(
            getattr(ddp_config, 'topk_adams_residual_cpu_offload', False)
        )
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
        self._payload_allreduce_stream: Optional[torch.cuda.Stream] = None
        self._param_offload_stream: Optional[torch.cuda.Stream] = None

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
        self._max_residual_stage_numel: int = 0
        self._residual_stage_buffers: List[Optional[torch.Tensor]] = [None, None]
        self._max_mask_stage_numel: int = 0
        self._mask_stage_buffers: List[Optional[torch.Tensor]] = [None]
        # Keep async next-mask sync staging disjoint from main-stream mask staging.
        # Sharing these pools can race when packed-mask unpack/pack runs concurrently
        # on different CUDA streams.
        self._mask_stage_buffers_async: List[Optional[torch.Tensor]] = [None]
        # A single fp32 scratch buffer is sufficient because per-layer work is
        # serialized; this trims one largest-layer-sized fp32 allocation.
        self._fp32_scratch_buffers: List[Optional[torch.Tensor]] = [None]

        if torch.cuda.is_available():
            try:
                self._async_mask_stream = torch.cuda.Stream(
                    device=torch.cuda.current_device()
                )
            except Exception:
                self._async_mask_stream = torch.cuda.Stream()
            try:
                self._payload_allreduce_stream = torch.cuda.Stream(
                    device=torch.cuda.current_device()
                )
            except Exception:
                self._payload_allreduce_stream = torch.cuda.Stream()
            if self._offload_full_param_to_cpu or self._offload_residual_to_cpu:
                try:
                    self._param_offload_stream = torch.cuda.Stream(
                        device=torch.cuda.current_device()
                    )
                except Exception:
                    self._param_offload_stream = torch.cuda.Stream()

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
                # Assign contiguous parameter slices by cumulative numel rather than
                # by slice count. This keeps ownership semantics unchanged (single
                # owner per slice, no overlap) while reducing all-gather padding
                # overhead for imbalanced layer sizes.
                total_slice_numel = int(
                    sum(max(0, int(end) - int(start)) for (start, end, _param) in sorted_slices)
                )
                target_numel_per_owner = (
                    float(total_slice_numel) / float(max(1, group_size))
                    if total_slice_numel > 0
                    else 0.0
                )
                owner = 0
                owner_numel = 0
                for cursor, (start, end, param) in enumerate(sorted_slices):
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
                    owner_numel += max(0, int(end) - int(start))
                    if owner < (group_size - 1):
                        remaining_slices = int(total_slices - (cursor + 1))
                        remaining_owners = int(group_size - (owner + 1))
                        reached_target = (
                            target_numel_per_owner > 0.0
                            and float(owner_numel) >= target_numel_per_owner
                        )
                        must_advance_for_coverage = remaining_slices <= remaining_owners
                        can_advance_for_balance = remaining_slices > remaining_owners
                        if (reached_target and can_advance_for_balance) or must_advance_for_coverage:
                            owner += 1
                            owner_numel = 0
            self._buffer_param_slices.append(param_slices)
            for param_slice in param_slices:
                slice_numel = int(param_slice.end - param_slice.start)
                self._max_residual_stage_numel = max(
                    self._max_residual_stage_numel,
                    slice_numel,
                )
                self._max_mask_stage_numel = max(
                    self._max_mask_stage_numel,
                    slice_numel,
                )

            mask_pack_segments, packed_mask_numel = self._build_mask_pack_layout(
                numel=numel,
                param_slices=param_slices,
            )
            (
                mask_allgather_segments_by_rank,
                mask_allgather_rank_payload_bytes,
                mask_allgather_send_bytes,
            ) = self._build_mask_allgather_layout(
                buffer=buffer,
                numel=numel,
                param_slices=param_slices,
                group_size=group_size,
            )
            if self._offload_residual_to_cpu:
                residual_storage = torch.zeros(numel, dtype=torch.float32, device='cpu')
                if torch.cuda.is_available():
                    try:
                        residual_storage = residual_storage.pin_memory()
                    except Exception:
                        pass
            else:
                residual_storage = torch.zeros(numel, dtype=torch.float32, device=grad_data.device)

            # Keep mask state in packed-bit format across all modes to minimize
            # persistent GPU memory usage. Per-slice uint8 views are staged on
            # demand via _mask_stage_buffers.
            use_packed_mask_only = True
            current_mask_u8: Optional[torch.Tensor]
            next_mask_u8: Optional[torch.Tensor]
            if use_packed_mask_only:
                current_mask_u8 = None
                next_mask_u8 = None
            else:
                current_mask_u8 = torch.zeros(numel, dtype=torch.uint8, device=grad_data.device)
                next_mask_u8 = torch.zeros(numel, dtype=torch.uint8, device=grad_data.device)

            state = _BufferState(
                residual=residual_storage,
                residual_cpu=residual_storage if self._offload_residual_to_cpu else None,
                current_mask_u8=current_mask_u8,
                next_mask_u8=next_mask_u8,
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
                mask_numel=numel,
                use_packed_mask_only=use_packed_mask_only,
                mask_allgather_segments_by_rank=mask_allgather_segments_by_rank,
                mask_allgather_rank_payload_bytes=mask_allgather_rank_payload_bytes,
                mask_allgather_send_bytes=mask_allgather_send_bytes,
            )
            self._buffer_states.append(state)

        if self._offload_residual_to_cpu and torch.cuda.is_available() and self._max_residual_stage_numel > 0:
            stage_numel = max(1, int(self._max_residual_stage_numel))
            try:
                stage_device = torch.device('cuda', torch.cuda.current_device())
                self._residual_stage_buffers[0] = torch.empty(
                    stage_numel,
                    dtype=torch.float32,
                    device=stage_device,
                )
                self._residual_stage_buffers[1] = torch.empty(
                    stage_numel,
                    dtype=torch.float32,
                    device=stage_device,
                )
            except Exception:
                self._residual_stage_buffers = [None, None]

        # Mask stage and fp32 scratch buffers are allocated lazily per slice and
        # grown only as needed, reducing startup-time GPU footprint.

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
        self._full_param_fp32: List[Optional[torch.Tensor]] = []
        self._warned_sparse_param_sync_overlap = False
        self._triton_sparse_nontopk_enabled = bool(_TRITON_AVAILABLE)
        self._triton_sparse_topk_enabled = bool(_TRITON_AVAILABLE)
        self._warned_triton_sparse_nontopk_fallback = False
        self._warned_triton_sparse_topk_fallback = False

        for buffer in self._buffers:
            param_data = buffer.param_data
            if param_data is None or param_data.numel() == 0:
                self._full_param_fp32.append(None)
            else:
                if self._offload_full_param_to_cpu:
                    full_param = param_data.detach().to(dtype=torch.float32, device='cpu').clone()
                    if torch.cuda.is_available():
                        try:
                            full_param = full_param.pin_memory()
                        except Exception:
                            pass
                    self._full_param_fp32.append(full_param)
                else:
                    self._full_param_fp32.append(param_data.detach().to(torch.float32).clone())

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

    def _log_profile_subset(self, step_hint: int, prefix: str, label: str) -> None:
        if not self._profile_enabled:
            return
        if not self._profile_should_log():
            return
        if step_hint >= 0 and int(step_hint) % self._profile_log_interval != 0:
            return

        subset_ms = [
            (key[len(prefix) :], total_ms)
            for key, total_ms in self._profile_step_ms.items()
            if key.startswith(prefix)
        ]
        subset_values = [
            (key[len(prefix) :], value)
            for key, value in self._profile_step_values.items()
            if key.startswith(prefix)
        ]
        if not subset_ms and not subset_values:
            return

        timing_parts: List[str] = []
        for key, total_ms in sorted(subset_ms, key=lambda item: item[1], reverse=True):
            calls = max(1, int(self._profile_step_calls.get(f'{prefix}{key}', 0)))
            timing_parts.append(
                f"{key}={total_ms:.3f}ms(calls={calls},avg={total_ms / calls:.3f}ms)"
            )

        value_parts = [f"{key}={value:.3f}" for key, value in sorted(subset_values)]
        step_label = str(int(step_hint)) if step_hint >= 0 else 'unknown'
        msg = f"{self.__class__.__name__} {label} profile step={step_label}"
        if timing_parts:
            msg += " timings: " + " | ".join(timing_parts)
        if value_parts:
            msg += " values: " + " | ".join(value_parts)
        logger.info(msg)

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
        if (not self._ddp_config.use_distributed_optimizer) and mapped_model_params < total_model_params:
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
                if state.use_packed_mask_only:
                    current_packed = state.current_mask_packed_u8
                    next_packed = state.next_mask_packed_u8
                    if current_packed is None or next_packed is None:
                        raise RuntimeError('Internal error: packed mask buffers are missing.')
                    current_packed.copy_(next_packed)
                else:
                    current_mask_u8 = state.current_mask_u8
                    next_mask_u8 = state.next_mask_u8
                    if current_mask_u8 is None or next_mask_u8 is None:
                        raise RuntimeError('Internal error: mask_u8 buffers are missing.')
                    current_mask_u8.copy_(next_mask_u8)
                state.has_current_mask = True
        self._prepared_iteration = int(train_iter)

    def can_replace_dense_param_all_gather(self) -> bool:
        """Whether sparse post-step sync can replace dense param all-gather."""
        return bool(
            self._ddp_config.use_distributed_optimizer
            and (not self._ddp_config.overlap_param_gather)
        )

    def _ensure_full_param_storage(
        self,
        buffer_idx: int,
        param_data: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        full_param = self._full_param_fp32[buffer_idx]
        if self._offload_full_param_to_cpu:
            needs_init = (
                full_param is None
                or full_param.numel() != param_data.numel()
                or full_param.device.type != 'cpu'
            )
            if needs_init:
                full_param = param_data.detach().to(dtype=torch.float32, device='cpu').clone()
                if torch.cuda.is_available():
                    try:
                        full_param = full_param.pin_memory()
                    except Exception:
                        pass
                self._full_param_fp32[buffer_idx] = full_param
            return full_param

        needs_init = (
            full_param is None
            or full_param.numel() != param_data.numel()
            or full_param.device != param_data.device
        )
        if needs_init:
            full_param = param_data.detach().to(torch.float32).clone()
            self._full_param_fp32[buffer_idx] = full_param
        return full_param

    def _prefetch_full_param_from_cpu(
        self,
        full_param_cpu: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.cuda.Event]]:
        full_param_gpu = torch.empty(
            full_param_cpu.numel(),
            dtype=torch.float32,
            device=device,
        )
        can_async = (
            self._param_offload_stream is not None
            and full_param_gpu.is_cuda
            and full_param_cpu.device.type == 'cpu'
            and full_param_cpu.is_pinned()
        )
        if can_async:
            with torch.cuda.stream(self._param_offload_stream):
                full_param_gpu.copy_(full_param_cpu, non_blocking=True)
                ready_event = torch.cuda.Event()
                ready_event.record(self._param_offload_stream)
            return full_param_gpu, ready_event

        full_param_gpu.copy_(full_param_cpu, non_blocking=False)
        return full_param_gpu, None

    def _queue_full_param_offload_to_cpu(
        self,
        full_param_gpu: torch.Tensor,
        full_param_cpu: torch.Tensor,
    ) -> bool:
        can_async = (
            self._param_offload_stream is not None
            and full_param_gpu.is_cuda
            and full_param_cpu.device.type == 'cpu'
            and full_param_cpu.is_pinned()
        )
        if can_async:
            current_stream = torch.cuda.current_stream(full_param_gpu.device)
            with torch.cuda.stream(self._param_offload_stream):
                self._param_offload_stream.wait_stream(current_stream)
                full_param_cpu.copy_(full_param_gpu, non_blocking=True)
            return True

        full_param_cpu.copy_(full_param_gpu.to(dtype=torch.float32, device='cpu'))
        return False
    
    
    
    def _residual_storage(self, state: _BufferState) -> torch.Tensor:
        return state.residual_cpu if state.residual_cpu is not None else state.residual

    def _residual_stage_view(
        self,
        stage_slot: int,
        numel: int,
        device: torch.device,
    ) -> torch.Tensor:
        if stage_slot not in (0, 1):
            stage_slot = int(stage_slot) & 1
        stage = self._residual_stage_buffers[stage_slot]
        need_alloc = (
            stage is None
            or stage.device != device
            or int(stage.numel()) < int(numel)
        )
        if need_alloc:
            alloc_numel = max(int(numel), int(self._max_residual_stage_numel), 1)
            stage = torch.empty(alloc_numel, dtype=torch.float32, device=device)
            self._residual_stage_buffers[stage_slot] = stage
        return stage[: int(numel)]

    def _prefetch_residual_slice_from_cpu(
        self,
        residual_cpu: torch.Tensor,
        start: int,
        end: int,
        device: torch.device,
        stage_slot: int,
    ) -> Tuple[torch.Tensor, Optional[torch.cuda.Event]]:
        numel = int(end - start)
        if numel <= 0:
            raise RuntimeError('Residual prefetch requested with non-positive slice length.')
        residual_gpu = self._residual_stage_view(stage_slot=stage_slot, numel=numel, device=device)
        residual_cpu_slice = residual_cpu[int(start) : int(end)]
        can_async = (
            self._param_offload_stream is not None
            and residual_gpu.is_cuda
            and residual_cpu.device.type == 'cpu'
            and residual_cpu.is_pinned()
        )
        if can_async:
            with torch.cuda.stream(self._param_offload_stream):
                residual_gpu.copy_(residual_cpu_slice, non_blocking=True)
                ready_event = torch.cuda.Event()
                ready_event.record(self._param_offload_stream)
            return residual_gpu, ready_event

        residual_gpu.copy_(residual_cpu_slice, non_blocking=False)
        return residual_gpu, None

    def _queue_residual_slice_offload_to_cpu(
        self,
        residual_gpu: torch.Tensor,
        residual_cpu: torch.Tensor,
        start: int,
        end: int,
    ) -> bool:
        residual_cpu_slice = residual_cpu[int(start) : int(end)]
        if int(residual_gpu.numel()) != int(residual_cpu_slice.numel()):
            raise RuntimeError(
                'Top-k residual offload found residual slice size mismatch: '
                f"gpu={int(residual_gpu.numel())} cpu={int(residual_cpu_slice.numel())}."
            )

        can_async = (
            self._param_offload_stream is not None
            and residual_gpu.is_cuda
            and residual_cpu.device.type == 'cpu'
            and residual_cpu.is_pinned()
        )
        if can_async:
            current_stream = torch.cuda.current_stream(residual_gpu.device)
            with torch.cuda.stream(self._param_offload_stream):
                self._param_offload_stream.wait_stream(current_stream)
                residual_cpu_slice.copy_(residual_gpu, non_blocking=True)
            return True

        residual_cpu_slice.copy_(residual_gpu.to(dtype=torch.float32, device='cpu'))
        return False

    def _mask_stage_view_from_pool(
        self,
        stage_pool: List[Optional[torch.Tensor]],
        stage_slot: int,
        numel: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not stage_pool:
            raise RuntimeError("Internal error: empty mask stage buffer pool.")
        stage_slot = int(stage_slot) % int(len(stage_pool))
        stage = stage_pool[stage_slot]
        need_alloc = (
            stage is None
            or stage.device != device
            or int(stage.numel()) < int(numel)
        )
        if need_alloc:
            alloc_numel = max(int(numel), 1)
            stage = torch.empty(alloc_numel, dtype=torch.uint8, device=device)
            stage_pool[stage_slot] = stage
        return stage[: int(numel)]

    def _mask_stage_view(
        self,
        stage_slot: int,
        numel: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self._mask_stage_view_from_pool(
            stage_pool=self._mask_stage_buffers,
            stage_slot=stage_slot,
            numel=numel,
            device=device,
        )

    def _fp32_scratch_view(
        self,
        stage_slot: int,
        numel: int,
        device: torch.device,
    ) -> torch.Tensor:
        _ = stage_slot
        scratch = self._fp32_scratch_buffers[0]
        need_alloc = (
            scratch is None
            or scratch.device != device
            or int(scratch.numel()) < int(numel)
        )
        if need_alloc:
            alloc_numel = max(int(numel), 1)
            scratch = torch.empty(alloc_numel, dtype=torch.float32, device=device)
            self._fp32_scratch_buffers[0] = scratch
        return scratch[: int(numel)]

    @staticmethod
    def _fill_packed_mask_ones(mask_packed: torch.Tensor, mask_numel: int) -> None:
        mask_packed.fill_(0xFF)
        valid_numel = int(mask_numel)
        extra_bits = int(mask_packed.numel()) * 8 - valid_numel
        if extra_bits <= 0 or int(mask_packed.numel()) <= 0:
            return
        valid_last_bits = 8 - extra_bits
        keep_mask = (1 << int(valid_last_bits)) - 1
        mask_packed[-1] = mask_packed[-1] & torch.tensor(
            keep_mask,
            dtype=mask_packed.dtype,
            device=mask_packed.device,
        )

    def _unpack_mask_slice_from_packed(
        self,
        packed_u8: torch.Tensor,
        start: int,
        end: int,
        device: torch.device,
        stage_slot: int,
        stage_pool: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        start_i = int(start)
        end_i = int(end)
        numel = int(end_i - start_i)
        if numel <= 0:
            raise RuntimeError('Mask unpack requested with non-positive slice length.')

        if stage_pool is None:
            dst = self._mask_stage_view(stage_slot=stage_slot, numel=numel, device=device)
        else:
            dst = self._mask_stage_view_from_pool(
                stage_pool=stage_pool,
                stage_slot=stage_slot,
                numel=numel,
                device=device,
            )
        bit_offset = start_i & 7
        byte_start = start_i >> 3
        byte_end = (end_i + 7) >> 3
        src = packed_u8[byte_start:byte_end]

        can_use_triton = (
            _TRITON_AVAILABLE
            and src.is_cuda
            and dst.is_cuda
            and src.is_contiguous()
            and dst.is_contiguous()
            and bit_offset == 0
            and int(src.numel()) == self._mask_pack_numel(numel)
        )
        if can_use_triton:
            _launch_mask_unpack_triton(src, dst, numel)
            return dst

        shifts = torch.arange(8, device=src.device, dtype=torch.int16)
        bits = ((src.to(torch.int16).unsqueeze(1) >> shifts) & 1).to(torch.uint8).reshape(-1)
        dst.copy_(bits[bit_offset : bit_offset + numel])
        return dst

    def _pack_mask_slice_to_packed(
        self,
        mask_slice_u8: torch.Tensor,
        packed_u8: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        start_i = int(start)
        end_i = int(end)
        numel = int(end_i - start_i)
        if numel <= 0:
            return
        if int(mask_slice_u8.numel()) != numel:
            raise RuntimeError(
                'Top-k mask pack found mask slice size mismatch: '
                f"slice={int(mask_slice_u8.numel())} expected={numel}."
            )

        bit_offset = start_i & 7
        byte_start = start_i >> 3
        byte_end = (end_i + 7) >> 3
        packed_slice = packed_u8[byte_start:byte_end]

        can_use_triton = (
            _TRITON_AVAILABLE
            and mask_slice_u8.is_cuda
            and packed_slice.is_cuda
            and mask_slice_u8.is_contiguous()
            and packed_slice.is_contiguous()
            and bit_offset == 0
            and int(mask_slice_u8.numel()) % 8 == 0
            and int(packed_slice.numel()) == (int(mask_slice_u8.numel()) // 8)
        )
        if can_use_triton:
            _launch_mask_pack_triton(mask_slice_u8, packed_slice)
            return

        shifts = torch.arange(8, device=packed_slice.device, dtype=torch.int16)
        bits = ((packed_slice.to(torch.int16).unsqueeze(1) >> shifts) & 1).to(torch.uint8).reshape(-1)
        bits[bit_offset : bit_offset + numel].copy_(mask_slice_u8.to(torch.uint8))
        bit_rows = bits.view(-1, 8).to(torch.int16)
        weights = (1 << shifts).view(1, 8)
        packed_vals = torch.sum(bit_rows * weights, dim=1).to(torch.uint8)
        packed_slice.copy_(packed_vals)

    def _set_packed_mask_slice_constant(
        self,
        packed_u8: torch.Tensor,
        start: int,
        end: int,
        value: int,
        stage_slot: int,
        device: torch.device,
    ) -> None:
        numel = int(end) - int(start)
        if numel <= 0:
            return
        mask_slice = self._mask_stage_view(
            stage_slot=stage_slot,
            numel=numel,
            device=device,
        )
        if int(value) != 0:
            mask_slice.fill_(1)
        else:
            mask_slice.zero_()
        self._pack_mask_slice_to_packed(mask_slice, packed_u8, int(start), int(end))

    def _set_packed_mask_slice_selected(
        self,
        packed_u8: torch.Tensor,
        start: int,
        end: int,
        selected_idx: torch.Tensor,
        stage_slot: int,
        device: torch.device,
    ) -> None:
        numel = int(end) - int(start)
        if numel <= 0:
            return
        mask_slice = self._mask_stage_view(
            stage_slot=stage_slot,
            numel=numel,
            device=device,
        )
        mask_slice.zero_()
        if int(selected_idx.numel()) > 0:
            mask_slice.index_fill_(0, selected_idx.to(torch.int64), 1)
        self._pack_mask_slice_to_packed(mask_slice, packed_u8, int(start), int(end))

    def _selected_local_idx_from_packed(
        self,
        packed_u8: torch.Tensor,
        start: int,
        end: int,
        device: torch.device,
        stage_slot: int,
        stage_pool: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        mask_slice = self._unpack_mask_slice_from_packed(
            packed_u8=packed_u8,
            start=int(start),
            end=int(end),
            device=device,
            stage_slot=stage_slot,
            stage_pool=stage_pool,
        )
        return torch.nonzero(mask_slice, as_tuple=False).flatten()

    def _selected_global_idx_from_packed_ranges(
        self,
        packed_u8: torch.Tensor,
        ranges: List[Tuple[int, int]],
        device: torch.device,
        stage_pool: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for range_pos, (start, end) in enumerate(ranges):
            start_i = int(start)
            end_i = int(end)
            if end_i <= start_i:
                continue
            selected_local = self._selected_local_idx_from_packed(
                packed_u8=packed_u8,
                start=start_i,
                end=end_i,
                device=device,
                stage_slot=(range_pos % 2),
                stage_pool=stage_pool,
            )
            if int(selected_local.numel()) > 0:
                chunks.append(selected_local + start_i)
        if chunks:
            return torch.cat(chunks, dim=0)
        return torch.empty(0, dtype=torch.int64, device=device)

    def _allreduce_packed_mask_staged_max_(
        self,
        packed_u8: torch.Tensor,
        ranges: List[Tuple[int, int]],
        mask_numel: int,
        group: torch.distributed.ProcessGroup,
        profile_key: str,
        stage_pool: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> None:
        world_size = self._group_size(group)
        if world_size <= 1:
            return
        if int(mask_numel) <= 0 or int(packed_u8.numel()) == 0:
            return

        _ = ranges
        t_total = self._profile_tic(packed_u8)

        t_unpack = self._profile_tic(packed_u8)
        mask_unpacked = self._unpack_mask_slice_from_packed(
            packed_u8=packed_u8,
            start=0,
            end=int(mask_numel),
            device=packed_u8.device,
            stage_slot=0,
            stage_pool=stage_pool,
        )
        self._profile_toc(f"{profile_key}.slice_unpack_ms", t_unpack, mask_unpacked)

        t_allreduce = self._profile_tic(mask_unpacked)
        torch.distributed.all_reduce(
            mask_unpacked,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
            async_op=False,
        )
        self._profile_toc(f"{profile_key}.slice_allreduce_ms", t_allreduce, mask_unpacked)

        t_pack = self._profile_tic(mask_unpacked)
        self._pack_mask_slice_to_packed(mask_unpacked, packed_u8, 0, int(mask_numel))
        self._profile_toc(f"{profile_key}.slice_pack_ms", t_pack, mask_unpacked)

        self._profile_toc(profile_key, t_total, packed_u8)

    def _build_packed_mask_from_corrected(
        self,
        grad_data: torch.Tensor,
        residual: torch.Tensor,
        param_slices: List[_ParamSlice],
        density: float,
        dense_mode: bool,
        out_mask_packed_u8: torch.Tensor,
        group_rank: int,
        profile_tag: str = 'mask.build.packed',
    ) -> None:
        t_zero = self._profile_tic(out_mask_packed_u8)
        out_mask_packed_u8.zero_()
        self._profile_toc(f'{profile_tag}.zero_ms', t_zero, out_mask_packed_u8)

        offloaded_residual = bool(
            self._offload_residual_to_cpu
            and residual.device.type == 'cpu'
            and grad_data.is_cuda
        )
        owner_work: List[Tuple[int, int]] = []
        prefetched_owner_pos: Optional[int] = None
        prefetched_residual: Optional[torch.Tensor] = None
        prefetched_ready_event: Optional[torch.cuda.Event] = None
        if offloaded_residual:
            for param_slice in param_slices:
                start = int(param_slice.start)
                end = int(param_slice.end)
                if end <= start:
                    continue
                if int(param_slice.owner) != int(group_rank):
                    continue
                owner_work.append((start, end))
            if owner_work:
                first_start, first_end = owner_work[0]
                prefetched_residual, prefetched_ready_event = self._prefetch_residual_slice_from_cpu(
                    residual_cpu=residual,
                    start=first_start,
                    end=first_end,
                    device=grad_data.device,
                    stage_slot=0,
                )
                prefetched_owner_pos = 0

        owner_work_pos = 0
        mask_stage_pos = 0
        for param_slice in param_slices:
            start = int(param_slice.start)
            end = int(param_slice.end)
            param = param_slice.param
            numel = int(end - start)
            if numel <= 0:
                continue
            if int(param_slice.owner) != int(group_rank):
                continue

            residual_slice: Optional[torch.Tensor] = None
            if offloaded_residual:
                if owner_work_pos >= len(owner_work):
                    raise RuntimeError('Residual prefetch worklist out of bounds in packed mask build.')
                expected_start, expected_end = owner_work[owner_work_pos]
                if expected_start != start or expected_end != end:
                    raise RuntimeError(
                        'Residual prefetch worklist mismatch in packed mask build: '
                        f'expected=[{expected_start}:{expected_end}] got=[{start}:{end}].'
                    )
                if prefetched_owner_pos != owner_work_pos or prefetched_residual is None:
                    prefetched_residual, prefetched_ready_event = self._prefetch_residual_slice_from_cpu(
                        residual_cpu=residual,
                        start=start,
                        end=end,
                        device=grad_data.device,
                        stage_slot=(owner_work_pos % 2),
                    )
                    prefetched_owner_pos = owner_work_pos
                residual_slice = prefetched_residual
                if residual_slice is None:
                    raise RuntimeError('Residual prefetch produced an empty staging buffer.')
                if prefetched_ready_event is not None:
                    torch.cuda.current_stream(grad_data.device).wait_event(prefetched_ready_event)
                next_owner_work_pos = owner_work_pos + 1
                if next_owner_work_pos < len(owner_work):
                    next_start, next_end = owner_work[next_owner_work_pos]
                    prefetched_residual, prefetched_ready_event = self._prefetch_residual_slice_from_cpu(
                        residual_cpu=residual,
                        start=next_start,
                        end=next_end,
                        device=grad_data.device,
                        stage_slot=(next_owner_work_pos % 2),
                    )
                    prefetched_owner_pos = next_owner_work_pos
                else:
                    prefetched_owner_pos = None
                    prefetched_residual = None
                    prefetched_ready_event = None
                owner_work_pos += 1
            else:
                residual_slice = residual[start:end]

            stage_slot = (mask_stage_pos % 2)
            mask_stage_pos += 1
            mask_slice = self._mask_stage_view(
                stage_slot=stage_slot,
                numel=numel,
                device=grad_data.device,
            )
            mask_slice.zero_()

            self._profile_add_value(f'{profile_tag}.candidate_layers', 1.0)
            self._profile_add_value(f'{profile_tag}.candidate_numel', float(numel))

            if param_slice.exclude_from_topk:
                mask_slice.fill_(1)
                self._profile_add_value(f'{profile_tag}.excluded_layers', 1.0)
                self._profile_add_value(f'{profile_tag}.excluded_numel', float(numel))
                self._profile_add_value(f'{profile_tag}.selected_elements', float(numel))
                self._pack_mask_slice_to_packed(mask_slice, out_mask_packed_u8, start, end)
                continue

            if dense_mode or density >= 1.0:
                mask_slice.fill_(1)
                self._profile_add_value(f'{profile_tag}.selected_elements', float(numel))
                self._pack_mask_slice_to_packed(mask_slice, out_mask_packed_u8, start, end)
                continue

            k = self._topk_k(density, numel)
            if k <= 0:
                self._pack_mask_slice_to_packed(mask_slice, out_mask_packed_u8, start, end)
                continue
            if k >= numel:
                mask_slice.fill_(1)
                self._profile_add_value(f'{profile_tag}.selected_elements', float(numel))
                self._pack_mask_slice_to_packed(mask_slice, out_mask_packed_u8, start, end)
                continue

            state = self._optimizer_state_entry(param)
            if state is None or 'exp_avg' not in state:
                mask_slice.fill_(1)
                self._profile_add_value(f'{profile_tag}.selected_elements', float(numel))
                self._pack_mask_slice_to_packed(mask_slice, out_mask_packed_u8, start, end)
                continue

            if residual_slice is None:
                raise RuntimeError('Residual slice is missing while building packed top-k mask.')

            t_corrected = self._profile_tic(grad_data)
            exp_avg_prev = state['exp_avg'].view(-1).to(torch.float32)
            group = self._param_to_group.get(param, {})
            beta1 = float(group.get('betas', (0.9, 0.999))[0])
            corrected = self._fp32_scratch_view(
                stage_slot=stage_slot,
                numel=numel,
                device=grad_data.device,
            )
            corrected.copy_(exp_avg_prev)
            corrected.mul_(beta1)
            corrected.add_(grad_data[start:end].to(torch.float32), alpha=1.0 - beta1)
            corrected.add_(residual_slice)
            self._profile_toc(f'{profile_tag}.corrected_ms', t_corrected, grad_data)

            t_topk = self._profile_tic(corrected)
            _, topk_idx = torch.topk(corrected.abs(), k=k, sorted=False)
            self._profile_toc(f'{profile_tag}.topk_ms', t_topk, corrected)

            t_index_fill = self._profile_tic(mask_slice)
            mask_slice.index_fill_(0, topk_idx, 1)
            self._profile_toc(f'{profile_tag}.index_fill_ms', t_index_fill, mask_slice)
            self._profile_add_value(f'{profile_tag}.selected_elements', float(k))
            self._profile_add_value(f'{profile_tag}.topk_calls', 1.0)

            self._pack_mask_slice_to_packed(mask_slice, out_mask_packed_u8, start, end)

    def _maybe_non_topk_decay_triton(
        self,
        full_slice_fp32: torch.Tensor,
        mask_slice_u8: torch.Tensor,
        decay_factor: float,
    ) -> bool:
        if not _TRITON_AVAILABLE:
            raise RuntimeError(
                "Top-k hard-fail mode: Triton is required for non-topk decay kernel."
            )
        if not self._triton_sparse_nontopk_enabled:
            raise RuntimeError(
                "Top-k hard-fail mode: Triton non-topk decay kernel is disabled after a prior failure."
            )
        if not (
            full_slice_fp32.is_cuda
            and mask_slice_u8.is_cuda
            and full_slice_fp32.is_contiguous()
            and mask_slice_u8.is_contiguous()
            and full_slice_fp32.dtype == torch.float32
            and mask_slice_u8.dtype == torch.uint8
            and int(full_slice_fp32.numel()) == int(mask_slice_u8.numel())
        ):
            raise RuntimeError(
                "Top-k hard-fail mode: invalid tensors for Triton non-topk decay kernel."
            )
        try:
            _launch_non_topk_decay_triton(full_slice_fp32, mask_slice_u8, decay_factor)
            return True
        except Exception as exc:
            self._triton_sparse_nontopk_enabled = False
            raise RuntimeError(
                "Top-k hard-fail mode: Triton non-topk decay kernel failed."
            ) from exc

    def _maybe_topk_writeback_triton(
        self,
        full_param_fp32: torch.Tensor,
        selected_global_idx: torch.Tensor,
        selected_updated_values: torch.Tensor,
    ) -> bool:
        if not _TRITON_AVAILABLE:
            raise RuntimeError(
                "Top-k hard-fail mode: Triton is required for top-k writeback kernel."
            )
        if not self._triton_sparse_topk_enabled:
            raise RuntimeError(
                "Top-k hard-fail mode: Triton top-k writeback kernel is disabled after a prior failure."
            )
        if not (
            full_param_fp32.is_cuda
            and selected_global_idx.is_cuda
            and selected_updated_values.is_cuda
            and full_param_fp32.is_contiguous()
            and selected_global_idx.is_contiguous()
            and selected_updated_values.is_contiguous()
            and full_param_fp32.dtype == torch.float32
            and selected_updated_values.dtype == torch.float32
            and selected_global_idx.dtype in (torch.int64, torch.int32)
            and int(selected_global_idx.numel()) == int(selected_updated_values.numel())
        ):
            raise RuntimeError(
                "Top-k hard-fail mode: invalid tensors for Triton top-k writeback kernel."
            )
        try:
            _launch_topk_scatter_writeback_triton(
                full_param_fp32,
                selected_global_idx,
                selected_updated_values,
            )
            return True
        except Exception as exc:
            self._triton_sparse_topk_enabled = False
            raise RuntimeError(
                "Top-k hard-fail mode: Triton top-k writeback kernel failed."
            ) from exc


    @torch.no_grad()
    def sync_sparse_params_from_local_shards(self) -> bool:
        """Synchronize only selected updated params and locally decay non-topk params.

        This path keeps a full FP32 param replica on each rank and avoids dense
        per-step parameter all-gather in distributed-optimizer mode.
        """
        if not self._ddp_config.use_distributed_optimizer:
            return False
        if self._ddp_config.overlap_param_gather:
            raise RuntimeError(
                "Top-k hard-fail mode: overlap_param_gather=True is unsupported for sparse param sync."
            )

        step_hint = int(self._prepared_iteration) if self._prepared_iteration is not None else -1
        sync_sparse_t0 = self._profile_tic()
        self._profile_add_value(
            'sync_sparse.offload_full_param_cpu',
            1.0 if self._offload_full_param_to_cpu else 0.0,
        )
        self._profile_add_value(
            'sync_sparse.triton_available',
            1.0 if _TRITON_AVAILABLE else 0.0,
        )
        self._profile_add_value(
            'sync_sparse.triton_nontopk_enabled',
            1.0 if self._triton_sparse_nontopk_enabled else 0.0,
        )
        self._profile_add_value(
            'sync_sparse.triton_topk_enabled',
            1.0 if self._triton_sparse_topk_enabled else 0.0,
        )

        t_work_items = self._profile_tic()
        work_items: List[Tuple[int, _ParamAndGradBuffer]] = []
        for buffer_idx, buffer in enumerate(self._buffers):
            param_data = buffer.param_data
            if param_data is None or param_data.numel() == 0:
                continue
            if not self._buffer_param_slices[buffer_idx]:
                continue
            work_items.append((buffer_idx, buffer))
        self._profile_toc('sync_sparse.build_work_items_ms', t_work_items)
        self._profile_add_value('sync_sparse.buffer_count', float(len(work_items)))

        group_payload_sync: Dict[int, Dict[str, Any]] = {}
        buffer_plans: Dict[int, Dict[str, Any]] = {}
        t_plan = self._profile_tic()
        for buffer_idx, buffer in work_items:
            param_data = buffer.param_data
            if param_data is None:
                continue

            state = self._buffer_states[buffer_idx]
            param_slices = self._buffer_param_slices[buffer_idx]
            group = buffer.data_parallel_group
            group_rank = self._buffer_group_ranks[buffer_idx]
            group_size = max(1, self._group_size(group))

            self._profile_add_value('sync_sparse.buffer_numel', float(param_data.numel()))
            self._profile_add_value('sync_sparse.param_slice_count', float(len(param_slices)))

            local_selected_by_range: Dict[Tuple[int, int], torch.Tensor] = {}
            if state.use_packed_mask_only:
                current_packed = state.current_mask_packed_u8
                if current_packed is None:
                    raise RuntimeError('Internal error: current packed mask buffer is missing.')
                selected_scan_pos = 0
                t_nonzero = self._profile_tic(current_packed)
                for param_slice in param_slices:
                    param = param_slice.param
                    local_start, local_end, _, _ = self._distopt_local_shard_range(
                        buffer=buffer,
                        param=param,
                        group_rank=group_rank,
                        group_size=group_size,
                    )
                    local_numel = int(local_end - local_start)
                    if local_numel <= 0:
                        continue
                    local_selected = self._selected_local_idx_from_packed(
                        packed_u8=current_packed,
                        start=int(local_start),
                        end=int(local_end),
                        device=param_data.device,
                        stage_slot=(selected_scan_pos % 2),
                    )
                    selected_scan_pos += 1
                    local_selected_by_range[(int(local_start), int(local_end))] = local_selected

                global_ranges = [
                    (int(param_slice.start), int(param_slice.end))
                    for param_slice in param_slices
                    if int(param_slice.end) > int(param_slice.start)
                ]
                selected_global_idx = self._selected_global_idx_from_packed_ranges(
                    packed_u8=current_packed,
                    ranges=global_ranges,
                    device=param_data.device,
                )
                self._profile_toc('sync_sparse.plan.nonzero_ms', t_nonzero, current_packed)
            else:
                mask_u8 = state.current_mask_u8
                if mask_u8 is None:
                    raise RuntimeError('Internal error: current mask_u8 buffer is missing.')
                if mask_u8.numel() != param_data.numel():
                    raise RuntimeError(
                        "Top-k sparse param sync mask/param size mismatch: "
                        f"mask={int(mask_u8.numel())} param={int(param_data.numel())}."
                    )

                t_mask_bool = self._profile_tic(mask_u8)
                current_mask = mask_u8.bool()
                self._profile_toc('sync_sparse.plan.mask_bool_ms', t_mask_bool, current_mask)

                t_nonzero = self._profile_tic(current_mask)
                selected_global_idx = torch.nonzero(current_mask, as_tuple=False).flatten()
                self._profile_toc('sync_sparse.plan.nonzero_ms', t_nonzero, selected_global_idx)

            selected_count = int(selected_global_idx.numel())
            self._profile_add_value('sync_sparse.selected_total', float(selected_count))
            self._profile_add_value('sync_sparse.total_numel', float(param_data.numel()))

            group_key = id(group)
            group_entry = group_payload_sync.get(group_key)
            if group_entry is None:
                group_entry = {
                    'group': group,
                    'group_size': int(group_size),
                    'device': param_data.device,
                    'plans': [],
                    'payload_numel': 0,
                    'payload': None,
                    'handle': None,
                    'on_side_stream': False,
                    'waited': False,
                    'payload_allreduce_total_t0': None,
                }
                group_payload_sync[group_key] = group_entry
            else:
                if int(group_entry['group_size']) != int(group_size):
                    raise RuntimeError(
                        'Top-k sparse payload packing found inconsistent group size '
                        f"for one data-parallel group: {int(group_entry['group_size'])} vs {int(group_size)}."
                    )
                if group_entry['device'] != param_data.device:
                    raise RuntimeError(
                        'Top-k sparse payload packing found mixed devices for one '
                        f"data-parallel group: {group_entry['device']} vs {param_data.device}."
                    )

            payload_offset = int(group_entry['payload_numel'])
            group_entry['payload_numel'] = payload_offset + selected_count

            plan = {
                'buffer_idx': int(buffer_idx),
                'buffer': buffer,
                'param_data': param_data,
                'state': state,
                'param_slices': param_slices,
                'group_key': group_key,
                'group_rank': int(group_rank),
                'group_size': int(group_size),
                'local_selected_by_range': local_selected_by_range,
                'mask_u8': state.current_mask_u8,
                'selected_global_idx': selected_global_idx,
                'selected_count': int(selected_count),
                'payload_offset': int(payload_offset),
            }
            group_entry['plans'].append(plan)
            buffer_plans[int(buffer_idx)] = plan
        self._profile_toc('sync_sparse.plan.total_ms', t_plan)
        self._profile_add_value('sync_sparse.group_count', float(len(group_payload_sync)))

        for group_entry in group_payload_sync.values():
            payload_numel = int(group_entry['payload_numel'])
            self._profile_add_value('sync_sparse.group_payload_numel', float(payload_numel))
            if payload_numel <= 0:
                continue

            plans = group_entry['plans']
            t_payload_alloc = self._profile_tic()
            payload = torch.zeros(
                payload_numel,
                dtype=torch.float32,
                device=group_entry['device'],
            )
            self._profile_toc('sync_sparse.group.payload_alloc_ms', t_payload_alloc, payload)
            self._profile_add_value('sync_sparse.selected_value_numel', float(payload_numel))

            for plan in plans:
                selected_count = int(plan['selected_count'])
                if selected_count <= 0:
                    continue

                self._profile_add_value('sync_sparse.group_plan_count', 1.0)
                payload_base = int(plan['payload_offset'])
                buffer = plan['buffer']
                param_data = plan['param_data']
                param_slices = plan['param_slices']
                group_rank = int(plan['group_rank'])
                group_size = int(plan['group_size'])
                state = plan['state']
                local_selected_by_range = plan['local_selected_by_range']
                mask_u8 = plan.get('mask_u8', None)
                selected_global_idx = plan['selected_global_idx']

                for param_slice in param_slices:
                    param = param_slice.param
                    t_range = self._profile_tic()
                    local_start, local_end, _, _ = self._distopt_local_shard_range(
                        buffer=buffer,
                        param=param,
                        group_rank=group_rank,
                        group_size=group_size,
                    )
                    self._profile_toc('sync_sparse.group.local_range_ms', t_range)
                    self._profile_add_value('sync_sparse.local_range_calls', 1.0)
                    local_numel = int(local_end - local_start)
                    if local_numel <= 0:
                        continue

                    t_search = self._profile_tic(selected_global_idx)
                    payload_start = int(
                        torch.searchsorted(
                            selected_global_idx,
                            torch.tensor(
                                local_start,
                                dtype=torch.int64,
                                device=selected_global_idx.device,
                            ),
                            right=False,
                        ).item()
                    )
                    payload_end = int(
                        torch.searchsorted(
                            selected_global_idx,
                            torch.tensor(
                                local_end,
                                dtype=torch.int64,
                                device=selected_global_idx.device,
                            ),
                            right=False,
                        ).item()
                    )
                    self._profile_toc(
                        'sync_sparse.group.searchsorted_ms',
                        t_search,
                        selected_global_idx,
                        calls=2,
                    )
                    self._profile_add_value('sync_sparse.searchsorted_calls', 2.0)
                    payload_len = max(0, payload_end - payload_start)
                    if payload_len <= 0:
                        continue

                    t_local_select = self._profile_tic(param_data)
                    if state.use_packed_mask_only:
                        local_selected_idx = local_selected_by_range.get(
                            (int(local_start), int(local_end)),
                            None,
                        )
                        if local_selected_idx is None:
                            current_packed = state.current_mask_packed_u8
                            if current_packed is None:
                                raise RuntimeError('Internal error: current packed mask buffer is missing.')
                            local_selected_idx = self._selected_local_idx_from_packed(
                                packed_u8=current_packed,
                                start=int(local_start),
                                end=int(local_end),
                                device=param_data.device,
                                stage_slot=(payload_start % 2),
                            )
                            local_selected_by_range[(int(local_start), int(local_end))] = local_selected_idx
                        local_selected_values = param_data[local_start:local_end].to(torch.float32).index_select(
                            0,
                            local_selected_idx.to(torch.int64),
                        )
                    else:
                        if mask_u8 is None:
                            raise RuntimeError('Internal error: current mask_u8 buffer is missing.')
                        local_mask = mask_u8[local_start:local_end] != 0
                        local_selected_values = param_data[local_start:local_end].to(torch.float32)[
                            local_mask
                        ]
                    self._profile_toc(
                        'sync_sparse.group.local_select_ms',
                        t_local_select,
                        local_selected_values,
                    )
                    self._profile_add_value(
                        'sync_sparse.local_payload_elements',
                        float(payload_len),
                    )
                    if int(local_selected_values.numel()) != payload_len:
                        raise RuntimeError(
                            'Top-k sparse param sync payload mismatch: local selected count '
                            'changed between mask and payload packing.'
                        )

                    t_payload_copy = self._profile_tic(payload)
                    payload[
                        payload_base + payload_start : payload_base + payload_end
                    ].copy_(local_selected_values)
                    self._profile_toc(
                        'sync_sparse.group.payload_copy_ms',
                        t_payload_copy,
                        payload,
                    )

            group_entry['payload'] = payload

            if int(group_entry['group_size']) > 1:
                t_payload_total = self._profile_tic()
                t_payload_dispatch = self._profile_tic()
                payload_allreduce_on_side_stream = False
                payload_allreduce_handle: Optional[Any] = None
                if self._payload_allreduce_stream is not None and payload.is_cuda:
                    payload_allreduce_on_side_stream = True
                    current_stream = torch.cuda.current_stream(payload.device)
                    self._payload_allreduce_stream.wait_stream(current_stream)
                    with torch.cuda.stream(self._payload_allreduce_stream):
                        payload_allreduce_handle = torch.distributed.all_reduce(
                            payload,
                            op=torch.distributed.ReduceOp.SUM,
                            group=group_entry['group'],
                            async_op=True,
                        )
                else:
                    payload_allreduce_handle = torch.distributed.all_reduce(
                        payload,
                        op=torch.distributed.ReduceOp.SUM,
                        group=group_entry['group'],
                        async_op=True,
                    )
                self._profile_toc(
                    'sync_sparse.group.payload_allreduce_dispatch_ms',
                    t_payload_dispatch,
                )
                group_entry['handle'] = payload_allreduce_handle
                group_entry['on_side_stream'] = bool(payload_allreduce_on_side_stream)
                group_entry['payload_allreduce_total_t0'] = t_payload_total

        pending_offload_tensors: List[torch.Tensor] = []
        prefetched_buffer_idx: Optional[int] = None
        prefetched_full_param: Optional[torch.Tensor] = None
        prefetched_ready_event: Optional[torch.cuda.Event] = None

        def launch_prefetch(
            work_pos: int,
        ) -> Tuple[Optional[int], Optional[torch.Tensor], Optional[torch.cuda.Event]]:
            if work_pos >= len(work_items):
                return None, None, None
            next_buffer_idx, next_buffer = work_items[work_pos]
            next_param_data = next_buffer.param_data
            if next_param_data is None:
                return next_buffer_idx, None, None
            self._profile_add_value('sync_sparse.prefetch_attempts', 1.0)
            t_ensure = self._profile_tic(next_param_data)
            full_storage = self._ensure_full_param_storage(next_buffer_idx, next_param_data)
            self._profile_toc(
                'sync_sparse.prefetch.ensure_storage_ms',
                t_ensure,
                next_param_data,
            )
            if full_storage is None:
                return next_buffer_idx, None, None
            if not self._offload_full_param_to_cpu:
                return next_buffer_idx, full_storage, None
            t_prefetch = self._profile_tic(next_param_data)
            full_param_gpu, ready_event = self._prefetch_full_param_from_cpu(
                full_storage, next_param_data.device
            )
            self._profile_toc('sync_sparse.prefetch.cpu_to_gpu_ms', t_prefetch, full_param_gpu)
            return next_buffer_idx, full_param_gpu, ready_event

        if self._offload_full_param_to_cpu and work_items:
            (
                prefetched_buffer_idx,
                prefetched_full_param,
                prefetched_ready_event,
            ) = launch_prefetch(0)

        for work_pos, (buffer_idx, buffer) in enumerate(work_items):
            plan = buffer_plans.get(int(buffer_idx))
            param_data = buffer.param_data
            if plan is None or param_data is None:
                continue

            param_slices = plan['param_slices']
            state = plan['state']
            mask_u8 = plan.get('mask_u8', None)
            local_selected_by_range = plan.get('local_selected_by_range', {})
            selected_global_idx = plan['selected_global_idx']
            selected_count = int(plan['selected_count'])

            if self._offload_full_param_to_cpu:
                if prefetched_buffer_idx != buffer_idx or prefetched_full_param is None:
                    (
                        prefetched_buffer_idx,
                        prefetched_full_param,
                        prefetched_ready_event,
                    ) = launch_prefetch(work_pos)
                full_param = prefetched_full_param
                if full_param is None:
                    continue
                if prefetched_ready_event is not None:
                    t_prefetch_wait = self._profile_tic()
                    torch.cuda.current_stream(param_data.device).wait_event(prefetched_ready_event)
                    self._profile_toc(
                        'sync_sparse.buffer.prefetch_wait_ms',
                        t_prefetch_wait,
                        param_data,
                    )

                (
                    prefetched_buffer_idx,
                    prefetched_full_param,
                    prefetched_ready_event,
                ) = launch_prefetch(work_pos + 1)
            else:
                t_full_storage = self._profile_tic(param_data)
                full_param = self._ensure_full_param_storage(buffer_idx, param_data)
                self._profile_toc(
                    'sync_sparse.buffer.ensure_storage_ms',
                    t_full_storage,
                    full_param if full_param is not None else param_data,
                )
                if full_param is None:
                    continue

            # For non-topk positions, apply decoupled weight decay update locally on FP32 replica.
            # This is intentionally overlapped with packed sparse payload communication above.
            decay_mask_work_pos = 0
            for param_slice in param_slices:
                start = int(param_slice.start)
                end = int(param_slice.end)
                param = param_slice.param
                if end <= start:
                    continue

                group_cfg = self._param_to_group.get(param, None)
                if group_cfg is None:
                    continue
                lr = float(group_cfg.get('lr', 0.0))
                weight_decay = self._group_weight_decay(group_cfg)
                if lr == 0.0 or weight_decay == 0.0:
                    continue

                decay_factor = 1.0 - lr * weight_decay
                if abs(decay_factor - 1.0) <= 1.0e-12:
                    continue

                full_slice = full_param[start:end]
                if state.use_packed_mask_only:
                    local_selected_idx = local_selected_by_range.get((start, end), None)
                    if local_selected_idx is None:
                        current_packed = state.current_mask_packed_u8
                        if current_packed is None:
                            raise RuntimeError('Internal error: current packed mask buffer is missing.')
                        local_selected_idx = self._selected_local_idx_from_packed(
                            packed_u8=current_packed,
                            start=start,
                            end=end,
                            device=full_slice.device,
                            stage_slot=(decay_mask_work_pos % 2),
                        )
                        local_selected_by_range[(start, end)] = local_selected_idx
                    mask_slice_u8 = self._mask_stage_view(
                        stage_slot=(decay_mask_work_pos % 2),
                        numel=int(end - start),
                        device=full_slice.device,
                    )
                    mask_slice_u8.zero_()
                    if int(local_selected_idx.numel()) > 0:
                        mask_slice_u8.index_fill_(0, local_selected_idx.to(torch.int64), 1)
                    decay_mask_work_pos += 1
                else:
                    if mask_u8 is None:
                        raise RuntimeError('Internal error: current mask_u8 buffer is missing.')
                    mask_slice_u8 = mask_u8[start:end]
                t_selected_count = self._profile_tic(mask_slice_u8)
                local_selected_count = int(mask_slice_u8.sum().item())
                self._profile_toc(
                    'sync_sparse.buffer.nontopk.selected_count_ms',
                    t_selected_count,
                    mask_slice_u8,
                )
                self._profile_add_value('sync_sparse.nontopk_selected_count_calls', 1.0)
                if local_selected_count >= int(end - start):
                    self._profile_add_value('sync_sparse.nontopk_full_selected_slices', 1.0)
                    continue

                self._profile_add_value('sync_sparse.nontopk_decay_slices', 1.0)
                if local_selected_count == 0:
                    self._profile_add_value('sync_sparse.nontopk_zero_selected_slices', 1.0)
                else:
                    self._profile_add_value('sync_sparse.nontopk_partial_selected_slices', 1.0)

                t_triton_decay = self._profile_tic(full_slice)
                triton_applied = self._maybe_non_topk_decay_triton(
                    full_slice,
                    mask_slice_u8,
                    decay_factor,
                )
                if not triton_applied:
                    raise RuntimeError(
                        "Top-k hard-fail mode: Triton non-topk decay path returned no kernel execution."
                    )

                self._profile_toc(
                    'sync_sparse.buffer.nontopk.triton_ms',
                    t_triton_decay,
                    full_slice,
                )
                self._profile_add_value('sync_sparse.triton_nontopk_calls', 1.0)
                self._profile_add_value(
                    'sync_sparse.triton_nontopk_numel',
                    float(end - start),
                )
                continue

            # For topk positions, use owner-updated values synchronized via one packed all-reduce.
            if selected_count > 0:
                group_entry = group_payload_sync[plan['group_key']]
                payload = group_entry.get('payload', None)
                if payload is None:
                    raise RuntimeError(
                        'Internal error: packed sparse payload buffer is missing for top-k sync.'
                    )
                if not bool(group_entry.get('waited', False)):
                    payload_allreduce_handle = group_entry.get('handle', None)
                    if payload_allreduce_handle is not None:
                        t_payload_wait = self._profile_tic()
                        payload_allreduce_handle.wait()
                        if bool(group_entry.get('on_side_stream', False)):
                            torch.cuda.current_stream(payload.device).wait_stream(
                                self._payload_allreduce_stream
                            )
                        self._profile_toc(
                            'sync_sparse.group.payload_allreduce_wait_ms',
                            t_payload_wait,
                            payload,
                        )
                        self._profile_toc(
                            'sync_sparse.group.payload_allreduce_total_ms',
                            group_entry.get('payload_allreduce_total_t0', None),
                            payload,
                        )
                    group_entry['waited'] = True

                payload_start = int(plan['payload_offset'])
                payload_end = payload_start + selected_count
                selected_updated_values = payload[payload_start:payload_end]
                t_topk_writeback = self._profile_tic(full_param)
                triton_applied = self._maybe_topk_writeback_triton(
                    full_param,
                    selected_global_idx,
                    selected_updated_values,
                )
                if not triton_applied:
                    raise RuntimeError(
                        "Top-k hard-fail mode: Triton top-k writeback path returned no kernel execution."
                    )
                self._profile_toc(
                    'sync_sparse.buffer.topk_writeback.triton_ms',
                    t_topk_writeback,
                    full_param,
                )
                self._profile_add_value('sync_sparse.triton_topk_calls', 1.0)
                self._profile_add_value(
                    'sync_sparse.triton_topk_numel',
                    float(selected_count),
                )

            # Materialize full updated params for next forward.
            t_materialize = self._profile_tic(full_param)
            param_data.copy_(full_param.to(param_data.dtype))
            self._profile_toc(
                'sync_sparse.buffer.materialize_param_ms',
                t_materialize,
                param_data,
            )
            self._profile_add_value('sync_sparse.materialize_numel', float(param_data.numel()))

            if self._offload_full_param_to_cpu:
                full_param_cpu = self._full_param_fp32[buffer_idx]
                if full_param_cpu is None or full_param_cpu.device.type != 'cpu':
                    raise RuntimeError(
                        'Top-k CPU offload expected CPU full-param storage, but found invalid buffer.'
                    )
                t_offload_queue = self._profile_tic(full_param)
                queued_async = self._queue_full_param_offload_to_cpu(full_param, full_param_cpu)
                self._profile_toc(
                    'sync_sparse.buffer.offload_queue_ms',
                    t_offload_queue,
                    full_param,
                )
                self._profile_add_value('sync_sparse.offload_queue_calls', 1.0)
                if queued_async:
                    pending_offload_tensors.append(full_param)
                    self._profile_add_value('sync_sparse.offload_async_buffers', 1.0)

        if pending_offload_tensors and self._param_offload_stream is not None:
            wait_device = pending_offload_tensors[0].device
            t_offload_wait = self._profile_tic()
            torch.cuda.current_stream(wait_device).wait_stream(self._param_offload_stream)
            self._profile_toc(
                'sync_sparse.offload_wait_ms',
                t_offload_wait,
                pending_offload_tensors[0],
            )
            pending_offload_tensors.clear()

        self._profile_toc('sync_sparse.total_ms', sync_sparse_t0)
        self._log_profile_subset(step_hint, 'sync_sparse.', 'sync_sparse')
        return True


    def _wait_next_mask_allreduce(
        self,
        state: _BufferState,
    ) -> None:
        handle = state.next_mask_allreduce_handle
        wait_event = state.next_mask_allreduce_event
        wait_tensor = state.next_mask_packed_u8 if state.use_packed_mask_only else state.next_mask_u8
        if wait_tensor is None:
            raise RuntimeError("Internal error: next mask wait tensor is missing.")
        if state.next_mask_allreduce_uses_packed and state.next_mask_allgather_recv_u8 is not None:
            wait_tensor = state.next_mask_allgather_recv_u8

        if handle is None and wait_event is None:
            state.next_mask_allreduce_uses_packed = bool(state.use_packed_mask_only)
            return

        if handle is not None:
            t_wait = self._profile_tic(wait_tensor)
            handle.wait()
            self._profile_toc("mask.next_async_wait_ms", t_wait, wait_tensor)

            if state.next_mask_allreduce_uses_packed:
                packed = state.next_mask_packed_u8
                recv_u8 = state.next_mask_allgather_recv_u8
                send_bytes = int(state.mask_allgather_send_bytes)
                if packed is None:
                    raise RuntimeError("Internal error: next packed mask buffer is missing.")
                if send_bytes > 0:
                    if recv_u8 is None:
                        raise RuntimeError("Internal error: next all-gather recv buffer is missing.")
                    t_stitch = self._profile_tic(packed)
                    self._stitch_allgather_recv_to_packed_mask(
                        recv_u8=recv_u8,
                        packed_u8=packed,
                        rank_segments_by_rank=state.mask_allgather_segments_by_rank,
                        rank_payload_bytes=state.mask_allgather_rank_payload_bytes,
                        send_bytes=send_bytes,
                        profile_key='mask.next_async_stitch_ms',
                    )
                    self._profile_toc('mask.next_async_stitch_ms.total_ms', t_stitch, packed)

                if not state.use_packed_mask_only:
                    next_mask_u8 = state.next_mask_u8
                    if next_mask_u8 is None:
                        raise RuntimeError("Internal error: next mask_u8 buffer is missing.")
                    t_unpack = self._profile_tic(next_mask_u8)
                    self._unpack_packed_mask_to_local_u8(
                        packed,
                        next_mask_u8,
                        state.mask_pack_segments,
                    )
                    self._profile_toc("mask.next_async_unpack_ms", t_unpack, next_mask_u8)

            state.next_mask_allreduce_handle = None

        if wait_event is not None:
            if wait_tensor.is_cuda:
                t_event_wait = self._profile_tic(wait_tensor)
                torch.cuda.current_stream(wait_tensor.device).wait_event(wait_event)
                self._profile_toc("mask.next_async_wait_ms", t_event_wait, wait_tensor)
            state.next_mask_allreduce_event = None

        state.next_mask_allreduce_handle = None
        state.next_mask_allreduce_event = None
        state.next_mask_allreduce_uses_packed = False
        if self._async_mask_stream is not None and wait_tensor.is_cuda:
            t_stream_wait = self._profile_tic(wait_tensor)
            torch.cuda.current_stream(wait_tensor.device).wait_stream(self._async_mask_stream)
            self._profile_toc("mask.next_async_stream_wait_ms", t_stream_wait, wait_tensor)
        state.has_next_mask = True

    def _launch_next_mask_allreduce_async(

        self,
        state: _BufferState,
        group: torch.distributed.ProcessGroup,
    ) -> None:
        # Re-launching is illegal while a previous async mask sync is inflight.
        self._wait_next_mask_allreduce(state)

        world_size = self._group_size(group)

        if state.use_packed_mask_only:
            next_packed = state.next_mask_packed_u8
            if next_packed is None:
                raise RuntimeError('Internal error: next packed mask buffer is missing.')
            if world_size <= 1 or next_packed.numel() == 0 or int(state.mask_allgather_send_bytes) <= 0:
                state.next_mask_allreduce_handle = None
                state.next_mask_allreduce_event = None
                state.next_mask_allreduce_uses_packed = True
                state.next_mask_allreduce_unpacked_u8 = None
                state.has_next_mask = True
                return

            state.has_next_mask = False
            state.next_mask_allreduce_event = None

            if self._async_mask_stream is not None and next_packed.is_cuda:
                current_stream = torch.cuda.current_stream(next_packed.device)
                t_stream_wait = self._profile_tic(next_packed)
                self._async_mask_stream.wait_stream(current_stream)
                self._profile_toc(
                    'mask.next_async_launch_wait_stream_ms',
                    t_stream_wait,
                    next_packed,
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
            return

        next_mask_u8 = state.next_mask_u8
        if next_mask_u8 is None:
            raise RuntimeError('Internal error: next mask_u8 buffer is missing.')

        if world_size <= 1 or next_mask_u8.numel() == 0:
            state.next_mask_allreduce_handle = None
            state.next_mask_allreduce_event = None
            state.next_mask_allreduce_uses_packed = False
            state.has_next_mask = True
            return

        state.has_next_mask = False

        if self._async_mask_stream is not None and next_mask_u8.is_cuda:
            current_stream = torch.cuda.current_stream(next_mask_u8.device)
            t_stream_wait = self._profile_tic(next_mask_u8)
            self._async_mask_stream.wait_stream(current_stream)
            self._profile_toc(
                'mask.next_async_launch_wait_stream_ms',
                t_stream_wait,
                next_mask_u8,
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

    @staticmethod
    def _coalesce_sorted_ranges(
        ranges: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if not ranges:
            return []
        sorted_ranges = sorted(ranges, key=lambda item: int(item[0]))
        merged: List[List[int]] = []
        for start, end in sorted_ranges:
            start_i = int(start)
            end_i = int(end)
            if end_i <= start_i:
                continue
            if not merged:
                merged.append([start_i, end_i])
                continue
            prev_start, prev_end = merged[-1]
            if start_i <= prev_end:
                if end_i > prev_end:
                    merged[-1][1] = end_i
            else:
                merged.append([start_i, end_i])
        return [(int(start), int(end)) for start, end in merged]

    def _build_mask_allgather_layout(
        self,
        buffer: _ParamAndGradBuffer,
        numel: int,
        param_slices: List[_ParamSlice],
        group_size: int,
    ) -> Tuple[
        Tuple[Tuple[Tuple[int, int, int, int], ...], ...],
        Tuple[int, ...],
        int,
    ]:
        total_numel = int(numel)
        world_size = max(1, int(group_size))
        if total_numel <= 0 or world_size <= 1:
            return tuple(), tuple(), 0

        rank_ranges: List[List[Tuple[int, int]]] = [list() for _ in range(world_size)]
        if self._ddp_config.use_distributed_optimizer:
            for param_slice in param_slices:
                param = param_slice.param
                for rank in range(world_size):
                    local_start, local_end, _local_param_start, _param_numel = self._distopt_local_shard_range(
                        buffer=buffer,
                        param=param,
                        group_rank=rank,
                        group_size=world_size,
                    )
                    start_i = max(0, min(int(local_start), total_numel))
                    end_i = max(0, min(int(local_end), total_numel))
                    if end_i > start_i:
                        rank_ranges[rank].append((start_i, end_i))
        else:
            for param_slice in param_slices:
                owner = int(param_slice.owner)
                if owner < 0 or owner >= world_size:
                    continue
                start_i = max(0, min(int(param_slice.start), total_numel))
                end_i = max(0, min(int(param_slice.end), total_numel))
                if end_i > start_i:
                    rank_ranges[owner].append((start_i, end_i))

        rank_segments: List[Tuple[Tuple[int, int, int, int], ...]] = []
        rank_payload_bytes: List[int] = []
        send_bytes = 0
        for rank in range(world_size):
            merged_ranges = self._coalesce_sorted_ranges(rank_ranges[rank])
            segments: List[Tuple[int, int, int, int]] = []
            payload_cursor = 0
            for start_i, end_i in merged_ranges:
                nbits = int(end_i - start_i)
                if nbits <= 0:
                    continue
                nbytes = self._mask_pack_numel(nbits)
                if nbytes <= 0:
                    continue
                segments.append((int(start_i), int(end_i), int(payload_cursor), int(nbytes)))
                payload_cursor += int(nbytes)
            rank_segments.append(tuple(segments))
            rank_payload_bytes.append(int(payload_cursor))
            send_bytes = max(send_bytes, int(payload_cursor))

        return tuple(rank_segments), tuple(rank_payload_bytes), int(send_bytes)

    def _ensure_mask_allgather_send_buffer(
        self,
        state: _BufferState,
        device: torch.device,
    ) -> torch.Tensor:
        send_bytes = int(state.mask_allgather_send_bytes)
        if send_bytes <= 0:
            return torch.empty(0, dtype=torch.uint8, device=device)
        send_buf = state.next_mask_allgather_send_u8
        need_alloc = (
            send_buf is None
            or send_buf.device != device
            or int(send_buf.numel()) < send_bytes
        )
        if need_alloc:
            send_buf = torch.empty(send_bytes, dtype=torch.uint8, device=device)
            state.next_mask_allgather_send_u8 = send_buf
        return send_buf[:send_bytes]

    def _ensure_mask_allgather_recv_buffer(
        self,
        state: _BufferState,
        device: torch.device,
        world_size: int,
    ) -> torch.Tensor:
        send_bytes = int(state.mask_allgather_send_bytes)
        recv_bytes = int(send_bytes) * int(world_size)
        if recv_bytes <= 0:
            return torch.empty(0, dtype=torch.uint8, device=device)
        recv_buf = state.next_mask_allgather_recv_u8
        need_alloc = (
            recv_buf is None
            or recv_buf.device != device
            or int(recv_buf.numel()) < recv_bytes
        )
        if need_alloc:
            recv_buf = torch.empty(recv_bytes, dtype=torch.uint8, device=device)
            state.next_mask_allgather_recv_u8 = recv_buf
        return recv_buf[:recv_bytes]

    def _pack_rank_owned_mask_to_allgather_send(
        self,
        packed_u8: torch.Tensor,
        send_u8: torch.Tensor,
        rank_segments: Tuple[Tuple[int, int, int, int], ...],
        profile_key: str,
        stage_pool: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> None:
        send_u8.zero_()
        if int(send_u8.numel()) == 0 or not rank_segments:
            return

        for seg_pos, (start_i, end_i, payload_offset, byte_len) in enumerate(rank_segments):
            start = int(start_i)
            end = int(end_i)
            payload_off = int(payload_offset)
            nbytes = int(byte_len)
            numel = int(end - start)
            if numel <= 0 or nbytes <= 0:
                continue

            t_unpack = self._profile_tic(packed_u8)
            mask_slice = self._unpack_mask_slice_from_packed(
                packed_u8=packed_u8,
                start=start,
                end=end,
                device=packed_u8.device,
                stage_slot=(seg_pos % 2),
                stage_pool=stage_pool,
            )
            self._profile_toc(f'{profile_key}.gather_pack_unpack_ms', t_unpack, mask_slice)

            send_slice = send_u8[payload_off : payload_off + nbytes]
            send_slice.zero_()
            t_pack = self._profile_tic(mask_slice)
            self._pack_mask_slice_to_packed(mask_slice, send_slice, 0, numel)
            self._profile_toc(f'{profile_key}.gather_pack_ms', t_pack, mask_slice)

    def _stitch_allgather_recv_to_packed_mask(
        self,
        recv_u8: torch.Tensor,
        packed_u8: torch.Tensor,
        rank_segments_by_rank: Tuple[Tuple[Tuple[int, int, int, int], ...], ...],
        rank_payload_bytes: Tuple[int, ...],
        send_bytes: int,
        profile_key: str,
        stage_pool: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> None:
        packed_u8.zero_()
        if int(send_bytes) <= 0 or int(recv_u8.numel()) == 0:
            return

        world_size = int(len(rank_segments_by_rank))
        if world_size <= 0:
            return

        for rank in range(world_size):
            payload_bytes = int(rank_payload_bytes[rank]) if rank < len(rank_payload_bytes) else 0
            if payload_bytes <= 0:
                continue
            base = int(rank) * int(send_bytes)
            rank_recv = recv_u8[base : base + int(send_bytes)]
            rank_segments = rank_segments_by_rank[rank]
            for seg_pos, (start_i, end_i, payload_offset, byte_len) in enumerate(rank_segments):
                start = int(start_i)
                end = int(end_i)
                payload_off = int(payload_offset)
                nbytes = int(byte_len)
                numel = int(end - start)
                if numel <= 0 or nbytes <= 0:
                    continue

                recv_slice = rank_recv[payload_off : payload_off + nbytes]
                t_unpack = self._profile_tic(recv_slice)
                mask_slice = self._unpack_mask_slice_from_packed(
                    packed_u8=recv_slice,
                    start=0,
                    end=numel,
                    device=packed_u8.device,
                    stage_slot=(seg_pos % 2),
                    stage_pool=stage_pool,
                )
                self._profile_toc(f'{profile_key}.gather_unpack_ms', t_unpack, mask_slice)

                t_pack = self._profile_tic(mask_slice)
                self._pack_mask_slice_to_packed(mask_slice, packed_u8, start, end)
                self._profile_toc(f'{profile_key}.gather_scatter_pack_ms', t_pack, mask_slice)

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

        raise RuntimeError(
            "Top-k hard-fail mode: Triton mask pack path unavailable. "
            "Torch pack fallback is disabled."
        )

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

        raise RuntimeError(
            "Top-k hard-fail mode: Triton mask unpack path unavailable. "
            "Torch unpack fallback is disabled."
        )

    def _allreduce_max_mask_sync(
        self,
        state: _BufferState,
        group: torch.distributed.ProcessGroup,
        profile_key: str,
    ) -> None:
        world_size = self._group_size(group)
        if state.use_packed_mask_only:
            packed = state.current_mask_packed_u8
            if packed is None:
                raise RuntimeError('Internal error: current packed mask buffer is missing.')
            if world_size <= 1 or packed.numel() == 0:
                return
            packed_ranges = [
                (int(start), int(end))
                for (start, end, _packed_offset, _byte_len, _owner) in state.mask_pack_segments
            ]
            self._allreduce_packed_mask_staged_max_(
                packed_u8=packed,
                ranges=packed_ranges,
                mask_numel=int(state.mask_numel),
                group=group,
                profile_key=profile_key,
            )
            return

        mask_u8 = state.current_mask_u8
        if mask_u8 is None:
            raise RuntimeError('Internal error: current mask_u8 buffer is missing.')
        if world_size <= 1 or mask_u8.numel() == 0:
            return

        # Legacy non-packed path: reduce directly on uint8 mask for correctness.
        self._allreduce_max_mask_(mask_u8, group, async_op=False, profile_key=profile_key)

    def _allreduce_max_mask_async_launch(
        self,
        state: _BufferState,
        group: torch.distributed.ProcessGroup,
        profile_key: str,
    ) -> Optional[Any]:
        world_size = self._group_size(group)
        if state.use_packed_mask_only:
            packed = state.next_mask_packed_u8
            if packed is None:
                raise RuntimeError('Internal error: next packed mask buffer is missing.')
            send_bytes = int(state.mask_allgather_send_bytes)
            if world_size <= 1 or packed.numel() == 0 or send_bytes <= 0:
                state.next_mask_allreduce_uses_packed = True
                return None

            rank = self._group_rank(group)
            rank_segments_by_rank = state.mask_allgather_segments_by_rank
            if rank < 0 or rank >= len(rank_segments_by_rank):
                raise RuntimeError(
                    'Top-k mask all-gather layout is inconsistent with process-group rank.'
                )
            local_payload_bytes = 0
            if rank < len(state.mask_allgather_rank_payload_bytes):
                local_payload_bytes = int(state.mask_allgather_rank_payload_bytes[rank])
            self._profile_add_value(
                f'{profile_key}.gather_send_bytes',
                float(send_bytes),
            )
            self._profile_add_value(
                f'{profile_key}.gather_local_payload_bytes',
                float(local_payload_bytes),
            )
            self._profile_add_value(
                f'{profile_key}.gather_recv_bytes',
                float(int(send_bytes) * int(world_size)),
            )


            send_u8 = self._ensure_mask_allgather_send_buffer(
                state,
                device=packed.device,
            )
            recv_u8 = self._ensure_mask_allgather_recv_buffer(
                state,
                device=packed.device,
                world_size=world_size,
            )

            t_pack_total = self._profile_tic(send_u8)
            self._pack_rank_owned_mask_to_allgather_send(
                packed_u8=packed,
                send_u8=send_u8,
                rank_segments=rank_segments_by_rank[rank],
                profile_key=profile_key,
                stage_pool=self._mask_stage_buffers_async,
            )
            self._profile_toc(f'{profile_key}.gather_pack_total_ms', t_pack_total, send_u8)

            state.next_mask_allreduce_uses_packed = True
            t_launch = self._profile_tic(None)
            try:
                handle = torch.distributed.all_gather_into_tensor(
                    recv_u8,
                    send_u8,
                    group=group,
                    async_op=True,
                )
            except Exception:
                recv_chunks = [
                    recv_u8[i * send_bytes : (i + 1) * send_bytes]
                    for i in range(int(world_size))
                ]
                handle = torch.distributed.all_gather(
                    recv_chunks,
                    send_u8,
                    group=group,
                    async_op=True,
                )
            self._profile_toc(f'{profile_key}.gather_launch_ms', t_launch, None)
            return handle

        mask_u8 = state.next_mask_u8
        if mask_u8 is None:
            raise RuntimeError('Internal error: next mask_u8 buffer is missing.')
        if world_size <= 1 or mask_u8.numel() == 0:
            state.next_mask_allreduce_uses_packed = False
            return None

        # Legacy non-packed path: reduce directly on uint8 mask for correctness.
        state.next_mask_allreduce_uses_packed = False
        return self._allreduce_max_mask_(
            mask_u8,
            group,
            async_op=True,
            profile_key=profile_key,
        )

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

        offloaded_residual = bool(
            self._offload_residual_to_cpu
            and residual.device.type == 'cpu'
            and grad_data.is_cuda
        )
        owner_work: List[Tuple[int, int]] = []
        prefetched_owner_pos: Optional[int] = None
        prefetched_residual: Optional[torch.Tensor] = None
        prefetched_ready_event: Optional[torch.cuda.Event] = None
        if offloaded_residual:
            for param_slice in param_slices:
                start = int(param_slice.start)
                end = int(param_slice.end)
                if end <= start:
                    continue
                if int(param_slice.owner) != int(group_rank):
                    continue
                owner_work.append((start, end))
            if owner_work:
                first_start, first_end = owner_work[0]
                prefetched_residual, prefetched_ready_event = self._prefetch_residual_slice_from_cpu(
                    residual_cpu=residual,
                    start=first_start,
                    end=first_end,
                    device=grad_data.device,
                    stage_slot=0,
                )
                prefetched_owner_pos = 0

        owner_work_pos = 0
        for param_slice in param_slices:
            start = int(param_slice.start)
            end = int(param_slice.end)
            param = param_slice.param
            numel = int(end - start)
            if numel <= 0:
                continue
            if int(param_slice.owner) != int(group_rank):
                continue

            residual_slice: Optional[torch.Tensor] = None
            if offloaded_residual:
                if owner_work_pos >= len(owner_work):
                    raise RuntimeError('Residual prefetch worklist out of bounds in mask build.')
                expected_start, expected_end = owner_work[owner_work_pos]
                if expected_start != start or expected_end != end:
                    raise RuntimeError(
                        'Residual prefetch worklist mismatch in mask build: '
                        f'expected=[{expected_start}:{expected_end}] got=[{start}:{end}].'
                    )
                if prefetched_owner_pos != owner_work_pos or prefetched_residual is None:
                    prefetched_residual, prefetched_ready_event = self._prefetch_residual_slice_from_cpu(
                        residual_cpu=residual,
                        start=start,
                        end=end,
                        device=grad_data.device,
                        stage_slot=(owner_work_pos % 2),
                    )
                    prefetched_owner_pos = owner_work_pos
                residual_slice = prefetched_residual
                if residual_slice is None:
                    raise RuntimeError('Residual prefetch produced an empty staging buffer.')
                if prefetched_ready_event is not None:
                    torch.cuda.current_stream(grad_data.device).wait_event(prefetched_ready_event)
                next_owner_work_pos = owner_work_pos + 1
                if next_owner_work_pos < len(owner_work):
                    next_start, next_end = owner_work[next_owner_work_pos]
                    prefetched_residual, prefetched_ready_event = self._prefetch_residual_slice_from_cpu(
                        residual_cpu=residual,
                        start=next_start,
                        end=next_end,
                        device=grad_data.device,
                        stage_slot=(next_owner_work_pos % 2),
                    )
                    prefetched_owner_pos = next_owner_work_pos
                else:
                    prefetched_owner_pos = None
                    prefetched_residual = None
                    prefetched_ready_event = None
                owner_work_pos += 1
            else:
                residual_slice = residual[start:end]

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

            if residual_slice is None:
                raise RuntimeError('Residual slice is missing while building top-k mask.')

            t_corrected = self._profile_tic(grad_data)
            exp_avg_prev = state['exp_avg'].view(-1).to(torch.float32)
            group = self._param_to_group.get(param, {})
            beta1 = float(group.get('betas', (0.9, 0.999))[0])

            grad_slice_fp32 = grad_data[start:end].to(torch.float32)
            corrected = exp_avg_prev * beta1
            corrected.add_(grad_slice_fp32, alpha=1.0 - beta1)
            corrected.add_(residual_slice)
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
        profile_tensor = mask_u8 if not async_op else None
        t_allreduce = self._profile_tic(profile_tensor)
        handle = torch.distributed.all_reduce(
            mask_u8,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
            async_op=async_op,
        )
        key = profile_key
        if key is None:
            key = 'mask.allreduce_async_launch_ms' if async_op else 'mask.allreduce_sync_ms'
        self._profile_toc(key, t_allreduce, profile_tensor)
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
            raise RuntimeError(
                "Top-k hard-fail mode: unexpected non-state mask tensor in synced mask build."
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
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.AVG,
            group=group,
        )
        self._profile_toc(f'{key_base}.allreduce_avg_ms', t_allreduce, tensor)

    def _fp8_allreduce_(
        self,
        tensor: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        profile_key: str = 'sparse.layer.fp8.payload',
    ) -> float:
        t_allreduce = self._profile_tic(tensor)
        try:
            torch.distributed.all_reduce(tensor, group=group)
        except Exception as exc:
            raise RuntimeError(
                "Top-k hard-fail mode: FP8 all-reduce failed; FP32 fallback is disabled."
            ) from exc
        return self._profile_toc(f'{profile_key}.allreduce_ms', t_allreduce, tensor)

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

    @staticmethod
    def _distopt_local_shard_range(
        buffer: _ParamAndGradBuffer,
        param: torch.nn.Parameter,
        group_rank: int,
        group_size: int,
    ) -> Tuple[int, int, int, int]:
        """Return local shard interval in grad-buffer space for one param."""
        param_start, param_end, bucket_id = buffer.param_index_map[param]
        param_start = int(param_start)
        param_end = int(param_end)
        param_numel = param_end - param_start
        bucket = buffer.buckets[int(bucket_id)]
        bucket_numel = int(bucket.grad_data.numel())
        if group_size <= 0:
            raise RuntimeError(f"Invalid group size {group_size} for distributed optimizer top-k.")
        if bucket_numel % group_size != 0:
            raise RuntimeError(
                f"Bucket size {bucket_numel} is not divisible by group size {group_size}."
            )
        shard_size = bucket_numel // group_size
        local_bucket_start = int(bucket.offset) + int(group_rank) * shard_size
        local_bucket_end = local_bucket_start + shard_size
        local_start = max(param_start, local_bucket_start)
        local_end = min(param_end, local_bucket_end)
        local_param_start = max(0, local_start - param_start)
        return local_start, local_end, local_param_start, param_numel

    @staticmethod
    def _stable_argsort_desc(values: torch.Tensor) -> torch.Tensor:
        try:
            return torch.argsort(values, descending=True, stable=True)
        except TypeError:
            # Fallback for torch builds that do not expose stable argsort.
            order = sorted(
                range(int(values.numel())),
                key=lambda i: float(values[i].item()),
                reverse=True,
            )
            return torch.tensor(order, device=values.device, dtype=torch.int64)

    def _distopt_global_topk_local_positions(
        self,
        local_scores_abs: torch.Tensor,
        local_param_start: int,
        global_param_numel: int,
        k: int,
        group: torch.distributed.ProcessGroup,
    ) -> torch.Tensor:
        """Select exact global top-k for a parameter and return local selected positions."""
        local_numel = int(local_scores_abs.numel())
        if k <= 0 or global_param_numel <= 0:
            return torch.empty(0, dtype=torch.int64, device=local_scores_abs.device)

        group_size = self._group_size(group)
        local_take = min(int(k), local_numel)
        if local_take > 0:
            local_scores, local_idx = torch.topk(local_scores_abs, k=local_take, sorted=False)
            local_global_idx = local_idx.to(torch.int64) + int(local_param_start)
            local_scores = local_scores.to(torch.float32)
        else:
            local_scores = torch.empty(0, dtype=torch.float32, device=local_scores_abs.device)
            local_global_idx = torch.empty(0, dtype=torch.int64, device=local_scores_abs.device)

        if group_size <= 1:
            if local_take <= 0:
                return torch.empty(0, dtype=torch.int64, device=local_scores_abs.device)
            if local_take > k:
                order = self._stable_argsort_desc(local_scores)
                local_global_idx = local_global_idx[order[:k]]
            return (local_global_idx - int(local_param_start)).to(torch.int64)

        take_tensor = torch.tensor([local_take], dtype=torch.int64, device=local_scores_abs.device)
        take_gather = [torch.zeros_like(take_tensor) for _ in range(group_size)]
        torch.distributed.all_gather(take_gather, take_tensor, group=group)
        gathered_take = [int(t.item()) for t in take_gather]
        max_take = max(gathered_take) if gathered_take else 0
        if max_take <= 0:
            return torch.empty(0, dtype=torch.int64, device=local_scores_abs.device)

        padded_scores = torch.full(
            (max_take,), -torch.inf, dtype=torch.float32, device=local_scores_abs.device
        )
        padded_idx = torch.full((max_take,), -1, dtype=torch.int64, device=local_scores_abs.device)
        if local_take > 0:
            padded_scores[:local_take].copy_(local_scores)
            padded_idx[:local_take].copy_(local_global_idx)

        gathered_scores = [torch.empty_like(padded_scores) for _ in range(group_size)]
        gathered_idx = [torch.empty_like(padded_idx) for _ in range(group_size)]
        torch.distributed.all_gather(gathered_scores, padded_scores, group=group)
        torch.distributed.all_gather(gathered_idx, padded_idx, group=group)

        score_chunks: List[torch.Tensor] = []
        idx_chunks: List[torch.Tensor] = []
        for rank_idx, rank_take in enumerate(gathered_take):
            if rank_take <= 0:
                continue
            score_chunks.append(gathered_scores[rank_idx][:rank_take])
            idx_chunks.append(gathered_idx[rank_idx][:rank_take])

        if not idx_chunks:
            return torch.empty(0, dtype=torch.int64, device=local_scores_abs.device)

        candidate_scores = torch.cat(score_chunks, dim=0)
        candidate_idx = torch.cat(idx_chunks, dim=0)
        valid = candidate_idx >= 0
        candidate_scores = candidate_scores[valid]
        candidate_idx = candidate_idx[valid]
        if candidate_idx.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=local_scores_abs.device)

        if int(candidate_idx.numel()) > int(k):
            # Deterministic tie-break: sort by idx ascending first, then stable
            # score sort descending.
            by_idx = torch.argsort(candidate_idx)
            candidate_idx = candidate_idx[by_idx]
            candidate_scores = candidate_scores[by_idx]
            by_score = self._stable_argsort_desc(candidate_scores)
            candidate_idx = candidate_idx[by_score[: int(k)]]

        local_begin = int(local_param_start)
        local_end = local_begin + local_numel
        local_selected = candidate_idx[
            (candidate_idx >= local_begin) & (candidate_idx < local_end)
        ]
        if local_selected.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=local_scores_abs.device)
        local_selected = local_selected - local_begin
        if local_selected.numel() > 1:
            local_selected = torch.unique(local_selected, sorted=True)
        return local_selected.to(torch.int64)

    def _reduce_dense_fallback(
        self,
        grad_data: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        state: _BufferState,
        profile_tag: str = 'dense.buffer',
    ) -> None:
        raise RuntimeError(
            "Top-k hard-fail mode: dense fallback path is disabled."
        )

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

        if force_all_reduce:
            raise RuntimeError(
                "Top-k hard-fail mode: force_all_reduce is disabled."
            )
        if int(train_iter) < self._start_iter:
            raise RuntimeError(
                "Top-k hard-fail mode: dense warmup mode is disabled. "
                f"Got train_iter={int(train_iter)} < topk_adams_start_iter={int(self._start_iter)}."
            )
        dense_mode = False
        use_fp8_quantized_payload = self._use_fp8_quantized_payload(int(train_iter))

        total_selected = 0
        total_numel = 0
        synced_grad_sq_sum: Optional[torch.Tensor] = None
        update_metric_sq_sum: Optional[torch.Tensor] = None
        move_clip_grad_to_reducer = bool(self._move_clip_grad_to_reducer)
        pending_updates: List[_PendingUpdate] = []
        synced_grad_buffers: List[Optional[torch.Tensor]] = [None] * len(self._buffers)

        if self._ddp_config.use_distributed_optimizer:
            local_grad_slices: List[torch.Tensor] = []
            residual_offload_pending_tensors: List[torch.Tensor] = []

            for buffer_idx, buffer in enumerate(self._buffers):
                grad_data = buffer.grad_data
                if grad_data is None or grad_data.numel() == 0:
                    continue

                state = self._buffer_states[buffer_idx]
                residual_storage = self._residual_storage(state)
                residual_cpu = state.residual_cpu if self._offload_residual_to_cpu else None

                param_slices = self._buffer_param_slices[buffer_idx]
                group = buffer.data_parallel_group
                group_rank = self._buffer_group_ranks[buffer_idx]
                group_size = max(1, self._group_size(group))
                next_dense_mode = bool(force_all_reduce or (int(train_iter) + 1) < self._start_iter)
                need_bootstrap_current_mask = (not dense_mode) and (not state.has_current_mask)

                if not param_slices:
                    if state.use_packed_mask_only:
                        current_packed = state.current_mask_packed_u8
                        next_packed = state.next_mask_packed_u8
                        if current_packed is None or next_packed is None:
                            raise RuntimeError('Internal error: packed mask buffers are missing.')
                        current_packed.zero_()
                        next_packed.zero_()
                    else:
                        current_mask_u8 = state.current_mask_u8
                        next_mask_u8 = state.next_mask_u8
                        if current_mask_u8 is None or next_mask_u8 is None:
                            raise RuntimeError('Internal error: mask_u8 buffers are missing.')
                        current_mask_u8.zero_()
                        next_mask_u8.zero_()
                    state.next_mask_allreduce_handle = None
                    state.next_mask_allreduce_event = None
                    state.next_mask_allreduce_uses_packed = False
                    state.next_mask_allreduce_unpacked_u8 = None
                    state.has_current_mask = False
                    state.has_next_mask = False
                    continue

                if dense_mode:
                    # Replace skipped default RS path with dense all-reduce in warmup/forced-dense mode.
                    self._allreduce_average_(
                        grad_data,
                        group,
                        profile_key='distopt.dense.buffer.grad_allreduce',
                    )
                    if next_dense_mode:
                        if state.use_packed_mask_only:
                            next_packed = state.next_mask_packed_u8
                            if next_packed is None:
                                raise RuntimeError('Internal error: next packed mask buffer is missing.')
                            self._fill_packed_mask_ones(next_packed, state.mask_numel)
                        else:
                            next_mask_u8 = state.next_mask_u8
                            if next_mask_u8 is None:
                                raise RuntimeError('Internal error: next mask_u8 buffer is missing.')
                            next_mask_u8.fill_(1)
                    else:
                        if state.use_packed_mask_only:
                            next_packed = state.next_mask_packed_u8
                            if next_packed is None:
                                raise RuntimeError('Internal error: next packed mask buffer is missing.')
                            next_packed.zero_()
                        else:
                            next_mask_u8 = state.next_mask_u8
                            if next_mask_u8 is None:
                                raise RuntimeError('Internal error: next mask_u8 buffer is missing.')
                            next_mask_u8.zero_()

                    dense_prefetch_work: List[Tuple[int, int]] = []
                    prefetched_dense_pos: Optional[int] = None
                    prefetched_dense: Optional[torch.Tensor] = None
                    prefetched_dense_event: Optional[torch.cuda.Event] = None
                    if (
                        (not next_dense_mode)
                        and self._offload_residual_to_cpu
                        and residual_cpu is not None
                    ):
                        for param_slice in param_slices:
                            param = param_slice.param
                            local_start, local_end, _, _ = self._distopt_local_shard_range(
                                buffer=buffer,
                                param=param,
                                group_rank=group_rank,
                                group_size=group_size,
                            )
                            if int(local_end - local_start) > 0:
                                dense_prefetch_work.append((int(local_start), int(local_end)))
                        if dense_prefetch_work:
                            first_start, first_end = dense_prefetch_work[0]
                            prefetched_dense, prefetched_dense_event = self._prefetch_residual_slice_from_cpu(
                                residual_cpu=residual_cpu,
                                start=first_start,
                                end=first_end,
                                device=grad_data.device,
                                stage_slot=0,
                            )
                            prefetched_dense_pos = 0
                    dense_work_pos = 0

                    for param_slice in param_slices:
                        param = param_slice.param
                        local_start, local_end, _, _ = self._distopt_local_shard_range(
                            buffer=buffer,
                            param=param,
                            group_rank=group_rank,
                            group_size=group_size,
                        )
                        local_numel = int(local_end - local_start)
                        if local_numel <= 0:
                            continue
                        grad_slice = grad_data[local_start:local_end]

                        residual_local: Optional[torch.Tensor] = None
                        if not next_dense_mode:
                            if self._offload_residual_to_cpu and residual_cpu is not None:
                                if dense_work_pos >= len(dense_prefetch_work):
                                    raise RuntimeError(
                                        'Residual dense prefetch worklist out of bounds.'
                                    )
                                expected_start, expected_end = dense_prefetch_work[dense_work_pos]
                                if expected_start != int(local_start) or expected_end != int(local_end):
                                    raise RuntimeError(
                                        'Residual dense prefetch worklist mismatch: '
                                        f'expected=[{expected_start}:{expected_end}] '
                                        f'got=[{int(local_start)}:{int(local_end)}].'
                                    )
                                if prefetched_dense_pos != dense_work_pos or prefetched_dense is None:
                                    prefetched_dense, prefetched_dense_event = self._prefetch_residual_slice_from_cpu(
                                        residual_cpu=residual_cpu,
                                        start=int(local_start),
                                        end=int(local_end),
                                        device=grad_data.device,
                                        stage_slot=(dense_work_pos % 2),
                                    )
                                    prefetched_dense_pos = dense_work_pos
                                residual_local = prefetched_dense
                                if residual_local is None:
                                    raise RuntimeError(
                                        'Residual dense prefetch returned an empty staging tensor.'
                                    )
                                if prefetched_dense_event is not None:
                                    t_prefetch_wait = self._profile_tic(grad_data)
                                    torch.cuda.current_stream(grad_data.device).wait_event(
                                        prefetched_dense_event
                                    )
                                    self._profile_toc(
                                        'distopt.residual.prefetch_wait_ms',
                                        t_prefetch_wait,
                                        grad_data,
                                    )
                                next_dense_work_pos = dense_work_pos + 1
                                if next_dense_work_pos < len(dense_prefetch_work):
                                    next_start, next_end = dense_prefetch_work[next_dense_work_pos]
                                    t_prefetch = self._profile_tic(grad_data)
                                    prefetched_dense, prefetched_dense_event = self._prefetch_residual_slice_from_cpu(
                                        residual_cpu=residual_cpu,
                                        start=next_start,
                                        end=next_end,
                                        device=grad_data.device,
                                        stage_slot=(next_dense_work_pos % 2),
                                    )
                                    self._profile_toc(
                                        'distopt.residual.prefetch.cpu_to_gpu_ms',
                                        t_prefetch,
                                        prefetched_dense,
                                    )
                                    self._profile_add_value('distopt.residual.prefetch_calls', 1.0)
                                    prefetched_dense_pos = next_dense_work_pos
                                else:
                                    prefetched_dense_pos = None
                                    prefetched_dense = None
                                    prefetched_dense_event = None
                                dense_work_pos += 1
                            else:
                                residual_local = residual_storage[local_start:local_end]

                        if not next_dense_mode:
                            if state.use_packed_mask_only:
                                next_packed = state.next_mask_packed_u8
                                if next_packed is None:
                                    raise RuntimeError('Internal error: next packed mask buffer is missing.')
                                next_mask_local = self._mask_stage_view(
                                    stage_slot=(dense_work_pos % 2),
                                    numel=local_numel,
                                    device=grad_data.device,
                                )
                                next_mask_local.zero_()
                                if param_slice.exclude_from_topk or next_density >= 1.0:
                                    next_mask_local.fill_(1)
                                else:
                                    state_entry = self._optimizer_state_entry(param)
                                    if (
                                        state_entry is None
                                        or 'exp_avg' not in state_entry
                                        or int(state_entry['exp_avg'].numel()) != local_numel
                                    ):
                                        # Keep behavior conservative when state mapping is incomplete.
                                        next_mask_local.fill_(1)
                                    else:
                                        if residual_local is None:
                                            raise RuntimeError(
                                                'Residual slice is missing in dense mask update.'
                                            )
                                        exp_avg_prev = state_entry['exp_avg'].view(-1).to(torch.float32)
                                        group_cfg = self._param_to_group.get(param, {})
                                        beta1, _ = group_cfg.get('betas', (0.9, 0.999))
                                        beta1 = float(beta1)
                                        corrected_local = exp_avg_prev * beta1
                                        corrected_local.add_(
                                            grad_slice.to(torch.float32), alpha=1.0 - beta1
                                        )
                                        corrected_local.add_(residual_local)
                                        k_next = self._topk_k(next_density, local_numel)
                                        if k_next >= local_numel:
                                            next_mask_local.fill_(1)
                                        elif k_next > 0:
                                            _, local_selected_next = torch.topk(
                                                corrected_local.abs(), k=k_next, sorted=False
                                            )
                                            next_mask_local.index_fill_(0, local_selected_next, 1)
                                self._pack_mask_slice_to_packed(
                                    next_mask_local,
                                    next_packed,
                                    int(local_start),
                                    int(local_end),
                                )
                            else:
                                if param_slice.exclude_from_topk or next_density >= 1.0:
                                    state.next_mask_u8[local_start:local_end].fill_(1)
                                else:
                                    state_entry = self._optimizer_state_entry(param)
                                    if (
                                        state_entry is None
                                        or 'exp_avg' not in state_entry
                                        or int(state_entry['exp_avg'].numel()) != local_numel
                                    ):
                                        # Keep behavior conservative when state mapping is incomplete.
                                        state.next_mask_u8[local_start:local_end].fill_(1)
                                    else:
                                        if residual_local is None:
                                            raise RuntimeError(
                                                'Residual slice is missing in dense mask update.'
                                            )
                                        exp_avg_prev = state_entry['exp_avg'].view(-1).to(torch.float32)
                                        group_cfg = self._param_to_group.get(param, {})
                                        beta1, _ = group_cfg.get('betas', (0.9, 0.999))
                                        beta1 = float(beta1)
                                        corrected_local = exp_avg_prev * beta1
                                        corrected_local.add_(
                                            grad_slice.to(torch.float32), alpha=1.0 - beta1
                                        )
                                        corrected_local.add_(residual_local)
                                        k_next = self._topk_k(next_density, local_numel)
                                        if k_next >= local_numel:
                                            state.next_mask_u8[local_start:local_end].fill_(1)
                                        elif k_next > 0:
                                            _, local_selected_next = torch.topk(
                                                corrected_local.abs(), k=k_next, sorted=False
                                            )
                                            selected_mask_next = torch.zeros(
                                                local_numel, dtype=torch.bool, device=grad_data.device
                                            )
                                            selected_mask_next.index_fill_(0, local_selected_next, True)
                                            state.next_mask_u8[local_start:local_end].copy_(
                                                selected_mask_next.to(torch.uint8)
                                            )

                        if self._offload_residual_to_cpu and residual_cpu is not None:
                            if next_dense_mode:
                                residual_cpu[local_start:local_end].zero_()
                            else:
                                if residual_local is None:
                                    raise RuntimeError(
                                        'Residual slice is missing for dense residual offload.'
                                    )
                                residual_local.zero_()
                                t_offload_queue = self._profile_tic(residual_local)
                                self._queue_residual_slice_offload_to_cpu(
                                    residual_gpu=residual_local,
                                    residual_cpu=residual_cpu,
                                    start=int(local_start),
                                    end=int(local_end),
                                )
                                self._profile_toc(
                                    'distopt.residual.offload_queue_ms',
                                    t_offload_queue,
                                    residual_local,
                                )
                                self._profile_add_value('distopt.residual.offload_queue_calls', 1.0)
                        else:
                            residual_storage[local_start:local_end].zero_()

                        if state.use_packed_mask_only:
                            current_packed = state.current_mask_packed_u8
                            if current_packed is None:
                                raise RuntimeError('Internal error: current packed mask buffer is missing.')
                            self._set_packed_mask_slice_constant(
                                packed_u8=current_packed,
                                start=int(local_start),
                                end=int(local_end),
                                value=1,
                                stage_slot=(dense_work_pos % 2),
                                device=grad_data.device,
                            )
                        else:
                            current_mask_u8 = state.current_mask_u8
                            if current_mask_u8 is None:
                                raise RuntimeError('Internal error: current mask_u8 buffer is missing.')
                            current_mask_u8[local_start:local_end].fill_(1)
                        total_selected += local_numel
                        total_numel += local_numel
                        if move_clip_grad_to_reducer:
                            local_grad_slices.append(grad_slice)
                        if synced_grad_sq_sum is None:
                            synced_grad_sq_sum = torch.zeros(
                                (), dtype=torch.float64, device=grad_slice.device
                            )
                        if param_is_not_shared(
                            param
                        ) and tensor_parallel.param_is_not_tensor_parallel_duplicate(param):
                            synced_grad_sq_sum.add_(grad_slice.to(torch.float64).pow(2).sum())
                    # Broadcast dense mask ownership so every rank keeps a global mask view.
                    self._allreduce_max_mask_sync(
                        state,
                        group,
                        profile_key='distopt.mask.current_sync_ms',
                    )
                    state.has_current_mask = True
                    if next_dense_mode:
                        self._wait_next_mask_allreduce(state)
                        if state.use_packed_mask_only:
                            next_packed = state.next_mask_packed_u8
                            if next_packed is None:
                                raise RuntimeError('Internal error: next packed mask buffer is missing.')
                            self._fill_packed_mask_ones(next_packed, state.mask_numel)
                        else:
                            next_mask_u8 = state.next_mask_u8
                            if next_mask_u8 is None:
                                raise RuntimeError('Internal error: next mask_u8 buffer is missing.')
                            next_mask_u8.fill_(1)
                        state.next_mask_allreduce_handle = None
                        state.next_mask_allreduce_event = None
                        state.next_mask_allreduce_uses_packed = False
                        state.next_mask_allreduce_unpacked_u8 = None
                        state.has_next_mask = True
                    else:
                        self._launch_next_mask_allreduce_async(state, group)
                    continue

                local_mapping_bad = 0
                for param_slice in param_slices:
                    param = param_slice.param
                    local_start, local_end, _, _ = self._distopt_local_shard_range(
                        buffer=buffer,
                        param=param,
                        group_rank=group_rank,
                        group_size=group_size,
                    )
                    local_numel = int(local_end - local_start)
                    if local_numel <= 0:
                        continue
                    state_entry = self._optimizer_state_entry(param)
                    if (
                        param not in self._param_to_group
                        or state_entry is None
                        or 'exp_avg' not in state_entry
                        or int(state_entry['exp_avg'].numel()) != local_numel
                    ):
                        local_mapping_bad = 1
                        break
                if local_mapping_bad != 0:
                    raise RuntimeError(
                        "Top-k hard-fail mode: incomplete sharded AdamS state mapping in distributed optimizer mode."
                    )

                if need_bootstrap_current_mask:
                    if state.use_packed_mask_only:
                        current_packed = state.current_mask_packed_u8
                        if current_packed is None:
                            raise RuntimeError('Internal error: current packed mask buffer is missing.')
                        current_packed.zero_()
                    else:
                        current_mask_u8 = state.current_mask_u8
                        if current_mask_u8 is None:
                            raise RuntimeError('Internal error: current mask_u8 buffer is missing.')
                        current_mask_u8.zero_()
                if state.use_packed_mask_only:
                    next_packed = state.next_mask_packed_u8
                    if next_packed is None:
                        raise RuntimeError('Internal error: next packed mask buffer is missing.')
                    next_packed.zero_()
                else:
                    next_mask_u8 = state.next_mask_u8
                    if next_mask_u8 is None:
                        raise RuntimeError('Internal error: next mask_u8 buffer is missing.')
                    next_mask_u8.zero_()
                local_entries: List[Dict[str, Any]] = []
                sparse_prefetch_work: List[Tuple[int, int]] = []
                prefetched_sparse_pos: Optional[int] = None
                prefetched_sparse: Optional[torch.Tensor] = None
                prefetched_sparse_event: Optional[torch.cuda.Event] = None
                if self._offload_residual_to_cpu and residual_cpu is not None:
                    for param_slice in param_slices:
                        param = param_slice.param
                        local_start, local_end, _, _ = self._distopt_local_shard_range(
                            buffer=buffer,
                            param=param,
                            group_rank=group_rank,
                            group_size=group_size,
                        )
                        if int(local_end - local_start) > 0:
                            sparse_prefetch_work.append((int(local_start), int(local_end)))
                    if sparse_prefetch_work:
                        first_start, first_end = sparse_prefetch_work[0]
                        t_prefetch = self._profile_tic(grad_data)
                        prefetched_sparse, prefetched_sparse_event = self._prefetch_residual_slice_from_cpu(
                            residual_cpu=residual_cpu,
                            start=first_start,
                            end=first_end,
                            device=grad_data.device,
                            stage_slot=0,
                        )
                        self._profile_toc(
                            'distopt.residual.prefetch.cpu_to_gpu_ms',
                            t_prefetch,
                            prefetched_sparse,
                        )
                        self._profile_add_value('distopt.residual.prefetch_calls', 1.0)
                        prefetched_sparse_pos = 0
                sparse_work_pos = 0
                single_pass_residual_offload = bool(
                    self._offload_residual_to_cpu
                    and residual_cpu is not None
                    and not use_fp8_quantized_payload
                )
                residual_offload_stage_events: List[Optional[torch.cuda.Event]] = [None, None]
                residual_offload_stage_pos = 0

                for param_slice in param_slices:
                    start = int(param_slice.start)
                    end = int(param_slice.end)
                    param = param_slice.param
                    if end <= start:
                        continue

                    local_start, local_end, _, _ = self._distopt_local_shard_range(
                        buffer=buffer,
                        param=param,
                        group_rank=group_rank,
                        group_size=group_size,
                    )
                    local_numel = max(0, int(local_end - local_start))
                    local_slice_end = local_start + local_numel
                    if local_numel <= 0:
                        continue

                    group_cfg = self._param_to_group.get(param, {})
                    beta1, _ = group_cfg.get('betas', (0.9, 0.999))
                    beta1 = float(beta1)

                    param_state = self._ensure_optimizer_state_entry(param)
                    exp_avg_prev = param_state['exp_avg'].view(-1).to(torch.float32).clone()
                    grad_slice = grad_data[local_start:local_slice_end]
                    grad_fp32 = grad_slice.to(torch.float32)
                    residual_slice: Optional[torch.Tensor] = None
                    if self._offload_residual_to_cpu and residual_cpu is not None:
                        if sparse_work_pos >= len(sparse_prefetch_work):
                            raise RuntimeError('Residual sparse prefetch worklist out of bounds.')
                        expected_start, expected_end = sparse_prefetch_work[sparse_work_pos]
                        if expected_start != int(local_start) or expected_end != int(local_slice_end):
                            raise RuntimeError(
                                'Residual sparse prefetch worklist mismatch: '
                                f'expected=[{expected_start}:{expected_end}] '
                                f'got=[{int(local_start)}:{int(local_slice_end)}].'
                            )
                        if prefetched_sparse_pos != sparse_work_pos or prefetched_sparse is None:
                            t_prefetch = self._profile_tic(grad_data)
                            prefetched_sparse, prefetched_sparse_event = self._prefetch_residual_slice_from_cpu(
                                residual_cpu=residual_cpu,
                                start=int(local_start),
                                end=int(local_slice_end),
                                device=grad_data.device,
                                stage_slot=(sparse_work_pos % 2),
                            )
                            self._profile_toc(
                                'distopt.residual.prefetch.cpu_to_gpu_ms',
                                t_prefetch,
                                prefetched_sparse,
                            )
                            self._profile_add_value('distopt.residual.prefetch_calls', 1.0)
                            prefetched_sparse_pos = sparse_work_pos
                        residual_slice = prefetched_sparse
                        if residual_slice is None:
                            raise RuntimeError('Residual sparse prefetch returned an empty staging tensor.')
                        if prefetched_sparse_event is not None:
                            t_prefetch_wait = self._profile_tic(grad_data)
                            torch.cuda.current_stream(grad_data.device).wait_event(
                                prefetched_sparse_event
                            )
                            self._profile_toc(
                                'distopt.residual.prefetch_wait_ms',
                                t_prefetch_wait,
                                grad_data,
                            )
                        next_sparse_work_pos = sparse_work_pos + 1
                        if next_sparse_work_pos < len(sparse_prefetch_work):
                            next_start, next_end = sparse_prefetch_work[next_sparse_work_pos]
                            t_prefetch = self._profile_tic(grad_data)
                            prefetched_sparse, prefetched_sparse_event = self._prefetch_residual_slice_from_cpu(
                                residual_cpu=residual_cpu,
                                start=next_start,
                                end=next_end,
                                device=grad_data.device,
                                stage_slot=(next_sparse_work_pos % 2),
                            )
                            self._profile_toc(
                                'distopt.residual.prefetch.cpu_to_gpu_ms',
                                t_prefetch,
                                prefetched_sparse,
                            )
                            self._profile_add_value('distopt.residual.prefetch_calls', 1.0)
                            prefetched_sparse_pos = next_sparse_work_pos
                        else:
                            prefetched_sparse_pos = None
                            prefetched_sparse = None
                            prefetched_sparse_event = None
                        sparse_work_pos += 1
                    else:
                        residual_slice = residual_storage[local_start:local_slice_end]

                    corrected_local = exp_avg_prev * beta1
                    corrected_local.add_(grad_fp32, alpha=1.0 - beta1)
                    corrected_local.add_(residual_slice)
                    local_scores_abs = corrected_local.abs()

                    if need_bootstrap_current_mask:
                        if param_slice.exclude_from_topk or current_density >= 1.0:
                            local_selected_current = torch.arange(
                                local_numel, device=grad_data.device, dtype=torch.int64
                            )
                        else:
                            # Bootstrap only once for step-0 sparse mode. Subsequent steps use
                            # the pre-synchronized previous-step mask from prepare_pre_forward.
                            k_current = self._topk_k(current_density, local_numel)
                            if k_current <= 0:
                                local_selected_current = torch.empty(
                                    0, dtype=torch.int64, device=grad_data.device
                                )
                            elif k_current >= local_numel:
                                local_selected_current = torch.arange(
                                    local_numel, device=grad_data.device, dtype=torch.int64
                                )
                            else:
                                _, local_selected_current = torch.topk(
                                    local_scores_abs, k=k_current, sorted=False
                                )
                        if state.use_packed_mask_only:
                            current_packed = state.current_mask_packed_u8
                            if current_packed is None:
                                raise RuntimeError('Internal error: current packed mask buffer is missing.')
                            self._set_packed_mask_slice_selected(
                                packed_u8=current_packed,
                                start=int(local_start),
                                end=int(local_slice_end),
                                selected_idx=local_selected_current,
                                stage_slot=(sparse_work_pos % 2),
                                device=grad_data.device,
                            )
                        else:
                            selected_mask_current = torch.zeros(
                                local_numel, dtype=torch.bool, device=grad_data.device
                            )
                            if local_selected_current.numel() > 0:
                                selected_mask_current.index_fill_(0, local_selected_current, True)
                            state.current_mask_u8[local_start:local_slice_end].copy_(
                                selected_mask_current.to(torch.uint8)
                            )

                    if param_slice.exclude_from_topk or next_dense_mode or next_density >= 1.0:
                        local_selected_next = torch.arange(
                            local_numel, device=grad_data.device, dtype=torch.int64
                        )
                    else:
                        # Build next-step ownership mask locally; ranks synchronize
                        # it asynchronously via all-gather at the end of reduce().
                        k_next = self._topk_k(next_density, local_numel)
                        if k_next <= 0:
                            local_selected_next = torch.empty(
                                0, dtype=torch.int64, device=grad_data.device
                            )
                        elif k_next >= local_numel:
                            local_selected_next = torch.arange(
                                local_numel, device=grad_data.device, dtype=torch.int64
                            )
                        else:
                            _, local_selected_next = torch.topk(
                                local_scores_abs, k=k_next, sorted=False
                            )
                    if state.use_packed_mask_only:
                        next_packed = state.next_mask_packed_u8
                        if next_packed is None:
                            raise RuntimeError('Internal error: next packed mask buffer is missing.')
                        self._set_packed_mask_slice_selected(
                            packed_u8=next_packed,
                            start=int(local_start),
                            end=int(local_slice_end),
                            selected_idx=local_selected_next,
                            stage_slot=(sparse_work_pos % 2),
                            device=grad_data.device,
                        )
                    else:
                        selected_mask_next = torch.zeros(
                            local_numel, dtype=torch.bool, device=grad_data.device
                        )
                        if local_selected_next.numel() > 0:
                            selected_mask_next.index_fill_(0, local_selected_next, True)
                        state.next_mask_u8[local_start:local_slice_end].copy_(
                            selected_mask_next.to(torch.uint8)
                        )
                    local_entries.append(
                        {
                            "param": param,
                            "local_start": local_start,
                            "local_slice_end": local_slice_end,
                            "beta1": beta1,
                            "exp_avg_prev": exp_avg_prev,
                            "corrected_local": corrected_local,
                            "grad_slice": grad_slice,
                            "payload_offset": 0,
                            "payload_length": 0,
                        }
                    )

                if need_bootstrap_current_mask:
                    # Bootstrap current mask once, then keep using previous-step mask.
                    self._allreduce_max_mask_sync(
                        state,
                        group,
                        profile_key='distopt.mask.current_bootstrap_sync_ms',
                    )
                    state.has_current_mask = True

                # Hide next-step mask synchronization under backward/optimizer work.
                self._launch_next_mask_allreduce_async(state, group)

                if state.use_packed_mask_only:
                    current_packed = state.current_mask_packed_u8
                    if current_packed is None:
                        raise RuntimeError('Internal error: current packed mask buffer is missing.')
                    global_ranges = [
                        (int(param_slice.start), int(param_slice.end))
                        for param_slice in param_slices
                        if int(param_slice.end) > int(param_slice.start)
                    ]
                    selected_global_idx = self._selected_global_idx_from_packed_ranges(
                        packed_u8=current_packed,
                        ranges=global_ranges,
                        device=grad_data.device,
                    )
                else:
                    current_mask_u8 = state.current_mask_u8
                    if current_mask_u8 is None:
                        raise RuntimeError('Internal error: current mask_u8 buffer is missing.')
                    selected_global_idx = torch.nonzero(current_mask_u8, as_tuple=False).flatten()

                for entry in local_entries:
                    local_start = int(entry["local_start"])
                    local_slice_end = int(entry["local_slice_end"])
                    local_numel = int(local_slice_end - local_start)
                    if local_numel <= 0:
                        entry["selected_idx"] = torch.empty(
                            0,
                            dtype=torch.int64,
                            device=grad_data.device,
                        )
                        entry["payload_offset"] = 0
                        entry["payload_length"] = 0
                        continue

                    payload_start = int(
                        torch.searchsorted(
                            selected_global_idx,
                            torch.tensor(
                                local_start,
                                dtype=torch.int64,
                                device=selected_global_idx.device,
                            ),
                            right=False,
                        ).item()
                    )
                    payload_end = int(
                        torch.searchsorted(
                            selected_global_idx,
                            torch.tensor(
                                local_slice_end,
                                dtype=torch.int64,
                                device=selected_global_idx.device,
                            ),
                            right=False,
                        ).item()
                    )
                    payload_length = max(0, payload_end - payload_start)
                    entry["payload_offset"] = payload_start
                    entry["payload_length"] = payload_length

                    if payload_length > 0:
                        selected_global_local = selected_global_idx[payload_start:payload_end]
                        local_selected_idx = (selected_global_local - int(local_start)).to(torch.int64)
                        if int(local_selected_idx.numel()) != payload_length:
                            raise RuntimeError(
                                "Top-k dist-opt payload selection mismatch: derived local selected "
                                "count does not match payload length."
                            )
                        if bool((local_selected_idx < 0).any().item()) or bool((local_selected_idx >= local_numel).any().item()):
                            raise RuntimeError(
                                "Top-k dist-opt payload selection mismatch: derived local selected "
                                "indices are out of local shard bounds."
                            )
                    else:
                        local_selected_idx = torch.empty(
                            0,
                            dtype=torch.int64,
                            device=grad_data.device,
                        )

                    entry["selected_idx"] = local_selected_idx
                    total_selected += int(payload_length)
                    total_numel += local_numel

                total_payload_numel = int(selected_global_idx.numel())
                synced_payload_flat: Optional[torch.Tensor] = None
                quant_error_flat: Optional[torch.Tensor] = None
                if total_payload_numel > 0:
                    packed_payload = torch.zeros(
                        total_payload_numel,
                        dtype=torch.float32,
                        device=grad_data.device,
                    )
                    for entry in local_entries:
                        local_selected_idx = entry.get("selected_idx", None)
                        if local_selected_idx is None:
                            raise RuntimeError('Internal error: missing local selected indices.')
                        payload_start = int(entry.get("payload_offset", 0))
                        payload_length = int(entry.get("payload_length", 0))
                        if payload_length <= 0:
                            continue
                        payload_end = payload_start + payload_length

                        local_selected_values = entry["corrected_local"].index_select(
                            0,
                            local_selected_idx.to(torch.int64),
                        )
                        if int(local_selected_values.numel()) != payload_length:
                            raise RuntimeError(
                                "Top-k dist-opt payload packing mismatch: local selected count "
                                "changed between mask sync and payload pack."
                            )
                        packed_payload[payload_start:payload_end].copy_(local_selected_values)

                    t_payload = self._profile_tic(packed_payload)
                    if use_fp8_quantized_payload:
                        synced_payload_flat, quant_error_flat = self._fp8_quantized_allreduce_packed(
                            packed_payload,
                            group,
                            train_iter=int(train_iter),
                            profile_key_prefix='distopt.sparse.buffer.fp8',
                        )
                    else:
                        synced_payload_flat = packed_payload
                        self._allreduce_average_(
                            synced_payload_flat,
                            group,
                            profile_key='distopt.sparse.buffer.payload_fp32',
                        )
                    self._profile_toc(
                        'distopt.sparse.buffer.payload_total_ms',
                        t_payload,
                        packed_payload,
                    )

                if single_pass_residual_offload and quant_error_flat is not None:
                    raise RuntimeError(
                        "Top-k hard-fail mode: dist-opt single-pass residual offload does not support FP8 quantized payload residual correction."
                    )

                for entry in local_entries:
                    param = entry["param"]
                    local_start = int(entry["local_start"])
                    local_slice_end = int(entry["local_slice_end"])
                    beta1 = float(entry["beta1"])
                    exp_avg_prev = entry["exp_avg_prev"]
                    corrected_local = entry["corrected_local"]
                    grad_slice = entry["grad_slice"]

                    local_numel = int(local_slice_end - local_start)
                    selected_idx = entry.get("selected_idx", None)
                    if selected_idx is None:
                        raise RuntimeError('Internal error: missing selected_idx for dist-opt sparse entry.')
                    selected_count = int(selected_idx.numel())

                    selected_idx_i64 = selected_idx.to(torch.int64)
                    offload_stage_slot = -1
                    if single_pass_residual_offload:
                        offload_stage_slot = int(residual_offload_stage_pos % 2)
                        residual_offload_stage_pos += 1
                        prior_stage_event = residual_offload_stage_events[offload_stage_slot]
                        if prior_stage_event is not None:
                            t_stage_wait = self._profile_tic(grad_data)
                            torch.cuda.current_stream(grad_data.device).wait_event(prior_stage_event)
                            self._profile_toc(
                                'distopt.residual.offload_stage_wait_ms',
                                t_stage_wait,
                                grad_data,
                            )
                            self._profile_add_value(
                                'distopt.residual.offload_stage_wait_calls',
                                1.0,
                            )
                            residual_offload_stage_events[offload_stage_slot] = None

                        t_stage_copy = self._profile_tic(corrected_local)
                        residual_updated = self._residual_stage_view(
                            stage_slot=offload_stage_slot,
                            numel=local_numel,
                            device=grad_data.device,
                        )
                        residual_updated.copy_(corrected_local)
                        self._profile_toc(
                            'distopt.residual.offload_stage_copy_ms',
                            t_stage_copy,
                            residual_updated,
                        )
                        self._profile_add_value(
                            'distopt.residual.offload_stage_copy_calls',
                            1.0,
                        )
                    else:
                        residual_updated = corrected_local.clone()

                    if selected_count > 0:
                        residual_updated.index_fill_(0, selected_idx_i64, 0.0)

                    synced_momentum = torch.zeros_like(corrected_local)
                    payload_length = int(entry.get("payload_length", 0))
                    if payload_length > 0:
                        if synced_payload_flat is None:
                            raise RuntimeError(
                                "Internal error: dist-opt sparse payload all-reduce output is missing."
                            )
                        payload_start = int(entry.get("payload_offset", 0))
                        payload_end = payload_start + payload_length
                        if int(selected_idx.numel()) != payload_length:
                            raise RuntimeError(
                                "Top-k dist-opt payload unpack mismatch: selected count changed "
                                "between pack and unpack passes."
                            )
                        synced_selected = synced_payload_flat[payload_start:payload_end]
                        synced_momentum.index_copy_(0, selected_idx_i64, synced_selected)
                        if quant_error_flat is not None:
                            residual_selected = residual_updated.index_select(0, selected_idx_i64)
                            residual_selected.add_(quant_error_flat[payload_start:payload_end])
                            residual_updated.index_copy_(0, selected_idx_i64, residual_selected)
                    if self._offload_residual_to_cpu and residual_cpu is not None:
                        t_offload_queue = self._profile_tic(residual_updated)
                        queued_async = self._queue_residual_slice_offload_to_cpu(
                            residual_gpu=residual_updated,
                            residual_cpu=residual_cpu,
                            start=local_start,
                            end=local_slice_end,
                        )
                        self._profile_toc(
                            'distopt.residual.offload_queue_ms',
                            t_offload_queue,
                            residual_updated,
                        )
                        self._profile_add_value('distopt.residual.offload_queue_calls', 1.0)
                        if queued_async:
                            residual_offload_pending_tensors.append(residual_updated)
                            self._profile_add_value('distopt.residual.offload_async_buffers', 1.0)
                            if (
                                single_pass_residual_offload
                                and offload_stage_slot >= 0
                                and self._param_offload_stream is not None
                            ):
                                done_event = torch.cuda.Event()
                                done_event.record(self._param_offload_stream)
                                residual_offload_stage_events[offload_stage_slot] = done_event
                        else:
                            self._profile_add_value('distopt.residual.offload_sync_calls', 1.0)
                    else:
                        residual_storage[local_start:local_slice_end].copy_(residual_updated)

                    entry["corrected_local"] = None

                    if abs(1.0 - beta1) < 1e-12:
                        synced_grad = torch.zeros_like(synced_momentum)
                    else:
                        synced_grad = (synced_momentum - exp_avg_prev * beta1) / (1.0 - beta1)
                    grad_slice.copy_(synced_grad.to(grad_slice.dtype))

                    if move_clip_grad_to_reducer and local_numel > 0:
                        local_grad_slices.append(grad_slice)
                    if synced_grad_sq_sum is None:
                        synced_grad_sq_sum = torch.zeros(
                            (), dtype=torch.float64, device=synced_grad.device
                        )
                    if param_is_not_shared(
                        param
                    ) and tensor_parallel.param_is_not_tensor_parallel_duplicate(param):
                        synced_grad_sq_sum.add_(synced_grad.to(torch.float64).pow(2).sum())

            if residual_offload_pending_tensors and self._param_offload_stream is not None:
                wait_tensor = residual_offload_pending_tensors[0]
                t_residual_wait = self._profile_tic(wait_tensor)
                torch.cuda.current_stream(wait_tensor.device).wait_stream(self._param_offload_stream)
                self._profile_toc(
                    'distopt.residual.offload_wait_ms',
                    t_residual_wait,
                    wait_tensor,
                )
                self._profile_add_value('distopt.residual.offload_wait_calls', 1.0)
                residual_offload_pending_tensors.clear()

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

            if move_clip_grad_to_reducer and local_grad_slices:
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
                    for grad_slice in local_grad_slices:
                        grad_slice.mul_(clip_coeff)

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
            self._last_update_metric_norm = 0.0
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
            return

        residual_offload_pending_tensors: List[torch.Tensor] = []
        for buffer_idx, buffer in enumerate(self._buffers):
            grad_data = buffer.grad_data
            if grad_data is None or grad_data.numel() == 0:
                continue

            state = self._buffer_states[buffer_idx]
            residual_storage = self._residual_storage(state)
            residual_cpu = state.residual_cpu if self._offload_residual_to_cpu else None
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
                raise RuntimeError(
                    "Top-k hard-fail mode: missing optimizer-state mapping for at least one model parameter."
                )

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
                    if state.use_packed_mask_only:
                        next_mask_packed = state.next_mask_packed_u8
                        if next_mask_packed is None:
                            raise RuntimeError('Internal error: next packed mask buffer is missing.')
                        self._build_packed_mask_from_corrected(
                            grad_data=grad_data,
                            residual=residual_storage,
                            param_slices=param_slices,
                            density=next_density,
                            dense_mode=False,
                            out_mask_packed_u8=next_mask_packed,
                            group_rank=group_rank,
                            profile_tag='mask.next_from_dense',
                        )
                    else:
                        next_mask_u8 = state.next_mask_u8
                        if next_mask_u8 is None:
                            raise RuntimeError('Internal error: next mask_u8 buffer is missing.')
                        self._build_mask_from_corrected(
                            grad_data=grad_data,
                            residual=residual_storage,
                            param_slices=param_slices,
                            density=next_density,
                            dense_mode=False,
                            out_mask_u8=next_mask_u8,
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
                if state.next_mask_allreduce_handle is not None or state.next_mask_allreduce_event is not None:
                    self._wait_next_mask_allreduce(state)
                if state.has_next_mask:
                    if state.use_packed_mask_only:
                        current_packed = state.current_mask_packed_u8
                        next_packed = state.next_mask_packed_u8
                        if current_packed is None or next_packed is None:
                            raise RuntimeError('Internal error: packed mask buffers are missing.')
                        state.current_mask_packed_u8, state.next_mask_packed_u8 = next_packed, current_packed
                    else:
                        current_mask_u8 = state.current_mask_u8
                        next_mask_u8 = state.next_mask_u8
                        if current_mask_u8 is None or next_mask_u8 is None:
                            raise RuntimeError('Internal error: mask_u8 buffers are missing.')
                        current_mask_u8.copy_(next_mask_u8)
                    state.has_current_mask = True
                else:
                    if state.use_packed_mask_only:
                        current_packed = state.current_mask_packed_u8
                        if current_packed is None:
                            raise RuntimeError('Internal error: current packed mask buffer is missing.')
                        self._build_packed_mask_from_corrected(
                            grad_data=grad_data,
                            residual=residual_storage,
                            param_slices=param_slices,
                            density=current_density,
                            dense_mode=dense_mode,
                            out_mask_packed_u8=current_packed,
                            group_rank=group_rank,
                            profile_tag='mask.current_sync',
                        )
                        if self._group_size(group) > 1 and current_packed.numel() > 0:
                            packed_ranges = [
                                (int(start), int(end))
                                for (start, end, _packed_offset, _byte_len, _owner) in state.mask_pack_segments
                            ]
                            self._allreduce_packed_mask_staged_max_(
                                packed_u8=current_packed,
                                ranges=packed_ranges,
                                mask_numel=int(state.mask_numel),
                                group=group,
                                profile_key='mask.current_sync.allreduce_sync_ms',
                            )
                    else:
                        current_mask_u8 = state.current_mask_u8
                        if current_mask_u8 is None:
                            raise RuntimeError('Internal error: current mask_u8 buffer is missing.')
                        self._build_synced_mask_from_corrected(
                            grad_data=grad_data,
                            residual=residual_storage,
                            param_slices=param_slices,
                            density=current_density,
                            dense_mode=dense_mode,
                            out_mask_u8=current_mask_u8,
                            state=state,
                            group=group,
                            group_rank=group_rank,
                            profile_tag='mask.current_sync',
                        )
                    state.has_current_mask = True

            next_dense_mode = bool(force_all_reduce or (int(train_iter) + 1) < self._start_iter)
            if state.use_packed_mask_only:
                next_packed = state.next_mask_packed_u8
                if next_packed is None:
                    raise RuntimeError('Internal error: next packed mask buffer is missing.')
                self._build_packed_mask_from_corrected(
                    grad_data=grad_data,
                    residual=residual_storage,
                    param_slices=param_slices,
                    density=next_density,
                    dense_mode=next_dense_mode,
                    out_mask_packed_u8=next_packed,
                    group_rank=group_rank,
                    profile_tag='mask.next_async',
                )
            else:
                next_mask_u8 = state.next_mask_u8
                if next_mask_u8 is None:
                    raise RuntimeError('Internal error: next mask_u8 buffer is missing.')
                self._build_mask_from_corrected(
                    grad_data=grad_data,
                    residual=residual_storage,
                    param_slices=param_slices,
                    density=next_density,
                    dense_mode=next_dense_mode,
                    out_mask_u8=next_mask_u8,
                    group_rank=group_rank,
                    profile_tag='mask.next_async',
                )
            self._launch_next_mask_allreduce_async(state, group)

            current_mask: Optional[torch.Tensor] = None
            if state.use_packed_mask_only:
                total_numel += int(state.mask_numel)
            else:
                current_mask_u8 = state.current_mask_u8
                if current_mask_u8 is None:
                    raise RuntimeError('Internal error: current mask_u8 buffer is missing.')
                current_mask = current_mask_u8.bool()
                total_selected += int(current_mask.sum().item())
                total_numel += int(current_mask.numel())

            single_pass_residual_offload = bool(
                self._offload_residual_to_cpu
                and residual_cpu is not None
                and not use_fp8_quantized_payload
            )
            residual_offload_stage_events: List[Optional[torch.cuda.Event]] = [None, None]
            sparse_slice_plans: List[_SparseSlicePlan] = []
            payload_chunks: List[torch.Tensor] = []
            payload_offset = 0

            sparse_prefetch_work: List[Tuple[int, int]] = []
            prefetched_sparse_pos: Optional[int] = None
            prefetched_sparse: Optional[torch.Tensor] = None
            prefetched_sparse_event: Optional[torch.cuda.Event] = None
            if self._offload_residual_to_cpu and residual_cpu is not None:
                for param_slice in param_slices:
                    start = int(param_slice.start)
                    end = int(param_slice.end)
                    if end > start:
                        sparse_prefetch_work.append((start, end))
                if sparse_prefetch_work:
                    first_start, first_end = sparse_prefetch_work[0]
                    t_prefetch = self._profile_tic(grad_data)
                    prefetched_sparse, prefetched_sparse_event = self._prefetch_residual_slice_from_cpu(
                        residual_cpu=residual_cpu,
                        start=first_start,
                        end=first_end,
                        device=grad_data.device,
                        stage_slot=0,
                    )
                    self._profile_toc(
                        'residual.prefetch.cpu_to_gpu_ms',
                        t_prefetch,
                        prefetched_sparse,
                    )
                    self._profile_add_value('residual.prefetch_calls', 1.0)
                    prefetched_sparse_pos = 0
            sparse_work_pos = 0
            mask_work_pos = 0
            residual_offload_stage_pos = 0

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
                if self._offload_residual_to_cpu and residual_cpu is not None:
                    if sparse_work_pos >= len(sparse_prefetch_work):
                        raise RuntimeError('Residual sparse prefetch worklist out of bounds.')
                    expected_start, expected_end = sparse_prefetch_work[sparse_work_pos]
                    if expected_start != int(start) or expected_end != int(end):
                        raise RuntimeError(
                            'Residual sparse prefetch worklist mismatch: '
                            f'expected=[{expected_start}:{expected_end}] got=[{int(start)}:{int(end)}].'
                        )
                    if prefetched_sparse_pos != sparse_work_pos or prefetched_sparse is None:
                        t_prefetch = self._profile_tic(grad_data)
                        prefetched_sparse, prefetched_sparse_event = self._prefetch_residual_slice_from_cpu(
                            residual_cpu=residual_cpu,
                            start=int(start),
                            end=int(end),
                            device=grad_data.device,
                            stage_slot=(sparse_work_pos % 2),
                        )
                        self._profile_toc(
                            'residual.prefetch.cpu_to_gpu_ms',
                            t_prefetch,
                            prefetched_sparse,
                        )
                        self._profile_add_value('residual.prefetch_calls', 1.0)
                        prefetched_sparse_pos = sparse_work_pos
                    residual_slice = prefetched_sparse
                    if residual_slice is None:
                        raise RuntimeError('Residual sparse prefetch returned an empty staging tensor.')
                    if prefetched_sparse_event is not None:
                        t_prefetch_wait = self._profile_tic(grad_data)
                        torch.cuda.current_stream(grad_data.device).wait_event(prefetched_sparse_event)
                        self._profile_toc(
                            'residual.prefetch_wait_ms',
                            t_prefetch_wait,
                            grad_data,
                        )
                    next_sparse_work_pos = sparse_work_pos + 1
                    if next_sparse_work_pos < len(sparse_prefetch_work):
                        next_start, next_end = sparse_prefetch_work[next_sparse_work_pos]
                        t_prefetch = self._profile_tic(grad_data)
                        prefetched_sparse, prefetched_sparse_event = self._prefetch_residual_slice_from_cpu(
                            residual_cpu=residual_cpu,
                            start=next_start,
                            end=next_end,
                            device=grad_data.device,
                            stage_slot=(next_sparse_work_pos % 2),
                        )
                        self._profile_toc(
                            'residual.prefetch.cpu_to_gpu_ms',
                            t_prefetch,
                            prefetched_sparse,
                        )
                        self._profile_add_value('residual.prefetch_calls', 1.0)
                        prefetched_sparse_pos = next_sparse_work_pos
                    else:
                        prefetched_sparse_pos = None
                        prefetched_sparse = None
                        prefetched_sparse_event = None
                    sparse_work_pos += 1
                else:
                    residual_slice = residual_storage[start:end]

                mask_stage_slot = (mask_work_pos % 2)
                mask_work_pos += 1
                if state.use_packed_mask_only:
                    current_packed = state.current_mask_packed_u8
                    if current_packed is None:
                        raise RuntimeError('Internal error: current packed mask buffer is missing.')
                    mask_slice = self._unpack_mask_slice_from_packed(
                        packed_u8=current_packed,
                        start=int(start),
                        end=int(end),
                        device=grad_data.device,
                        stage_slot=mask_stage_slot,
                    ).bool()
                else:
                    if current_mask is None:
                        raise RuntimeError('Internal error: current mask tensor is missing.')
                    mask_slice = current_mask[start:end]
                selected_count = int(mask_slice.sum().item())
                if state.use_packed_mask_only:
                    total_selected += int(selected_count)

                t_corrected = self._profile_tic(grad_slice)
                corrected = self._fp32_scratch_view(
                    stage_slot=mask_stage_slot,
                    numel=int(end - start),
                    device=grad_data.device,
                )
                corrected.copy_(exp_avg_prev)
                corrected.mul_(beta1)
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

                if selected_count > 0:
                    t_select_idx = self._profile_tic(mask_slice)
                    selected_idx = torch.nonzero(mask_slice, as_tuple=False).flatten()
                    selected = corrected.index_select(0, selected_idx)
                    self._profile_toc('sparse.layer.select_index_ms', t_select_idx, mask_slice)
                    payload_chunks.append(selected)

                if not (self._offload_residual_to_cpu and residual_cpu is not None):
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
                elif single_pass_residual_offload:
                    offload_stage_slot = int(residual_offload_stage_pos % 2)
                    residual_offload_stage_pos += 1
                    prior_stage_event = residual_offload_stage_events[offload_stage_slot]
                    if prior_stage_event is not None:
                        torch.cuda.current_stream(grad_data.device).wait_event(prior_stage_event)
                        residual_offload_stage_events[offload_stage_slot] = None

                    t_residual = self._profile_tic(grad_slice)
                    residual_stage = self._residual_stage_view(
                        stage_slot=offload_stage_slot,
                        numel=int(end - start),
                        device=grad_data.device,
                    )
                    residual_stage.copy_(corrected)
                    if selected_count > 0:
                        residual_stage.masked_fill_(mask_slice, 0.0)
                        layer_profile['residual_update_ms'] += self._profile_toc(
                            'sparse.layer.residual_update_ms',
                            t_residual,
                            residual_stage,
                        )
                    else:
                        layer_profile['residual_update_ms'] += self._profile_toc(
                            'sparse.layer.residual_copy_ms',
                            t_residual,
                            residual_stage,
                        )

                    t_offload_queue = self._profile_tic(residual_stage)
                    queued_async = self._queue_residual_slice_offload_to_cpu(
                        residual_gpu=residual_stage,
                        residual_cpu=residual_cpu,
                        start=int(start),
                        end=int(end),
                    )
                    self._profile_toc(
                        'residual.offload_queue_ms',
                        t_offload_queue,
                        residual_stage,
                    )
                    self._profile_add_value('residual.offload_queue_calls', 1.0)
                    if queued_async:
                        self._profile_add_value('residual.offload_async_buffers', 1.0)
                        if self._param_offload_stream is not None:
                            done_event = torch.cuda.Event()
                            done_event.record(self._param_offload_stream)
                            residual_offload_stage_events[offload_stage_slot] = done_event
                        residual_offload_pending_tensors.append(residual_stage)

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

            if single_pass_residual_offload and quant_error_flat is not None:
                raise RuntimeError(
                    "Top-k hard-fail mode: single-pass residual offload does not support FP8 quantized payload residual correction."
                )

            sparse_update_prefetch_work: List[Tuple[int, int]] = []
            prefetched_update_pos: Optional[int] = None
            prefetched_update: Optional[torch.Tensor] = None
            prefetched_update_event: Optional[torch.cuda.Event] = None
            if (
                self._offload_residual_to_cpu
                and residual_cpu is not None
                and not single_pass_residual_offload
            ):
                for slice_plan in sparse_slice_plans:
                    start = int(slice_plan.start)
                    end = int(slice_plan.end)
                    if end > start:
                        sparse_update_prefetch_work.append((start, end))
                if sparse_update_prefetch_work:
                    first_start, first_end = sparse_update_prefetch_work[0]
                    t_prefetch = self._profile_tic(grad_data)
                    prefetched_update, prefetched_update_event = self._prefetch_residual_slice_from_cpu(
                        residual_cpu=residual_cpu,
                        start=first_start,
                        end=first_end,
                        device=grad_data.device,
                        stage_slot=0,
                    )
                    self._profile_toc(
                        'residual.prefetch.cpu_to_gpu_ms',
                        t_prefetch,
                        prefetched_update,
                    )
                    self._profile_add_value('residual.prefetch_calls', 1.0)
                    prefetched_update_pos = 0
            sparse_update_work_pos = 0
            mask_update_work_pos = 0

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
                if self._offload_residual_to_cpu and residual_cpu is not None:
                    if single_pass_residual_offload:
                        residual_before = None
                    else:
                        if sparse_update_work_pos >= len(sparse_update_prefetch_work):
                            raise RuntimeError('Residual update prefetch worklist out of bounds.')
                        expected_start, expected_end = sparse_update_prefetch_work[sparse_update_work_pos]
                        if expected_start != int(start) or expected_end != int(end):
                            raise RuntimeError(
                                'Residual update prefetch worklist mismatch: '
                                f'expected=[{expected_start}:{expected_end}] got=[{int(start)}:{int(end)}].'
                            )
                        if prefetched_update_pos != sparse_update_work_pos or prefetched_update is None:
                            t_prefetch = self._profile_tic(grad_data)
                            prefetched_update, prefetched_update_event = self._prefetch_residual_slice_from_cpu(
                                residual_cpu=residual_cpu,
                                start=int(start),
                                end=int(end),
                                device=grad_data.device,
                                stage_slot=(sparse_update_work_pos % 2),
                            )
                            self._profile_toc(
                                'residual.prefetch.cpu_to_gpu_ms',
                                t_prefetch,
                                prefetched_update,
                            )
                            self._profile_add_value('residual.prefetch_calls', 1.0)
                            prefetched_update_pos = sparse_update_work_pos
                        residual_before = prefetched_update
                        if residual_before is None:
                            raise RuntimeError('Residual update prefetch returned an empty staging tensor.')
                        if prefetched_update_event is not None:
                            t_prefetch_wait = self._profile_tic(grad_data)
                            torch.cuda.current_stream(grad_data.device).wait_event(prefetched_update_event)
                            self._profile_toc(
                                'residual.prefetch_wait_ms',
                                t_prefetch_wait,
                                grad_data,
                            )
                        next_update_work_pos = sparse_update_work_pos + 1
                        if next_update_work_pos < len(sparse_update_prefetch_work):
                            next_start, next_end = sparse_update_prefetch_work[next_update_work_pos]
                            t_prefetch = self._profile_tic(grad_data)
                            prefetched_update, prefetched_update_event = self._prefetch_residual_slice_from_cpu(
                                residual_cpu=residual_cpu,
                                start=next_start,
                                end=next_end,
                                device=grad_data.device,
                                stage_slot=(next_update_work_pos % 2),
                            )
                            self._profile_toc(
                                'residual.prefetch.cpu_to_gpu_ms',
                                t_prefetch,
                                prefetched_update,
                            )
                            self._profile_add_value('residual.prefetch_calls', 1.0)
                            prefetched_update_pos = next_update_work_pos
                        else:
                            prefetched_update_pos = None
                            prefetched_update = None
                            prefetched_update_event = None
                        sparse_update_work_pos += 1
                else:
                    residual_before = residual_storage[start:end]

                mask_stage_slot = (mask_update_work_pos % 2)
                mask_update_work_pos += 1
                if state.use_packed_mask_only:
                    current_packed = state.current_mask_packed_u8
                    if current_packed is None:
                        raise RuntimeError('Internal error: current packed mask buffer is missing.')
                    mask_slice = self._unpack_mask_slice_from_packed(
                        packed_u8=current_packed,
                        start=int(start),
                        end=int(end),
                        device=grad_data.device,
                        stage_slot=mask_stage_slot,
                    ).bool()
                else:
                    if current_mask is None:
                        raise RuntimeError('Internal error: current mask tensor is missing.')
                    mask_slice = current_mask[start:end]
                selected_count = int(mask_slice.sum().item())

                synced_momentum = torch.zeros_like(exp_avg_prev)
                residual_updated: Optional[torch.Tensor] = None
                if not single_pass_residual_offload:
                    t_residual = self._profile_tic(grad_slice)
                    residual_updated = exp_avg_prev * slice_plan.beta1
                    residual_updated.add_(grad_slice.to(torch.float32), alpha=1.0 - slice_plan.beta1)
                    residual_updated.add_(residual_before)
                    if selected_count > 0:
                        residual_updated.masked_fill_(mask_slice, 0.0)

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
                        if residual_updated is None:
                            raise RuntimeError(
                                'Top-k hard-fail mode: residual correction requires residual_updated tensor.'
                            )
                        residual_selected = residual_updated.index_select(0, selected_idx)
                        residual_selected.add_(quant_error_flat[payload_start:payload_end])
                        residual_updated.index_copy_(0, selected_idx, residual_selected)

                if not single_pass_residual_offload:
                    if residual_updated is None:
                        raise RuntimeError('Internal error: residual_updated tensor is missing.')
                    if self._offload_residual_to_cpu and residual_cpu is not None:
                        t_offload_queue = self._profile_tic(residual_updated)
                        queued_async = self._queue_residual_slice_offload_to_cpu(
                            residual_gpu=residual_updated,
                            residual_cpu=residual_cpu,
                            start=int(start),
                            end=int(end),
                        )
                        self._profile_toc(
                            'residual.offload_queue_ms',
                            t_offload_queue,
                            residual_updated,
                        )
                        self._profile_add_value('residual.offload_queue_calls', 1.0)
                        if queued_async:
                            residual_offload_pending_tensors.append(residual_updated)
                            self._profile_add_value('residual.offload_async_buffers', 1.0)
                    else:
                        residual_storage[start:end].copy_(residual_updated)
                    if selected_count > 0:
                        slice_plan.layer_profile['residual_update_ms'] += self._profile_toc(
                            'sparse.layer.residual_update_ms',
                            t_residual,
                            residual_updated,
                        )
                    else:
                        slice_plan.layer_profile['residual_update_ms'] += self._profile_toc(
                            'sparse.layer.residual_copy_ms',
                            t_residual,
                            residual_updated,
                        )

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

        if residual_offload_pending_tensors and self._param_offload_stream is not None:
            wait_tensor = residual_offload_pending_tensors[0]
            t_residual_wait = self._profile_tic(wait_tensor)
            torch.cuda.current_stream(wait_tensor.device).wait_stream(self._param_offload_stream)
            self._profile_toc(
                'residual.offload_wait_ms',
                t_residual_wait,
                wait_tensor,
            )
            self._profile_add_value('residual.offload_wait_calls', 1.0)
            residual_offload_pending_tensors.clear()

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

            pending_mask_work_pos = 0
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

                pending_state = self._buffer_states[pending.buffer_idx]
                if pending_state.use_packed_mask_only:
                    pending_packed = pending_state.current_mask_packed_u8
                    if pending_packed is None:
                        raise RuntimeError('Internal error: pending packed mask buffer is missing.')
                    mask_slice = self._unpack_mask_slice_from_packed(
                        packed_u8=pending_packed,
                        start=int(pending.start),
                        end=int(pending.end),
                        device=grad_slice.device,
                        stage_slot=(pending_mask_work_pos % 2),
                    ).to(torch.float32)
                else:
                    current_mask_u8 = pending_state.current_mask_u8
                    if current_mask_u8 is None:
                        raise RuntimeError('Internal error: pending mask_u8 buffer is missing.')
                    mask_slice = current_mask_u8[pending.start : pending.end].to(torch.float32)
                pending_mask_work_pos += 1

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
            residual_storage = state.residual_cpu if state.residual_cpu is not None else state.residual
            state_entry: Dict[str, Any] = {
                'residual': residual_storage.detach().cpu().clone(),
                'has_current_mask': bool(state.has_current_mask),
                'has_next_mask': bool(state.has_next_mask),
                'use_packed_mask_only': bool(state.use_packed_mask_only),
            }
            if state.current_mask_packed_u8 is not None:
                state_entry['current_mask_packed_u8'] = (
                    state.current_mask_packed_u8.detach().to(torch.uint8).cpu().clone()
                )
            if state.next_mask_packed_u8 is not None:
                state_entry['next_mask_packed_u8'] = (
                    state.next_mask_packed_u8.detach().to(torch.uint8).cpu().clone()
                )
            if state.current_mask_u8 is not None:
                state_entry['current_mask_u8'] = (
                    state.current_mask_u8.detach().to(torch.uint8).cpu().clone()
                )
            if state.next_mask_u8 is not None:
                state_entry['next_mask_u8'] = (
                    state.next_mask_u8.detach().to(torch.uint8).cpu().clone()
                )
            buffer_states.append(state_entry)
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

            residual_storage = state.residual_cpu if state.residual_cpu is not None else state.residual
            residual = loaded.get('residual', None)
            residual_ok = (
                isinstance(residual, torch.Tensor)
                and residual.shape == residual_storage.shape
                and residual.numel() == residual_storage.numel()
            )
            if residual_ok:
                residual_storage.copy_(
                    residual.to(device=residual_storage.device, dtype=residual_storage.dtype)
                )
            else:
                residual_storage.zero_()
            if state.residual_cpu is not None:
                state.residual = state.residual_cpu

            current_mask = loaded.get('current_mask_u8', None)
            next_mask = loaded.get('next_mask_u8', None)
            current_mask_packed_loaded_tensor = loaded.get('current_mask_packed_u8', None)
            next_mask_packed_loaded_tensor = loaded.get('next_mask_packed_u8', None)

            current_packed_loaded = False
            next_packed_loaded = False

            current_packed = state.current_mask_packed_u8
            if current_packed is not None:
                if (
                    isinstance(current_mask_packed_loaded_tensor, torch.Tensor)
                    and current_mask_packed_loaded_tensor.shape == current_packed.shape
                    and current_mask_packed_loaded_tensor.numel() == current_packed.numel()
                ):
                    current_packed.copy_(
                        current_mask_packed_loaded_tensor.to(
                            device=current_packed.device,
                            dtype=torch.uint8,
                        )
                    )
                    current_packed_loaded = True
                elif (
                    isinstance(current_mask, torch.Tensor)
                    and current_mask.numel() == int(state.mask_numel)
                ):
                    current_mask_dev = current_mask.to(device=current_packed.device, dtype=torch.uint8)
                    self._pack_mask_segments_to_packed(
                        current_mask_dev,
                        current_packed,
                        state.mask_pack_segments,
                    )
                    current_packed_loaded = True
                else:
                    current_packed.zero_()

            next_packed = state.next_mask_packed_u8
            if next_packed is not None:
                if (
                    isinstance(next_mask_packed_loaded_tensor, torch.Tensor)
                    and next_mask_packed_loaded_tensor.shape == next_packed.shape
                    and next_mask_packed_loaded_tensor.numel() == next_packed.numel()
                ):
                    next_packed.copy_(
                        next_mask_packed_loaded_tensor.to(
                            device=next_packed.device,
                            dtype=torch.uint8,
                        )
                    )
                    next_packed_loaded = True
                elif (
                    isinstance(next_mask, torch.Tensor)
                    and next_mask.numel() == int(state.mask_numel)
                ):
                    next_mask_dev = next_mask.to(device=next_packed.device, dtype=torch.uint8)
                    self._pack_mask_segments_to_packed(
                        next_mask_dev,
                        next_packed,
                        state.mask_pack_segments,
                    )
                    next_packed_loaded = True
                else:
                    next_packed.zero_()

            current_mask_loaded = False
            next_mask_loaded = False

            current_mask_u8 = state.current_mask_u8
            if current_mask_u8 is not None:
                current_mask_loaded = (
                    isinstance(current_mask, torch.Tensor)
                    and current_mask.shape == current_mask_u8.shape
                    and current_mask.numel() == current_mask_u8.numel()
                )
                if current_mask_loaded:
                    current_mask_u8.copy_(
                        current_mask.to(device=current_mask_u8.device, dtype=torch.uint8)
                    )
                elif current_packed_loaded and current_packed is not None:
                    self._unpack_packed_mask_to_local_u8(
                        current_packed,
                        current_mask_u8,
                        state.mask_pack_segments,
                    )
                    current_mask_loaded = True
                else:
                    current_mask_u8.zero_()
            elif state.use_packed_mask_only:
                current_mask_loaded = current_packed_loaded

            next_mask_u8 = state.next_mask_u8
            if next_mask_u8 is not None:
                next_mask_loaded = (
                    isinstance(next_mask, torch.Tensor)
                    and next_mask.shape == next_mask_u8.shape
                    and next_mask.numel() == next_mask_u8.numel()
                )
                if next_mask_loaded:
                    next_mask_u8.copy_(
                        next_mask.to(device=next_mask_u8.device, dtype=torch.uint8)
                    )
                elif next_packed_loaded and next_packed is not None:
                    self._unpack_packed_mask_to_local_u8(
                        next_packed,
                        next_mask_u8,
                        state.mask_pack_segments,
                    )
                    next_mask_loaded = True
                else:
                    next_mask_u8.zero_()
            elif state.use_packed_mask_only:
                next_mask_loaded = next_packed_loaded

            # Backward-compatibility: if explicit mask-valid flags are absent in older
            # checkpoints, infer validity from whether masks were loaded successfully.
            current_mask_flag = loaded.get('has_current_mask', current_mask_loaded)
            next_mask_flag = loaded.get('has_next_mask', next_mask_loaded)

            state.has_current_mask = bool(current_mask_flag) and bool(current_mask_loaded)
            state.next_mask_allreduce_handle = None
            state.next_mask_allreduce_event = None
            state.next_mask_allreduce_uses_packed = False
            state.next_mask_allreduce_unpacked_u8 = None
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
