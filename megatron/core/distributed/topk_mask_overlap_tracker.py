import csv
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TrackedMomentumSlice:
    key: Tuple[int, int, int]
    buffer_idx: int
    start: int
    end: int
    param: torch.nn.Parameter
    name: str
    safe_name: str
    output_stem: str


class TopKMaskOverlapTracker:
    """Track top-k mask overlap on unsparsified AdamS momentum.

    The tracker derives momentum from synchronized dense DDP gradients using
    `m_t = beta1 * m_{t-1} + (1 - beta1) * g_t`, then records top-k Jaccard and
    energy-overlap metrics for each parameter tensor.
    """

    def __init__(
        self,
        buffers: List[_ParamAndGradBuffer],
        param_to_name: Dict[torch.nn.Parameter, str],
        ddp_config: DistributedDataParallelConfig,
    ) -> None:
        self._buffers = list(buffers)
        self._param_to_name = dict(param_to_name)
        self._output_root = os.path.abspath(str(ddp_config.topk_mask_overlap_output_dir).strip())
        self._reference_mode = str(ddp_config.topk_mask_overlap_reference_mode).strip().lower()
        self._start_iter = max(0, int(ddp_config.topk_mask_overlap_start_iter))
        self._max_steps = max(0, int(ddp_config.topk_mask_overlap_max_steps))
        self._topk_density = float(ddp_config.topk_mask_overlap_density)
        self._default_beta1 = 0.9
        self._eps = 1e-12

        self._current_train_iter = 0
        self._processed_steps = 0
        self._max_steps_log_emitted = False

        self._reference_indices: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self._reference_steps: Dict[Tuple[int, int, int], int] = {}
        self._beta1_by_key: Dict[Tuple[int, int, int], float] = {}
        self._momentum_buffers: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self._jaccard_headers_written: Set[str] = set()
        self._energy_headers_written: Set[str] = set()
        self._warned_missing_optimizer_mapping = False

        self._global_rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._global_rank = int(torch.distributed.get_rank())

        self._tracked_slices = self._build_tracked_slices()

        if self._global_rank == 0:
            logger.info(
                "TopKMaskOverlapTracker enabled with output_root=%s, reference_mode=%s, "
                "topk_density=%s, start_iter=%s, max_steps=%s (tracking AdamS momentum)",
                self._output_root,
                self._reference_mode,
                self._topk_density,
                self._start_iter,
                self._max_steps,
            )

    @staticmethod
    def _iter_optimizer_wrappers(optimizer: Any) -> List[Any]:
        wrappers = getattr(optimizer, 'chained_optimizers', None)
        if wrappers is None:
            return [optimizer]
        return list(wrappers)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return str(name).replace('/', '_').replace('.', '_')

    @staticmethod
    def _group_rank(group: torch.distributed.ProcessGroup) -> int:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return 0
        try:
            return int(torch.distributed.get_rank(group=group))
        except Exception:
            return 0

    def _build_tracked_slices(self) -> List[List[_TrackedMomentumSlice]]:
        tracked_by_buffer: List[List[_TrackedMomentumSlice]] = []
        safe_name_counts: Dict[str, int] = {}
        raw_entries: List[List[Tuple[int, int, torch.nn.Parameter, str, str]]] = []

        for buffer_idx, buffer in enumerate(self._buffers):
            entries: List[Tuple[int, int, torch.nn.Parameter, str, str]] = []
            for param in buffer.params:
                start, end, _ = buffer.param_index_map[param]
                name = self._param_to_name.get(param, f'buffer_{buffer_idx}_offset_{int(start)}')
                safe_name = self._sanitize_name(name)
                safe_name_counts[safe_name] = safe_name_counts.get(safe_name, 0) + 1
                entries.append((int(start), int(end), param, name, safe_name))
            entries.sort(key=lambda item: item[0])
            raw_entries.append(entries)

        for buffer_idx, entries in enumerate(raw_entries):
            tracked_entries: List[_TrackedMomentumSlice] = []
            for start, end, param, name, safe_name in entries:
                output_stem = safe_name
                if safe_name_counts.get(safe_name, 0) > 1:
                    output_stem = f'{safe_name}__buffer_{buffer_idx}_offset_{start}'
                tracked_entries.append(
                    _TrackedMomentumSlice(
                        key=(buffer_idx, start, end),
                        buffer_idx=buffer_idx,
                        start=start,
                        end=end,
                        param=param,
                        name=name,
                        safe_name=safe_name,
                        output_stem=output_stem,
                    )
                )
            tracked_by_buffer.append(tracked_entries)

        return tracked_by_buffer

    def set_optimizer(self, optimizer: Any) -> None:
        beta1_by_optim_param: Dict[torch.Tensor, float] = {}

        for wrapper in self._iter_optimizer_wrappers(optimizer):
            if getattr(wrapper, 'is_stub_optimizer', False):
                continue
            inner_optimizer = getattr(wrapper, 'optimizer', wrapper)
            param_groups = getattr(inner_optimizer, 'param_groups', None)
            if param_groups is None:
                continue

            for group in param_groups:
                betas = group.get('betas', (self._default_beta1, 0.999))
                beta1 = float(betas[0])
                for param in group.get('params', []):
                    beta1_by_optim_param[param] = beta1

        self._beta1_by_key.clear()
        missing = 0
        total = 0
        for tracked_slices in self._tracked_slices:
            for tracked_slice in tracked_slices:
                total += 1
                optim_param = getattr(tracked_slice.param, 'main_param', tracked_slice.param)
                beta1 = beta1_by_optim_param.get(optim_param, None)
                if beta1 is None:
                    beta1 = beta1_by_optim_param.get(tracked_slice.param, None)
                if beta1 is None:
                    missing += 1
                    beta1 = self._default_beta1
                self._beta1_by_key[tracked_slice.key] = beta1

        if missing > 0 and not self._warned_missing_optimizer_mapping:
            logger.warning(
                'TopKMaskOverlapTracker optimizer mapping incomplete: mapped %d / %d parameters; '
                'defaulting beta1 to %.3f for the rest.',
                total - missing,
                total,
                self._default_beta1,
            )
            self._warned_missing_optimizer_mapping = True

    def prepare_pre_forward(self, train_iter: int) -> None:
        self._current_train_iter = int(train_iter)

    def _resolve_k(self, numel: int) -> int:
        if numel <= 0:
            return 0
        if self._topk_density <= 0.0:
            return 1
        if self._topk_density < 1.0:
            k = int(math.ceil(float(numel) * self._topk_density))
        else:
            k = int(self._topk_density)
        return max(1, min(k, int(numel)))

    def _get_csv_paths(
        self, dtype: torch.dtype, tracked_slice: _TrackedMomentumSlice
    ) -> Tuple[str, str]:
        dtype_label = str(dtype).replace(' ', '')
        base_dir = os.path.join(
            self._output_root,
            f'rank_{self._global_rank:05d}',
            dtype_label,
        )
        jaccard_dir = os.path.join(base_dir, 'jaccard')
        energy_dir = os.path.join(base_dir, 'energy')
        os.makedirs(jaccard_dir, exist_ok=True)
        os.makedirs(energy_dir, exist_ok=True)
        file_name = f'{tracked_slice.output_stem}.csv'
        return (
            os.path.join(jaccard_dir, file_name),
            os.path.join(energy_dir, file_name),
        )

    def _append_jaccard_csv(
        self,
        csv_path: str,
        train_iter: int,
        ref_step: int,
        k_ref: int,
        k_cur: int,
        inter_cnt: int,
        union_cnt: int,
        jaccard: float,
    ) -> None:
        write_header = False
        if csv_path not in self._jaccard_headers_written:
            file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
            write_header = not file_exists
            self._jaccard_headers_written.add(csv_path)

        with open(csv_path, 'a', newline='') as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(
                    [
                        'step',
                        'ref_step',
                        'reference_mode',
                        'k_ref',
                        'k_cur',
                        'intersection',
                        'union',
                        'jaccard',
                    ]
                )
            writer.writerow(
                [
                    int(train_iter),
                    int(ref_step),
                    self._reference_mode,
                    int(k_ref),
                    int(k_cur),
                    int(inter_cnt),
                    int(union_cnt),
                    float(jaccard),
                ]
            )

    def _append_energy_csv(
        self,
        csv_path: str,
        train_iter: int,
        ref_step: int,
        k_ref: int,
        k_cur: int,
        inter_cnt: int,
        total_energy: float,
        intersection_energy: float,
        energy_overlap: float,
    ) -> None:
        write_header = False
        if csv_path not in self._energy_headers_written:
            file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
            write_header = not file_exists
            self._energy_headers_written.add(csv_path)

        with open(csv_path, 'a', newline='') as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(
                    [
                        'step',
                        'ref_step',
                        'reference_mode',
                        'k_ref',
                        'k_cur',
                        'intersection',
                        'total_energy',
                        'intersection_energy',
                        'energy_overlap',
                    ]
                )
            writer.writerow(
                [
                    int(train_iter),
                    int(ref_step),
                    self._reference_mode,
                    int(k_ref),
                    int(k_cur),
                    int(inter_cnt),
                    float(total_energy),
                    float(intersection_energy),
                    float(energy_overlap),
                ]
            )

    @staticmethod
    def _intersection_mask(
        current_indices: torch.Tensor, reference_indices: torch.Tensor
    ) -> torch.Tensor:
        try:
            return torch.isin(current_indices, reference_indices)
        except Exception:
            if reference_indices.numel() == 0:
                return torch.zeros_like(current_indices, dtype=torch.bool)
            return (current_indices.view(-1, 1) == reference_indices.view(1, -1)).any(dim=1)

    def _get_or_init_momentum_buffer(
        self, tracked_slice: _TrackedMomentumSlice, device: torch.device, numel: int
    ) -> torch.Tensor:
        momentum = self._momentum_buffers.get(tracked_slice.key, None)
        needs_init = (
            momentum is None
            or int(momentum.numel()) != int(numel)
            or momentum.device != device
            or momentum.dtype != torch.float32
        )
        if needs_init:
            momentum = torch.zeros(numel, dtype=torch.float32, device=device)
            self._momentum_buffers[tracked_slice.key] = momentum
        return momentum

    def _record_slice_overlap(
        self,
        train_iter: int,
        dtype: torch.dtype,
        tracked_slice: _TrackedMomentumSlice,
        grad_slice: torch.Tensor,
    ) -> None:
        layer_numel = int(grad_slice.numel())
        if layer_numel <= 0:
            return

        k_cur = self._resolve_k(layer_numel)
        if k_cur <= 0:
            return

        grad_fp32 = grad_slice if grad_slice.dtype == torch.float32 else grad_slice.to(torch.float32)
        momentum = self._get_or_init_momentum_buffer(tracked_slice, grad_fp32.device, layer_numel)
        beta1 = float(self._beta1_by_key.get(tracked_slice.key, self._default_beta1))
        momentum.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)

        cur_idx = torch.topk(momentum.abs(), k_cur, largest=True, sorted=False).indices
        cur_vals = momentum.index_select(0, cur_idx)

        ref_idx = self._reference_indices.get(tracked_slice.key, None)
        ref_step = self._reference_steps.get(tracked_slice.key, int(train_iter))
        if ref_idx is None:
            ref_idx = cur_idx.detach().clone()
            ref_step = int(train_iter)
            self._reference_indices[tracked_slice.key] = ref_idx
            self._reference_steps[tracked_slice.key] = ref_step

        inter_mask = self._intersection_mask(cur_idx, ref_idx)
        inter_cnt = int(inter_mask.sum().item())
        k_ref = int(ref_idx.numel())
        union_cnt = int(k_ref + k_cur - inter_cnt)
        jaccard = float(inter_cnt / union_cnt) if union_cnt > 0 else 1.0

        total_energy = float(torch.sum(momentum * momentum).item())
        if inter_cnt > 0:
            inter_vals = cur_vals[inter_mask]
            intersection_energy = float(torch.sum(inter_vals * inter_vals).item())
        else:
            intersection_energy = 0.0
        energy_overlap = (
            float(intersection_energy / (total_energy + self._eps))
            if total_energy > 0.0
            else 0.0
        )

        jaccard_csv_path, energy_csv_path = self._get_csv_paths(dtype, tracked_slice)
        self._append_jaccard_csv(
            csv_path=jaccard_csv_path,
            train_iter=train_iter,
            ref_step=ref_step,
            k_ref=k_ref,
            k_cur=k_cur,
            inter_cnt=inter_cnt,
            union_cnt=union_cnt,
            jaccard=jaccard,
        )
        self._append_energy_csv(
            csv_path=energy_csv_path,
            train_iter=train_iter,
            ref_step=ref_step,
            k_ref=k_ref,
            k_cur=k_cur,
            inter_cnt=inter_cnt,
            total_energy=total_energy,
            intersection_energy=intersection_energy,
            energy_overlap=energy_overlap,
        )

        if self._reference_mode == 'prev':
            self._reference_indices[tracked_slice.key] = cur_idx.detach().clone()
            self._reference_steps[tracked_slice.key] = int(train_iter)

    def maybe_record(self, train_iter: Optional[int] = None) -> None:
        current_iter = int(self._current_train_iter if train_iter is None else train_iter)

        if current_iter < self._start_iter:
            if current_iter % 100 == 0 and self._global_rank == 0:
                logger.info('TopKMaskOverlapTracker warm-up step %d', current_iter)
            return

        if self._max_steps > 0 and self._processed_steps >= self._max_steps:
            if self._global_rank == 0 and not self._max_steps_log_emitted:
                logger.info(
                    'TopKMaskOverlapTracker reached max tracked steps (%d); '
                    'continuing training without additional overlap logging.',
                    self._max_steps,
                )
                self._max_steps_log_emitted = True
            return

        self._processed_steps += 1

        for buffer_idx, buffer in enumerate(self._buffers):
            if self._group_rank(buffer.data_parallel_group) != 0:
                continue
            grad_flat = buffer.grad_data.view(-1)
            dtype = grad_flat.dtype
            for tracked_slice in self._tracked_slices[buffer_idx]:
                grad_slice = grad_flat[tracked_slice.start : tracked_slice.end]
                self._record_slice_overlap(
                    train_iter=current_iter,
                    dtype=dtype,
                    tracked_slice=tracked_slice,
                    grad_slice=grad_slice,
                )
