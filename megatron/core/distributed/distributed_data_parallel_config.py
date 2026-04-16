from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DistributedDataParallelConfig:
    """Configuration for DistributedDataParallel."""

    grad_reduce_in_fp32: bool = False
    """If true, reduce grads in fp32."""

    overlap_grad_reduce: bool = False
    """If true, overlap grad all-reduce / reduce-scatter with backward compute."""

    overlap_param_gather: bool = False
    """If true, overlap param all-gather with forward compute."""

    align_param_gather: bool = False
    """If true, all PP stages will launch param all-gathers simultaneously. Otherwise, each
    PP stage will independently launch as needed.
    """

    use_distributed_optimizer: bool = False
    """If true, issue reduce-scatter collectives to aggregate gradients and clean up
       originally allocated model parameters, otherwise issue all-reduce collectives.
    """

    num_distributed_optimizer_instances: int = 1
    """Sets the factor by which the DP domain is sharded to have the partial DistOpt
       enabled. Defaults to 1, which means DistOpt is across entire DP domain.
    """

    check_for_nan_in_grad: bool = False
    """
    If true, check for NaNs and Infs in gradients _before_ communication collective.
    Invoked by `start_grad_sync` such as in the Megatron-LM DDP training API.
    """

    check_for_large_grads: bool = False
    """If true, check for unexpectedly large gradients _before_ communication collective."""

    bucket_size: Optional[int] = None
    """Maximum number of parameters in each bucket. If unspecified, MCore uses a default
       value of max(40000000, 1000000 * dp_size) parameters (larger DP sizes need larger
       buckets to ensure collectives do not become latency-bound)."""

    pad_buckets_for_high_nccl_busbw: bool = False
    """If true, make sure the bucket size is divisible by a large power of 2 (2^16) to
       ensure NCCL collectives have high bus bandwidth at large DP counts, since NCCL
       message size (which for ring algorithms is bucket_size / dp_size) apparently needs
       to be divisible by a power of 2 for high busbw."""

    reduce_scatter_with_fp32_accumulation: bool = False
    """If true, use a reduce-scatter implementation which sends lower-precision values
       over the wire (using an all-to-all to keep total communication overhead in line
       with the standard ring implementation) but performs accumulation locally in FP32."""

    average_in_collective: bool = False
    """If true, compute average in collective directly, as opposed to dividing by the
       dp_size first and then computing sum in the collective."""

    fp8_param_gather: bool = False
    """If true, keep the compute param in fp8 (do not use any other intermediate dtype) and
       perform the param all-gather in fp8."""

    reuse_grad_buf_for_mxfp8_param_ag: bool = False
    """If true, reuse the grad buffer for param AG when using mxfp8 recipe. Should be 
       set to True only when fp8_recipe is mxfp8 and fp8_param_gather is True."""

    use_megatron_fsdp: bool = False
    """If true, use the FSDP code path for DDP."""

    use_custom_fsdp: bool = False
    """
    NOTE: The flag `use_custom_fsdp` is deprecated and will be removed in future versions.
    Please use `use_megatron_fsdp` instead, as all functionality will be migrated there.
    Future updates will drop support for `use_custom_fsdp` to avoid confusion.
    """

    data_parallel_sharding_strategy: str = 'no_shard'
    """Sharding strategy for FSDP. Valid values are 'no_shard', 'optim',
      'optim_grads', 'optim_grads_params'."""

    gradient_reduce_div_fusion: bool = True
    """If true, perform gradient reduce and division fusion."""

    suggested_communication_unit_size: int = None
    """Specifies the number of elements to communicate at once during
      FSDP (Fully Sharded Data Parallel) operations. 
      This flag also affects FSDP all-gather prefetch behavior. Setting a larger
      value increases the communication buffer size, while a smaller value
      disables prefetching and may degrade performance. Adjust this value
      based on your system's memory and performance requirements."""

    keep_fp8_transpose_cache: bool = False
    """If true, keep the fp8 transpose cache when using Megatron FSDP."""

    nccl_ub: bool = False
    """If true, allocate and register NCCL userbuffer for param and grad buffer.
      This flag enables SM efficient nccl algorithm that could improve the performance
      of FSDP and DP with comm_overlap. This flag will be much more effective when used
      together with sharp. 
      The follwoing will be the expected number of SM usage for various cases.
      (Note that this is just a reference number and the number of SM usage could vary 
      on message size, communication domain size and nccl version.)
      | Communication domain | use_sharp | SM usage of "AG/RS" |
      |----------------------|-----------|---------------------|
      | NVL                  | N/A       | 4 / 5               |
      | NVL+IB               | False     | 16 / 16             |
      | NVL+IB               | True      | 6 / 6               |
      | IB                   | False     | 1 / 4               |
      | IB                   | True      | 1 / 1               |
    """

    fsdp_double_buffer: bool = False
    """If true, use persistently allocated double buffers for the 
      temporary memory needed in the Megatron FSDP communications.
      This option will cause additional memory overhead, however, it is necessary for
      to register user buffer (nccl_ub=True) for the Megatron FSDP. 
      This option will be automatically set to True when nccl_ub=True.
    """

    fsdp_db_use_persist_buf_on_alloc_fail: bool = False
    """Whether to fall back to persistent buffer when a bucket does not
       fit FSDP double buffer size. If true, FSDP will use the persistently 
       allocated buffer for the bucket that does not fit, it will enable NCCL 
       user buffer with the cost of more memory usage. If false, FSDP will use
       Dynamic memory allocator, NCCL user buffer won't not enabled, which 
       usually leads to low performance.
    """

    fsdp_all_gather_in_start_param_sync: bool = True
    """
    If True, use all-gather during the initial Megatron-FSDP parameter
    synchronization step. This can increase overlap between the first
    parameter all-gather and computation, helping to better hide the
    initial communication cost.
    """

    outer_dp_sharding_strategy: str = 'no_shard'
    """
    Sharding strategy for outer data parallel group in Hybrid Sharded Data Parallel (HSDP) mode.
    Valid values are 'no_shard', 'optim'. This option is only effective when Hybrid FSDP is enabled.
    """

    disable_symmetric_registration: bool = False
    """If true, disable symmetric (window) registration for NCCL userbuffer registration.
      This option will force to use conventional (local) userbuffer registration 
      when nccl_ub is set.
    """

    fsdp_manual_registration: bool = False
    """If true, manually register the FSDP communication buffers to NCCL user buffer.
      This option is only effective when use_megatron_fsdp and nccl_ub is set.
      For symmetric registration with large models, the registration itself can take 
      a significant amount of time. This option minimizes the number of registration calls
      to minimize the registration time.
    """

    delay_wgrad_compute: bool = False
    """Delay the weight gradient computation to improve batch-level communication overlapping"""

    megatron_fsdp_main_params_dtype: Optional[torch.dtype] = torch.float32
    """Data type for the main weight buffer utilized for distributed optimization
      and quantization with Megatron-FSDP. If set to None, the model compute weight
      buffer will take the role of the main weights, or when no sharding is applied,
      the native model weights become the main weights. Defaults to torch.float32.
    """

    megatron_fsdp_main_grads_dtype: Optional[torch.dtype] = None
    """Data type for the main gradient buffer utilized for distributed optimization with
      Megatron-FSDP. If set to None, main gradients will match the dtype of the model
      compute parameters specified by the user model. Defaults to None.
    """

    megatron_fsdp_grad_comm_dtype: Optional[torch.dtype] = None
    """Data type for gradient gather / scatter communications. Can be utilized to reduce
      communication latency, but adds overhead for type-casting and copy operations.
      If using NCCL UBR v2.27+, gradient reduction may be performed in high-precision
      depending on the network domain (NVLink or IB), and can enable mixed-precision
      communication and accumulation, e.g. setting grad_comm_dtype to `BF16` can support
      `FP32` reduction even though we have `BF16` input and output communication buffers.
      If set to None, the `main_grads_dtype` is used. If using HSDP (either DP-Replicate
      or DP-Outer in `outer_dp_sharding_strategy`), `no_shard`, `optim`, or a
      `FixedPoolAllocator` (`fsdp_double_buffer`), allocating `dtype`-custom gradient
      communication buffers (per FSDP group) adds memory overhead. Defaults to None.
      No additional memory is allocated when `grad_comm_dtype == main_grads_dtype`.
    """

    use_topk_adams_reducer: bool = False
    """If true, replace default grad all-reduce path with Top-K AdamS reducer."""

    topk_adams_density: float = 0.1
    """Target top-k density used by Top-K AdamS reducer."""

    topk_adams_start_iter: int = 0
    """Iteration to start sparse top-k communication."""

    topk_adams_density_start: float = 1.0
    """Initial top-k density used before warmup reaches target density."""

    topk_adams_density_warmup_steps: int = 0
    """Number of steps to geometrically warm up density from start to target."""

    topk_adams_density_cooldown_steps: int = 0
    """Optional cooldown steps to geometrically increase density from target to 1.0."""

    topk_adams_density_cooldown_start_step: int = -1
    """Step to start cooldown; negative means use topk_adams_start_iter."""

    topk_adams_use_exclude_from_topk: bool = True
    """If true, apply name-based exclusion (e.g., LayerNorm/RMSNorm) from top-k sparsification."""

    move_clip_grad_to_reducer: bool = False
    """If true, apply global grad clipping inside top-k reducer on raw synced gradients."""

    use_fp8_topk_quant: bool = False
    """If true, use FP8 quantized sparse payloads in Top-K AdamS reducer."""

    topk_adams_full_param_cpu_offload: bool = False
    """If true, offload Top-K reducer full FP32 param replica to CPU between steps."""

    topk_adams_residual_cpu_offload: bool = False
    """If true, offload Top-K reducer residual (error-feedback) buffers to CPU with layer-staged GPU buffers."""

    use_topk_mask_overlap_tracker: bool = False
    """If true, record per-parameter top-k mask overlap on AdamS momentum."""

    topk_mask_overlap_density: float = 0.01
    """Top-k density (<1.0) or absolute k (>=1) used by the overlap tracker."""

    topk_mask_overlap_start_iter: int = 0
    """Iteration to start recording top-k mask overlap metrics."""

    topk_mask_overlap_reference_mode: str = "first"
    """Reference mask mode for the overlap tracker: 'first' or 'prev'."""

    topk_mask_overlap_output_dir: str = "/tmp/topk_mask_overlap"
    """Directory where the overlap tracker writes CSV outputs."""

    topk_mask_overlap_max_steps: int = 0
    """Maximum number of post-start steps to record; 0 means unlimited."""

    def __post_init__(self):
        import os

        """Check the validity of the config."""
        if self.reuse_grad_buf_for_mxfp8_param_ag:
            assert self.fp8_param_gather, "Reuse grad buffer only when keeping params in MXFP8."

        if self.use_topk_adams_reducer:
            if self.overlap_grad_reduce:
                raise ValueError(
                    "Top-K AdamS reducer does not support overlap_grad_reduce. "
                    "Disable --overlap-grad-reduce."
                )
            if not (0.0 < self.topk_adams_density <= 1.0):
                raise ValueError(
                    f"topk_adams_density must be in (0, 1], got {self.topk_adams_density}."
                )
            if not (0.0 < self.topk_adams_density_start <= 1.0):
                raise ValueError(
                    "topk_adams_density_start must be in (0, 1], "
                    f"got {self.topk_adams_density_start}."
                )
            if self.topk_adams_density_warmup_steps < 0:
                raise ValueError(
                    "topk_adams_density_warmup_steps must be >= 0, "
                    f"got {self.topk_adams_density_warmup_steps}."
                )
            if self.topk_adams_density_cooldown_steps < 0:
                raise ValueError(
                    "topk_adams_density_cooldown_steps must be >= 0, "
                    f"got {self.topk_adams_density_cooldown_steps}."
                )
            if self.topk_adams_density_cooldown_start_step < -1:
                raise ValueError(
                    "topk_adams_density_cooldown_start_step must be >= -1, "
                    f"got {self.topk_adams_density_cooldown_start_step}."
                )
            if self.topk_adams_start_iter < 0:
                raise ValueError(
                    f"topk_adams_start_iter must be >= 0, got {self.topk_adams_start_iter}."
                )
            if self.topk_adams_full_param_cpu_offload and self.overlap_param_gather:
                raise ValueError(
                    "topk_adams_full_param_cpu_offload requires overlap_param_gather=False "
                    "because sparse param sync replaces dense param all-gather."
                )
            if self.topk_adams_full_param_cpu_offload and (not self.use_distributed_optimizer):
                raise ValueError(
                    "topk_adams_full_param_cpu_offload requires use_distributed_optimizer=True."
                )
        elif self.move_clip_grad_to_reducer:
            raise ValueError(
                "move_clip_grad_to_reducer requires use_topk_adams_reducer."
            )
        elif self.topk_adams_full_param_cpu_offload:
            raise ValueError(
                "topk_adams_full_param_cpu_offload requires use_topk_adams_reducer."
            )
        elif self.topk_adams_residual_cpu_offload:
            raise ValueError(
                "topk_adams_residual_cpu_offload requires use_topk_adams_reducer."
            )

        if self.use_topk_mask_overlap_tracker:
            if self.use_topk_adams_reducer:
                raise ValueError(
                    "Top-k mask overlap tracker does not support use_topk_adams_reducer."
                )
            if self.use_distributed_optimizer:
                raise ValueError(
                    "Top-k mask overlap tracker does not support distributed optimizer."
                )
            if self.use_megatron_fsdp or self.use_custom_fsdp:
                raise ValueError(
                    "Top-k mask overlap tracker is supported only on the standard MCore DDP path."
                )
            if self.topk_mask_overlap_density <= 0.0:
                raise ValueError(
                    "topk_mask_overlap_density must be > 0, "
                    f"got {self.topk_mask_overlap_density}."
                )
            if self.topk_mask_overlap_start_iter < 0:
                raise ValueError(
                    "topk_mask_overlap_start_iter must be >= 0, "
                    f"got {self.topk_mask_overlap_start_iter}."
                )
            if self.topk_mask_overlap_max_steps < 0:
                raise ValueError(
                    "topk_mask_overlap_max_steps must be >= 0, "
                    f"got {self.topk_mask_overlap_max_steps}."
                )
            reference_mode = str(self.topk_mask_overlap_reference_mode).strip().lower()
            if reference_mode not in ("first", "prev"):
                raise ValueError(
                    "topk_mask_overlap_reference_mode must be either 'first' or 'prev', "
                    f"got {self.topk_mask_overlap_reference_mode!r}."
                )

        if self.nccl_ub:
            if 'expandable_segments:True' in os.getenv('PYTORCH_CUDA_ALLOC_CONF', '').split(','):
                raise ValueError(
                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is currently not supported "
                    "with nccl_ub due to compatibility issue with torch.cuda.MemPool API."
                )
