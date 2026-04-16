# SCAPE

`SCAPE` is the SC26 AD/AE artifact repo for SCAPE. It is a Megatron-LM fork with sparse AdamS communication, density scheduling, CPU offload variants, and checked-in Slurm launchers for the current VISTA experiments.

## Artifact Scope

This checkout still contains the usual Megatron-LM codebase, but the artifact is centered on:

- `pretrain_gpt.py`: the training entrypoint used by the checked-in launchers
- `slurm_scripts/vista/`: baseline, SCAPE, scaling, and memory-study launchers
- `tools/`: current data-prep, checkpoint-conversion, evaluation, and log-summary helpers
- `megatron/`: the SCAPE and distributed-training implementation

The rest of the repository, including `docs/`, `examples/`, `scripts/`, and `tests/`, is mostly upstream/reference material from the fork.

## Current Repository Layout

Top-level paths that matter most for the artifact:

- `pretrain_gpt.py`
- `megatron/`
- `slurm_scripts/`
- `tools/`
- `README.md`

Current `slurm_scripts/vista/` layout:

```text
slurm_scripts/vista/
  baseline/
    AdamS/
      gpt-345M/
      llama-H2O-500M/
    AdamW/
      gpt-345M/
      llama-H2O-500M/
  scape/
    gpt-345M/
      d_0.01/
      d_0.1/
    llama-500M/
      d_0.01/
      d_0.1/
  scaling_efficiency/
    AdamS/
      500M_micro_bs_8/
        comp_comm_breakdown/
        dist-optm-comp_comm_breakdown/
      1.8B_micro_bs_8/
        comp_comm_breakdown/
        dist-optm-comp_comm_breakdown/
    SCAPE/
      d_0.01/
        no_distributed_optimizer/
        use_distributed_optimizer/
      d_0.1/
        no_distributed_optimizer/
        use_distributed_optimizer/
    SCAPE_cpu_offload/
      d_0.01/
        use_distributed_optimizer/
      d_0.1/
        use_distributed_optimizer/
  memory_usage_micro_bs_8/
    AdamS_baseline/
      500M/
      1.8B/
    AdamS_baseline_dist-optm/
      500M/
      1.8B/
    SCAPE_d_0.1_no_residual_offload/
      500M/
      1.8B/
    SCAPE_d_0.1_residual_offload/
      500M/
      1.8B/
    SCAPE_d_0.1_dist-optm_no_residual_model_offload/
      500M/
      1.8B/
    SCAPE_d_0.1_dist-optm_residual_model_offload/
      500M/
      1.8B/
```

Important layout notes:

- The current tree uses `SCAPE_cpu_offload`, not `SCAPE_cpu_offloadv2`.
- The current tree includes `slurm_scripts/vista/scape/`; it does not use the older `reducers/` or `gpt/reducers/` layout from earlier notes.
- Selected directories also contain generated `.out`, `.err`, `.txt`, and cache files. Treat the checked-in `.sh` and `.slurm` launchers as the source of truth.

## Current Launcher Families

Dense baselines:

- `slurm_scripts/vista/baseline/AdamS/gpt-345M/`
- `slurm_scripts/vista/baseline/AdamW/gpt-345M/`
- `slurm_scripts/vista/baseline/AdamS/llama-H2O-500M/`
- `slurm_scripts/vista/baseline/AdamW/llama-H2O-500M/`

Sparse SCAPE launchers:

- `slurm_scripts/vista/scape/gpt-345M/d_0.01/`
- `slurm_scripts/vista/scape/gpt-345M/d_0.1/`
- `slurm_scripts/vista/scape/llama-500M/d_0.01/`
- `slurm_scripts/vista/scape/llama-500M/d_0.1/`

Scaling-efficiency studies:

- dense AdamS launchers for `500M_micro_bs_8` and `1.8B_micro_bs_8`
- sparse SCAPE launchers for `d_0.01` and `d_0.1`
- distributed-optimizer and non-distributed-optimizer branches under SCAPE
- selected `64gpus/` and `128gpus/` `.slurm` wrappers for fixed-scale runs
- CPU-offload scaling launchers under `slurm_scripts/vista/scaling_efficiency/SCAPE_cpu_offload/`

Memory studies:

- dense AdamS baseline
- dense AdamS baseline with distributed optimizer
- SCAPE without residual offload
- SCAPE with residual offload
- SCAPE with distributed optimizer and full-model offload
- SCAPE with distributed optimizer and residual-model offload

## Running the Current Launchers

The checked-in launchers now invoke `pretrain_gpt.py` from the current working directory. Run them from the repository root.

Example shell launcher usage:

```bash
cd /path/to/SCAPE_SC26_ADAE
bash ./slurm_scripts/vista/baseline/AdamW/gpt-345M/pretrain_gpt_345M.sh
```

Another example:

```bash
cd /path/to/SCAPE_SC26_ADAE
bash ./slurm_scripts/vista/scape/llama-500M/d_0.1/scape_llama_500M.sh
```

Important execution behavior:

- `.sh` launchers expect to run inside an active Slurm allocation
- if `SLURM_JOB_ID` is missing, those scripts exit immediately
- once inside an allocation, the `.sh` launchers typically self-launch one worker task per node with `srun`
- `.slurm` launchers are direct `sbatch` entrypoints

A typical shell-launcher workflow is:

```bash
cd /path/to/SCAPE_SC26_ADAE
salloc -A <account> -p <partition> -N 4 -t 02:00:00
bash ./slurm_scripts/vista/scape/llama-500M/d_0.1/scape_llama_500M.sh
```

A typical `.slurm` workflow is:

```bash
cd /path/to/SCAPE_SC26_ADAE
sbatch ./slurm_scripts/vista/memory_usage_micro_bs_8/SCAPE_d_0.1_dist-optm_no_residual_model_offload/1.3B/scape_llama2_1.3B_baseline.slurm
```

## Required Edits Before Launching

Most checked-in launchers are templates and still contain placeholders. Before running them, replace or export the following values.

| Item | Where it is used | What to provide |
| --- | --- | --- |
| `YOUR_ACCOUNT` | `#SBATCH -A ...`, `salloc`, `srun` | Your Slurm allocation/account |
| `YOUR_PARTITION` | `#SBATCH -p ...`, `salloc`, `srun` | Your Slurm partition/queue |
| `YOUR_EMAIL_ADDRESS` | `#SBATCH --mail-user=...` | Your email for job notifications |
| `/path/to/repo` | `CONTAINER_CMD` | Absolute path to this checkout |
| `/path/to/data` | `CONTAINER_CMD`, `DATA_PATH` | Host path containing preprocessed data |
| `/path/to/ckpts` | `CONTAINER_CMD`, `CHECKPOINT_PATH` | Writable checkpoint/output path |
| `/path/to/pytorch_26.01-py3.sif` | `CONTAINER_CMD` | Apptainer image based on `nvcr.io/nvidia/pytorch:26.01-py3` |
| `HF_TOKEN` or `HF_TOKEN_FILE` | Llama tokenizer access | Hugging Face access token for gated assets when needed |
| `WANDB_API_KEY` or `WANDB_API_KEY_FILE` | Optional logging | Needed only if you want W&B logging |
| `GPT2_VOCAB_FILE`, `GPT2_MERGE_FILE` | GPT-345M launchers | GPT-2 vocab and merges files |

Common environment variables used by the current launchers:

- `DATA_PATH`
- `CHECKPOINT_PATH`
- `TOKENIZER_MODEL`
- `HF_TOKEN`
- `HF_TOKEN_FILE`
- `WANDB_API_KEY`
- `WANDB_API_KEY_FILE`
- `GPT2_VOCAB_FILE`
- `GPT2_MERGE_FILE`

Important portability note:

- run shell launchers from the repository root with `bash ./slurm_scripts/...`
- the repository path used in `CONTAINER_CMD` must match the real checkout path
- that same path must be bind-mounted into the container so `pretrain_gpt.py` is still visible from the current working directory inside Apptainer

## Container Setup

The launchers currently use a placeholder command of the form:

```bash
CONTAINER_CMD="apptainer exec --nv --bind /path/to/repo --bind /path/to/data --bind /path/to/ckpts --fakeroot /path/to/pytorch_26.01-py3.sif"
```

A typical setup looks like:

```bash
export REPO_ROOT=/path/to/SCAPE_SC26_ADAE
export DATA_ROOT=/path/to/data
export CKPT_ROOT=/path/to/ckpts
export CONTAINER=/path/to/pytorch_26.01-py3.sif

ml tacc-apptainer
apptainer exec --nv \
  --bind ${REPO_ROOT}:${REPO_ROOT} \
  --bind ${DATA_ROOT}:${DATA_ROOT} \
  --bind ${CKPT_ROOT}:${CKPT_ROOT} \
  --fakeroot \
  ${CONTAINER} \
  bash
```

If needed, pull the base image first:

```bash
apptainer pull /path/to/pytorch_26.01-py3.sif docker://nvcr.io/nvidia/pytorch:26.01-py3
```

Credential examples:

```bash
echo <huggingface_token> > $HOME/hf_token
echo <wandb_api_key> > $HOME/wandb_key
```

## SlimPajama-6B Download and Preprocessing

The current Llama launchers expect a Megatron preprocessed dataset prefix such as:

- `${OUTPUT_PREFIX}_text_document.bin`
- `${OUTPUT_PREFIX}_text_document.idx`

For Llama runs, `DATA_PATH` should point at the shared prefix without the suffix, for example:

```bash
DATA_PATH=/path/to/processed/slimpajama/slimpajama6b_llama2_text_document
```

Current checked-in helpers for this workflow:

- `tools/download_slimpajama_6b.py`
- `tools/run_prepare_slimpajama_6b.sh`
- `tools/preprocess_data.py`

`tools/run_prepare_slimpajama_6b.sh` currently honors these variables:

- `RAW_DIR`
- `RAW_JSONL`
- `OUTPUT_PREFIX`
- `TOKENIZER_MODEL`
- `WORKERS`
- `PYTHON_BIN`
- `VENV_ACTIVATE`

Example preprocessing workflow:

```bash
cd /path/to/SCAPE_SC26_ADAE
export RAW_DIR=/path/to/raw/slimpajama6b
export OUTPUT_PREFIX=/path/to/processed/slimpajama/slimpajama6b_llama2
export TOKENIZER_MODEL=meta-llama/Llama-2-7b-hf
export WORKERS=32
bash tools/run_prepare_slimpajama_6b.sh
```

The helper prints the final dataset prefix as:

```text
DATA_PATH=${OUTPUT_PREFIX}_text_document
```

## Export to HF and Run LM Evaluation

`tools/run_convert_torch_dist_to_hf.sh` converts a Megatron `torch_dist` checkpoint into a Hugging Face export by launching `examples/post_training/modelopt/export.py` inside Apptainer.

Current variables used by the export script:

- `CKPT_ROOT`: input `torch_dist` checkpoint root; it must contain `latest_checkpointed_iteration.txt`
- `HF_OUT`: output HF directory; defaults to `${CKPT_ROOT}_hf`
- `IMG`: Apptainer image path
- `MEGATRON`: path to this `SCAPE_SC26_ADAE` checkout
- `HF_TOKEN_FILE`: Hugging Face token file; defaults to `$HOME/hf_token`

Current environment requirements for the export step:

- `ml tacc-apptainer` must be available on the host
- the checkpoint, output, and repo paths must be reachable through the script's current bind mounts
- `HF_TOKEN_FILE` must exist before launching the export

Example export workflow:

```bash
cd /path/to/SCAPE_SC26_ADAE
export CKPT_ROOT=/path/to/torch_dist_ckpt
export HF_OUT=${CKPT_ROOT}_hf
export IMG=/path/to/pytorch_26.02-py3.sif
export MEGATRON=/path/to/SCAPE_SC26_ADAE
bash tools/run_convert_torch_dist_to_hf.sh
```

`tools/run_downstream_lm_eval.sh` runs `lm_eval` on the converted HF checkpoint.

Current variables used by the evaluation script:

- `HF_MODEL`: path to the converted HF checkpoint
- `OUT_DIR`: directory for lm-eval outputs

Environment requirement for downstream evaluation:

- run `tools/run_downstream_lm_eval.sh` from a Python environment where the `lm_eval` command is installed and available on `PATH`
- that environment should also include a compatible CUDA-enabled PyTorch plus the Hugging Face runtime packages used by `lm_eval`, such as `transformers` and `accelerate`
- the script reads `HF_TOKEN` from `$HOME/hf_token`
- the script currently uses `--device cuda:1` and `--batch_size 128`, so adjust the script or runtime environment if your machine differs

Example LM-eval workflow:

```bash
source /path/to/lm-eval-venv/bin/activate
command -v lm_eval
cd /path/to/SCAPE_SC26_ADAE
export HF_MODEL=/path/to/hf_ckpt
export OUT_DIR=/path/to/lm_eval_logs
bash tools/run_downstream_lm_eval.sh
```

## Helper Scripts

Current artifact-relevant helpers include:

- `tools/compute_avg_step_time.py`
- `tools/summarize_topk_runtime.py`
- `tools/download_slimpajama_6b.py`
- `tools/run_prepare_slimpajama_6b.sh`
- `tools/run_convert_torch_dist_to_hf.sh`
- `tools/run_downstream_lm_eval.sh`

Example step-time summary usage:

```bash
cd /path/to/SCAPE_SC26_ADAE
python3 tools/compute_avg_step_time.py \
  --paths ./slurm_scripts/vista/scape/llama-500M/d_0.1 \
  --recursive \
  --skip-steps 10
```

## Quick Checklist

Before submitting jobs, make sure that:

- you are in the repository root
- the launcher path you use matches the current `slurm_scripts/vista/` tree above
- placeholder account, partition, email, container, data, and checkpoint paths have been replaced
- `DATA_PATH` points to a real preprocessed dataset prefix
- GPT launchers have valid GPT-2 vocab and merges files
- Llama launchers have a valid tokenizer model and token if needed
- `.sh` launchers are run inside an active Slurm allocation
- `.slurm` launchers are submitted with `sbatch`
- HF export runs have valid `CKPT_ROOT`, `HF_OUT`, `IMG`, `MEGATRON`, and `HF_TOKEN_FILE` settings
- downstream evaluation is run from an environment where `lm_eval` is installed, along with compatible `torch`, `transformers`, and `accelerate`
