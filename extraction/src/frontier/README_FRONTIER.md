# Frontier: Muse-Glimmer-30B extraction benchmark

The scripts in this directory match this repository layout:

```text
project_root/
├── human/reviewer_1.csv
├── papers/{1244,347,393,459,605}.pdf
└── src/frontier/
```

The local BF16 checkpoint defaults to:

```text
/lustre/orion/csc677/scratch/domsoos/models/Muse-Glimmer-30B
```

The serving stack uses the official ROCm vLLM image pinned by default to **v0.30.0**, because current Muse-Glimmer support and the `muse_glimmer` reasoning parser are required. The job exposes an OpenAI-compatible endpoint only on the allocated compute node.

## 1. Pull the repo on Frontier

From the project root after your normal `git pull`:

```bash
pwd
# should be the directory containing human/, papers/, and src/
```

## 2. One-time client environment

```bash
bash src/frontier/setup_client_env.sh
```

This creates a lightweight Python client environment at:

```text
/lustre/orion/csc677/scratch/domsoos/envs/glimmer-extraction-client
```

## 3. One-time vLLM container pull

```bash
bash src/frontier/pull_vllm_container.sh
```

Default image:

```text
docker://vllm/vllm-openai-rocm:v0.30.0
```

The resulting SIF is kept under your project scratch container directory.

## 4. Preflight

```bash
bash src/frontier/preflight.sh
```

This checks all five PDFs, the completed reviewer CSV, the local Glimmer weights, client environment, and vLLM SIF. It also prints model config information and hashes the human reference/model metadata.

## 5. Smoke test first

```bash
sbatch src/frontier/smoke_test_frontier.slurm
```

This allocates one Frontier node, starts Muse-Glimmer using 4 MI250X GCDs by default, and runs only **S0 on 1244.pdf**.

Monitor:

```bash
squeue -u $USER
# after/while running:
tail -f glimmer_smoke-<JOBID>.out
tail -f logs/vllm-smoke-<JOBID>.log
```

A successful smoke test writes:

```text
smoke_results/model_extractions.csv
smoke_results/1244/single.json
```

## 6. Full five-paper S0/S1/M1 experiment

```bash
sbatch src/frontier/run_frontier_benchmark.slurm
```

Monitor:

```bash
squeue -u $USER
tail -f glimmer_extract-<JOBID>.out
tail -f logs/vllm-<JOBID>.log
```

The full job performs:

1. Muse-Glimmer vLLM startup;
2. Covidence reviewer export -> internal human-reference schema;
3. S0, S1, and M1 extraction for all five PDFs;
4. automatic field comparison with the completed human extraction;
5. creation of `manual_review.csv` for later semantic correctness scoring;
6. benchmark table generation;
7. reproducibility metadata/hashes.

The extraction run uses `--resume`, so if a job is restarted it reuses completed `<paper>/<mode>.json` outputs.

## Useful overrides

Environment variables can be placed before `sbatch`, for example:

```bash
TP_SIZE=4 MAX_MODEL_LEN=65536 REASONING_STRENGTH=medium \
  sbatch src/frontier/run_frontier_benchmark.slurm
```

Other supported overrides include `MODEL_PATH`, `CLIENT_ENV`, `VLLM_SIF`, `PORT`, `GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`, and `USE_OLCF_GPU_BIND`.

If the official container reports a ROCm library conflict with OLCF's experimental GPU-binding helper, retry the smoke test with:

```bash
USE_OLCF_GPU_BIND=0 sbatch src/frontier/smoke_test_frontier.slurm
```

Do this only if the default runtime path fails; the default follows OLCF's documented Apptainer GPU-helper workflow.

## After semantic manual scoring

Open `results/manual_review.csv` and fill `human_judgment` with one of:

```text
correct
partial
incorrect
unsupported
missing
```

Then run on a login node:

```bash
bash src/frontier/post_review_summary.sh
```

This updates `results/manual_accuracy_summary.csv` and the final `results/benchmark_table.{csv,md}`.
