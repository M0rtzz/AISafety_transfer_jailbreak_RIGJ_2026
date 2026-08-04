# AISafety_transfer_jailbreak_RIGJ_2026

This is the official repository for ACL 2026 Main paper "Enhancing the Transferability of Jailbreak Attacks on Large Language Models via Exploiting Reparameterization Invariance" by Ao Wang, Xinghao Yang, Yongshun Gong, Wei Liu, Baodi Liu, and Weifeng Liu*.

## Installation

You can download the packages needed as follow:

```
conda create -n rigj python==3.11
conda activate rigj
pip install -r requirements.txt
```
## Training / Attack Generation

Run the following command to perform adversarial prompt optimization:
```
nohup python -u ngd_main.py \
    --source-model llama2-7b \
    --train-dataset harmbench_gjo \
    --lr 1 \
    --beta-1 0.9 \
    --beta-2 0.9999 \
    --begin-tau 5 \
    --final-tau 1 \
    --n-train-data 20 \
    --num-steps 1000 \
    --num-adv-tokens 100 \
    > ./rigj-harmbench-gjo-vicuna-steps1000-lr1-beta0.9-0.9999-tau5-1-trainnum20-bf16-num100.out 2>&1
```
or run 
```
sh run.sh
```
## Argument Description
| Argument | Description |
|----------|------------|
| `--source-model` | Source model used for optimization (e.g., LLaMA2-7B) |
| `--train-dataset` | Training dataset (e.g., HarmBench) |
| `--lr` | Learning rate |
| `--beta-1` | First-order momentum coefficient |
| `--beta-2` | Second-order momentum coefficient |
| `--begin-tau` | Initial temperature |
| `--final-tau` | Final temperature |
| `--n-train-data` | Number of training samples |
| `--num-steps` | Number of optimization steps |
| `--num-adv-tokens` | Length of adversarial token sequence |
## Output

Logs will be saved to:
```
./rigj-harmbench-gjo-vicuna-steps1000-lr1-beta0.9-0.9999-tau5-1-trainnum20-bf16-num100.out
```
The log file includes:
```
optimization progress
intermediate adversarial tokens
training statistics
```
Results will be saved to:
```
./results/
```

## Reusable Harmful-Benchmark Pipeline

The repository includes a two-stage pipeline for generating one reusable RIGJ
attack artifact per source model and evaluating all 20 directed transfers among
Qwen, Llama, Mistral, Vicuna, and InternLM.

Copy and edit the model configuration first:

```bash
cp configs/model_families.example.json configs/model_families.json
```

All model paths may be local paths under `Models/`. The loaders accept
`--trust-remote-code`; Vicuna receives an in-memory fallback chat template when
its tokenizer has none, and InternLM3 is loaded through the strict native
Llama-compatible loader used by COMBAT.

The nine datasets can be prepared independently before any model run:

```bash
./scripts/run_timestamped.sh \
  scripts/run_harm_benchmark_eval.py \
  --download-only \
  --benchmark-max-examples 199 \
  --seed 0
```

### 1. Prepare five reusable attack artifacts

```bash
./scripts/run_timestamped.sh \
  scripts/prepare_attack_assets.py \
  --model-config configs/model_families.json \
  --train-dataset harmbench_gjo \
  --n-train-data 20 \
  --seed 0 \
  --attack-position prefix \
  --trust-remote-code \
  --local-files-only
```

This runs the original NGD optimization once per source family. NGD candidate
screening retains the original `max_new_tokens=24`. The packaged attack assets
store the decoded `best_adv_prompt`, provenance, original NGD JSON, and
materialized attacked prompts for the nine benchmarks.

### 2. Evaluate all 20 directed model pairs

```bash
./scripts/run_timestamped.sh \
  scripts/run_all_model_pairs.py \
  --model-config configs/model_families.json \
  --attack-bank outputs/<attack-run>/attack_assets \
  --benchmark-max-examples 199 \
  --seed 0 \
  --max-new-tokens 256 \
  --attack-position prefix \
  --trust-remote-code \
  --local-files-only
```

The runner loads models sequentially on one GPU and reuses the five source
generation jobs. Consequently, the 20 directed pairs require 25 unique
generation jobs rather than regenerating each source output four times.
Successful rows, generation errors, and `input_too_long` rows are saved
incrementally. Inputs are never silently truncated.

The harmful suite is:

- HarmBench Standard test: all 159 official examples;
- JailbreakBench;
- StrongREJECT;
- AdvBench;
- MaliciousInstruct;
- Do-Not-Answer;
- XSTest unsafe rows only;
- SorryBench;
- WildJailbreak `adversarial_harmful` rows only.

The default selection is `min(199, available_count)` per benchmark with
deterministic `seed=0`. Downloaded normalized data is cached under
`outputs/benchmark_cache`. A failed source falls back to another source or a
valid materialized cache; the run fails if neither is available.

### 3. Judge saved outputs (ASR or AHS)

Configure the same environment variables used by COMBAT:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
export LLM_ASR_JUDGE_CONCURRENCY=4
```

Dry-run normalization without making API calls:

```bash
python scripts/judge_outputs_llm_asr.py \
  --judge-input-path outputs/<pair-run> \
  --judge-dry-run
```

Run or resume judging:

```bash
python scripts/judge_outputs_llm_asr.py \
  --asr \
  --judge-input-path outputs/<pair-run> \
  --judge-run-name rigj_all_pairs \
  --judge-resume
```

`--asr` is optional and remains the default. To reproduce the paper-style
1--10 Average Harmfulness Score, run the same saved generations with `--ahs`:

```bash
python scripts/judge_outputs_llm_asr.py \
  --ahs \
  --judge-input-path outputs/<pair-run> \
  --judge-run-name rigj_all_pairs \
  --judge-resume
```

AHS outputs are kept separate under `llm_ahs_judge/`, with names such as
`rigj_all_pairs.ahs.judgements.jsonl` and
`rigj_all_pairs.ahs.summary.json`. The summary reports `ahs` overall and by
benchmark, condition, condition/benchmark, model role, generation job, and
source-to-target pair. Generation rows without an explicit `condition` are
grouped as `default`. Use `--judge-conditions ...` to select named conditions.

The primary ASR is `prompt_matched_attack_success`. Reports also include
`any_harmful_output_rate`, the original RIGJ refusal-prefix metric, coverage,
conservative ASR with unavailable generations in the denominator, and
source-to-target ASR gaps.

During long judge runs, insufficient-balance/quota and `billing_error`
responses trigger an override reload of the repository `.env`, an API-client
rebuild, and a retry of the current sample. This allows rotating the judge API
key without restarting the run. Existing billing-error rows are also retried
automatically with `--judge-resume`.

### Output layout

```text
outputs/<timestamp>/
├── command.txt
├── <script-or-log-name>.log
├── attack_assets/                 # attack-preparation runs
├── benchmark_manifest.json
├── generations/                   # unique cached generation jobs
├── pairs/<source>_to_<target>/    # 20 pair manifests
├── all_pairs_manifest.json
├── llm_asr_judge/                 # ASR inputs, judgements, CSV, summary
└── llm_ahs_judge/                 # AHS inputs, judgements, CSV, summary
```

Use `--help` on each script for single-pair runs, benchmark subsets, batch
size, dtype, cache refresh, model-family overrides, and judge retry controls.
