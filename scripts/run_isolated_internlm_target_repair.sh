#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat >&2 <<'EOF'
Usage:
    ORIGINAL_PAIR_RUN=/abs/original/run \
    REPAIR_ROOT=/abs/new/repair/root \
    ATTACK_BANK=/abs/attack_assets \
    ./scripts/run_isolated_internlm_target_repair.sh \
        qwen|llama|mistral|vicuna

The script copies only the source/self and source/InternLM generation jobs into
an isolated repair directory. It never modifies the original generation cache.
EOF
}

[[ $# -eq 1 ]] || {
    usage
    exit 2
}

source_family="$1"
case "${source_family}" in
    qwen|llama|mistral|vicuna) ;;
    *)
        echo "error: unsupported source family: ${source_family}" >&2
        usage
        exit 2
        ;;
esac

: "${ORIGINAL_PAIR_RUN:?Set ORIGINAL_PAIR_RUN to the completed 20-pair run}"
: "${REPAIR_ROOT:?Set REPAIR_ROOT to a new directory}"
: "${ATTACK_BANK:?Set ATTACK_BANK to the attack_assets directory}"

MODEL_CONFIG="${MODEL_CONFIG:-${ROOT_DIR}/configs/model_families.json}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
PYTHON_BIN="${RIGJ_PYTHON:-python3}"

cd "${ROOT_DIR}"

original_pair_run="$(realpath "${ORIGINAL_PAIR_RUN}")"
attack_bank="$(realpath "${ATTACK_BANK}")"
model_config="$(realpath "${MODEL_CONFIG}")"
mkdir -p "${REPAIR_ROOT}"
repair_root="$(realpath "${REPAIR_ROOT}")"

[[ "${repair_root}" != "${original_pair_run}" ]] || {
    echo "error: REPAIR_ROOT must differ from ORIGINAL_PAIR_RUN" >&2
    exit 2
}

source_model="$(
    jq -er --arg family "${source_family}" '.models[$family].path' "${model_config}"
)"
target_model="$(jq -er '.models.internlm.path' "${model_config}")"

original_generations="${original_pair_run}/generations"
repair_dir="${repair_root}/${source_family}_to_internlm"
repair_generations="${repair_dir}/generations"
mkdir -p "${repair_generations}"

shopt -s nullglob
source_jobs=(
    "${original_generations}/${source_family}_attack__on__${source_family}_"*
)
target_jobs=(
    "${original_generations}/${source_family}_attack__on__internlm_"*
)
shopt -u nullglob

[[ ${#source_jobs[@]} -eq 1 ]] || {
    echo "error: expected one source/self generation job, found ${#source_jobs[@]}" >&2
    exit 2
}
[[ ${#target_jobs[@]} -eq 1 ]] || {
    echo "error: expected one source/InternLM generation job, found ${#target_jobs[@]}" >&2
    exit 2
}

copy_job_once() {
    local source_job="$1"
    local destination="${repair_generations}/$(basename "${source_job}")"
    if [[ -e "${destination}" ]]; then
        [[ -f "${destination}/job_manifest.json" ]] || {
            echo "error: incomplete existing repair job: ${destination}" >&2
            exit 2
        }
        echo "[repair] reusing isolated job copy: ${destination}"
        return
    fi
    echo "[repair] copying job without modifying original: ${source_job}"
    cp -a --reflink=auto "${source_job}" "${repair_generations}/"
}

copy_job_once "${source_jobs[0]}"
copy_job_once "${target_jobs[0]}"

echo "[repair] pair=${source_family}_to_internlm"
echo "[repair] output=${repair_dir}"
echo "[repair] generation_cache=${repair_generations}"

"${PYTHON_BIN}" -u scripts/run_harm_benchmark_eval.py \
    --output-dir "${repair_dir}" \
    --source-model "${source_model}" \
    --target-model "${target_model}" \
    --source-family "${source_family}" \
    --target-family internlm \
    --attack-artifact "${attack_bank}/${source_family}/attack.json" \
    --attack-position prefix \
    --benchmarks \
        harmbench \
        jailbreakbench \
        strongreject \
        advbench \
        malicious_instruct \
        do_not_answer \
        xstest_unsafe \
        sorrybench \
        wildjailbreak \
    --benchmark-cache-dir outputs/benchmark_cache \
    --benchmark-max-examples 199 \
    --benchmark-timeout-seconds 10 \
    --seed 0 \
    --gen-batch-size "${GEN_BATCH_SIZE}" \
    --max-new-tokens 256 \
    --device-map auto \
    --torch-dtype bfloat16 \
    --generation-cache-dir "${repair_generations}" \
    --trust-remote-code \
    --local-files-only

