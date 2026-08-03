#!/usr/bin/env bash

set -euo pipefail
trap '' INT QUIT TSTP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat >&2 <<'EOF'
Usage:
    ./scripts/wait_then_run_all_20_pairs.sh [--poll-seconds N]
        [--wait-gpus GPU ... --min-free-gpu-memory-gib GIB]

Wait until the fixed five-model prefix attack bank is complete and, when GPU
waiting is enabled, every selected GPU has more than the requested amount of
free memory. Then run all 20 directed model pairs in a timestamped directory.

All experiment paths and pair arguments are defined inside this script.
The command retries with generation batch sizes 4, 2, then 1 on failure.
EOF
}

poll_seconds=30
wait_gpus=()
min_free_gpu_memory_gib=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --poll-seconds|-p)
            if [[ $# -lt 2 ]]; then
                echo "error: --poll-seconds requires a positive integer" >&2
                exit 1
            fi
            poll_seconds="$2"
            shift 2
            ;;
        --wait-gpus)
            shift
            gpu_count_before="${#wait_gpus[@]}"
            while [[ $# -gt 0 && "$1" != --* ]]; do
                wait_gpus+=("$1")
                shift
            done
            if [[ "${#wait_gpus[@]}" -eq "${gpu_count_before}" ]]; then
                echo "error: --wait-gpus requires one or more GPU indices" >&2
                exit 1
            fi
            ;;
        --min-free-gpu-memory-gib)
            if [[ $# -lt 2 ]]; then
                echo "error: --min-free-gpu-memory-gib requires a positive integer" >&2
                exit 1
            fi
            min_free_gpu_memory_gib="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if ! [[ "${poll_seconds}" =~ ^[0-9]+$ ]] || [[ "${poll_seconds}" -le 0 ]]; then
    echo "error: --poll-seconds must be a positive integer: ${poll_seconds}" >&2
    exit 1
fi

for wait_gpu in "${wait_gpus[@]}"; do
    if ! [[ "${wait_gpu}" =~ ^[0-9]+$ ]]; then
        echo "error: GPU index must be a non-negative integer: ${wait_gpu}" >&2
        exit 1
    fi
done

if [[ "${#wait_gpus[@]}" -gt 0 ]]; then
    if ! [[ "${min_free_gpu_memory_gib}" =~ ^[0-9]+$ ]] || [[ "${min_free_gpu_memory_gib}" -le 0 ]]; then
        echo "error: --wait-gpus requires --min-free-gpu-memory-gib with a positive integer" >&2
        exit 1
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "error: nvidia-smi is required for GPU memory waiting" >&2
        exit 1
    fi
elif [[ -n "${min_free_gpu_memory_gib}" ]]; then
    echo "error: --min-free-gpu-memory-gib requires --wait-gpus" >&2
    exit 1
fi

# Fixed inputs for this experiment. These definitions deliberately do not
# depend on variables exported by the shell that launches this script.
readonly ATTACK_RUN="${ROOT_DIR}/outputs/attack_banks/rigj_qwen3-8b_llama3.1-8b_mistral7b-v0.3_vicuna7b-v1.5_internlm3-8b_prefix_harmbench-gjo_train20_seed0_steps1000_tokens100"
readonly ATTACK_BANK="${ATTACK_RUN}/attack_assets"
readonly MODEL_CONFIG="${ROOT_DIR}/configs/model_families.json"
readonly BENCHMARK_CACHE_DIR="${ROOT_DIR}/outputs/benchmark_cache"
readonly WAIT_FILE="${ATTACK_BANK}/attack_bank_manifest.json"
export ATTACK_RUN ATTACK_BANK MODEL_CONFIG BENCHMARK_CACHE_DIR

PAIR_COMMON=(
    --model-config "${MODEL_CONFIG}"
    --attack-bank "${ATTACK_BANK}"
    --benchmarks
        harmbench
        jailbreakbench
        strongreject
        advbench
        malicious_instruct
        do_not_answer
        xstest_unsafe
        sorrybench
        wildjailbreak
    --benchmark-cache-dir "${BENCHMARK_CACHE_DIR}"
    --benchmark-max-examples 199
    --benchmark-timeout-seconds 10
    --seed 0
    --attack-position prefix
    --max-new-tokens 256
    --device-map auto
    --torch-dtype bfloat16
    --trust-remote-code
    --local-files-only
)

[[ -f "${MODEL_CONFIG}" ]] || {
    echo "error: model config not found: ${MODEL_CONFIG}" >&2
    exit 1
}
[[ -d "${BENCHMARK_CACHE_DIR}" ]] || {
    echo "error: benchmark cache directory not found: ${BENCHMARK_CACHE_DIR}" >&2
    exit 1
}

attack_bank_ready() {
    local family
    [[ -s "${WAIT_FILE}" ]] || return 1
    for family in qwen llama mistral vicuna internlm; do
        [[ -s "${ATTACK_BANK}/${family}/attack.json" ]] || return 1
    done
}

declare -A gpu_free_mib=()
gpu_free_memory_snapshot() {
    local index free_mib
    gpu_free_mib=()
    while IFS=',' read -r index free_mib; do
        index="${index//[[:space:]]/}"
        free_mib="${free_mib//[[:space:]]/}"
        if [[ "${index}" =~ ^[0-9]+$ && "${free_mib}" =~ ^[0-9]+$ ]]; then
            gpu_free_mib["${index}"]="${free_mib}"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
    [[ "${#gpu_free_mib[@]}" -gt 0 ]]
}

echo "[wait_then_run] attack_run=${ATTACK_RUN}"
echo "[wait_then_run] waiting for completed attack bank: ${WAIT_FILE}"
echo "[wait_then_run] poll_seconds=${poll_seconds}"
if [[ "${#wait_gpus[@]}" -gt 0 ]]; then
    echo "[wait_then_run] waiting for GPUs ${wait_gpus[*]} to each have >${min_free_gpu_memory_gib} GiB free"
fi

while true; do
    bank_ready=false
    if attack_bank_ready; then
        bank_ready=true
    fi

    gpu_ready=true
    gpu_status="disabled"
    if [[ "${#wait_gpus[@]}" -gt 0 ]]; then
        gpu_status=""
        threshold_mib=$((min_free_gpu_memory_gib * 1024))
        if gpu_free_memory_snapshot; then
            for wait_gpu in "${wait_gpus[@]}"; do
                free_mib="${gpu_free_mib[${wait_gpu}]:-missing}"
                gpu_status+="gpu${wait_gpu}=${free_mib}MiB "
                if [[ "${free_mib}" == "missing" ]] || (( free_mib <= threshold_mib )); then
                    gpu_ready=false
                fi
            done
        else
            gpu_ready=false
            gpu_status="nvidia-smi-query-failed"
        fi
    fi

    if [[ "${bank_ready}" == true && "${gpu_ready}" == true ]]; then
        break
    fi

    now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    completed=()
    for family in qwen llama mistral vicuna internlm; do
        if [[ -s "${ATTACK_BANK}/${family}/attack.json" ]]; then
            completed+=("${family}")
        fi
    done
    echo "[wait_then_run] ${now}; bank_ready=${bank_ready}; completed=${#completed[@]}/5: ${completed[*]:-none}; gpu_free=${gpu_status}; sleep ${poll_seconds}s"
    sleep "${poll_seconds}"
done

echo "[wait_then_run] attack bank and GPU conditions are satisfied"
cd "${ROOT_DIR}"

run_step() {
    echo "[wait_then_run] launching next command:"
    printf '  %q' "$@"
    printf '\n'
    "$@"
}

run_step_with_gen_batch_fallback() {
    local batch_size
    for batch_size in 4 2 1; do
        echo "[wait_then_run] trying --gen-batch-size ${batch_size}"
        if run_step "$@" --gen-batch-size "${batch_size}"; then
            echo "[wait_then_run] selected_gen_batch_size=${batch_size}"
            return 0
        fi
        echo "[wait_then_run] --gen-batch-size ${batch_size} failed; trying next fallback"
    done
    echo "[wait_then_run] error: all fallback batch sizes failed: 4 2 1" >&2
    return 1
}

run_step_with_gen_batch_fallback \
    env CUDA_VISIBLE_DEVICES=0,1 \
    ./scripts/run_timestamped.sh \
    --log-name all_20_pairs_prefix \
    scripts/run_all_model_pairs.py \
    "${PAIR_COMMON[@]}"
