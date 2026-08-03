#!/usr/bin/env bash

set -euo pipefail
trap '' INT QUIT TSTP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat >&2 << 'EOF'
Usage:
    ./scripts/run_timestamped.sh [--log-name NAME] <script.py> [script args...]

The wrapper creates outputs/<timestamp>/, injects --output-dir for experiment
scripts, captures stdout/stderr, and refuses a manually supplied --output-dir.
judge_outputs_llm_asr.py manages its report directory from --judge-input-path.
Set RIGJ_PYTHON to override the interpreter. Otherwise python3 from the current
shell PATH is used.
EOF
}

log_name=""
if [[ "${1:-}" == "--log-name" ]]; then
    [[ $# -ge 2 ]] || {
        usage
        exit 1
    }
    log_name="$2"
    shift 2
fi
[[ $# -ge 1 ]] || {
    usage
    exit 1
}

script_arg="$1"
shift
case "${script_arg}" in
    /*) script_path="${script_arg}" ;;
    ./*) script_path="${ROOT_DIR}/${script_arg#./}" ;;
    *) script_path="${ROOT_DIR}/${script_arg}" ;;
esac
[[ -f "${script_path}" ]] || {
    echo "error: script not found: ${script_arg}" >&2
    exit 1
}

for arg in "$@"; do
    if [[ "${arg}" == "--output-dir" || "${arg}" == --output-dir=* ]]; then
        echo "error: --output-dir is managed by run_timestamped.sh" >&2
        exit 1
    fi
done

if [[ -z "${log_name}" ]]; then
    log_name="$(basename "${script_path}" .py)"
fi

timestamp="$(date '+%Y-%m-%d_%H-%M-%S')"
run_dir="${ROOT_DIR}/outputs/${timestamp}"
suffix=1
while [[ -e "${run_dir}" ]]; do
    run_dir="${ROOT_DIR}/outputs/${timestamp}_${suffix}"
    suffix=$((suffix + 1))
done
mkdir -p "${run_dir}"

script_base="$(basename "${script_path}")"
script_args=("$@")
python_bin="${RIGJ_PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
    python_bin="$(command -v python3)"
elif [[ "${python_bin}" != */* ]]; then
    python_bin="$(command -v "${python_bin}")"
fi
[[ -x "${python_bin}" ]] || {
    echo "error: Python interpreter is not executable: ${python_bin}" >&2
    exit 1
}
if [[ "${script_base}" == "judge_outputs_llm_asr.py" ]]; then
    cmd=("${python_bin}" -u "${script_path}" "${script_args[@]}")
else
    cmd=("${python_bin}" -u "${script_path}" --output-dir "${run_dir}" "${script_args[@]}")
fi

{
    printf 'working_directory=%q\n' "${ROOT_DIR}"
    printf 'run_directory=%q\n' "${run_dir}"
    printf 'command='
    printf '%q ' "${cmd[@]}"
    printf '\n'
} > "${run_dir}/command.txt"

echo "[run] directory: ${run_dir}"
echo "[run] log: ${run_dir}/${log_name}.log"
printf '[run] command: '
printf '%q ' "${cmd[@]}"
printf '\n'

cd "${ROOT_DIR}"
set +e
"${cmd[@]}" 2>&1 | tee "${run_dir}/${log_name}.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${status}" > "${run_dir}/exit_status.txt"
exit "${status}"
