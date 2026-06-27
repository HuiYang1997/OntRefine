#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-axiom_repair}"
SKIP_SETUP="${SKIP_SETUP:-0}"
FORCE_SETUP="${FORCE_SETUP:-0}"

INPUT_OWL="${INPUT_OWL:-${SCRIPT_DIR}/examples/custom_ontology.owl}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  INPUT_OWL="$1"
  shift
fi

python_runner=()

have_conda() { command -v conda >/dev/null 2>&1; }
conda_env_exists() { conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; }

setup_conda() {
  if ! conda_env_exists; then
    echo "Creating conda environment: ${CONDA_ENV}"
    conda env create -f "${SCRIPT_DIR}/environment.yml" -n "${CONDA_ENV}"
  elif [[ "${FORCE_SETUP}" == "1" ]]; then
    echo "Updating conda environment: ${CONDA_ENV}"
    conda env update -f "${SCRIPT_DIR}/environment.yml" -n "${CONDA_ENV}" --prune
  else
    echo "Using existing conda environment: ${CONDA_ENV}"
  fi
  python_runner=(conda run --no-capture-output -n "${CONDA_ENV}" python)
}

setup_venv() {
  local venv_dir="${SCRIPT_DIR}/.venv"
  if [[ ! -d "${venv_dir}" ]]; then
    echo "Creating virtual environment: ${venv_dir}"
    python3 -m venv "${venv_dir}"
  fi
  if [[ ! -f "${venv_dir}/.setup_complete" || "${FORCE_SETUP}" == "1" ]]; then
    "${venv_dir}/bin/python" -m pip install --upgrade pip
    "${venv_dir}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
    touch "${venv_dir}/.setup_complete"
  else
    echo "Using existing .venv"
  fi
  python_runner=("${venv_dir}/bin/python")
}

if [[ "${SKIP_SETUP}" == "1" ]]; then
  python_runner=(python)
elif have_conda; then
  setup_conda
else
  setup_venv
fi

echo "======================================================"
echo "  OWL Axiom Repair"
echo "======================================================"
echo "  Input OWL: ${INPUT_OWL}"
echo "  Default LLM: local Qwen unless overridden"
echo "======================================================"

"${python_runner[@]}" "${SCRIPT_DIR}/src/run_pipeline.py" \
  --config "${SCRIPT_DIR}/config.yaml" \
  --input "${INPUT_OWL}" \
  "$@"
