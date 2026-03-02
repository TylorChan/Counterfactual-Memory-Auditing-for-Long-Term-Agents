#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECREATE="${RECREATE:-0}" # set RECREATE=1 to remove and recreate envs

CONDA_SH=""
for candidate in \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"; do
  if [[ -f "${candidate}" ]]; then
    CONDA_SH="${candidate}"
    break
  fi
done

if [[ -z "${CONDA_SH}" ]]; then
  echo "[ERROR] conda.sh not found. Install Miniconda/Anaconda first." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"

create_env_if_needed() {
  local env_name="$1"
  local py_ver="$2"

  if [[ "${RECREATE}" == "1" ]]; then
    conda remove -n "${env_name}" --all -y >/dev/null 2>&1 || true
  fi

  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[SKIP] ${env_name} exists"
  else
    echo "[CREATE] ${env_name}"
    conda create -n "${env_name}" "python=${py_ver}" -y
  fi
}

echo "[INFO] ROOT_DIR=${ROOT_DIR}"

create_env_if_needed "anna-lme" "3.10"
create_env_if_needed "share-lme" "3.10"
create_env_if_needed "memos-lme" "3.10"
create_env_if_needed "ld-lme" "3.10"

export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "[SETUP] anna-lme"
conda activate anna-lme
python -m pip install --upgrade pip
python -m pip install \
  openai==1.12.0 \
  httpx==0.27.2 \
  tqdm \
  environs \
  pydantic \
  python-dotenv \
  PyYAML
python -m pip install -e "${ROOT_DIR}/AnnaAgent"

echo "[SETUP] share-lme"
conda activate share-lme
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/share_longmemeval_bridge/requirements.txt"

echo "[SETUP] memos-lme"
conda activate memos-lme
python -m pip install --upgrade pip setuptools wheel
conda install -n memos-lme -y -c conda-forge faiss-cpu
TMP_REQ="$(mktemp /tmp/memoryos_requirements_cpu.XXXXXX.txt)"
grep -v '^faiss-gpu' "${ROOT_DIR}/MemoryOS/memoryos-pypi/requirements.txt" > "${TMP_REQ}"
python -m pip install -r "${TMP_REQ}"
python -m pip install tqdm
rm -f "${TMP_REQ}"

echo "[SETUP] ld-lme"
conda activate ld-lme
python -m pip install --upgrade pip
python -m pip install \
  openai==1.12.0 \
  httpx==0.27.2 \
  chromadb==0.4.22 \
  spacy==3.7.4 \
  tqdm \
  pillow
python -m pip install \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

echo "[OK] Environments ready: anna-lme, share-lme, memos-lme, ld-lme"
echo "[TIP] Run with: cd ${ROOT_DIR} && bash run_mals.sh"
