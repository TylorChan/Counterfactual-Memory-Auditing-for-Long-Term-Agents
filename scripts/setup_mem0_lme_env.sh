#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${WORKDIR}/scripts/conda_bootstrap.sh"
source_conda_sh

ENV_NAME="${ENV_NAME:-mem0-lme}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
MEM0_VERSION="${MEM0_VERSION:-2.0.1}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env ${ENV_NAME} already exists. Installing/updating mem0 dependencies."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  "mem0ai==${MEM0_VERSION}" \
  "fastembed>=0.3.1" \
  "openai>=1.90.0" \
  "tqdm>=4.66.0"

conda run -n "${ENV_NAME}" python -c "import fastembed, mem0, openai, qdrant_client; print('mem0:', getattr(mem0, '__file__', 'ok')); print('openai:', openai.__version__); print('qdrant_client: ok'); print('fastembed: ok')"
