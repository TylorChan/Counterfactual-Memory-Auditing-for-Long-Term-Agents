#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="${MEM0_BENCHMARKS_DIR:-${WORKDIR}/external/memory-benchmarks}"
BENCH_REF="${MEM0_BENCHMARKS_REF:-f75666d33ef560f0f196746e0e16c515d17e6856}"
ENV_NAME="${ENV_NAME:-mem0-lme}"

source "${WORKDIR}/scripts/conda_bootstrap.sh"
source_conda_sh

if [[ ! -d "${BENCH_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${BENCH_DIR}")"
  git clone https://github.com/mem0ai/memory-benchmarks.git "${BENCH_DIR}"
fi

git -C "${BENCH_DIR}" fetch --all --tags --depth 1 || true
git -C "${BENCH_DIR}" checkout "${BENCH_REF}"

conda activate "${ENV_NAME}"
python -m pip install -r "${BENCH_DIR}/requirements.txt"
python - <<PY
import importlib
for name in ['aiohttp', 'aiolimiter', 'openai', 'tqdm', 'pydantic']:
    importlib.import_module(name)
print('memory-benchmarks ready:', '${BENCH_DIR}')
print('ref:', '${BENCH_REF}')
PY
