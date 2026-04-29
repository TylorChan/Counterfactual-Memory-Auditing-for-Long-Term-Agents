#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORKDIR}"

source "${WORKDIR}/scripts/conda_bootstrap.sh"
source_conda_sh

load_env_file() {
  local env_file="$1"
  if [[ ! -f "${env_file}" ]]; then
    return 0
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line#export }"
    if [[ "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      local key="${BASH_REMATCH[1]}"
      local value="${BASH_REMATCH[2]}"
      if [[ "${value}" =~ ^\".*\"$ ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "${value}" =~ ^\'.*\'$ ]]; then
        value="${value:1:${#value}-2}"
      fi
      export "${key}=${value}"
    fi
  done < "${env_file}"
}

load_env_file "${WORKDIR}/.env"

KEY_VAR="${KEY_VAR:-OPENAI_API_KEY_1}"
if [[ -n "${!KEY_VAR:-}" ]]; then
  export OPENAI_API_KEY="${!KEY_VAR}"
elif [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Missing ${KEY_VAR} or OPENAI_API_KEY." >&2
  exit 1
fi

MEM0_OFFICIAL_BACKEND="${MEM0_OFFICIAL_BACKEND:-cloud}"
if [[ "${MEM0_OFFICIAL_BACKEND}" == "cloud" && -z "${MEM0_API_KEY:-}" ]]; then
  echo "Missing MEM0_API_KEY for cloud smoke. Set MEM0_OFFICIAL_BACKEND=oss to test against a local official OSS server." >&2
  exit 1
fi

if [[ ! -d "${MEM0_BENCHMARKS_DIR:-${WORKDIR}/external/memory-benchmarks}" ]]; then
  bash "${WORKDIR}/scripts/setup_mem0_official_benchmarks.sh"
fi

conda activate mem0-lme

STAMP="$(date +%m_%d_%H_%M)"
RUN_TAG="${RUN_TAG:-mem0_official_smoke_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/LongMemEval/${RUN_TAG}}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
DATA_FILE="${DATA_FILE:-${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_100_balanced_seed8980.json}"
LIMIT="${LIMIT:-10}"
OFFSET="${OFFSET:-0}"
ANSWERER_MODEL="${ANSWERER_MODEL:-gpt-5}"
TOP_K="${MEM0_OFFICIAL_TOP_K:-200}"
ANSWER_CUTOFF="${MEM0_OFFICIAL_ANSWER_CUTOFF:-200}"
ENABLE_CF_WRAPPER="${ENABLE_CF_WRAPPER:-1}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-prompt}"
CF_MAX_WRITES="${CF_MAX_WRITES:-3}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"
CF_RULE_MODE="${CF_RULE_MODE:-rollback-only}"
BENCH_DIR="${MEM0_BENCHMARKS_DIR:-${WORKDIR}/external/memory-benchmarks}"
BENCH_REF="${MEM0_BENCHMARKS_REF:-f75666d33ef560f0f196746e0e16c515d17e6856}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

CF_ARGS=()
if [[ "${ENABLE_CF_WRAPPER}" == "1" ]]; then
  CF_ARGS=(
    --enable-cf-wrapper
    --cf-target-scope "${CF_TARGET_SCOPE}"
    --cf-max-writes "${CF_MAX_WRITES}"
    --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}"
  )
fi

HOST_ARGS=()
if [[ -n "${MEM0_HOST:-}" ]]; then
  HOST_ARGS=(--mem0-host "${MEM0_HOST}")
fi
BASE_URL_ARGS=()
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  BASE_URL_ARGS=(--openai-base-url "${OPENAI_BASE_URL}")
fi

cat > "${OUTPUT_ROOT}/run_config.txt" <<CONFIG
run_tag=${RUN_TAG}
backend=${MEM0_OFFICIAL_BACKEND}
answerer_model=${ANSWERER_MODEL}
data_file=${DATA_FILE}
limit=${LIMIT}
offset=${OFFSET}
top_k=${TOP_K}
answer_cutoff=${ANSWER_CUTOFF}
enable_cf_wrapper=${ENABLE_CF_WRAPPER}
cf_rule_mode=${CF_RULE_MODE}
cf_target_scope=${CF_TARGET_SCOPE}
cf_max_writes=${CF_MAX_WRITES}
bench_dir=${BENCH_DIR}
CONFIG

python "${WORKDIR}/mem0_official_longmemeval_bridge/run_infer.py" \
  --memory-benchmarks-dir "${BENCH_DIR}" \
  --memory-benchmarks-ref "${BENCH_REF}" \
  --longmemeval-file "${DATA_FILE}" \
  --out-jsonl "${OUTPUT_ROOT}/preds_mem0_official_${RUN_TAG}_s1.jsonl" \
  --trace-jsonl "${OUTPUT_ROOT}/preds_mem0_official_${RUN_TAG}_s1.trace.jsonl" \
  --project-name "mem0-official-smoke" \
  --backend "${MEM0_OFFICIAL_BACKEND}" \
  --answerer-model "${ANSWERER_MODEL}" \
  --top-k "${TOP_K}" \
  --answer-cutoff "${ANSWER_CUTOFF}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET}" \
  --cf-rule-mode "${CF_RULE_MODE}" \
  "${HOST_ARGS[@]}" \
  "${BASE_URL_ARGS[@]}" \
  "${CF_ARGS[@]}" \
  > "${LOG_ROOT}/mem0_official_smoke.out" \
  2> "${LOG_ROOT}/mem0_official_smoke.err"

echo "${OUTPUT_ROOT}"
echo "${LOG_ROOT}/mem0_official_smoke.out"
echo "${LOG_ROOT}/mem0_official_smoke.err"
