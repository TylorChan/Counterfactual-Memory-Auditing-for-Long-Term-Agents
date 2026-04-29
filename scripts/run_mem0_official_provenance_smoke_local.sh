#!/usr/bin/env bash
set -euo pipefail

# One-question Mem0 Cloud provenance smoke test.
# Purpose: verify whether Mem0 Cloud search results can be mapped back to the
# source_write_id metadata we attach at write time. This is not an accuracy run.

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORKDIR}"

# -----------------------------
# Fixed smoke configuration
# -----------------------------
DATA_FILE="${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json"
RUN_TAG="mem0_official_provenance_smoke_$(date +%m_%d_%H_%M)"
OUTPUT_ROOT="${WORKDIR}/LongMemEval/${RUN_TAG}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
KEY_VAR="OPENAI_API_KEY_1"
MEM0_OFFICIAL_BACKEND="cloud"
OFFSET="2"                 # qid=852ce960, smallest write count in the 50-question subset
LIMIT="1"
ANSWERER_MODEL="gpt-4o-mini" # cheaper smoke; final CF50 script uses gpt-5
TOP_K="200"
ANSWER_CUTOFF="200"
ENABLE_CF_WRAPPER="1"
CF_RULE_MODE="rollback-only"
CF_TARGET_SCOPE="prompt"
CF_MAX_WRITES="1"          # one rollback to conserve Mem0 add quota
CF_DOMINANCE_THRESHOLD="0.75"
BENCH_DIR="${MEM0_BENCHMARKS_DIR:-${WORKDIR}/external/memory-benchmarks}"
BENCH_REF="${MEM0_BENCHMARKS_REF:-f75666d33ef560f0f196746e0e16c515d17e6856}"

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

if [[ -z "${MEM0_API_KEY:-}" ]]; then
  echo "Missing MEM0_API_KEY in ${WORKDIR}/.env" >&2
  exit 1
fi
if [[ -z "${!KEY_VAR:-}" ]]; then
  echo "Missing ${KEY_VAR} in ${WORKDIR}/.env" >&2
  exit 1
fi
export OPENAI_API_KEY="${!KEY_VAR}"
export MEM0_TELEMETRY=false
export ANONYMIZED_TELEMETRY=False
export LME_PROMPT_CACHE_ENABLED=1
export LME_PROMPT_CACHE_KEY_PREFIX="lme-longmemeval"
export LME_PROMPT_CACHE_LOG=1
export PYTHONUNBUFFERED=1

if [[ ! -d "${BENCH_DIR}" ]]; then
  bash "${WORKDIR}/scripts/setup_mem0_official_benchmarks.sh"
fi

conda activate mem0-lme
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

cat > "${OUTPUT_ROOT}/run_config.txt" <<CONFIG
run_tag=${RUN_TAG}
backend=${MEM0_OFFICIAL_BACKEND}
data_file=${DATA_FILE}
offset=${OFFSET}
limit=${LIMIT}
answerer_model=${ANSWERER_MODEL}
top_k=${TOP_K}
answer_cutoff=${ANSWER_CUTOFF}
enable_cf_wrapper=${ENABLE_CF_WRAPPER}
cf_rule_mode=${CF_RULE_MODE}
cf_target_scope=${CF_TARGET_SCOPE}
cf_max_writes=${CF_MAX_WRITES}
bench_dir=${BENCH_DIR}
bench_ref=${BENCH_REF}
CONFIG

cat <<INFO
============================================================
Mem0 official provenance smoke
output_root=${OUTPUT_ROOT}
backend=${MEM0_OFFICIAL_BACKEND}
data_file=${DATA_FILE}
offset=${OFFSET} limit=${LIMIT}
answerer_model=${ANSWERER_MODEL}
top_k=${TOP_K} answer_cutoff=${ANSWER_CUTOFF}
cf_rule_mode=${CF_RULE_MODE} cf_max_writes=${CF_MAX_WRITES}
Progress bars should appear below for baseline and CF ingestion.
============================================================
INFO

python "${WORKDIR}/mem0_official_longmemeval_bridge/run_infer.py" \
  --memory-benchmarks-dir "${BENCH_DIR}" \
  --memory-benchmarks-ref "${BENCH_REF}" \
  --longmemeval-file "${DATA_FILE}" \
  --out-jsonl "${OUTPUT_ROOT}/preds_mem0_official_${RUN_TAG}_s1.jsonl" \
  --trace-jsonl "${OUTPUT_ROOT}/preds_mem0_official_${RUN_TAG}_s1.trace.jsonl" \
  --project-name "mem0-official-provenance-smoke" \
  --backend "${MEM0_OFFICIAL_BACKEND}" \
  --answerer-model "${ANSWERER_MODEL}" \
  --top-k "${TOP_K}" \
  --answer-cutoff "${ANSWER_CUTOFF}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET}" \
  --cf-rule-mode "${CF_RULE_MODE}" \
  --enable-cf-wrapper \
  --cf-target-scope "${CF_TARGET_SCOPE}" \
  --cf-max-writes "${CF_MAX_WRITES}" \
  --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
  --show-ingest-progress \
  > >(tee "${LOG_ROOT}/mem0_official_provenance_smoke.out") \
  2> >(tee "${LOG_ROOT}/mem0_official_provenance_smoke.err" >&2)

RUN_DIR="${OUTPUT_ROOT}" python - <<'PY'
import glob
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
audit_paths = sorted(run_dir.glob("*.trace.audit_queries.jsonl"))
cf_query_paths = sorted(run_dir.glob("*.trace.cf_queries.jsonl"))
cf_run_paths = sorted(run_dir.glob("*.trace.cf_runs.jsonl"))
trace_paths = sorted(run_dir.glob("*.trace.jsonl"))
print("\n============================================================")
print("Smoke provenance summary")
print("run_dir:", run_dir)
for label, paths in [("audit_queries", audit_paths), ("cf_queries", cf_query_paths), ("cf_runs", cf_run_paths), ("trace", trace_paths)]:
    print(label + ":", paths[0] if paths else "MISSING")

def read_first(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None

baseline = read_first(audit_paths[0] if audit_paths else None)
cf_summary = read_first(cf_query_paths[0] if cf_query_paths else None)
if baseline:
    print("\nBaseline query:")
    print("question_id:", baseline.get("question_id"))
    print("retrieved_items:", len(baseline.get("retrieved_items") or []))
    print("prompt_items:", len(baseline.get("prompt_items") or []))
    print("bridge_items:", len(baseline.get("bridge_items") or []))
    print("retrieved_write_ids:", len(baseline.get("retrieved_write_ids") or []))
    print("prompt_write_ids:", len(baseline.get("prompt_write_ids") or []))
    sample_item = (baseline.get("retrieved_items") or baseline.get("prompt_items") or [{}])[0]
    print("sample_source_write_ids:", sample_item.get("source_write_ids"))
    print("sample_source_form:", sample_item.get("source_form"))
if cf_summary:
    print("\nCF summary:")
    for key in [
        "retrieved_item_coverage",
        "prompt_item_coverage",
        "baseline_retrieval_correct",
        "baseline_exposure_correct",
        "rollback_gini",
        "rollback_mean_influence",
        "query_dominance_label",
        "cf_run_count",
    ]:
        print(f"{key}:", cf_summary.get(key))
print("============================================================")
PY
