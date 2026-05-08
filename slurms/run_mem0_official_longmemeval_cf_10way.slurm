#!/bin/bash -l
# Official Mem0 memory-benchmarks LongMemEval baseline+rollback-CF run.

#SBATCH --job-name=mem0off_cf
#SBATCH --partition=msilong
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=48g
#SBATCH --tmp=10g
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7751@umn.edu

set -euo pipefail

WORKDIR="/users/9/chen7751/csci8980"
JOB_ID="${SLURM_JOB_ID:-manual_$$}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/LongMemEval}"
cd "${WORKDIR}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

module load parallel

DATE_SUFFIX="$(date +%m_%d_%H_%M)"
RUN_TAG="${RUN_TAG:-s_100_mem0_official_cf_${DATE_SUFFIX}}"
DATA_FILE="${DATA_FILE:-${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_100_balanced_seed8980.json}"
TOTAL_QUESTIONS="${TOTAL_QUESTIONS:-100}"
SHARDS="${SHARDS:-10}"
KEY_SLOTS="${KEY_SLOTS:-10}"
ANSWERER_MODEL="${ANSWERER_MODEL:-gpt-5}"
STAGGER_SECONDS="${STAGGER_SECONDS:-30}"
JOBS="${JOBS:-${SLURM_NTASKS:-${SHARDS}}}"
ENABLE_CF_WRAPPER="${ENABLE_CF_WRAPPER:-1}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-prompt}"
CF_MAX_WRITES="${CF_MAX_WRITES:-3}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"
CF_RULE_MODE="${CF_RULE_MODE:-rollback-only}"
MEM0_OFFICIAL_BACKEND="${MEM0_OFFICIAL_BACKEND:-cloud}"
MEM0_OFFICIAL_TOP_K="${MEM0_OFFICIAL_TOP_K:-200}"
MEM0_OFFICIAL_ANSWER_CUTOFF="${MEM0_OFFICIAL_ANSWER_CUTOFF:-200}"
MEM0_OFFICIAL_MAX_ADD_RETRIES="${MEM0_OFFICIAL_MAX_ADD_RETRIES:-8}"
MEM0_OFFICIAL_ADD_EVENT_RETRIES="${MEM0_OFFICIAL_ADD_EVENT_RETRIES:-3}"
MEM0_OFFICIAL_RETRY_DELAY="${MEM0_OFFICIAL_RETRY_DELAY:-10}"
MEM0_OFFICIAL_REQUEST_TIMEOUT="${MEM0_OFFICIAL_REQUEST_TIMEOUT:-300}"
MEM0_OFFICIAL_EVENT_POLL_TIMEOUT="${MEM0_OFFICIAL_EVENT_POLL_TIMEOUT:-1800}"
COMMANDS_FILE="${LOG_ROOT}/mem0off_cf_${JOB_ID}.commands.txt"
PARALLEL_JOBLOG="${LOG_ROOT}/mem0off_cf_${JOB_ID}.parallel.log"
export OUTPUT_ROOT DATA_FILE RUN_TAG MEM0_OFFICIAL_BACKEND MEM0_OFFICIAL_TOP_K MEM0_OFFICIAL_ANSWER_CUTOFF
export ENABLE_CF_WRAPPER CF_TARGET_SCOPE CF_MAX_WRITES CF_DOMINANCE_THRESHOLD CF_RULE_MODE
export MEM0_OFFICIAL_MAX_ADD_RETRIES MEM0_OFFICIAL_ADD_EVENT_RETRIES MEM0_OFFICIAL_RETRY_DELAY
export MEM0_OFFICIAL_REQUEST_TIMEOUT MEM0_OFFICIAL_EVENT_POLL_TIMEOUT

TASK_SPECS=()
shard_base=$((TOTAL_QUESTIONS / SHARDS))
shard_remainder=$((TOTAL_QUESTIONS % SHARDS))
offset=0
for shard_idx in $(seq 1 "${SHARDS}"); do
  limit="${shard_base}"
  if (( shard_idx <= shard_remainder )); then
    limit=$((limit + 1))
  fi
  key_idx=$(( ((shard_idx - 1) % KEY_SLOTS) + 1 ))
  TASK_SPECS+=("s${shard_idx}|${offset}|${limit}|OPENAI_API_KEY_${key_idx}")
  offset=$((offset + limit))
done

cat <<INFO
============================================================
[$(date)] job=${JOB_ID} launching official Mem0 CF ${SHARDS} shards over ${TOTAL_QUESTIONS} questions
host=$(hostname)
cwd=${WORKDIR}
run_tag=${RUN_TAG}
data_file=${DATA_FILE}
backend=${MEM0_OFFICIAL_BACKEND}
answerer_model=${ANSWERER_MODEL}
top_k=${MEM0_OFFICIAL_TOP_K}
answer_cutoff=${MEM0_OFFICIAL_ANSWER_CUTOFF}
parallel_jobs=${JOBS}
key_slots=${KEY_SLOTS}
enable_cf_wrapper=${ENABLE_CF_WRAPPER}
cf_rule_mode=${CF_RULE_MODE}
cf_target_scope=${CF_TARGET_SCOPE}
cf_max_writes=${CF_MAX_WRITES}
============================================================
INFO

rm -f "${COMMANDS_FILE}" "${PARALLEL_JOBLOG}"

i=0
for spec in "${TASK_SPECS[@]}"; do
  IFS='|' read -r PART_TAG OFFSET LIMIT KEY_VAR <<< "${spec}"
  DELAY=$((i * STAGGER_SECONDS))
  printf "%q " bash "${WORKDIR}/scripts/run_mem0_official_cf_shard.sh" \
    "${WORKDIR}" \
    "${PART_TAG}" \
    "${OFFSET}" \
    "${LIMIT}" \
    "${KEY_VAR}" \
    "${RUN_TAG}" \
    "${ANSWERER_MODEL}" \
    "${DELAY}" \
    "${ENABLE_CF_WRAPPER}" >> "${COMMANDS_FILE}"
  printf "> %q 2> %q\n" \
    "${LOG_ROOT}/mem0off_cf_${JOB_ID}_${PART_TAG}.out" \
    "${LOG_ROOT}/mem0off_cf_${JOB_ID}_${PART_TAG}.err" >> "${COMMANDS_FILE}"
  i=$((i + 1))
done

parallel --jobs "${JOBS}" --joblog "${PARALLEL_JOBLOG}" < "${COMMANDS_FILE}"

echo "[$(date)] Parallel shards completed; merging and aggregating official Mem0 CF artifacts."

mapfile -t PRED_FILES < <(
  find "${OUTPUT_ROOT:-${WORKDIR}/LongMemEval}" -maxdepth 1 -type f \
    -name "preds_mem0_official_${RUN_TAG}_s*.jsonl" \
    ! -name "*.trace*" | sort
)
mapfile -t CF_QUERY_FILES < <(
  find "${OUTPUT_ROOT:-${WORKDIR}/LongMemEval}" -maxdepth 1 -type f \
    -name "preds_mem0_official_${RUN_TAG}_s*.trace.cf_queries.jsonl" | sort
)
mapfile -t CF_RUN_FILES < <(
  find "${OUTPUT_ROOT:-${WORKDIR}/LongMemEval}" -maxdepth 1 -type f \
    -name "preds_mem0_official_${RUN_TAG}_s*.trace.cf_runs.jsonl" | sort
)

if (( ${#PRED_FILES[@]} == 0 || ${#CF_QUERY_FILES[@]} == 0 || ${#CF_RUN_FILES[@]} == 0 )); then
  echo "Missing expected output artifacts for RUN_TAG=${RUN_TAG}" >&2
  echo "pred_files=${#PRED_FILES[@]} cf_query_files=${#CF_QUERY_FILES[@]} cf_run_files=${#CF_RUN_FILES[@]}" >&2
  exit 1
fi

BASELINE_MERGED="${OUTPUT_ROOT}/preds_mem0_official_${RUN_TAG}_baseline_merged.jsonl"
CF_QUERIES_MERGED="${OUTPUT_ROOT}/preds_mem0_official_${RUN_TAG}_merged.trace.cf_queries.jsonl"
CF_RUNS_MERGED="${OUTPUT_ROOT}/preds_mem0_official_${RUN_TAG}_merged.trace.cf_runs.jsonl"
AFTER_CF_MERGED="${OUTPUT_ROOT}/preds_mem0_official_${RUN_TAG}_after_cf_merged.jsonl"
METRICS_JSON="${OUTPUT_ROOT}/mem0_official_${RUN_TAG}_cf_metrics.json"
QUERY_MATRIX_JSON="${OUTPUT_ROOT}/mem0_official_${RUN_TAG}_query_matrix.json"
QUERY_MATRIX_ROWS="${OUTPUT_ROOT}/mem0_official_${RUN_TAG}_query_matrix_rows.jsonl"

cat "${PRED_FILES[@]}" > "${BASELINE_MERGED}"
cat "${CF_QUERY_FILES[@]}" > "${CF_QUERIES_MERGED}"
cat "${CF_RUN_FILES[@]}" > "${CF_RUNS_MERGED}"

python "${WORKDIR}/scripts/aggregate_cf_metrics.py" \
  "${CF_QUERY_FILES[@]}" \
  --out-json "${METRICS_JSON}"

python "${WORKDIR}/scripts/aggregate_cf_query_matrix.py" \
  --cf-queries "${CF_QUERIES_MERGED}" \
  --cf-runs "${CF_RUNS_MERGED}" \
  --out-json "${QUERY_MATRIX_JSON}" \
  --out-queries "${QUERY_MATRIX_ROWS}"

python "${WORKDIR}/scripts/export_cf_eval_predictions.py" \
  --cf-queries "${CF_QUERY_FILES[@]}" \
  --cf-runs "${CF_RUN_FILES[@]}" \
  --out "${AFTER_CF_MERGED}" \
  --mode max_influence

python - "${TOTAL_QUESTIONS}" "${BASELINE_MERGED}" "${CF_QUERIES_MERGED}" "${CF_RUNS_MERGED}" "${METRICS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

expected = int(sys.argv[1])
baseline_path = Path(sys.argv[2])
cf_queries_path = Path(sys.argv[3])
cf_runs_path = Path(sys.argv[4])
metrics_path = Path(sys.argv[5])

def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

baseline_n = count_jsonl(baseline_path)
cf_query_n = count_jsonl(cf_queries_path)
cf_run_n = count_jsonl(cf_runs_path)
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
agent = metrics.get("mem0_official")
if not agent:
    raise SystemExit("metrics JSON does not contain mem0_official")
required = [
    "answer_flip_rate_mean",
    "query_fragility_rate",
    "abstention_flip_rate_mean",
    "mean_answer_distance",
    "max_answer_distance_per_query_mean",
    "mean_influence_per_query",
    "rollback_gini_mean",
    "retrieved_item_coverage_mean",
    "prompt_item_coverage_mean",
    "confusion_matrix",
    "etdl_survival_curve",
]
missing = [key for key in required if key not in agent]
if missing:
    raise SystemExit(f"metrics missing required currentplan fields: {missing}")
if baseline_n != expected:
    raise SystemExit(f"baseline merged row count mismatch: got {baseline_n}, expected {expected}")
if cf_query_n != expected:
    raise SystemExit(f"cf query summary row count mismatch: got {cf_query_n}, expected {expected}")
if cf_run_n < expected:
    raise SystemExit(f"too few cf runs: got {cf_run_n}, expected at least {expected}")
if agent.get("primary_retrieval_schema_rows") != expected:
    raise SystemExit(
        f"primary retrieval schema incomplete: got {agent.get('primary_retrieval_schema_rows')}, expected {expected}"
    )
print(
    json.dumps(
        {
            "validated": True,
            "baseline_rows": baseline_n,
            "cf_query_rows": cf_query_n,
            "cf_run_rows": cf_run_n,
            "metrics_path": str(metrics_path),
        },
        indent=2,
    )
)
PY

cat <<INFO
============================================================
[$(date)] Official Mem0 CF postprocess complete
baseline_merged=${BASELINE_MERGED}
after_cf_merged=${AFTER_CF_MERGED}
cf_queries_merged=${CF_QUERIES_MERGED}
cf_runs_merged=${CF_RUNS_MERGED}
metrics_json=${METRICS_JSON}
query_matrix_json=${QUERY_MATRIX_JSON}
query_matrix_rows=${QUERY_MATRIX_ROWS}
parallel_joblog=${PARALLEL_JOBLOG}
============================================================
INFO
