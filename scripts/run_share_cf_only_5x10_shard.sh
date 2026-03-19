#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 13 ]]; then
  echo "Usage: $0 <workdir> <baseline_tag> <cf_tag> <shard_name> <offset> <limit> <key_var> <llm_model> <openai_base_url> <start_delay_s> <cf_target_scope> <cf_max_writes> <cf_dominance_threshold>" >&2
  exit 2
fi

WORKDIR="$1"
BASELINE_TAG="$2"
CF_TAG="$3"
SHARD_NAME="$4"
OFFSET="$5"
LIMIT="$6"
KEY_VAR="$7"
LLM_MODEL="$8"
OPENAI_BASE_URL="$9"
START_DELAY_S="${10}"
CF_TARGET_SCOPE="${11}"
CF_MAX_WRITES="${12}"
CF_DOMINANCE_THRESHOLD="${13}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate share-lme

export ANONYMIZED_TELEMETRY=False
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

load_env_file() {
  local env_file="$1"
  if [[ ! -f "${env_file}" ]]; then
    return 0
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    line="${line#export }"
    if [[ "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
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
KEY_VALUE="${!KEY_VAR:-${OPENAI_API_KEY:-}}"
if [[ -z "${KEY_VALUE}" ]]; then
  echo "Missing OpenAI key: checked ${KEY_VAR}, then OPENAI_API_KEY." >&2
  exit 1
fi
export OPENAI_API_KEY="${KEY_VALUE}"

if [[ "${START_DELAY_S}" != "0" ]]; then
  sleep "${START_DELAY_S}"
fi

DATA_FILE="${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json"
SRC_Q1="${WORKDIR}/LongMemEval/preds_share_s_50_unifiedqa_${BASELINE_TAG}_p1.trace.audit_queries.jsonl"
SRC_W1="${WORKDIR}/LongMemEval/preds_share_s_50_unifiedqa_${BASELINE_TAG}_p1.trace.audit_writes.jsonl"
SRC_Q2="${WORKDIR}/LongMemEval/preds_share_s_50_unifiedqa_${BASELINE_TAG}_p2.trace.audit_queries.jsonl"
SRC_W2="${WORKDIR}/LongMemEval/preds_share_s_50_unifiedqa_${BASELINE_TAG}_p2.trace.audit_writes.jsonl"

for path in "${SRC_Q1}" "${SRC_W1}" "${SRC_Q2}" "${SRC_W2}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing baseline audit artifact: ${path}" >&2
    exit 1
  fi
done

RUNTIME_DIR="${WORKDIR}/cf_only_runtime/${CF_TAG}/share_${SHARD_NAME}"
mkdir -p "${RUNTIME_DIR}"
SUBSET_Q="${RUNTIME_DIR}/baseline_subset.audit_queries.jsonl"
SUBSET_W="${RUNTIME_DIR}/baseline_subset.audit_writes.jsonl"
OUTPUT_TRACE_BASE="${WORKDIR}/LongMemEval/preds_share_s_50_unifiedqa_${BASELINE_TAG}_${SHARD_NAME}.trace.jsonl"

python - <<'PY' "${DATA_FILE}" "${OFFSET}" "${LIMIT}" "${SRC_Q1}" "${SRC_W1}" "${SRC_Q2}" "${SRC_W2}" "${SUBSET_Q}" "${SUBSET_W}"
import json, sys
from pathlib import Path

data_file = Path(sys.argv[1])
offset = int(sys.argv[2])
limit = int(sys.argv[3])
src_q1 = Path(sys.argv[4])
src_w1 = Path(sys.argv[5])
src_q2 = Path(sys.argv[6])
src_w2 = Path(sys.argv[7])
out_q = Path(sys.argv[8])
out_w = Path(sys.argv[9])

data = json.loads(data_file.read_text(encoding='utf-8'))
subset = data[offset:offset+limit]
qid_order = [item['question_id'] for item in subset]
qid_set = set(qid_order)

query_records = {}
for src in (src_q1, src_q2):
    for line in src.open(encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        qid = rec.get('question_id')
        if qid in qid_set and qid not in query_records:
            query_records[qid] = rec

write_records = []
for src in (src_w1, src_w2):
    for line in src.open(encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get('question_id') in qid_set:
            write_records.append(rec)

missing = [qid for qid in qid_order if qid not in query_records]
if missing:
    raise SystemExit(f'Missing baseline queries for qids: {missing}')

out_q.parent.mkdir(parents=True, exist_ok=True)
out_w.parent.mkdir(parents=True, exist_ok=True)
with out_q.open('w', encoding='utf-8') as f:
    for qid in qid_order:
        f.write(json.dumps(query_records[qid], ensure_ascii=False) + '\n')
with out_w.open('w', encoding='utf-8') as f:
    for rec in write_records:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
PY

echo "============================================================"
echo "[$(date)] SHARE CF-only shard=${SHARD_NAME} offset=${OFFSET} limit=${LIMIT}"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "baseline_tag=${BASELINE_TAG}"
echo "cf_tag=${CF_TAG}"
echo "key_var=${KEY_VAR}"
echo "cf_target_scope=${CF_TARGET_SCOPE}"
echo "cf_max_writes=${CF_MAX_WRITES}"
echo "cf_dominance_threshold=${CF_DOMINANCE_THRESHOLD}"
echo "baseline_subset_queries=${SUBSET_Q}"
echo "baseline_subset_writes=${SUBSET_W}"
echo "============================================================"

python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
  --agent share \
  --share-dir "${WORKDIR}/SHARE" \
  --longmemeval-file "${DATA_FILE}" \
  --baseline-trace-jsonl "${OUTPUT_TRACE_BASE}" \
  --baseline-audit-queries "${SUBSET_Q}" \
  --baseline-audit-writes "${SUBSET_W}" \
  --cf-tag "${CF_TAG}" \
  --llm-model "${LLM_MODEL}" \
  --openai-base-url "${OPENAI_BASE_URL}" \
  --cf-target-scope "${CF_TARGET_SCOPE}" \
  --cf-max-writes "${CF_MAX_WRITES}" \
  --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
  --cf-rule-mode rollback-only \
  --runtime-dir "${RUNTIME_DIR}"

echo "[$(date)] Completed SHARE CF-only shard=${SHARD_NAME}"
