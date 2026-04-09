#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_smoke_ltm.sh [--preset wiring|logic] [--agents a,b,c] [--limit N] [--offset N]
                   [--cf-max-writes N] [--baseline-tag TAG] [--cf-tag TAG]
                   [--llm-model MODEL] [--openai-base-url URL] [--key-var ENV_NAME]

Presets:
  wiring  Run the smallest end-to-end smoke test on all 5 agents.
          Defaults: agents=anna,share,memoryos,ldagent,theanine limit=1 cf_max_writes=1

  logic   Run a slightly larger smoke test on the two most important agents.
          Defaults: agents=memoryos,share limit=3 cf_max_writes=3

Examples:
  ./scripts/run_smoke_ltm.sh
  ./scripts/run_smoke_ltm.sh --preset logic
  ./scripts/run_smoke_ltm.sh --agents memoryos,share --limit 2 --cf-max-writes 2
EOF
}

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
PRESET="wiring"
AGENTS=""
LIMIT=""
OFFSET="0"
CF_MAX_WRITES=""
BASELINE_TAG="smoke_$(date +%m%d_%H%M%S)"
CF_TAG=""
LLM_MODEL="${SMOKE_LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
KEY_VAR="OPENAI_API_KEY"
CF_DOMINANCE_THRESHOLD="${SMOKE_CF_DOMINANCE_THRESHOLD:-0.75}"
PART_TAG="smoke"

cd "${WORKDIR}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --agents)
      AGENTS="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --offset)
      OFFSET="$2"
      shift 2
      ;;
    --cf-max-writes)
      CF_MAX_WRITES="$2"
      shift 2
      ;;
    --baseline-tag)
      BASELINE_TAG="$2"
      shift 2
      ;;
    --cf-tag)
      CF_TAG="$2"
      shift 2
      ;;
    --llm-model)
      LLM_MODEL="$2"
      shift 2
      ;;
    --openai-base-url)
      OPENAI_BASE_URL="$2"
      shift 2
      ;;
    --key-var)
      KEY_VAR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${PRESET}" in
  wiring)
    : "${AGENTS:=anna,share,memoryos,ldagent,theanine}"
    : "${LIMIT:=1}"
    : "${CF_MAX_WRITES:=1}"
    ;;
  logic)
    : "${AGENTS:=memoryos,share}"
    : "${LIMIT:=3}"
    : "${CF_MAX_WRITES:=3}"
    ;;
  *)
    echo "Unsupported preset: ${PRESET}" >&2
    exit 2
    ;;
esac

: "${CF_TAG:=cf_${BASELINE_TAG}}"

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

resolve_openai_key() {
  local resolved="${!KEY_VAR:-${OPENAI_API_KEY:-}}"
  if [[ -z "${resolved}" ]]; then
    echo "Missing OpenAI key: checked ${KEY_VAR}, then OPENAI_API_KEY." >&2
    exit 1
  fi
  export OPENAI_API_KEY="${resolved}"
}

load_env_file "${WORKDIR}/.env"
resolve_openai_key

activate_env() {
  local agent="$1"
  case "${agent}" in
    anna) conda activate anna-lme ;;
    share) conda activate share-lme ;;
    memoryos) conda activate memos-lme ;;
    ldagent) conda activate ld-lme ;;
    theanine) conda activate theanine-lme ;;
    *)
      echo "Unknown agent: ${agent}" >&2
      exit 2
      ;;
  esac
}

preflight_agent_import() {
  local agent="$1"
  local module=""
  case "${agent}" in
    anna) module="anna_longmemeval_bridge.run_infer" ;;
    share) module="share_longmemeval_bridge.run_infer" ;;
    memoryos) module="memos_longmemeval_bridge.run_infer" ;;
    ldagent) module="ldagent_longmemeval_bridge.run_infer" ;;
    theanine) module="theanine_longmemeval_bridge.run_infer" ;;
    *)
      echo "Unknown agent for preflight: ${agent}" >&2
      exit 2
      ;;
  esac

  if ! PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}" python -c "import ${module}" >/dev/null 2>&1; then
    echo "Preflight import failed for ${agent} (${module}) in env $(conda info --json | python - <<'PY'\nimport json,sys\nprint(json.load(sys.stdin).get('active_prefix_name','unknown'))\nPY)." >&2
    echo "This usually means the conda env exists but its Python dependencies are not installed." >&2
    if [[ "${agent}" == "anna" ]]; then
      echo "Expected setup is documented in /Users/daqingchen/csci8980/anna_longmemeval_bridge/README.md and /Users/daqingchen/csci8980/anna_longmemeval_bridge/requirements.txt." >&2
    fi
    exit 1
  fi
}

baseline_trace_path() {
  local agent="$1"
  echo "${WORKDIR}/LongMemEval/preds_${agent}_${BASELINE_TAG}_${PART_TAG}.trace.jsonl"
}

baseline_audit_query_path() {
  local agent="$1"
  local trace_path
  trace_path="$(baseline_trace_path "${agent}")"
  echo "${trace_path%.jsonl}.audit_queries.jsonl"
}

cf_run_path() {
  local agent="$1"
  local stem
  stem="$(baseline_trace_path "${agent}")"
  stem="${stem%.jsonl}"
  echo "${stem}.${CF_TAG}.cf_runs.jsonl"
}

cf_query_path() {
  local agent="$1"
  local stem
  stem="$(baseline_trace_path "${agent}")"
  stem="${stem%.jsonl}"
  echo "${stem}.${CF_TAG}.cf_queries.jsonl"
}

matrix_json_path() {
  local agent="$1"
  local stem
  stem="$(baseline_trace_path "${agent}")"
  stem="${stem%.trace.jsonl}"
  echo "${stem}.${CF_TAG}.query_matrix.json"
}

run_baseline() {
  local agent="$1"
  echo "[$(date)] Baseline smoke: agent=${agent} offset=${OFFSET} limit=${LIMIT}"
  activate_env "${agent}"
  preflight_agent_import "${agent}"
  "${WORKDIR}/scripts/run_unifiedqa_shard.sh" \
    "${WORKDIR}" \
    "${agent}" \
    "${PART_TAG}" \
    "${OFFSET}" \
    "${LIMIT}" \
    "${KEY_VAR}" \
    "${BASELINE_TAG}" \
    "${LLM_MODEL}" \
    "${OPENAI_BASE_URL}" \
    "0"

  local audit_query_path
  audit_query_path="$(baseline_audit_query_path "${agent}")"
  if [[ ! -f "${audit_query_path}" ]]; then
    echo "Baseline smoke for ${agent} did not produce ${audit_query_path}." >&2
    exit 1
  fi

  local baseline_count
  baseline_count="$(python - <<'PY' "${audit_query_path}"
import json, sys
count = 0
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("trace_kind") == "baseline_query":
            count += 1
print(count)
PY
)"
  if [[ "${baseline_count}" == "0" ]]; then
    echo "Baseline smoke for ${agent} produced zero successful baseline_query records." >&2
    echo "Most likely causes are an invalid API key or missing upstream dependencies." >&2
    echo "Aborting before CF stage for ${agent}." >&2
    exit 1
  fi
}

run_cf() {
  local agent="$1"
  local trace_path
  local runtime_dir
  trace_path="$(baseline_trace_path "${agent}")"
  runtime_dir="${WORKDIR}/cf_only_runtime/${CF_TAG}/${agent}_${PART_TAG}"

  if [[ ! -f "${trace_path}" ]]; then
    echo "Missing baseline trace for ${agent}: ${trace_path}" >&2
    exit 1
  fi

  echo "[$(date)] CF smoke: agent=${agent} trace=$(basename "${trace_path}")"
  activate_env "${agent}"
  preflight_agent_import "${agent}"

  case "${agent}" in
    anna)
      python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
        --agent anna \
        --anna-agent-dir "${WORKDIR}/AnnaAgent" \
        --longmemeval-file "${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json" \
        --baseline-trace-jsonl "${trace_path}" \
        --cf-tag "${CF_TAG}" \
        --llm-model "${LLM_MODEL}" \
        --openai-base-url "${OPENAI_BASE_URL}" \
        --cf-target-scope prompt \
        --cf-max-writes "${CF_MAX_WRITES}" \
        --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
        --cf-rule-mode rollback-only \
        --runtime-dir "${runtime_dir}"
      ;;
    share)
      python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
        --agent share \
        --share-dir "${WORKDIR}/SHARE" \
        --longmemeval-file "${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json" \
        --baseline-trace-jsonl "${trace_path}" \
        --cf-tag "${CF_TAG}" \
        --llm-model "${LLM_MODEL}" \
        --openai-base-url "${OPENAI_BASE_URL}" \
        --cf-target-scope prompt \
        --cf-max-writes "${CF_MAX_WRITES}" \
        --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
        --cf-rule-mode rollback-only \
        --runtime-dir "${runtime_dir}"
      ;;
    memoryos)
      python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
        --agent memoryos \
        --memoryos-dir "${WORKDIR}/MemoryOS" \
        --longmemeval-file "${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json" \
        --baseline-trace-jsonl "${trace_path}" \
        --cf-tag "${CF_TAG}" \
        --llm-model "${LLM_MODEL}" \
        --openai-base-url "${OPENAI_BASE_URL}" \
        --cf-target-scope prompt \
        --cf-max-writes "${CF_MAX_WRITES}" \
        --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
        --cf-rule-mode rollback-only \
        --runtime-dir "${runtime_dir}"
      ;;
    ldagent)
      python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
        --agent ldagent \
        --ld-agent-dir "${WORKDIR}/LD-Agent" \
        --longmemeval-file "${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json" \
        --baseline-trace-jsonl "${trace_path}" \
        --cf-tag "${CF_TAG}" \
        --llm-model "${LLM_MODEL}" \
        --openai-base-url "${OPENAI_BASE_URL}" \
        --cf-target-scope prompt \
        --cf-max-writes "${CF_MAX_WRITES}" \
        --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
        --cf-rule-mode rollback-only \
        --runtime-dir "${runtime_dir}"
      ;;
    theanine)
      local theanine_repo
      theanine_repo="${WORKDIR}/Theanine_${PART_TAG}_repo"
      if [[ ! -d "${theanine_repo}" ]]; then
        theanine_repo="${WORKDIR}/Theanine"
      fi
      python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
        --agent theanine \
        --theanine-dir "${theanine_repo}" \
        --longmemeval-file "${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json" \
        --baseline-trace-jsonl "${trace_path}" \
        --cf-tag "${CF_TAG}" \
        --llm-model "${LLM_MODEL}" \
        --openai-base-url "${OPENAI_BASE_URL}" \
        --cf-target-scope prompt \
        --cf-max-writes "${CF_MAX_WRITES}" \
        --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
        --cf-rule-mode rollback-only \
        --runtime-dir "${runtime_dir}"
      ;;
  esac
}

run_matrix() {
  local agent="$1"
  local run_path query_path matrix_path
  run_path="$(cf_run_path "${agent}")"
  query_path="$(cf_query_path "${agent}")"
  matrix_path="$(matrix_json_path "${agent}")"

  if [[ ! -f "${run_path}" || ! -f "${query_path}" ]]; then
    echo "Missing CF outputs for ${agent}: ${run_path} / ${query_path}" >&2
    exit 1
  fi

  python "${WORKDIR}/scripts/aggregate_cf_query_matrix.py" \
    --cf-queries "${query_path}" \
    --cf-runs "${run_path}" \
    --out-json "${matrix_path}" >/dev/null

  echo "[$(date)] Matrix written: ${matrix_path}"
}

IFS=',' read -r -a AGENT_LIST <<< "${AGENTS}"

echo "============================================================"
echo "Smoke preset      : ${PRESET}"
echo "Agents            : ${AGENTS}"
echo "Offset            : ${OFFSET}"
echo "Limit             : ${LIMIT}"
echo "CF max writes     : ${CF_MAX_WRITES}"
echo "Baseline tag      : ${BASELINE_TAG}"
echo "CF tag            : ${CF_TAG}"
echo "LLM model         : ${LLM_MODEL}"
echo "OpenAI base URL   : ${OPENAI_BASE_URL}"
echo "============================================================"

for agent in "${AGENT_LIST[@]}"; do
  run_baseline "${agent}"
  run_cf "${agent}"
  run_matrix "${agent}"
done

echo "[$(date)] Smoke run complete."
