#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <github-username>" >&2
  exit 2
fi

GH_USER="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FILE="${ROOT_DIR}/LongMemEval/data/longmemeval_s_cleaned_50.json"
BACKUP_DIR="${ROOT_DIR}/data_backups"
BACKUP_FILE="${BACKUP_DIR}/longmemeval_s_cleaned_50.json"

ANNA_URL="https://github.com/${GH_USER}/AnnaAgent.git"
LD_URL="https://github.com/${GH_USER}/LD-Agent.git"
LME_URL="https://github.com/${GH_USER}/LongMemEval.git"
MEMOS_URL="https://github.com/${GH_USER}/MemoryOS.git"
SHARE_URL="https://github.com/${GH_USER}/SHARE.git"

echo "[INFO] Validating fork repos exist and are accessible..."
for url in \
  "${ANNA_URL}" \
  "${LD_URL}" \
  "${LME_URL}" \
  "${MEMOS_URL}" \
  "${SHARE_URL}"; do
  if ! git ls-remote "${url}" >/dev/null 2>&1; then
    echo "[ERROR] Missing or inaccessible fork: ${url}" >&2
    exit 3
  fi
done

mkdir -p "${BACKUP_DIR}"
if [[ -f "${DATA_FILE}" ]]; then
  cp -f "${DATA_FILE}" "${BACKUP_FILE}"
  echo "[INFO] Backed up dataset to ${BACKUP_FILE}"
fi

cat > "${ROOT_DIR}/.gitmodules" <<EOF
[submodule "AnnaAgent"]
	path = AnnaAgent
	url = ${ANNA_URL}
[submodule "LD-Agent"]
	path = LD-Agent
	url = ${LD_URL}
[submodule "LongMemEval"]
	path = LongMemEval
	url = ${LME_URL}
[submodule "MemoryOS"]
	path = MemoryOS
	url = ${MEMOS_URL}
[submodule "SHARE"]
	path = SHARE
	url = ${SHARE_URL}
EOF

cd "${ROOT_DIR}"
git submodule sync --recursive

git -C AnnaAgent remote set-url origin "${ANNA_URL}"
git -C LD-Agent remote set-url origin "${LD_URL}"
git -C LongMemEval remote set-url origin "${LME_URL}"
git -C MemoryOS remote set-url origin "${MEMOS_URL}"
git -C SHARE remote set-url origin "${SHARE_URL}"

if [[ -f "${DATA_FILE}" ]]; then
  git -C LongMemEval add data/longmemeval_s_cleaned_50.json || true
  if ! git -C LongMemEval diff --cached --quiet; then
    git -C LongMemEval commit -m "Add longmemeval_s_cleaned_50 subset for experiments"
  fi
  git -C LongMemEval push -u origin main
fi

git add .gitmodules LongMemEval
if ! git diff --cached --quiet; then
  git commit -m "Point submodules to personal forks and update LongMemEval subset commit"
  git push origin main
fi

echo "[OK] Submodules now point to ${GH_USER} forks."
