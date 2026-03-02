#!/bin/bash -l
#SBATCH --job-name=probe_local_openai
#SBATCH --partition=msismall
#SBATCH --constraint=genoa
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "===== NODE ====="
hostname
date

echo "===== LISTEN 8000/8001/8002 ====="
ss -ltnp | rg ':(8000|8001|8002)\b' || echo "No local listeners on 8000/8001/8002"

echo "===== /v1/models CHECK ====="
for p in 8000 8001 8002; do
  echo "--- localhost:${p} ---"
  if curl -sS -m 5 -D "/tmp/h_${SLURM_JOB_ID}_${p}.txt" "http://localhost:${p}/v1/models" -o "/tmp/b_${SLURM_JOB_ID}_${p}.json"; then
    echo "[HEADERS]"
    sed -n '1,20p' "/tmp/h_${SLURM_JOB_ID}_${p}.txt"
    echo "[BODY]"
    sed -n '1,60p' "/tmp/b_${SLURM_JOB_ID}_${p}.json"
  else
    echo "curl failed (service not reachable)"
  fi
done

echo "===== PROCESS HINT ====="
ps -ef | rg 'vllm|uvicorn|fastapi|litellm|ollama|tgi|text-generation|proxy|nginx|caddy|8000|8001|8002' | rg -v 'rg ' || true

echo "===== DONE ====="
