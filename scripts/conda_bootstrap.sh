#!/usr/bin/env bash

source_conda_sh() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local exe_dir
    exe_dir="$(cd "$(dirname "${CONDA_EXE}")" && pwd)"
    local candidate="${exe_dir%/bin}/etc/profile.d/conda.sh"
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      return 0
    fi
  fi

  if command -v conda >/dev/null 2>&1; then
    local conda_base=""
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "${conda_base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi

  local candidate=""
  for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      return 0
    fi
  done

  echo "Could not locate conda.sh. Checked CONDA_EXE, 'conda info --base', and common install paths." >&2
  return 1
}
