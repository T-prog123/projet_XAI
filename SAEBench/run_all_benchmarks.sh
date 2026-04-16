#!/usr/bin/env bash

set -u

mkdir -p logs

scripts=(
  "sae/benchmark_sae_absorption.py"
  "sae/benchmark_sae_core.py"
  "sae/benchmark_sae_sparse_probing.py"

  "sst/benchmark_sst_absorption.py"
  "sst/benchmark_sst_core.py"
  "sst/benchmark_sst_sparse_probing.py"

  "st/benchmark_st_absorption.py"
  "st/benchmark_st_core.py"
  "st/benchmark_st_sparse_probing.py"
)

successes=()
failures=()

for script in "${scripts[@]}"; do
  name=$(basename "$script" .py)
  log_file="logs/${name}.log"

  echo "========================================"
  echo "Running: $script"
  echo "Log: $log_file"
  echo "========================================"

  python "$script" >"$log_file" 2>&1
  exit_code=$?

  if [ $exit_code -eq 0 ]; then
    echo "[OK] $script"
    successes+=("$script")
  else
    echo "[FAIL] $script (exit code $exit_code)"
    failures+=("$script")
  fi

  echo
done

echo "========================================"
echo "DONE"
echo "========================================"

echo "Successful runs: ${#successes[@]}"
for s in "${successes[@]}"; do
  echo "  [OK]   $s"
done

echo
echo "Failed runs: ${#failures[@]}"
for s in "${failures[@]}"; do
  echo "  [FAIL] $s"
done