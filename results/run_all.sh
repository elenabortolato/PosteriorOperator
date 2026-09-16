#!/usr/bin/env bash
# Regenerate every captured result in this directory.
#
# Run from the repository root:  bash results/run_all.sh
# Takes roughly 40 minutes in total on 4 CPU cores; the per-script cost is
# dominated by network training, and 07 is the slowest per unit of output
# because it refits the model 84 times to measure a sampling distribution.
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p results
status=0

for script in examples/lfi/0*.py; do
    name="$(basename "${script%.py}")"
    out="results/${name}.txt"
    printf '%-48s -> %s\n' "$script" "$out"
    start=$SECONDS
    if python "$script" > "$out" 2>&1; then
        printf '  ok (%ds)\n' "$((SECONDS - start))"
    else
        printf '  FAILED (%ds) -- see %s\n' "$((SECONDS - start))" "$out"
        status=1
    fi
done

exit "$status"
