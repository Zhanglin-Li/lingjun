#!/bin/bash
# Hidden size sweep runner
# Runs sweep configs sequentially with resume support.
# If interrupted, just re-run — already-completed configs are skipped.

cd "$(dirname "$0")/.."  # ensure we're in project root

LOG="experiments/experiment_log.csv"

# All sweep configs in order (baseline + variants)
CONFIGS=(
    "config/config.yaml"
    "config/hidden_size/s02.yaml"
    "config/hidden_size/s03.yaml"
    "config/hidden_size/s04.yaml"
    "config/hidden_size/s05.yaml"
    "config/hidden_size/s06.yaml"
    "config/hidden_size/s07.yaml"
    "config/hidden_size/s08.yaml"
    "config/hidden_size/s09.yaml"
    "config/hidden_size/s10.yaml"
)

# Count how many experiments already in log (skip header)
if [ -f "$LOG" ]; then
    EXISTING=$(tail -n +2 "$LOG" | wc -l)
else
    EXISTING=0
fi

echo "Found $EXISTING completed experiments in log"

TOTAL=${#CONFIGS[@]}
SKIP=$EXISTING

if [ "$SKIP" -ge "$TOTAL" ]; then
    echo "All $TOTAL configs already completed. Nothing to do."
    exit 0
fi

echo "Running configs $((SKIP+1)) to $TOTAL of $TOTAL"
echo ""

for ((i=SKIP; i<TOTAL; i++)); do
    CFG="${CONFIGS[$i]}"
    NUM=$((i+1))
    echo "========================================="
    echo "[$NUM/$TOTAL] $CFG"
    echo "========================================="

    .venv/bin/python train.py "$CFG"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: $CFG failed with exit code $EXIT_CODE"
        echo "Fix the issue and re-run: bash scripts/run_sweep.sh"
        exit $EXIT_CODE
    fi

    echo ""
done

echo "========================================="
echo "All $TOTAL sweep configs completed!"
echo "========================================="
