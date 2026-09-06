#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKER="$SCRIPT_DIR/model-checker"
MODEL_DIR="$SCRIPT_DIR/models"

if [ ! -x "$CHECKER" ]; then
    echo "Error: checker not found or not executable: $CHECKER"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: model directory not found: $MODEL_DIR"
    exit 1
fi

total=0
passed=0
failed=0

for model in "$MODEL_DIR"/*.tflite; do
    [ -e "$model" ] || continue

    total=$((total + 1))
    name="$(basename "$model")"

    output=$("$CHECKER" "$model" 2>&1)
    status=$?

    if [ "$status" -eq 0 ]; then
        passed=$((passed + 1))
        printf "PASS  %s\n" "$name"
    else
        failed=$((failed + 1))
        printf "FAIL  %s\n" "$name"

        printf '%s\n' "$output" > "${model%.tflite}.fail.log"
    fi
done

echo
echo "Total: $total  Pass: $passed  Fail: $failed"

exit "$failed"