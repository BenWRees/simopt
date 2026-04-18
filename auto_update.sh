#!/bin/bash

set -u

ENV_PATH="/Users/benjaminrees/miniconda3/envs/simopt"
DATE=$(date "+%Y-%m-%d %H:%M:%S")
LOG_FILE=$(mktemp)
TMP_DIR=$(mktemp -d)
TEST_FILE="$TMP_DIR/generated_tests.py"

MAX_PARALLEL=4

echo "AUTO-UPDATE LOG - $DATE" > "$LOG_FILE"
echo "---------------------------------" >> "$LOG_FILE"

MODIFIED_FILES=$(git diff --name-only | grep '\.py$')

if [ -z "$MODIFIED_FILES" ]; then
    echo "No modified Python files found."
    exit 0
fi

echo "Running Ruff checks..."

PASSING_FILES=()
FAILURES=0

# --- Step 1: Ruff filtering ---
for FILE in $MODIFIED_FILES; do
    RUFF_OUT=$(conda run -p "$ENV_PATH" ruff check "$FILE" 2>&1)
    if [ $? -ne 0 ]; then
        FAILURES=1
        echo "FAILED RUFF: $FILE" >> "$LOG_FILE"
        echo "$RUFF_OUT" >> "$LOG_FILE"
        echo "---------------------------------" >> "$LOG_FILE"
    else
        PASSING_FILES+=("$FILE")
    fi
done

if [ ${#PASSING_FILES[@]} -eq 0 ]; then
    echo "No files passed Ruff."
    exit 0
fi

echo "Building dependency graph..."

LAYERS=$(printf "%s\n" "${PASSING_FILES[@]}" | python build_dependency_graph.py)

COMMITS_MADE=0

# --- Step 2: Layered pytest execution ---
while IFS= read -r LAYER; do
    echo "Testing layer: $LAYER"

    # Generate pytest file for this layer
    python generate_pytest_wrapper.py "$TEST_FILE" $LAYER

    PYTEST_OUT=$(conda run -p "$ENV_PATH" pytest "$TEST_FILE" -q 2>&1)
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        FAILURES=1
        echo "PYTEST FAILURE (layer): $LAYER" >> "$LOG_FILE"
        echo "$PYTEST_OUT" >> "$LOG_FILE"
        echo "---------------------------------" >> "$LOG_FILE"
    else
        for FILE in $LAYER; do
            git add "$FILE"
            COMMITS_MADE=1
        done
    fi

done <<< "$LAYERS"

# Commit + push
if [ $COMMITS_MADE -eq 1 ]; then
    git commit -m "Auto-Commit batch on $DATE"
    git push origin master
fi

# Email failures
if [ $FAILURES -eq 1 ]; then
    mail -s "AUTO-UPDATE $DATE: Message Log" B.Rees@soton.ac.uk < "$LOG_FILE"
fi

rm -rf "$TMP_DIR"
rm "$LOG_FILE"

echo "Done."