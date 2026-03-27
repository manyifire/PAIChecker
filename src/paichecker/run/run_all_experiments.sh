#!/usr/bin/env bash
# run_all_experiments.sh — run all 20 experiments (5 methods × 4 models) in parallel.
#
# Each model gets its own background process so they hit different API endpoints
# and don't compete on the same rate limit.
#
# Usage:
#   bash run_all_experiments.sh                    # all 5 models
#   bash run_all_experiments.sh gpt claude         # specific models only

set -euo pipefail

# ── Shared config ──────────────────────────────────────────────────────────
export GITHUB_TOKEN="${GITHUB_TOKEN:?Set GITHUB_TOKEN}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"
export MSWEA_SILENT_STARTUP=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INPUT_PATH="$PROJECT_ROOT/data/minisweagent_sample.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/data/outputs"
SUB_AGENT_DIR="$PROJECT_ROOT/data/sub_agent_outputs"
LOG_DIR="$PROJECT_ROOT/logs"
TOTAL_LINES=$(wc -l < "$INPUT_PATH")

export INPUT_PATH OUTPUT_DIR SUB_AGENT_DIR LOG_DIR TOTAL_LINES

mkdir -p "$LOG_DIR"

# ── Model name resolution (bash 3.2 compatible, no associative arrays) ─────
full_model_name() {
    case "$1" in
        gpt)      echo "openai/gpt-5.3-codex" ;;
        claude)   echo "openai/claude-sonnet-4-6" ;;
        gemini)   echo "openai/gemini-3.1-pro-preview" ;;
        qwen)     echo "openai/qwen3.5-plus" ;;
        *)        echo "$1" ;;   # allow passing a full name directly
    esac
}

# ── Per-model worker (runs all 5 methods sequentially) ─────────────────────
run_model() {
    local model="$1"
    local full_name
    full_name="$(full_model_name "$model")"
    local log="$LOG_DIR/${model}.log"

    # Rotate old log
    [ -f "$log" ] && mv "$log" "${log}.prev"

    echo "=== START $(date '+%Y-%m-%d %H:%M:%S') model=$model full=$full_name ===" | tee -a "$log"

    # ── 1. Baselines: zero-shot, few-shot, cot ────────────────────────────
    for method in zero-shot few-shot cot; do
        echo "[$(date +%H:%M:%S)] [$model] $method — start" | tee -a "$log"
        python -m paichecker.run.baselines \
            --input  "$INPUT_PATH" \
            --method "$method" \
            --model  "$model" \
            --output-dir "$OUTPUT_DIR" \
            >> "$log" 2>&1
        echo "[$(date +%H:%M:%S)] [$model] $method — done" | tee -a "$log"
    done

    # ── 2. mini-swe-agent ─────────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] [$model] mini-swe-agent — start" | tee -a "$log"
    mini_output="$OUTPUT_DIR/mini-swe-agent/misalign_outputs_${model}_all.jsonl"
    mkdir -p "$(dirname "$mini_output")"
    for ((i = 0; i < TOTAL_LINES; i++)); do
        python -m paichecker.run.mini_swe_detector \
            --input  "$INPUT_PATH" \
            --index  "$i" \
            --model  "$full_name" \
            --output "$mini_output" \
            >> "$log" 2>&1
    done
    echo "[$(date +%H:%M:%S)] [$model] mini-swe-agent — done" | tee -a "$log"

    # ── 3. multi-agent checker ────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] [$model] multi-agent — start" | tee -a "$log"
    checker_output="$OUTPUT_DIR/checker/misalign_outputs_${model}_all.jsonl"
    mkdir -p "$(dirname "$checker_output")"
    for ((i = 0; i < TOTAL_LINES; i++)); do
        python -m paichecker.run.multi_swe_detector \
            --input      "$INPUT_PATH" \
            --index      "$i" \
            --model      "$full_name" \
            --output     "$checker_output" \
            --output-dir "$SUB_AGENT_DIR" \
            >> "$log" 2>&1
    done
    echo "[$(date +%H:%M:%S)] [$model] multi-agent — done" | tee -a "$log"

    echo "=== DONE $(date '+%Y-%m-%d %H:%M:%S') model=$model ===" | tee -a "$log"
}

export -f run_model full_model_name

# ── Determine which models to run ─────────────────────────────────────────
if [ $# -eq 0 ]; then
    MODELS=(gpt claude gemini qwen)
else
    MODELS=("$@")
fi

# ── Launch one background process per model ───────────────────────────────
PIDS=()
for model in "${MODELS[@]}"; do
    echo "Launching worker: $model  (log → $LOG_DIR/${model}.log)"
    bash -c "run_model '$model'" &
    PIDS+=($!)
done

echo ""
echo "${#PIDS[@]} model workers running in parallel."
echo "Monitor progress:  tail -f $LOG_DIR/<model>.log"
echo ""

# ── Wait for all workers, report results ─────────────────────────────────
FAILED=()
for i in "${!PIDS[@]}"; do
    model="${MODELS[$i]}"
    pid="${PIDS[$i]}"
    if wait "$pid"; then
        echo "[OK]   $model"
    else
        echo "[FAIL] $model  (check $LOG_DIR/${model}.log)"
        FAILED+=("$model")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All experiments finished successfully."
    echo "Run evaluation:  python evaluate.py"
else
    echo "Failed models: ${FAILED[*]}"
    exit 1
fi
