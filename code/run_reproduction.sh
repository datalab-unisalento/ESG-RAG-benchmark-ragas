#!/usr/bin/env bash
# Controlled, resumable reproduction of the paper's experiments.
#
# Stages (each resumes from its own manifest if interrupted):
#   1. record environment
#   2. RAG generation (frozen retrieval, fixed decoding)      -> outputs/repro/rag_responses
#   3. RAGAS judging, all seven metrics, three judges          -> outputs/repro/judges
#   4. generation timing, one model loaded at a time           -> outputs/repro/benchmark
#   5. QA consistency check on the 284-pair testset            -> outputs/repro/qa_validation
#
# Run detached so it survives logout:
#   setsid nohup bash code/run_reproduction.sh > outputs/repro/driver.log 2>&1 < /dev/null &
#
# A stage that makes no progress for STALL_SECONDS is stopped, its models are unloaded,
# and the stage is restarted with --resume, up to MAX_TRIES times.

set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python
ROOT=outputs/repro
STALL_SECONDS=900
MAX_TRIES=3
MODELS=(qwen3:4b-instruct nemotron-3.5-lightning:30b cogito:32b gemma4:12b ministral-3:8b gemma3:4b glm-4.7-flash:q4_K_M nemotron-3-nano:4b)

mkdir -p "$ROOT/logs"

log() { echo "[$(date '+%F %T')] $*"; }

record_environment() {
    {
        echo "{"
        echo "  \"git_commit\": \"$(git rev-parse HEAD 2>/dev/null)\","
        echo "  \"git_branch\": \"$(git branch --show-current 2>/dev/null)\","
        echo "  \"date_utc\": \"$(date -u '+%F %T')\","
        echo "  \"host\": \"$(hostname)\","
        echo "  \"gpu\": \"$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)\","
        echo "  \"ollama_version\": \"$(ollama --version 2>/dev/null | tail -1)\","
        echo "  \"python\": \"$($PY --version 2>&1)\""
        echo "}"
    } > "$ROOT/environment.json"
    $PY -m pip freeze > "$ROOT/pip_freeze.txt" 2>/dev/null
    ollama list > "$ROOT/ollama_models.txt" 2>/dev/null
    log "environment recorded in $ROOT/environment.json"
}

# Each stage is a function so that a retry picks up --resume when a manifest exists.
stage_generation() {
    local flag=""
    [ -f "$ROOT/rag_responses/manifest.json" ] && flag="--resume"
    $PY code/02_run_rag_generation.py --n 100 --output-dir "$ROOT/rag_responses" $flag
}

stage_judges() {
    local flag=""
    [ -f "$ROOT/judges/manifest.json" ] && flag="--resume"
    $PY code/04_compare_ragas_judges.py \
        --input-dir "$ROOT/rag_responses" \
        --output-dir "$ROOT/judges" \
        --judges qwen3:4b-instruct nemotron-3.5-lightning:30b cogito:32b \
        $flag
}

stage_timing() {
    local flag=""
    [ -f "$ROOT/benchmark/manifest.json" ] && flag="--resume"
    $PY code/06_benchmark_tokens_latency.py --rag-dir "$ROOT/rag_responses" --n 50 \
        --output-dir "$ROOT/benchmark" $flag
}

stage_validation() {
    local flag=""
    [ -f "$ROOT/qa_validation/manifest.json" ] && flag="--resume"
    $PY code/05_validate_synthetic_testset.py \
        --input outputs/testset/01_esg_testset.csv \
        --corpus-dir data/documents \
        --models gemma4:12b ministral-3:8b nemotron-3.5-lightning:30b \
        --output-dir "$ROOT/qa_validation" $flag
}

unload_all() {
    for m in "${MODELS[@]}"; do ollama stop "$m" >/dev/null 2>&1; done
}

# last_activity LOG DIR: newest modification time (epoch seconds) among the log and the output files.
# Some stages write progress to output files rather than to the log.
last_activity() {
    find "$1" "$2" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1
}

# run_stage NAME FUNCTION OUTPUT_DIR: runs FUNCTION in the background and restarts it on stall or failure.
run_stage() {
    local name=$1 fn=$2 out_dir=$3 log_file="$ROOT/logs/$1.log"
    local try pid rc stalled
    for try in $(seq 1 "$MAX_TRIES"); do
        log "$name: attempt $try"
        "$fn" >> "$log_file" 2>&1 &
        pid=$!
        stalled=0
        while kill -0 "$pid" 2>/dev/null; do
            sleep 60
            if [ $(( $(date +%s) - $(last_activity "$log_file" "$out_dir") )) -gt "$STALL_SECONDS" ]; then
                log "$name: no progress for ${STALL_SECONDS}s, restarting"
                pkill -P "$pid" 2>/dev/null
                kill "$pid" 2>/dev/null
                stalled=1
                break
            fi
        done
        wait "$pid" 2>/dev/null
        rc=$?
        if [ "$stalled" -eq 0 ] && [ "$rc" -eq 0 ]; then
            log "$name: done"
            return 0
        fi
        log "$name: exit code $rc (stalled=$stalled)"
        unload_all
        sleep 30
    done
    log "$name: FAILED after $MAX_TRIES attempts"
    return 1
}

record_environment
run_stage generation stage_generation "$ROOT/rag_responses" || exit 1
run_stage judges stage_judges "$ROOT/judges" || exit 1
run_stage timing stage_timing "$ROOT/benchmark" || exit 1
run_stage validation stage_validation "$ROOT/qa_validation" || exit 1
log "all stages complete"
