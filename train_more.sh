#!/usr/bin/env bash

# set parallelism using PARALLEL_JOBS env var or first positional arg (default=2)
# configure retries using RETRIES env var (default=1)
# configure sleep between retries using SLEEP_BETWEEN_RETRIES env var (default=5
# edit COMMANDS array below to define the jobs

set -u

# -------- configuration --------
JOBS=${1:-${PARALLEL_JOBS:-2}}          # parallel workers (positional arg wins, then env)
RETRIES=${RETRIES:-1}                  # number of retries per command (0 = no retry)
SLEEP_BETWEEN_RETRIES=${SLEEP_BETWEEN_RETRIES:-5}
LOG_ROOT=${LOG_ROOT:-logs/train_more}

# -------- environment --------
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

timestamp=$(date +"%Y%m%d-%H%M%S")
LOG_DIR="${LOG_ROOT}/${timestamp}"
mkdir -p "$LOG_DIR"

# -------- workload definition --------
COMMANDS=(
  'python3 train_images.py --base "MPPCA"  --flow "OTCFM" --dataset "celeba"        --epochs 50'
  'python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "celeba"        --epochs 50'
  'python3 train_images.py --base "MPPCA"  --flow "VPCFM" --dataset "celeba"        --epochs 50'
  'python3 train_images.py --base "Normal" --flow "VPCFM" --dataset "celeba"        --epochs 50'

  'python3 train_images.py --base "MPPCA"  --flow "OTCFM" --dataset "celeba-64x64"  --epochs 50'
  'python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "celeba-64x64"  --epochs 50'
  'python3 train_images.py --base "MPPCA"  --flow "VPCFM" --dataset "celeba-64x64"  --epochs 50'
  'python3 train_images.py --base "Normal" --flow "VPCFM" --dataset "celeba-64x64"  --epochs 50'

  'python3 train_images.py --base "MPPCA"  --flow "OTCFM" --dataset "celeba-128x128" --epochs 50'
  'python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "celeba-128x128" --epochs 50'
  'python3 train_images.py --base "MPPCA"  --flow "VPCFM" --dataset "celeba-128x128" --epochs 50'
  'python3 train_images.py --base "Normal" --flow "VPCFM" --dataset "celeba-128x128" --epochs 50'

  'python3 train_images.py --base "MPPCA"  --flow "OTCFM" --dataset "fashion"       --epochs 100'
  'python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "fashion"       --epochs 100'
  'python3 train_images.py --base "MPPCA"  --flow "VPCFM" --dataset "fashion"       --epochs 100'
  'python3 train_images.py --base "Normal" --flow "VPCFM" --dataset "fashion"       --epochs 100'

  'python3 train_images.py --base "MPPCA"  --flow "OTCFM" --dataset "cifar10"       --epochs 100'
  'python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "cifar10"       --epochs 100'
  'python3 train_images.py --base "MPPCA"  --flow "VPCFM" --dataset "cifar10"       --epochs 100'
  'python3 train_images.py --base "Normal" --flow "VPCFM" --dataset "cifar10"       --epochs 100'

  'python3 train_images.py --base "MPPCA"  --flow "VPCFM" --dataset "fashion" --epochs 100 --n_factors 3'
  'python3 train_images.py --base "MPPCA"  --flow "VPCFM" --dataset "fashion" --epochs 100 --n_factors 9'
  'python3 train_images.py --base "MPPCA"  --flow "VPCFM" --dataset "fashion" --epochs 100 --n_factors 12'
  'python3 train_images.py --base "MPPCA"  --flow "VPCFM" --dataset "fashion" --epochs 100 --n_factors 15'
  'python3 train_images.py --base "MPPCA"  --flow "VPCFM" --dataset "fashion" --epochs 100 --n_factors 18'
)

# -------- helpers --------
log_msg() {
    local level=$1 msg=$2
    printf '%s [%s] %s\n' "$(date +"%Y-%m-%d %H:%M:%S")" "$level" "$msg"
}

run_with_retry() {
    local idx=$1 cmd=$2
    local attempt=0
    local rc=0

    while :; do
        attempt=$((attempt + 1))
        local log_file="$LOG_DIR/$(printf '%02d' "$idx")-attempt${attempt}.log"
        log_msg "START" "cmd #$idx attempt $attempt: $cmd" | tee -a "$log_file"
        bash -lc "$cmd" >>"$log_file" 2>&1
        rc=$?
        if [[ $rc -eq 0 ]]; then
            log_msg "OK" "cmd #$idx succeeded on attempt $attempt" | tee -a "$log_file"
            return 0
        fi

        if (( attempt > RETRIES )); then
            log_msg "FAIL" "cmd #$idx failed after $attempt attempt(s) (rc=$rc)" | tee -a "$log_file"
            echo "$idx,$cmd,$rc" >> "$LOG_DIR/failed.csv"
            return "$rc"
        fi

        log_msg "RETRY" "cmd #$idx rc=$rc; waiting ${SLEEP_BETWEEN_RETRIES}s" | tee -a "$log_file"
        sleep "$SLEEP_BETWEEN_RETRIES"
    done
}

# -------- scheduler --------
log_msg "INFO" "Starting $((${#COMMANDS[@]})) jobs with JOBS=$JOBS, RETRIES=$RETRIES"
echo "log_dir=$LOG_DIR" > "$LOG_DIR/summary.txt"

failures=0
pids=()

for i in "${!COMMANDS[@]}"; do
    # throttle parallelism
    while (( ${#pids[@]} >= JOBS )); do
        if wait "${pids[0]}"; then :; else failures=$((failures + 1)); fi
        pids=("${pids[@]:1}")
    done

    run_with_retry "$i" "${COMMANDS[$i]}" &
    pids+=("$!")
done

# wait for remaining jobs
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else failures=$((failures + 1)); fi
done

log_msg "INFO" "Completed with failures=$failures. Logs: $LOG_DIR"

if (( failures > 0 )); then
    exit 1
fi
