#!/bin/bash
#SBATCH --job-name=softcot_train_eval
#SBATCH --account=def-sdrew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=logs/softcot/softcot_train_eval_%j.out
#SBATCH --error=logs/softcot/softcot_train_eval_%j.err

set -euo pipefail

module load StdEnv/2023
module load python/3.13
module load gcc arrow/21.0.0
source /home/jyang001/jyang001/projects/envs/quin/bin/activate

HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "[HF] ERROR: HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set."
  exit 1
fi
HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_TOKEN HUGGINGFACE_HUB_TOKEN

echo "[HF] token prefix: ${HF_TOKEN:0:8}"
python - <<'PY'
import os
from huggingface_hub import HfApi, hf_hub_download
t = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
print("[HF] env token present:", bool(t), "prefix:", (t or "")[:8])
api = HfApi(token=t)
print("[HF] whoami:", api.whoami()["name"])
print("[HF] sha:", api.model_info("meta-llama/Llama-3.2-1B-Instruct", timeout=10).sha)
cfg = hf_hub_download("meta-llama/Llama-3.2-1B-Instruct", "config.json", token=t)
print("[HF] config path:", cfg)
PY

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/jyang001/jyang001/projects/SoftCoT}}"
HOME_PROJECT_ROOT="${HOME_PROJECT_ROOT:-/home/jyang001/jyang001/projects/SoftCoT}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/jyang001/scratch}"

SOFTCOT_DATA_DIR="${SOFTCOT_DATA_DIR:-$PROJECT_ROOT/data}"
SOFTCOT_RESULTS_DIR="${SOFTCOT_RESULTS_DIR:-$SCRATCH_ROOT/SoftCoT/results}"
SOFTCOT_LOGS_DIR="${SOFTCOT_LOGS_DIR:-$SCRATCH_ROOT/SoftCoT/train_logs}"
SOFTCOT_CKPT_DIR="${SOFTCOT_CKPT_DIR:-$SCRATCH_ROOT/SoftCoT}"

export SOFTCOT_DATA_DIR SOFTCOT_RESULTS_DIR SOFTCOT_LOGS_DIR SOFTCOT_CKPT_DIR

mkdir -p "$PROJECT_ROOT/logs/softcot" "$SOFTCOT_RESULTS_DIR" "$SOFTCOT_LOGS_DIR" "$SOFTCOT_CKPT_DIR"
cd "$PROJECT_ROOT"

BASE_MODEL_ID="${BASE_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
ASSIST_MODEL_ID="${ASSIST_MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
BASE_MODEL_NAME="${BASE_MODEL_ID##*/}"
ASSIST_MODEL_NAME="${ASSIST_MODEL_ID##*/}"

TASK_NAME_RAW="${TASK_NAME:-gsm8k}"
TASK_NAME_NORMALIZED="${TASK_NAME_RAW,,}"
case "$TASK_NAME_NORMALIZED" in
  strategyqa|strategy-qa)
    TASK_NAME="strategyqa"
    ;;
  aqua|aqua-rat|aqua_rat)
    TASK_NAME="aqua"
    ;;
  gsm8k|asdiv-aug|du)
    TASK_NAME="$TASK_NAME_NORMALIZED"
    ;;
  *)
    echo "Unsupported TASK_NAME: $TASK_NAME_RAW" >&2
    echo "Supported values: gsm8k, strategyqa, asdiv-aug, aqua, du" >&2
    exit 2
    ;;
esac

NUM_THOUGHT_TOKENS="${NUM_THOUGHT_TOKENS:-2}"
TEST_K="${TEST_K:-0}"
K_SHOT="${K_SHOT:-0}"
SEED="${SEED:-42}"
EVAL_ONLY="${EVAL_ONLY:-false}"
PARAMS_FILE_NAME="${PARAMS_FILE_NAME:-}"
GSM8K_TRAIN_FILE="${GSM8K_TRAIN_FILE:-}"
GSM8K_DEV_FILE="${GSM8K_DEV_FILE:-}"
GSM8K_TEST_FILE="${GSM8K_TEST_FILE:-}"

OUTPUT_NAME="${OUTPUT_NAME:-softcot-run}"
BATCH_SIZE="${BATCH_SIZE:-2}"
N_EPOCHS="${N_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
OPTIM="${OPTIM:-adamw_torch}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BF16="${BF16:-true}"
PAD_TO_MAX="${PAD_TO_MAX:-false}"
TUNE_ASSISTANT="${TUNE_ASSISTANT:-true}"
TUNE_BASE="${TUNE_BASE:-false}"

RUNS="${RUNS:-1}"
RUN_START="${RUN_START:-1}"
OUTPUT_BASE="${OUTPUT_NAME}"

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || (( RUNS < 1 )); then
  echo "RUNS must be a positive integer; got: $RUNS" >&2
  exit 2
fi
if ! [[ "$RUN_START" =~ ^[0-9]+$ ]] || (( RUN_START < 1 )); then
  echo "RUN_START must be a positive integer; got: $RUN_START" >&2
  exit 2
fi

resolve_dataset_path() {
  local base_dir="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo ""
    return
  fi
  if [[ "$value" = /* ]]; then
    echo "$value"
  else
    echo "${base_dir%/}/$value"
  fi
}

if [[ "$TASK_NAME" == "gsm8k" ]]; then
  GSM8K_ROOT="${SOFTCOT_DATA_DIR%/}/gsm8k"
  if [[ ! -d "$GSM8K_ROOT" && -d "${SOFTCOT_DATA_DIR%/}/GSM8K" ]]; then
    GSM8K_ROOT="${SOFTCOT_DATA_DIR%/}/GSM8K"
  fi
  if [[ -n "$GSM8K_TRAIN_FILE" ]]; then
    export SOFTCOT_GSM8K_TRAIN="$(resolve_dataset_path "$GSM8K_ROOT" "$GSM8K_TRAIN_FILE")"
  fi
  if [[ -n "$GSM8K_DEV_FILE" ]]; then
    export SOFTCOT_GSM8K_DEV="$(resolve_dataset_path "$GSM8K_ROOT" "$GSM8K_DEV_FILE")"
  fi
  if [[ -n "$GSM8K_TEST_FILE" ]]; then
    export SOFTCOT_GSM8K_TEST="$(resolve_dataset_path "$GSM8K_ROOT" "$GSM8K_TEST_FILE")"
  fi
  if [[ -n "${SOFTCOT_GSM8K_TRAIN:-}" && -n "${SOFTCOT_GSM8K_TEST:-}" && -z "${SOFTCOT_GSM8K_DEV:-}" ]]; then
    export SOFTCOT_GSM8K_DEV="$SOFTCOT_GSM8K_TEST"
  fi
fi

N_EPOCHS_TAG="$(python -c "print(float('${N_EPOCHS}'))")"

ASSIST_TUNE_FLAG=()
BASE_TUNE_FLAG=()
BF16_FLAG=(--bf16)
PAD_FLAG=(--no_pad_to_max)
MAX_SAMPLES_ARG=()

if [[ "${TUNE_ASSISTANT,,}" == "true" || "${TUNE_ASSISTANT}" == "1" ]]; then
  ASSIST_TUNE_FLAG+=(--tune_assistant_model)
fi
if [[ "${TUNE_BASE,,}" == "true" || "${TUNE_BASE}" == "1" ]]; then
  BASE_TUNE_FLAG+=(--tune_base_model)
fi
if [[ "${BF16,,}" == "false" || "${BF16}" == "0" ]]; then
  BF16_FLAG=(--no_bf16)
fi
if [[ "${PAD_TO_MAX,,}" == "true" || "${PAD_TO_MAX}" == "1" ]]; then
  PAD_FLAG=(--pad_to_max)
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLES_ARG=(--max_samples "$MAX_SAMPLES")
fi

for offset in $(seq 0 $((RUNS - 1))); do
  run_idx=$((RUN_START + offset))
  RUN_TAG="run${run_idx}"

  if (( RUNS == 1 )); then
    RUN_RESULTS_DIR="$SOFTCOT_RESULTS_DIR"
    RUN_LOGS_DIR="$SOFTCOT_LOGS_DIR"
    RUN_CKPT_BASE="$SOFTCOT_CKPT_DIR"
    OUTPUT_NAME_RUN="$OUTPUT_BASE"
  else
    RUN_RESULTS_DIR="${SOFTCOT_RESULTS_DIR%/}/${RUN_TAG}"
    RUN_LOGS_DIR="${SOFTCOT_LOGS_DIR%/}/${RUN_TAG}"
    RUN_CKPT_BASE="${SOFTCOT_CKPT_DIR%/}/${RUN_TAG}"
    OUTPUT_NAME_RUN="${OUTPUT_BASE}-${RUN_TAG}"
  fi

  mkdir -p "$RUN_RESULTS_DIR" "$RUN_LOGS_DIR" "$RUN_CKPT_BASE"

  # Export per-run dirs so downstream code can honor them.
  export CKPT_BASE="$RUN_CKPT_BASE"
  export RESULTS_BASE="$RUN_RESULTS_DIR"
  export TRAIN_LOG_BASE="$RUN_LOGS_DIR"

  CKPT_DIR="$RUN_CKPT_BASE/${OUTPUT_NAME_RUN}-${TASK_NAME}-${N_EPOCHS_TAG}-${NUM_THOUGHT_TOKENS}-${BASE_MODEL_NAME}-${ASSIST_MODEL_NAME}"
  PROJ_BIN="$CKPT_DIR/projection.bin"
  PRED_FILE="$RUN_RESULTS_DIR/predictions-${TASK_NAME}.jsonl"

  if [[ "${EVAL_ONLY,,}" == "true" || "${EVAL_ONLY}" == "1" ]]; then
    if [[ -z "$PARAMS_FILE_NAME" ]]; then
      echo "[SoftCoT][${RUN_TAG}] ERROR: EVAL_ONLY is enabled but PARAMS_FILE_NAME is empty." >&2
      exit 2
    fi
    PROJ_BIN="$PARAMS_FILE_NAME"
    echo "[SoftCoT][${RUN_TAG}] EVAL_ONLY mode; skipping training."
  else
    echo "[SoftCoT][${RUN_TAG}] Starting training for task=${TASK_NAME}..."
    python train_softcot.py \
      --large_model_id "$BASE_MODEL_ID" \
      --small_model_id "$ASSIST_MODEL_ID" \
      --task_name "$TASK_NAME" \
      --output_name "$OUTPUT_NAME_RUN" \
      --batch_size "$BATCH_SIZE" \
      --num_thought_tokens "$NUM_THOUGHT_TOKENS" \
      --n_epochs "$N_EPOCHS" \
      --learning_rate "$LEARNING_RATE" \
      --weight_decay "$WEIGHT_DECAY" \
      --grad_accum "$GRAD_ACCUM" \
      --warmup_ratio "$WARMUP_RATIO" \
      --logging_steps "$LOGGING_STEPS" \
      --save_steps "$SAVE_STEPS" \
      --save_strategy "$SAVE_STRATEGY" \
      --optim "$OPTIM" \
      --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
      --max_grad_norm "$MAX_GRAD_NORM" \
      --seed "$SEED" \
      --max_length "$MAX_LENGTH" \
      --k_shot "$K_SHOT" \
      "${MAX_SAMPLES_ARG[@]}" \
      "${BF16_FLAG[@]}" \
      "${PAD_FLAG[@]}" \
      "${ASSIST_TUNE_FLAG[@]}" \
      "${BASE_TUNE_FLAG[@]}"

    echo "[SoftCoT][${RUN_TAG}] Training finished. Checking projection file at $PROJ_BIN"
    if [[ ! -f "$PROJ_BIN" ]]; then
      echo "[SoftCoT][${RUN_TAG}] ERROR: projection.bin not found at $PROJ_BIN"
      exit 1
    fi
  fi

  if [[ ! -f "$PROJ_BIN" ]]; then
    echo "[SoftCoT][${RUN_TAG}] ERROR: params file not found at $PROJ_BIN"
    exit 1
  fi

  echo "[SoftCoT][${RUN_TAG}] Starting evaluation..."
  python evaluate_softcot.py \
    --task_name "$TASK_NAME" \
    --params_file_name "$PROJ_BIN" \
    --num_thought_tokens "$NUM_THOUGHT_TOKENS" \
    --num_return_sequences 1 \
    --seed "$SEED" \
    --test_k "$TEST_K" \
    --pred_file "$PRED_FILE" \
    "${ASSIST_TUNE_FLAG[@]}" \
    "${BASE_TUNE_FLAG[@]}" \
    --base_model_id "$BASE_MODEL_ID" \
    --assistant_model_id "$ASSIST_MODEL_ID"

  echo "[SoftCoT][${RUN_TAG}] Done. Saved predictions to $PRED_FILE"
done
