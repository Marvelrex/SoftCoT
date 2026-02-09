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

# Hugging Face auth for gated models (pass via sbatch --export)
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "[HF] ERROR: HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set."
  exit 1
fi
HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_TOKEN HUGGINGFACE_HUB_TOKEN

# Pre-flight check: verify token is present and can download gated files
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

PROJECT_ROOT="${PROJECT_ROOT:-/home/jyang001/jyang001/projects/SoftCoT}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/jyang001/scratch}"
CKPT_BASE="${CKPT_BASE:-$SCRATCH_ROOT/SoftCoT/ckpt}"
SOFTCOT_DATA_DIR="${SOFTCOT_DATA_DIR:-$PROJECT_ROOT/data}"
export SOFTCOT_DATA_DIR

mkdir -p "$PROJECT_ROOT/logs/softcot" "$PROJECT_ROOT/outputs" "$CKPT_BASE"
cd "$PROJECT_ROOT"

# Core models
BASE_MODEL_ID="${BASE_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
ASSIST_MODEL_ID="${ASSIST_MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
BASE_MODEL_NAME="${BASE_MODEL_ID##*/}"
ASSIST_MODEL_NAME="${ASSIST_MODEL_ID##*/}"

# Task/dataset settings
TASK_NAME="${TASK_NAME:-gsm8k}"
NUM_THOUGHT_TOKENS="${NUM_THOUGHT_TOKENS:-2}"
TEST_K="${TEST_K:-0}"            # 0 = full test set
K_SHOT="${K_SHOT:-0}"            # 0 = full train set
SEED="${SEED:-42}"
PRED_BASE="${PRED_BASE:-$PROJECT_ROOT/outputs}"

# Training hyperparameters
OUTPUT_NAME="${OUTPUT_NAME:-softcot-run}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_EPOCHS="${N_EPOCHS:-3}"
TUNE_ASSISTANT="${TUNE_ASSISTANT:-true}"   # set to false to skip LoRA on assistant
TUNE_BASE="${TUNE_BASE:-false}"            # set true to also LoRA-tune the base model
RUNS="${RUNS:-10}"
OUTPUT_BASE="${OUTPUT_NAME}"

ASSIST_TUNE_FLAG=()
BASE_TUNE_FLAG=()
if [[ "${TUNE_ASSISTANT,,}" == "true" || "${TUNE_ASSISTANT}" == "1" ]]; then
  ASSIST_TUNE_FLAG+=(--tune_assistant_model)
fi
if [[ "${TUNE_BASE,,}" == "true" || "${TUNE_BASE}" == "1" ]]; then
  BASE_TUNE_FLAG+=(--tune_base_model)
fi

for i in $(seq 1 "$RUNS"); do
  RUN_TAG="run${i}"
  OUTPUT_NAME_RUN="${OUTPUT_BASE}-${RUN_TAG}"

  # Derived checkpoint path (matches train_softcot.py naming)
  CKPT_DIR="$CKPT_BASE/${OUTPUT_NAME_RUN}-${TASK_NAME}-${N_EPOCHS}-${NUM_THOUGHT_TOKENS}-${BASE_MODEL_NAME}-${ASSIST_MODEL_NAME}"
  PROJ_BIN="$CKPT_DIR/projection.bin"

  RUN_PRED_DIR="$PRED_BASE/${RUN_TAG}"
  PRED_FILE="$RUN_PRED_DIR/predictions.jsonl"
  mkdir -p "$RUN_PRED_DIR"

  echo "[SoftCoT][${RUN_TAG}] Starting training..."
  python train_softcot.py \
    --task_name "$TASK_NAME" \
    --output_name "$OUTPUT_NAME_RUN" \
    --batch_size "$BATCH_SIZE" \
    --num_thought_tokens "$NUM_THOUGHT_TOKENS" \
    --n_epochs "$N_EPOCHS" \
    --k_shot "$K_SHOT" \
    "${ASSIST_TUNE_FLAG[@]}" \
    "${BASE_TUNE_FLAG[@]}"

  echo "[SoftCoT][${RUN_TAG}] Training finished. Checking projection file at $PROJ_BIN"
  if [[ ! -f "$PROJ_BIN" ]]; then
    echo "[SoftCoT][${RUN_TAG}] ERROR: projection.bin not found at $PROJ_BIN"
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

  echo "[SoftCoT][${RUN_TAG}] Done."
done
