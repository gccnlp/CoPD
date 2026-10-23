#!/usr/bin/env bash
# CoPD two-branch training.
set -euo pipefail

# Model and data
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
MODEL_PATH_A=${MODEL_PATH_A:-$MODEL_PATH}
MODEL_PATH_B=${MODEL_PATH_B:-$MODEL_PATH}
TEXT_TRAIN_DATA=${TEXT_TRAIN_DATA:-}
VISION_TRAIN_DATA=${VISION_TRAIN_DATA:-}
# Use the same mixed text/image validation set for adaptive bk.
TEXT_VAL_DATA=${TEXT_VAL_DATA:-}
VISION_VAL_DATA=${VISION_VAL_DATA:-}

# Distillation coefficients
BK=${BK:-0.01}
REVERSE_KL_COEF_1=${REVERSE_KL_COEF_1:-$BK}  # image student
REVERSE_KL_COEF_2=${REVERSE_KL_COEF_2:-$BK}  # text student

# Resources
NNODES=${NNODES:-${WORLD_SIZE:-1}}
GPUS_PER_NODE=${GPUS_PER_NODE:-${NPROC_PER_NODE:-8}}
GPUS_PER_BRANCH=${GPUS_PER_BRANCH:-}  # defaults to half the GPUs per node
TP_SIZE=${TP_SIZE:-2}

# Configuration and outputs
CONFIG_PATH=${CONFIG_PATH:-configs/copd.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/text_image_4b_bk${BK}}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-text_image_4b_bk${BK}}

if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: bash scripts/train_dual.sh [--dry-run] [key=value ...]

Edit this script or use environment variables.
Trailing key=value arguments take precedence.
--dry-run prints the command without starting training.
HELP
    exit 0
fi

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
: "${TEXT_TRAIN_DATA:?Set TEXT_TRAIN_DATA to the text training file.}"
: "${VISION_TRAIN_DATA:?Set VISION_TRAIN_DATA to the image training file.}"
for value in "$NNODES" "$GPUS_PER_NODE"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Node/GPU counts must be positive integers.' >&2; exit 2; }
done
GPUS_PER_BRANCH=${GPUS_PER_BRANCH:-$((GPUS_PER_NODE / 2))}
for value in "$GPUS_PER_BRANCH" "$TP_SIZE"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Branch GPU count and TP_SIZE must be positive integers.' >&2; exit 2; }
done
if (( 2 * GPUS_PER_BRANCH > GPUS_PER_NODE || (NNODES * GPUS_PER_BRANCH) % TP_SIZE != 0 )); then
    echo 'Two branches must fit the GPUs; branch world size must be divisible by TP_SIZE.' >&2
    exit 2
fi
if (( NNODES > 1 )) && [[ -z "${RAY_ADDRESS:-}" ]]; then
    echo 'Start a Ray cluster and set RAY_ADDRESS for multi-node training.' >&2
    exit 2
fi
for file in "$TEXT_TRAIN_DATA" "$VISION_TRAIN_DATA" "$CONFIG_PATH"; do
    [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 2; }
done
for file in "${TEXT_VAL_DATA:-}" "${VISION_VAL_DATA:-}"; do
    [[ -z "$file" || -f "$file" ]] || { echo "Missing validation file: $file" >&2; exit 2; }
done

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800000}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$OUTPUT_DIR/tensorboard_log}
cmd=("${PYTHON_BIN:-python}" -m verl.trainer.main
    "config=$CONFIG_PATH"
    "worker.actor.model.model_path=$MODEL_PATH"
    "parallel.model_path_a=$MODEL_PATH_A"
    "parallel.model_path_b=$MODEL_PATH_B"
    "data.text_train_files=$TEXT_TRAIN_DATA"
    "data.vision_train_files=$VISION_TRAIN_DATA"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.save_checkpoint_path=$OUTPUT_DIR"
    "trainer.nnodes=$NNODES"
    "trainer.n_gpus_per_node=$GPUS_PER_NODE"
    "parallel.n_gpus_per_model=$GPUS_PER_BRANCH"
    "worker.rollout.tensor_parallel_size=$TP_SIZE"
    "coev.reverse_kl_coef=$BK"
    "coev.reverse_kl_coef_vision=$REVERSE_KL_COEF_1"
    "coev.reverse_kl_coef_text=$REVERSE_KL_COEF_2")

add_env_override() {
    local variable_name="$1" config_key="$2"
    if [[ -n "${!variable_name:-}" ]]; then
        cmd+=("$config_key=${!variable_name}")
    fi
}
add_env_override TEXT_VAL_DATA data.text_val_files
add_env_override VISION_VAL_DATA data.vision_val_files
add_env_override FOREIGN_GRPO_LOSS_ENABLED coev.foreign_grpo_loss_enabled
add_env_override CROSS_REWARD_LOSS_ENABLED coev.cross_reward_loss_enabled
add_env_override CROSS_REWARD_STYLE coev.cross_reward_style
add_env_override ANCHOR_COEF coev.anchor_coef
add_env_override OPD_OFF_STEPS coev.opd_off_steps
add_env_override OPD_ON_STEPS coev.opd_on_steps
add_env_override VAL_FREQ trainer.val_freq
add_env_override SAVE_FREQ trainer.save_freq
add_env_override SAVE_LIMIT trainer.save_limit
add_env_override MAX_STEPS trainer.max_steps
cmd+=("$@")
if "$dry_run"; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi
exec "${cmd[@]}"
