#!/usr/bin/env bash
#SBATCH --job-name=recaptcha-grid-predict
#SBATCH --partition=steins
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME=singularity
else
    echo "apptainer/singularity が見つかりません" >&2
    exit 1
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="${IMAGE:-${PROJECT_DIR}/re-recaptcha.sif}"
MODEL_PATH="${MODEL_PATH:-runs/yolov8_segmentation/fog_mosaic_finetune/weights/best.pt}"
SOURCE="${SOURCE:-src/yolo_dataset/images/test}"
LABELS_DIR="${LABELS_DIR:-src/yolo_dataset/labels/test}"
TARGET="${TARGET:-bus}"
CONF="${CONF:-0.25}"
MIN_TILE_MASK_RATIO="${MIN_TILE_MASK_RATIO:-0.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/segment/grid_tile_predictions/predict}"

cd "${PROJECT_DIR}"
mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/runs" \
    "${PROJECT_DIR}/.cache/matplotlib" \
    "${PROJECT_DIR}/.cache/ultralytics"

if [[ ! -f "${IMAGE}" ]]; then
    echo "Apptainerイメージが見つかりません: ${IMAGE}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "学習済みモデルが見つかりません: ${MODEL_PATH}" >&2
    echo "MODEL_PATHを指定して再実行してください。" >&2
    exit 1
fi

if [[ "${RUNTIME}" == "apptainer" ]]; then
    export APPTAINERENV_MPLCONFIGDIR=/workspace/.cache/matplotlib
    export APPTAINERENV_YOLO_CONFIG_DIR=/workspace/.cache/ultralytics
else
    export SINGULARITYENV_MPLCONFIGDIR=/workspace/.cache/matplotlib
    export SINGULARITYENV_YOLO_CONFIG_DIR=/workspace/.cache/ultralytics
fi

echo "model=${MODEL_PATH}"
echo "source=${SOURCE}"
echo "target=${TARGET} conf=${CONF} min_tile_mask_ratio=${MIN_TILE_MASK_RATIO}"

"${RUNTIME}" exec --nv --cleanenv \
    --bind "${PROJECT_DIR}:/workspace" \
    --pwd /workspace \
    "${IMAGE}" \
    python src/segmentation/predict_grid_tiles.py \
        --model-path "${MODEL_PATH}" \
        --source "${SOURCE}" \
        --labels-dir "${LABELS_DIR}" \
        --target "${TARGET}" \
        --conf "${CONF}" \
        --device 0 \
        --min-tile-mask-ratio "${MIN_TILE_MASK_RATIO}" \
        --output-root "${OUTPUT_ROOT}"
