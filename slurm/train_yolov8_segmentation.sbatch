#!/usr/bin/env bash
#SBATCH --job-name=recaptcha-yolo
#SBATCH --partition=steins
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
IMAGE="${PROJECT_DIR}/re-recaptcha.sif"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs" "${PROJECT_DIR}/.cache/matplotlib" "${PROJECT_DIR}/.cache/ultralytics"

export SINGULARITYENV_MPLCONFIGDIR=/workspace/.cache/matplotlib
export SINGULARITYENV_YOLO_CONFIG_DIR=/workspace/.cache/ultralytics

run_in_container() {
    "${RUNTIME}" exec --nv --cleanenv \
        --bind "${PROJECT_DIR}:/workspace" \
        --pwd /workspace \
        "${IMAGE}" \
        "$@"
}

TRAIN_DATA_ARGS=()
if [[ "${PREPARE_DATASET:-0}" == "1" ]]; then
    echo "初回用の画像・ラベル生成を実行します。"
    run_in_container \
        python src/segmentation/add_noise.py \
        --seed 42
    run_in_container \
        python src/segmentation/extract_and_apply_labels.py
    TRAIN_DATA_ARGS+=(--reshuffle)
else
    echo "既存の画像・ラベル・split manifestを再利用します。"
fi

run_in_container \
    python src/segmentation/train_yolov8_segmentation.py \
        --model yolov8n-seg.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch 16 \
        --device 0 \
        --workers "${SLURM_CPUS_PER_TASK}" \
        --project runs/yolov8_segmentation \
        --name fog_mosaic_finetune \
        "${TRAIN_DATA_ARGS[@]}"
