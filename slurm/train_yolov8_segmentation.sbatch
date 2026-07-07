#!/usr/bin/env bash
#SBATCH --job-name=recaptcha-yolo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Adjust these module names to match the department server.
module purge
module load singularity

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
IMAGE="${PROJECT_DIR}/re-recaptcha.sif"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs" "${PROJECT_DIR}/.cache/matplotlib" "${PROJECT_DIR}/.cache/ultralytics"

export SINGULARITYENV_MPLCONFIGDIR=/workspace/.cache/matplotlib
export SINGULARITYENV_YOLO_CONFIG_DIR=/workspace/.cache/ultralytics

singularity exec --nv --cleanenv \
    --bind "${PROJECT_DIR}:/workspace" \
    --pwd /workspace \
    "${IMAGE}" \
    python src/train_yolov8_segmentation.py \
        --model yolov8n-seg.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch 16 \
        --device 0 \
        --workers "${SLURM_CPUS_PER_TASK}" \
        --project runs/yolov8_segmentation \
        --name fog_mosaic_finetune
