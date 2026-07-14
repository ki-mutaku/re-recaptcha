#!/usr/bin/env bash
#SBATCH --job-name=recaptcha-cls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

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

"${RUNTIME}" exec --nv --cleanenv \
    --bind "${PROJECT_DIR}:/workspace" \
    --pwd /workspace \
    "${IMAGE}" \
    python src/image_classification/classification.py --image-dir img --target bus --threshold 0.05
