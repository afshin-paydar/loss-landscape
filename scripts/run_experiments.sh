#!/bin/bash
# =============================================================================
# Joint Pruning + Mixed-Precision Quantization: Full Experiment Pipeline
#
# This script runs all 8 experiments from the mathematical formulation document.
# Adjust paths, GPU count, and hyperparameters as needed.
#
# Prerequisites:
#   - Trained baseline ResNet-56 checkpoint (see Step 0)
#   - CUDA-capable GPU(s)
#   - pip install mpi4py h5py matplotlib (if not already installed)
# =============================================================================

set -e  # Exit on error

cd "$(dirname "$0")/.."  # Navigate to repo root
SAVE_DIR="cifar10/trained_nets/resnet56_joint"
NGPU=1  # Set to number of available GPUs

echo "============================================="
echo " Joint Pruning + Quantization Experiments"
echo "============================================="

# -----------------------------------------------------------------
# Step 0: Train baseline ResNet-56 (skip if you already have one)
# -----------------------------------------------------------------
# Uncomment the following if you need to train from scratch:
# echo "[Step 0] Training baseline ResNet-56..."
# python cifar10/main.py --model resnet56 --epochs 300
# BASELINE="cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7"

# SET THIS to your actual baseline checkpoint path:
BASELINE="${1:-cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7}"

if [ ! -f "$BASELINE" ]; then
    echo "ERROR: Baseline model not found at $BASELINE"
    echo "Usage: bash scripts/run_experiments.sh <path_to_baseline_model>"
    echo ""
    echo "To train a baseline first, run:"
    echo "  python cifar10/main.py --model resnet56 --epochs 300"
    exit 1
fi

echo "Using baseline: $BASELINE"

# -----------------------------------------------------------------
# Step 1: Train joint pruned + quantized model (4 stages)
# -----------------------------------------------------------------
echo ""
echo "[Step 1] Training joint model (4 stages)..."
echo "  Stage 1: Warm-up (20 epochs)"
echo "  Stage 2: Joint search (100 epochs)"
echo "  Stage 3: Discretization"
echo "  Stage 4: Fine-tuning (100 epochs)"

python train_joint.py \
    --model resnet56 \
    --pretrained "$BASELINE" \
    --batch_size 128 \
    --warmup_epochs 20 \
    --joint_epochs 100 \
    --finetune_epochs 100 \
    --lambda_s 0.1 \
    --gamma 0.01 \
    --S_max_ratio 0.5 \
    --tau_start 5.0 \
    --tau_end 0.1 \
    --cuda \
    --save_dir "$SAVE_DIR"

JOINT_MODEL="$SAVE_DIR/final.pth"

# -----------------------------------------------------------------
# E1: Baseline task loss surface (original Li et al. workflow)
# -----------------------------------------------------------------
echo ""
echo "[E1] Baseline task loss surface..."

python plot_surface.py \
    --model resnet56 \
    --model_file "$BASELINE" \
    --x=-1:1:51 --y=-1:1:51 \
    --dir_type weights \
    --xnorm filter --xignore biasbn \
    --ynorm filter --yignore biasbn \
    --cuda --plot

# -----------------------------------------------------------------
# E2: Compressed model task loss surface
# -----------------------------------------------------------------
echo ""
echo "[E2] Compressed model task loss surface..."

python plot_surface_joint.py \
    --model resnet56 --joint \
    --model_file "$JOINT_MODEL" \
    --x=-1:1:51 --y=-1:1:51 \
    --xnorm filter --xignore biasbn \
    --ynorm filter --yignore biasbn \
    --subspace all \
    --lambda_s 0.0 --gamma 0.0 --S_max_ratio 0.0 \
    --cuda --plot

# -----------------------------------------------------------------
# E3: Lagrangian surface at different lambda values
# -----------------------------------------------------------------
echo ""
echo "[E3] Lagrangian surfaces at varying lambda_s..."

for LAMBDA in 0.01 0.1 1.0; do
    echo "  lambda_s = $LAMBDA"
    python plot_surface_joint.py \
        --model resnet56 --joint \
        --model_file "$JOINT_MODEL" \
        --x=-1:1:51 --y=-1:1:51 \
        --xnorm filter --xignore biasbn \
        --ynorm filter --yignore biasbn \
        --subspace all \
        --lambda_s "$LAMBDA" --gamma 0.01 --S_max_ratio 0.5 \
        --cuda --plot
done

# -----------------------------------------------------------------
# E4: Weight-only subspace slice
# -----------------------------------------------------------------
echo ""
echo "[E4] Weight-only subspace..."

python plot_surface_joint.py \
    --model resnet56 --joint \
    --model_file "$JOINT_MODEL" \
    --x=-1:1:51 --y=-1:1:51 \
    --xnorm filter --xignore biasbn \
    --ynorm filter --yignore biasbn \
    --subspace weights_only \
    --lambda_s 0.1 --gamma 0.01 --S_max_ratio 0.5 \
    --cuda --plot

# -----------------------------------------------------------------
# E5: Pruning-only subspace slice
# -----------------------------------------------------------------
echo ""
echo "[E5] Pruning-only (alpha) subspace..."

python plot_surface_joint.py \
    --model resnet56 --joint \
    --model_file "$JOINT_MODEL" \
    --x=-1:1:51 --y=-1:1:51 \
    --xnorm filter --xignore biasbn \
    --ynorm filter --yignore biasbn \
    --subspace alpha_only \
    --lambda_s 0.1 --gamma 0.01 --S_max_ratio 0.5 \
    --cuda --plot

# -----------------------------------------------------------------
# E6: Quantization-only subspace slice
# -----------------------------------------------------------------
echo ""
echo "[E6] Quantization-only (beta) subspace..."

python plot_surface_joint.py \
    --model resnet56 --joint \
    --model_file "$JOINT_MODEL" \
    --x=-1:1:51 --y=-1:1:51 \
    --xnorm filter --xignore biasbn \
    --ynorm filter --yignore biasbn \
    --subspace beta_only \
    --lambda_s 0.1 --gamma 0.01 --S_max_ratio 0.5 \
    --cuda --plot

# -----------------------------------------------------------------
# E7: Hessian eigenvalue ratio map
# -----------------------------------------------------------------
echo ""
echo "[E7] Hessian eigenvalue analysis (requires more compute)..."
echo "  (This uses the existing plot_hessian_eigen.py on the baseline)"

python plot_hessian_eigen.py \
    --model resnet56 \
    --model_file "$BASELINE" \
    --x=-1:1:21 --y=-1:1:21 \
    --dir_type weights \
    --xnorm filter --xignore biasbn \
    --ynorm filter --yignore biasbn \
    --cuda

# -----------------------------------------------------------------
# E8: Pareto front (sweep lambda_s, collect accuracy vs BOPs)
# -----------------------------------------------------------------
echo ""
echo "[E8] Pareto front sweep..."

PARETO_DIR="$SAVE_DIR/pareto"
mkdir -p "$PARETO_DIR"

for RATIO in 0.25 0.35 0.50 0.65 0.75; do
    echo "  S_max_ratio = $RATIO"
    python train_joint.py \
        --model resnet56 \
        --pretrained "$BASELINE" \
        --warmup_epochs 10 \
        --joint_epochs 50 \
        --finetune_epochs 50 \
        --lambda_s 0.1 \
        --gamma 0.01 \
        --S_max_ratio "$RATIO" \
        --cuda \
        --save_dir "$PARETO_DIR/ratio_${RATIO}"
done

echo ""
echo "============================================="
echo " All experiments complete!"
echo " Results saved in: $SAVE_DIR"
echo ""
echo " Surface HDF5 files can be visualized with:"
echo "   python plot_2D.py --surf_file <file.h5> --surf_name task_loss"
echo "   python plot_2D.py --surf_file <file.h5> --surf_name lagrangian_loss"
echo "============================================="
