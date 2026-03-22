#!/bin/bash
# Download and prepare PanNuke dataset for CellViT training
# Usage: bash setup_pannuke.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANNUKE_DIR="$SCRIPT_DIR/PanNuke"
RAW_DIR="$PANNUKE_DIR/raw"

echo "=== Downloading PanNuke dataset ==="
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

for fold in 1 2 3; do
    if [ ! -f "fold_${fold}.zip" ]; then
        echo "Downloading fold_${fold}..."
        wget -q --show-progress "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_${fold}.zip"
    else
        echo "fold_${fold}.zip already exists, skipping"
    fi
done

echo "=== Extracting ==="
for fold in 1 2 3; do
    unzip -oq "fold_${fold}.zip"
done

echo "=== Reorganizing folds ==="
for i in 0 1 2; do
    src_fold=$((i + 1))
    mkdir -p "$PANNUKE_DIR/fold${i}"
    cp "Fold ${src_fold}/images/fold${src_fold}/images.npy" "$PANNUKE_DIR/fold${i}/images.npy"
    cp "Fold ${src_fold}/masks/fold${src_fold}/masks.npy" "$PANNUKE_DIR/fold${i}/masks.npy"
    cp "Fold ${src_fold}/images/fold${src_fold}/types.npy" "$PANNUKE_DIR/fold${i}/types.npy"
done

echo "=== Running prepare_pannuke.py ==="
cd "$SCRIPT_DIR"
PYTHONPATH="$SCRIPT_DIR" python3 cell_segmentation/datasets/prepare_pannuke.py \
    --input_path "$PANNUKE_DIR" \
    --output_path "$PANNUKE_DIR"

echo "=== Copying dataset configs ==="
cp -r configs/datasets/PanNuke/* "$PANNUKE_DIR/"

echo "=== Downloading SAM-H encoder ==="
mkdir -p models/pretrained/SAM
if [ ! -f "models/pretrained/SAM/sam_vit_h.pth" ]; then
    wget -q --show-progress "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
        -O models/pretrained/SAM/sam_vit_h.pth
else
    echo "SAM-H already exists, skipping"
fi

echo "=== Done! ==="
echo "Dataset ready at: $PANNUKE_DIR"
echo "SAM-H ready at: models/pretrained/SAM/sam_vit_h.pth"
echo ""
echo "To train: python3 cell_segmentation/run_cellvit.py --config configs/training/cellvit_samh_pannuke_fold1.yaml"
