#!/usr/bin/env bash
# Quickstart Benchmark & Visual Evaluation Commands for Delto-3F Gripper

set -e

# Change directory to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "================================================================="
echo " Delto-3F In-Hand Manipulation Evaluation Suite"
echo "================================================================="
echo "Select evaluation mode:"
echo "  1) Live 3D Visual Demo - Clean Presentation Baseline"
echo "  2) Live 3D Visual Demo - Sim-to-Real Domain Randomization"
echo "  3) 16-Worker Parallel Benchmark - Stressed DR (600 episodes)"
echo "  4) Re-compile 55-Page PDF Documentation"
echo "================================================================="

case "$1" in
  1)
    echo "[Mode 1] Running Clean Baseline 3D Viewer..."
    python3 in_hand_manipulation/evaluate.py \
      --checkpoint best_model/best_model_clean.pth \
      --render \
      --hold_time 1.5
    ;;
  2)
    echo "[Mode 2] Running Sim-to-Real DR 3D Viewer..."
    python3 in_hand_manipulation/evaluate_dr.py \
      --checkpoint best_model/best_model_with_dr.pth \
      --render \
      --smooth_alpha 0.50
    ;;
  3)
    echo "[Mode 3] Running 16-Worker Parallel DR Benchmark..."
    python3 in_hand_manipulation/evaluate_dr.py \
      --checkpoint best_model/best_model_with_dr.pth \
      --episodes_per_range 100 \
      --num_workers 16 \
      --smooth_alpha 0.50 \
      --epsilon_ratio 0.10 \
      --report_file EVAL/EVALUATION_DR_REPORT.md
    ;;
  4)
    echo "[Mode 4] Compiling PDF Documentation..."
    python3 scripts/generate_documentation_pdf.py
    ;;
  *)
    echo "Usage: bash scripts/run_eval_demo.sh [1|2|3|4]"
    exit 1
    ;;
esac

