#!/usr/bin/env bash
# End-to-end pipeline. Run stages individually while iterating; this is the full sweep.
set -euo pipefail
cd "$(dirname "$0")/.."

ENCODERS=("clap_general" "panns_cnn14")   # add "beats" "m2d" "audiomae" once their loaders are active

echo "==> 0. sanity tests (no data needed)"
python tests/test_metrics.py
python tests/test_pipeline_synthetic.py

echo "==> 1. build synthetic domain (skip if you already generated it)"
# python -m src.generate_synthetic --per-class 300 --steps 100 --cfg 7
# python -m src.generate_synthetic --captions data/raw/fsd50k_captions.csv   # paired/instance-level

echo "==> 2. build manifest from DCASE-T7 real+generated and our synth"
python -m src.manifest

echo "==> 3. embed with every encoder"
for e in "${ENCODERS[@]}"; do
  python -m src.embed --encoder "$e"
done

echo "==> 4. baseline cross-domain retrieval + diagnostics (frozen encoders)"
for e in "${ENCODERS[@]}"; do
  python -m src.evaluate --encoder "$e"
  python -m src.domain_probe --encoder "$e"
done

echo "==> 5. domain bridging (method) + re-evaluation"
for e in "${ENCODERS[@]}"; do
  python -m src.bridge --encoder "$e" --supcon 1.0 --dann 0.3 --irm 0.1 --epochs 50
  python -m src.evaluate --encoder "$e" --emb "data/embeddings/${e}_bridged.npz"
  python -m src.domain_probe --encoder "$e" --emb "data/embeddings/${e}_bridged.npz"
done

echo "==> done. Reports in results/."
