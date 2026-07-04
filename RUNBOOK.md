# SoundMatch-SR — reproduction runbook

The exact pipeline used to produce the v1 results. GPU work runs on **Modal**
(`~/.venv-modal/bin/modal`); analysis runs locally on CPU. Heavy artifacts live outside the
code tree via `SMSR_DATA` (data root) and `SMSR_RAW` (extracted-corpora root).

## 0. One-time
```bash
M=~/.venv-modal/bin/modal
export SMSR_DATA=/home/elliott/data/doppelganger            # local analysis paths
```

## 1. Stage the corpus onto the Modal volume (downloads Zenodo 8091972, ~4 GB, ~5 min)
```bash
$M run modal_app.py::stage
```

## 2. Build the manifest (local, from a local extract; or run on the volume) and push it
```bash
SMSR_RAW=$SMSR_DATA/raw/extracted python -m src.manifest_dcase      # -> $SMSR_DATA/manifest.csv
$M volume put soundmatch-sr-data $SMSR_DATA/manifest.csv /manifest.csv --force
```
31,450 rows; real eval=test; 662 BBC clips flagged `is_cc=0`.

## 3. Embed on Modal (A10G, 16-worker parallel decode, ~20 min/encoder)
```bash
$M run modal_app.py --encoder clap_general          # -> /embeddings/clap_general.npz on volume
$M run modal_app.py::embed --encoder panns_cnn14
$M volume get soundmatch-sr-data /embeddings/clap_general.npz $SMSR_DATA/embeddings/ --force
```

## 4. Train the two heads + ablations + E6 on Modal (cached embeddings, ~2-3 min each)
```bash
$M run modal_app.py::bridge --encoder clap_general --objective invariant --supcon 1 --dann .3 --irm .1
$M run modal_app.py::bridge --encoder clap_general --objective sensitive --supcon 1
# ablations
for t in "supcon 1 --tag inv_supcon" "supcon 1 --coral 1 --tag inv_coral" \
         "supcon 1 --dann .3 --tag inv_dann" "supcon 1 --irm .1 --tag inv_irm"; do
  $M run modal_app.py::bridge --encoder clap_general --objective invariant --$t
done
# E6: hold Track-A generators out of training
$M run modal_app.py::bridge --encoder clap_general --objective invariant --supcon 1 --dann .3 --irm .1 \
    --holdout-track A --tag invariant_noA
$M volume get soundmatch-sr-data /embeddings $SMSR_DATA/embeddings --force
```

## 5. Analyze locally (CPU): tables + figures
```bash
export CUDA_VISIBLE_DEVICES=""
E=$SMSR_DATA/embeddings
python -m src.analyze --encoder clap_general \
  --variant frozen=$E/clap_general.npz \
  --variant invariant=$E/clap_general_invariant.npz \
  --variant sensitive=$E/clap_general_sensitive.npz
python -m src.ablation_e6          # Table 2 + Table 4
# -> results/leaderboard.md, ablation_e6.md, summary.json, fig_umap.png, fig_probes.png, fig_morphology.png
```

## 6. SSL encoders (BEATs / M2D / AudioMAE) -> six-encoder dissociation
All in `modal_ssl.py` (separate file/app from `modal_app.py`; same volume + env conventions).
```bash
$M run modal_ssl.py::fetch_ckpts                     # one-time: ckpts -> /data/ckpts, HF cache
$M run modal_ssl.py::smoke --encoder beats           # sanity gate (pairs beat cross-category)
for e in beats m2d audiomae; do
  $M run --detach modal_ssl.py::embed --encoder $e --corpus dcase
  $M run --detach modal_ssl.py::embed --encoder $e --corpus ucs \
      --manifest /data/manifest_ucs_paired.csv --suffix _ucs_paired
done
# 5-fold leave-classes-out heads (class-supcon + instance), tags match src/kfold_eval.py
for e in beats m2d audiomae panns_cnn14 ast; do $M run modal_ssl.py::kfold_heads --encoder $e; done
# pull npz/heads, then assemble the 6-encoder table locally (CPU)
SMSR_DATA=~/data/doppelganger SMSR_MANIFEST=~/data/doppelganger/manifest_ucs_paired.csv \
  python -m scripts.ssl_dissociation      # -> results/ssl_dissociation.{json,md}
```

## Notes
- The local GB10 GPU is **not** used (cooling fault); all GPU goes to Modal.
- Determinism: single `SEED` in `config.py`; splits are hash-bucketed.
- `tests/test_metrics.py` and `tests/test_pipeline_synthetic.py` gate the metric + pipeline math.
