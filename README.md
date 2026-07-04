# Doppelganger: Sound Effects and Their Synthetic Twins

A benchmark for matching a **synthetic** sound effect to the **real** recording it was generated
from. Audio-conditioned generators now produce a synthetic twin for a real clip, so real and
synthetic versions of an event increasingly coexist in sound libraries and training corpora.
Doppelganger measures whether an audio representation can recover the specific real source of a
synthetic clip across that boundary, and separates two capabilities that are usually conflated:
recognizing the *same specific event* across the synthetic–real boundary, and recognizing the
*same kind of event*.

Headline finding — a **dissociation**: training a head on instance *pairs* (a real clip and its own
audio-conditioned twin) generalizes to sound events unseen in head training (full-gallery R@1 ≈
0.80, chance 0.0003), while training on class *labels* collapses below the frozen encoder. Category
structure transfers for no objective. The effect requires audio-conditioned generation and is
specific to a generator family.

Paper: [`paper/doppelganger.pdf`](paper/doppelganger.pdf) · datasheet: [`paper/DATASHEET.md`](paper/DATASHEET.md).

## Where the artifacts live

This repo holds **code, the paper, and small results**. Large derived artifacts are hosted
separately:

| Artifact | Home |
|---|---|
| Code, paper, results JSON, figures, corpus manifests | **this repo** |
| Precomputed embeddings (`.npz`) and generated twin audio | HuggingFace **dataset** `elliottash/doppelganger` |
| Trained heads (`*.head.pt`) and fine-tuned encoders (`ckpts/`) | HuggingFace **model** repo |

Embeddings and checkpoints are intentionally **not** in git — they are large binaries and are
reproducible from the code plus the manifests.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Validate the install (no data, no GPU)
```bash
python tests/test_metrics.py             # retrieval math vs hand-computed values
python tests/test_pipeline_synthetic.py  # whole pipeline on synthetic embeddings
```

## Reproduce
GPU work runs on Modal (`modal_app.py`, `modal_ssl.py`, `modal_baselines.py`).

```bash
# 1. Corpora: DCASE-2023 Task-7 (ready-made core) + the UCS corpus built from FSD50K
#    via src/ingest_fsd50k.py -> src/taxonomy_ucs.py -> src/clap_verify.py.
#    Manifests (one row per clip, leakage-safe splits) are in data/manifests/.
# 2. Twins: src/generate_synthetic.py / src/gen_pairs.py (Stable-Audio-Open audio-init).
# 3. Embed with a frozen encoder (src/embed.py, src/encoders.py).
# 4. Frozen-gap diagnostics: src/evaluate.py, src/domain_probe.py.
# 5. Heads (invariant / sensitive / instance / classifier): src/bridge.py --objective ...
# 6. The dissociation (5-fold leave-classes-out): src/kfold_eval.py, scripts/ssl_dissociation.py.
# 7. Extra analyses: src/extra_metrics.py, src/validate_sensitive.py,
#    src/cross_generator_eval.py, src/spectrum_eval.py; figures via src/make_figs.py.
```

## Layout
```
config.py                  taxonomy, encoder registry, generation prompts, paths (env-driven)
src/                       pipeline: manifests, encoders, embedding, heads, evaluation, figures
modal_*.py                 Modal apps for GPU embedding / SSL encoders / baselines / data stats
data/manifests/            corpus manifests (one row per clip; the reproducibility source of truth)
results/                   metrics JSON + paper figures
paper/                     doppelganger.tex/.pdf, references.bib, DATASHEET.md, style file
arxiv/                     flat arXiv package (pdflatex-only) + tarball
human_study/               human annotation baseline: task, trial builder, analysis, aggregates
                           (raw participant responses are NOT redistributed)
tests/                     metric + end-to-end synthetic tests
```

## License and data notes
Code is **MIT** (see `LICENSE`). Real audio is redistributed only where the source license permits;
restricted sources (e.g. BBC) are referenced by ID, not redistributed. Synthetic twins are generated
with Stable Audio Open (Stability AI Community License). The human annotation baseline redistributes
the task, trial definitions, and aggregated results only — never raw participant data.
