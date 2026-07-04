# SoundMatch-SR — v1 results (real, end-to-end)

What the blueprint promised but the delivery environment could not run (no GPU/weights), now run
for real on the **DCASE-2023 Task-7** core corpus. All GPU work ran on **Modal**; analysis local.

- **Working repo:** `SoundMatch-SR_extracted/sfx-domain-benchmark/`
- **Paper draft:** `PAPER_DRAFT.md` (full numbers) · **Datasheet:** repo `paper/DATASHEET.md`
- **Reproduce:** repo `RUNBOOK.md` · **Figures:** `figures/`
- **Data/embeddings** live off the Dropbox tree at `/home/elliott/data/doppelganger/` (31,450
  clips; CLAP + PANNs embeddings; 8 head variants).

## Corpus actually built
31,450 clips = 5,550 real (`dev`/`eval`, source-disjoint) + 25,900 synthetic from **37 generator
systems** (9 Track-A, 27 Track-B, +baseline) across 7 classes. Leakage-safe splits (real test =
DCASE `eval`; dev train/val by Freesound recording id). 662 BBC clips flagged non-redistributable.

## Headline numbers

**Frozen encoders carry a real synthetic↔real gap (Table 1).**

| frozen | control real→real | synth→real mAP | gap | PAD | domain-probe | event-probe |
|---|---|---|---|---|---|---|
| CLAP | 0.935 | 0.774 [0.766, 0.783] | **+0.161** | 1.27 | 0.90 | 0.95 |
| PANNs | 0.798 | 0.674 | +0.124 | 1.39 | 0.90 | 0.94 |

**The two heads you asked for — the same contrastive head, opposite labels (Table 2).**

| CLAP + head | control | synth→real mAP | gap | PAD | sil(event) | sil(domain) |
|---|---|---|---|---|---|---|
| frozen | 0.935 | 0.774 | +0.161 | 1.27 | 0.176 | 0.043 |
| **invariant** (insensitive) | 0.988 | **0.970** | **+0.018** | 1.38 | **0.751** | **−0.002** |
| **sensitive** | 0.206 | 0.157 | +0.049 | **1.88** | **−0.040** | **0.747** |

- **Invariant** closes the gap 89% (0.161→0.018), synth→real 0.774→0.970, *and* improves
  same-domain retrieval. Same event, synth or real → same representation. (`fig_umap.png` middle.)
- **Sensitive** flips the geometry to organize by domain (silhouette 0.04→0.75), PAD→1.88 — a
  real-vs-synthetic **fidelity axis** (directly useful for scoring your game-SFX generator).

**Ablation (Table 3):** `supcon` alone closes the gap (0.019); DANN/CORAL/IRM add <0.002 — the
cross-domain contrastive term does the work.

**Generalization to unseen generators (Table 4, E6):** a head trained with all Track-A generators
held out still closes the Track-A gap 0.136→0.023 (0.799→0.966) — within 0.008 of the head that
saw them. Not memorized fingerprints.

## Figures
- `figures/fig_umap.png` — frozen vs invariant vs sensitive, colored by event and by domain (the
  money shot: the geometry flips).
- `figures/fig_morphology.png` — per-event gap (keyboard/footstep hardest; motor/gunshot easiest).
- `figures/fig_probes.png` — PAD + event/domain probe accuracy across variants.

## Status & next steps
- **Done:** benchmark built, key systems validated on GPU (encoders.py/bridge.py — the unrun
  pieces), both heads trained + validated, leaderboard + ablation + E6, paper draft + datasheet.
- **Toward arXiv:** activate BEATs/M2D/AudioMAE loaders (2 lines each) for a fuller leaderboard;
  convert `PAPER_DRAFT.md` → LaTeX; optional instance-level pairs + breadth classes via Stable
  Audio Open (repo `src/generate_synthetic.py`).
- **Game-SFX tie-in (secondary):** the sensitive head is a ready fidelity metric; next is to score
  the procedural SFX generator's outputs against real refs on this axis.
