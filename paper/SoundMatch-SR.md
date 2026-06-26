# The Rendering Transformation Is Learnable; the Taxonomy Is Not: A Dissociation in Synthetic↔Real Sound-Effect Retrieval

**SoundMatch-SR.** Draft v2. Numbers are on held-out test splits; ± is std over 5 category folds
or bootstrap 95% CIs where noted. Figures in `results/`.

## Abstract

Perceptual studies report that listeners cannot reliably distinguish a well-made Foley or
text-to-audio rendering of an event from a real recording of the same event — the *identity* of
the event survives the rendering. We ask whether modern audio embeddings represent that identity
invariantly to how a sound was produced, and find a sharp **dissociation**. We introduce
**SoundMatch-SR**, a benchmark for cross-domain (synthetic↔real) sound-effect retrieval, built on
(i) the category-aligned DCASE-2023 Task-7 Foley corpus (5,550 real + 25,900 synthetic clips from
37 generator systems, 7 classes) and (ii) a new **34-category UCS-labelled corpus** (FSD50K real,
CLAP-verified, with 10,420 instance-level Stable-Audio twins). Off-the-shelf encoders carry a
real, linearly-accessible synthetic↔real gap (CLAP: 0.161 mAP; Proxy-A-distance 1.27). A
lightweight head on frozen embeddings closes it within the training taxonomy (0.161→0.018), but —
crucially — a **class-supervised** head does **not** transfer to unseen categories: held-out
classes collapse *below* the frozen baseline, on both 7 and 34 categories. The fix is to train on
the **pairs, not the labels**: an instance-contrastive objective (a clip and its generated twin)
learns the rendering *mapping* rather than category identity, and **generalizes** — on categories
never seen in training it retrieves the exact real twin with R@1 0.83 ± 0.05 (vs 0.64 frozen,
0.33 class-supervised), across 5 folds and on a second encoder (PANNs). Category *clustering* does
not transfer for any objective. We release the benchmark, the UCS corpus recipe, the trained
heads, and a reusable transform.

## 1. Introduction

Foley research finds well-synthesized everyday sounds perceptually indistinguishable from
recordings of the same action; the event's identity survives the synthesis→recording boundary for
a human listener. We make this a measurable question about machine representations:

> **Is "sound identity" recoverable in embedding space, invariant to the rendering process — and
> if so, does that invariance transfer to sounds the model was not trained on?**

We operationalize it as **retrieval** (detection rewards *exploiting* the synthetic↔real gap;
retrieval/invariance reward *removing* it) and answer it with a dissociation that, to our
knowledge, is new: the synthetic→real *transformation* is a learnable, transferable object, while
category identity is not. The practical upshot for sound-effect tooling (search, dedup, generator
evaluation) is that matching a generated sound to its real counterpart generalizes across a
taxonomy, whereas category retrieval must be trained per category.

**Contributions.**
- **C1.** A two-part benchmark: a controlled 7-class corpus (DCASE-T7) and a diverse,
  instance-paired **34-category UCS corpus** (FSD50K + Stable Audio), with leakage-safe splits.
- **C2.** A measured **dissociation**: an instance-contrastive objective generalizes the
  synthetic→real mapping to unseen categories; class-supervised invariance does not (and can
  degrade unseen classes below baseline). Robust over 5 folds and 2 encoders.
- **C3.** A **fidelity-spectrum** analysis showing the instance head recovers the mapping across
  twin-fidelity levels, collapsing only when the twin is independent of its source — i.e. it is
  not exploiting near-copies.
- **C4.** Released artifacts: benchmark, UCS recipe, CLAP-verification, trained heads + a reusable
  transform that adjusts any new clip's embedding (`src/apply_head.py`).

## 2. Related work

General audio-embedding benchmarks (HEAR, X-ARES, MSEB) treat all audio as one domain and never
test invariance to *how* a sound was produced. Deepfake-environmental-audio detection on DCASE-T7
is the mirror objective (exploit the gap); we remove it, and reproduce detection as a controlled
*sensitive* head. We borrow Proxy-A-distance, DANN, Deep CORAL, supervised-contrastive and IRM
from domain adaptation, and connect to instance-discrimination / contrastive self-supervision:
our generalizing objective is instance-level (a clip and its twin), and our negative result is
that class-level supervision does not transfer the invariance — a domain-generalization framing.

## 3. The SoundMatch-SR benchmark

### 3.1 Core corpus (DCASE-T7) — controlled measurement
DCASE-2023 Task 7: 7 classes, 5,550 real clips (FSD50K/UrbanSound8K/BBC/Freesound; DCASE's
source-disjoint dev/eval = our train-val/test) and 25,900 synthetic clips from 37 generator
systems (Track-A data-restricted, Track-B external, + baseline). 662 BBC clips are flagged
non-redistributable. This is the controlled setting: a clean gap measurement and a 7-class
leave-one-class-out probe.

### 3.2 UCS corpus — diverse, instance-paired
To test generalization we need many categories and instance-level pairs. We label real audio with
the industry-standard **UCS taxonomy** (CatID), mapping FSD50K's 200 AudioSet labels onto 34 UCS
categories spanning all morphologies (impact, texture, tonal, whoosh, vocal, mechanical,
ambience). We **CLAP-verify** every clip (keep iff its UCS label is top-5 by audio-text cosine),
retaining 10,420 of 13,579 (77%) — dropping FSD50K's noisiest multi-labels. For each verified
anchor we generate a **synthetic twin with Stable Audio Open (audio-init)**, conditioned on the
real clip so the twin shares the event yet is rendered by the model — 10,420 instance pairs
(20,840 clips). Splits are FSD50K's own train/val/eval, keyed on the Freesound uploader.

### 3.3 Tasks and metrics
**Category retrieval**: query a clip; rank opposite-domain clips; relevant iff same category. mAP,
P@1, with a same-domain control → **domain gap** = control − cross. **Instance retrieval** (UCS):
relevant iff the clip's *exact* cross-domain twin (shared `instance_id`); R@1, MRR. Bootstrap CIs;
on UCS, k-fold over held-out categories.

## 4. Method: heads on frozen embeddings

A small MLP (CLAP-512→256, L2-normalised), frozen after training; the head is the *adjusted
embedding* and applies to any new clip. The objective decides what it learns:
- **invariant** (class-supervised contrastive, cross-domain positives up-weighted, +DANN/IRM):
  pulls *same-category* clips together, removes domain.
- **sensitive** (domain-supervised, within-category): the deliberate mirror — organises by
  rendering; its real-vs-synth direction is a fidelity axis.
- **instance** (the key): positive = a clip and *its own* generated twin (pair-aware batching so
  twins co-occur). Learns the synthetic→real *mapping*, not class identity.

## 5. Experiments

### 5.1 Frozen encoders carry a real gap (DCASE-T7)
| frozen | control real→real | synth→real mAP | gap | PAD | domain-probe | event-probe |
|---|---|---|---|---|---|---|
| CLAP | 0.935 | 0.774 [0.766,0.783] | +0.161 | 1.27 | 0.90 | 0.95 |
| PANNs CNN14 | 0.798 | 0.674 | +0.124 | 1.39 | 0.90 | 0.94 |

The synthetic↔real axis is real and linearly accessible in both encoders.

### 5.2 Within-taxonomy, the gap closes — and the two heads are mirror images (DCASE-T7)
| CLAP + head | control | synth→real mAP | gap | PAD | silhouette(event) | silhouette(domain) |
|---|---|---|---|---|---|---|
| frozen | 0.935 | 0.774 | +0.161 | 1.27 | 0.176 | 0.043 |
| invariant | 0.988 | **0.970** | **+0.018** | 1.38 | **0.751** | **−0.002** |
| sensitive | 0.206 | 0.157 | — | **1.88** | −0.040 | **0.747** |

The invariant head closes the gap 89% while *raising* same-domain control; the sensitive head
flips the geometry to organise by rendering (the fidelity axis). Ablation: cross-domain supcon
alone closes the gap (0.019); DANN/CORAL/IRM add <0.002. Generator-generalization (E6): a head
trained without Track-A generators still closes the Track-A gap 0.136→0.023.

### 5.3 The closed-world failure: class-supervised invariance does NOT generalize
Holding **one class out of training** and evaluating on it (DCASE-T7): the held-out class
collapses *below* frozen (keyboard 0.69→0.43, gunshot 0.81→0.31, rain 0.75→0.30). The
class-supervised head learns cluster alignments, not a transferable transformation. **Adding
categories does not fix this**: on the 34-category UCS corpus, a class-supervised head's held-out
categories still fall (0.40→0.19 full-gallery).

### 5.4 The fix and the dissociation: instance pairs generalize (UCS, 5-fold leave-classes-out)
Training on the *pairs* (instance objective) rather than the labels. Mean ± std over 5 folds, on
**held-out (unseen) categories**:

| variant (unseen cats) | category-mAP | **instance-R@1** | instance-MRR |
|---|---|---|---|
| frozen | 0.615 ± 0.052 | 0.642 ± 0.094 | 0.731 ± 0.081 |
| class-supcon | 0.459 ± 0.072 | 0.334 ± 0.064 | 0.439 ± 0.061 |
| **instance** | 0.492 ± 0.047 | **0.829 ± 0.045** | 0.888 ± 0.034 |

**The dissociation (Fig. `fig_dissociation.png`):** the *instance* objective generalizes the
synthetic→real mapping to categories it never trained on (instance-R@1 0.83, +0.19 over frozen,
every fold), while class-supcon *hurts* it (0.33). **No** objective generalizes *category*
clustering (all ≤ frozen on unseen-category mAP). The transformation transfers; the taxonomy does
not. It holds on **three encoders** — CLAP, PANNs (instance 0.747 vs frozen 0.655 vs class 0.378),
and AST (instance 0.945 vs frozen 0.849 vs class 0.380) — so it is a property of the objective,
not of any one encoder.

### 5.6 Cross-generator scope: instance pairing requires audio conditioning
We add 670 ElevenLabs twins across all categories, captioned from each clip's Freesound title+tags
(text-to-SFX, no audio-init). Their instance correspondence is **weak** — frozen instance-R@1 0.11
(vs 0.55–0.98 for audio-init twins) — because the caption is a bottleneck that discards
instance-specific detail; the twins are category-appropriate (cat-mAP 0.48) but not renderings of
the *specific* clip, so the instance head has no mapping to recover (0.10). The transferable
synthetic→real mapping is thus specific to generators that **condition on the source audio**; for
purely text-conditioned generators, instance-level pairing is not available. This scopes C2 and
explains our use of audio-init for the paired corpus.

### 5.5 The instance head isn't exploiting near-copies (fidelity spectrum)
We generate twins across fidelity levels (measured as CLAP cosine of a twin to its source) and
plot instance R@1 (Fig. `fig_fidelity.png`):

| twin fidelity | 0.73 | 0.64 | 0.55 | ~0.11 (text-only / degenerate) |
|---|---|---|---|---|
| frozen R@1 | 0.55 | 0.39 | 0.27 | ~0.00 |
| instance-head R@1 | 0.98 | 0.91 | 0.70 | ~0.00 |

As twins diverge, frozen retrieval degrades but the instance head holds up (advantage +0.43…+0.52);
both collapse to chance only when the twin is independent of its source — i.e. the head recovers a
rendering *mapping*, not a near-copy shortcut.

## 6. Application: game / generative-SFX tooling
The instance head is a transferable synthetic↔real matcher (find a real reference for a generated
sound; dedup across real/synthetic sources); the sensitive head is a realness/fidelity axis for
scoring a generator. Both ship as reusable transforms (`apply_head`) on top of CLAP. See
`HANDOFF_SFX.md`.

## 7. Limitations
Category clustering does not transfer to unseen categories (train per target taxonomy); twins are
Stable-Audio audio-init at one operating point per corpus (the spectrum covers fidelity but one
generator family); frozen-embedding heads cannot recover information the backbone discarded; CLAP
backbone bounds very-short/heavily-designed sounds. SSL encoders (BEATs/M2D/AudioMAE) are stubs.

## 8. Release
Code (MIT), DCASE + UCS manifests, the CLAP-verification, the generation recipe (not restricted
audio), the trained heads + `apply_head`, a datasheet (`DATASHEET.md`), and `RUNBOOK.md`.
