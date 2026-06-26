# SoundMatch-SR — Benchmark Blueprint & Delivery Plan

**Audience:** the engineering/research team building this.
**Goal:** ship a reusable benchmark + a method, and a paper strong enough for ICASSP / INTERSPEECH / WASPAA / a NeurIPS Datasets & Benchmarks submission.
**One sentence:** *We measure whether general-purpose audio embeddings can retrieve the real-world counterpart of a synthetic sound effect (and vice versa), show that off-the-shelf encoders are confounded by the rendering process, and provide an invariance objective that recovers cross-domain matches.*

This document is the contract. The companion code in this repo implements every piece described here; where a section says "see `src/...`," that file is the reference implementation.

---

## 1. Why this is novel (and defensible to reviewers)

Three bodies of work sit near this, and none occupies our spot:

1. **General audio-embedding benchmarks** — HEAR (19 tasks), X-ARES, and the 2025 MSEB/MAEB benchmarks evaluate frozen embeddings on classification, retrieval, and clustering. They treat all audio as one domain. *They never ask whether the embedding is invariant to how a sound was produced.*
2. **Synthetic/"deepfake" environmental-audio detection** — built directly on the DCASE-2023 Task-7 corpus, these works train classifiers to tell fake from real. *That is the opposite of our goal:* they exploit the synthetic↔real gap; we want representations that are invariant to it. Their existence is useful to us — it proves the gap is real and linearly accessible — but the task framing (detect the domain) is the mirror image of ours (ignore the domain, preserve the event).
3. **Sim-to-real transfer in music IR** (e.g., drum transcription with synthetic training data) — quantifies a transfer gap for a *supervised* task. *We study retrieval/representation geometry, not a single supervised head, and on sound effects rather than music.*

**The intellectual hook** (put this in the intro): perceptual studies show Foley and real recordings of the same action are essentially indistinguishable to human listeners — i.e., the *identity* of a sound event survives the synthetic→real transformation for human ears. So the empirical question is sharp and falsifiable: **is "sound identity" an identifiable construct in embedding space, separable from the rendering process?** If yes, an invariant representation should exist and we should be able to find it. If general-purpose encoders fail at it, that is a concrete, measurable limitation of today's audio foundation models.

**Contribution claims (what gets cited):**
- C1. A benchmark and protocol for **cross-domain sound-effect retrieval** (synthetic↔real), at two correspondence levels (category and instance/paired).
- C2. An **analysis** decomposing how much of each popular encoder's geometry is event-semantic vs production-artifact (Proxy-A-distance, probes, silhouettes).
- C3. A **bridging method** (cross-domain supervised contrastive + invariance penalties) that reduces the gap, with ablations isolating which mechanism matters.
- C4. The first **leaderboard** across CLAP / BEATs / M2D / AudioMAE / supervised baselines on this axis.

---

## 2. Research questions → hypotheses → where each is answered

| RQ | Question | Hypothesis | Evidence (table/figure) |
|----|----------|-----------|-------------------------|
| RQ1 | How large is the synthetic↔real retrieval gap for SOTA frozen encoders? | Large and encoder-dependent; CLAP < supervised PANNs (CLAP's text grounding should be more semantic). | `domain_gap_mAP` per encoder → **Table 1 (leaderboard)** |
| RQ2 | Is the gap driven by event-correlated differences or domain-global nuisance? | Mostly nuisance: domains are linearly separable even within an event. | Proxy-A-distance, event-vs-domain probe accuracy → **Fig 2** |
| RQ3 | Which sound morphologies suffer most? | Texture/tonal (rain, piano) transfer better than transient/mechanical (gunshot, engine), where synthesis artifacts are audible. | per-event / per-morphology mAP → **Fig 3** |
| RQ4 | Can a lightweight invariance objective close the gap without hurting same-domain retrieval? | Yes; cross-domain supcon + IRM raises cross-domain mAP and lowers PAD with ≤2 pt same-domain cost. | bridged vs frozen → **Table 2 + Fig 4** |
| RQ5 | Does instance-level (paired) matching track category-level? | Correlated but strictly harder; recall@1 stays low even when category mAP is high. | instance metrics → **Table 3** |

Pre-register these (OSF or an appendix) before running the bridging experiments to keep C3 honest.

---

## 3. The two tasks

Everything is evaluated on a held-out **test** split.

- **Category-level retrieval.** Query a clip; rank gallery clips of the *opposite domain*; a gallery item is relevant iff it shares the query's event class. Run **both directions** (`synth→real`, `real→synth`). Metrics: mAP (headline), R@{1,5,10}, nDCG@{1,5,10}, MRR. Report a **same-domain control** (`real→real`) so the gap is `control_mAP − cross_mAP`, not an absolute number that conflates "bad encoder" with "big gap."
- **Instance-level (paired) retrieval.** Only over clips that have a cross-domain partner (a synthetic clip generated from a specific real clip, sharing an `instance_id`). Exactly one relevant gallery item. Metrics: R@{1,5,10} (= hit@k), MRR, median rank of the true match. This is the literal "match synthetic and real *variants of the same sound*" task and is the hardest, most novel number.

Implementation: `src/evaluate.py` (both levels, both directions, bootstrap 95% CIs by resampling queries). Metric definitions and unit tests: `src/metrics.py`, `tests/test_metrics.py`.

---

## 4. Data — sources, construction, licensing, splits

### 4.1 Core corpus (use this first — minimal new collection)
**DCASE-2023 Challenge Task 7 (Foley Sound Synthesis)** is, in effect, a ready-made paired corpus:
- **Real ("nonfake"):** ~5,550 real clips across 7 classes — *dog_bark, footstep, gunshot, keyboard, moving_motor_vehicle, rain, sneeze_cough* — sourced from UrbanSound8K, FSD50K, and BBC Sound Effects. Mono, 16-bit, 22.05 kHz, 4 s.
- **Synthetic ("fake"):** ~25,200 clips generated by the challenge participants' systems (many different synthesis methods → built-in synthesizer diversity, which is great for generalization claims).
- Hosted on Zenodo (record `8091972`) with `DevMeta.csv` / `EvalMeta.csv` provenance.

This gives **category-level** correspondence immediately and spans transient, texture, tonal-ish, mechanical, and vocal/animal morphologies. It is the backbone of C1, C2, C4.

### 4.2 Extension classes (breadth)
Add 12 classes (see `config.EXTENSION`: glass_break, piano_note, drum_hit, bell_ring, applause, wind, …) so the benchmark isn't 7-class and isn't dominated by one morphology. Real audio: pull from **FSD50K** and **ESC-50** (both Creative-Commons, ESC-50 is conveniently 5 s clips). Synthetic audio: generate (next).

### 4.3 Synthetic generation (your synthesizer, modern + releasable)
Generate with **Stable Audio Open 1.0** (`stabilityai/stable-audio-open-1.0`): open weights, single consumer GPU, strong on SFX/field recordings, and — critically for a *redistributable* benchmark — trained only on CC0/CC-BY/CC Sampling+ audio. For synthesizer diversity (so reviewers can't say results are Stable-Audio-specific), add a second generator: **AudioGen** (Meta audiocraft) and/or **Stable Audio Open Small-SFX**. Code: `src/generate_synthetic.py` (category mode and paired mode), prompts in `config.py`.

**Building instance-level pairs (for the paired task):** caption a set of real FSD50K clips (use their tags, or an audio-captioning model), then generate one synthetic clip per real clip from that caption; emit `pairs.csv` so each synthetic clip inherits the real clip's `instance_id`. Optionally also do audio-to-audio (Stable Audio Open supports audio init) for tighter pairing. This yields genuinely paired synth/real of "the same" sound.

### 4.4 Licensing rules (decide before any release)
- **Code:** MIT or Apache-2.0.
- **Real audio:** redistribute only CC-licensed audio (FSD50K, ESC-50, UrbanSound8K). **Do not redistribute BBC Sound Effects** clips from the DCASE set — they are research-use-only; instead release a manifest + a fetch script (AudioSet-style "we ship IDs, not audio").
- **Synthetic audio:** ship the **generation log** (event, prompt, template, seed, steps, cfg — `src/generate_synthetic.py` writes it) so anyone can regenerate bit-comparable clips; optionally also release the WAVs under the Stability AI Community License. This sidesteps any ambiguity about redistributing model outputs and makes the benchmark fully reproducible.
- Document everything in a **datasheet** (Gebru et al. format) — D&B reviewers expect it.

### 4.5 Splits — leakage is the #1 way this paper dies
`src/manifest.py` assigns splits that are **source- and instance-disjoint**:
- paired clips are keyed on `instance_id` so a synth/real pair never straddles splits;
- unpaired clips are keyed on a `(source, recording-stem)` proxy for the Freesound uploader, so near-duplicates from the same upload don't leak across splits.
Run a duplicate check too: embed everything, flag intra-class cross-split pairs with cosine > 0.98, and quarantine. A benchmark that leaks recordings across the train/test boundary is the first thing a sharp reviewer will probe.

### 4.6 Target statistics
Aim for ≥ 19 classes, ≥ 300 real + ≥ 300 synthetic clips per class (more for the core 7), and ≥ 3,000 instance-level pairs. Report the full per-class, per-domain, per-split count table (an appendix table reviewers will check).

---

## 5. Baselines (the leaderboard)

Encoders live in `config.ENCODERS`; add a row to benchmark a new one. Evaluate **frozen** first (this is the honest "what do today's models do" result), then optionally fine-tuned.

- **CLAP** — `laion/clap-htsat-unfused` (general), `laion/clap-htsat-fused`, and the music+speech+audioset checkpoint. Fully implemented via `transformers` in `src/encoders.py`. Expectation: strongest, because text grounding pushes toward semantics over texture.
- **BEATs** — `BEATs_iter3_plus_AS2M`. De-facto strong general-audio SSL backbone.
- **M2D** — masked-modeling-duo; strong on HEAR.
- **AudioMAE** — masked-autoencoder; clean SSL reference.
- **PANNs CNN14** — supervised AudioSet features; the "production-artifact-heavy" reference and likely worst-case gap.

(Loaders for the SSL encoders are documented stubs in `src/encoders.py`: each needs two lines filled to load its upstream checkpoint. CLAP and PANNs are ready.)

---

## 6. The method — domain bridging (C3)

A lightweight projection head trained on **frozen** embeddings (so it's cheap — minutes on cached vectors). Four combinable objectives, all in `src/bridge.py`:

- **Cross-domain supervised contrastive (workhorse).** Positives = same event; **cross-domain positives are up-weighted**, so the head is explicitly rewarded for pulling a synthetic clip toward the real clip of the same event rather than just tightening within-domain clusters.
- **DANN** (gradient-reversal domain discriminator) — makes domain unpredictable from the representation.
- **Deep CORAL** — aligns synth vs real feature covariances.
- **IRM** — treats {synthetic, real} as *environments* and penalizes representations whose optimal event-classifier differs across them.

**Framing for the write-up (and a genuine connection to invariance/causal inference):** model the rendering process (synthesized vs recorded) as a **nuisance variable** that must not influence the representation, while the event label is the signal to preserve. IRM operationalizes "the event predictor should be simultaneously optimal in both domains" — i.e., domain is a confounder to be marginalized, not a cause of identity. This is the formal version of the Foley-indistinguishability intuition and is the conceptual spine of the method section.

**Ablation grid** (→ Table 2): each objective alone, supcon+each, all-on; sweep the cross-domain positive weight; show frozen → bridged deltas in *both* cross-domain mAP (up) and Proxy-A-distance (down), plus the same-domain control to prove you didn't just collapse everything.

---

## 7. Diagnostics that become figures (C2)

`src/domain_probe.py` produces, per encoder, the analysis that carries the paper independent of any retrieval score:
- **Proxy-A-distance** — `2(1−2·err)` of a linear domain classifier. Near 2 = trivially separable (big gap); near 0 = invariant (the target). This is the standard DA gap proxy.
- **Event vs domain linear probes** on the same features — a good identity space has high event accuracy *and* low domain accuracy (`identity_minus_domain`).
- **Silhouette ratio** — silhouette of points labeled by event vs by domain. If domain-silhouette > event-silhouette, the geometry is organized by *how it was made*, not *what it is* — the headline pathology, one sentence and one figure.

Add **UMAP plots** colored by event vs by domain (frozen vs bridged) — the qualitative money shot for Fig 4.

---

## 8. Experiment matrix → paper skeleton

| # | Experiment | Command | Lands as |
|---|-----------|---------|----------|
| E1 | Frozen cross-domain retrieval, all encoders | `src.evaluate --encoder X` | Table 1 (leaderboard), RQ1 |
| E2 | Gap decomposition (PAD, probes, silhouettes) | `src.domain_probe --encoder X` | Fig 2, RQ2 |
| E3 | Per-event / per-morphology breakdown | in E1 report (`per_event_*`) | Fig 3, RQ3 |
| E4 | Bridging ablations + re-eval | `src.bridge ...` then `src.evaluate --emb ...` | Table 2, Fig 4, RQ4 |
| E5 | Instance-level paired retrieval | in E1 report (`instance`) | Table 3, RQ5 |
| E6 | Synthesizer-generalization (train bridge on Stable-Audio synth, test on AudioGen synth) | bridge on one source, eval on held-out source | Table 4 (generalization), pre-empts "overfit to one generator" |

Paper skeleton: Intro (the identifiability hook) → Related work (the three bodies in §1) → Benchmark (data, tasks, protocol, datasheet) → Baselines & gap analysis (E1–E3) → Method & ablations (E4, E6) → Instance-level (E5) → Limitations → Release.

---

## 9. Reproducibility & engineering standards

- **One source of truth:** `data/manifest.csv`. Every script joins on `clip_id`. Embeddings are cached as aligned `.npz` (`ids` + `emb`).
- **Determinism:** single `SEED` in `config.py`; `seed_everything` in `src/utils.py`; hash-bucketed splits; logged generation seeds.
- **Fixed pre-processing:** every clip mono, resampled, peak-normalized, padded/center-cropped to a fixed ≤5 s window (`src/utils.load_audio`) so the gap is never confounded by duration or loudness differences between domains — a subtle but real trap.
- **Confidence intervals:** bootstrap over queries (`src/evaluate`), report 95% CIs on every headline mAP; an improvement without overlapping-CI separation is not a result.
- **Compute budget (estimate):** embedding ~50k clips with CLAP ≈ 1–2 GPU-hours; bridging head trains in minutes on cached vectors; synthetic generation dominates (~0.5–2 s/clip on one GPU → a day for tens of thousands). All feasible on a single A100/3090-class GPU; no multi-node training.
- **Tests as a gate:** `tests/test_metrics.py` (hand-checked metric values) and `tests/test_pipeline_synthetic.py` (full pipeline on synthetic embeddings; also asserts the gap metric *responds* to a planted shift) must pass in CI.

---

## 10. Work breakdown & ~10-week timeline

| Wk | Milestone | Owner |
|----|-----------|-------|
| 1 | Repo + env + tests green; download DCASE-T7; remap class folders to `config` names | Eng |
| 2 | Manifest v1 (core 7 classes), splits + duplicate audit | Eng |
| 3 | Embedding pipeline for CLAP + PANNs; E1/E2 on core corpus | Eng + Research |
| 4 | Synthetic generation (Stable Audio Open) for extension classes; instance pairs (captions → gen) | Eng |
| 5 | Manifest v2 (≥19 classes, paired set); re-run E1–E3, first leaderboard | Research |
| 6 | Activate BEATs/M2D/AudioMAE loaders; full leaderboard | Eng |
| 7 | Bridging method + ablation grid (E4) | Research |
| 8 | Synthesizer-generalization (E6) + instance-level (E5); UMAP figures | Research |
| 9 | Datasheet, license clearance, release packaging (manifest + gen-log + scripts) | Eng + Legal |
| 10 | Writing, CI freeze, internal review, submit | All |

Parallelizable: data/eng track (wks 1–6, 9) vs method/analysis track (wks 7–8) can overlap once Manifest v1 exists.

---

## 11. Risks & mitigations

- **"Off-the-shelf already solves it."** Then C1+C2 (a clean benchmark + the first quantification + per-morphology analysis) still stands as a D&B contribution, and C3 becomes "and it's near-saturated, here's the residual hard slice (instance-level, transient sounds)." Either outcome is publishable — design the framing so both win.
- **Synthetic too easy or too hard.** Calibrate with multiple synthesizers and the fidelity knob is observable via PAD; report the spectrum rather than a single point. The DCASE generated set already spans many synthesis methods.
- **Leakage.** §4.5 disjoint splits + cosine duplicate audit + a held-out-source generalization test (E6).
- **License exposure.** §4.4: ship manifests + generation recipe, not restricted audio; datasheet; pick a permissive code license.
- **Reviewer: "why retrieval, not detection?"** Because detection rewards exploiting the gap (already done); retrieval/invariance rewards *removing* it, which is the useful capability for SFX search, dataset dedup, and Foley tooling. Say this explicitly.

---

## 12. Target venues & positioning

- **NeurIPS / ICLR Datasets & Benchmarks track** — strongest fit for the benchmark + leaderboard + datasheet (C1, C2, C4).
- **ICASSP / INTERSPEECH / WASPAA** — fit for the method + analysis framing (C2, C3); WASPAA especially for the audio-representation angle.
- **DCASE Workshop** — natural community fit given the corpus lineage; good for an early version / to recruit external baselines.
- **ISMIR** — if you spin out the music-subset (piano/drum/bell) cross-domain results as a focused companion.

What makes it "top": a crisp, falsifiable hook (identity-vs-rendering identifiability), a *reusable* artifact with a leaderboard, an analysis that exposes a real limitation of audio foundation models, and a method with clean ablations — not just a number.

---

## 13. Seed bibliography (read/cite)

- FSD50K (Fonseca et al.) — real SFX dataset, AudioSet ontology, CC-licensed.
- ESC-50 (Piczak) — 5 s environmental clips; UrbanSound8K (Salamon et al.).
- DCASE-2023 Task-7 Foley Sound Synthesis (Choi et al.) — the core corpus.
- "Detection of Deepfake Environmental Audio" — the mirror-image (detection) framing on the same data; cite to contrast.
- Foley-vs-real perception studies (e.g., "Real, Foley or synthetic? An evaluation of everyday walking sounds") — the human-indistinguishability hook.
- CLAP (Wu et al. / Elizalde et al.); BEATs (Chen et al.); M2D (Niizumi et al.); AudioMAE (Huang et al.); PANNs (Kong et al.) — encoders.
- HEAR (Turian et al.); X-ARES; MSEB (NeurIPS 2025) — embedding benchmarks to position against.
- Stable Audio Open (Evans et al., arXiv) — the synthesizer.
- Domain adaptation: DANN (Ganin & Lempitsky 2015); Deep CORAL (Sun & Saenko 2016); Proxy-A-distance (Ben-David et al. 2010); SupCon (Khosla et al. 2020); IRM (Arjovsky et al. 2019).

*(Pull exact citations/years from the sources as you write; this is the reading list, not the .bib.)*
