# Datasheet — SoundMatch-SR (v1, core corpus)

Following Gebru et al., *Datasheets for Datasets*.

## Motivation
- **Purpose.** Measure whether general-purpose audio embeddings represent the *identity* of a
  sound event invariantly to whether it was synthesized or recorded — operationalized as
  cross-domain (synthetic↔real) retrieval, with a method to make the representation invariant
  (or, deliberately, sensitive) to the rendering process.
- **Who built it.** The SoundMatch-SR authors. v1 is a re-framing of an existing corpus (no new
  audio collection); extensions add generated audio.

## Composition
- **Instances.** Short (≤4 s) mono sound-effect clips, each labelled with an event class, a
  domain (`real`/`synth`), provenance, and (for synthetic) the generator system id.
- **Counts (v1 core).** 31,450 clips: 5,550 real + 25,900 synthetic across 7 event classes
  (dog_bark, footstep, gunshot, keyboard, moving_motor_vehicle, rain, sneeze_cough). Synthetic
  clips come from 37 generator identities (9 Track-A, 27 Track-B challenge systems + 1 baseline).
- **Source corpus.** DCASE-2023 Challenge Task 7 (Foley Sound Synthesis), Zenodo 8091972. Real
  audio originates from FSD50K, UrbanSound8K, BBC Sound Effects, and Freesound; synthetic audio
  is the released challenge submissions.
- **Splits.** Real: DCASE `eval` = test (source-disjoint from `dev` by construction); `dev`
  split train/val by original Freesound recording id. Synthetic: hashed 70/15/15 with all
  generators present in every split; a held-out-generator split (Track-A unseen) is provided for
  the generalization experiment.
- **Labels.** Event class (from the DCASE folder taxonomy); domain; `system_id`, `track`;
  `orig_dataset`, `orig_id` (source provenance from `DevMeta.csv`/`EvalMeta.csv`).
- **Missing data.** Synthetic clips have no source recording id (generated). No instance-level
  (paired) correspondence on the core corpus — DCASE generators are category-conditional.

## Collection process
No primary collection. We programmatically wrap the released archive: walk the directory layout,
join real clips to their provenance metadata, assign leakage-safe splits, and emit a manifest.
A near-duplicate audit (intra-class cross-split cosine > 0.98) is provided.

## Preprocessing
Each clip is loaded mono, resampled to 48 kHz, peak-normalized, and padded/center-cropped to a
fixed 5 s window so the synthetic/real gap is not confounded by duration or loudness. Embeddings
are cached as aligned `(ids, emb)` `.npz` per encoder.

## Uses
- **Intended.** Benchmarking audio representations for production-invariant retrieval; training
  invariance heads; building real-vs-synthetic fidelity metrics for generative SFX evaluation.
- **Not recommended.** Building a deployed "AI-audio detector" — the sensitive head is provided
  as a scientific control, not a forensics tool.

## Composition — UCS corpus (v2, the diverse generalization substrate)
- **Taxonomy.** 34 UCS (Universal Category System) CatIDs, mapping FSD50K's 200 AudioSet labels
  onto the industry-standard SFX taxonomy (`src/taxonomy_ucs.py`), spanning all morphologies.
- **Real.** 13,579 FSD50K clips selected balanced per CatID, then **CLAP-verified** (kept iff the
  clip's audio matches its UCS label at top-5 by CLAP audio-text cosine) → 10,420 verified clips.
  Splits: FSD50K's own dev train/val + eval test, keyed on the Freesound uploader.
- **Synthetic.** 10,420 instance twins: one per verified anchor, generated with Stable Audio Open
  **audio-init** (conditioned on the real clip), sharing the anchor's `instance_id` and split.
  A second generator (ElevenLabs, text-only, captioned from the clip's Freesound title+tags) adds
  ~680 twins across all categories for the cross-generator check. A fidelity-spectrum subset
  (~1,360 anchors) is generated at multiple init_noise levels + text-only.
- **Provenance.** Each real clip carries `orig_dataset=fsd50k`, `orig_id` (Freesound id);
  synthetic carries `system_id` (sao/elevenlabs) and `cond` (fidelity level). All CC.

## Distribution & licensing
- **Code.** MIT/Apache-2.0 (to be finalized at release).
- **Real audio.** Redistribute only CC-licensed sources (FSD50K, UrbanSound8K, Freesound). The
  **662 BBC Sound Effects clips are research-only and are NOT redistributed**; we ship a manifest
  + fetch script (`is_cc=0` rows). 
- **Synthetic audio.** We ship the manifest; the challenge submissions are redistributable under
  the DCASE terms. For generated extension audio we ship the generation log
  (event, prompt, seed, steps, cfg) so clips are bit-reproducible with Stable Audio Open.
- **Maintenance.** Versioned manifests; the code regenerates all derived artifacts from the
  manifest + the source archive.
