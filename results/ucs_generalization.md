# UCS generalization (leave-6-classes-out: GUN, WATR, BELL, BIRD, DOOR, WHSH)

34-category UCS corpus (FSD50K real + Stable Audio audio-init twins, 10,420 instance pairs).
Held-out categories = never seen in training. Two retrieval regimes:

| head (held-out cats) | category-mAP | instance-R@1 | instance-MRR |
|---|---|---|---|
| frozen CLAP | 0.691 | 0.625 | 0.734 |
| class-supcon (unseen) | 0.513 | 0.321 | 0.428 |
| **instance (unseen)** | 0.536 | **0.827** | **0.886** |
| instance_event (unseen) | 0.451 | 0.335 | 0.441 |
| instance_event (seen, ceiling) | 0.875 | 0.308 | 0.411 |

Full-gallery (34-cat) per-category cross-domain mAP, class-supcon: held-out avg frozen 0.403 ->
seen-when-trained 0.659 -> unseen 0.186 (class clusters don't transfer; can fall below frozen).

## Findings
1. INSTANCE matching generalizes to unseen categories via the pure `instance` objective
   (R@1 0.83 unseen, beats frozen 0.63; class-supcon HURTS to 0.32). Learns the synth->real
   MAPPING, not class identity -> transfers.
2. CATEGORY clustering does NOT generalize (no objective beats frozen on unseen-cat category
   retrieval) but is strong on trained categories (instance_event seen 0.875).
3. Contrast with 7-class DCASE: there a held-out class collapsed to 0.30 (below frozen) — the
   same class-supcon failure; breadth (34 cats) did not fix it, instance pairs did (for the
   instance regime).

## 5-fold leave-classes-out (robust, all 34 categories held out once) — mean ± std
| variant (unseen cats) | category-mAP | instance-R@1 | instance-MRR |
|---|---|---|---|
| frozen | 0.615 ± 0.052 | 0.642 ± 0.094 | 0.731 ± 0.081 |
| class-supcon | 0.459 ± 0.072 | 0.334 ± 0.064 | 0.439 ± 0.061 |
| **instance** | 0.492 ± 0.047 | **0.829 ± 0.045** | 0.888 ± 0.034 |

Dissociation holds in every fold: instance mapping generalizes (R@1 0.83 unseen, +0.19 vs frozen,
all 5 folds); class-supcon hurts instance retrieval; neither transfers category structure.

## Fidelity spectrum (P0a) — instance retrieval R@1 vs measured twin fidelity
Twin fidelity = mean CLAP cosine(synth twin, its real source). (init_noise is non-monotonic:
0.3 underflows sigma_min and degenerates; 0.6 is most faithful.)

| condition | twin-fidelity | frozen R@1 | instance-head R@1 |
|---|---|---|---|
| n06 | 0.73 | 0.552 | 0.980 |
| n09 | 0.64 | 0.387 | 0.914 |
| n12 | 0.55 | 0.273 | 0.704 |
| n03 (degenerate) | 0.12 | 0.001 | 0.001 |
| text (no init) | 0.11 | 0.001 | 0.001 |

The instance head holds up as twins diverge (advantage +0.43..+0.52 over frozen across the
meaningful range); both collapse to chance only when the twin is independent of its source
(text-only / degenerate) — i.e. the head recovers a rendering MAPPING, not near-copies.
See results/fig_fidelity.png.

## Not CLAP-specific (P1) — PANNs CNN14 dissociation on held-out-6 categories
| variant (unseen) | category-mAP | instance-R@1 |
|---|---|---|
| frozen PANNs | 0.587 | 0.655 |
| class-supcon | 0.448 | 0.378 |
| instance | 0.465 | 0.747 |
Same dissociation as CLAP (instance generalizes, class hurts, category doesn't transfer) on a
different supervised 2048-d encoder.

## Cross-generator probe (P2) — ElevenLabs (text-only) twins
670 ElevenLabs twins across 34 categories, captioned from each clip's Freesound title+tags
(text-to-SFX, no audio-init). Instance head trained on Stable Audio audio-init twins.
| EL twins | cat-mAP | inst-R@1 |
|---|---|---|
| frozen CLAP | 0.484 | 0.107 |
| + instance head | 0.223 | 0.099 |
Finding: text-conditioned twins have WEAK instance correspondence (frozen R@1 0.11 vs 0.55-0.98
for audio-init) — the caption is a bottleneck that discards instance detail. So instance-level
pairing requires the generator to condition on the SOURCE AUDIO; the learned mapping is specific
to audio-conditioned rendering, not arbitrary cross-generator transfer. (ElevenLabs twins are
category-appropriate, cat-mAP 0.48, but not renderings of the specific clip.)

## 3rd encoder (AST) — dissociation holds (held-out-6)
| variant (unseen) | cat-mAP | inst-R@1 |
|---|---|---|
| frozen AST | 0.560 | 0.849 |
| class-supcon | 0.439 | 0.380 |
| instance | 0.486 | 0.945 |
Dissociation confirmed on THREE encoders (CLAP, PANNs, AST): instance generalizes (R@1 up),
class-supcon hurts, category doesn't transfer.

## CORRECTED (post-review): full-gallery, committed folds, CIs, CE baseline
Instance R@1 against the FULL real test gallery (N=3,065, chance 0.0003), 5-fold leave-classes-out,
bootstrap 95% CIs. Folds generated deterministically in kfold_eval.make_folds (seed 1234).
| variant (unseen cats) | instance-R@1 (full gallery) | per-fold |
|---|---|---|
| frozen | 0.611 [0.594,0.629] | — |
| class-supcon | 0.269 [0.254,0.285] | — |
| class-CE (classifier baseline) | 0.282 [0.266,0.297] | 0.31/0.25/0.31/0.31/0.18 |
| **instance** | **0.800 [0.786,0.813]** | 0.83/0.71/0.83/0.85/0.73 |
instance > frozen in all 5 folds (min +0.148); > class in all 5 (min +0.493). Both class-label
objectives (supcon, CE) collapse equally -> the failure is class-label supervision, not the
contrastive form. Per-encoder full-gallery instance R@1: CLAP 0.80 / PANNs 0.69 / AST 0.93 (vs
class 0.25/0.32/0.29); PANNs+AST did not do CLAP-verification, so not a filter artifact.
NB the headline restricted-gallery number (0.829) drops only to 0.800 on the full gallery.
