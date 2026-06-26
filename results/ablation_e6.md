## Table 2 — ablation: closing the gap (frozen CLAP, TEST)

| objective | control mAP | synth→real mAP | P@1 | gap | PAD | domain-probe |
|---|---|---|---|---|---|---|
| frozen | 0.935 | 0.774 | 0.829 | +0.161 | 1.27 | 0.902 |
| supcon only | 0.991 | 0.972 | 0.966 | +0.019 | 1.38 | 0.846 |
| supcon+CORAL | 0.988 | 0.971 | 0.964 | +0.017 | 1.38 | 0.844 |
| supcon+DANN | 0.990 | 0.971 | 0.964 | +0.019 | 1.38 | 0.846 |
| supcon+IRM | 0.989 | 0.972 | 0.964 | +0.017 | 1.38 | 0.846 |
| supcon+DANN+IRM (full) | 0.988 | 0.970 | 0.963 | +0.018 | 1.38 | 0.844 |

## Table 4 — E6: generalization to unseen generators (synth = Track A only)

| head | synth→real mAP (Track A) | P@1 | gap |
|---|---|---|---|
| frozen | 0.799 | 0.849 | +0.136 |
| invariant (trained on A+B) | 0.974 | 0.971 | +0.015 |
| invariant_noA (A unseen) | 0.966 | 0.958 | +0.023 |
