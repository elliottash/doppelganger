# SoundMatch-SR

**A benchmark for synthetic↔real sound-effect retrieval — and a dissociation: the rendering
transformation is learnable, the taxonomy is not.**

Can an audio embedding tell that a *synthesized* glass-break and a *recorded* glass-break are the
same kind of sound — or does it just hear "rendered by a diffusion model" vs. "recorded with a
microphone"? SoundMatch-SR measures exactly that: cross-domain (synthetic↔real) retrieval of short
sound effects, with both category-level and instance-level correspondence.

📄 Paper: [`paper/PAPER_NeurIPS.pdf`](paper/) · 🤗 Data: `huggingface.co/datasets/elliottash/soundmatch-sr`

## The headline result

Perceptual studies find synthetic and real Foley of the same event indistinguishable to humans.
In embedding space, I find a **dissociation** on categories *never seen in training*:

| objective (unseen categories) | category-mAP | **instance-R@1** |
|---|---|---|
| frozen CLAP | 0.62 | 0.64 |
| class-supervised | 0.46 | 0.33 (hurts) |
| **instance-contrastive** | 0.49 | **0.83 ± 0.05** |

- **Instance matching generalizes** — trained on real↔synthetic *pairs* (a clip and its generated
  twin), the head learns the synthetic→real *mapping* and transfers to new sound types (R@1 0.83,
  every fold, on CLAP / PANNs / AST).
- **Category clustering does not** — no objective beats frozen on unseen-category retrieval.
- **Class-supervised invariance fails** on unseen categories (collapses below frozen), even at 34
  categories.

See [`results/ucs_generalization.md`](results/ucs_generalization.md) for all numbers.

## What's in the benchmark

- **DCASE-T7 core** (controlled): 7 classes, 5,550 real + 25,900 synthetic clips from 37 generators.
- **UCS corpus** (diverse): 34 Universal-Category-System categories, FSD50K real audio
  (CLAP-verified), 10,420 instance-level Stable-Audio-Open twins (audio-init).
- Leakage-safe splits, a generation recipe, and trained embedding heads + a reusable transform.

## Quickstart

```bash
pip install -r requirements.txt
python tests/test_metrics.py            # retrieval math
python tests/test_pipeline_synthetic.py # end-to-end on synthetic embeddings
```

GPU work (embedding, generation, head training) runs on [Modal](https://modal.com) via
`modal_app.py`; analysis is CPU. See [`RUNBOOK.md`](RUNBOOK.md) for the full pipeline and
[`BENCHMARK_BLUEPRINT.md`](BENCHMARK_BLUEPRINT.md) for the design.

```bash
M=~/.venv-modal/bin/modal
$M run modal_app.py::stage                       # download DCASE-T7 to the volume
$M run modal_app.py --encoder clap_general       # embed
$M run modal_app.py::bridge --encoder clap_general --objective instance --supcon 1 --dann .3
python -m src.kfold_eval                          # the dissociation
```

## Use the adjusted embeddings on your own audio

```python
from src.apply_head import load_head, transform
from src.encoders import load_encoder
from src.utils import load_audio
from config import TARGET_SR

head, meta, dev = load_head("clap_general_ucs_paired_instance.head.pt")  # from the HF dataset
clap = load_encoder("clap_general")
adjusted = transform(head, dev, clap.embed([load_audio("my.wav")], sr=TARGET_SR))
```

## Layout
```
src/            metrics, manifest builders, encoders, embed, evaluate, domain_probe,
                bridge (the heads), taxonomy_ucs, clap_verify, gen_pairs, gen_elevenlabs,
                kfold_eval, spectrum_eval, apply_head
modal_app.py    Modal app: stage / embed / verify / bridge / generate
paper/          LaTeX (NeurIPS format) + bib + compiled PDF
data/manifests/ the manifests (one row per clip; provenance + leakage-safe splits)
results/        figures + result tables
```

## License
Code: MIT. Audio: CC-licensed sources redistributed where permitted; restricted audio (e.g. BBC)
shipped as a fetch manifest + generation recipe rather than files. Synthetic audio generated with
Stable Audio Open (Stability AI Community License). See the datasheet in `paper/DATASHEET.md`.

## Citation
```bibtex
@misc{ash2026soundmatchsr,
  title  = {SoundMatch-SR: The Rendering Transformation Is Learnable, the Taxonomy Is Not},
  author = {Elliott Ash},
  year   = {2026},
  note   = {https://github.com/elliottash/soundmatch-sr}
}
```
