"""SSL encoders (BEATs / M2D / AudioMAE) on Modal — kept SEPARATE from modal_app.py so this
work never conflicts with concurrent edits there. Shares the same volume + env conventions.

Functions
  fetch_ckpts()            download the three checkpoints once to the volume (/data/ckpts, HF cache)
  smoke(encoder)           sanity-check an encoder on a few real/synth PAIRS before the full run
  embed(encoder, ...)      embed a manifest (mirrors modal_app.py::embed)
  bridge(...)              train a projection head on cached embeddings (mirrors modal_app.py::bridge)

Typical run (from the repo dir), M=~/.venv-modal/bin/modal:
  $M run modal_ssl.py::fetch_ckpts
  $M run modal_ssl.py::smoke --encoder beats
  $M run modal_ssl.py::embed --encoder beats --corpus dcase
  $M run modal_ssl.py::embed --encoder beats --corpus ucs --manifest /data/manifest_ucs_paired.csv --suffix _ucs_paired
  $M run modal_ssl.py::kfold_heads --encoder beats            # 5 folds x {class,instance} heads
"""
from __future__ import annotations

import pathlib
import modal

_REPO = pathlib.Path(__file__).resolve().parent

# BEATs inference code is standalone (no fairseq); fetch just the module files instead of
# cloning the multi-GB unilm monorepo. M2D ships a portable single-file wrapper in its repo.
_BEATS_FILES = ["BEATs.py", "backbone.py", "modules.py", "quantizer.py", "Tokenizers.py"]
_BEATS_RAW = "https://raw.githubusercontent.com/microsoft/unilm/master/beats"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "wget", "git", "unzip", "ffmpeg")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        "numpy<2", "scipy", "scikit-learn",
        "soundfile", "librosa", "resampy", "tqdm",
        "transformers==4.44.2", "huggingface_hub",
        "timm", "einops", "nnAudio",            # M2D portable wrapper + AudioMAE timm port
        "sentence-transformers==3.0.1",         # imported at module level by portable_m2d
    )
    .run_commands(
        "mkdir -p /opt/beats && cd /opt/beats && "
        + " && ".join(f"wget -q {_BEATS_RAW}/{f}" for f in _BEATS_FILES),
        "git clone --depth 1 https://github.com/nttcslab/m2d /opt/m2d",
    )
    .add_local_dir(str(_REPO), remote_path="/root/app",
                   ignore=["data", "results", "__pycache__", "*.pyc", ".git",
                           "*.zip", "SoundMatch-SR_extracted", ".venv",
                           # concurrently-written / heavy dirs that the code never imports
                           "human_study", "paper", "arxiv", "_archive",
                           "*.wav", "*.pdf", "*.png"])
)

app = modal.App("soundmatch-sr-ssl")
vol = modal.Volume.from_name("soundmatch-sr-data", create_if_missing=True)

VOL = "/data"
ENV = {"SMSR_DATA": VOL, "SMSR_RAW": f"{VOL}/raw/extracted", "HF_HOME": f"{VOL}/hf_cache"}
ENV_UCS = {"SMSR_DATA": VOL, "SMSR_RAW": VOL, "HF_HOME": f"{VOL}/hf_cache",
           "SMSR_MANIFEST": f"{VOL}/manifest_ucs.csv"}


def _env_for(corpus: str) -> dict:
    return ENV_UCS if corpus == "ucs" else ENV


def _setup(corpus: str, manifest: str = ""):
    import os, sys
    env = dict(_env_for(corpus))
    if manifest:
        env["SMSR_MANIFEST"] = manifest
    os.environ.update(env)
    if "/root/app" not in sys.path:
        sys.path.insert(0, "/root/app")


_M2D_RELEASE = ("https://github.com/nttcslab/m2d/releases/download/v0.1.0/"
                "m2d_vit_base-80x608p16x16-221006-mr7_enconly.zip")
_BEATS_HF = ("Bencr/beats-checkpoints", "BEATs_iter3_plus_AS2M.pt")
_AUDIOMAE_HF = "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m"


@app.function(image=image, volumes={VOL: vol}, timeout=60 * 60)
def fetch_ckpts():
    """Download all three checkpoints once to the volume (idempotent)."""
    import os, shutil, subprocess
    os.environ.update(ENV)
    ck = pathlib.Path(VOL) / "ckpts"

    # BEATs: official links are OneDrive (unscriptable); use the HF hub mirror.
    beats_dir = ck / "beats"; beats_dir.mkdir(parents=True, exist_ok=True)
    dst = beats_dir / _BEATS_HF[1]
    if not dst.exists():
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=_BEATS_HF[0], filename=_BEATS_HF[1], repo_type="dataset")
        shutil.copy(p, dst)
    print(f"beats ckpt: {dst} ({dst.stat().st_size/1e6:.0f} MB)")

    # M2D base (non-CLAP) enc-only checkpoint from the official GitHub release.
    m2d_dir = ck / "m2d"; m2d_dir.mkdir(parents=True, exist_ok=True)
    if not list(m2d_dir.rglob("*.pth")):
        z = m2d_dir / "m2d.zip"
        subprocess.run(["wget", "-q", "-O", str(z), _M2D_RELEASE], check=True)
        subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(m2d_dir)], check=True)
        z.unlink()
    print("m2d ckpt:", [str(p) for p in m2d_dir.rglob("*.pth")])

    # AudioMAE: timm port on the HF hub; instantiating caches the weights in HF_HOME (volume).
    import timm
    timm.create_model(_AUDIOMAE_HF, pretrained=True)
    print("audiomae weights cached in", os.environ["HF_HOME"])
    vol.commit()


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", timeout=60 * 60)
def smoke(encoder: str = "beats", n_pairs: int = 4):
    """Sanity gate before the full embed run: embeddings must be unit-norm, non-degenerate,
    and rank a clip's paired twin above cross-category clips (pair cos > mismatched cos)."""
    import csv
    import numpy as np
    _setup("ucs", manifest=f"{VOL}/manifest_ucs_paired.csv")
    from config import MANIFEST, TARGET_SR, resolve_audio_path
    from src.encoders import load_encoder
    from src.utils import load_audio

    rows = list(csv.DictReader(open(MANIFEST)))
    by_iid = {}
    for r in rows:
        by_iid.setdefault(r["instance_id"], {})[r["domain"]] = r
    # n_pairs complete pairs from n_pairs different categories (deterministic pick)
    pairs, seen_ev = [], set()
    for iid in sorted(by_iid):
        d = by_iid[iid]
        if "real" in d and "synth" in d and d["real"]["event"] not in seen_ev:
            seen_ev.add(d["real"]["event"]); pairs.append((d["real"], d["synth"]))
        if len(pairs) >= n_pairs:
            break
    wavs, labels = [], []
    for real, synth in pairs:
        for r in (real, synth):
            wavs.append(load_audio(str(resolve_audio_path(r["path"]))))
            labels.append(f"{r['event']}/{r['domain']}")
    enc = load_encoder(encoder)
    E = enc.embed(wavs, sr=TARGET_SR)
    norms = np.linalg.norm(E, axis=1)
    sims = E @ E.T
    pair_cos = np.array([sims[2 * i, 2 * i + 1] for i in range(len(pairs))])
    off = np.array([sims[2 * i, 2 * j + 1] for i in range(len(pairs))
                    for j in range(len(pairs)) if i != j])
    spread = float(E.std(axis=0).mean())
    print(f"encoder={encoder}  shape={E.shape}  norms=[{norms.min():.4f},{norms.max():.4f}]")
    print(f"per-dim spread (std over clips, mean over dims) = {spread:.4f}")
    print(f"pair cos      = {np.round(pair_cos, 3)}  (mean {pair_cos.mean():.3f})")
    print(f"cross-cat cos = mean {off.mean():.3f}  max {off.max():.3f}")
    for lbl, row in zip(labels, np.round(sims, 3)):
        print(f"  {lbl:12s} {row}")
    ok = (E.shape[1] == enc.dim and abs(float(norms.mean()) - 1) < 1e-3
          and spread > 1e-3 and pair_cos.mean() > off.mean())
    print("SMOKE", "PASS" if ok else "FAIL")
    if not ok:
        raise RuntimeError(f"smoke test failed for {encoder}")


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", cpu=16.0, timeout=6 * 60 * 60)
def embed(encoder: str = "beats", corpus: str = "dcase", manifest: str = "", suffix: str = ""):
    """Embed a manifest with an SSL encoder (mirror of modal_app.py::embed)."""
    _setup(corpus, manifest)
    print(f"embedding with {encoder} (corpus={corpus}, manifest={manifest or 'default'})")
    suf = suffix if suffix else ("" if corpus == "dcase" else f"_{corpus}")
    from src import embed as E
    E.main(encoder, out_suffix=suf)
    vol.commit()


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", timeout=2 * 60 * 60)
def bridge(encoder: str = "beats", supcon: float = 1.0, dann: float = 0.0,
           coral: float = 0.0, irm: float = 0.0, objective: str = "invariant",
           epochs: int = 50, tag: str = "", holdout_track: str = "",
           holdout_events: str = "", corpus: str = "ucs", manifest: str = "",
           emb_suffix: str = ""):
    """Train a projection head on cached embeddings (mirror of modal_app.py::bridge)."""
    _setup(corpus, manifest)
    from src import bridge as B
    B.train(encoder + emb_suffix, w_supcon=supcon, w_dann=dann, w_coral=coral, w_irm=irm,
            objective=objective, epochs=epochs, tag=tag, holdout_track=holdout_track,
            holdout_events=holdout_events)
    vol.commit()


@app.local_entrypoint()
def kfold_heads(encoder: str = "beats", epochs: int = 50):
    """Train the 5-fold leave-classes-out heads for one encoder on the UCS paired corpus:
    objectives class (=class-supcon, bridge objective 'invariant') and instance, tags matching
    src/kfold_eval.py ('<enc>_ucs_paired_kf%d_class' / '_instance'). Folds are the committed
    deterministic partition (kfold_eval.make_folds, seed 1234)."""
    import csv, random
    # folds from the LOCAL copy of the paired manifest (same file as on the volume);
    # stdlib-only replica of src.kfold_eval.make_folds (the modal venv has no numpy)
    local_manifest = pathlib.Path.home() / "data" / "doppelganger" / "manifest_ucs_paired.csv"
    rows = list(csv.DictReader(open(local_manifest)))
    cats = sorted({r["event"] for r in rows})
    random.Random(1234).shuffle(cats)
    folds = [cats[i::5] for i in range(5)]
    print("folds:", folds)
    jobs = []
    for i, held in enumerate(folds):
        ho = ",".join(held)
        for objective, short in (("invariant", "class"), ("instance", "instance")):
            jobs.append(dict(encoder=encoder, objective=objective, epochs=epochs,
                             tag=f"kf{i}_{short}", holdout_events=ho, corpus="ucs",
                             manifest=f"{VOL}/manifest_ucs_paired.csv",
                             emb_suffix="_ucs_paired"))
    handles = [bridge.spawn(**j) for j in jobs]
    for h, j in zip(handles, jobs):
        h.get()
        print("done:", j["tag"], j["objective"])
    print(f"all {len(jobs)} heads trained for {encoder}")


@app.function(image=image, volumes={VOL: vol}, timeout=60 * 60)
def pack_embeddings(prefixes: str = "beats,m2d,audiomae"):
    """Tar the requested encoders' embedding/head files into /data/embeddings_ssl.tar.gz
    (modal volume get of deep dirs is unreliable; pull the single tarball instead)."""
    import subprocess
    emb = pathlib.Path(VOL) / "embeddings"
    want = [p.strip() for p in prefixes.split(",") if p.strip()]
    files = sorted(f.name for f in emb.iterdir()
                   if any(f.name.startswith(w) for w in want))
    out = f"{VOL}/embeddings_ssl.tar.gz"
    subprocess.run(["tar", "czf", out, "-C", str(emb)] + files, check=True)
    vol.commit()
    print(f"packed {len(files)} files -> {out}")
    for f in files:
        print(" ", f)
