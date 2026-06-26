"""SoundMatch-SR on Modal — all GPU work (the local GB10 box overheats under load).

Mirrors the repo's other Modal tool (tools/synth_inverter/modal_train.py): one App, one
persistent Volume holding the corpus + embeddings + HF cache, the code mounted via
add_local_dir, run with the repo's modal venv.

Functions
  stage()            download DCASE-T7 (Zenodo 8091972) into the volume and extract  (CPU)
  embed(encoder)     CLAP/PANNs over the manifest, write <encoder>.npz to the volume   (GPU)
  bridge(...)        train a projection head on cached embeddings, write *_bridged.npz  (GPU)

Typical run (from the repo dir):
  M=~/.venv-modal/bin/modal
  $M run modal_app.py::stage                                   # one-time, ~5 min
  $M volume put soundmatch-sr-data <local manifest.csv> /manifest.csv   # push the manifest
  $M run modal_app.py --encoder clap_general                   # embed (local_entrypoint)
  $M volume get soundmatch-sr-data /embeddings ./pulled        # pull npz back to analyze
"""
from __future__ import annotations

import pathlib
import modal

_REPO = pathlib.Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "wget", "tar", "ffmpeg")  # ffmpeg: decode ElevenLabs mp3 twins
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        "numpy<2", "scipy", "scikit-learn",
        "soundfile", "librosa", "resampy", "tqdm",
        "transformers==4.44.2", "huggingface_hub",
        "panns-inference",
    )
    # mount only the code; heavy/local-only dirs stay out of the image
    .add_local_dir(str(_REPO), remote_path="/root/app",
                   ignore=["data", "results", "__pycache__", "*.pyc", ".git",
                           "*.zip", "SoundMatch-SR_extracted", ".venv"])
)

app = modal.App("soundmatch-sr")
vol = modal.Volume.from_name("soundmatch-sr-data", create_if_missing=True)

# Heavier image for Stable Audio Open generation (kept separate from the embedding image).
gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "numpy<2", "soundfile", "librosa",
                 "resampy", "einops", "stable-audio-tools", "huggingface_hub")
    .add_local_dir(str(_REPO), remote_path="/root/app",
                   ignore=["data", "results", "__pycache__", "*.pyc", ".git",
                           "*.zip", "SoundMatch-SR_extracted", ".venv"])
)
hf_secret = modal.Secret.from_name("huggingface-soundmatch")

# Light image for downloading/unpacking corpora (no torch).
data_image = modal.Image.debian_slim().apt_install("wget", "zip", "unzip")

VOL = "/data"
# env so config.py points all heavy paths at the volume, and HF weights cache there too
ENV = {"SMSR_DATA": VOL, "SMSR_RAW": f"{VOL}/raw/extracted", "HF_HOME": f"{VOL}/hf_cache"}
# the UCS corpus has its own manifest and a different raw root (FSD50K audio under /data/fsd50k)
ENV_UCS = {"SMSR_DATA": VOL, "SMSR_RAW": VOL, "HF_HOME": f"{VOL}/hf_cache",
           "SMSR_MANIFEST": f"{VOL}/manifest_ucs.csv"}


def _env_for(corpus: str) -> dict:
    return ENV_UCS if corpus == "ucs" else ENV

_ZENODO = "https://zenodo.org/records/8091972/files"
_TARBALLS = {
    "Dataset.tar.gz": "DCASE_2023_Challenge_Task_7_Dataset.tar.gz",
    "Submission.tar.gz": "DCASE_2023_Challenge_Task_7_Submission.tar.gz",
    "Baseline.tar.gz": "DCASE_2023_Challenge_Task_7_Baseline.tar.gz",
}


@app.function(image=image, volumes={VOL: vol}, timeout=60 * 60)
def stage():
    """Download + extract the corpus into the volume (idempotent)."""
    import os, subprocess
    raw = pathlib.Path(VOL) / "raw"
    ext = raw / "extracted"
    ext.mkdir(parents=True, exist_ok=True)
    for local, remote in _TARBALLS.items():
        tgz = raw / local
        if not tgz.exists():
            print(f"downloading {remote} ...")
            subprocess.run(["wget", "-q", "-O", str(tgz), f"{_ZENODO}/{remote}?download=1"],
                           check=True)
        print(f"extracting {local} ...")
        subprocess.run(["tar", "xzf", str(tgz), "-C", str(ext)], check=True)
    # tiny meta csv too
    subprocess.run(["wget", "-q", "-O", str(raw / "submission_meta.csv"),
                    f"{_ZENODO}/dcase_task7_submission_meta.csv?download=1"], check=False)
    vol.commit()
    n = sum(1 for _ in ext.rglob("*.wav"))
    print(f"staged. {n} wav files under {ext}")


_FSD_ZENODO = "https://zenodo.org/records/4060432/files"
_FSD_FILES = [
    "FSD50K.dev_audio.z01", "FSD50K.dev_audio.z02", "FSD50K.dev_audio.z03",
    "FSD50K.dev_audio.z04", "FSD50K.dev_audio.z05", "FSD50K.dev_audio.zip",
    "FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip",
    "FSD50K.ground_truth.zip", "FSD50K.metadata.zip",
]


@app.function(image=data_image, volumes={VOL: vol}, timeout=3 * 60 * 60)
def fsd50k_stage():
    """Download FSD50K (24.7 GB multipart zips), reassemble + unzip onto the volume."""
    import subprocess
    root = pathlib.Path(VOL) / "fsd50k"
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # download all parts CONCURRENTLY (Zenodo throttles per-connection; parallel is ~Nx faster).
    # Always wget -c: resumes any partial from a previous run, no-ops a complete file.
    procs = []
    for f in _FSD_FILES:
        print(f"downloading {f} ...")
        procs.append(subprocess.Popen(
            ["wget", "-q", "-c", "-O", str(raw / f), f"{_FSD_ZENODO}/{f}?download=1"]))
    for p in procs:
        if p.wait() != 0:
            raise RuntimeError("a wget failed")
    vol.commit()
    print("all parts downloaded")
    # reassemble split zips and extract (audio + ground truth + metadata)
    for name in ("FSD50K.dev_audio", "FSD50K.eval_audio"):
        out = raw / f"{name}.unsplit.zip"
        if not (root / name).exists():
            print(f"reassembling {name} ...")
            subprocess.run(["zip", "-q", "-s", "0", str(raw / f"{name}.zip"),
                            "--out", str(out)], check=True)
            subprocess.run(["unzip", "-q", "-o", str(out), "-d", str(root)], check=True)
            out.unlink(missing_ok=True)
    for meta in ("FSD50K.ground_truth.zip", "FSD50K.metadata.zip"):
        subprocess.run(["unzip", "-q", "-o", str(raw / meta), "-d", str(root)], check=False)
    vol.commit()
    n = sum(1 for _ in root.rglob("*.wav"))
    print(f"FSD50K staged: {n} wav files under {root}")


def _run_module(encoder: str, corpus: str = "dcase", manifest: str = "", suffix: str = ""):
    """Import the mounted repo and run its embed pipeline against the volume."""
    import os, sys
    env = dict(_env_for(corpus))
    if manifest:
        env["SMSR_MANIFEST"] = manifest
    os.environ.update(env)
    sys.path.insert(0, "/root/app")
    # embeddings are tagged so corpora don't collide
    suf = suffix if suffix else ("" if corpus == "dcase" else f"_{corpus}")
    from src import embed
    embed.main(encoder, out_suffix=suf)
    vol.commit()


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", cpu=16.0, timeout=4 * 60 * 60)
def embed(encoder: str = "clap_general", corpus: str = "dcase", manifest: str = "",
          suffix: str = ""):
    print(f"embedding with {encoder} (corpus={corpus}, manifest={manifest or 'default'})")
    _run_module(encoder, corpus, manifest, suffix)


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", timeout=60 * 60)
def verify_ucs(encoder: str = "clap_general", topk: int = 3,
               template: str = "the sound of {}"):
    """CLAP zero-shot label verification of the UCS corpus -> manifest_ucs_verified.csv."""
    import os, sys
    os.environ.update(ENV_UCS)
    sys.path.insert(0, "/root/app")
    from src import clap_verify
    clap_verify.verify(encoder, suffix="_ucs", topk=topk, template=template)
    vol.commit()


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", timeout=2 * 60 * 60)
def bridge(encoder: str = "clap_general", supcon: float = 1.0, dann: float = 0.0,
           coral: float = 0.0, irm: float = 0.0,
           objective: str = "invariant", epochs: int = 50, tag: str = "",
           holdout_track: str = "", holdout_events: str = "",
           corpus: str = "dcase", manifest: str = "", emb_suffix: str = ""):
    import os, sys
    env = dict(_env_for(corpus))
    if manifest:
        env["SMSR_MANIFEST"] = manifest
    os.environ.update(env)
    sys.path.insert(0, "/root/app")
    from src import bridge as B
    B.train(encoder + emb_suffix, w_supcon=supcon, w_dann=dann, w_coral=coral, w_irm=irm,
            objective=objective, epochs=epochs, tag=tag, holdout_track=holdout_track,
            holdout_events=holdout_events)
    vol.commit()


@app.function(image=gen_image, volumes={VOL: vol}, gpu="A10G", secrets=[hf_secret],
              timeout=30 * 60)
def demo_listen(n_cats: int = 12, steps: int = 50, noise: float = 0.6):
    """Generate ONE real/synth demo pair per category (audio-init) into /data/synth/demo for
    quick listening — separate namespace so it never races the big fan-out."""
    import os, sys, csv, json
    os.environ.update(ENV_UCS); os.environ["SMSR_MANIFEST"] = f"{VOL}/manifest_ucs_verified.csv"
    sys.path.insert(0, "/root/app")
    import soundfile as sf
    from config import MANIFEST, resolve_audio_path, EVENT_PHRASE
    from src.utils import load_audio
    from src.generate_synthetic import StableAudioOpen
    from src.taxonomy_ucs import UCS_CATEGORIES, verify_prompt
    rows = [r for r in csv.DictReader(open(MANIFEST)) if r["domain"] == "real"]
    # one test-split anchor per category
    picks, seen = [], set()
    for r in sorted(rows, key=lambda r: (r["event"], r["clip_id"])):
        if r["split"] == "test" and r["event"] not in seen:
            seen.add(r["event"]); picks.append(r)
        if len(picks) >= n_cats:
            break
    be = StableAudioOpen(seconds=5.0)
    demo = pathlib.Path(VOL) / "synth" / "demo"; demo.mkdir(parents=True, exist_ok=True)
    out = []
    from config import TARGET_SR
    for r in picks:
        real = load_audio(str(resolve_audio_path(r["path"])))
        prompt = verify_prompt(r["event"]) + ", isolated sound effect, dry"
        wav = be.generate_init(prompt, real, TARGET_SR, seed=42, steps=steps, init_noise_level=noise)
        sf.write(str(demo / f"{r['event']}_synth.wav"), wav, be.sr)
        sf.write(str(demo / f"{r['event']}_real.wav"), real, TARGET_SR)
        out.append({"event": r["event"], "name": UCS_CATEGORIES[r["event"]][0],
                    "prompt": prompt, "real_clip_id": r["clip_id"]})
        print(f"  {r['event']:5s} {UCS_CATEGORIES[r['event']][0]}")
    json.dump(out, open(demo / "demo.json", "w"), indent=2)
    vol.commit()
    print(f"demo pairs -> {demo}")


@app.function(image=gen_image, volumes={VOL: vol}, gpu="A10G", secrets=[hf_secret],
              timeout=30 * 60)
def warm_sao():
    """Download Stable Audio Open into the volume HF cache once, before fan-out."""
    import os
    os.environ.update(ENV)
    from stable_audio_tools import get_pretrained_model
    get_pretrained_model("stabilityai/stable-audio-open-1.0")
    vol.commit()
    print("SAO model cached on volume")


@app.function(image=gen_image, volumes={VOL: vol}, gpu="A10G", secrets=[hf_secret],
              cpu=4.0, timeout=4 * 60 * 60)
def gen_pairs(shard: int, n_shards: int, per_class, max_anchors, cc_only: bool,
              steps: int, noise: float, corpus: str = "dcase", manifest: str = "",
              balanced_per_cat=None, out_tag: str = "", mode: str = "init"):
    import os, sys
    env = dict(_env_for(corpus))
    if manifest:
        env["SMSR_MANIFEST"] = manifest          # e.g. the CLAP-verified UCS manifest
    os.environ.update(env)
    sys.path.insert(0, "/root/app")
    from src import gen_pairs as G
    G.run(shard, n_shards, per_class, max_anchors, cc_only, steps, noise=noise,
          balanced_per_cat=balanced_per_cat, out_tag=out_tag, mode=mode)
    vol.commit()


@app.local_entrypoint()
def gen_spectrum(manifest: str = "/data/manifest_ucs_verified.csv", per_cat: int = 40,
                 n_shards: int = 16, steps: int = 50, warm: bool = True):
    """Fidelity sweep: for a balanced subset, generate twins at several init_noise levels plus a
    text-only (no-init) condition, each in its own namespace, for the spectrum experiment."""
    if warm:
        warm_sao.remote()
    conds = [("_n03", 0.3, "init"), ("_n06", 0.6, "init"), ("_n09", 0.9, "init"),
             ("_n12", 1.2, "init"), ("_text", 0.0, "text")]
    for tag, noise, mode in conds:
        args = [(s, n_shards, None, None, False, steps, noise, "ucs", manifest, per_cat, tag, mode)
                for s in range(n_shards)]
        list(gen_pairs.starmap(args))
        print(f"condition {tag} done")
    print("spectrum generation complete")


@app.local_entrypoint()
def main(encoder: str = "clap_general", do_stage: bool = False):
    if do_stage:
        stage.remote()
    embed.remote(encoder)


@app.local_entrypoint()
def generate(n_shards: int = 10, max_anchors: int = 0, per_class: int = 0,
             cc_only: bool = False, steps: int = 60, noise: float = 0.6, warm: bool = True,
             corpus: str = "dcase", manifest: str = ""):
    """Fan out Stable Audio audio-init paired generation across n_shards GPUs (one per shard).
    For the UCS corpus: --corpus ucs --manifest /data/manifest_ucs_verified.csv --n-shards 20."""
    if warm:
        warm_sao.remote()
    pc = per_class or None
    ma = max_anchors or None
    args = [(s, n_shards, pc, ma, cc_only, steps, noise, corpus, manifest)
            for s in range(n_shards)]
    list(gen_pairs.starmap(args))
    print(f"all {n_shards} shards done")
