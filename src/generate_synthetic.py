"""Generate the 'synthetic' domain with Stable Audio Open 1.0 (text-to-audio).

Why Stable Audio Open: it is open-weight, runs on a single consumer GPU, is stronger on
sound effects / field recordings than on music, and (important for a *releasable* benchmark)
was trained only on CC0 / CC-BY / CC Sampling+ audio. Weights + licence:
    https://huggingface.co/stabilityai/stable-audio-open-1.0
For broader synthetic coverage you can additionally generate with AudioGen (Meta, audiocraft)
or Stable Audio Open Small/SFX; just point a second pass at a different `--backend`.

Reproducibility contract: we log (event, template, adjective, seed, steps, cfg) for every
clip into synth/generation_log.csv and we DO NOT need to redistribute model audio -- anyone
can regenerate bit-comparable clips from the log. (Release the log + code; optionally release
the WAVs under the model's community licence.)

Two modes:
  * category mode (default): N clips per class from prompt templates. Gives category-level
    correspondence to real clips. instance_id = -1.
  * paired mode (--captions captions.csv with columns real_clip_id,event,caption): generate
    one synthetic clip per real clip from that real clip's caption, and emit pairs.csv so the
    synthetic clip shares an instance_id with its real source -> instance-level retrieval.

Usage:
    python -m src.generate_synthetic --per-class 200 --steps 100 --cfg 7
    python -m src.generate_synthetic --captions data/raw/fsd50k_captions.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import random
from pathlib import Path

import numpy as np

from config import (EVENT_CLASSES, EVENT_PHRASE, PROMPT_TEMPLATES, PROMPT_ADJECTIVES,
                    SYNTH, CLIP_SECONDS, SEED)


def build_prompt(event: str, rng: random.Random) -> tuple[str, str, str]:
    template = rng.choice(PROMPT_TEMPLATES)
    adj = rng.choice(PROMPT_ADJECTIVES)
    phrase = EVENT_PHRASE[event]
    return template.format(adj=adj, event_phrase=phrase), template, adj


class StableAudioOpen:
    """Thin wrapper over stable-audio-tools. Install: pip install stable-audio-tools"""
    def __init__(self, device: str | None = None, seconds: float = CLIP_SECONDS):
        import torch
        from stable_audio_tools import get_pretrained_model
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        self.model = self.model.to(self.device)
        self.sr = cfg["sample_rate"]
        self.sample_size = int(seconds * self.sr)
        self.seconds = seconds

    def generate(self, prompt: str, seed: int, steps: int = 100, cfg_scale: float = 7.0) -> np.ndarray:
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": self.seconds}]
        audio = generate_diffusion_cond(
            self.model, steps=steps, cfg_scale=cfg_scale, conditioning=conditioning,
            sample_size=self.sample_size, sigma_min=0.3, sigma_max=500,
            sampler_type="dpmpp-3m-sde", device=self.device, seed=seed,
        )  # (1, C, T)
        wav = audio.squeeze(0).to(self.torch.float32).cpu().numpy()
        wav = wav.mean(axis=0)  # downmix to mono
        peak = np.max(np.abs(wav)) + 1e-9
        return (wav / peak).astype(np.float32)

    def generate_init(self, prompt: str, init_mono: np.ndarray, in_sr: int, seed: int,
                      steps: int = 60, cfg_scale: float = 7.0,
                      init_noise_level: float = 0.6) -> np.ndarray:
        """Audio-to-audio: condition generation on a REAL clip (instance-level pairing).
        The synth twin shares the event identity / coarse character of `init_mono` but is
        rendered by the diffusion model. `init_noise_level` in [~0.3 weak .. ~1.2 strong]
        trades instance-fidelity (low) vs how-synthetic (high)."""
        import torch
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        wav = torch.tensor(np.asarray(init_mono, dtype=np.float32))
        if in_sr != self.sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav, in_sr, self.sr)
        # pad / center-trim to the model window, then make stereo [2, T]
        n = self.sample_size
        if wav.numel() < n:
            wav = torch.nn.functional.pad(wav, (0, n - wav.numel()))
        else:
            s = (wav.numel() - n) // 2
            wav = wav[s:s + n]
        init = wav.unsqueeze(0).repeat(2, 1)  # (2, T)
        conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": self.seconds}]
        audio = generate_diffusion_cond(
            self.model, steps=steps, cfg_scale=cfg_scale, conditioning=conditioning,
            sample_size=self.sample_size, sigma_min=0.3, sigma_max=500,
            sampler_type="dpmpp-3m-sde", device=self.device, seed=seed,
            init_audio=(self.sr, init), init_noise_level=init_noise_level,
        )
        out = audio.squeeze(0).to(self.torch.float32).cpu().numpy().mean(axis=0)
        peak = np.max(np.abs(out)) + 1e-9
        return (out / peak).astype(np.float32)


def _write_wav(path: Path, wav: np.ndarray, sr: int):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr)


def run_category(per_class: int, steps: int, cfg: float, backend: StableAudioOpen):
    rng = random.Random(SEED)
    log_path = SYNTH / "generation_log.csv"
    with open(log_path, "w", newline="") as fh:
        log = csv.writer(fh); log.writerow(["filename", "event", "prompt", "template", "adj", "seed", "steps", "cfg"])
        for event in EVENT_CLASSES:
            for j in range(per_class):
                seed = rng.randint(0, 2**31 - 1)
                prompt, template, adj = build_prompt(event, rng)
                wav = backend.generate(prompt, seed=seed, steps=steps, cfg_scale=cfg)
                fname = f"{event}_{j:05d}.wav"
                _write_wav(SYNTH / event / fname, wav, backend.sr)
                log.writerow([fname, event, prompt, template, adj, seed, steps, cfg])
            print(f"  generated {per_class} for {event}")
    print(f"log -> {log_path}")


def run_paired(captions_csv: Path, steps: int, cfg: float, backend: StableAudioOpen):
    rng = random.Random(SEED)
    pairs_path = SYNTH / "pairs.csv"
    log_path = SYNTH / "generation_log.csv"
    with open(captions_csv) as fh:
        caps = list(csv.DictReader(fh))   # columns: real_clip_id, event, caption
    with open(pairs_path, "w", newline="") as pf, open(log_path, "w", newline="") as lf:
        pairs = csv.writer(pf); pairs.writerow(["synth_filename", "real_clip_id", "instance_id"])
        log = csv.writer(lf); log.writerow(["filename", "event", "prompt", "seed", "steps", "cfg", "real_clip_id"])
        for iid, r in enumerate(caps):
            seed = rng.randint(0, 2**31 - 1)
            event, prompt = r["event"], r["caption"]
            wav = backend.generate(prompt, seed=seed, steps=steps, cfg_scale=cfg)
            fname = f"{event}_pair{iid:06d}.wav"
            _write_wav(SYNTH / event / fname, wav, backend.sr)
            pairs.writerow([fname, r["real_clip_id"], iid])      # synth pair shares instance_id...
            log.writerow([fname, event, prompt, seed, steps, cfg, r["real_clip_id"]])
            # NOTE: you must also stamp the SAME instance_id on the real clip row in the manifest
            # (join real_clip_id -> instance_id) so the pair is linked for instance-level eval.
    print(f"pairs -> {pairs_path}\nlog -> {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--captions", type=str, default=None)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.0)
    a = ap.parse_args()
    be = StableAudioOpen()
    if a.captions:
        run_paired(Path(a.captions), a.steps, a.cfg, be)
    else:
        run_category(a.per_class, a.steps, a.cfg, be)
