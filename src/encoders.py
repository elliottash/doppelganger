"""Pluggable audio-encoder interface.

Every encoder exposes the same contract:

    enc = load_encoder("clap_general")
    vecs = enc.embed(list_of_waveforms, sr)   # -> (N, D) float32, L2-normalised

`embed` takes a list of mono float32 numpy waveforms (any sample rate; the encoder
resamples to its own expected rate) and returns one pooled, L2-normalised vector per clip.
L2-normalising here means cosine similarity == dot product everywhere downstream.

CLAP is fully implemented through HuggingFace transformers. The self-supervised encoders
(BEATs / M2D / AudioMAE) are activated below (checkpoints under DATA/ckpts/, fetched by
modal_ssl.py::fetch_ckpts); PANNs loads via panns_inference. The point of the ABC is that
evaluate.py / bridge.py never need to know which encoder produced a matrix.
"""
from __future__ import annotations

import abc
import numpy as np

from config import ENCODERS


def l2norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


class Encoder(abc.ABC):
    dim: int
    sr: int

    @abc.abstractmethod
    def embed(self, waveforms: list[np.ndarray], sr: int) -> np.ndarray:
        ...


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    import librosa
    return librosa.resample(wav.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)


# --------------------------------------------------------------------------------------
# CLAP via HuggingFace transformers (fully working)
# --------------------------------------------------------------------------------------
class ClapHF(Encoder):
    def __init__(self, hf_id: str, dim: int, sr: int, device: str | None = None, batch_size: int = 32):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(hf_id).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(hf_id)
        self.dim, self.sr, self.bs = dim, sr, batch_size

    def embed(self, waveforms, sr):
        out = []
        for i in range(0, len(waveforms), self.bs):
            chunk = [ _resample(w, sr, self.sr) for w in waveforms[i:i + self.bs] ]
            inputs = self.processor(audios=chunk, sampling_rate=self.sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.inference_mode():
                feats = self.model.get_audio_features(**inputs)  # already L2-normalised by CLAP
            out.append(feats.float().cpu().numpy())
        return l2norm(np.concatenate(out, axis=0))

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """CLAP text embeddings in the SHARED audio-text space (for zero-shot verification:
        cosine(audio, text) == how well a clip matches a category description)."""
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            feats = self.model.get_text_features(**inputs)
        return l2norm(feats.float().cpu().numpy())


class ASTHf(Encoder):
    """Audio Spectrogram Transformer (MIT/ast-finetuned-audioset). Supervised AudioSet ViT over
    log-mel patches; mean-pool the token hidden states -> 768-d. Clean transformers API."""
    def __init__(self, hf_id: str, dim: int, sr: int, device: str | None = None, batch_size: int = 16):
        import torch
        from transformers import ASTModel, AutoFeatureExtractor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ASTModel.from_pretrained(hf_id).to(self.device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(hf_id)
        self.dim, self.sr, self.bs = dim, sr, batch_size

    def embed(self, waveforms, sr):
        out = []
        for i in range(0, len(waveforms), self.bs):
            chunk = [_resample(w, sr, self.sr) for w in waveforms[i:i + self.bs]]
            inp = self.fe(chunk, sampling_rate=self.sr, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with self.torch.inference_mode():
                h = self.model(**inp).last_hidden_state.mean(dim=1)  # (B, 768)
            out.append(h.float().cpu().numpy())
        return l2norm(np.concatenate(out, axis=0))


class ClapLaionCkpt(Encoder):
    """LAION CLAP loaded from a raw .pt checkpoint via the `laion_clap` pip package
    (needed for the music/speech/audioset checkpoints not mirrored on the HF hub)."""
    def __init__(self, amodel: str, ckpt: str, dim: int, sr: int, device: str | None = None):
        import torch, laion_clap
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel=amodel, device=self.device)
        self.model.load_ckpt(ckpt)           # download from the LAION-AI/CLAP release page
        self.dim, self.sr = dim, sr

    def embed(self, waveforms, sr):
        wavs = np.stack([_resample(w, sr, self.sr) for w in waveforms]).astype(np.float32)
        emb = self.model.get_audio_embedding_from_data(x=wavs, use_tensor=False)
        return l2norm(np.asarray(emb))


# --------------------------------------------------------------------------------------
# Self-supervised encoders (BEATs / M2D / AudioMAE) -- activated.
# Each returns frame embeddings; we mean-pool over time to a clip vector.
# Checkpoints live under DATA/ckpts/<encoder-name>/ (on Modal: /data/ckpts/...; see
# modal_ssl.py::fetch_ckpts which downloads them once to the volume).
# --------------------------------------------------------------------------------------
def _ckpt_path(name: str, ckpt: str):
    """Resolve a registry `ckpt` entry: absolute paths pass through; otherwise look under
    DATA/ckpts/<encoder name>/."""
    from pathlib import Path
    from config import DATA
    p = Path(ckpt)
    return p if p.is_absolute() else (DATA / "ckpts" / name / ckpt)


class _SSLStub(Encoder):
    name = "ssl"
    def __init__(self, ckpt: str, dim: int, sr: int):
        self.ckpt, self.dim, self.sr = ckpt, dim, sr
        self._model = None  # lazy

    def _load(self):
        raise NotImplementedError(
            f"Activate {self.name}: load the model from {self.ckpt} and set self._model. "
            "See the docstring for the upstream repo/checkpoint."
        )

    def embed(self, waveforms, sr):
        if self._model is None:
            self._load()
        # Expected per-encoder forward: frames (T, D) -> mean-pool -> (D,)
        vecs = [self._frame_features(_resample(w, sr, self.sr)).mean(0) for w in waveforms]
        return l2norm(np.stack(vecs))

    def _frame_features(self, wav: np.ndarray) -> np.ndarray:  # (T, D)
        raise NotImplementedError


class BEATs(_SSLStub):
    """BEATs (Microsoft unilm). Checkpoint BEATs_iter3_plus_AS2M.pt (pretrained iter3+, AS2M)
    from github.com/microsoft/unilm/tree/master/beats (OneDrive links; mirrored on the HF hub
    at datasets/Bencr/beats-checkpoints). The repo's standalone inference code (BEATs.py,
    backbone.py, modules.py -- no fairseq needed) must be on sys.path; set $BEATS_CODE_DIR
    (default /opt/beats, where modal_ssl.py's image fetches it)."""
    name = "beats"

    def _load(self):
        import os, sys
        import torch
        code = os.environ.get("BEATS_CODE_DIR", "/opt/beats")
        if code not in sys.path:
            sys.path.insert(0, code)
        from BEATs import BEATs as _BEATs, BEATsConfig  # noqa: N811
        ck = torch.load(str(_ckpt_path(self.name, self.ckpt)), map_location="cpu",
                        weights_only=False)
        cfg = BEATsConfig(ck["cfg"])
        m = _BEATs(cfg)
        m.load_state_dict(ck["model"])
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = m.eval().to(self.device)

    def _frame_features(self, wav):
        t = self.torch.from_numpy(np.ascontiguousarray(wav)).float()[None].to(self.device)
        with self.torch.inference_mode():
            feats, _ = self._model.extract_features(t, padding_mask=None)  # (1, T, 768)
        return feats[0].float().cpu().numpy()


class M2D(_SSLStub):
    """M2D (nttcslab/m2d), BASE SSL checkpoint (NOT the CLAP variant):
    m2d_vit_base-80x608p16x16-221006-mr7_enconly (release v0.1.0 zip). Loaded through the
    repo's standalone wrapper examples/portable_m2d.py; the cloned repo dir is $M2D_CODE_DIR
    (default /opt/m2d). forward(wav) -> (1, T, 3840) where 3840 = 5 freq patches x 768; we
    average the freq patches to per-frame 768-d features (registry dim stays 768)."""
    name = "m2d"

    def _load(self):
        import glob, os, sys
        import torch
        code = os.environ.get("M2D_CODE_DIR", "/opt/m2d")
        for p in (code, os.path.join(code, "examples")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from portable_m2d import PortableM2D
        base = _ckpt_path(self.name, self.ckpt)
        if base.is_file():
            weight = str(base)
        else:  # registry names the release prefix; find the extracted .pth under ckpts/m2d/
            hits = sorted(glob.glob(str(base.parent / "**" / "*.pth"), recursive=True))
            hits = [h for h in hits if "clap" not in h.lower()]
            if not hits:
                raise FileNotFoundError(f"no M2D .pth under {base.parent} (run fetch_ckpts)")
            weight = hits[0]
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = PortableM2D(weight_file=weight).eval().to(self.device)

    def _frame_features(self, wav):
        t = self.torch.from_numpy(np.ascontiguousarray(wav)).float()[None].to(self.device)
        with self.torch.inference_mode():
            feats = self._model(t)                    # (1, T, F*768), F freq patches
        f = feats[0].float().cpu().numpy()
        T, D = f.shape
        return f.reshape(T, D // self.dim, self.dim).mean(axis=1)  # (T, 768)


class AudioMAE(_SSLStub):
    """AudioMAE (facebookresearch/AudioMAE) ViT-B/16, AS2M self-supervised pretrain
    (`pretrained.pth`). The official checkpoint is behind a Google Drive link, so we load the
    timm port of the SAME weights from the HF hub: gaunernst/vit_base_patch16_1024_128.audiomae_as2m.
    Preprocessing per the port's model card: kaldi fbank (128 mels, hanning, htk_compat),
    pad/trim to 1024 frames, normalise with (mean, std*2) = (-4.2677393, 4.5689974*2).
    We take patch tokens (64 time x 8 freq), average freq, and keep only the time patches
    covering real (non-padding) frames -> (T, 768) frame features."""
    name = "audiomae"
    MEAN, STD = -4.2677393, 4.5689974
    N_FRAMES, N_MELS = 1024, 128

    def _load(self):
        import timm
        import torch
        ref = self.ckpt if self.ckpt.startswith("hf_hub:") else f"hf_hub:{self.ckpt}"
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = timm.create_model(ref, pretrained=True).eval().to(self.device)

    def _frame_features(self, wav):
        torch = self.torch
        import torch.nn.functional as F
        from torchaudio.compliance import kaldi
        t = torch.from_numpy(np.ascontiguousarray(wav)).float()[None]
        mel = kaldi.fbank(t, htk_compat=True, window_type="hanning",
                          num_mel_bins=self.N_MELS, sample_frequency=float(self.sr))
        n_frames = mel.shape[0]
        if n_frames < self.N_FRAMES:
            mel = F.pad(mel, (0, 0, 0, self.N_FRAMES - n_frames))
        else:
            mel = mel[: self.N_FRAMES]
            n_frames = self.N_FRAMES
        mel = (mel - self.MEAN) / (self.STD * 2)
        x = mel.view(1, 1, self.N_FRAMES, self.N_MELS).to(self.device)
        with torch.inference_mode():
            tok = self._model.forward_features(x)      # (1, prefix + 64*8, 768)
        tok = tok[0, getattr(self._model, "num_prefix_tokens", 0):].float().cpu().numpy()
        grid_t, grid_f = self.N_FRAMES // 16, self.N_MELS // 16      # 64 x 8, row-major
        frames = tok.reshape(grid_t, grid_f, -1).mean(axis=1)        # (64, 768)
        n_valid = max(1, int(np.ceil(n_frames / 16)))                # drop pure-padding patches
        return frames[:n_valid]


class PANNs(Encoder):
    """Supervised baseline (CNN14 trained on AudioSet) via the `panns_inference` package.
    Embedding is the 2048-d penultimate layer. Useful as a 'supervised features' reference;
    expect it to encode production artefacts strongly -> a good worst-case domain-gap point."""
    def __init__(self, ckpt: str, dim: int, sr: int, device: str | None = None):
        import torch
        from panns_inference import AudioTagging
        from config import DATA
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # cache the checkpoint on the (persistent) data root; panns_inference auto-downloads
        # Cnn14_mAP=0.431.pth to this path if missing.
        cache = DATA / "panns_data"; cache.mkdir(parents=True, exist_ok=True)
        self.at = AudioTagging(checkpoint_path=str(cache / ckpt), device=self.device)
        self.dim, self.sr = dim, sr

    def embed(self, waveforms, sr):
        wavs = np.stack([_resample(w, sr, self.sr) for w in waveforms]).astype(np.float32)
        _, emb = self.at.inference(wavs)   # (clipwise_output, embedding)
        return l2norm(np.asarray(emb))


_KIND_TO_CLASS = {
    "clap_hf": ClapHF,
    "ast_hf": ASTHf,
    "clap_laion_ckpt": ClapLaionCkpt,
    "beats": BEATs,
    "m2d": M2D,
    "audiomae": AudioMAE,
    "panns": PANNs,
}


def load_encoder(name: str, **kwargs) -> Encoder:
    spec = dict(ENCODERS[name])
    kind = spec.pop("kind")
    cls = _KIND_TO_CLASS[kind]
    # Pass only the kwargs each class accepts (specs are intentionally minimal).
    return cls(**{**spec, **kwargs})
