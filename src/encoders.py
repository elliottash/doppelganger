"""Pluggable audio-encoder interface.

Every encoder exposes the same contract:

    enc = load_encoder("clap_general")
    vecs = enc.embed(list_of_waveforms, sr)   # -> (N, D) float32, L2-normalised

`embed` takes a list of mono float32 numpy waveforms (any sample rate; the encoder
resamples to its own expected rate) and returns one pooled, L2-normalised vector per clip.
L2-normalising here means cosine similarity == dot product everywhere downstream.

CLAP is fully implemented through HuggingFace transformers. The self-supervised encoders
(BEATs / M2D / AudioMAE) and the PANNs baseline have loader stubs with exact checkpoint
sources documented; fill the two marked lines per encoder to activate them. The point of
the ABC is that evaluate.py / bridge.py never need to know which encoder produced a matrix.
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
# Self-supervised encoders -- loader stubs with exact sources.
# Each returns frame embeddings; we mean-pool over time to a clip vector.
# --------------------------------------------------------------------------------------
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
    """BEATs (Microsoft unilm). Checkpoint BEATs_iter3_plus_AS2M.pt from
    github.com/microsoft/unilm/tree/master/beats . Load with the repo's BEATs class:
        from BEATs import BEATs, BEATsConfig
        ck = torch.load(self.ckpt); cfg = BEATsConfig(ck['cfg']); m = BEATs(cfg)
        m.load_state_dict(ck['model']); m.eval()
        feats, _ = m.extract_features(wav_tensor[None], padding_mask=None)  # (1,T,768)
    """
    name = "beats"


class M2D(_SSLStub):
    """M2D (nttcslab/m2d). Use the repo's portable wrapper `examples/portable_m2d.py`:
        from portable_m2d import PortableM2D; m = PortableM2D(weight_file=self.ckpt)
        feats = m.encode_clap_audio(...) or m.get_timestamp_embeddings(...)  # (T,768)
    """
    name = "m2d"


class AudioMAE(_SSLStub):
    """AudioMAE (facebookresearch/AudioMAE). Build the ViT, load `pretrained.pth`, run the
    encoder on a 128-mel log-spectrogram patchified to 16x16; take the patch tokens -> (T,768).
    """
    name = "audiomae"


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
