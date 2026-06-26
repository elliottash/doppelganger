"""Apply a trained bridging head (the ADJUSTED EMBEDDING) to new audio or new CLAP vectors.

The head is a frozen projection on top of CLAP, so the adjusted embedding of any clip is just
head(clap_audio_features(clip)). Use this to embed game SFX, score realness, dedup, or match a
generated sound to its real counterpart with the `instance` head (which generalizes to unseen
categories).

    # transform an existing CLAP .npz through a head
    python -m src.apply_head --head data/embeddings/clap_general_ucs_paired_instance.head.pt \
        --emb data/embeddings/clap_general_ucs.npz --out adjusted.npz

    # or embed raw wavs and adjust them in one step
    python -m src.apply_head --head ....head.pt --wavs a.wav b.wav --out adjusted.npz
"""
from __future__ import annotations

import argparse
import numpy as np


def load_head(head_path, device=None):
    import torch
    import torch.nn as nn
    ck = torch.load(head_path, map_location="cpu")

    class Head(nn.Module):
        def __init__(self, d_in, d_hidden=512, d_proj=256, n_events=0):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(),
                nn.Linear(d_hidden, d_proj))
            self.event_head = nn.Linear(d_proj, n_events)
            self.domain_head = nn.Sequential(nn.Linear(d_proj, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, x):
            return nn.functional.normalize(self.backbone(x), dim=-1)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head = Head(ck["d_in"], n_events=ck["n_events"]).to(dev)
    head.load_state_dict(ck["state_dict"]); head.eval()
    return head, ck, dev


def transform(head, dev, X: np.ndarray) -> np.ndarray:
    """X: (N, d_in) CLAP audio features -> (N, d_proj) adjusted, L2-normalised."""
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), 4096):
            xb = torch.tensor(np.asarray(X[i:i + 4096], dtype=np.float32)).to(dev)
            out.append(head(xb).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--emb", help="existing CLAP .npz (ids+emb) to transform")
    ap.add_argument("--wavs", nargs="*", help="raw wavs to embed+transform (needs CLAP)")
    ap.add_argument("--encoder", default="clap_general")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    head, ck, dev = load_head(a.head)
    print(f"head: objective={ck['objective']} tag={ck['tag']} d_in={ck['d_in']}")

    if a.emb:
        d = np.load(a.emb, allow_pickle=True)
        ids, X = d["ids"], d["emb"]
    else:
        from src.encoders import load_encoder
        from src.utils import load_audio
        from config import TARGET_SR
        enc = load_encoder(a.encoder)
        X = enc.embed([load_audio(w) for w in a.wavs], sr=TARGET_SR)
        ids = np.array(a.wavs)
    Z = transform(head, dev, X)
    np.savez(a.out, ids=ids, emb=Z)
    print(f"adjusted {Z.shape} -> {a.out}")


if __name__ == "__main__":
    main()
