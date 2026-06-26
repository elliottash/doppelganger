"""Extract and cache embeddings for every clip in the manifest, for a chosen encoder.

Output: data/embeddings/<encoder>.npz holding
    ids   : (N,) clip_id strings, aligned row-for-row with
    emb   : (N, D) float32 L2-normalised embeddings
Row order matches the manifest order, so evaluate.py can join on clip_id safely.

Usage:
    python -m src.embed --encoder clap_general
    python -m src.embed --encoder beats
"""
from __future__ import annotations

import argparse
import csv
import numpy as np

from config import MANIFEST, EMB, resolve_audio_path
from src.encoders import load_encoder
from src.utils import load_audio, seed_everything
from config import SEED


def read_manifest():
    with open(MANIFEST) as fh:
        return list(csv.DictReader(fh))


def _decode(args):
    """Top-level so it's picklable for the process pool. Returns (clip_id, wav | None)."""
    clip_id, path = args
    try:
        return clip_id, load_audio(str(resolve_audio_path(path)))
    except Exception as e:  # noqa: BLE001
        return clip_id, None


def main(encoder_name: str, batch: int = 64, workers: int | None = None, shard: int = 2048,
         out_suffix: str = ""):
    """Audio decode (librosa resample) is the bottleneck and is CPU-bound, so we decode in a
    process pool while the GPU encodes the previous shard's batches. Memory stays bounded by
    processing `shard` clips at a time rather than holding all 31k waveforms at once."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    seed_everything(SEED)
    from config import TARGET_SR
    rows = read_manifest()
    enc = load_encoder(encoder_name)
    workers = workers or min(16, (os.cpu_count() or 4))
    print(f"embedding {len(rows)} clips with {encoder_name} using {workers} decode workers")

    ids, embs = [], []
    items = [(r["clip_id"], r["path"]) for r in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for s in range(0, len(items), shard):
            chunk = items[s:s + shard]
            decoded = list(pool.map(_decode, chunk, chunksize=16))
            buf_id = [cid for cid, w in decoded if w is not None]
            buf_wav = [w for _, w in decoded if w is not None]
            for cid, w in decoded:
                if w is None:
                    print(f"  skip {cid}")
            for b in range(0, len(buf_wav), batch):
                vecs = enc.embed(buf_wav[b:b + batch], sr=TARGET_SR)
                embs.append(vecs); ids.extend(buf_id[b:b + batch])
            print(f"  embedded {min(s + shard, len(items))}/{len(items)}")

    emb = np.concatenate(embs, axis=0).astype(np.float32)
    ids = np.array(ids)
    out = EMB / f"{encoder_name}{out_suffix}.npz"
    np.savez(out, ids=ids, emb=emb)
    print(f"saved {emb.shape} embeddings -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(a.encoder, a.batch, a.workers)
