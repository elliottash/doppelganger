"""Triviality controls for the instance-retrieval task: can off-the-shelf near-duplicate
methods solve twin retrieval WITHOUT learned embeddings?

Baselines (all non-learned):
  - logmel : 64-mel log-spectrogram, mean+std over time (128-d), cosine
  - mfcc   : 20 MFCCs + deltas, mean+std (80-d), cosine
  - chroma : Chromaprint/AcoustID fingerprint, prefix bit-similarity (the standard
             audio-dedup fingerprint; what a library dedup tool would use)

Task: synth test query (N=3,065) -> full real test gallery (N=3,065), exact-twin R@1,
identical to the paper's full-gallery protocol. Runs entirely on Modal CPU.

  ~/.venv-modal/bin/modal run modal_baselines.py::run          # features (sharded) + score
"""
from __future__ import annotations
import modal

VOL = "/data"
app = modal.App("doppelganger-baselines")
vol = modal.Volume.from_name("soundmatch-sr-data")

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libchromaprint-tools")
         .pip_install("numpy", "librosa", "soundfile"))

N_MELS, SR, MAX_S = 64, 22050, 10.0
FP_LEN = 120  # int32 words of chromaprint prefix kept


def _test_rows():
    import csv
    rows = list(csv.DictReader(open(f"{VOL}/manifest_ucs_paired.csv")))
    g = [r for r in rows if r["split"] == "test" and r["domain"] == "real"]
    gi = {int(r["instance_id"]) for r in g}
    q = [r for r in rows if r["split"] == "test" and r["domain"] == "synth"
         and int(r["instance_id"]) in gi]
    return q, g


@app.function(image=image, volumes={VOL: vol}, cpu=8.0, timeout=3 * 60 * 60)
def features(shard: int, n_shards: int):
    import json, subprocess
    import numpy as np, librosa
    from pathlib import Path

    q, g = _test_rows()
    todo = (q + g)[shard::n_shards]
    out = {}
    for r in todo:
        p = Path(VOL) / r["path"]
        try:
            wav, _ = librosa.load(str(p), sr=SR, mono=True, duration=MAX_S)
            if len(wav) < SR // 4:
                raise ValueError("too short")
            m = librosa.feature.melspectrogram(y=wav, sr=SR, n_mels=N_MELS)
            lm = np.log(m + 1e-6)
            mf = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=20)
            dm = librosa.feature.delta(mf)
            fp = np.zeros(FP_LEN, np.int64)
            try:
                j = json.loads(subprocess.run(
                    ["fpcalc", "-raw", "-json", "-length", str(int(MAX_S)), str(p)],
                    capture_output=True, text=True, timeout=60).stdout)
                raw = np.asarray(j.get("fingerprint", [])[:FP_LEN], np.int64)
                fp[: len(raw)] = raw
            except Exception:
                pass
            out[r["clip_id"]] = {
                "logmel": np.concatenate([lm.mean(1), lm.std(1)]).astype(np.float32),
                "mfcc": np.concatenate([mf.mean(1), mf.std(1), dm.mean(1), dm.std(1)]).astype(np.float32),
                "chroma": fp}
        except Exception as e:
            print("skip", r["clip_id"], e)
    import numpy as np
    Path(f"{VOL}/baselines").mkdir(exist_ok=True)
    ids = list(out)
    np.savez(f"{VOL}/baselines/feats_{shard:02d}.npz",
             ids=np.array(ids),
             logmel=np.stack([out[i]["logmel"] for i in ids]) if ids else np.zeros((0, 2 * N_MELS)),
             mfcc=np.stack([out[i]["mfcc"] for i in ids]) if ids else np.zeros((0, 80)),
             chroma=np.stack([out[i]["chroma"] for i in ids]) if ids else np.zeros((0, FP_LEN)))
    vol.commit()
    return len(ids)


@app.function(image=image, volumes={VOL: vol}, cpu=8.0, memory=16384, timeout=60 * 60)
def score():
    import glob, json
    import numpy as np

    feats = {}
    for f in sorted(glob.glob(f"{VOL}/baselines/feats_*.npz")):
        d = np.load(f, allow_pickle=True)
        for i, a, b, c in zip(d["ids"], d["logmel"], d["mfcc"], d["chroma"]):
            feats[str(i)] = (a, b, c)
    q, g = _test_rows()
    q = [r for r in q if r["clip_id"] in feats]
    g = [r for r in g if r["clip_id"] in feats]
    qi = np.array([int(r["instance_id"]) for r in q])
    gi = np.array([int(r["instance_id"]) for r in g])

    def cos_r1(idx):
        Q = np.stack([feats[r["clip_id"]][idx] for r in q]).astype(np.float64)
        G = np.stack([feats[r["clip_id"]][idx] for r in g]).astype(np.float64)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-9
        ranks = []
        for s in range(0, len(q), 256):
            sims = Q[s:s + 256] @ G.T
            order = np.argsort(-sims, axis=1)
            ranked = gi[order]
            for row, target in zip(ranked, qi[s:s + 256]):
                ranks.append(int(np.where(row == target)[0][0]) + 1)
        rk = np.array(ranks)
        return {"R@1": float((rk == 1).mean()), "R@5": float((rk <= 5).mean()),
                "MRR": float((1.0 / rk).mean())}

    def chroma_r1():
        Q = np.stack([feats[r["clip_id"]][2] for r in q]).astype(np.int64)
        G = np.stack([feats[r["clip_id"]][2] for r in g]).astype(np.int64)
        # bit-similarity on the shared prefix: 1 - popcount(xor)/32L
        lut = np.array([bin(x).count("1") for x in range(65536)], np.int64)

        def pop(x):
            return (lut[x & 0xFFFF] + lut[(x >> 16) & 0xFFFF]
                    + lut[(x >> 32) & 0xFFFF] + lut[(x >> 48) & 0xFFFF])
        ranks = []
        for s in range(0, len(q), 32):
            x = Q[s:s + 32, None, :] ^ G[None, :, :]
            d = pop(x).sum(-1)  # lower = more similar
            order = np.argsort(d, axis=1)
            ranked = gi[order]
            for row, target in zip(ranked, qi[s:s + 32]):
                ranks.append(int(np.where(row == target)[0][0]) + 1)
        rk = np.array(ranks)
        return {"R@1": float((rk == 1).mean()), "R@5": float((rk <= 5).mean()),
                "MRR": float((1.0 / rk).mean())}

    res = {"n_queries": len(q), "gallery_N": len(g), "chance_R@1": 1.0 / len(g),
           "logmel_cos": cos_r1(0), "mfcc_cos": cos_r1(1), "chromaprint": chroma_r1()}
    json.dump(res, open(f"{VOL}/baselines/baseline_scores.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(res, indent=1))
    return res


@app.local_entrypoint()
def run(n_shards: int = 16, skip_features: bool = False):
    if not skip_features:
        counts = list(features.starmap([(s, n_shards) for s in range(n_shards)]))
        print("features done:", sum(counts), "clips")
    print(score.remote())
