"""Score the benchmark models on the IDENTICAL human-study trials (both blocks).

Block "retrieval" (6-way): cosine similarity of the synthetic query embedding
against the 6 real candidate embeddings, argmax = the model's pick. Two models:
  - clap_frozen:    raw CLAP audio embeddings (embeddings/clap_general_ucs_paired.npz)
  - clap_instance:  the same embeddings passed through the trained instance head
                    (heads_only/clap_general_ucs_paired_instance.head.pt, CPU)

Block "2afc" (real vs own twin): both clips' frozen-CLAP embeddings are passed
through the SENSITIVE head (heads_only/clap_general_ucs_paired_sensitive.head.pt)
and projected onto the real-minus-synth centroid axis in sensitive space (centroids
from the paired TRAIN split -- same construction as src/validate_sensitive.py); the
higher-scoring clip is picked as "real".
  - sensitive_axis: machine 2AFC accuracy on the identical pairs

Writes human_study/model_on_trials.json with per-trial picks + summary accuracies.
CPU-only (tiny MLP heads), allowed under the no-local-GPU policy.

Run:  python3 model_on_trials.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # repo root: import src.apply_head
DATA = Path.home() / "data/doppelganger"
EMB = DATA / "embeddings/clap_general_ucs_paired.npz"
MANIFEST = DATA / "manifest_ucs_paired.csv"
INSTANCE_HEAD = DATA / "heads_only/clap_general_ucs_paired_instance.head.pt"
SENSITIVE_HEAD = DATA / "heads_only/clap_general_ucs_paired_sensitive.head.pt"


def main():
    trials = [json.loads(l) for l in open(HERE / "trials.jsonl")]
    retrieval = [t for t in trials if t["block"] == "retrieval" and not t["catch"]]
    afc = [t for t in trials if t["block"] == "2afc"]

    d = np.load(EMB, allow_pickle=True)
    all_ids = list(d["ids"])
    ids = {cid: i for i, cid in enumerate(all_ids)}
    X = d["emb"].astype(np.float32)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)

    from src.apply_head import load_head, transform

    # ---- retrieval block: frozen CLAP + instance head, cosine argmax over 6 ----
    head_i, _, dev = load_head(INSTANCE_HEAD, device="cpu")
    Z = transform(head_i, dev, X)             # already L2-normalised
    out, acc = {}, {"clap_frozen": 0, "clap_instance": 0}
    for t in retrieval:
        qi = ids[t["query_clip_id"]]
        ci = [ids[c["clip_id"]] for c in t["candidates"]]
        rec = {"block": "retrieval", "category": t["category"],
               "answer": t["answer"]}
        for name, M in (("clap_frozen", Xn), ("clap_instance", Z)):
            sims = M[ci] @ M[qi]
            pick = int(np.argmax(sims))
            rec[name] = {"pick": pick, "correct": pick == t["answer"],
                         "sims": [round(float(s), 4) for s in sims]}
            acc[name] += pick == t["answer"]
        out[t["trial_id"]] = rec

    # ---- 2afc block: sensitive head, real-minus-synth train-centroid axis ----
    head_s, _, dev = load_head(SENSITIVE_HEAD, device="cpu")
    Zs = transform(head_s, dev, X)
    rows = list(csv.DictReader(open(MANIFEST)))
    tr_real = [ids[r["clip_id"]] for r in rows if r["split"] == "train"
               and r["domain"] == "real" and r["clip_id"] in ids]
    tr_synth = [ids[r["clip_id"]] for r in rows if r["split"] == "train"
                and r["domain"] == "synth" and r["clip_id"] in ids]
    zr, zs_ = Zs[tr_real].mean(0), Zs[tr_synth].mean(0)
    axis = (zr - zs_) / (np.linalg.norm(zr - zs_) + 1e-9)

    afc_ok = 0
    for t in afc:
        # order as presented (a, b): answer is the index of the REAL clip
        a_id = t["real_clip_id"] if t["answer"] == 0 else t["synth_clip_id"]
        b_id = t["synth_clip_id"] if t["answer"] == 0 else t["real_clip_id"]
        sa = float(Zs[ids[a_id]] @ axis)
        sb = float(Zs[ids[b_id]] @ axis)
        pick = 0 if sa >= sb else 1
        correct = pick == t["answer"]
        afc_ok += correct
        out[t["trial_id"]] = {"block": "2afc", "category": t["category"],
                              "answer": t["answer"],
                              "sensitive_axis": {"pick": pick, "correct": correct,
                                                 "score_a": round(sa, 4),
                                                 "score_b": round(sb, 4)}}

    summary = {m: {"correct": int(k), "n": len(retrieval),
                   "accuracy": round(k / len(retrieval), 4), "block": "retrieval",
                   "chance": round(1 / 6, 4)}
               for m, k in acc.items()}
    summary["sensitive_axis"] = {"correct": afc_ok, "n": len(afc),
                                 "accuracy": round(afc_ok / len(afc), 4),
                                 "block": "2afc", "chance": 0.5}
    with open(HERE / "model_on_trials.json", "w") as fh:
        json.dump({"summary": summary, "trials": out}, fh, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
