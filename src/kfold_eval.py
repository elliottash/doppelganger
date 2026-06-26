"""k-fold leave-classes-out evaluation with bootstrap CIs. For each fold, the head was trained
with that fold's categories held out; we evaluate cross-domain retrieval on exactly those
held-out (unseen) categories. Compares the `instance` objective (learns the mapping) vs
`class`-supcon (learns clusters) vs frozen, pooled across folds.

Metrics on held-out categories (gallery = held-out real test clips):
  category-mAP : retrieve any same-category real clip
  instance-R@1 : retrieve the clip's EXACT real twin
"""
from __future__ import annotations
import csv, json
import numpy as np
from src import metrics as M

E = "/home/elliott/synthmatch_data/embeddings"
MANI = "/home/elliott/synthmatch_data/manifest_ucs_paired.csv"
ROWS = list(csv.DictReader(open(MANI)))
FOLDS = json.load(open("/tmp/folds.json"))


def _vecs(npz):
    d = np.load(npz, allow_pickle=True)
    return {c: v for c, v in zip(d["ids"], d["emb"])}


def fold_scores(npz, held):
    v = _vecs(npz); held = set(held)
    sI = [r for r in ROWS if r["split"] == "test" and r["domain"] == "synth" and r["event"] in held and r["clip_id"] in v]
    rI = [r for r in ROWS if r["split"] == "test" and r["domain"] == "real" and r["event"] in held and r["clip_id"] in v]
    S = np.stack([v[r["clip_id"]] for r in sI]).astype(np.float64)
    R = np.stack([v[r["clip_id"]] for r in rI]).astype(np.float64)
    sims = S @ R.T
    sev = np.array([r["event"] for r in sI]); rev = np.array([r["event"] for r in rI])
    sii = np.array([int(r["instance_id"]) for r in sI]); rii = np.array([int(r["instance_id"]) for r in rI])
    cat = M.evaluate_retrieval(sims, sev[:, None] == rev[None, :])["mAP"]
    ins = M.evaluate_retrieval(sims, sii[:, None] == rii[None, :])
    return cat, ins["R@1"], ins["MRR"]


def main():
    rows_out = {"frozen": [], "class": [], "instance": []}
    for i, held in enumerate(FOLDS):
        rows_out["frozen"].append(fold_scores(f"{E}/clap_general_ucs_paired.npz", held))
        rows_out["class"].append(fold_scores(f"{E}/clap_general_ucs_paired_kf{i}_class.npz", held))
        rows_out["instance"].append(fold_scores(f"{E}/clap_general_ucs_paired_kf{i}_instance.npz", held))
    print("\n=== 5-fold leave-classes-out (held-out/unseen categories), mean +/- std over folds ===")
    print(f"{'variant':10s} {'cat-mAP':>16s} {'inst-R@1':>16s} {'inst-MRR':>16s}")
    for k in ("frozen", "class", "instance"):
        a = np.array(rows_out[k])  # (folds, 3)
        m, s = a.mean(0), a.std(0)
        print(f"{k:10s} {m[0]:.3f} +/- {s[0]:.3f}   {m[1]:.3f} +/- {s[1]:.3f}   {m[2]:.3f} +/- {s[2]:.3f}")
    json.dump(rows_out, open("results/kfold_scores.json", "w"), indent=2)
    print("\nper-fold instance-R@1 (unseen):")
    for i in range(len(FOLDS)):
        print(f"  fold{i} ({len(FOLDS[i])} cats): frozen {rows_out['frozen'][i][1]:.3f}  "
              f"class {rows_out['class'][i][1]:.3f}  instance {rows_out['instance'][i][1]:.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
