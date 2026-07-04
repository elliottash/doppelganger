"""Extended benchmark metrics on the existing kfold embeddings (CPU, no training).

Adds, per the NeurIPS review sweep:
  1. MRR and R@5 alongside R@1 (the paper promised MRR),
  2. the reverse direction (real->synth) instance retrieval,
  3. per-UCS-category instance R@1 on unseen categories,
  4. a dataset-hygiene (dedup) operating demo: half the queries have their twin
     removed from the gallery; sweep a cosine threshold and report P/R/F1/AUPRC.

Protocol matches src/kfold_eval.py exactly: deterministic folds (seed 1234), queries are
synth test clips of the fold's held-out categories, gallery is the FULL real test set.

Run:  SMSR_DATA=/home/elliott/data/doppelganger python -m src.extra_metrics
"""
from __future__ import annotations
import csv, json, os
from pathlib import Path
import numpy as np

DATA = Path(os.environ.get("SMSR_DATA", "/home/elliott/data/doppelganger"))
EMB = DATA / "embeddings"
MANI = DATA / "manifest_ucs_paired.csv"
RESULTS = Path(__file__).resolve().parent.parent / "results"
SEED, K = 1234, 5

VARIANTS = {"frozen": "clap_general_ucs_paired.npz",
            "class": "clap_general_ucs_paired_kf%d_class.npz",
            "classifier": "clap_general_ucs_paired_kf%d_classifier.npz",
            "instance": "clap_general_ucs_paired_kf%d_instance.npz"}


def make_folds(events, k=K, seed=SEED):
    import random
    cats = sorted(set(events))
    random.Random(seed).shuffle(cats)
    return [cats[i::k] for i in range(k)]


def _vecs(npz):
    d = np.load(npz, allow_pickle=True)
    return {c: v for c, v in zip(d["ids"], d["emb"])}


def _rows():
    return list(csv.DictReader(open(MANI)))


def _ranks(rows, v, held, direction="s2r"):
    """Per-query rank of the true twin. s2r: synth query, full real-test gallery.
    r2s: real query (only those with a twin), full synth-test gallery."""
    held = set(held)
    qd, gd = ("synth", "real") if direction == "s2r" else ("real", "synth")
    g = [r for r in rows if r["split"] == "test" and r["domain"] == gd and r["clip_id"] in v]
    gi = np.array([int(r["instance_id"]) for r in g])
    giset = set(gi.tolist())
    q = [r for r in rows if r["split"] == "test" and r["domain"] == qd
         and r["event"] in held and r["clip_id"] in v and int(r["instance_id"]) in giset]
    S = np.stack([v[r["clip_id"]] for r in q]).astype(np.float64)
    G = np.stack([v[r["clip_id"]] for r in g]).astype(np.float64)
    sims = S @ G.T
    qi = np.array([int(r["instance_id"]) for r in q])
    order = np.argsort(-sims, axis=1)
    ranked = gi[order]
    rank = np.array([int(np.where(ranked[i] == qi[i])[0][0]) + 1 for i in range(len(qi))])
    events = [r["event"] for r in q]
    top1 = sims[np.arange(len(q)), order[:, 0]]
    hit1 = (ranked[:, 0] == qi)
    return rank, np.array(events), top1, hit1, len(g)


def _boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def kfold_metrics(rows, folds):
    out = {}
    for name, tag in VARIANTS.items():
        for direction in ("s2r", "r2s"):
            ranks, Ns = [], []
            for i, held in enumerate(folds):
                npz = EMB / (tag % i if "%" in tag else tag)
                rk, _, _, _, N = _ranks(rows, _vecs(npz), held, direction)
                ranks.append(rk); Ns.append(N)
            rk = np.concatenate(ranks)
            r1, r5, mrr = (rk == 1).astype(float), (rk <= 5).astype(float), 1.0 / rk
            out[f"{name}/{direction}"] = {
                "R@1": float(r1.mean()), "R@1_ci": _boot_ci(r1),
                "R@5": float(r5.mean()), "MRR": float(mrr.mean()), "MRR_ci": _boot_ci(mrr),
                "gallery_N": int(np.mean(Ns)), "n_queries": int(len(rk))}
            print(f"{name:10s} {direction}  R@1 {r1.mean():.3f}  R@5 {r5.mean():.3f}  "
                  f"MRR {mrr.mean():.3f}  (N={int(np.mean(Ns))}, q={len(rk)})")
    return out


def per_category(rows, folds):
    """Unseen-category instance R@1 per UCS category, instance head vs frozen (s2r)."""
    per = {}
    for name in ("frozen", "instance"):
        tag = VARIANTS[name]
        ev_all, hit_all = [], []
        for i, held in enumerate(folds):
            npz = EMB / (tag % i if "%" in tag else tag)
            rk, ev, _, hit, _ = _ranks(rows, _vecs(npz), held, "s2r")
            ev_all.append(ev); hit_all.append(hit)
        ev, hit = np.concatenate(ev_all), np.concatenate(hit_all)
        per[name] = {c: {"R@1": float(hit[ev == c].mean()), "n": int((ev == c).sum())}
                     for c in sorted(set(ev.tolist()))}
    print("\nper-category instance R@1 (unseen cats, s2r): worst 8 for the instance head")
    worst = sorted(per["instance"].items(), key=lambda kv: kv[1]["R@1"])[:8]
    for c, d in worst:
        print(f"  {c:5s} {d['R@1']:.2f} (n={d['n']}, frozen {per['frozen'][c]['R@1']:.2f})")
    return per


def dedup_demo(rows, folds, drop_frac=0.5, seed=7):
    """Hygiene demo: for each unseen-cat synth query, is its real source in the pool?
    Remove the twin from the gallery for a random half of queries; predict `duplicate`
    iff top-1 cosine > theta AND treat the retrieved clip as the source. A prediction is
    a true positive only if the twin was present AND retrieved at rank 1."""
    out = {}
    for name in ("frozen", "instance"):
        tag = VARIANTS[name]
        scores, correct, present = [], [], []
        rng = np.random.default_rng(seed)
        for i, held in enumerate(folds):
            npz = EMB / (tag % i if "%" in tag else tag)
            v = _vecs(npz)
            held = set(held)
            g = [r for r in rows if r["split"] == "test" and r["domain"] == "real" and r["clip_id"] in v]
            gi = np.array([int(r["instance_id"]) for r in g])
            giset = set(gi.tolist())
            q = [r for r in rows if r["split"] == "test" and r["domain"] == "synth"
                 and r["event"] in held and r["clip_id"] in v and int(r["instance_id"]) in giset]
            G = np.stack([v[r["clip_id"]] for r in g]).astype(np.float64)
            for r in q:
                has_twin = rng.random() > drop_frac
                qi = int(r["instance_id"])
                mask = np.ones(len(g), bool)
                if not has_twin:
                    mask &= (gi != qi)
                sims = G[mask] @ np.asarray(v[r["clip_id"]], np.float64)
                j = int(np.argmax(sims))
                scores.append(float(sims[j]))
                correct.append(bool(has_twin and gi[mask][j] == qi))
                present.append(has_twin)
        scores, correct, present = map(np.array, (scores, correct, present))
        # sweep threshold: predict-dup iff score > theta; TP = correct retrieval of a present twin
        ths = np.quantile(scores, np.linspace(0.01, 0.99, 99))
        best = {"f1": -1.0}
        curve = []
        for th in ths:
            pred = scores > th
            tp = float((pred & correct).sum())
            fp = float((pred & ~correct).sum())
            fn = float((~pred & present).sum() + (pred & present & ~correct).sum())
            p = tp / max(tp + fp, 1e-9); rc = tp / max(tp + fn, 1e-9)
            f1 = 2 * p * rc / max(p + rc, 1e-9)
            curve.append({"theta": float(th), "precision": p, "recall": rc, "f1": f1})
            if f1 > best["f1"]:
                best = {"theta": float(th), "precision": p, "recall": rc, "f1": f1}
        out[name] = {"best": best, "n_queries": int(len(scores)),
                     "frac_with_twin": float(present.mean()), "curve": curve[::7]}
        print(f"dedup {name:9s} best-F1 {best['f1']:.3f} (P {best['precision']:.3f} "
              f"R {best['recall']:.3f} @ theta {best['theta']:.3f})")
    return out


def main():
    rows = _rows()
    folds = make_folds([r["event"] for r in rows])
    res = {"kfold_metrics": kfold_metrics(rows, folds),
           "per_category": per_category(rows, folds),
           "dedup": dedup_demo(rows, folds)}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(RESULTS / "extra_metrics.json", "w"), indent=1)
    print(f"\nwrote {RESULTS/'extra_metrics.json'}")


if __name__ == "__main__":
    main()
