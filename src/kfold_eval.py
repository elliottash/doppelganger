"""k-fold leave-classes-out evaluation, reported HONESTLY.

For each fold, the head was trained with that fold's categories held out; I evaluate on exactly
those held-out (unseen) categories. The instance-retrieval gallery is the FULL real test set (all
34 categories as distractors) -- NOT restricted to the held-out categories -- so R@1 is not
inflated by a small, oracle-selected, category-homogeneous pool. I report the gallery size N and
chance = 1/N, bootstrap 95% CIs over queries, per-fold numbers, and the paired instance-vs-class
margin. The restricted-gallery number is also shown for transparency / comparison.

The folds are generated deterministically here (no external /tmp file).

Generalised over encoders: `evaluate_encoder("<enc>_ucs_paired")` expects the cached embeddings
    <enc>_ucs_paired.npz                      (frozen)
    <enc>_ucs_paired_kf%d_class.npz           (class-supcon head, fold %d held out)
    <enc>_ucs_paired_kf%d_instance.npz        (instance head, fold %d held out)
and reports, per variant: full-gallery instance R@1 (+CI, per fold), restricted-gallery R@1,
and category-mAP on the unseen categories (full + restricted gallery).
"""
from __future__ import annotations
import csv, json, random
import numpy as np
from config import EMB, RESULTS
from src import metrics as M

E = str(EMB)
SEED = 1234
K = 5


def make_folds(events, k=K, seed=SEED):
    """Deterministic k-fold partition of the category list (committed, reproducible)."""
    cats = sorted(set(events))
    random.Random(seed).shuffle(cats)
    return [cats[i::k] for i in range(k)]


def _vecs(npz):
    d = np.load(npz, allow_pickle=True)
    return {c: v for c, v in zip(d["ids"], d["emb"])}


def _query_gallery(rows, v, held, full_gallery):
    held = set(held)
    q = [r for r in rows if r["split"] == "test" and r["domain"] == "synth"
         and r["event"] in held and r["clip_id"] in v]
    if full_gallery:
        g = [r for r in rows if r["split"] == "test" and r["domain"] == "real" and r["clip_id"] in v]
    else:
        g = [r for r in rows if r["split"] == "test" and r["domain"] == "real"
             and r["event"] in held and r["clip_id"] in v]
    S = np.stack([v[r["clip_id"]] for r in q]).astype(np.float64)
    G = np.stack([v[r["clip_id"]] for r in g]).astype(np.float64)
    return q, g, S @ G.T


def _per_query_hits(rows, v, held, full_gallery):
    """Return (hit@1 array, rank array, gallery_N) for synth->real instance retrieval on the
    held-out categories. full_gallery=True uses ALL real test clips as the gallery."""
    q, g, sims = _query_gallery(rows, v, held, full_gallery)
    qi = np.array([int(r["instance_id"]) for r in q])
    gi = np.array([int(r["instance_id"]) for r in g])
    order = np.argsort(-sims, axis=1)
    ranked_gi = gi[order]
    hit1 = (ranked_gi[:, 0] == qi).astype(float)
    rank = np.array([int(np.where(ranked_gi[i] == qi[i])[0][0]) + 1 for i in range(len(qi))])
    return hit1, rank, len(g)


def _per_query_ap(rows, v, held, full_gallery):
    """Per-query category Average Precision (relevant = same event) for synth->real retrieval
    on the held-out categories."""
    q, g, sims = _query_gallery(rows, v, held, full_gallery)
    qe = np.array([r["event"] for r in q])
    ge = np.array([r["event"] for r in g])
    rel = qe[:, None] == ge[None, :]
    aps = np.array([M.average_precision(rel[i][np.argsort(-sims[i], kind="stable")])
                    for i in range(len(q))])
    return aps


def _boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_encoder(enc_tag="clap_general_ucs_paired", out_name="kfold_scores.json",
                     verbose=True):
    """Run the full 5-fold leave-classes-out evaluation for one encoder tag; returns the
    results dict and (if out_name) writes it to RESULTS/<out_name>."""
    import config
    rows = list(csv.DictReader(open(config.MANIFEST)))
    folds = make_folds([r["event"] for r in rows])
    variants = {"frozen": f"{enc_tag}.npz",
                "class": f"{enc_tag}_kf%d_class",
                "instance": f"{enc_tag}_kf%d_instance"}
    out = {"encoder": enc_tag, "folds": folds, "gallery": "full (all real test clips)"}
    if verbose:
        print(f"[{enc_tag}] 5-fold leave-classes-out, instance R@1 on UNSEEN categories (FULL gallery)\n")
        print(f"{'variant':9s} {'R@1 (full)':>22s} {'restricted':>10s} {'catmAP':>7s} {'per-fold R@1 (full)':>34s}")
    for name, tag in variants.items():
        ph_full, ph_restr, perfold, Ns = [], [], [], []
        ap_full, ap_restr, perfold_ap = [], [], []
        for i, held in enumerate(folds):
            npz = f"{E}/{tag % i}.npz" if "%d" in tag else f"{E}/{tag}"
            v = _vecs(npz)
            hf, _, N = _per_query_hits(rows, v, held, True)
            hr, _, _ = _per_query_hits(rows, v, held, False)
            af = _per_query_ap(rows, v, held, True)
            ar = _per_query_ap(rows, v, held, False)
            ph_full.append(hf); ph_restr.append(hr); perfold.append(hf.mean()); Ns.append(N)
            ap_full.append(af); ap_restr.append(ar); perfold_ap.append(float(af.mean()))
        allq = np.concatenate(ph_full)
        lo, hi = _boot_ci(allq)
        allap = np.concatenate(ap_full)
        aplo, aphi = _boot_ci(allap)
        out[name] = {"R1_full": float(allq.mean()), "ci95": [lo, hi],
                     "R1_restricted": float(np.concatenate(ph_restr).mean()),
                     "per_fold": [float(x) for x in perfold],
                     "catmAP_full": float(allap.mean()), "catmAP_ci95": [aplo, aphi],
                     "catmAP_restricted": float(np.concatenate(ap_restr).mean()),
                     "catmAP_per_fold": perfold_ap,
                     "n_queries": int(len(allq)), "gallery_N": int(np.mean(Ns))}
        if verbose:
            pf = " ".join(f"{x:.2f}" for x in perfold)
            print(f"{name:9s} {allq.mean():.3f} [{lo:.3f},{hi:.3f}] "
                  f"{np.concatenate(ph_restr).mean():10.3f} {allap.mean():7.3f}   {pf}")
    N = out["instance"]["gallery_N"]
    out["chance_R1"] = 1.0 / N
    if verbose:
        print(f"\ngallery N = {N}  chance R@1 = {1/N:.4f}")
        inst = np.array(out["instance"]["per_fold"]); cls = np.array(out["class"]["per_fold"])
        frz = np.array(out["frozen"]["per_fold"])
        print(f"paired per-fold margin  instance-class: {(inst-cls).mean():+.3f} (min {(inst-cls).min():+.3f}, all>0={(inst>cls).all()})")
        print(f"paired per-fold margin  instance-frozen:{(inst-frz).mean():+.3f} (min {(inst-frz).min():+.3f}, all>0={(inst>frz).all()})")
    if out_name:
        json.dump(out, open(RESULTS / out_name, "w"), indent=2)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-tag", default="clap_general_ucs_paired",
                    help="embedding tag prefix, e.g. beats_ucs_paired")
    ap.add_argument("--out", default=None,
                    help="output json name under results/ (default kfold_scores.json for the "
                         "CLAP default, kfold_scores_<enc>.json otherwise)")
    a = ap.parse_args()
    out_name = a.out or ("kfold_scores.json" if a.enc_tag == "clap_general_ucs_paired"
                         else f"kfold_scores_{a.enc_tag}.json")
    evaluate_encoder(a.enc_tag, out_name)
    print("DONE")


if __name__ == "__main__":
    main()
