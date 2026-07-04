"""Validate the SENSITIVE head as a per-clip realness/fidelity score (C3).

Two candidate scores from the trained sensitive head:
  (a) its domain-probe logit (domain_head output),
  (b) projection on the real-minus-synth centroid axis in sensitive space
      (centroids from the UCS train split).

Tests, all on frozen-CLAP inputs transformed through the head:
  1. Monotonicity vs operating point on the fidelity-spectrum set
     (cond in {real, n06, n09, n12, text}; n03 is a known-degenerate underflow).
  2. Per-clip Spearman correlation with measured twin fidelity
     (CLAP cosine between each twin and its own real source).
  3. Separation AUCs: real vs each synthetic condition; SAO(n06) vs ElevenLabs.

Run:  SMSR_DATA=/home/elliott/data/doppelganger python -m src.validate_sensitive
"""
from __future__ import annotations
import csv, json, os
from pathlib import Path
import numpy as np

DATA = Path(os.environ.get("SMSR_DATA", "/home/elliott/data/doppelganger"))
EMB = DATA / "embeddings"
RESULTS = Path(__file__).resolve().parent.parent / "results"
HEAD = DATA / "heads_only" / "clap_general_ucs_paired_sensitive.head.pt"


def load(npz):
    d = np.load(EMB / npz, allow_pickle=True)
    return {c: v for c, v in zip(d["ids"], d["emb"])}


def rows(name):
    return list(csv.DictReader(open(DATA / name)))


def head_transform(vecs_dict, head, dev):
    import torch
    ids = list(vecs_dict)
    X = torch.tensor(np.stack([vecs_dict[i] for i in ids]), dtype=torch.float32, device=dev)
    with torch.no_grad():
        Z = head(X)
        logit = head.domain_head(head.backbone(X)).squeeze(-1)
    return ids, Z.cpu().numpy(), logit.cpu().numpy()


def auc(pos, neg):
    """rank-based AUC of score separating pos (higher) from neg."""
    x = np.concatenate([pos, neg])
    r = x.argsort().argsort() + 1
    rp = r[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from apply_head import load_head
    head, meta, dev = load_head(str(HEAD))

    frozen_paired = load("clap_general_ucs_paired.npz")
    spec_rows = rows("manifest_spectrum.csv")
    el_rows = rows("manifest_el.csv")
    spec = load("clap_general_spectrum.npz")
    el = load("clap_general_el.npz")

    # ---- sensitive-space representation of everything ----
    ids_s, Z_s, logit_s = head_transform(spec, head, dev)
    ids_e, Z_e, logit_e = head_transform(el, head, dev)
    zmap = {i: z for i, z in zip(ids_s, Z_s)}; zmap.update({i: z for i, z in zip(ids_e, Z_e)})
    lmap = {i: l for i, l in zip(ids_s, logit_s)}; lmap.update({i: l for i, l in zip(ids_e, logit_e)})

    # centroid axis from the paired TRAIN split (real minus synth), in sensitive space
    paired = rows("manifest_ucs_paired.csv")
    tr = [r for r in paired if r["split"] == "train" and r["clip_id"] in frozen_paired]
    ids_t, Z_t, _ = head_transform({r["clip_id"]: frozen_paired[r["clip_id"]] for r in tr}, head, dev)
    dom_t = {r["clip_id"]: r["domain"] for r in tr}
    zr = Z_t[[dom_t[i] == "real" for i in ids_t]].mean(0)
    zs = Z_t[[dom_t[i] == "synth" for i in ids_t]].mean(0)
    axis = (zr - zs) / (np.linalg.norm(zr - zs) + 1e-9)

    def scores(rws, which):
        sel = [r for r in rws if r["cond"] == which and r["clip_id"] in zmap]
        ax = np.array([float(zmap[r["clip_id"]] @ axis) for r in sel])
        lg = np.array([float(lmap[r["clip_id"]]) for r in sel])
        return sel, ax, lg

    conds = ["real", "n06", "n09", "n12", "text"]
    by_cond = {}
    for c in conds:
        sel, ax, lg = scores(spec_rows, c)
        by_cond[c] = {"axis_mean": float(ax.mean()), "axis_sd": float(ax.std()),
                      "logit_mean": float(lg.mean()), "n": len(sel)}
    sel_el, ax_el, lg_el = scores(el_rows, "elevenlabs")
    by_cond["elevenlabs"] = {"axis_mean": float(ax_el.mean()), "axis_sd": float(ax_el.std()),
                             "logit_mean": float(lg_el.mean()), "n": len(sel_el)}

    # ---- per-clip fidelity (frozen CLAP cosine of twin to its own real source) ----
    real_spec = {int(r["instance_id"]): r["clip_id"] for r in spec_rows if r["cond"] == "real"}
    fid, ax_by_clip = [], []
    for c in ("n06", "n09", "n12"):
        sel, ax, _ = scores(spec_rows, c)
        for r, a in zip(sel, ax):
            src = real_spec.get(int(r["instance_id"]))
            if src and src in spec and r["clip_id"] in spec:
                u, v = spec[src], spec[r["clip_id"]]
                fid.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
                ax_by_clip.append(float(a))
    fid, ax_by_clip = np.array(fid), np.array(ax_by_clip)

    def spearman(a, b):
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        ra, rb = ra - ra.mean(), rb - rb.mean()
        return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-9))

    rho = spearman(ax_by_clip, fid)

    # ---- separation AUCs on the axis score (real = high) ----
    _, ax_real, lg_real = scores(spec_rows, "real")
    aucs = {}
    for c in ("n06", "n09", "n12", "text"):
        _, ax_c, _ = scores(spec_rows, c)
        aucs[f"real_vs_{c}"] = auc(ax_real, ax_c)
    aucs["real_vs_elevenlabs"] = auc(ax_real, ax_el)
    _, ax_n06, _ = scores(spec_rows, "n06")
    aucs["n06_vs_elevenlabs"] = auc(ax_n06, ax_el)   # SAO twins score realer than EL?

    res = {"score_by_cond": by_cond, "spearman_axis_vs_twin_fidelity": rho,
           "n_fidelity_pairs": int(len(fid)), "auc": aucs,
           "head": str(HEAD), "axis": "real-minus-synth centroid, train split"}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(RESULTS / "sensitive_validation.json", "w"), indent=1)

    print("sensitive-axis score by condition (higher = realer):")
    for c in conds + ["elevenlabs"]:
        d = by_cond[c]
        print(f"  {c:12s} axis {d['axis_mean']:+.3f} ± {d['axis_sd']:.3f}   logit {d['logit_mean']:+.2f}   n={d['n']}")
    print(f"\nSpearman(axis score, per-clip twin fidelity) = {rho:+.3f}  (n={len(fid)})")
    for k, v in aucs.items():
        print(f"  AUC {k:22s} {v:.3f}")
    print(f"\nwrote {RESULTS/'sensitive_validation.json'}")


if __name__ == "__main__":
    main()
