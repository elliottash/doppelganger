"""Fidelity-spectrum evaluation: instance retrieval (find the exact real twin) as a function of
how far the synthetic twin is from its source (init_noise level + text-only).

Shows the instance task is NOT trivially easy (frozen R@1 falls as fidelity drops) and that the
instance head holds up where frozen fails -> the headline isn't an artifact of near-copy twins.

    python -m src.spectrum_eval --frozen clap_general_spectrum.npz --head <instance head.pt>
"""
from __future__ import annotations
import argparse, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import MANIFEST, RESULTS
from src import metrics as M
from src.apply_head import load_head, transform

ORDER = ["n03", "n06", "n09", "n12", "text"]
XLAB = {"n03": "0.3", "n06": "0.6", "n09": "0.9", "n12": "1.2", "text": "text\n(no init)"}


def instance_scores(emb_map, rows, cond):
    q = [r for r in rows if r["domain"] == "synth" and r["cond"] == cond and r["clip_id"] in emb_map]
    g = [r for r in rows if r["domain"] == "real" and r["clip_id"] in emb_map]
    if not q:
        return None
    Q = np.stack([emb_map[r["clip_id"]] for r in q]).astype(np.float64)
    G = np.stack([emb_map[r["clip_id"]] for r in g]).astype(np.float64)
    sims = Q @ G.T
    qi = np.array([int(r["instance_id"]) for r in q]); gi = np.array([int(r["instance_id"]) for r in g])
    res = M.evaluate_retrieval(sims, qi[:, None] == gi[None, :])
    return res["R@1"], res["MRR"]


def twin_fidelity(froz_map, rows, cond):
    """Mean CLAP cosine of each synth twin to its OWN real source — the measured fidelity axis
    (nominal init_noise is not monotonic; init_noise=0.3 underflows sigma_min and degenerates)."""
    real = {int(r["instance_id"]): r["clip_id"] for r in rows if r["domain"] == "real"}
    cos = []
    for r in rows:
        if r["domain"] == "synth" and r["cond"] == cond:
            i = int(r["instance_id"])
            if i in real and r["clip_id"] in froz_map and real[i] in froz_map:
                cos.append(float(froz_map[r["clip_id"]] @ froz_map[real[i]]))
    return float(np.mean(cos)) if cos else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--head", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(MANIFEST)))
    d = np.load(a.frozen, allow_pickle=True)
    froz = {c: v for c, v in zip(d["ids"], d["emb"])}
    head, meta, dev = load_head(a.head)
    ids = list(froz); X = np.stack([froz[i] for i in ids])
    adj = {i: v for i, v in zip(ids, transform(head, dev, X))}

    out = {"frozen": {}, "instance": {}, "fidelity": {}}
    print(f"{'cond':6s} {'twin-fid':>9s} {'frozen R@1':>11s} {'instance R@1':>13s} {'inst MRR':>9s}")
    for c in ORDER:
        f = instance_scores(froz, rows, c); i = instance_scores(adj, rows, c)
        if f is None:
            continue
        fid = twin_fidelity(froz, rows, c)
        out["frozen"][c] = f; out["instance"][c] = i; out["fidelity"][c] = fid
        print(f"{c:6s} {fid:9.3f} {f[0]:11.3f} {i[0]:13.3f} {i[1]:9.3f}")
    json.dump(out, open(RESULTS / "spectrum_scores.json", "w"), indent=2)

    # figure: R@1 vs MEASURED twin fidelity (sorted), not nominal noise
    cs = sorted(out["fidelity"], key=lambda c: out["fidelity"][c])
    fx = [out["fidelity"][c] for c in cs]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(fx, [out["frozen"][c][0] for c in cs], "o-", label="frozen CLAP", color="gray")
    ax.plot(fx, [out["instance"][c][0] for c in cs], "s-", label="instance head", color="tab:red")
    for c in cs:
        ax.annotate(XLAB.get(c, c).split("\n")[0], (out["fidelity"][c], out["instance"][c][0]),
                    textcoords="offset points", xytext=(0, 7), fontsize=8, color="tab:red")
    ax.set_xlabel("twin fidelity  =  mean CLAP cosine(synth twin, its real source)  →")
    ax.set_ylabel("instance retrieval R@1  (find the exact real twin)")
    ax.set_title("Fidelity spectrum: the instance head holds up as twins diverge")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3); ax.legend(loc="center right")
    plt.tight_layout(); plt.savefig(RESULTS / "fig_fidelity.png", dpi=130); plt.close()
    print(f"wrote {RESULTS}/fig_fidelity.png")


if __name__ == "__main__":
    main()
