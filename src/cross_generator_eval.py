"""Cross-generator transfer matrix: does the learned synthetic->real INSTANCE correspondence
(trained on Stable-Audio-Open audio-init twins) transfer to twin sets from OTHER generator
families?

Twin sets (rows): SAO init_noise 0.6 (deployed operating point) / 0.45 / 0.75, AudioLDM
audio-to-audio (different architecture + training data), and two TEXT-only generators
(Sony Woosh DFlow on whoosh anchors, ElevenLabs) that share captions grounded in each clip's
Freesound metadata but never hear the source audio.

Heads (columns): frozen CLAP; the SAO-trained instance head (all categories,
clap_general_ucs_paired_instance.head.pt); the 5 k-fold leave-classes-out instance heads
pooled over each fold's UNSEEN categories (the paper's protocol, kfold_eval.py); and
optionally an ALDM-trained instance head (the reverse direction of the matrix).

Protocol: synthetic->real instance retrieval. Queries = a set's twins in the TEST split;
gallery = the FULL real test set (N=3,065; chance 1/N), identical to kfold_eval.py. Sets with
no test twins (ElevenLabs was generated on train/val anchors only) are evaluated against the
full real gallery of their own splits, with gallery N reported. Twin fidelity = mean frozen
CLAP cosine(twin, its real source), the measured (not nominal) fidelity axis.

Runs locally on CPU (embeddings + heads must be pulled from the volume first):

    SMSR_DATA=~/data/doppelganger python -m src.cross_generator_eval
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from config import DATA, EMB, RESULTS
from src.apply_head import load_head, transform
from src.kfold_eval import make_folds

# name -> (paired manifest, frozen npz, conditioning label)
SETS = {
    "sao06": ("manifest_ucs_paired.csv", "clap_general_ucs_paired.npz",
              "audio-init (SAO, noise 0.6 = deployed)"),
    "sao45": ("manifest_ucs_paired_sao45.csv", "clap_general_ucs_paired_sao45.npz",
              "audio-init (SAO, noise 0.45 = closer to source)"),
    "sao75": ("manifest_ucs_paired_sao75.csv", "clap_general_ucs_paired_sao75.npz",
              "audio-init (SAO, noise 0.75 = more synthetic)"),
    "aldm":  ("manifest_ucs_paired_aldm.csv", "clap_general_ucs_paired_aldm.npz",
              "audio-init (AudioLDM style_transfer, strength 0.5)"),
    "woosh": ("manifest_ucs_paired_woosh.csv", "clap_general_ucs_paired_woosh.npz",
              "TEXT-only (Woosh DFlow, Freesound-metadata captions; WHSH anchors)"),
    "el":    ("manifest_el.csv", "clap_general_el.npz",
              "TEXT-only (ElevenLabs, Freesound-metadata captions)"),
}
AUDIO_SETS = ("sao06", "sao45", "sao75", "aldm")


def _vecs(npz):
    d = np.load(npz, allow_pickle=True)
    return {c: v for c, v in zip(d["ids"], d["emb"])}


def _r1(qrows, grows, vec_q, vec_g, head=None, dev=None):
    """Instance R@1 (+MRR) for query twin rows against gallery real rows."""
    q = [r for r in qrows if r["clip_id"] in vec_q]
    g = [r for r in grows if r["clip_id"] in vec_g]
    if not q:
        return None
    Q = np.stack([vec_q[r["clip_id"]] for r in q])
    G = np.stack([vec_g[r["clip_id"]] for r in g])
    if head is not None:
        Q = transform(head, dev, Q)
        G = transform(head, dev, G)
    sims = Q.astype(np.float64) @ G.astype(np.float64).T
    qi = np.array([int(r["instance_id"]) for r in q])
    gi = np.array([int(r["instance_id"]) for r in g])
    order = np.argsort(-sims, axis=1)
    ranked = gi[order]
    hit1 = (ranked[:, 0] == qi).astype(float)
    rr = np.array([1.0 / (int(np.where(ranked[i] == qi[i])[0][0]) + 1) if (ranked[i] == qi[i]).any()
                   else 0.0 for i in range(len(qi))])
    return {"R1": float(hit1.mean()), "MRR": float(rr.mean()), "n_q": len(q), "n_g": len(g),
            "hits": hit1}


def _fidelity(vec, qrows, real_by_iid):
    cos = [float(np.dot(vec[r["clip_id"]], vec[real_by_iid[int(r["instance_id"])]]))
           for r in qrows
           if r["clip_id"] in vec and int(r["instance_id"]) in real_by_iid
           and real_by_iid[int(r["instance_id"])] in vec]
    return float(np.mean(cos)) if cos else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--inst-head", default=str(EMB / "clap_general_ucs_paired_instance.head.pt"))
    ap.add_argument("--aldm-head", default=str(EMB / "clap_general_ucs_paired_aldm_instance.head.pt"),
                    help="reverse-direction head trained on ALDM twins (optional; skipped if absent)")
    a = ap.parse_args()

    base_rows = list(csv.DictReader(open(DATA / "manifest_ucs_paired.csv")))
    base_vec = _vecs(EMB / "clap_general_ucs_paired.npz")
    real_rows = [r for r in base_rows if r["domain"] == "real"]
    real_by_split = {}
    for r in real_rows:
        real_by_split.setdefault(r["split"], []).append(r)
    folds = make_folds([r["event"] for r in base_rows])

    heads = {"inst_sao": a.inst_head}
    if Path(a.aldm_head).exists():
        heads["inst_aldm"] = a.aldm_head
    loaded = {k: load_head(p, device="cpu") for k, p in heads.items()}
    kf_heads = []
    for i in range(len(folds)):
        p = EMB / f"clap_general_ucs_paired_kf{i}_instance.head.pt"
        kf_heads.append(load_head(str(p), device="cpu") if p.exists() else None)

    out = {}
    for name in a.sets.split(","):
        mani, npz, cond = SETS[name]
        mpath, epath = DATA / mani, EMB / npz
        if not mpath.exists() or not epath.exists():
            print(f"[skip] {name}: missing {mpath.name if not mpath.exists() else epath.name}")
            continue
        rows = list(csv.DictReader(open(mpath)))
        vec = _vecs(epath)
        synth = [r for r in rows if r["domain"] == "synth" and int(r["instance_id"]) >= 0]
        # query universe: test split if the set has test twins, else its own splits
        test_q = [r for r in synth if r["split"] == "test"]
        if test_q:
            qrows, splits = test_q, ["test"]
        else:
            splits = sorted({r["split"] for r in synth})
            qrows = synth
        grows = [r for s in splits for r in real_by_split.get(s, [])]
        real_by_iid = {int(r["instance_id"]): r["clip_id"] for r in grows
                       if int(r["instance_id"]) >= 0}

        res = {"conditioning": cond, "splits": splits,
               "fidelity_clap_cos": _fidelity({**base_vec, **vec}, qrows, real_by_iid)}
        fr = _r1(qrows, grows, vec, base_vec)
        res["frozen"] = {k: v for k, v in fr.items() if k != "hits"}
        for hname, (head, ck, dev) in loaded.items():
            r = _r1(qrows, grows, vec, base_vec, head=head, dev=dev)
            res[hname] = {k: v for k, v in r.items() if k != "hits"}
        # k-fold pooled UNSEEN-category protocol (test split only; matches kfold_eval.py)
        if test_q and all(h is not None for h in kf_heads):
            hits, nq = [], 0
            for i, held in enumerate(folds):
                fq = [r for r in test_q if r["event"] in set(held)]
                if not fq:
                    continue
                head, ck, dev = kf_heads[i]
                r = _r1(fq, real_by_split["test"], vec, base_vec, head=head, dev=dev)
                if r:
                    hits.append(r["hits"]); nq += r["n_q"]
            if hits:
                allh = np.concatenate(hits)
                res["kfold_unseen_instance"] = {"R1": float(allh.mean()), "n_q": int(nq),
                                                "n_g": len(real_by_split["test"])}
        res["chance"] = 1.0 / res["frozen"]["n_g"]
        out[name] = res
        print(f"{name:6s} frozen R@1={res['frozen']['R1']:.3f} "
              f"inst_sao R@1={res['inst_sao']['R1']:.3f} "
              f"fid={res['fidelity_clap_cos']:.2f} "
              f"(n_q={res['frozen']['n_q']}, N={res['frozen']['n_g']})")

    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "cross_generator.json", "w"), indent=2)
    _write_md(out)
    print(f"-> {RESULTS / 'cross_generator.json'} + cross_generator.md")


def _write_md(out):
    lines = [
        "# Cross-generator transfer matrix (synthetic -> real instance retrieval)",
        "",
        "Queries = a twin set's synthetic clips (test split unless noted); gallery = the FULL",
        "real test set of the same splits (N and chance = 1/N per row). `inst_sao` = the",
        "instance head trained on SAO(0.6) twins; `kfold-unseen` = the 5 leave-classes-out",
        "instance heads pooled over their unseen categories (the paper's headline protocol).",
        "`fidelity` = mean frozen CLAP cosine(twin, its real source).", "",
        "| twin set | conditioning | fidelity | frozen R@1 | inst_sao R@1 | kfold-unseen R@1 | inst_aldm R@1 | n_q | gallery N | chance |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = [k for k in ("sao45", "sao06", "sao75", "aldm", "woosh", "el") if k in out]
    for k in order:
        r = out[k]
        kf = r.get("kfold_unseen_instance")
        al = r.get("inst_aldm")
        lines.append("| {} | {} | {:.2f} | {:.3f} | {:.3f} | {} | {} | {} | {} | {:.4f} |".format(
            k, r["conditioning"], r["fidelity_clap_cos"], r["frozen"]["R1"],
            r["inst_sao"]["R1"], f"{kf['R1']:.3f}" if kf else "—",
            f"{al['R1']:.3f}" if al else "—", r["frozen"]["n_q"], r["frozen"]["n_g"],
            r["chance"]))
    lines += [
        "",
        "Notes: sao06 is the deployed operating point the heads were trained on (its inst_sao",
        "number is therefore within-generator, seen-category); kfold-unseen is the honest",
        "unseen-category protocol. el has no test-split twins, so its row uses train+val",
        "queries against the full train+val real gallery (larger N). woosh twins are",
        "CC-BY-NC research-only (is_cc=0 in the manifest) and cover only the WHSH category.",
    ]
    (RESULTS / "cross_generator.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
