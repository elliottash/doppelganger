"""Cross-domain retrieval evaluation: synthetic<->real.

Given a cached embedding matrix (from embed.py) and the manifest, evaluate on the TEST split:

  Category-level:
    s2r : query = synthetic test clips,  gallery = real test clips,  relevant = same event
    r2s : query = real test clips,       gallery = synthetic test clips, relevant = same event
  Instance-level (only over clips with instance_id >= 0):
    s2r_pair / r2s_pair : exactly one relevant gallery item (the paired variant)

It also reports a same-domain control (real->real, synth->synth, category-level) so you can
separate "the embedding is generally good" from "the embedding bridges the gap". The headline
gap number is: same-domain mAP minus cross-domain mAP.

Bootstrap 95% CIs are computed by resampling queries.

Usage:
    python -m src.evaluate --encoder clap_general
    python -m src.evaluate --encoder clap_general --emb data/embeddings/clap_general_bridged.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import numpy as np

from config import MANIFEST, EMB, RESULTS, EVENT_CLASSES
from src import metrics as M


def _load(encoder_name, emb_path=None):
    rows = list(csv.DictReader(open(MANIFEST)))
    data = np.load(emb_path or (EMB / f"{encoder_name}.npz"), allow_pickle=True)
    id2vec = {cid: v for cid, v in zip(data["ids"], data["emb"])}
    # keep only manifest rows we have embeddings for, preserving manifest order
    rows = [r for r in rows if r["clip_id"] in id2vec]
    emb = np.stack([id2vec[r["clip_id"]] for r in rows]).astype(np.float64)
    return rows, emb


def _subset(rows, emb, **eq):
    idx = [i for i, r in enumerate(rows) if all(r[k] == v for k, v in eq.items())]
    return [rows[i] for i in idx], emb[idx]


def _arr(rows, key):
    return np.array([r[key] for r in rows])


def _bootstrap_ci(sims, rel, ks, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    nq = sims.shape[0]
    maps = []
    for _ in range(n_boot):
        samp = rng.integers(0, nq, size=nq)
        maps.append(M.evaluate_retrieval(sims[samp], rel[samp], ks=ks)["mAP"])
    lo, hi = np.nanpercentile(maps, [2.5, 97.5])
    return float(lo), float(hi)


def _run_pair(q_rows, q_emb, g_rows, g_emb, level, ks, n_boot):
    sims = q_emb @ g_emb.T
    if level == "category":
        rel = M.relevance_by_label(_arr(q_rows, "event"), _arr(g_rows, "event"))
    else:  # instance
        rel = M.relevance_by_instance(
            _arr(q_rows, "instance_id").astype(int), _arr(g_rows, "instance_id").astype(int))
    res = M.evaluate_retrieval(sims, rel, ks=ks)
    res["mAP_ci95"] = _bootstrap_ci(sims, rel, ks, n_boot=n_boot)
    return res, sims, rel


def per_event_map(q_rows, sims, rel, ks):
    """mAP computed within each event class (queries grouped by event). Iterates the events
    actually present (works for any taxonomy: DCASE-7, the extension classes, or UCS CatIDs)."""
    ev = _arr(q_rows, "event")
    out = {}
    for e in sorted(set(ev.tolist())):
        mask = ev == e
        if mask.sum() == 0:
            continue
        out[e] = M.evaluate_retrieval(sims[mask], rel[mask], ks=ks)["mAP"]
    return out


def evaluate(encoder_name, emb_path=None, ks=(1, 5, 10), n_boot=1000, synth_track=None):
    """synth_track: if set (e.g. 'A'), restrict the synthetic side to that track only -- used
    for the held-out-generator generalization test (E6)."""
    rows, emb = _load(encoder_name, emb_path)
    te = lambda **eq: _subset(rows, emb, split="test", **eq)

    real_rows, real_emb = te(domain="real")
    synth_rows, synth_emb = te(domain="synth")
    if synth_track is not None:
        keep = [i for i, r in enumerate(synth_rows) if r["track"] == synth_track]
        synth_rows = [synth_rows[i] for i in keep]; synth_emb = synth_emb[keep]

    report = {"encoder": encoder_name, "emb": emb_path or f"{encoder_name}.npz",
              "synth_track": synth_track,
              "n_real_test": len(real_rows), "n_synth_test": len(synth_rows)}

    # ---- category-level cross-domain ----
    s2r, s2r_sims, s2r_rel = _run_pair(synth_rows, synth_emb, real_rows, real_emb, "category", ks, n_boot)
    r2s, *_ = _run_pair(real_rows, real_emb, synth_rows, synth_emb, "category", ks, n_boot)
    report["category"] = {"synth->real": s2r, "real->synth": r2s}
    report["per_event_synth->real_mAP"] = per_event_map(synth_rows, s2r_sims, s2r_rel, ks)

    # ---- same-domain control (real->real, exclude self) ----
    rr_sims = real_emb @ real_emb.T
    rr_rel = M.relevance_by_label(_arr(real_rows, "event"), _arr(real_rows, "event"))
    rr_excl = np.eye(len(real_rows), dtype=bool)
    rr = M.evaluate_retrieval(rr_sims, rr_rel, ks=ks, exclude=rr_excl)
    report["control"] = {"real->real": rr}
    report["domain_gap_mAP"] = float(rr["mAP"] - s2r["mAP"])  # the headline gap

    # ---- instance-level (only if paired clips exist) ----
    real_pair = [r for r in real_rows if int(r["instance_id"]) >= 0]
    synth_pair = [r for r in synth_rows if int(r["instance_id"]) >= 0]
    if real_pair and synth_pair:
        rp_emb = np.stack([emb[rows.index(r)] for r in real_pair])
        sp_emb = np.stack([emb[rows.index(r)] for r in synth_pair])
        s2r_p, *_ = _run_pair(synth_pair, sp_emb, real_pair, rp_emb, "instance", ks, n_boot)
        report["instance"] = {"synth->real_paired": s2r_p}

    out = RESULTS / f"{encoder_name}.json"
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("encoder", "domain_gap_mAP")}, indent=2))
    print(f"full report -> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--emb", default=None, help="override embedding npz (e.g. a bridged matrix)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--synth-track", default=None, help="restrict synth side to a track (E6)")
    a = ap.parse_args()
    evaluate(a.encoder, emb_path=a.emb, n_boot=a.n_boot, synth_track=a.synth_track)
