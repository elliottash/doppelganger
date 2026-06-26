"""CLAP zero-shot label verification for the UCS corpus.

FSD50K labels are multi-label and weak. We keep a clip only if CLAP agrees its audio matches
its assigned UCS category: cosine(audio_i, text_c) in CLAP's shared space, and the clip's own
CatID must rank in the TOP-K of all categories (lenient, to tolerate legitimate co-labels and
CLAP's per-category miscalibration). Drops the gross mismatches, not subtle ones.

This needs CLAP's text encoder, so it runs where CLAP lives (Modal). It writes a filtered
manifest and prints per-category drop rates so we can sanity-check the filter isn't too harsh.

    python -m src.clap_verify --encoder clap_general --topk 3 --template "the sound of {}"
"""
from __future__ import annotations

import argparse
import csv
import numpy as np

from config import MANIFEST, EMB
from src.taxonomy_ucs import UCS_CATEGORIES, verify_prompt


def _audio_emb(encoder_name, suffix):
    data = np.load(EMB / f"{encoder_name}{suffix}.npz", allow_pickle=True)
    return {cid: v for cid, v in zip(data["ids"], data["emb"])}


def verify(encoder_name="clap_general", suffix="_ucs", topk=3, template="the sound of {}",
           out_manifest=None):
    from src.encoders import load_encoder

    rows = [r for r in csv.DictReader(open(MANIFEST))]
    id2vec = _audio_emb(encoder_name, suffix)
    rows = [r for r in rows if r["clip_id"] in id2vec]

    cats = list(UCS_CATEGORIES)
    ci = {c: i for i, c in enumerate(cats)}
    enc = load_encoder(encoder_name)
    prompts = [template.format(verify_prompt(c)) for c in cats]
    T = enc.embed_text(prompts)                      # (C, D)

    A = np.stack([id2vec[r["clip_id"]] for r in rows]).astype(np.float32)   # (N, D)
    sims = A @ T.T                                   # (N, C) cosine
    # rank of each clip's assigned category (0 = best match)
    order = np.argsort(-sims, axis=1)
    keep = np.zeros(len(rows), dtype=bool)
    rank_of = np.zeros(len(rows), dtype=int)
    for i, r in enumerate(rows):
        c = ci[r["event"]]
        rank_of[i] = int(np.where(order[i] == c)[0][0])
        keep[i] = rank_of[i] < topk

    # per-category drop rate
    from collections import Counter
    tot, dropped = Counter(), Counter()
    for i, r in enumerate(rows):
        tot[r["event"]] += 1
        if not keep[i]:
            dropped[r["event"]] += 1
    print(f"kept {keep.sum()}/{len(rows)} ({100*keep.mean():.1f}%) at top-{topk}")
    print("per-category drop rate (drop/total):")
    for c in sorted(cats, key=lambda c: -(dropped[c] / max(tot[c], 1))):
        if tot[c]:
            print(f"  {c:5s} {UCS_CATEGORIES[c][0]:26s} {dropped[c]:4d}/{tot[c]:<4d}"
                  f" {100*dropped[c]/tot[c]:5.1f}%  mean-rank {np.mean(rank_of[[i for i,r in enumerate(rows) if r['event']==c]]):.1f}")

    out = out_manifest or str(MANIFEST).replace(".csv", "_verified.csv")
    kept_rows = [r for i, r in enumerate(rows) if keep[i]]
    cols = list(kept_rows[0].keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(kept_rows)
    print(f"wrote {len(kept_rows)} verified rows -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="clap_general")
    ap.add_argument("--suffix", default="_ucs")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--template", default="the sound of {}")
    a = ap.parse_args()
    verify(a.encoder, a.suffix, a.topk, a.template)
