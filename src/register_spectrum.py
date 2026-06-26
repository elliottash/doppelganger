"""Build a manifest for the fidelity-spectrum experiment: the balanced real subset plus its
synthetic twins at every init_noise level and the text-only condition. Each synth row carries a
`cond` (n03/n06/n09/n12/text) so we can measure instance retrieval as a function of fidelity.

    python -m src.register_spectrum --real manifest_ucs_verified.csv --per-cat 40 \
        --synth-dir <synth> --out manifest_spectrum.csv
"""
from __future__ import annotations
import argparse, csv, glob
from pathlib import Path
from collections import defaultdict
from config import SYNTH, MORPHOLOGY
from src.gen_pairs import select_anchors, instance_id_for

_MORPH = {ev: g for g, evs in MORPHOLOGY.items() for ev in evs}
CONDS = {"_n03": 0.3, "_n06": 0.6, "_n09": 0.9, "_n12": 1.2, "_text": 0.0}


def build(per_cat: int, synth_dir: Path):
    anchors = select_anchors(None, None, False, balanced_per_cat=per_cat)  # reads config.MANIFEST
    cols = list(anchors[0].keys())
    rows = []
    for a in anchors:                                   # real subset rows
        a = dict(a); a["instance_id"] = instance_id_for(a["clip_id"])
        a["cond"] = "real"; rows.append(a)
    for tag, noise in CONDS.items():
        pairs = {}
        for f in sorted(glob.glob(str(synth_dir / f"sao_pairs{tag}_*.csv"))):
            for r in csv.DictReader(open(f)):
                pairs[r["real_clip_id"]] = r
        for a in anchors:
            p = pairs.get(a["clip_id"])
            if not p:
                continue
            iid = p["instance_id"]
            row = {k: "" for k in cols}
            row.update(clip_id=f"sao{tag}:{p['event']}:{iid}", domain="synth", event=p["event"],
                       morphology=_MORPH.get(p["event"], "other"), source="stable_audio_open",
                       system_id="sao", track="synth", orig_dataset="generated", orig_id=a["clip_id"],
                       dcase_subset="gen", is_cc="1", instance_id=iid, split=a["split"],
                       path=f"synth/{p['synth_path']}", cond=tag.strip("_"))
            rows.append(row)
    return rows, cols + ["cond"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=40)
    ap.add_argument("--synth-dir", default=str(SYNTH))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, cols = build(a.per_cat, Path(a.synth_dir))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    c = Counter(r["cond"] for r in rows)
    print(f"wrote {len(rows)} rows -> {a.out}")
    for k in ("real",) + tuple(t.strip("_") for t in CONDS):
        print(f"  {k:6s}: {c[k]}")


if __name__ == "__main__":
    main()
