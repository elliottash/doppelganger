"""Ingest FSD50K into a UCS-labelled manifest (the diverse training substrate).

Reads only the small ground-truth + metadata CSV/JSON (the 24 GB of audio stays on the volume);
maps each clip's AudioSet labels -> a UCS CatID (src/taxonomy_ucs), stratified-samples per CatID,
keys splits on the Freesound uploader (leakage-safe), and emits manifest_ucs.csv in our schema.

Label quality is then VERIFIED separately by zero-shot CLAP at embed time (drop clips whose audio
disagrees with their CatID) — this script does the cheap label-driven selection.

Paths are written relative to the data root (so set SMSR_RAW=/data on Modal): "fsd50k/<...>.wav".

Usage (after fsd50k_stage; pull ground_truth+metadata locally or point --fsd-root at them):
    python -m src.ingest_fsd50k --fsd-root /path/to/fsd50k --per-cat 300
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from config import DATA, SEED
from src.taxonomy_ucs import label_to_catid, MORPHOLOGY_OF, UCS_CATEGORIES

COLS = ["clip_id", "domain", "event", "morphology", "source", "system_id", "track",
        "orig_dataset", "orig_id", "dcase_subset", "is_cc", "instance_id", "split", "path"]


def _bucket(key: str, n=1000) -> float:
    return int(hashlib.sha1(f"{SEED}:{key}".encode()).hexdigest(), 16) % n / n


def _load_uploaders(meta_dir: Path, subset: str) -> dict:
    """fname -> uploader (for leakage-safe splits). Tolerates absent metadata."""
    out = {}
    for cand in (f"{subset}_clips_info_FSD50K.json", f"{subset}.json"):
        p = meta_dir / cand
        if p.exists():
            info = json.load(open(p))
            for fname, rec in info.items():
                out[str(fname)] = (rec or {}).get("uploader", "unknown")
            break
    return out


def _rows_for_subset(gt_dir: Path, meta_dir: Path, subset: str, audio_subdir: str):
    """subset in {'dev','eval'}. dev.csv has a train/val 'split' column; eval -> test."""
    csv_path = gt_dir / f"{subset}.csv"
    if not csv_path.exists():
        return []
    uploaders = _load_uploaders(meta_dir, subset)
    rows = []
    for r in csv.DictReader(open(csv_path)):
        labels = [x for x in r.get("labels", "").split(",") if x]
        cid = label_to_catid(labels)
        if cid is None:
            continue
        fname = r["fname"]
        up = uploaders.get(fname, f"f{fname}")
        if subset == "eval":
            split = "test"
        else:
            split = r.get("split", "") or ("val" if _bucket(f"u:{up}") < 0.15 else "train")
        rows.append(dict(
            clip_id=f"fsd:{subset}:{cid}:{fname}", domain="real", event=cid,
            morphology=MORPHOLOGY_OF.get(cid, "other"), source="fsd50k", system_id="real",
            track="real", orig_dataset="fsd50k", orig_id=str(fname), dcase_subset=subset,
            is_cc=1, instance_id=-1, split=split, uploader=up,
            path=f"fsd50k/{audio_subdir}/{fname}.wav"))
    return rows


def build(fsd_root: Path, per_cat: int, min_cat: int, per_cat_test: int = 150):
    gt = fsd_root / "FSD50K.ground_truth"
    meta = fsd_root / "FSD50K.metadata"
    rows = (_rows_for_subset(gt, meta, "dev", "FSD50K.dev_audio")
            + _rows_for_subset(gt, meta, "eval", "FSD50K.eval_audio"))
    # stratified, uploader-diverse cap per CatID for BOTH train+val and test (keeps the
    # benchmark balanced across categories rather than dominated by big classes like MUSC).
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["event"]].append(r)
    kept = []
    for cid, rs in by_cat.items():
        if len(rs) < min_cat:
            continue
        div = lambda r: (_bucket(r["uploader"] + cid), r["orig_id"])  # diversity-first order
        test = sorted([r for r in rs if r["split"] == "test"], key=div)[:per_cat_test]
        trainval = sorted([r for r in rs if r["split"] != "test"], key=div)[:per_cat]
        kept += test + trainval
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsd-root", required=True)
    ap.add_argument("--per-cat", type=int, default=300, help="max train+val clips per CatID")
    ap.add_argument("--min-cat", type=int, default=40, help="drop CatIDs with fewer than this")
    ap.add_argument("--out", default=str(DATA / "manifest_ucs.csv"))
    a = ap.parse_args()
    rows = build(Path(a.fsd_root), a.per_cat, a.min_cat)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    c = Counter(r["event"] for r in rows)
    cs = Counter((r["event"], r["split"]) for r in rows)
    print(f"wrote {len(rows)} rows, {len(c)} UCS categories -> {a.out}")
    for cid in sorted(c, key=lambda k: -c[k]):
        name = UCS_CATEGORIES.get(cid, (cid,))[0]
        print(f"  {cid:5s} {name:28s} {c[cid]:5d}  "
              f"(tr {cs[(cid,'train')]} / va {cs[(cid,'val')]} / te {cs[(cid,'test')]})")


if __name__ == "__main__":
    main()
