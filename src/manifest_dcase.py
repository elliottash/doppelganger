"""Build the SoundMatch-SR manifest from the *actual* DCASE-2023 Task-7 layout.

This is the corpus-specific builder (the generic `manifest.py` assumes a flat
root/<event>/*.wav layout; the released DCASE archive is structured differently).

Layout under RAW_ROOT (the extracted Zenodo record 8091972):
  real :  DCASE_2023_Challenge_Task_7_Dataset/{dev,eval}/<event>/<stem>.wav
          + DevMeta.csv / EvalMeta.csv  (original_dataset = FSD50k|UrbanSound8k|BBC|freesound;
                                         original_file_name = the Freesound/source id)
  synth:  AudioFiles/Submissions/<track A|B>/<system_id>/<event>/<stem>.wav   (36 systems)
          DCASE_2023_Challenge_Task_7_Baseline/<event>/<stem>.wav             (challenge baseline)

Manifest schema (one row per clip). `path` is RELATIVE to config.RAW_ROOT so the same
manifest resolves locally and on a Modal volume.
    clip_id domain event morphology source system_id track orig_dataset orig_id
    dcase_subset is_cc instance_id split path

Split protocol (leakage-safe, and documented in the paper):
  * real  : DCASE's own dev/eval boundary is source-disjoint by construction ->
            eval = TEST; dev = TRAIN/VAL, split by hash(original_file_name) so multiple
            4 s crops of one Freesound recording never straddle train/val.
  * synth : hash(clip_id) -> train/val/test (70/15/15). All 36 generator systems appear in
            every split (the realistic "you know your generators" setting). The
            held-out-generator setting is a SEPARATE split (see split_by_system, for E6).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from config import (EVENT_CLASSES, MORPHOLOGY, MANIFEST, RAW_ROOT, SEED, SPLITS)

_MORPH_OF = {ev: g for g, evs in MORPHOLOGY.items() for ev in evs}

# meta CamelCase category token -> our snake_case event label
_CAT2EVENT = {
    "DogBark": "dog_bark", "Footstep": "footstep", "GunShot": "gunshot",
    "Keyboard": "keyboard", "MovingMotorVehicle": "moving_motor_vehicle",
    "Rain": "rain", "Sneeze": "sneeze_cough",
}
# original_dataset values that are CC-licensed and therefore redistributable
_CC_SOURCES = {"FSD50k", "UrbanSound8k", "freesound.org"}

_DATASET_DIR = "DCASE_2023_Challenge_Task_7_Dataset"
_BASELINE_DIR = "DCASE_2023_Challenge_Task_7_Baseline"
_SUBMISSIONS_DIR = "AudioFiles/Submissions"

COLS = ["clip_id", "domain", "event", "morphology", "source", "system_id", "track",
        "orig_dataset", "orig_id", "dcase_subset", "is_cc", "instance_id", "split", "path"]


def _bucket(key: str, n: int = 10_000) -> float:
    h = int(hashlib.sha1(f"{SEED}:{key}".encode()).hexdigest(), 16)
    return (h % n) / n


def _load_meta(meta_csv: Path) -> dict:
    """(event, stem) -> (orig_dataset, orig_id). Tolerates the stale extra rows in DevMeta."""
    out = {}
    if not meta_csv.exists():
        return out
    with open(meta_csv) as fh:
        for r in csv.DictReader(fh):
            event = _CAT2EVENT.get(r["category"])
            if event is None:
                continue
            stem = Path(r["current_file_path"]).stem
            out[(event, stem)] = (r.get("original_dataset", "unknown"),
                                  r.get("original_file_name", "unknown"))
    return out


def _real_rows(root: Path):
    rows, match, total = [], 0, 0
    for subset in ("dev", "eval"):
        meta = _load_meta(root / _DATASET_DIR / f"{subset.capitalize()}Meta.csv")
        for event in EVENT_CLASSES:
            d = root / _DATASET_DIR / subset / event
            if not d.exists():
                continue
            for f in sorted(d.glob("*.wav")):
                total += 1
                od, oid = meta.get((event, f.stem), ("unknown", "unknown"))
                match += int(oid != "unknown")
                rows.append(dict(
                    clip_id=f"real:{subset}:{event}:{f.stem}", domain="real", event=event,
                    morphology=_MORPH_OF.get(event, "other"), source=f"dcase_{subset}",
                    system_id="real", track="real", orig_dataset=od, orig_id=oid,
                    dcase_subset=subset, is_cc=int(od in _CC_SOURCES), instance_id=-1,
                    split="", path=str((root / _DATASET_DIR / subset / event / f.name)
                                       .relative_to(root))))
    print(f"  real: {total} clips, {match} matched to source meta ({100*match/max(total,1):.0f}%)")
    return rows


def _synth_rows(root: Path):
    rows = []
    sub = root / _SUBMISSIONS_DIR
    for track in ("A", "B"):
        tdir = sub / track
        if not tdir.exists():
            continue
        for sysdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            system_id = sysdir.name
            for event in EVENT_CLASSES:
                d = sysdir / event
                if not d.exists():
                    continue
                for f in sorted(d.glob("*.wav")):
                    rows.append(dict(
                        clip_id=f"synth:{system_id}:{event}:{f.stem}", domain="synth",
                        event=event, morphology=_MORPH_OF.get(event, "other"),
                        source=system_id, system_id=system_id, track=track,
                        orig_dataset="generated", orig_id="-1", dcase_subset="submission",
                        is_cc=1, instance_id=-1, split="",
                        path=str(f.relative_to(root))))
    # challenge baseline = another synthetic system
    bdir = root / _BASELINE_DIR
    for event in EVENT_CLASSES:
        d = bdir / event
        if not d.exists():
            continue
        for f in sorted(d.glob("*.wav")):
            rows.append(dict(
                clip_id=f"synth:baseline:{event}:{f.stem}", domain="synth", event=event,
                morphology=_MORPH_OF.get(event, "other"), source="baseline",
                system_id="baseline", track="baseline", orig_dataset="generated",
                orig_id="-1", dcase_subset="baseline", is_cc=1, instance_id=-1,
                split="", path=str(f.relative_to(root))))
    print(f"  synth: {len(rows)} clips across "
          f"{len({r['system_id'] for r in rows})} systems")
    return rows


def assign_splits(rows, dev_val_frac=0.15, synth_fracs=(0.70, 0.15, 0.15)):
    for r in rows:
        if r["domain"] == "real":
            if r["dcase_subset"] == "eval":
                r["split"] = "test"
            else:  # dev -> train/val, keyed on the source recording id (not the crop)
                key = r["orig_id"] if r["orig_id"] != "unknown" else r["clip_id"]
                r["split"] = "val" if _bucket(f"realdev:{key}") < dev_val_frac else "train"
        else:  # synth
            b = _bucket(f"synth:{r['clip_id']}")
            tr, va, _ = synth_fracs
            r["split"] = "train" if b < tr else ("val" if b < tr + va else "test")
    return rows


def write_manifest(rows, path: Path = MANIFEST):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def summary(rows):
    from collections import Counter
    print("\nper (domain, split):")
    c = Counter((r["domain"], r["split"]) for r in rows)
    for k in sorted(c):
        print(f"  {k}: {c[k]}")
    print("\nper (event, domain) [test split only]:")
    c = Counter((r["event"], r["domain"]) for r in rows if r["split"] == "test")
    for e in EVENT_CLASSES:
        print(f"  {e:24s} real={c[(e,'real')]:4d}  synth={c[(e,'synth')]:5d}")
    nbbc = sum(1 for r in rows if r["domain"] == "real" and not int(r["is_cc"]))
    print(f"\nnon-CC real clips (BBC/unknown, DO NOT redistribute): {nbbc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_ROOT), help="extracted-corpora root")
    a = ap.parse_args()
    root = Path(a.raw)
    print(f"building manifest from {root}")
    rows = _real_rows(root) + _synth_rows(root)
    rows = assign_splits(rows)
    write_manifest(rows)
    summary(rows)
