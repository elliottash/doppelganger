"""Merge the sharded sao_pairs_*.csv into a PAIRED manifest: real anchors get their
instance_id set, and a synth-twin row is added for each, sharing that instance_id and the
anchor's split. The result is a real<->synth paired corpus for instance-level training/eval.

    python -m src.register_pairs --real manifest_ucs_verified.csv --out manifest_ucs_paired.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

from config import SYNTH, MORPHOLOGY


_MORPH = {ev: g for g, evs in MORPHOLOGY.items() for ev in evs}


def load_pairs(synth_dir: Path):
    """real_clip_id -> dict(instance_id, synth_path, event, split) from all shard csvs."""
    pairs = {}
    for f in sorted(glob.glob(str(synth_dir / "sao_pairs_*.csv"))):
        for r in csv.DictReader(open(f)):
            pairs[r["real_clip_id"]] = r
    return pairs


def build(real_manifest: Path, synth_dir: Path):
    pairs = load_pairs(synth_dir)
    real_rows = list(csv.DictReader(open(real_manifest)))
    cols = list(real_rows[0].keys())
    out = []
    n_paired = 0
    for r in real_rows:
        p = pairs.get(r["clip_id"])
        if p:
            r["instance_id"] = p["instance_id"]; n_paired += 1
        out.append(r)
        if p:
            iid = p["instance_id"]
            synth = {k: "" for k in cols}
            synth.update(
                clip_id=f"sao:{p['event']}:{iid}", domain="synth", event=p["event"],
                morphology=_MORPH.get(p["event"], "other"), source="stable_audio_open",
                system_id="sao", track="synth", orig_dataset="generated",
                orig_id=r["clip_id"], dcase_subset="gen", is_cc="1",
                instance_id=iid, split=p["split"], path=f"synth/{p['synth_path']}")
            out.append(synth)
    return out, cols, n_paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="verified real manifest")
    ap.add_argument("--synth-dir", default=str(SYNTH))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, cols, n_paired = build(Path(a.real), Path(a.synth_dir))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    dom = Counter(r["domain"] for r in rows)
    print(f"wrote {len(rows)} rows ({dom['real']} real, {dom['synth']} synth, "
          f"{n_paired} pairs) -> {a.out}")


if __name__ == "__main__":
    main()
