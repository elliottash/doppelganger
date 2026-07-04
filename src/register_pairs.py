"""Merge the sharded <prefix>_*.csv into a PAIRED manifest: real anchors get their
instance_id set, and a synth-twin row is added for each, sharing that instance_id and the
anchor's split. The result is a real<->synth paired corpus for instance-level training/eval.

    python -m src.register_pairs --real manifest_ucs_verified.csv --out manifest_ucs_paired.csv

Cross-generator twin sets use the same machinery with a different shard-csv prefix and
provenance stamp (same instance_ids — derived from the anchor clip_id — so the sets align):

    python -m src.register_pairs --real manifest_ucs_verified.csv --prefix aldm \
        --source audioldm --system-id aldm --out manifest_ucs_paired_aldm.csv
    python -m src.register_pairs --real ... --prefix woosh --source woosh_dflow \
        --system-id woosh --is-cc 0 --out manifest_ucs_paired_woosh.csv   # CC-BY-NC weights
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import Counter
from pathlib import Path

from config import SYNTH, MORPHOLOGY


_MORPH = {ev: g for g, evs in MORPHOLOGY.items() for ev in evs}


def load_pairs(synth_dir: Path, prefix: str = "sao_pairs"):
    """real_clip_id -> dict(instance_id, synth_path, event, split) from all shard csvs.

    Shard csvs are <prefix>_<shard>.csv; the \\d+ requirement keeps e.g. the fidelity-sweep
    csvs (sao_pairs_n03_0.csv) out of the plain sao_pairs manifest."""
    pat = re.compile(rf"^{re.escape(prefix)}_\d+\.csv$")
    pairs = {}
    for f in sorted(glob.glob(str(synth_dir / f"{prefix}_*.csv"))):
        if not pat.match(Path(f).name):
            continue
        for r in csv.DictReader(open(f)):
            pairs[r["real_clip_id"]] = r
    return pairs


def build(real_manifest: Path, synth_dir: Path, prefix: str = "sao_pairs",
          source: str = "stable_audio_open", system_id: str = "sao", is_cc: str = "1"):
    pairs = load_pairs(synth_dir, prefix)
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
                clip_id=f"{system_id}:{p['event']}:{iid}", domain="synth", event=p["event"],
                morphology=_MORPH.get(p["event"], "other"), source=source,
                system_id=system_id, track="synth", orig_dataset="generated",
                orig_id=r["clip_id"], dcase_subset="gen", is_cc=is_cc,
                instance_id=iid, split=p["split"], path=f"synth/{p['synth_path']}")
            out.append(synth)
    return out, cols, n_paired


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="verified real manifest")
    ap.add_argument("--synth-dir", default=str(SYNTH))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="sao_pairs", help="shard-csv / twin-dir namespace")
    ap.add_argument("--source", default="stable_audio_open")
    ap.add_argument("--system-id", default="sao")
    ap.add_argument("--is-cc", default="1",
                    help="0 for research-only generators (e.g. Woosh CC-BY-NC weights)")
    a = ap.parse_args(argv)
    rows, cols, n_paired = build(Path(a.real), Path(a.synth_dir), a.prefix, a.source,
                                 a.system_id, a.is_cc)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    dom = Counter(r["domain"] for r in rows)
    print(f"wrote {len(rows)} rows ({dom['real']} real, {dom['synth']} synth, "
          f"{n_paired} pairs) -> {a.out}")


if __name__ == "__main__":
    main()
