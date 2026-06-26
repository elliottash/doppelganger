"""Build the benchmark manifest -- the single source of truth.

Schema (one row per clip):
    clip_id       unique string id
    path          absolute path to the audio file
    domain        'real' | 'synth'
    event         one of config.EVENT_CLASSES
    morphology    coarse group (filled from config.MORPHOLOGY)
    source        provenance: 'dcase_t7' | 'fsd50k' | 'esc50' | 'urbansound8k' |
                  'stable_audio_open' | 'audiogen' | ...
    instance_id   integer; SHARED across a matched synth<->real pair, else a unique sentinel.
                  -1 means "no cross-domain partner" (category-level only).
    split         'train' | 'val' | 'test'

This file gives you two builders:
  * build_from_dcase_t7(): wraps the DCASE-2023 Task-7 corpus (real 'dev'/'eval' + the
    released generated submissions) into category-level rows. Zero generation needed.
  * register_generated(): appends rows for clips you synthesise with generate_synthetic.py,
    optionally carrying an instance_id that ties each synthetic clip to a specific real clip
    (instance-level / paired retrieval).

Splitting is SOURCE- and INSTANCE-disjoint to prevent leakage: a Freesound uploader (or a
matched pair) never appears in more than one split. See `assign_splits`.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from config import EVENT_CLASSES, MORPHOLOGY, MANIFEST, RAW, SYNTH, SEED, SPLITS

_MORPH_OF = {ev: g for g, evs in MORPHOLOGY.items() for ev in evs}


def _hash_bucket(key: str, n: int) -> int:
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    return h % n


def _rows_for_dir(root: Path, domain: str, source: str):
    """Expect root/<event>/<file>.wav layout. Yields manifest rows (instance_id = -1)."""
    rows = []
    for event in EVENT_CLASSES:
        d = root / event
        if not d.exists():
            continue
        for f in sorted(d.glob("*.wav")):
            cid = f"{source}:{domain}:{event}:{f.stem}"
            rows.append(dict(clip_id=cid, path=str(f.resolve()), domain=domain,
                             event=event, morphology=_MORPH_OF.get(event, "other"),
                             source=source, instance_id=-1, split=""))
    return rows


def build_from_dcase_t7(dev_real_dir: Path, generated_dir: Path):
    """dev_real_dir: the DCASE-T7 real 'dev' (and/or 'eval') folders organised by class.
       generated_dir: the released generated submissions organised by class.
    Class folder names must be remapped to config names (DogBark->dog_bark etc.) beforehand;
    a remap helper is left to the team since the released layout uses CamelCase."""
    rows = _rows_for_dir(dev_real_dir, "real", "dcase_t7")
    rows += _rows_for_dir(generated_dir, "synth", "dcase_t7_generated")
    return rows


def register_generated(synth_root: Path = SYNTH, pairs_csv: Path | None = None):
    """Append rows for clips produced by generate_synthetic.py.

    If `pairs_csv` is given (columns: synth_filename,real_clip_id,instance_id), the synthetic
    clips inherit the supplied instance_id so they form instance-level pairs with real clips
    that already carry the same instance_id. Otherwise instance_id = -1 (category-level only).
    """
    pair_map = {}
    if pairs_csv and Path(pairs_csv).exists():
        with open(pairs_csv) as fh:
            for r in csv.DictReader(fh):
                pair_map[r["synth_filename"]] = int(r["instance_id"])
    rows = []
    for event in EVENT_CLASSES:
        d = synth_root / event
        if not d.exists():
            continue
        for f in sorted(d.glob("*.wav")):
            iid = pair_map.get(f.name, -1)
            cid = f"gen:synth:{event}:{f.stem}"
            rows.append(dict(clip_id=cid, path=str(f.resolve()), domain="synth",
                             event=event, morphology=_MORPH_OF.get(event, "other"),
                             source="stable_audio_open", instance_id=iid, split=""))
    return rows


def assign_splits(rows, fracs=(0.7, 0.15, 0.15)):
    """Source- and instance-disjoint split assignment.

    Grouping key:
      * paired clips (instance_id >= 0): key on the instance_id so a pair stays together;
      * unpaired clips: key on (source, basename-stem-prefix) as a proxy for the Freesound
        uploader/recording so near-duplicates do not straddle splits.
    Hash bucketing makes the assignment deterministic and reproducible.
    """
    assert abs(sum(fracs) - 1.0) < 1e-6
    edges = [fracs[0], fracs[0] + fracs[1]]  # cum thresholds in [0,1)
    for r in rows:
        if int(r["instance_id"]) >= 0:
            key = f"pair:{r['instance_id']}"
        else:
            stem_prefix = Path(r["path"]).stem.split("_")[0]
            key = f"{r['source']}:{r['event']}:{stem_prefix}"
        b = _hash_bucket(f"{SEED}:{key}", 1000) / 1000.0
        r["split"] = SPLITS[0] if b < edges[0] else (SPLITS[1] if b < edges[1] else SPLITS[2])
    return rows


def write_manifest(rows, path: Path = MANIFEST):
    cols = ["clip_id", "path", "domain", "event", "morphology", "source", "instance_id", "split"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")
    return path


if __name__ == "__main__":
    # Example end-to-end manifest build. Adjust the input dirs to your download layout.
    rows = []
    dcase_real = RAW / "dcase_t7" / "real"
    dcase_gen = RAW / "dcase_t7" / "generated"
    if dcase_real.exists():
        rows += build_from_dcase_t7(dcase_real, dcase_gen)
    rows += register_generated(SYNTH, pairs_csv=SYNTH / "pairs.csv")
    if not rows:
        print("No source audio found yet. Download DCASE-T7 / generate synth first.")
    else:
        write_manifest(assign_splits(rows))
