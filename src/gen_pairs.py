"""Generate instance-level synthetic twins of REAL anchors (Stable Audio Open, audio-init).

Each real anchor clip gets one synth twin conditioned on the real audio, sharing an
`instance_id`. The twin INHERITS the anchor's split (dev->train/val, eval->test) so paired
clips never straddle the split boundary. Writes:
  <SYNTH>/sao_pairs/<event>/<instance_id>.wav
  <SYNTH>/sao_pairs.csv   (synth_path, real_clip_id, instance_id, event, split, seed, noise)

Sharded for Modal fan-out: process anchors where (index % n_shards == shard). Each shard
appends its own pairs_<shard>.csv (merged later) to avoid write races on the volume.

Anchor selection is deterministic (sorted by clip_id) so shards are disjoint and reproducible.
`instance_id` = a stable positive int derived from the anchor clip_id (hash), so re-runs and
the manifest wiring agree without a global counter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from config import MANIFEST, SYNTH, SEED, EVENT_CLASSES, EVENT_PHRASE, resolve_audio_path
from src.utils import load_audio


def instance_id_for(clip_id: str) -> int:
    """Stable, collision-resistant positive int for an anchor (shared by its synth twin)."""
    return int(hashlib.sha1(clip_id.encode()).hexdigest()[:12], 16)


def select_anchors(per_class: int | None, max_anchors: int | None, cc_only: bool,
                   balanced_per_cat: int | None = None, split: str = ""):
    rows = [r for r in csv.DictReader(open(MANIFEST)) if r["domain"] == "real"]
    if cc_only:
        rows = [r for r in rows if r.get("is_cc", "1") == "1"]
    if split:  # BEFORE the caps, so '--split test --max-anchors 5' smokes work
        keep = {s.strip() for s in split.split(",") if s.strip()}
        rows = [r for r in rows if r["split"] in keep]
    rows.sort(key=lambda r: r["clip_id"])
    if balanced_per_cat:  # N per ACTUAL event present (works for UCS CatIDs, any taxonomy)
        from collections import defaultdict
        seen = defaultdict(int); out = []
        for r in rows:
            if seen[r["event"]] < balanced_per_cat:
                out.append(r); seen[r["event"]] += 1
        rows = out
    elif per_class:
        out, seen = [], {e: 0 for e in EVENT_CLASSES}
        for r in rows:
            if seen.get(r["event"], per_class) < per_class:
                out.append(r); seen[r["event"]] += 1
        rows = out
    if max_anchors:
        rows = rows[:max_anchors]
    return rows


def run(shard: int, n_shards: int, per_class=None, max_anchors=None, cc_only=False,
        steps=None, cfg=None, noise=None, balanced_per_cat=None, out_tag="", mode="init",
        generator="sao", prefix="", split=""):
    """out_tag namespaces the output (sao_pairs<out_tag>/ + sao_pairs<out_tag>_<shard>.csv) so
    the fidelity sweep's noise levels don't collide. mode='text' = text-only prompt (no init
    audio): a category-level twin with NO instance signal, the spectrum's lower bound.

    generator selects the backend family (see generate_synthetic.GENERATORS): 'sao' (default,
    the original behavior) or 'aldm' (AudioLDM audio-to-audio) for the cross-generator
    transfer experiment. steps/cfg/noise default to the generator's deployed operating point
    when left None, so existing recipes stay bit-reproducible. prefix overrides the output
    namespace (default: the generator's, e.g. sao_pairs/ or aldm/). split ('test' or
    'train,val') restricts anchors to those splits (e.g. the extra SAO operating points are
    test-split-only). The PROMPT construction is deliberately identical across generators so
    the transfer comparison isn't confounded by prompt wording."""
    import soundfile as sf
    from src.generate_synthetic import GENERATORS
    from config import TARGET_SR

    g = GENERATORS[generator]
    steps = g["steps"] if steps is None else steps
    cfg = g["cfg"] if cfg is None else cfg
    noise = g["noise"] if noise is None else noise
    prefix = prefix or g["prefix"]

    anchors = select_anchors(per_class, max_anchors, cc_only, balanced_per_cat, split=split)
    mine = [a for i, a in enumerate(anchors) if i % n_shards == shard]
    print(f"shard {shard}/{n_shards}: {len(mine)} of {len(anchors)} anchors "
          f"(gen={generator} prefix={prefix} tag={out_tag} mode={mode} noise={noise} "
          f"steps={steps} cfg={cfg} split={split or 'all'})")
    if not mine:
        return

    be = g["cls"](seconds=5.0)
    out_csv = SYNTH / f"{prefix}{out_tag}_{shard}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["synth_path", "real_clip_id", "instance_id", "event", "split", "seed", "noise"])
        for k, a in enumerate(mine):
            iid = instance_id_for(a["clip_id"])
            seed = (SEED + iid) % (2**31 - 1)
            prompt = EVENT_PHRASE.get(a["event"], a["event"].replace("_", " ")) + \
                ", isolated sound effect, dry"
            try:
                real = load_audio(str(resolve_audio_path(a["path"])))  # mono @ TARGET_SR
                if mode == "text":
                    wav = be.generate(prompt, seed=seed, steps=steps, cfg_scale=cfg)
                else:
                    wav = be.generate_init(prompt, real, TARGET_SR, seed=seed, steps=steps,
                                           cfg_scale=cfg, init_noise_level=noise)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {a['clip_id']}: {e}")
                continue
            rel = Path(f"{prefix}{out_tag}") / a["event"] / f"{iid}.wav"
            dst = SYNTH / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), wav, be.sr)
            w.writerow([str(rel), a["clip_id"], iid, a["event"], a["split"], seed, noise])
            if (k + 1) % 50 == 0:
                print(f"  shard {shard}: {k+1}/{len(mine)}")
    print(f"shard {shard} done -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--max-anchors", type=int, default=None)
    ap.add_argument("--cc-only", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--generator", default="sao", choices=["sao", "aldm"])
    ap.add_argument("--prefix", default="")
    ap.add_argument("--split", default="")
    a = ap.parse_args()
    run(a.shard, a.n_shards, a.per_class, a.max_anchors, a.cc_only, a.steps, noise=a.noise,
        generator=a.generator, prefix=a.prefix, split=a.split)
