"""Generate Sony Woosh (DFlow, TEXT-conditioned) twins for the UCS whoosh-morphology anchors.

Purpose: a SECOND text-only generator (besides ElevenLabs) to test whether the text-only
instance-correspondence failure (EL frozen R@1 0.11) is model-specific or a property of the
text bottleneck. Woosh has no audio conditioning, so — exactly like gen_elevenlabs — each
anchor is captioned from its OWN Freesound metadata (title + tags from FSD50K clips_info),
making the twin instance-MEDIATED (grounded in the clip's description) but not
audio-conditioned.

LICENSE: Woosh weights are CC-BY-NC (research-only). Twins are registered with is_cc=0 and
source=woosh_dflow so they can be excluded from any commercial/redistribution path.

Runs LOCALLY with the modal venv python (stdlib + modal only; the GPU work happens in the
already-deployed `diffusion-sfx-woosh` app's generate_pack function). Wavs land under
<SMSR_DATA>/synth/woosh/<event>/<iid>.wav plus a shard csv synth/woosh_0.csv (same schema as
the SAO/ALDM shard csvs, + caption), ready for `modal volume put` + src.register_pairs.

    ~/.venv-modal/bin/python -m src.gen_woosh --max-anchors 5      # smoke
    ~/.venv-modal/bin/python -m src.gen_woosh                      # all whoosh anchors
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

SEED = 1234                      # config.SEED (kept inline: this script must not need numpy)
WOOSH_APP = "diffusion-sfx-woosh"
CATNAME = {"WHSH": "whoosh / swish"}          # UCS_CATEGORIES human names for whoosh morphology


def instance_id_for(clip_id: str) -> int:     # same derivation as src.gen_pairs
    return int(hashlib.sha1(clip_id.encode()).hexdigest()[:12], 16)


def load_clip_info(fsd_meta: Path) -> dict:
    info = {}
    for f in ("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"):
        p = fsd_meta / f
        if p.exists():
            info.update(json.load(open(p)))
    return info


def caption_for(anchor: dict, info: dict, catname: str) -> str:   # mirrors gen_elevenlabs
    rec = info.get(str(anchor["orig_id"]), {})
    title = re.sub(r"\.(wav|aiff?|flac|mp3)$", "", rec.get("title", ""), flags=re.I)
    title = re.sub(r"[_\-]+", " ", title).strip()
    tags = ", ".join(rec.get("tags", [])[:6])
    cap = ", ".join(x for x in (title, tags) if x) or catname
    return f"{cap}. {catname}, isolated sound effect"[:300]


def select_whoosh_anchors(manifest: Path, max_anchors: int | None, split: str = ""):
    rows = [r for r in csv.DictReader(open(manifest))
            if r["domain"] == "real" and r["morphology"] == "whoosh"]
    if split:
        keep = {s.strip() for s in split.split(",")}
        rows = [r for r in rows if r["split"] in keep]
    rows.sort(key=lambda r: r["clip_id"])
    return rows[:max_anchors] if max_anchors else rows


def run(manifest: Path, fsd_meta: Path, out_root: Path, max_anchors=None, split="",
        chunk=96, cfg=4.5, steps=4, frames=501):
    import modal
    anchors = select_whoosh_anchors(manifest, max_anchors, split)
    info = load_clip_info(fsd_meta)
    print(f"{len(anchors)} whoosh anchors (splits: "
          f"{sorted(set(a['split'] for a in anchors))})")
    gen = modal.Function.from_name(WOOSH_APP, "generate_pack")

    reqs, meta = [], []
    for a in anchors:
        iid = instance_id_for(a["clip_id"])
        cap = caption_for(a, info, CATNAME.get(a["event"], a["event"]))
        seed = (SEED + iid) % (2 ** 31 - 1)
        reqs.append({"prompt": cap, "seed": seed, "cfg": cfg, "steps": steps,
                     "frames": frames})
        meta.append((a, iid, cap, seed))

    out_rows = []
    for s in range(0, len(reqs), chunk):
        part, pmeta = reqs[s:s + chunk], meta[s:s + chunk]
        print(f"generating {s}..{s + len(part) - 1} of {len(reqs)} ...", flush=True)
        blob = gen.remote(requests=part, model="Woosh-DFlow")
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            names = sorted(n for n in zf.namelist() if n.endswith(".wav"))
            # generate_pack names files f"{i:03d}_..."; join on the in-chunk index.
            by_idx = {int(n.split("_", 1)[0]): n for n in names}
            for j, (a, iid, cap, seed) in enumerate(pmeta):
                if j not in by_idx:
                    print(f"  MISSING output for {a['clip_id']}")
                    continue
                rel = Path("woosh") / a["event"] / f"{iid}.wav"
                dst = out_root / "synth" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(zf.read(by_idx[j]))
                out_rows.append([str(rel), a["clip_id"], iid, a["event"], a["split"],
                                 seed, 0.0, cap])

    out_csv = out_root / "synth" / "woosh_0.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["synth_path", "real_clip_id", "instance_id", "event", "split", "seed",
                    "noise", "caption"])
        w.writerows(out_rows)
    print(f"{len(out_rows)} woosh twins -> {out_root / 'synth' / 'woosh'}\ncsv -> {out_csv}")
    print("Next: push to the soundmatch volume and register:\n"
          f"  modal volume put soundmatch-sr-data {out_root}/synth/woosh /synth/woosh\n"
          f"  modal volume put soundmatch-sr-data {out_csv} /synth/woosh_0.csv\n"
          "  modal run modal_app.py::register --prefix woosh --source woosh_dflow "
          "--system-id woosh --is-cc 0 --out manifest_ucs_paired_woosh.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    d = os.environ.get("SMSR_DATA", str(Path.home() / "data" / "doppelganger"))
    ap.add_argument("--manifest", default=f"{d}/manifest_ucs_verified.csv")
    ap.add_argument("--fsd-meta", default=f"{d}/fsd50k_meta/FSD50K.metadata")
    ap.add_argument("--out-root", default=d)
    ap.add_argument("--max-anchors", type=int, default=None)
    ap.add_argument("--split", default="", help="e.g. 'test' to restrict")
    a = ap.parse_args()
    run(Path(a.manifest), Path(a.fsd_meta), Path(a.out_root), a.max_anchors, a.split)
