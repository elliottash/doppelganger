"""Generate ElevenLabs sound-effect twins for a balanced subset of UCS anchors — a 2nd,
text-only generator for the cross-generator robustness check. ElevenLabs has no audio-init, so we
caption each anchor from its OWN Freesound metadata (title + tags, from FSD50K clips_info) -> the
twin is instance-mediated (grounded in the real clip's description). Runs LOCALLY (API only).

    ELEVENLABS_API_KEY in .env. Output: <SYNTH>/el_pairs/<event>/<iid>.mp3 + el_pairs.csv

    python -m src.gen_elevenlabs --per-cat 20 --fsd-meta /path/FSD50K.metadata
"""
from __future__ import annotations
import argparse, csv, json, os, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

from config import SYNTH, MORPHOLOGY
from src.gen_pairs import select_anchors, instance_id_for
from src.taxonomy_ucs import UCS_CATEGORIES

_MORPH = {ev: g for g, evs in MORPHOLOGY.items() for ev in evs}
_API = "https://api.elevenlabs.io/v1/sound-generation"


def _key():
    if os.environ.get("ELEVENLABS_API_KEY"):
        return os.environ["ELEVENLABS_API_KEY"]
    for env in (Path.cwd() / ".env",):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ELEVENLABS_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("ELEVENLABS_API_KEY not found")


def load_clip_info(fsd_meta: Path) -> dict:
    info = {}
    for f in ("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"):
        p = fsd_meta / f
        if p.exists():
            info.update(json.load(open(p)))
    return info


def caption_for(anchor, info, catname):
    rec = info.get(str(anchor["orig_id"]), {})
    title = re.sub(r"\.(wav|aiff?|flac|mp3)$", "", rec.get("title", ""), flags=re.I)
    title = re.sub(r"[_\-]+", " ", title).strip()
    tags = ", ".join(rec.get("tags", [])[:6])
    cap = ", ".join(x for x in (title, tags) if x) or catname
    return f"{cap}. {catname}, isolated sound effect"[:300]


def generate_one(text, key, seconds=4.0, infl=0.5, retries=4):
    headers = {"xi-api-key": key, "Content-Type": "application/json"}
    body = {"text": text, "duration_seconds": seconds, "prompt_influence": infl}
    for attempt in range(retries):
        r = requests.post(_API, headers=headers, json=body, timeout=90)
        if r.status_code == 200:
            return r.content
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"EL {r.status_code}: {r.text[:200]}")
    raise RuntimeError("EL retries exhausted")


def run(per_cat, fsd_meta: Path, seconds=4.0, workers=4):
    key = _key()
    info = load_clip_info(fsd_meta)
    anchors = select_anchors(None, None, False, balanced_per_cat=per_cat)  # reads config.MANIFEST
    out_dir = SYNTH / "el_pairs"
    rows = []

    def task(a):
        iid = instance_id_for(a["clip_id"])
        catname = UCS_CATEGORIES.get(a["event"], (a["event"],))[0]
        cap = caption_for(a, info, catname)
        try:
            audio = generate_one(cap, key, seconds=seconds)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {a['clip_id']}: {e}"); return None
        rel = Path("el_pairs") / a["event"] / f"{iid}.mp3"
        dst = SYNTH / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(audio)
        return [str(rel), a["clip_id"], iid, a["event"], a["split"], cap]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(task, anchors)):
            if res:
                rows.append(res)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(anchors)}")
    out_csv = SYNTH / "el_pairs.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["synth_path", "real_clip_id", "instance_id", "event", "split", "caption"])
        w.writerows(rows)
    print(f"generated {len(rows)} ElevenLabs twins -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=20)
    ap.add_argument("--fsd-meta", required=True)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.per_cat, Path(a.fsd_meta), a.seconds, a.workers)
