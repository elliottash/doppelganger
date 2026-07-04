"""Build the Doppelganger human-performance study trials (two blocks).

Block "retrieval" (fixed design): 6-way forced choice. Participant hears one
SYNTHETIC clip (a Stable-Audio audio-init twin) and picks which of 6 REAL recordings
it was generated from. The 5 distractors are real clips from the SAME UCS category
(chance = 16.7%). 300 unique query trials stratified over the 34 UCS categories
(9 per category, 8 for the 6 smallest), TEST split only, CC-licensed real clips only
(is_cc == "1"). Plus a pool of catch trials where the "synthetic" query is the real
recording itself.

Block "2afc": pair real-vs-synthetic discrimination. Trial = one real recording and
its OWN synthetic twin, presented as A and B in random order; "Which one is a REAL
recording?"; chance 50%. 150 trials = seeded stratified subsample of the 300
retrieval targets, reusing the exact same mp3s (no extra audio to fetch).

Steps (run in order; each is idempotent):
    python3 build_trials.py sample        # -> trials.jsonl (+ fetch_list.txt)
    python3 build_trials.py fetch         # pull needed wavs from Modal volume
    python3 build_trials.py transcode     # centre 3.5s + loudnorm + 128k mp3 -> audio/
    python3 build_trials.py verify        # every trial's mp3s exist & decode

Audio note: the synthetic twins were generated (src/gen_pairs.py) from a 5-second
CENTRE-CROP of the real clip (src/utils.load_audio, config.CLIP_SECONDS=5.0), and the
benchmark models embed that full 5 s window. Humans are presented the CENTRAL 3.5 s
(PRESENT_SECONDS) of that same window (centre crops nest, so the human window is a
strict subset of the model window) -- a small comparability caveat: models hear 5 s,
humans 3.5 s.
"""
from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = Path.home() / "data/doppelganger/manifest_ucs_paired.csv"
MODAL = str(Path.home() / ".venv-modal/bin/modal")
VOLUME = "soundmatch-sr-data"
RAW = Path.home() / "data/doppelganger/human_study_raw"  # fetched wavs (bulk data
# stays out of the Dropbox-synced tree; audio/ holds only the 57 MB of served mp3s)
AUDIO = HERE / "audio"        # transcoded mp3s served to participants
TRIALS = HERE / "trials.jsonl"

SEED = 20260701
N_TRIALS = 300
N_2AFC = 150                  # pair real-vs-synth 2AFC trials (block "2afc")
N_CATCH = 20                  # pool; each participant gets 2
N_EXTRA_DISTRACTORS = 6       # per-category real clips beyond the targets
N_CANDIDATES = 6              # forced-choice options per trial (1 target + 5 distractors)
PRESENT_SECONDS = 3.5         # centre of the 5 s model window presented to humans
CLIP_SECONDS = 5.0            # must match config.CLIP_SECONDS of the benchmark


def _san(clip_id: str) -> str:
    return clip_id.replace(":", "_").replace("/", "_")


def load_manifest():
    rows = list(csv.DictReader(open(MANIFEST)))
    real = [r for r in rows if r["domain"] == "real" and r["split"] == "test"
            and r["is_cc"] == "1"]
    synth = {(r["event"], r["instance_id"]): r for r in rows if r["domain"] == "synth"}
    return real, synth


def sample():
    rng = random.Random(SEED)
    real, synth = load_manifest()
    by_cat: dict[str, list] = {}
    for r in real:
        if (r["event"], r["instance_id"]) in synth:   # need the twin to exist
            by_cat.setdefault(r["event"], []).append(r)
    cats = sorted(by_cat)
    assert len(cats) == 34, f"expected 34 UCS categories, got {len(cats)}"

    # 300 = 28 cats x 9 + 6 cats x 8; the 6 smallest pools get 8.
    smallest = sorted(cats, key=lambda c: len(by_cat[c]))[:6]
    quota = {c: (8 if c in smallest else 9) for c in cats}
    assert sum(quota.values()) == N_TRIALS

    trials, pools = [], {}
    for cat in cats:
        pool = sorted(by_cat[cat], key=lambda r: r["clip_id"])
        rng.shuffle(pool)
        n_t = quota[cat]
        targets = pool[:n_t]
        extras = pool[n_t:n_t + N_EXTRA_DISTRACTORS]
        pools[cat] = targets + extras
        for t in targets:
            cand_pool = [r for r in pools[cat] if r["clip_id"] != t["clip_id"]]
            distractors = rng.sample(cand_pool, N_CANDIDATES - 1)
            cands = distractors + [t]
            rng.shuffle(cands)
            s = synth[(t["event"], t["instance_id"])]
            trials.append({
                "trial_id": f"t{len(trials):03d}",
                "block": "retrieval",
                "catch": False,
                "category": cat,
                "query_clip_id": s["clip_id"],
                "query_file": f"synth_{_san(s['clip_id'])}.mp3",
                "query_wav_remote": s["path"],
                "candidates": [{"clip_id": c["clip_id"],
                                "file": f"real_{_san(c['clip_id'])}.mp3",
                                "wav_remote": c["path"]} for c in cands],
                "answer": next(i for i, c in enumerate(cands)
                               if c["clip_id"] == t["clip_id"]),
            })

    # Catch trials: the "synthetic" query is the real recording itself.
    catch_cats = rng.sample(cats, N_CATCH) if N_CATCH <= len(cats) else \
        [cats[i % len(cats)] for i in range(N_CATCH)]
    for k, cat in enumerate(catch_cats):
        pool = pools[cat]
        t = rng.choice(pool)
        distractors = rng.sample([r for r in pool if r["clip_id"] != t["clip_id"]], N_CANDIDATES - 1)
        cands = distractors + [t]
        rng.shuffle(cands)
        trials.append({
            "trial_id": f"c{k:02d}",
            "block": "retrieval",
            "catch": True,
            "category": cat,
            "query_clip_id": t["clip_id"],           # the real clip IS the query
            "query_file": f"real_{_san(t['clip_id'])}.mp3",
            "query_wav_remote": t["path"],
            "candidates": [{"clip_id": c["clip_id"],
                            "file": f"real_{_san(c['clip_id'])}.mp3",
                            "wav_remote": c["path"]} for c in cands],
            "answer": next(i for i, c in enumerate(cands)
                           if c["clip_id"] == t["clip_id"]),
        })

    # Block 2: pair 2AFC (real vs its own twin). Stratified seeded subsample of the
    # retrieval targets; reuses the target real mp3 + synth query mp3 (no new audio).
    rng2 = random.Random(SEED + 1)
    scored = [t for t in trials if not t["catch"]]
    by_cat_t: dict[str, list] = {}
    for t in scored:
        by_cat_t.setdefault(t["category"], []).append(t)
    picked = []
    leftover = []
    for cat in cats:
        ts = list(by_cat_t[cat])
        rng2.shuffle(ts)
        k = len(ts) // 2
        picked += ts[:k]
        leftover += ts[k:]
    rng2.shuffle(leftover)
    picked += leftover[:N_2AFC - len(picked)]
    assert len(picked) == N_2AFC
    for k, t in enumerate(sorted(picked, key=lambda t: t["trial_id"])):
        target = t["candidates"][t["answer"]]
        pair = [{"clip_id": target["clip_id"], "file": target["file"],
                 "domain": "real"},
                {"clip_id": t["query_clip_id"], "file": t["query_file"],
                 "domain": "synth"}]
        rng2.shuffle(pair)
        trials.append({
            "trial_id": f"p{k:03d}",
            "block": "2afc",
            "catch": False,
            "category": t["category"],
            "a_file": pair[0]["file"], "b_file": pair[1]["file"],
            "real_clip_id": target["clip_id"],
            "synth_clip_id": t["query_clip_id"],
            "answer": next(i for i, c in enumerate(pair) if c["domain"] == "real"),
        })

    with open(TRIALS, "w") as fh:
        for t in trials:
            fh.write(json.dumps(t) + "\n")

    # Unique wavs to fetch (remote path -> local mp3 name stem).
    # 2afc trials reuse retrieval files, so only retrieval trials contribute.
    need = {}
    for t in trials:
        if t["block"] != "retrieval":
            continue
        need[t["query_wav_remote"]] = t["query_file"]
        for c in t["candidates"]:
            need[c["wav_remote"]] = c["file"]
    with open(HERE / "fetch_list.txt", "w") as fh:
        for remote, mp3 in sorted(need.items()):
            fh.write(f"{remote}\t{mp3}\n")
    n_ret = sum(1 for t in trials if t["block"] == "retrieval" and not t["catch"])
    n_catch = sum(1 for t in trials if t["catch"])
    n_afc = sum(1 for t in trials if t["block"] == "2afc")
    print(f"{len(trials)} trials ({n_ret} retrieval + {n_catch} catch + {n_afc} 2afc), "
          f"{len(need)} unique clips -> {TRIALS.name}, fetch_list.txt")


def _fetch_one(remote: str) -> tuple[str, bool]:
    dst = RAW / remote
    if dst.exists() and dst.stat().st_size > 1000:
        return remote, True
    dst.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        p = subprocess.run([MODAL, "volume", "get", "--force", VOLUME,
                            "/" + remote, str(dst)],
                           capture_output=True, text=True)
        if p.returncode == 0 and dst.exists() and dst.stat().st_size > 1000:
            return remote, True
    return remote, False


def fetch():
    prof = subprocess.run([MODAL, "profile", "current"], capture_output=True,
                          text=True).stdout.strip()
    if prof != "elliottash":
        sys.exit(f"modal profile is '{prof}', expected 'elliottash' -- aborting")
    remotes = [l.split("\t")[0] for l in open(HERE / "fetch_list.txt")]
    RAW.mkdir(exist_ok=True)
    ok = missing = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_fetch_one, r) for r in remotes]
        for i, f in enumerate(as_completed(futs)):
            remote, good = f.result()
            ok += good
            if not good:
                missing += 1
                print(f"  MISSING {remote}")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(remotes)}")
    print(f"fetched {ok}/{len(remotes)} ({missing} missing)")
    if missing:
        sys.exit(1)


def _transcode_one(remote: str, mp3: str) -> tuple[str, bool]:
    src, dst = RAW / remote, AUDIO / mp3
    if dst.exists() and dst.stat().st_size > 1000:
        return mp3, True
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    try:
        dur = float(p.stdout.strip())
    except ValueError:
        return mp3, False
    # centre-crop to CLIP_SECONDS (same window the twin was generated from)
    args = ["ffmpeg", "-y", "-v", "error"]
    if dur > PRESENT_SECONDS:
        args += ["-ss", f"{(dur - PRESENT_SECONDS) / 2:.3f}"]
    args += ["-i", str(src), "-t", f"{PRESENT_SECONDS}",
             "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
             "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k",
             str(dst)]
    r = subprocess.run(args, capture_output=True, text=True)
    return mp3, r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000


def transcode():
    AUDIO.mkdir(exist_ok=True)
    pairs = [l.rstrip("\n").split("\t") for l in open(HERE / "fetch_list.txt")]
    bad = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_transcode_one, r, m) for r, m in pairs]
        for i, f in enumerate(as_completed(futs)):
            mp3, good = f.result()
            if not good:
                bad += 1
                print(f"  FAILED {mp3}")
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(pairs)}")
    print(f"transcoded {len(pairs)-bad}/{len(pairs)}")
    if bad:
        sys.exit(1)


def verify():
    bad = 0
    for line in open(TRIALS):
        t = json.loads(line)
        files = [t["a_file"], t["b_file"]] if t["block"] == "2afc" else \
            [t["query_file"]] + [c["file"] for c in t["candidates"]]
        for f in files:
            p = AUDIO / f
            if not p.exists() or p.stat().st_size < 1000:
                print(f"  {t['trial_id']}: missing {f}")
                bad += 1
    print("all trial audio present" if not bad else f"{bad} missing files")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sample"
    {"sample": sample, "fetch": fetch, "transcode": transcode,
     "verify": verify}[cmd]()
