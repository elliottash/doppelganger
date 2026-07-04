# Doppelganger human-performance study

Measures human accuracy on the paper's synthetic-to-real twin tasks so the benchmark
can report a **human reference point** (difficulty calibration).

**Two blocks per session:**
- **Part 1 — pair 2AFC (16 trials):** one real recording and its OWN synthetic twin,
  presented as A/B in random order; "Which one is a REAL recording?"; chance 50%.
  150 unique pair trials (seeded stratified subsample of the retrieval targets, same
  mp3s — no extra audio).
- **Part 2 — retrieval (24 + 2 catch trials):** 6-way forced choice; the participant
  hears one SYNTHETIC clip (a Stable-Audio audio-init twin) and picks which of 6 REAL
  recordings it was generated from. The 5 distractors are real clips from the SAME
  UCS category (hardest realistic condition). Chance = 16.7%. 300 unique trials
  stratified over all 34 UCS categories (9 per category; 8 for the 6 smallest), TEST
  split only, CC-licensed real clips only. Catch = the "synthetic" query is the real
  recording itself; missing BOTH catches => excluded.

**Audio.** The twins were generated from a 5 s centre-crop of the real clip
(`config.CLIP_SECONDS = 5.0`) and the benchmark models embed that full 5 s window.
Humans are presented the **central 3.5 s** of that same window (centre crops nest,
so the human stimulus is a strict subset of the model's) — note this comparability
caveat when reporting: models hear 5 s, humans 3.5 s. Loudness normalised (ffmpeg
loudnorm, I=-14 LUFS), 128 kbps mono mp3.

**Assignment.** Balanced per block (least-assigned trials first); 25 participants
~ 400 2AFC judgments (~2.7/pair) and ~600 retrieval judgments (=2/trial).

**Prolific:** 25 places x GBP 1.80 for ~12 min (>= GBP 9/hr).
Cost: GBP 45.00 reward + ~33% fee (GBP 14.85) ~ **GBP 59.85** (+VAT on the fee where
applicable).

**Prolific draft (already created, UNPUBLISHED — publishing costs money and is done
manually):**
- study id: `<STUDY_ID>`
- completion code: `<COMPLETION_CODE>` (also in `study_config.json`, read by `serve.py`)
- review at: https://app.prolific.com/researcher/studies/<STUDY_ID>

## Files

| file | what |
|---|---|
| `build_trials.py` | sample trials / fetch wavs from Modal / transcode to mp3 / verify |
| `trials.jsonl` | 300 retrieval + 20 catch + 150 2afc trials (with answers) |
| `model_on_trials.py` | frozen-CLAP + instance-head (retrieval) and sensitive-axis (2afc) on the same trials |
| `model_on_trials.json` | per-trial model picks + summary |
| `serve.py` | stdlib task server (two-block assignment, audio w/ opaque alias names, responses, completion code) |
| `task.html` | participant page: Part 1 A/B UI, interstitial, Part 2 6-way UI |
| `analyze.py` | per-block human accuracy, exclusions, agreement, model comparison |
| `audio/` | 802 study mp3s (served); fetched wavs live in `~/data/doppelganger/human_study_raw/` |
| `responses/<PID>.jsonl` | one line per answer (block, choice, correct, rt_ms, plays) |
| `sessions/`, `assign_counts.json` | per-participant assignment state |

Anti-cheat note: the mp3s on disk are named `real_*`/`synth_*`, which would reveal
the 2AFC answer in the browser's network tab — `serve.py` therefore exposes 2AFC
sides and retrieval queries under opaque aliases (`pair_<tid>_a.mp3`, `q_<tid>.mp3`).

## Launch runbook

1. **Rebuild audio if needed** (idempotent; requires `modal profile current` ==
   `elliottash`):

       python3 build_trials.py sample     # only if trials.jsonl is missing
       python3 build_trials.py fetch      # pulls 802 wavs from soundmatch-sr-data
       python3 build_trials.py transcode  # -> audio/*.mp3 (central 3.5 s, -14 LUFS)
       python3 build_trials.py verify     # every trial's mp3 exists

   NOTE: re-running `sample` re-randomises trials — don't do it after data
   collection has started (trials.jsonl is the frozen design). If you change
   PRESENT_SECONDS, `rm audio/*.mp3` first (transcode skips existing files).

2. **Start the server** (port 8766 — 8765 belongs to an unrelated MIDI server):

       python3 serve.py --port 8766

3. **Expose it** with a cloudflared quick tunnel (prints an `https://....trycloudflare.com` URL):

       cloudflared tunnel --url http://localhost:8766

   Quick-tunnel URLs die with the process — keep both processes in tmux for the
   session. (For anything longer than a day, use a named tunnel.)

4. **Smoke-test through the tunnel yourself**: open
   `https://<tunnel>/?PROLIFIC_PID=smoketest_you` and do both parts with sound on.
   Delete the test artifacts afterwards:
   `rm responses/smoketest_you.jsonl sessions/smoketest_you.json`
   (or reset assignment state by deleting `assign_counts.json` and all `sessions/`
   BEFORE launch, never after.)

5. **Point the draft at the tunnel** (do NOT publish from the CLI):

       cd ../../../../tools   # multiverse/tools
       python3 -m benchmark.study.prolific_api set-url \
           --id <STUDY_ID> --url https://<tunnel>.trycloudflare.com

6. **Publish manually** in the Prolific dashboard
   (https://app.prolific.com/researcher/studies/<STUDY_ID>) — this
   spends money, which is why no script here calls publish. Monitor with:

       python3 -m benchmark.study.prolific_api submissions --id <STUDY_ID>

7. **Analyze**:

       python3 analyze.py

## Model reference numbers (same trials, computed 2026-07-01)

| model | block | accuracy | chance |
|---|---|---|---|
| frozen CLAP (cosine argmax) | retrieval 6-way | 0.920 | 0.167 |
| instance head (cosine argmax) | retrieval 6-way | 0.990 | 0.167 |
| sensitive head, real-minus-synth train-centroid axis | 2AFC | 1.000 | 0.500 |
