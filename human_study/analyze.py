"""Analyze the Doppelganger human-study responses (two blocks).

Reads responses/*.jsonl, applies the exclusion rule (a participant who misses BOTH
catch trials is excluded), and reports:
  - human accuracy per block: "2afc" (real vs own twin, chance 50%) and
    "retrieval" (6-way, chance 16.7%), each with a per-participant bootstrap 95% CI
  - per-category accuracy within each block
  - per-participant table (per-block accuracy, catch performance, median RT)
  - per-trial agreement (majority vote, mean pairwise agreement) per block
  - model-on-same-trials comparison (model_on_trials.json): frozen CLAP + instance
    head on retrieval, sensitive-axis on 2afc

Comparability caveat: humans hear the CENTRAL 3.5 s of the model's 5 s window
(centre crops nest), so the human stimulus is a strict subset of what the models
embed.

Run:  python3 analyze.py [--responses responses/]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANCE = {"2afc": 0.5, "retrieval": 1 / 6}


def load(resp_dir: Path):
    per_pid = {}
    for p in sorted(resp_dir.glob("*.jsonl")):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if rows:
            # keep only the first answer per trial (guards double-submits)
            seen, uniq = set(), []
            for r in rows:
                if r["trial_id"] not in seen:
                    seen.add(r["trial_id"])
                    uniq.append(r)
            per_pid[rows[0]["pid"]] = uniq
    return per_pid


def boot_ci(included: dict, block: str, n=2000):
    """participant-level bootstrap CI over the given block's rows."""
    pids = [p for p in included if any(r["block"] == block for r in included[p])]
    if not pids:
        return float("nan"), float("nan")
    rng = random.Random(0)
    boots = []
    for _ in range(n):
        sample = [r for pid in rng.choices(pids, k=len(pids))
                  for r in included[pid] if r["block"] == block]
        boots.append(sum(r["correct"] for r in sample) / len(sample))
    boots.sort()
    return boots[int(0.025 * n)], boots[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(HERE / "responses"))
    a = ap.parse_args()
    per_pid = load(Path(a.responses))
    if not per_pid:
        print("no responses yet")
        return

    print(f"{'pid':<26} {'2afc':>6} {'retr':>6} {'catch':>5} {'medRT_s':>7}  excluded?")
    included = {}
    for pid, rows in sorted(per_pid.items()):
        catch = [r for r in rows if r["catch"]]
        catch_ok = sum(r["correct"] for r in catch)
        excl = len(catch) >= 2 and catch_ok == 0
        accs = {}
        for blk in ("2afc", "retrieval"):
            sc = [r for r in rows if r["block"] == blk and not r["catch"]]
            accs[blk] = sum(r["correct"] for r in sc) / len(sc) if sc else float("nan")
        rts = [r["rt_ms"] / 1000 for r in rows if r.get("rt_ms") and not r["catch"]]
        med = statistics.median(rts) if rts else float("nan")
        print(f"{pid:<26} {accs['2afc']:>6.3f} {accs['retrieval']:>6.3f} "
              f"{catch_ok}/{len(catch)} {med:>7.1f}  {'EXCLUDED' if excl else ''}")
        if not excl:
            included[pid] = [r for r in rows if not r["catch"]]

    if not included:
        print("\nno included participants")
        return

    for blk in ("2afc", "retrieval"):
        rows = [r for rs in included.values() for r in rs if r["block"] == blk]
        if not rows:
            continue
        n_ok = sum(r["correct"] for r in rows)
        lo, hi = boot_ci(included, blk)
        print(f"\nHUMAN {blk}: {n_ok/len(rows):.3f} ({n_ok}/{len(rows)} judgments, "
              f"{len(included)} participants)  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"chance {CHANCE[blk]:.3f}")
        per_cat = defaultdict(list)
        for r in rows:
            per_cat[r["category"]].append(r["correct"])
        print(f"per-category ({blk}):")
        for cat in sorted(per_cat, key=lambda c: -statistics.mean(per_cat[c])):
            v = per_cat[cat]
            print(f"  {cat:<5} {statistics.mean(v):.3f}  (n={len(v)})")

        by_trial = defaultdict(list)
        for r in rows:
            by_trial[r["trial_id"]].append(r)
        multi = {t: rs for t, rs in by_trial.items() if len(rs) >= 2}
        if multi:
            maj_ok, agree = 0, []
            for t, rs in multi.items():
                top, cnt = Counter(r["choice"] for r in rs).most_common(1)[0]
                maj_ok += (top == rs[0]["answer"]) and (cnt * 2 > len(rs))
                pairs = same = 0
                for i in range(len(rs)):
                    for j in range(i + 1, len(rs)):
                        pairs += 1
                        same += rs[i]["choice"] == rs[j]["choice"]
                agree.append(same / pairs)
            print(f"trials with >=2 raters: {len(multi)}/{len(by_trial)}; "
                  f"strict-majority-vote accuracy {maj_ok/len(multi):.3f}; "
                  f"mean pairwise agreement {statistics.mean(agree):.3f} "
                  f"(chance ~{CHANCE[blk]:.3f})")

    # model comparison on the SAME trials
    mp = HERE / "model_on_trials.json"
    if mp.exists():
        m = json.load(open(mp))
        print("\nMODEL on all study trials (humans heard the central 3.5 s of the "
              "model's 5 s window):")
        for name, s in m["summary"].items():
            print(f"  {name:<14} [{s['block']}] {s['accuracy']:.3f}  "
                  f"({s['correct']}/{s['n']}, chance {s['chance']})")
        answered_rows = [r for rs in included.values() for r in rs]
        answered = defaultdict(set)
        for r in answered_rows:
            answered[r["block"]].add(r["trial_id"])
        for name, s in m["summary"].items():
            tids = answered.get(s["block"], set())
            sub = [m["trials"][t][name]["correct"] for t in tids
                   if t in m["trials"] and name in m["trials"][t]]
            if sub:
                print(f"  {name:<14} [{s['block']}] {sum(sub)/len(sub):.3f} on the "
                      f"{len(sub)} human-answered trials")


if __name__ == "__main__":
    main()
