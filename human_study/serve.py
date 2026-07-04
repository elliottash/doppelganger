"""Single-file stdlib server for the Doppelganger human study (two blocks).

Session = Part 1: 16 pair 2AFC trials (real vs its own synthetic twin, "which is
REAL?", chance 50%), then Part 2: 24 retrieval trials + 2 catch trials (6-way
"which real recording was this generated from?", chance 16.7%). Assignment is
balanced per block (least-assigned trials first). Answers + timings go to
responses/<PROLIFIC_PID>.jsonl (each row carries a "block" field); the completion
code is shown only after BOTH parts are finished.

Run:  python3 serve.py [--port 8766]
Then expose via cloudflared:  cloudflared tunnel --url http://localhost:8766

Endpoints:
  GET  /?PROLIFIC_PID=..&STUDY_ID=..&SESSION_ID=..   task page
  GET  /api/session?pid=PID     assign (or resume) this participant's trial list
  POST /api/answer              {pid, trial_id, position, choice, rt_ms, plays}
  GET  /audio/<file>.mp3        study audio (Range supported)
  GET  /health                  liveness probe
"""
from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
AUDIO = HERE / "audio"
RESPONSES = HERE / "responses"
SESSIONS = HERE / "sessions"
COUNTS = HERE / "assign_counts.json"
CONFIG = HERE / "study_config.json"

N_2AFC_PER_SESSION = 16
N_SCORED_PER_SESSION = 24
N_CATCH_PER_SESSION = 2

_lock = threading.Lock()
_pid_re = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

TRIALS = {t["trial_id"]: t for t in
          (json.loads(l) for l in open(HERE / "trials.jsonl"))}
SCORED_IDS = [tid for tid, t in TRIALS.items()
              if t["block"] == "retrieval" and not t["catch"]]
CATCH_IDS = [tid for tid, t in TRIALS.items() if t["catch"]]
AFC_IDS = [tid for tid, t in TRIALS.items() if t["block"] == "2afc"]
PAGE = (HERE / "task.html").read_bytes()
COMPLETION_CODE = json.load(open(CONFIG))["completion_code"] if CONFIG.exists() \
    else "TESTCODE"


def _load_counts():
    return json.load(open(COUNTS)) if COUNTS.exists() else {}


def _assign(pid: str) -> dict:
    """Create or resume a session. Balanced per block: least-assigned trials first.
    Order: Part 1 = 2afc trials, Part 2 = retrieval trials with catches inserted."""
    sess_path = SESSIONS / f"{pid}.json"
    with _lock:
        if sess_path.exists():
            return json.load(open(sess_path))
        counts = _load_counts()
        rng = random.Random(f"{pid}:{time.time_ns()}")

        def pick_balanced(pool, k):
            order = sorted(pool, key=lambda t: (counts.get(t, 0), rng.random()))
            chosen = order[:k]
            rng.shuffle(chosen)
            return chosen

        part1 = pick_balanced(AFC_IDS, N_2AFC_PER_SESSION)
        part2 = pick_balanced(SCORED_IDS, N_SCORED_PER_SESSION)
        catches = rng.sample(CATCH_IDS, N_CATCH_PER_SESSION)
        # insert catches away from the very start/end of part 2
        for c in catches:
            part2.insert(rng.randint(4, max(5, len(part2) - 4)), c)
        chosen = part1 + part2
        for t in chosen:
            counts[t] = counts.get(t, 0) + 1
        json.dump(counts, open(COUNTS, "w"))
        sess = {"pid": pid, "trial_ids": chosen, "created": time.time()}
        SESSIONS.mkdir(exist_ok=True)
        json.dump(sess, open(sess_path, "w"))
        return sess


def _answered(pid: str) -> set:
    p = RESPONSES / f"{pid}.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["trial_id"] for l in open(p) if l.strip()}


def _client_trials(sess: dict) -> list:
    """Trial list with NO answer / catch / domain leakage.

    2afc sides and query files are served under OPAQUE alias names (pair_<tid>_a.mp3,
    q_<tid>.mp3): the underlying mp3s are named real_*/synth_*, which would give the
    2afc answer (and catch-ness) away in the browser's network tab."""
    out = []
    for tid in sess["trial_ids"]:
        t = TRIALS[tid]
        if t["block"] == "2afc":
            out.append({"trial_id": tid, "block": "2afc",
                        "category": t["category"],
                        "a": f"pair_{tid}_a.mp3", "b": f"pair_{tid}_b.mp3"})
        else:
            out.append({"trial_id": tid, "block": "retrieval",
                        "category": t["category"], "query": f"q_{tid}.mp3",
                        "candidates": [c["file"] for c in t["candidates"]]})
    return out


_alias_re = re.compile(r"^(?:pair_([A-Za-z0-9]+)_([ab])|q_([A-Za-z0-9]+))\.mp3$")


def _resolve_alias(name: str) -> str | None:
    """Map an alias filename to the real mp3 name (or pass through real names)."""
    m = _alias_re.match(name)
    if not m:
        return name
    if m.group(3):                             # q_<tid>.mp3
        t = TRIALS.get(m.group(3))
        return t["query_file"] if t and t["block"] == "retrieval" else None
    t = TRIALS.get(m.group(1))                 # pair_<tid>_<a|b>.mp3
    if not t or t["block"] != "2afc":
        return None
    return t["a_file"] if m.group(2) == "a" else t["b_file"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/" or u.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif u.path == "/health":
            self._json({"ok": True, "trials": len(TRIALS)})
        elif u.path == "/api/session":
            q = parse_qs(u.query)
            pid = q.get("pid", [""])[0]
            if not _pid_re.match(pid):
                return self._json({"error": "bad pid"}, 400)
            sess = _assign(pid)
            done = _answered(pid)
            out = {"pid": pid, "trials": _client_trials(sess),
                   "answered": sorted(done), "total": len(sess["trial_ids"])}
            if len(done) >= len(sess["trial_ids"]):
                out["completion_code"] = COMPLETION_CODE
            self._json(out)
        elif u.path.startswith("/audio/"):
            self._serve_audio(u.path[len("/audio/"):])
        else:
            self._json({"error": "not found"}, 404)

    def _serve_audio(self, name: str):
        if "/" in name or ".." in name or not name.endswith(".mp3"):
            return self._json({"error": "bad path"}, 400)
        name = _resolve_alias(name)
        if name is None:
            return self._json({"error": "missing"}, 404)
        p = AUDIO / name
        if not p.exists():
            return self._json({"error": "missing"}, 404)
        data = p.read_bytes()
        rng = self.headers.get("Range")
        start, end = 0, len(data) - 1
        status = 200
        if rng and rng.startswith("bytes="):
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), end)
                status = 206
        chunk = data[start:end + 1]
        self.send_response(status)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Cache-Control", "public, max-age=86400")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.end_headers()
        self.wfile.write(chunk)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/answer":
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            pid = body["pid"]
            tid = body["trial_id"]
            choice = int(body["choice"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return self._json({"error": "bad body"}, 400)
        if not _pid_re.match(pid) or tid not in TRIALS:
            return self._json({"error": "bad answer"}, 400)
        t = TRIALS[tid]
        n_opts = 2 if t["block"] == "2afc" else 6
        if not 0 <= choice < n_opts:
            return self._json({"error": "bad answer"}, 400)
        sess_path = SESSIONS / f"{pid}.json"
        if not sess_path.exists():
            return self._json({"error": "no session"}, 400)
        sess = json.load(open(sess_path))
        if tid not in sess["trial_ids"]:
            return self._json({"error": "trial not in session"}, 400)
        rec = {"pid": pid, "trial_id": tid, "block": t["block"],
               "category": t["category"], "catch": t["catch"], "choice": choice,
               "answer": t["answer"], "correct": choice == t["answer"],
               "position": body.get("position"),
               "rt_ms": body.get("rt_ms"), "plays": body.get("plays"),
               "ts": time.time(),
               "study_id": body.get("study_id"), "session_id": body.get("session_id")}
        with _lock:
            RESPONSES.mkdir(exist_ok=True)
            with open(RESPONSES / f"{pid}.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        done = _answered(pid)
        finished = len(done) >= len(sess["trial_ids"])
        out = {"ok": True, "answered": len(done), "total": len(sess["trial_ids"])}
        if finished:
            out["completion_code"] = COMPLETION_CODE
        self._json(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    print(f"{len(AFC_IDS)} 2afc + {len(SCORED_IDS)} retrieval + {len(CATCH_IDS)} "
          f"catch trials; completion code {COMPLETION_CODE}")
    print(f"serving on http://0.0.0.0:{a.port}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
