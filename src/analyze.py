"""Aggregate evaluate + domain_probe across embedding variants into the paper's tables and
figures. Operates purely on cached .npz embeddings (CPU) -- no GPU, no audio.

For each variant (frozen / invariant / sensitive / ablations) it computes the retrieval report
and the domain diagnostics, then emits:
  results/leaderboard.md     Table 1 + Table 2 (frozen vs bridged, with 95% CIs)
  results/diagnostics.md     Fig 2 data (PAD, event/domain probes, silhouettes)
  results/fig_morphology.png Fig 3 (per-morphology cross-domain mAP, frozen)
  results/fig_probes.png     Fig 2 (PAD + probe bars across variants)
  results/fig_umap.png       Fig 4 (UMAP coloured by event vs domain, frozen vs the two heads)
  results/summary.json       machine-readable everything

Usage:
  python -m src.analyze --encoder clap_general \
      --variant frozen=data/embeddings/clap_general.npz \
      --variant invariant=data/embeddings/clap_general_invariant.npz \
      --variant sensitive=data/embeddings/clap_general_sensitive.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MANIFEST, RESULTS, EVENT_CLASSES, MORPHOLOGY
from src import evaluate as EV
from src import domain_probe as DP


def _reports(encoder, variants):
    out = {}
    for name, path in variants.items():
        ev = EV.evaluate(encoder, emb_path=path, n_boot=500)
        dp = DP.run(encoder, emb_path=path)
        out[name] = {"eval": ev, "diag": dp}
    return out


def leaderboard_md(reports):
    L = ["# SoundMatch-SR results\n",
         "## Table 1/2 — cross-domain retrieval (TEST split)\n",
         "| variant | real→real mAP (control) | synth→real mAP | real→synth mAP | "
         "**domain gap** | synth→real P@1 | instance MRR |",
         "|---|---|---|---|---|---|---|"]
    for name, r in reports.items():
        e = r["eval"]
        ctrl = e["control"]["real->real"]["mAP"]
        s2r = e["category"]["synth->real"]
        r2s = e["category"]["real->synth"]["mAP"]
        gap = e["domain_gap_mAP"]
        lo, hi = s2r.get("mAP_ci95", (float("nan"), float("nan")))
        inst = e.get("instance", {}).get("synth->real_paired", {}).get("MRR", float("nan"))
        L.append(f"| {name} | {ctrl:.3f} | {s2r['mAP']:.3f} [{lo:.3f},{hi:.3f}] | "
                 f"{r2s:.3f} | {gap:+.3f} | {s2r['P@1']:.3f} | {inst:.3f} |")
    return "\n".join(L) + "\n"


def diagnostics_md(reports):
    L = ["\n## Fig 2 — domain-gap diagnostics (TEST split)\n",
         "| variant | Proxy-A-dist | event-probe acc | domain-probe acc | "
         "identity−domain | silhouette(event) | silhouette(domain) |",
         "|---|---|---|---|---|---|---|"]
    for name, r in reports.items():
        d = r["diag"]
        L.append(f"| {name} | {d['proxy_a_distance']:.3f} | {d['event_probe_acc']:.3f} | "
                 f"{d['domain_probe_acc']:.3f} | {d['identity_minus_domain']:+.3f} | "
                 f"{d['silhouette_event']:.3f} | {d['silhouette_domain']:.3f} |")
    return "\n".join(L) + "\n"


def fig_morphology(reports, frozen_name):
    """Per-morphology cross-domain mAP for the frozen encoder (Fig 3)."""
    if frozen_name not in reports:
        return
    per_event = reports[frozen_name]["eval"].get("per_event_synth->real_mAP", {})
    morph_of = {ev: g for g, evs in MORPHOLOGY.items() for ev in evs}
    groups = {}
    for ev, v in per_event.items():
        groups.setdefault(morph_of.get(ev, "other"), []).append((ev, v))
    fig, ax = plt.subplots(figsize=(9, 4))
    xs, labels, colors = [], [], []
    palette = {g: c for g, c in zip(MORPHOLOGY, plt.cm.tab10.colors)}
    i = 0
    for g, items in groups.items():
        for ev, v in sorted(items, key=lambda t: -t[1]):
            xs.append(v); labels.append(ev); colors.append(palette.get(g, "gray")); i += 1
    order = np.argsort(xs)
    ax.barh([labels[k] for k in order], [xs[k] for k in order],
            color=[colors[k] for k in order])
    ax.set_xlabel("synth→real mAP (frozen)"); ax.set_title("Per-event cross-domain retrieval")
    plt.tight_layout(); plt.savefig(RESULTS / "fig_morphology.png", dpi=130); plt.close()


def fig_probes(reports):
    names = list(reports)
    pad = [reports[n]["diag"]["proxy_a_distance"] for n in names]
    dom = [reports[n]["diag"]["domain_probe_acc"] for n in names]
    ev = [reports[n]["diag"]["event_probe_acc"] for n in names]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, pad, w, label="Proxy-A-distance (↓ invariant)")
    ax.bar(x, dom, w, label="domain-probe acc")
    ax.bar(x + w, ev, w, label="event-probe acc")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.legend()
    ax.set_title("Domain separability vs event recoverability")
    plt.tight_layout(); plt.savefig(RESULTS / "fig_probes.png", dpi=130); plt.close()


def fig_umap(encoder, variants, n_per_cell=200, seed=0):
    import umap
    rows = list(csv.DictReader(open(MANIFEST)))
    rng = np.random.default_rng(seed)
    # balanced TEST subsample across (event, domain)
    by_cell = {}
    for i, r in enumerate(rows):
        if r["split"] != "test":
            continue
        by_cell.setdefault((r["event"], r["domain"]), []).append(i)
    pick = []
    for cell, idxs in by_cell.items():
        pick += list(rng.choice(idxs, size=min(n_per_cell, len(idxs)), replace=False))
    pick = np.array(sorted(pick))
    ev = np.array([rows[i]["event"] for i in pick])
    dom = np.array([rows[i]["domain"] for i in pick])
    id_of = {r["clip_id"]: i for i, r in enumerate(rows)}
    cids = [rows[i]["clip_id"] for i in pick]

    fig, axes = plt.subplots(2, len(variants), figsize=(5 * len(variants), 9), squeeze=False)
    ev_colors = {e: c for e, c in zip(EVENT_CLASSES, list(plt.cm.tab20.colors) * 2)}
    for col, (name, path) in enumerate(variants.items()):
        data = np.load(path, allow_pickle=True)
        v = {cid: vec for cid, vec in zip(data["ids"], data["emb"])}
        X = np.stack([v[c] for c in cids]).astype(np.float32)
        Z = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                      random_state=seed).fit_transform(X)
        for e in np.unique(ev):
            m = ev == e
            axes[0][col].scatter(Z[m, 0], Z[m, 1], s=5, color=ev_colors.get(e), label=e)
        axes[0][col].set_title(f"{name}: coloured by EVENT")
        for d, c in (("real", "tab:blue"), ("synth", "tab:red")):
            m = dom == d
            axes[1][col].scatter(Z[m, 0], Z[m, 1], s=5, color=c, label=d, alpha=0.5)
        axes[1][col].set_title(f"{name}: coloured by DOMAIN")
        for ax in (axes[0][col], axes[1][col]):
            ax.set_xticks([]); ax.set_yticks([])
    axes[1][0].legend(markerscale=3, loc="best")
    plt.tight_layout(); plt.savefig(RESULTS / "fig_umap.png", dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--variant", action="append", default=[],
                    help="name=path/to.npz ; repeatable. 'frozen' should be the raw encoder npz")
    ap.add_argument("--no-umap", action="store_true")
    a = ap.parse_args()
    variants = {}
    for spec in a.variant:
        name, path = spec.split("=", 1)
        variants[name] = path
    reports = _reports(a.encoder, variants)
    md = leaderboard_md(reports) + diagnostics_md(reports)
    (RESULTS / "leaderboard.md").write_text(md)
    json.dump({k: {"eval": v["eval"], "diag": v["diag"]} for k, v in reports.items()},
              open(RESULTS / "summary.json", "w"), indent=2)
    fig_morphology(reports, "frozen" if "frozen" in reports else list(reports)[0])
    fig_probes(reports)
    if not a.no_umap:
        fig_umap(a.encoder, variants)
    print(md)
    print(f"\nwrote {RESULTS}/leaderboard.md, summary.json, fig_*.png")


if __name__ == "__main__":
    main()
