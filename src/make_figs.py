"""Regenerate the teaser and fidelity figures cleanly (reads results/*.json; illustrative panels
use a fixed RNG seed). Splits the old combined dissociation/fidelity figure follow-up is done in
LaTeX; this script owns fig_teaser.png and fig_fidelity.png.

    python -m src.make_figs
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from config import RESULTS

RNG = np.random.default_rng(7)
GRAY, BLUE, RED = "#7f7f7f", "#1f77b4", "#d62728"


def fig_fidelity():
    d = json.load(open(RESULTS / "spectrum_scores.json"))
    fid, froz, inst = d["fidelity"], d["frozen"], d["instance"]
    # audio-init points that actually rendered a faithful twin (mid/high fidelity)
    faithful = sorted(["n12", "n09", "n06"], key=lambda c: fid[c])
    fx = [fid[c] for c in faithful]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    # faithful audio-init: lines
    ax.plot(fx, [froz[c][0] for c in faithful], "o-", color=GRAY, lw=2, ms=7, label="frozen CLAP")
    ax.plot(fx, [inst[c][0] for c in faithful], "s-", color=RED, lw=2, ms=7, label="instance head")
    ax.set_xlim(0.5, 0.78)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("twin fidelity  =  CLAP cosine(synthetic twin, its real source)")
    ax.set_ylabel("instance R@1  (retrieve the exact real twin)")
    ax.set_title("The instance head tracks twin fidelity and stays above frozen")
    ax.grid(alpha=.3)
    ax.legend(loc="upper left", framealpha=.95)
    plt.tight_layout()
    plt.savefig(RESULTS / "fig_fidelity.png", dpi=150)
    plt.close()
    print("wrote fig_fidelity.png")


def _wave(ax, color, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 900)
    env = np.exp(-3 * t) * (0.4 + 0.6 * rng.random())
    y = env * np.sin(2 * np.pi * (18 + 40 * t) * t) * (0.6 + 0.4 * rng.standard_normal(t.size))
    ax.plot(t, y, color=color, lw=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(-1.1, 1.1); ax.axis("off")


def _scatter(ax, split, title):
    # 3 event clusters; each has real (blue) + synth (red). split=True -> domains separate.
    ax.set_title(title, fontsize=9)
    centers = [(-1.4, 1.1), (1.3, 0.9), (0.0, -1.2)]
    for cx, cy in centers:
        for dom, col in [(0, BLUE), (1, RED)]:
            off = (0.9 if split else 0.0) * (1 if dom else -1)
            x = cx + off + 0.35 * RNG.standard_normal(26)
            y = cy + 0.35 * RNG.standard_normal(26)
            ax.scatter(x, y, s=9, color=col, alpha=.8, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-3, 3); ax.set_ylim(-2.6, 2.4)


def _bars(ax):
    # real numbers from tab:diss (CLAP, 5-fold leave-classes-out, unseen categories);
    # both columns are full-gallery (instance R@1 and full-gallery category mAP)
    groups = ["find exact twin\n(instance R@1)", "same category\n(category mAP)"]
    frozen = [0.611, 0.343]
    clazz = [0.269, 0.169]
    inst = [0.800, 0.262]
    x = np.arange(2); w = 0.26
    ax.bar(x - w, frozen, w, color=GRAY, label="frozen")
    ax.bar(x,     clazz,  w, color=BLUE, label="class-sup.")
    ax.bar(x + w, inst,   w, color=RED,  label="instance")
    for xi, f in zip(x, frozen):
        ax.plot([xi - 1.6 * w, xi + 1.6 * w], [f, f], ls=":", color="0.4", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("score", fontsize=9)
    ax.set_title("on categories unseen in training", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=.95)
    ax.tick_params(labelsize=8)


def fig_teaser():
    fig = plt.figure(figsize=(13, 3.25))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.25, 1.5, 0.05, 1.35], wspace=0.28)
    # --- concept column ---
    gsl = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.55)
    a1 = fig.add_subplot(gsl[0]); _wave(a1, BLUE, 1); a1.set_title("real recording", color=BLUE, fontsize=10)
    a2 = fig.add_subplot(gsl[1]); _wave(a2, RED, 2); a2.set_title("synthetic twin", color=RED, fontsize=10)
    fig.text(0.135, 0.5, "audio-conditioned\ngeneration  $\\rightarrow$", ha="center", va="center",
             fontsize=8.5, color="0.4")
    fig.text(0.135, 0.06, "same event, different rendering  ·  DCASE-T7 (7 cls) + UCS (34 cls, 10.4k pairs)",
             ha="center", fontsize=8, color="0.45")
    # --- geometry column ---
    gsm = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.12)
    _scatter(fig.add_subplot(gsm[0]), False, "identity space\n(domains mixed)")
    _scatter(fig.add_subplot(gsm[1]), True, "rendering space\n(domains split)")
    # --- result column ---
    _bars(fig.add_subplot(gs[3]))
    fig.suptitle("Doppelganger: is a sound's identity separable from how it was rendered?",
                 fontsize=12.5, fontweight="bold", x=0.5, y=1.02)
    fig.text(0.86, -0.03, "instance correspondence transfers; category structure does not",
             ha="center", fontsize=8.5, style="italic", color="0.3")
    plt.savefig(RESULTS / "fig_teaser.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote fig_teaser.png")


if __name__ == "__main__":
    fig_fidelity()
    fig_teaser()
