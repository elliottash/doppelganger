"""Table 2 (ablation: which objective closes the gap) and Table 4 (E6: generalization to
unseen generators). Runs on cached embeddings; writes results/ablation_e6.md + .json."""
from __future__ import annotations
import json
import numpy as np
from config import EMB, RESULTS
from src import evaluate as EV
from src import domain_probe as DP

ENC = "clap_general"


def _row(name, emb, synth_track=None, do_pad=True):
    e = EV.evaluate(ENC, emb_path=emb, n_boot=50, synth_track=synth_track)
    ctrl = e["control"]["real->real"]["mAP"]
    s2r = e["category"]["synth->real"]["mAP"]
    p1 = e["category"]["synth->real"]["P@1"]
    gap = e["domain_gap_mAP"]
    pad = dom = float("nan")
    if do_pad:
        d = DP.run(ENC, emb_path=emb)
        pad, dom = d["proxy_a_distance"], d["domain_probe_acc"]
    return dict(name=name, control=ctrl, s2r=s2r, p1=p1, gap=gap, pad=pad, dom=dom,
                n_synth=e["n_synth_test"])


def main():
    P = lambda v: str(EMB / f"{ENC}_{v}.npz")
    F = str(EMB / f"{ENC}.npz")

    # ---- Table 2: ablation ----
    abl = [
        _row("frozen", F),
        _row("supcon only", P("inv_supcon")),
        _row("supcon+CORAL", P("inv_coral")),
        _row("supcon+DANN", P("inv_dann")),
        _row("supcon+IRM", P("inv_irm")),
        _row("supcon+DANN+IRM (full)", P("invariant")),
    ]
    # ---- Table 4: E6 generalization to unseen generators (Track A held out of training) ----
    e6 = [
        _row("frozen", F, synth_track="A", do_pad=False),
        _row("invariant (trained on A+B)", P("invariant"), synth_track="A", do_pad=False),
        _row("invariant_noA (A unseen)", P("invariant_noA"), synth_track="A", do_pad=False),
    ]

    L = ["## Table 2 — ablation: closing the gap (frozen CLAP, TEST)\n",
         "| objective | control mAP | synth→real mAP | P@1 | gap | PAD | domain-probe |",
         "|---|---|---|---|---|---|---|"]
    for r in abl:
        L.append(f"| {r['name']} | {r['control']:.3f} | {r['s2r']:.3f} | {r['p1']:.3f} | "
                 f"{r['gap']:+.3f} | {r['pad']:.2f} | {r['dom']:.3f} |")
    L += ["\n## Table 4 — E6: generalization to unseen generators (synth = Track A only)\n",
          "| head | synth→real mAP (Track A) | P@1 | gap |", "|---|---|---|---|"]
    for r in e6:
        L.append(f"| {r['name']} | {r['s2r']:.3f} | {r['p1']:.3f} | {r['gap']:+.3f} |")
    md = "\n".join(L) + "\n"
    (RESULTS / "ablation_e6.md").write_text(md)
    json.dump({"ablation": abl, "e6": e6}, open(RESULTS / "ablation_e6.json", "w"), indent=2)
    print(md)


if __name__ == "__main__":
    main()
