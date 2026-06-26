"""End-to-end smoke test with synthetic embeddings -- no audio, no model downloads.

Verifies that manifest -> embedding cache -> evaluate -> domain_probe runs, and that the
headline `domain_gap_mAP` and Proxy-A-distance both INCREASE when we inject a larger
synthetic-vs-real shift. This lets the team validate an install in seconds and guards the
evaluator against regressions.

Run: python tests/test_pipeline_synthetic.py
"""
import os, sys, csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src import evaluate, domain_probe

RNG = np.random.default_rng(0)
EVENTS = ["dog_bark", "rain", "gunshot", "piano_note"]
D = 64
PER_CELL = 80  # clips per (event, domain)


def make_dataset(encoder_name, shift, fidelity):
    """Event prototypes + noise. The synthetic domain gets (a) a uniform shift and (b) an
    event-signal scaled by `fidelity` in [0,1]: fidelity=1 -> faithful synth (small gap);
    low fidelity -> synth loses event identity -> cross-domain retrieval collapses (large gap)."""
    rows, vecs, ids = [], [], []
    protos = {e: RNG.normal(size=D) for e in EVENTS}
    shift_dir = RNG.normal(size=D); shift_dir /= np.linalg.norm(shift_dir)
    for e in EVENTS:
        for domain in ("real", "synth"):
            for j in range(PER_CELL):
                if domain == "real":
                    v = protos[e] + 0.35 * RNG.normal(size=D)
                else:
                    v = fidelity * protos[e] + 0.35 * RNG.normal(size=D) + shift * shift_dir
                v = v / (np.linalg.norm(v) + 1e-9)
                cid = f"{domain}:{e}:{j}"
                rows.append(dict(clip_id=cid, path="NA", domain=domain, event=e,
                                 morphology="x", source="synthetic_test",
                                 instance_id=-1, split="test" if j % 5 else "train"))
                vecs.append(v.astype(np.float32)); ids.append(cid)
    cols = ["clip_id", "path", "domain", "event", "morphology", "source", "instance_id", "split"]
    with open(config.MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    np.savez(config.EMB / f"{encoder_name}.npz", ids=np.array(ids), emb=np.stack(vecs))


def test_gap_responds_to_shift():
    make_dataset("smoke_small", shift=0.3, fidelity=1.0)   # faithful synth -> small gap
    small = evaluate.evaluate("smoke_small", n_boot=200)
    diag_small = domain_probe.run("smoke_small")

    make_dataset("smoke_large", shift=1.5, fidelity=0.15)   # unfaithful synth -> large gap
    large = evaluate.evaluate("smoke_large", n_boot=200)
    diag_large = domain_probe.run("smoke_large")

    # sanity: all metrics finite and in range
    for r in (small, large):
        assert 0.0 <= r["category"]["synth->real"]["mAP"] <= 1.0
        assert 0.0 <= r["control"]["real->real"]["mAP"] <= 1.0

    # a bigger planted shift -> bigger measured gap and more separable domains
    assert large["domain_gap_mAP"] > small["domain_gap_mAP"], (
        small["domain_gap_mAP"], large["domain_gap_mAP"])
    assert diag_large["proxy_a_distance"] > diag_small["proxy_a_distance"], (
        diag_small["proxy_a_distance"], diag_large["proxy_a_distance"])
    print("\nPASS: gap and PAD both increase with injected shift")
    print(f"  small: gap={small['domain_gap_mAP']:.3f}  PAD={diag_small['proxy_a_distance']:.3f}")
    print(f"  large: gap={large['domain_gap_mAP']:.3f}  PAD={diag_large['proxy_a_distance']:.3f}")


if __name__ == "__main__":
    test_gap_responds_to_shift()
