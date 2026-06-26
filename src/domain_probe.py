"""Diagnostics that quantify *how much* of an embedding is domain (synthetic vs real) vs
semantic (event). These produce the analysis figures that make the paper, independent of any
retrieval number.

1) Proxy-A-distance (PAD): train a linear classifier to predict domain from the (frozen)
   embedding. PAD = 2 * (1 - 2*err). PAD near 2 -> domains trivially separable (large gap);
   PAD near 0 -> domains indistinguishable (the goal after bridging). This is the standard
   domain-adaptation gap proxy (Ben-David et al.).

2) Event-vs-domain linear probes: accuracy of a linear probe for event and for domain, on the
   same features. A good "sound-identity" space has high event accuracy and low domain accuracy.

3) Silhouette ratio: silhouette of points labelled by event vs labelled by domain. If domain
   silhouette > event silhouette, the embedding geometry is organised by production process,
   not by what the sound is -- the core pathology this benchmark exposes.

Usage:
    python -m src.domain_probe --encoder clap_general
"""
from __future__ import annotations

import argparse
import csv
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import cross_val_score

from config import MANIFEST, EMB, RESULTS


def _load(encoder_name, emb_path=None):
    rows = list(csv.DictReader(open(MANIFEST)))
    data = np.load(emb_path or (EMB / f"{encoder_name}.npz"), allow_pickle=True)
    id2vec = {cid: v for cid, v in zip(data["ids"], data["emb"])}
    rows = [r for r in rows if r["clip_id"] in id2vec]
    emb = np.stack([id2vec[r["clip_id"]] for r in rows]).astype(np.float64)
    return rows, emb


def proxy_a_distance(X, domain_y, cv=5):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    acc = cross_val_score(clf, X, domain_y, cv=cv, scoring="accuracy").mean()
    err = 1.0 - acc
    pad = 2.0 * (1.0 - 2.0 * err)
    return float(pad), float(acc)


def linear_probe(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(X_te)))


def run(encoder_name, emb_path=None):
    rows, emb = _load(encoder_name, emb_path)
    split = np.array([r["split"] for r in rows])
    domain = np.array([r["domain"] for r in rows])
    event = np.array([r["event"] for r in rows])

    tr = split == "train"; te = split == "test"
    report = {"encoder": encoder_name}

    # 1) PAD on the test split (held-out domain separability)
    pad, dom_acc = proxy_a_distance(emb[te], (domain[te] == "synth").astype(int))
    report["proxy_a_distance"] = pad
    report["domain_cv_acc"] = dom_acc

    # 2) linear probes (train->test)
    report["event_probe_acc"] = linear_probe(emb[tr], event[tr], emb[te], event[te])
    report["domain_probe_acc"] = linear_probe(emb[tr], domain[tr], emb[te], domain[te])
    report["identity_minus_domain"] = report["event_probe_acc"] - report["domain_probe_acc"]

    # 3) silhouettes on a capped sample (silhouette is O(n^2))
    n = emb[te].shape[0]
    idx = np.random.default_rng(0).choice(n, size=min(n, 3000), replace=False)
    Xs = emb[te][idx]
    report["silhouette_event"] = float(silhouette_score(Xs, event[te][idx]))
    report["silhouette_domain"] = float(silhouette_score(Xs, domain[te][idx]))

    out = RESULTS / f"{encoder_name}_diagnostics.json"
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--emb", default=None)
    a = ap.parse_args()
    run(a.encoder, a.emb)
