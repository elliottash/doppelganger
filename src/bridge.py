"""Learn a lightweight projection on top of FROZEN encoder embeddings. The SAME contrastive
machinery, pointed at a different label, yields the two complementary embeddings this paper
studies:

  objective="invariant"  (the bridging method, C3)
      Pull synthetic and real variants of the SAME EVENT together while removing domain
      (production) information. Positives = same event; cross-domain positives are up-weighted
      so the head is rewarded for closing the gap, not just tightening within-domain clusters.
      Optional invariance penalties stack on top:
        dann   domain-adversarial gradient reversal (Ganin & Lempitsky 2015)
        coral  align synth/real feature covariances (Sun & Saenko 2016)
        irm    {synth,real} as environments; event predictor simultaneously optimal in both
               (Arjovsky et al. 2019) -- the formal "rendering is a nuisance, not identity".
      -> low Proxy-A-distance, event-probe high, domain-probe ~chance, cross-domain mAP up.

  objective="sensitive"  (the deliberate mirror image)
      A DOMAIN-supervised contrastive head, but the contrast is restricted to WITHIN-EVENT
      candidates: for each clip the positives are same-event/same-domain and the negatives are
      same-event/other-domain. So within every event class, synthetic and real are pushed
      apart -- the representation encodes "how it was rendered", controlled for what it is.
      -> high Proxy-A-distance, domain-probe ~100%, and a usable real-vs-synth *fidelity axis*
      (e.g. for scoring a game-SFX generator's realism). This is the construct the
      deepfake-detection papers exploit; we build it on purpose as the contrast to "invariant".

Train on the TRAIN split, re-embed everything, save data/embeddings/<encoder>_<tag>.npz so
evaluate.py / domain_probe.py re-run unchanged.

Usage:
    python -m src.bridge --encoder clap_general --objective invariant --supcon 1 --dann .3 --irm .1
    python -m src.bridge --encoder clap_general --objective sensitive --supcon 1
"""
from __future__ import annotations

import argparse
import csv
import numpy as np

from config import MANIFEST as _MANIFEST, EMB as _EMB


def _load(encoder_name):
    import torch
    rows = list(csv.DictReader(open(_MANIFEST)))
    data = np.load(_EMB / f"{encoder_name}.npz", allow_pickle=True)
    id2vec = {cid: v for cid, v in zip(data["ids"], data["emb"])}
    rows = [r for r in rows if r["clip_id"] in id2vec]
    emb = np.stack([id2vec[r["clip_id"]] for r in rows]).astype(np.float32)
    return rows, emb, torch.tensor(emb)


def _build():
    import torch
    import torch.nn as nn

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lamb):
            ctx.lamb = lamb
            return x.view_as(x)
        @staticmethod
        def backward(ctx, g):
            return -ctx.lamb * g, None

    def grad_reverse(x, lamb=1.0):
        return GradReverse.apply(x, lamb)

    class Head(nn.Module):
        def __init__(self, d_in, d_hidden=512, d_proj=256, n_events=0):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(),
                nn.Linear(d_hidden, d_proj),
            )
            self.event_head = nn.Linear(d_proj, n_events)       # for IRM / aux CE
            self.domain_head = nn.Sequential(nn.Linear(d_proj, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, x):
            z = self.backbone(x)
            z = nn.functional.normalize(z, dim=-1)              # embeddings live on the sphere
            return z

    def masked_supcon(z, pos_mask, cand_mask, temp=0.1, weight=None):
        """Supervised-contrastive loss with explicit positive and candidate (denominator) masks.

        pos_mask[i,j]  = j is a positive for anchor i   (self already excluded)
        cand_mask[i,j] = j may appear in i's softmax denominator (positives must be a subset)
        weight         = optional (n,n) multiplier on positive terms (cross-domain up-weighting)
        """
        sim = z @ z.t() / temp
        sim = sim.masked_fill(~cand_mask, -1e9)                 # contrast only against candidates
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        w = pos_mask.float()
        if weight is not None:
            w = w * weight
        denom = w.sum(1).clamp_min(1e-9)
        loss = -(w * logp).sum(1) / denom
        valid = pos_mask.any(1)
        if valid.sum() == 0:
            return z.new_zeros(())
        return loss[valid].mean()

    def coral_loss(z, domains):
        zr = z[domains == 0]; zs = z[domains == 1]
        if len(zr) < 2 or len(zs) < 2:
            return z.new_zeros(())
        def cov(a):
            a = a - a.mean(0, keepdim=True)
            return (a.t() @ a) / (a.size(0) - 1)
        d = z.size(1)
        return ((cov(zr) - cov(zs)) ** 2).sum() / (4 * d * d)

    def irm_penalty(logits, y, env_mask):
        """Per-environment gradient penalty w.r.t. a dummy scale (Arjovsky et al.)."""
        scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
        ce = nn.functional.cross_entropy
        pen = logits.new_zeros(())
        for e in env_mask.unique():
            m = env_mask == e
            if m.sum() == 0:
                continue
            loss_e = ce(logits[m] * scale, y[m])
            g = torch.autograd.grad(loss_e, [scale], create_graph=True)[0]
            pen = pen + (g ** 2)
        return pen

    return dict(Head=Head, grad_reverse=grad_reverse, masked_supcon=masked_supcon,
                coral_loss=coral_loss, irm_penalty=irm_penalty)


def _masks(yb, db, objective, parts, iidb=None, cross_domain_weight=2.0):
    """Build (pos_mask, cand_mask, weight). yb=event ids, db=domain, iidb=instance ids.

    invariant : positives = same EVENT (class clusters). Doesn't transfer to unseen classes.
    instance  : positives = same INSTANCE (a real clip and its generated twin). Learns the
                domain-removal MAPPING itself, not class identity -> the objective that can
                generalize to unseen categories.
    instance_event : instance pairs + same-event positives (instance up-weighted) -> keeps
                category structure while still learning the cross-domain mapping.
    sensitive : positives = same (event, domain), contrasted within event -> domain axis.
    """
    import torch
    n = yb.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=yb.device)
    not_self = ~eye
    same_event = (yb.view(-1, 1) == yb.view(1, -1))
    same_dom = (db.view(-1, 1) == db.view(1, -1))
    if objective == "invariant":
        pos = same_event & not_self
        cand = not_self
        w = torch.ones((n, n), device=yb.device)
        w[pos & ~same_dom] = cross_domain_weight
        return pos, cand, w
    if objective in ("instance", "instance_event"):
        same_iid = (iidb.view(-1, 1) == iidb.view(1, -1)) & (iidb.view(-1, 1) >= 0)
        pos = same_iid & not_self                    # a clip and its cross-domain twin
        if objective == "instance_event":
            pos = pos | (same_event & not_self)
        cand = not_self
        w = torch.ones((n, n), device=yb.device)
        w[same_iid & not_self] = cross_domain_weight  # the exact twin counts most
        return pos, cand, w
    if objective == "sensitive":
        pos = same_event & same_dom & not_self
        cand = same_event & not_self
        return pos, cand, None
    raise ValueError(objective)


def train(encoder_name, w_supcon=1.0, w_dann=0.0, w_coral=0.0, w_irm=0.0,
          objective="invariant", epochs=50, batch=1024, lr=1e-3, tag="",
          holdout_track="", holdout_events="", device=None):
    """holdout_track: if set (e.g. 'A'), synthetic clips from that track are EXCLUDED from
        training -> eval on that track measures generalization to unseen generators (E6).
    holdout_events: comma-separated event classes EXCLUDED from training (the head never sees
        them) -> per-event eval on a held-out class measures generalization to UNSEEN events
        (the leave-class-out test for the 7-class closed-world concern)."""
    import torch
    import torch.nn as nn

    rows, emb_np, emb = _load(encoder_name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    split = np.array([r["split"] for r in rows])
    track = np.array([r.get("track", "") for r in rows])
    event_arr = np.array([r["event"] for r in rows])
    events = sorted({r["event"] for r in rows})
    ev2i = {e: i for i, e in enumerate(events)}
    y = torch.tensor([ev2i[r["event"]] for r in rows])
    dom = torch.tensor([0 if r["domain"] == "real" else 1 for r in rows])
    iid = torch.tensor([int(r.get("instance_id", -1)) for r in rows])

    tr = np.flatnonzero(split == "train")
    if holdout_track:
        tr = np.array([i for i in tr if track[i] != holdout_track])
        print(f"E6: holding out track {holdout_track} from training -> {len(tr)} train rows")
    if holdout_events:
        ho = set(e.strip() for e in holdout_events.split(",") if e.strip())
        tr = np.array([i for i in tr if event_arr[i] not in ho])
        print(f"leave-class-out: holding out {ho} from training -> {len(tr)} train rows")
    parts = _build()
    head = parts["Head"](emb.size(1), n_events=len(events)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    Xtr = emb[tr].to(device); ytr = y[tr].to(device); dtr = dom[tr].to(device)
    itr = iid[tr].to(device)
    n = len(tr)

    # pair-aware batching: group train indices by instance_id so a real clip and its synth
    # twin land in the SAME batch (essential for the instance objective; neutral for the rest).
    import collections
    groups = collections.defaultdict(list)
    for local_i in range(n):
        k = int(itr[local_i].item())
        groups[k if k >= 0 else (-2 - local_i)].append(local_i)
    units = [torch.tensor(u, device=device) for u in groups.values()]

    for ep in range(epochs):
        head.train()
        uorder = torch.randperm(len(units))
        flat = torch.cat([units[u] for u in uorder.tolist()])   # pairs kept adjacent
        lamb = 2.0 / (1.0 + np.exp(-10 * ep / epochs)) - 1.0   # DANN schedule
        tot = 0.0
        for i in range(0, flat.numel(), batch):
            b = flat[i:i + batch]
            if b.numel() < 8:
                continue
            xb, yb, db, ib = Xtr[b], ytr[b], dtr[b], itr[b]
            z = head(xb)
            loss = z.new_zeros(())
            if objective == "classifier":
                # compute-matched supervised baseline: cross-entropy on category labels, no
                # contrastive term; the retrieval embedding is the backbone output z.
                loss = loss + nn.functional.cross_entropy(head.event_head(z), yb)
            elif w_supcon:
                pos, cand, wmat = _masks(yb, db, objective, parts, iidb=ib)
                loss = loss + w_supcon * parts["masked_supcon"](z, pos, cand, weight=wmat)
            if objective != "sensitive":   # domain-invariance penalties (not for the sensitive head)
                if w_coral:
                    loss = loss + w_coral * parts["coral_loss"](z, db)
                if w_dann:
                    d_logit = head.domain_head(parts["grad_reverse"](z, lamb)).squeeze(-1)
                    loss = loss + w_dann * nn.functional.binary_cross_entropy_with_logits(
                        d_logit, db.float())
                if w_irm:
                    ev_logits = head.event_head(z)
                    loss = loss + nn.functional.cross_entropy(ev_logits, yb)        # ERM term
                    loss = loss + w_irm * parts["irm_penalty"](ev_logits, yb, db)   # invariance
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1}/{epochs}  loss={tot/n:.4f}  lambda={lamb:.3f}")

    head.eval()
    with torch.inference_mode():
        bridged = []
        for i in range(0, emb.size(0), 2048):
            bridged.append(head(emb[i:i+2048].to(device)).cpu().numpy())
    bridged = np.concatenate(bridged, 0).astype(np.float32)
    ids = np.array([r["clip_id"] for r in rows])
    tag = tag or objective
    out = _EMB / f"{encoder_name}_{tag}.npz"
    np.savez(out, ids=ids, emb=bridged)
    # ALSO save the reusable head (weights + arch) so the adjusted embedding can be applied to
    # NEW audio (e.g. game SFX): load CLAP -> embed -> head(emb). See src/apply_head.py.
    head_path = _EMB / f"{encoder_name}_{tag}.head.pt"
    torch.save({"state_dict": head.state_dict(), "d_in": int(emb.size(1)),
                "n_events": len(events), "objective": objective, "tag": tag,
                "events": events}, head_path)
    print(f"saved {objective} embeddings -> {out}")
    print(f"saved reusable head -> {head_path}")
    print(f"Now run: python -m src.evaluate --encoder {encoder_name} --emb {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--objective",
                    choices=["invariant", "sensitive", "instance", "instance_event", "classifier"],
                    default="invariant")
    ap.add_argument("--supcon", type=float, default=1.0)
    ap.add_argument("--dann", type=float, default=0.0)
    ap.add_argument("--coral", type=float, default=0.0)
    ap.add_argument("--irm", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--tag", default="")
    ap.add_argument("--holdout-track", default="")
    ap.add_argument("--holdout-events", default="")
    a = ap.parse_args()
    train(a.encoder, a.supcon, a.dann, a.coral, a.irm,
          objective=a.objective, epochs=a.epochs, tag=a.tag, holdout_track=a.holdout_track,
          holdout_events=a.holdout_events)
