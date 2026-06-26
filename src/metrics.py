"""
Retrieval metrics for cross-domain (synthetic <-> real) sound-effect retrieval.

The functions here are deliberately pure: they operate on a similarity matrix and a
boolean relevance matrix, with no dependency on the embedding model or the metadata
schema. `evaluate.py` is responsible for *building* `sims` and `rel` from embeddings +
manifest; this module just scores them. Keeping the scoring isolated is what lets us
unit-test it against hand-computed numbers (see tests/test_metrics.py).

Two retrieval regimes share this code:
  * Category-level: a gallery item is relevant iff it shares the query's event class.
    `rel[i]` then has many True entries (all real clips of that class).
  * Instance-level: a gallery item is relevant iff it is the paired variant of the query
    (same underlying sound). `rel[i]` then has exactly one True entry, so recall@k == hit@k
    and MRR is the headline number.
"""

from __future__ import annotations

import numpy as np


def _ranked_relevance(sims_row: np.ndarray,
                      rel_row: np.ndarray,
                      exclude_row: np.ndarray | None) -> np.ndarray:
    """Return the relevance flags of gallery items ordered by descending similarity.

    `exclude_row` (bool) marks gallery items that must not be ranked at all -- e.g. the
    query's own item when query and gallery sets overlap. Excluded items are dropped
    before ranking so they never occupy a rank slot.
    """
    if exclude_row is not None:
        keep = ~exclude_row
        sims_row = sims_row[keep]
        rel_row = rel_row[keep]
    # argsort descending; stable so ties are reproducible across runs
    order = np.argsort(-sims_row, kind="stable")
    return rel_row[order].astype(bool)


def average_precision(ranked_rel: np.ndarray) -> float:
    """Standard information-retrieval Average Precision for a single query.

    AP = (1 / R) * sum over hit positions k of precision@k, where R is the number of
    relevant items. Returns nan if there are no relevant items (caller should skip).
    """
    n_rel = int(ranked_rel.sum())
    if n_rel == 0:
        return float("nan")
    hits = np.flatnonzero(ranked_rel)            # 0-indexed positions of relevant items
    ranks = hits + 1                             # 1-indexed ranks
    precision_at_hits = (np.arange(1, n_rel + 1)) / ranks
    return float(precision_at_hits.sum() / n_rel)


def reciprocal_rank(ranked_rel: np.ndarray) -> float:
    """1 / rank of the first relevant item; 0 if none retrieved."""
    hits = np.flatnonzero(ranked_rel)
    if hits.size == 0:
        return 0.0
    return float(1.0 / (hits[0] + 1))


def recall_at_k(ranked_rel: np.ndarray, k: int) -> float:
    """Fraction of relevant items found in the top k. For single-relevant queries this
    is hit@k (0/1)."""
    n_rel = int(ranked_rel.sum())
    if n_rel == 0:
        return float("nan")
    return float(ranked_rel[:k].sum() / n_rel)


def precision_at_k(ranked_rel: np.ndarray, k: int) -> float:
    """Fraction of the top-k that are relevant. For category-level retrieval (many relevants)
    this is the interpretable companion to mAP; P@1 == 'is the nearest opposite-domain clip the
    same event?'. recall@k, by contrast, divides by the (large) number of relevants."""
    if ranked_rel.size == 0:
        return float("nan")
    kk = min(k, ranked_rel.size)
    return float(ranked_rel[:kk].sum() / kk)


def ndcg_at_k(ranked_rel: np.ndarray, k: int) -> float:
    """Binary-relevance nDCG@k."""
    n_rel = int(ranked_rel.sum())
    if n_rel == 0:
        return float("nan")
    topk = ranked_rel[:k].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, topk.size + 2))
    dcg = float((topk * discounts).sum())
    ideal = np.zeros(k, dtype=float)
    ideal[: min(n_rel, k)] = 1.0
    idcg = float((ideal * (1.0 / np.log2(np.arange(2, k + 2)))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def first_hit_rank(ranked_rel: np.ndarray) -> float:
    """Rank (1-indexed) of the first relevant item; inf if none."""
    hits = np.flatnonzero(ranked_rel)
    return float(hits[0] + 1) if hits.size else float("inf")


def evaluate_retrieval(sims: np.ndarray,
                       rel: np.ndarray,
                       ks=(1, 5, 10),
                       exclude: np.ndarray | None = None) -> dict:
    """Aggregate retrieval metrics over all queries.

    Args:
        sims:    (Nq, Ng) similarity matrix (cosine == dot product for L2-normed embeddings).
        rel:     (Nq, Ng) boolean relevance matrix.
        ks:      iterable of cutoffs for recall@k and nDCG@k.
        exclude: optional (Nq, Ng) boolean mask of items to drop from ranking.

    Returns a dict of scalar metrics plus bookkeeping (n_queries_scored, the number of
    queries that had >=1 relevant gallery item; queries with none are skipped).
    """
    sims = np.asarray(sims, dtype=np.float64)
    rel = np.asarray(rel, dtype=bool)
    assert sims.shape == rel.shape, f"shape mismatch: {sims.shape} vs {rel.shape}"
    nq = sims.shape[0]
    ks = list(ks)

    aps, rrs, fhr = [], [], []
    recalls = {k: [] for k in ks}
    precs = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}

    for i in range(nq):
        excl_row = exclude[i] if exclude is not None else None
        ranked = _ranked_relevance(sims[i], rel[i], excl_row)
        if ranked.sum() == 0:
            continue  # no relevant item available for this query -> not scorable
        aps.append(average_precision(ranked))
        rrs.append(reciprocal_rank(ranked))
        fhr.append(first_hit_rank(ranked))
        for k in ks:
            recalls[k].append(recall_at_k(ranked, k))
            precs[k].append(precision_at_k(ranked, k))
            ndcgs[k].append(ndcg_at_k(ranked, k))

    out = {
        "mAP": float(np.mean(aps)) if aps else float("nan"),
        "MRR": float(np.mean(rrs)) if rrs else float("nan"),
        "median_first_hit_rank": float(np.median(fhr)) if fhr else float("nan"),
        "n_queries_scored": len(aps),
        "n_queries_total": nq,
    }
    for k in ks:
        out[f"R@{k}"] = float(np.mean(recalls[k])) if recalls[k] else float("nan")
    for k in ks:
        out[f"P@{k}"] = float(np.mean(precs[k])) if precs[k] else float("nan")
    for k in ks:
        out[f"nDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else float("nan")
    return out


# ----------------------------------------------------------------------------------
# Helpers to build the relevance matrix from manifest columns. `evaluate.py` calls these.
# ----------------------------------------------------------------------------------
def relevance_by_label(query_labels: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    """rel[i,j] = (query_labels[i] == gallery_labels[j]). Category-level retrieval."""
    return query_labels[:, None] == gallery_labels[None, :]


def relevance_by_instance(query_instance: np.ndarray, gallery_instance: np.ndarray) -> np.ndarray:
    """rel[i,j] = (query_instance[i] == gallery_instance[j]). Instance-level (paired)
    retrieval. Instance ids must be shared across domains for the matched pair and unique
    otherwise (use -1 / a sentinel for clips with no cross-domain partner)."""
    q = query_instance[:, None]
    g = gallery_instance[None, :]
    return (q == g)
