"""Hand-computed checks for src/metrics.py. Run: python -m pytest tests/ -q  (or python tests/test_metrics.py)"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import metrics as M


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_three_query_example():
    # sims (3 queries x 4 gallery), rel = relevance
    sims = np.array([
        [0.90, 0.10, 0.80, 0.20],   # q0 best ranking, relevant = {0,2}
        [0.30, 0.70, 0.20, 0.60],   # q1 single relevant = {1}, ranked first
        [0.20, 0.90, 0.85, 0.10],   # q2 poor ranking, relevant = {0,3}
    ])
    rel = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 1],
    ], dtype=bool)

    out = M.evaluate_retrieval(sims, rel, ks=(1, 2, 3))

    # Hand-computed (see blueprint derivation):
    # AP:  q0=1.0, q1=1.0, q2=0.41666667  -> mAP = 0.80555556
    # RR:  q0=1.0, q1=1.0, q2=0.33333333  -> MRR = 0.77777778
    # R@1: 0.5, 1.0, 0.0 -> 0.5 ; R@2: 1.0,1.0,0.0 -> 0.66666667 ; R@3: 1.0,1.0,0.5 -> 0.83333333
    assert approx(out["mAP"], 0.80555556, 1e-6), out["mAP"]
    assert approx(out["MRR"], 0.77777778, 1e-6), out["MRR"]
    assert approx(out["R@1"], 0.5, 1e-9), out["R@1"]
    assert approx(out["R@2"], 0.66666667, 1e-6), out["R@2"]
    assert approx(out["R@3"], 0.83333333, 1e-6), out["R@3"]
    assert out["n_queries_scored"] == 3
    # first-hit ranks: q0=1, q1=1, q2=3 -> median 1
    assert approx(out["median_first_hit_rank"], 1.0, 1e-9), out["median_first_hit_rank"]


def test_ndcg_single_query():
    # q2 in isolation: ranked relevance [0,0,1,1], n_rel=2
    ranked = np.array([0, 0, 1, 1], dtype=bool)
    # nDCG@3: DCG = 1/log2(4) = 0.5 ; IDCG@3 = 1/log2(2)+1/log2(3) = 1.63092975
    assert approx(M.ndcg_at_k(ranked, 3), 0.5 / 1.63092975, 1e-6)
    # AP for this query = (1/3 + 2/4)/2 = 0.41666667
    assert approx(M.average_precision(ranked), 0.41666667, 1e-6)


def test_exclude_self():
    # gallery == query set; excluding the diagonal must change the ranking
    sims = np.array([[1.0, 0.2], [0.2, 1.0]])
    rel = np.array([[1, 1], [1, 1]], dtype=bool)  # everything relevant
    exclude = np.eye(2, dtype=bool)               # drop self-match
    out = M.evaluate_retrieval(sims, rel, ks=(1,), exclude=exclude)
    # after excluding self, each query has exactly 1 gallery item, relevant -> perfect
    assert approx(out["R@1"], 1.0, 1e-9)
    assert approx(out["MRR"], 1.0, 1e-9)


def test_relevance_builders():
    ql = np.array(["dog", "rain"])
    gl = np.array(["rain", "dog", "dog"])
    rel = M.relevance_by_label(ql, gl)
    assert rel.tolist() == [[False, True, True], [True, False, False]]

    qi = np.array([5, 9])
    gi = np.array([9, 5, -1])
    rel2 = M.relevance_by_instance(qi, gi)
    assert rel2.tolist() == [[False, True, False], [True, False, False]]


if __name__ == "__main__":
    test_three_query_example()
    test_ndcg_single_query()
    test_exclude_self()
    test_relevance_builders()
    print("all metric tests passed")
