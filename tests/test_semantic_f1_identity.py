from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize, MultiLabelBinarizer

from semantic_f1_score import pointwise_semantic_f1_score, semantic_f1_score


def identity_corr(labels: list[str]) -> pd.DataFrame:
    n = len(labels)
    mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return pd.DataFrame(mat, index=labels, columns=labels)


def positive_corr(labels: list[str], seed: int = 0) -> pd.DataFrame:
    """Construct a symmetric, non-identity correlation-like matrix with
    ones on the diagonal and off-diagonal in (0,1). Deterministic via seed.
    """
    import random

    rng = random.Random(seed)
    n = len(labels)
    mat = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                val = 1.0
            else:
                # strictly between 0 and 1 to ensure non-identity similarities
                val = rng.uniform(0.05, 0.95)
            mat[i][j] = val
            mat[j][i] = val
    return pd.DataFrame(mat, index=labels, columns=labels)


def _binarize_single_label(
    trues: list[str], preds: list[str], labels: list[str]
):
    y_true_bin = label_binarize(trues, classes=labels)
    y_pred_bin = label_binarize(preds, classes=labels)
    return y_true_bin, y_pred_bin


def test_single_label_wrappers_match_sklearn_with_identity():
    labels = ["a", "b", "c"]
    S = identity_corr(labels)

    trues = ["a", "c", "b", "c", "b"]
    preds = ["a", "b", "b", "c", "a"]

    # single-label via common function
    micro_sem = semantic_f1_score(trues, preds, S, average="micro")
    macro_sem = semantic_f1_score(trues, preds, S, average="macro")
    mean_point_sem = semantic_f1_score(trues, preds, S, average="samples")

    # sklearn references
    micro_ref = f1_score(trues, preds, average="micro", labels=labels)
    macro_ref = f1_score(trues, preds, average="macro", labels=labels)
    y_true_bin, y_pred_bin = _binarize_single_label(trues, preds, labels)
    samples_ref = f1_score(y_true_bin, y_pred_bin, average="samples")
    weighted_ref = f1_score(trues, preds, average="weighted", labels=labels)

    assert pytest.approx(micro_sem, rel=0, abs=1e-12) == micro_ref
    assert pytest.approx(macro_sem, rel=0, abs=1e-12) == macro_ref
    # mean pointwise semantic F1 reduces to per-sample F1 (== accuracy here)
    assert pytest.approx(mean_point_sem, rel=0, abs=1e-12) == samples_ref
    # also equals micro F1 in single-label identity setting
    assert pytest.approx(mean_point_sem, rel=0, abs=1e-12) == micro_ref

    # weighted average should also match sklearn in identity case
    weighted_sem = semantic_f1_score(trues, preds, S, average="weighted")
    assert pytest.approx(weighted_sem, rel=0, abs=1e-12) == weighted_ref

    # spot-check pointwise behavior (1 if match else 0)
    for p, t in zip(preds, trues):
        f1_pt, *_ = pointwise_semantic_f1_score([p], [t], S)
        expected = 1.0 if p == t else 0.0
        assert pytest.approx(f1_pt, rel=0, abs=1e-12) == expected


def test_singleton_sets_match_sklearn_with_identity():
    labels = ["a", "b", "c"]
    S = identity_corr(labels)

    trues = [["a"], ["c"], ["b"], ["c"], ["b"]]
    preds = [["a"], ["b"], ["b"], ["c"], ["a"]]

    micro_sem = semantic_f1_score(trues, preds, S, average="micro")
    macro_sem = semantic_f1_score(trues, preds, S, average="macro")
    mean_point_sem = semantic_f1_score(trues, preds, S, average="samples")

    flat_trues = [t[0] for t in trues]
    flat_preds = [p[0] for p in preds]
    micro_ref = f1_score(flat_trues, flat_preds, average="micro", labels=labels)
    macro_ref = f1_score(flat_trues, flat_preds, average="macro", labels=labels)
    y_true_bin, y_pred_bin = _binarize_single_label(
        flat_trues, flat_preds, labels
    )
    samples_ref = f1_score(y_true_bin, y_pred_bin, average="samples")

    assert pytest.approx(micro_sem, rel=0, abs=1e-12) == micro_ref
    assert pytest.approx(macro_sem, rel=0, abs=1e-12) == macro_ref
    assert pytest.approx(mean_point_sem, rel=0, abs=1e-12) == samples_ref

    # pointwise function on singletons behaves like exact-match F1
    for P, T in zip(preds, trues):
        f1_pt, *_ = pointwise_semantic_f1_score(P, T, S)
        expected = 1.0 if P == T else 0.0
        assert pytest.approx(f1_pt, rel=0, abs=1e-12) == expected


def test_multilabel_matches_sklearn_with_identity():
    labels = ["x", "y", "z"]
    S = identity_corr(labels)

    trues = [["x", "y"], [], ["z"], ["x"], ["y", "z"]]
    preds = [["x"], ["y"], ["z"], [], ["y", "z"]]

    # semantic micro/macro with identity should reduce to standard multilabel F1
    micro_sem = semantic_f1_score(trues, preds, S, average="micro")
    macro_sem = semantic_f1_score(trues, preds, S, average="macro")
    mean_point_sem = semantic_f1_score(trues, preds, S, average="samples")
    weighted_sem = semantic_f1_score(trues, preds, S, average="weighted")

    # sklearn multilabel references
    mlb = MultiLabelBinarizer(classes=labels)
    y_true_bin = mlb.fit_transform(trues)
    y_pred_bin = mlb.transform(preds)
    micro_ref = f1_score(y_true_bin, y_pred_bin, average="micro")
    macro_ref = f1_score(y_true_bin, y_pred_bin, average="macro")
    samples_ref = f1_score(y_true_bin, y_pred_bin, average="samples")
    weighted_ref = f1_score(y_true_bin, y_pred_bin, average="weighted")

    assert pytest.approx(micro_sem, rel=0, abs=1e-12) == micro_ref
    assert pytest.approx(macro_sem, rel=0, abs=1e-12) == macro_ref
    assert pytest.approx(mean_point_sem, rel=0, abs=1e-12) == samples_ref
    assert pytest.approx(weighted_sem, rel=0, abs=1e-12) == weighted_ref


def test_perfect_multilabel_scores_one_with_positive_corr():
    """For a non-identity positive correlation matrix, identical multilabel
    predictions should yield a perfect semantic score of 1 for pointwise,
    samples, micro, and macro averages.
    """
    labels = ["x", "y", "z", "w"]
    S = positive_corr(labels, seed=42)

    trues = [["x", "y"], [], ["z"], ["x", "w"], ["y", "z", "w"], ["x"]]
    preds = [list(t) for t in trues]  # exact matches

    # pointwise: each example should be perfect
    for P, T in zip(preds, trues):
        f1_pt, *_ = pointwise_semantic_f1_score(P, T, S)
        assert pytest.approx(f1_pt, rel=0, abs=1e-12) == 1.0

    # samples average: mean of perfect pointwise scores
    samples_sem = semantic_f1_score(trues, preds, S, average="samples")
    assert pytest.approx(samples_sem, rel=0, abs=1e-12) == 1.0

    # micro and macro semantic F1 should also be perfect under exact matches
    micro_sem = semantic_f1_score(trues, preds, S, average="micro")
    macro_sem = semantic_f1_score(trues, preds, S, average="macro")
    assert pytest.approx(micro_sem, rel=0, abs=1e-12) == 1.0
    assert pytest.approx(macro_sem, rel=0, abs=1e-12) == 1.0

