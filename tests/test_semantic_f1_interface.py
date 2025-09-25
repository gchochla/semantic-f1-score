from __future__ import annotations

import pandas as pd
import pytest
from sklearn.preprocessing import MultiLabelBinarizer

from semantic_f1_score import semantic_f1_score
from semantic_f1_score.hungarian import hungarian_score


def identity_corr(labels: list[str]) -> pd.DataFrame:
    mat = [[1.0 if i == j else 0.0 for j in range(len(labels))] for i in range(len(labels))]
    return pd.DataFrame(mat, index=labels, columns=labels)


@pytest.mark.parametrize("average", ["samples", "micro", "macro", "weighted"])
def test_semantic_f1_accepts_strings_and_onehot_inputs(average: str):
    labels = ["joy", "anger", "surprise"]
    S = identity_corr(labels)

    trues = [["joy"], ["anger", "surprise"], ["surprise"], [], ["anger"]]
    preds = [["joy"], ["surprise"], ["surprise"], ["joy"], ["anger", "joy"]]

    mlb = MultiLabelBinarizer(classes=labels)
    y_true_onehot = mlb.fit_transform(trues).astype(int).tolist()
    y_pred_onehot = mlb.transform(preds).astype(int).tolist()

    kwargs: dict[str, object] = {"average": average}
    if average != "samples":
        kwargs["labels"] = labels

    string_score = semantic_f1_score(trues, preds, S, **kwargs)
    onehot_score = semantic_f1_score(y_true_onehot, y_pred_onehot, S, **kwargs)

    assert pytest.approx(onehot_score, rel=0, abs=1e-12) == string_score


def test_hungarian_score_accepts_strings_and_onehot_inputs():
    labels = ["joy", "anger", "surprise"]
    S = identity_corr(labels)

    trues = [["joy"], ["anger", "surprise"], ["surprise"], [], ["anger"]]
    preds = [["joy", "surprise"], ["surprise"], ["surprise"], ["joy"], ["anger", "joy"]]

    mlb = MultiLabelBinarizer(classes=labels)
    y_true_onehot = mlb.fit_transform(trues).astype(int).tolist()
    y_pred_onehot = mlb.transform(preds).astype(int).tolist()

    string_score = hungarian_score(trues, preds, S)
    onehot_score = hungarian_score(y_true_onehot, y_pred_onehot, S)

    assert pytest.approx(onehot_score, rel=0, abs=1e-12) == string_score
