from typing import Callable, Iterable
from statistics import harmonic_mean, mean

import pandas as pd
from scipy.optimize import linear_sum_assignment

from semantic_f1_score.semantic_f1 import (
    _ensure_iter_of_sets,
    _maybe_map_nums,
    _materialize_examples,
    _looks_like_onehot,
    _onehot_to_labels,
)


# Function to compute best match for labels
def extended_hungarian_match(
    pred_set: list[str],
    true_set: list[str],
    correlation_matrix: pd.DataFrame,
    mean_type: str = "arithmetic",
) -> tuple[float, list[tuple[str, str]]]:
    if not (pred_set or true_set):
        return 1.0, []
    if not (pred_set and true_set):
        return 0.0, []

    pairs = []
    similarities = []

    # keep only the relevant part of the correlation matrix
    correlation_matrix = correlation_matrix.loc[true_set, pred_set]

    # hungarian match
    cost_matrix = 1.0 - correlation_matrix.to_numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for r, c in zip(row_ind, col_ind):
        true_label = correlation_matrix.index[r]
        pred_label = correlation_matrix.columns[c]
        sim = correlation_matrix.iat[r, c]
        # print(f"Matched {pred_label} to {true_label} with similarity {sim:.4f}")
        pairs.append((pred_label, true_label))
        similarities.append(sim)

    # match remaining items to their closest match
    matched_pred = {p for p, _ in pairs}
    matched_true = {t for _, t in pairs}
    for t in true_set:
        if t not in matched_true:
            best_pred = correlation_matrix.loc[t].idxmax()
            best_sim = correlation_matrix.loc[t, best_pred]
            pairs.append((best_pred, t))
            similarities.append(best_sim)
            # print(
            #     f"[Extension] Matched {best_pred} to {t} with similarity {best_sim:.4f}"
            # )
            matched_pred.add(best_pred)
            matched_true.add(t)
    for p in pred_set:
        if p not in matched_pred:
            best_true = correlation_matrix.loc[:, p].idxmax()
            best_sim = correlation_matrix.loc[best_true, p]
            pairs.append((p, best_true))
            similarities.append(best_sim)
            # print(
            #     f"[Extension] Matched {p} to {best_true} with similarity {best_sim:.4f}"
            # )
            matched_pred.add(p)
            matched_true.add(best_true)

    if mean_type == "arithmetic":
        mean_similarity = mean(similarities)
    else:
        mean_similarity = harmonic_mean(similarities)

    # print(f"Mean similarity: {mean_similarity:.4f}")

    return mean_similarity, pairs


def hungarian_score(
    preds: Iterable[Iterable[str | int]],
    trues: Iterable[Iterable[str | int]],
    correlation_matrix: pd.DataFrame,
    num_to_str_fn: Callable | None = None,
) -> float | dict[str, float]:
    """
    Mean pointwise Hungarian score across examples ("samples" average).

    Aligns input handling with semantic F1:
    - normalizes inputs to sets
    - optional numeric-to-string mapping applied once up-front
    - when return_components=True, also returns mean precision/recall
    """

    preds_rows = _materialize_examples(preds)
    trues_rows = _materialize_examples(trues)

    if num_to_str_fn is not None:
        preds_rows = _maybe_map_nums(preds_rows, num_to_str_fn)
        trues_rows = _maybe_map_nums(trues_rows, num_to_str_fn)

    matrix_labels = correlation_matrix.columns.tolist()
    if num_to_str_fn is None and _looks_like_onehot(preds_rows, matrix_labels):
        preds_rows = _onehot_to_labels(preds_rows, matrix_labels)
        trues_rows = _onehot_to_labels(trues_rows, matrix_labels)

    preds_sets = _ensure_iter_of_sets(preds_rows)
    trues_sets = _ensure_iter_of_sets(trues_rows)

    score = 0
    total = len(preds_sets)
    if total == 0:
        return 0.0

    for P, T in zip(preds_sets, trues_sets):
        s, _ = extended_hungarian_match(
            list(P), list(T), correlation_matrix, mean_type="arithmetic"
        )
        score += s / total

    return score
