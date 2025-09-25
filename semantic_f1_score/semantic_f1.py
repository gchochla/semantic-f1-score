from typing import Callable, Iterable, Any
from statistics import mean, harmonic_mean

import pandas as pd


# Function to compute best match for labels
def best_matching_pairs(
    A: list[str],
    B: list[str],
    correlation_matrix: pd.DataFrame,
    mean_type: str = "arithmetic",
) -> tuple[float, list[tuple[str, str]]]:
    if not (A or B):
        return 1.0, []
    if not (A and B):
        return 0.0, []

    pairs = []
    similarities = []

    for a in A:
        best_match = None
        best_similarity = float("-inf")
        for b in B:
            similarity = correlation_matrix.loc[b, a]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = b

        if best_match is not None:
            pairs.append((a, best_match))
            similarities.append(best_similarity)

    if mean_type == "arithmetic":
        average_similarity = mean(similarities)
    else:
        average_similarity = (
            harmonic_mean(similarities) if similarities else 0.0
        )

    return average_similarity, pairs


# Main function to compute overall similarity for preds to gt
def pointwise_semantic_f1_score(
    pred_set: Iterable[str] | set[str],
    true_set: Iterable[str] | set[str],
    correlation_matrix: pd.DataFrame,
    num_to_str_fn: Callable | None = None,
    *,
    return_components: bool = False,
) -> (
    tuple[float, list[tuple[str, str]], list[tuple[str, str]]] | dict[str, Any]
):
    """Computes the semantic F1 score between two sets of labels
    (i.e. for a specific example / point only). The semantic F1 score
    is a measure of how well the predicted labels match the ground
    truth in a multilabel setting, but instead of the hard match, it uses the
    semantic similarity between them. The semantic similarity
    is defined, e.g., by the cosine similarity between the emotions in
    Plutchik's wheel, or the correlation between moral foundations.

    The algorithm is the following:
    1. For each predicted label, find the best matching GT label
    2. Calculate the similarity in [0, 1] between the pairs
    3. Compute the average similarity (the semantic precision)
    4. And vice versa (the semantic recall)
    5. Compute the harmonic mean of the precision and recall

    Args:
        pred_set: The list of predicted str labels
        true_set: The list of ground-truth str labels
        correlation_matrix: The matrix of similarities
            between the labels
        return_components: when True, returns a dict including
            precision/recall and the pairing details.

    Returns:
        If return_components is False (default):
            (f1, forward_pairs, reverse_pairs)
        If return_components is True:
            {"f1", "precision", "recall", "forward_pairs", "reverse_pairs"}
    """

    if num_to_str_fn is not None:
        pred_set = num_to_str_fn(pred_set)
        true_set = num_to_str_fn(true_set)

    # precision: of all the retrieved items, how many were true
    # semantic precision: of all the retrieved items, how many were relevant
    semantic_precision, forward_pairs = best_matching_pairs(
        pred_set, true_set, correlation_matrix
    )

    # recall: of all the true items, how many were retrieved
    # semantic recall: of all the true items, how many were well approximated
    semantic_recall, reverse_pairs = best_matching_pairs(
        true_set, pred_set, correlation_matrix
    )

    semantic_f1 = harmonic_mean([semantic_precision, semantic_recall])

    if not return_components:
        return semantic_f1, forward_pairs, reverse_pairs

    return {
        "f1": float(semantic_f1),
        "precision": float(semantic_precision),
        "recall": float(semantic_recall),
        "forward_pairs": forward_pairs,
        "reverse_pairs": reverse_pairs,
    }


def samples_semantic_f1_score(
    preds: Iterable[Iterable[str | int]],
    trues: Iterable[Iterable[str | int]],
    correlation_matrix: pd.DataFrame,
    num_to_str_fn: Callable | None = None,
    *,
    return_components: bool = False,
) -> float | dict[str, float]:
    """
    Mean pointwise semantic F1 across examples ("samples" average).

    Aligns input handling with semantic_micro_f1_score/semantic_macro_f1_score:
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

    total_f1 = 0.0
    total_p = 0.0
    total_r = 0.0
    count = 0
    for P, T in zip(preds_sets, trues_sets):
        res = pointwise_semantic_f1_score(
            P, T, correlation_matrix, num_to_str_fn=None, return_components=True
        )
        total_f1 += float(res["f1"])  # type: ignore[index]
        total_p += float(res["precision"])  # type: ignore[index]
        total_r += float(res["recall"])  # type: ignore[index]
        count += 1

    if count == 0:
        return (
            {"f1": 0.0, "precision": 0.0, "recall": 0.0}
            if return_components
            else 0.0
        )

    mean_f1 = total_f1 / count
    if not return_components:
        return mean_f1
    return {
        "f1": mean_f1,
        "precision": total_p / count,
        "recall": total_r / count,
    }


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (
        (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        ),
        precision,
        recall,
    )


def _onehot_to_labels(
    y: Iterable[Iterable[int]], labels: list[str]
) -> list[list[str]]:
    return [[labels[i] for i, v in enumerate(row) if v] for row in y]


def _looks_like_onehot(
    rows: Iterable[Iterable[int | float]], labels: list[str]
) -> bool:
    rows = list(rows)
    if not rows:
        return False
    expected_length = len(labels)
    if expected_length == 0:
        return False
    allowed_values = {0, 0.0, 1, 1.0}
    for row in rows:
        if len(row) != expected_length:
            return False
        if not all(isinstance(v, (int, float)) for v in row):
            return False
        if not set(row).issubset(allowed_values):
            return False
    return True


def _materialize_examples(
    xs: Iterable[Iterable[str | int]],
) -> list[list[str | int]]:
    materialized: list[list[str | int]] = []
    for x in xs:
        if isinstance(x, set):
            materialized.append(list(x))
        elif isinstance(x, (list, tuple)):
            materialized.append(list(x))
        else:
            materialized.append([x])
    return materialized


def _ensure_iter_of_sets(xs: Iterable[Iterable[str]]):
    # normalize to list[set[str]]
    return [set(x) if not isinstance(x, set) else x for x in xs]


def _maybe_map_nums(xs, fn: Callable | None):
    if fn is None:
        return xs
    mapped = []
    for x in xs:
        mapped_row = fn(list(x))
        mapped.append(list(mapped_row))
    return mapped


def _labels_from_matrix(corr: pd.DataFrame) -> list[str]:
    # assumes symmetric, labeled by class names on both axes
    return list(corr.index)


def semantic_micro_f1_score(
    preds: Iterable[Iterable[str | int]],
    trues: Iterable[Iterable[str | int]],
    correlation_matrix: pd.DataFrame,
    num_to_str_fn: Callable | None = None,
    *,
    labels: list[str] | None = None,
    return_components: bool = False,
) -> float | dict[str, float]:
    """
    Semantic micro F1 (one-vs-rest, similarity-weighted).
    Identity matrix => standard micro F1.

    Args:
        return_components: when True, returns a dict with keys
            {"f1", "precision", "recall"}; otherwise returns a float F1.
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

    labels = labels or _labels_from_matrix(correlation_matrix)

    tp = fp = fn = 0.0

    for P, T in zip(preds_sets, trues_sets):
        res = pointwise_semantic_f1_score(
            P,
            T,
            correlation_matrix,
            num_to_str_fn=None,
            return_components=True,
        )

        forward_pairs = {p: t for p, t in res["forward_pairs"]}
        reverse_pairs = {t: p for t, p in res["reverse_pairs"]}

        tp += sum(
            (
                (
                    correlation_matrix.loc[forward_pairs[p], p]
                    if p in forward_pairs
                    else 0
                )
                for p in P
            )
        )
        fp += sum(
            (
                (
                    1 - correlation_matrix.loc[forward_pairs[p], p]
                    if p in forward_pairs
                    else 1
                )
                for p in P
            )
        )
        fn += sum(
            (
                (
                    1 - correlation_matrix.loc[t, reverse_pairs[t]]
                    if t in reverse_pairs
                    else 1
                )
                for t in T
            )
        )

    f1, precision, recall = _f1_from_counts(tp, fp, fn)
    if not return_components:
        return f1
    return {"f1": f1, "precision": precision, "recall": recall}


def semantic_macro_f1_score(
    preds: Iterable[Iterable[str | int]],
    trues: Iterable[Iterable[str | int]],
    correlation_matrix: pd.DataFrame,
    num_to_str_fn: Callable | None = None,
    *,
    average: str | None = "macro",
    labels: list[str] | None = None,
    return_components: bool = False,
) -> dict[str, float] | float | dict[str, dict[str, float]]:
    """
    Semantic macro/weighted F1 (one-vs-rest, similarity-weighted).

    Args:
        preds, trues: list-like of sets/lists of predicted and true labels
        correlation_matrix: pd.DataFrame of similarities in [0,1]
        num_to_str_fn: optional mapping function if labels are numeric
        average:
            - "macro": unweighted mean of per-class semantic F1
            - "weighted": support-weighted mean of per-class semantic F1
            - None: return dict of {class: semantic F1}
        return_components:
            - if average is "macro"/"weighted": return dict {"f1","precision","recall"}
            - if average is None: return dict {class: {"f1","precision","recall"}}

    Returns:
        float if average is "macro"/"weighted", else dict[str, float];
        when return_components is True, returns component dicts as noted above.
    """
    assert average in {"macro", "weighted", None}

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

    labels = labels or _labels_from_matrix(correlation_matrix)

    # accumulate per-class soft counts
    cls_tp = {c: 0.0 for c in labels}
    cls_fp = {c: 0.0 for c in labels}
    cls_fn = {c: 0.0 for c in labels}
    supports = {c: 0 for c in labels}

    for P, T in zip(preds_sets, trues_sets):
        res = pointwise_semantic_f1_score(
            P,
            T,
            correlation_matrix,
            num_to_str_fn=None,
            return_components=True,
        )

        forward_pairs = {p: t for p, t in res["forward_pairs"]}
        reverse_pairs = {t: p for t, p in res["reverse_pairs"]}

        for c in labels:
            supports[c] += 1 if c in T else 0
            cls_tp[c] += (
                correlation_matrix.loc[forward_pairs[c], c]
                if c in P and c in forward_pairs
                else 0.0
            )
            cls_fp[c] += (
                1
                - (
                    correlation_matrix.loc[forward_pairs[c], c]
                    if c in P and c in forward_pairs
                    else 0.0
                )
                if c in P
                else 0.0
            )
            cls_fn[c] += (
                1
                - (
                    correlation_matrix.loc[c, reverse_pairs[c]]
                    if c in T and c in reverse_pairs
                    else 0.0
                )
                if c in T
                else 0.0
            )

    per_class_precision = {
        c: (
            cls_tp[c] / (cls_tp[c] + cls_fp[c])
            if (cls_tp[c] + cls_fp[c]) > 0
            else 0.0
        )
        for c in labels
    }
    per_class_recall = {
        c: (
            cls_tp[c] / (cls_tp[c] + cls_fn[c])
            if (cls_tp[c] + cls_fn[c]) > 0
            else 0.0
        )
        for c in labels
    }
    per_class_f1 = {
        c: _f1_from_counts(cls_tp[c], cls_fp[c], cls_fn[c])[0] for c in labels
    }

    if average is None:
        if not return_components:
            return per_class_f1
        return {
            c: {
                "f1": per_class_f1[c],
                "precision": per_class_precision[c],
                "recall": per_class_recall[c],
            }
            for c in labels
        }

    if average == "macro":
        f1 = sum(per_class_f1.values()) / len(labels) if labels else 0.0
        if not return_components:
            return f1
        precision = (
            sum(per_class_precision.values()) / len(labels) if labels else 0.0
        )
        recall = sum(per_class_recall.values()) / len(labels) if labels else 0.0
        return {"f1": f1, "precision": precision, "recall": recall}
    else:  # weighted
        total_support = sum(supports.values())
        if total_support == 0:
            # no positives anywhere -> fall back to macro mean
            f1 = sum(per_class_f1.values()) / len(labels) if labels else 0.0
            if not return_components:
                return f1
            precision = (
                sum(per_class_precision.values()) / len(labels)
                if labels
                else 0.0
            )
            recall = (
                sum(per_class_recall.values()) / len(labels) if labels else 0.0
            )
        else:
            f1 = (
                sum(per_class_f1[c] * supports[c] for c in labels)
                / total_support
                if labels
                else 0.0
            )
            if not return_components:
                return f1
            precision = (
                sum(per_class_precision[c] * supports[c] for c in labels)
                / total_support
                if labels
                else 0.0
            )
            recall = (
                sum(per_class_recall[c] * supports[c] for c in labels)
                / total_support
                if labels
                else 0.0
            )
        return {"f1": f1, "precision": precision, "recall": recall}


def _wrap_single_label_to_singletons(
    y: Iterable[str | int] | Iterable[Iterable[str | int]],
) -> list[list[str | int]]:
    # Determine if input is flat (single-label) or already iterable of iterables
    ys = list(y)
    if len(ys) == 0:
        return []
    first = ys[0]
    if isinstance(first, (list, set, tuple)):
        return [list(v) for v in ys]
    # flat
    return [[v] for v in ys]


def semantic_f1_score(
    y_true: Iterable[str | int] | Iterable[Iterable[str | int]],
    y_pred: Iterable[str | int] | Iterable[Iterable[str | int]],
    correlation_matrix: pd.DataFrame,
    *,
    labels: list[str] | None = None,
    average: str | None = "samples",
) -> float | list[float]:
    """
    Semantic F1 with sklearn-compatible interface plus similarity matrix.

    Args mirror sklearn.metrics.f1_score (y_true first), with:
    - average in {"samples","micro","macro","weighted","binary", None}
    - labels: restrict classes for micro/macro/weighted/None
    """
    # Normalize to list[list[...]] (single-label => singleton lists)
    y_true_wrapped = _wrap_single_label_to_singletons(y_true)
    y_pred_wrapped = _wrap_single_label_to_singletons(y_pred)

    matrix_labels = _labels_from_matrix(correlation_matrix)
    if _looks_like_onehot(y_true_wrapped, matrix_labels) and _looks_like_onehot(
        y_pred_wrapped, matrix_labels
    ):
        y_true_labels = _onehot_to_labels(y_true_wrapped, matrix_labels)
        y_pred_labels = _onehot_to_labels(y_pred_wrapped, matrix_labels)
    else:
        # Attempt to auto-detect string vs int labels; if not strings, cast to str
        # Note: callers who need numeric-to-string mapping should pass already mapped.
        y_true_labels = [list(map(str, row)) for row in y_true_wrapped]
        y_pred_labels = [list(map(str, row)) for row in y_pred_wrapped]

    if average == "samples":
        return float(
            samples_semantic_f1_score(
                y_pred_labels, y_true_labels, correlation_matrix
            )
        )
    elif average == "micro":
        return float(
            semantic_micro_f1_score(
                y_pred_labels, y_true_labels, correlation_matrix, labels=labels
            )
        )
    elif average in {"macro", "weighted"}:
        return float(
            semantic_macro_f1_score(
                y_pred_labels,
                y_true_labels,
                correlation_matrix,
                average=average,
                labels=labels,
            )
        )
    elif average is None:
        # Return per-class scores in the provided label order (or matrix order)
        per_class = semantic_macro_f1_score(
            y_pred_labels,
            y_true_labels,
            correlation_matrix,
            average=None,
            labels=labels,
        )
        # Convert dict to list ordered by labels
        order = labels or _labels_from_matrix(correlation_matrix)
        return [float(per_class[c]) for c in order]

    else:
        raise ValueError(
            "average must be one of {'samples','micro','macro','weighted','binary', None}"
        )
