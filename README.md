# Semantic F1 Score

<p align="center">
  <img src="media/qual_thumbnail.png" alt="Qualitative example of Semantic F1 bidirectional matching" width="720">
  <br>
  <em>Semantic F1 grants partial credit by matching predictions and gold labels in both directions.</em>
</p>

Semantic F1 is a drop-in replacement for `sklearn` hard `f1_score` in subjective or fuzzy multi-label classification. It keeps the familiar precision-recall framing while using a domain similarity matrix to acknowledge when "wrong" labels are still semantically close. The package is the reference implementation accompanying the <em>Semantic F1 Scores: Fair Evaluation Under Fuzzy Class Boundaries</em>.

## Installation
```bash
pip install semantic-f1-score
```

The library depends only on `pandas` and `scipy`. For development extras, install with `pip install semantic_f1_score[test]` and run `pytest`.

## Highlights
- Two-step semantic precision/recall penalizes both over-prediction and under-coverage, avoiding the forced matches that plague single-pass or Hungarian-style alignment metrics.
- When the similarity matrix is the identity, every variant (pointwise, samples, micro, macro) reduces exactly to the standard F1, so existing evaluation pipelines stay compatible.
- Operates on metric and non-metric label spaces, and even continuous embeddings, making it suitable for emotions, moral foundations, negotiation strategies, and other fuzzy domains.
- Empirically monotonic with error rate and magnitude, robust to partially misspecified similarity matrices, and better aligned with downstream outcomes such as donation success in negotiation datasets (see paper for details).
- Lightweight pandas-based API with helpers for pointwise inspection and scikit-learn compatible averaging schemes.



## Quick Start
```python
import pandas as pd
from semantic_f1_score import semantic_f1_score, pointwise_semantic_f1_score

labels = ["anger", "disgust", "joy"]
S = pd.DataFrame(
    [
        [1.0, 0.7, 0.1],
        [0.7, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ],
    index=labels,
    columns=labels,
)

# Multi-label examples
y_true = [["anger", "disgust"], ["joy"], ["disgust"]]
y_pred = [["anger"], ["joy"], ["anger"]]

# also supports one-hot encoding with the same order as the similarity matrix S
print("Semantic micro F1", semantic_f1_score(y_true, y_pred, S, average="micro"))
print("Semantic macro F1", semantic_f1_score(y_true, y_pred, S, average="macro"))
print("Semantic samples F1", semantic_f1_score(y_true, y_pred, S, average="samples"))

# Inspect a single example
components = pointwise_semantic_f1_score(
    y_pred[0],
    y_true[0],
    S,
    return_components=True,
)
print("Pointwise components", components)
```

By design, using an identity matrix will give you the exact same scores as scikit-learn's F1 implementations. One-hot encoded inputs are detected automatically, and you can supply numeric labels via a mapping callback.

## Metric Variants
- `pointwise_semantic_f1_score` - semantic precision/recall plus harmonic mean for a single example, optionally returning the matched pairs.
- `semantic_f1_score(..., average="samples")` - mean of pointwise scores across a batch (default behaviour).
- `semantic_f1_score(..., average="micro"|"macro"|"weighted"|None)` - scikit-learn style aggregations that treat partial credit as soft counts.
- `semantic_f1_score(..., average=None)` - per-class semantic F1 values, ordered by the similarity matrix labels.
- `extended_hungarian_match` / `hungarian_score` - reproduces the Hungarian-style baseline analysed in the paper for comparison purposes.

## Crafting a Similarity Matrix
Semantic F1 only assumes a symmetric square matrix with values in \[0, 1]. In practice you can:
- Derive similarities from theory-driven structures (e.g. Plutchik's wheel of emotions, moral foundation clusters).
- Estimate them from data, such as normalized label co-occurrence or correlation matrices.
- Project labels into shared embeddings (e.g. sentence-level or concept-level encoders) and convert distances to similarities.
- Start with the identity matrix when no partial credit is desired, scores remain exact F1 while the API stays consistent.

Section B of the paper discusses best practices, including keeping on-diagonal values at 1, capping cross-cluster credit in non-metric spaces, and stress-testing metrics against perturbed matrices.


## Development
```bash
# Clone and install in editable mode
pip install -e .[dev,test]

# Run the regression tests
pytest -q
```

Pull requests and issues are welcome on GitHub.

## Citation

TBD

## License
Released under the MIT License. See `LICENSE` for details.
