from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def quality_stability(fold_ndcg: Sequence[float]) -> dict[str, float]:
    """Return the median and IQR used as selection evidence, not a fifth objective."""

    values = np.asarray(fold_ndcg, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or np.any(~np.isfinite(values)):
        raise ValueError("at least three finite fold values are required")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("fold NDCG values must be in [0, 1]")
    return {
        "quality_median": float(np.median(values)),
        "quality_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    }
