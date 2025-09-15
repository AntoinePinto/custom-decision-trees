from typing import Dict, Tuple

import numpy as np

from .metric_base import MetricBase


def compute_gini(metric_data: np.ndarray) -> float:
    """
    Compute the Gini impurity for a binary classification dataset.

    Parameters
    ----------
    metric_data : np.ndarray
        A 2D NumPy array where the first column represents binary class labels (0 or 1).

    Returns
    -------
    float
        The Gini impurity score. Returns 0.5 if the input array is empty.
    """

    y = metric_data[:, 0]

    if len(y) == 0:
        return 0.5

    prop0 = np.sum(y == 0) / len(y)
    prop1 = np.sum(y == 1) / len(y)

    metric = 1 - (prop0**2 + prop1**2)

    return float(metric)

class Gini(MetricBase):
    """
    A class that implements the Gini impurity metric for decision trees.
    Inherits from MetricBase.
    """

    def __init__(self,) -> None:
        pass

    def compute_metric(
            self,
            metric_data: np.ndarray,
            mask: np.ndarray
        ) -> Tuple[float, Dict]:
        """
        Compute the Gini gain (delta impurity) from a potential split.

        Parameters
        ----------
        metric_data : np.ndarray
            A 2D NumPy array of metric-related data. The first column should contain
            binary labels.
        mask : np.ndarray
            A boolean mask indicating which rows belong to the first side of the split.

        Returns
        -------
        Tuple[float, Dict]
            A tuple containing:
            - The computed Gini gain (float).
            - A dictionary with the Gini value for the first split side.
        """

        gini_parent = compute_gini(metric_data)
        gini_side1 = compute_gini(metric_data[mask])
        gini_side2 = compute_gini(metric_data[~mask])

        delta = (
            gini_parent -
            gini_side1 * np.mean(mask) -
            gini_side2 * (1 - np.mean(mask))
        )

        metadata = {"gini": round(gini_side1, 3)}

        return float(delta), metadata
