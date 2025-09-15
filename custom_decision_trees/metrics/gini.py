from typing import Dict, Tuple

import numpy as np

from .metric_base import MetricBase


class Gini(MetricBase):
    """
    A class that implements the Gini impurity metric for decision trees.
    Inherits from MetricBase.
    """

    def __init__(
            self,
            n_classes: int = 2,
        ) -> None:
        
        self.n_classes = n_classes
        self.max_impurity = 1 - 1 / n_classes

    def compute_gini(
            self,
            metric_data: np.ndarray,
        ) -> float:
        """
        Compute the Gini impurity for a binary classification dataset.

        Parameters
        ----------
        metric_data : np.ndarray
            A 2D NumPy array where the first column represents binary class labels (0 or 1).

        Returns
        -------
        float
            The Gini impurity score. Returns the max impurity if the input array is empty.
        """

        y = metric_data[:, 0]
        
        nb_obs = len(y)

        if nb_obs == 0:
            return self.max_impurity

        props = [(np.sum(y == i) / nb_obs) for i in np.unique(y)]

        metric = 1 - np.sum([prop**2 for prop in props])

        return float(metric)

    def compute_metric(
            self,
            metric_data: np.ndarray,
            mask: np.ndarray,
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

        gini_parent = self.compute_gini(metric_data)
        gini_side1 = self.compute_gini(metric_data[mask])
        gini_side2 = self.compute_gini(metric_data[~mask])

        delta = (
            gini_parent -
            gini_side1 * np.mean(mask) -
            gini_side2 * (1 - np.mean(mask))
        )

        metadata = {"gini": round(gini_side1, 3)}

        return float(delta), metadata
