from typing import Dict, Tuple

import numpy as np

from .metric_base import MetricBase


class Entropy(MetricBase):
    """
    A class that implements the Entropy (Information Gain) metric for decision trees.
    Supports multi-class classification.
    Inherits from MetricBase.
    """

    def __init__(
            self,
            n_classes: int = 2
        ) -> None:
        """
        Initialize the Entropy metric.

        Parameters
        ----------
        n_classes : int
            Number of classes in the dataset. Default is 2.
        """
        self.n_classes = n_classes
        self.max_entropy = np.log2(n_classes)

    def compute_entropy(
            self,
            metric_data: np.ndarray
        ) -> float:
        """
        Compute the entropy for a classification dataset.

        Parameters
        ----------
        metric_data : np.ndarray
            A 2D NumPy array where the first column represents class labels.

        Returns
        -------
        float
            The entropy score. Returns max_entropy if the input array is empty.
        """
        y = metric_data[:, 0]
        nb_obs = len(y)

        if nb_obs == 0:
            return self.max_entropy

        # Calculate proportions for each class
        props = [(np.sum(y == i) / nb_obs) for i in np.unique(y)]

        # Compute entropy
        entropy = -np.sum([p * np.log2(p) for p in props if p > 0])

        return float(entropy)

    def compute_metric(
            self,
            metric_data: np.ndarray,
            mask: np.ndarray
        ) -> Tuple[float, Dict]:
        """
        Compute the entropy gain (information gain) from a potential split.

        Parameters
        ----------
        metric_data : np.ndarray
            A 2D NumPy array of metric-related data. The first column should contain
            labels.
        mask : np.ndarray
            A boolean mask indicating which rows belong to the first side of the split.

        Returns
        -------
        Tuple[float, Dict]
            A tuple containing:
            - The computed entropy gain (float).
            - A dictionary with the entropy value for the first split side.
        """
        entropy_parent = self.compute_entropy(metric_data)
        entropy_side1 = self.compute_entropy(metric_data[mask])
        entropy_side2 = self.compute_entropy(metric_data[~mask])

        delta = (
            entropy_parent -
            entropy_side1 * np.mean(mask) -
            entropy_side2 * (1 - np.mean(mask))
        )

        metadata = {"entropy": round(entropy_side1, 3)}

        return float(delta), metadata
