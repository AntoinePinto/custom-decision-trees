import numpy as np

from custom_tree_classifier.metrics.metric_base import MetricBase


class WeightedEntropy(MetricBase):
    """
    A class that implements the weighted entropy metric for decision trees.
    """

    @staticmethod
    def compute_metric(
            metric_data: np.ndarray
        ) -> np.float64:
        """
        Compute the weighted entropy of the given data.

        Parameters
        ----------
        metric_data : np.ndarray
            The data used to compute the weighted entropy. The target variable
            is assumed to be in the first column, and weights in the second column.

        Returns
        -------
        np.float64
            The computed weighted entropy.
        """

        if len(metric_data) == 0:
            return np.float64(0.0)

        y = metric_data[:, 0]
        w = metric_data[:, 1]

        w_total = np.sum(w)
        if w_total == 0:
            return np.float64(0.0)

        w0 = np.sum(w[y == 0])
        w1 = w_total - w0

        # If either is 0, entropy is 0 (pure node)
        if w0 == 0 or w1 == 0:
            return np.float64(0.0)

        p0 = w0 / w_total
        p1 = w1 / w_total

        # Weighted entropy
        entropy = - (p0 * np.log2(p0) + p1 * np.log2(p1))
        return entropy

    @staticmethod
    def compute_delta(
            split: np.ndarray,
            metric_data: np.ndarray
        ) -> np.float64:
        """
        Compute the reduction in weighted entropy (delta) for a given split.

        Parameters
        ----------
        split : np.ndarray
            A boolean array indicating the split (True for left, False for right).
        metric_data : np.ndarray
            The data used to compute the weighted entropy. The second column contains
            weights.

        Returns
        -------
        np.float64
            The reduction in weighted entropy for the split.
        """

        parent_metric = WeightedEntropy.compute_metric(metric_data)

        left_metric = WeightedEntropy.compute_metric(metric_data[split])
        right_metric = WeightedEntropy.compute_metric(metric_data[~split])

        w = metric_data[:, 1]
        w_total = np.sum(w)
        if w_total == 0:
            return np.float64(0.0)

        w_left = np.sum(w[split])
        p_left = w_left / w_total
        p_right = 1.0 - p_left

        delta = parent_metric - (left_metric * p_left + right_metric * p_right)
        return delta
