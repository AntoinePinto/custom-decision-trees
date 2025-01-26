import numpy as np

from custom_tree_classifier.metrics.metric_base import MetricBase


class Entropy(MetricBase):
    """
    A class that implements the entropy metric for decision trees.
    """

    @staticmethod
    def compute_metric(metric_data: np.ndarray) -> np.float64:
        """
        Compute the entropy of the given data.

        Parameters
        ----------
        metric_data : np.ndarray
            The data used to compute the entropy. The target variable is assumed
            to be in the first column.

        Returns
        -------
        np.float64
            The computed entropy.
        """

        # y is assumed to be in the 0th column of metric_data
        y = metric_data[:, 0]
        n = len(y)

        if n == 0:
            return np.float64(0.0)

        prop0 = np.sum(y == 0) / n
        prop1 = 1.0 - prop0

        # Small epsilon to avoid log(0)
        prop0 = max(prop0, 1e-15)
        prop1 = max(prop1, 1e-15)

        entropy = - (prop0 * np.log2(prop0) + prop1 * np.log2(prop1))
        return entropy

    @staticmethod
    def compute_delta(
            split: np.ndarray,
            metric_data: np.ndarray
        ) -> np.float64:
        """
        Compute the information gain (delta) for a given split.

        Parameters
        ----------
        split : np.ndarray
            A boolean array indicating the split (True for left, False for right).
        metric_data : np.ndarray
            The data used to compute the entropy.

        Returns
        -------
        np.float64
            The computed information gain for the split.
        """
        # Overall (parent) entropy
        parent_metric = Entropy.compute_metric(metric_data)

        # Entropy of the left (True) and right (False) partitions
        left_metric = Entropy.compute_metric(metric_data[split])
        right_metric = Entropy.compute_metric(metric_data[~split])

        # Proportion of samples going left/right
        p_left = np.mean(split)
        p_right = 1.0 - p_left

        # Information gain = parent - [weighted children entropies]
        delta = parent_metric - (left_metric * p_left + right_metric * p_right)
        return delta
