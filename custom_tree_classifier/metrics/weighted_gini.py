import numpy as np

from custom_tree_classifier.metrics.metric_base import MetricBase


class WeightedGini(MetricBase):
    """
    A class that implements the weighted Gini impurity metric for decision trees.
    """

    @staticmethod
    def compute_metric(metric_data: np.ndarray) -> np.float64:
        """
        Compute the weighted Gini impurity of the given data.

        Parameters
        ----------
        metric_data : np.ndarray
            The data used to compute the weighted Gini impurity. The target variable
            is assumed to be in the first column, and weights in the second column.

        Returns
        -------
        np.float64
            The computed weighted Gini impurity.
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

        prop0 = w0 / w_total
        prop1 = w1 / w_total

        gini = 1.0 - (prop0**2 + prop1**2)
        return gini

    @staticmethod
    def compute_delta(
            split: np.ndarray,
            metric_data: np.ndarray
        ) -> np.float64:
        """
        Compute the reduction in weighted Gini impurity (delta) for a given split.

        Parameters
        ----------
        split : np.ndarray
            A boolean array indicating the split (True for left, False for right).
        metric_data : np.ndarray
            The data used to compute the weighted Gini impurity. The second column
            contains weights.

        Returns
        -------
        np.float64
            The reduction in weighted Gini impurity for the split.
        """

        parent_metric = WeightedGini.compute_metric(metric_data)

        left_metric = WeightedGini.compute_metric(metric_data[split])
        right_metric = WeightedGini.compute_metric(metric_data[~split])

        w = metric_data[:, 1]
        w_total = np.sum(w)

        if w_total == 0:
            return np.float64(0.0)

        w_left = np.sum(w[split])
        p_left = w_left / w_total
        p_right = 1.0 - p_left

        delta = parent_metric - (left_metric * p_left + right_metric * p_right)
        return delta
