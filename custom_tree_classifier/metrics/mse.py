import numpy as np

from custom_tree_classifier.metrics.metric_base import MetricBase


class MeanSquaredError(MetricBase):
    """
    A class that implements the mean squared error (MSE) metric for decision trees.
    """

    @staticmethod
    def compute_metric(metric_data: np.ndarray) -> np.float64:
        """
        Compute the mean squared error (MSE) of the given data.

        Parameters
        ----------
        metric_data : np.ndarray
            The data used to compute the MSE. The target variable is assumed
            to be in the first column.

        Returns
        -------
        np.float64
            The computed mean squared error.
        """

        y = metric_data[:, 0]
        n = len(y)

        if n == 0:
            return np.float64(0.0)

        p = np.mean(y)  # proportion of '1'
        mse = np.float64(p * (1.0 - p))  # p(1 - p)
        return mse

    @staticmethod
    def compute_delta(
            split: np.ndarray,
            metric_data: np.ndarray
        ) -> np.float64:
        """
        Compute the reduction in mean squared error (delta) for a given split.

        Parameters
        ----------
        split : np.ndarray
            A boolean array indicating the split (True for left, False for right).
        metric_data : np.ndarray
            The data used to compute the MSE.

        Returns
        -------
        np.float64
            The reduction in mean squared error for the split.
        """

        parent_metric = MeanSquaredError.compute_metric(metric_data)

        left_metric = MeanSquaredError.compute_metric(metric_data[split])
        right_metric = MeanSquaredError.compute_metric(metric_data[~split])

        p_left = np.mean(split)
        p_right = 1.0 - p_left

        delta = parent_metric - (left_metric * p_left + right_metric * p_right)
        return delta
