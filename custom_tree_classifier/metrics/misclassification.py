import numpy as np

from custom_tree_classifier.metrics.metric_base import MetricBase


class MisclassificationError(MetricBase):
    """
    A class that implements the misclassification error metric for decision trees.
    """

    @staticmethod
    def compute_metric(metric_data: np.ndarray) -> np.float64:
        """
        Compute the misclassification error of the given data.

        Parameters
        ----------
        metric_data : np.ndarray
            The data used to compute the misclassification error. The target
            variable is assumed to be in the first column.

        Returns
        -------
        np.float64
            The computed misclassification error.
        """

        y = metric_data[:, 0]
        n = len(y)

        if n == 0:
            return np.float64(0.0)

        prop0 = np.sum(y == 0) / n
        prop1 = 1.0 - prop0

        # Misclassification error = 1 - maximum class proportion
        error = 1.0 - max(prop0, prop1)
        return error

    @staticmethod
    def compute_delta(
            split: np.ndarray,
            metric_data: np.ndarray
        ) -> np.float64:
        """
        Compute the reduction in misclassification error (delta) for a given split.

        Parameters
        ----------
        split : np.ndarray
            A boolean array indicating the split (True for left, False for right).
        metric_data : np.ndarray
            The data used to compute the misclassification error.

        Returns
        -------
        np.float64
            The reduction in misclassification error for the split.
        """


        parent_metric = MisclassificationError.compute_metric(metric_data)

        left_metric = MisclassificationError.compute_metric(metric_data[split])
        right_metric = MisclassificationError.compute_metric(metric_data[~split])

        p_left = np.mean(split)
        p_right = 1.0 - p_left

        delta = parent_metric - (left_metric * p_left + right_metric * p_right)
        return delta
