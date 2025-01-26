import numpy as np

from custom_tree_classifier.metrics.metric_base import MetricBase


class Gini(MetricBase):
    """
    A class that implements the Gini impurity metric for decision trees.
    """

    @staticmethod
    def compute_metric(metric_data: np.ndarray) -> np.float64:
        """
        Compute the Gini impurity of the given data.

        Parameters
        ----------
        metric_data : np.ndarray
            The data used to compute the Gini impurity. The target variable is
            assumed to be in the first column.

        Returns
        -------
        np.float64
            The computed Gini impurity.
        """

        y = metric_data[:, 0]

        prop0 = np.sum(y == 0) / len(y)
        prop1 = np.sum(y == 1) / len(y)

        metric = 1 - (prop0**2 + prop1**2)

        return metric

    @staticmethod
    def compute_delta(
            split: np.ndarray,
            metric_data: np.ndarray
        ) -> np.float64:
        """
        Compute the reduction in Gini impurity (delta) for a given split.

        Parameters
        ----------
        split : np.ndarray
            A boolean array indicating the split (True for left, False for right).
        metric_data : np.ndarray
            The data used to compute the Gini impurity.

        Returns
        -------
        np.float64
            The reduction in Gini impurity for the split.
        """

        delta = (
            Gini.compute_metric(metric_data) -
            Gini.compute_metric(metric_data[split]) * np.mean(split) -
            Gini.compute_metric(metric_data[np.invert(split)]) * (1 - np.mean(split))
        )

        return delta
