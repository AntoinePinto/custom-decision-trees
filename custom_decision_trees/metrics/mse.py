from typing import Dict, Tuple

import numpy as np

from .metric_base import MetricBase


class MSE(MetricBase):
    """
    A class that implements the Mean Squared Error (MSE) metric for decision trees.
    Inherits from MetricBase.
    """

    def __init__(self) -> None:
        pass

    def compute_mse(
            self,
            metric_data: np.ndarray,
        ) -> float:
        """
        Compute the Mean Squared Error (variance proxy) for regression data.

        Parameters
        ----------
        metric_data : np.ndarray
            A 2D NumPy array where the first column represents continuous target values.

        Returns
        -------
        float
            The MSE (variance around the mean). Returns 0 if input is empty.
        """

        y = metric_data[:, 0]

        nb_obs = len(y)

        if nb_obs == 0:
            return 0

        mean_y = np.mean(y)
        mse = np.mean((y - mean_y) ** 2)

        return float(mse)

    def compute_metric(
            self,
            metric_data: np.ndarray,
            mask: np.ndarray,
        ) -> Tuple[float, Dict]:
        """
        Compute the MSE reduction (delta impurity) from a potential split.

        Parameters
        ----------
        metric_data : np.ndarray
            A 2D NumPy array of metric-related data. The first column should contain
            continuous target values.
        mask : np.ndarray
            A boolean mask indicating which rows belong to the first side of the split.

        Returns
        -------
        Tuple[float, Dict]
            A tuple containing:
            - The computed MSE reduction (float).
            - A dictionary with the MSE value for the first split side.
        """

        mse_parent = self.compute_mse(metric_data)
        mse_side1 = self.compute_mse(metric_data[mask])
        mse_side2 = self.compute_mse(metric_data[~mask])

        delta = (
            mse_parent
            - mse_side1 * np.mean(mask)
            - mse_side2 * (1 - np.mean(mask))
        )

        metadata = {"mse": round(mse_side1, 3)}

        return float(delta), metadata
