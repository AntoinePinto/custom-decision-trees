from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class MetricBase(ABC):
    """
    Abstract base class for split evaluation metrics used in decision
    trees.

    Concrete implementations must provide a `compute_metric` method,
    which evaluates the effectiveness of a potential split using a
    user-defined metric.
    """

    @abstractmethod
    def __init__(
            self,
            **kwargs
        ) -> None:
        """
        Initialize the metric base class.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional arguments that may be needed for specific metric implementations.
        """

        pass

    @abstractmethod
    def compute_metric(
            self,
            metric_data: np.ndarray,
            mask: np.ndarray
        ) -> Tuple[float, Dict]:
        """
        Compute the metric delta (gain) resulting from a proposed data split.

        Parameters
        ----------
        metric_data : np.ndarray
            A 2D NumPy array of data required by the metric. Typically includes
            target labels or prediction-relevant values.
        mask : np.ndarray
            A boolean array indicating which samples go to the first branch of the
            split. The complement defines the second branch.

        Returns
        -------
        Tuple[float, Dict]
            A tuple where the first element is the metric gain (float),
            and the second is a dictionary containing any relevant metadata or
            diagnostics.
        """

        pass
