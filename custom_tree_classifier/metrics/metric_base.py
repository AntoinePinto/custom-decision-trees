from abc import ABC, abstractmethod

import numpy as np


class MetricBase(ABC):
    """
    Abstract base class for metric computation.
    """

    @staticmethod
    @abstractmethod
    def compute_metric(metric_data: np.ndarray) -> np.float64:
        """
        Compute the metric value.
        """
        pass

    @staticmethod
    @abstractmethod
    def compute_delta(split: np.ndarray, metric_data: np.ndarray) -> np.float64:
        """
        Compute the delta of the metric based on a split.
        """
        pass
