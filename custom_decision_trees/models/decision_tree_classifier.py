from typing import Callable, List, Literal

import numpy as np

from custom_decision_trees.metrics import Gini, MetricBase
from custom_decision_trees.models.decision_tree import DecisionTree


class DecisionTreeClassifier(DecisionTree):
    """
    A custom implementation of a decision tree classifier with support for different
    splitting metrics.
    """

    def __init__(
            self,
            metric: MetricBase | None = None,
            max_depth: int = 5,
            min_samples_split: int | float = 2,
            min_samples_leaf: int | float = 2,
            max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
            nb_max_conditions_per_node: int = 1,
            nb_max_cut_options_per_var: int = 10,
            nb_max_split_options_per_node: int | None = None,
            random_state: int | None = None,
            n_jobs: int = 1,
        ) -> None:
        """
        Custom Decision Tree classifier with support for different splitting metrics.

        This implementation allows control over splitting criteria, depth, minimum
        samples, feature sampling, and parallelization. It supports visualization
        and prediction functionalities.

        Parameters
        ----------
        metric : MetricBase or None, default=None
            The splitting metric to use. If None, defaults to Gini index.
        max_depth : int, default=5
            The maximum depth of the tree.
        min_samples_split : int or float, default=2
            The minimum number of samples required to split a node. If float, represents
            a fraction of the dataset.
        min_samples_leaf : int or float, default=2
            The minimum number of samples required at a leaf node. If float, represents
            a fraction of the dataset.
        max_features : str, int, float, or None, default='sqrt'
            The number of features to consider when looking for the best split. Can be
            'sqrt', 'log2', None, int or float.
        nb_max_conditions_per_node : int, default=2
            The maximum number of conditions per node.
        nb_max_cut_options_per_var : int, default=2
            The maximum number of cut options to evaluate per variable.
        nb_max_split_options_per_node
            Maximum number of splits to be tested per node to avoid overly long
            calculations in multi-condition mode
        random_state : int or None, default=None
            Seed for reproducibility.
        n_jobs : int, default=1
            Number of parallel jobs to run.
        """

        if metric is None:
            metric = Gini()

        super().__init__(
            metric=metric,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            nb_max_conditions_per_node=nb_max_conditions_per_node,
            nb_max_cut_options_per_var=nb_max_cut_options_per_var,
            nb_max_split_options_per_node=nb_max_split_options_per_node,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def get_partition_value(
            self,
            y: np.ndarray,
        ) -> List[int]:
        """
        Compute the class distribution of the target variable.

        Parameters
        ----------
        y : np.ndarray
            The target variable.

        Returns
        -------
        list[int]
            A list containing the count of each class.
        """

        repartition = [int(np.sum(y == i)) for i in self.classes]

        return repartition

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            metric_data: np.ndarray,
            batch_size: int = 1000,
            tqdm_func: Callable | None = None,
            classes: np.ndarray | None = None,
        ) -> None:
        """
        Fit the decision tree classifier on the provided data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target values.
        metric_data : np.ndarray
            Data used by the splitting metric.
        classes: np.ndarray
            List of possible classes
        batch_size : int, default=1000
            Batch size for parallel computation of split evaluations.
        tqdm_func : callable or None, default=None
            Optional progress bar function.
        """

        self.classes = np.unique(y) if classes is None else classes

        super().fit(
            X=X,
            y=y,
            metric_data=metric_data,
            batch_size=batch_size,
            tqdm_func=tqdm_func,
        )

    def predict_proba(
            self,
            X: np.ndarray,
        ) -> np.ndarray:
        """
        Predict class probabilities as an array.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """

        predictions = self.predict_all(X=X)
        probas = np.array([p.value for p in predictions])

        return probas

    def predict(
            self,
            X: np.ndarray,
        ) -> np.ndarray:
        """
        Predict class labels for the given input samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).
        """

        probas = self.predict_proba(
            X=X
        )

        predictions = self.classes[np.argmax(probas, axis=1)]

        return predictions
