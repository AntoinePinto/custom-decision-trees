from typing import Callable

import numpy as np

from custom_decision_trees import DecisionTreeRegressor
from custom_decision_trees.metrics import MetricBase
from custom_decision_trees.models.random_forest import RandomForest


class RandomForestRegressor(RandomForest):
    """
    A implementation of a random forest with configurable estimators and splitting
    metrics.
    """

    def __init__(
            self,
            metric: MetricBase | None = None,
            n_estimators: int = 100,
            max_depth: int = 5,
            min_samples_split: int | float = 2,
            min_samples_leaf: int | float = 2,
            max_features: str | int | None = "sqrt",
            nb_max_conditions_per_node = 2,
            nb_max_cut_options_per_var = 2,
            nb_max_split_options_per_node: int | None = None,
            bootstrap: bool = True,
            max_samples: int | float | None = None,
            random_state: int | None = None,
            n_jobs: int = 1,
        ) -> None:
        """
        An implementation of a random forest classifier using custom decision trees and
        flexible metrics.

        Parameters
        ----------
        metric : MetricBase
            The metric used to evaluate split quality in each tree.
        n_estimators : int, default=100
            The number of trees in the forest.
        max_depth : int, default=5
            The maximum depth of each individual decision tree.
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node.
            If float, represents a fraction of the number of samples.
        min_samples_leaf : int or float, default=2
            The minimum number of samples required to be at a leaf node.
            If float, represents a fraction of the number of samples.
        max_features : {'sqrt', 'log2'} or int or float or None, default='sqrt'
            The number of features to consider when looking for the best split.
            If float, represents a fraction of the number of features.
        nb_max_conditions_per_node : int, default=2
            The maximum number of conditions combined per decision node (via AND).
        nb_max_cut_options_per_var : int, default=2
            The maximum number of cut options to evaluate per variable.
        nb_max_split_options_per_node
            Maximum number of splits to be tested per node to avoid overly long
            calculations in multi-condition mode
        bootstrap : bool, default=True
            Whether to bootstrap samples when building trees.
        max_samples : int, float or None, default=None
            The number of samples to draw when bootstrapping.
            If float, represents a fraction of the training set. If None, use all
            samples.
        random_state : int or None, default=None
            Controls the randomness of the bootstrapping and other stochastic processes.
        n_jobs : int, default=1
            The number of parallel jobs to run when building the forest.
        """

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

    def train_decision_tree(
            self,
            X: np.ndarray,
            y: np.ndarray,
            metric_data: np.ndarray,
            max_samples: int,
            batch_size: int = 1000,
            tqdm_func: Callable = None,
        ) -> DecisionTreeRegressor:
        """
        Train a single decision tree on a bootstrapped subset of the data.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        metric_data : np.ndarray
            Metric-specific data for split evaluation.
        max_samples : int
            Number of samples to use for training this tree.
        batch_size : int, default=1000
            Batch size for parallel evaluations.
        tqdm_func : callable, optional
            Progress bar utility (e.g., `tqdm`).

        Returns
        -------
        decision_tree : DecisionTree
            The trained decision tree.
        """

        observations_sample = np.array(range(X.shape[0]))
        if self.bootstrap is True:
            observations_sample = np.random.choice(
                observations_sample,
                size=max_samples,
                replace=True,
            )

        decision_tree = DecisionTreeRegressor(
            metric=self.metric,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            nb_max_conditions_per_node=self.nb_max_conditions_per_node,
            nb_max_cut_options_per_var=self.nb_max_cut_options_per_var,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        decision_tree.fit(
            X=X[observations_sample,:],
            y=y[observations_sample],
            metric_data=metric_data[observations_sample],
            batch_size=batch_size,
            tqdm_func=tqdm_func,
        )

        return decision_tree
