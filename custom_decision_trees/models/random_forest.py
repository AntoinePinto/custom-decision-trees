import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import numpy as np
from joblib import Parallel, delayed

from custom_decision_trees.metrics import MetricBase
from custom_decision_trees.models.decision_tree import DecisionTree
from custom_decision_trees.models.schemas import RandomForestPrediction


class RandomForest(ABC):
    """
    A implementation of a random forest with configurable estimators and splitting
    metrics.
    """

    forest: list[DecisionTree]

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

        self.metric = metric
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.nb_max_conditions_per_node = nb_max_conditions_per_node
        self.nb_max_cut_options_per_var = nb_max_cut_options_per_var
        self.bootstrap = bootstrap

        if isinstance(max_samples, float):
            if (max_samples <= 0) | (max_samples >= 1):
                raise ValueError(
                    "`max_samples`, when defined as float, correspond to a fraction of "
                    "the data set, i.e. it must be defined between 0 and 1."
                )
        self.max_samples = max_samples

        self.random_state = random_state
        np.random.seed(random_state)
        self.n_jobs = n_jobs

    def get_max_samples(
            self,
            X: np.ndarray,
        ) -> int:
        """
        Compute the number of samples to use per tree.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        max_samples : int
            Number of samples to draw.
        """

        if isinstance(self.max_samples, int):
            max_samples = self.max_samples
        elif isinstance(self.max_samples, float):
            max_samples = max([1, math.ceil(self.max_samples * len(X))])
        elif self.max_samples is None:
            max_samples = len(X)

        return max_samples

    @abstractmethod
    def train_decision_tree(
            self,
            X: np.ndarray,
            y: np.ndarray,
            metric_data: np.ndarray,
            max_samples: int,
            batch_size: int = 1000,
            tqdm_func: Callable = None,
        ) -> DecisionTree:
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

        pass

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            metric_data: np.ndarray,
            batch_size: int = 1000,
            tqdm_func: Callable | None = None,
        ) -> None:
        """
        Fit the random forest classifier.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        metric_data : np.ndarray
            Additional data used by the metric for evaluation.
        batch_size : int, default=1000
            Batch size for parallel evaluation.
        tqdm_func : callable, optional
            Optional progress bar function.

        Raises
        ------
        ValueError
            If X, y, or metric_data contain NaN values.
        """

        if np.any(np.isnan(X)):
            raise ValueError("Input X contains NaN values.")

        if np.any(np.isnan(y)):
            raise ValueError("Input y contains NaN values.")

        if np.any(np.isnan(metric_data)):
            raise ValueError("Input metric_data contains NaN values.")

        max_samples = self.get_max_samples(X)

        tasks = []
        for _ in range(self.n_estimators):

            task = delayed(self.train_decision_tree)(
                X=X,
                y=y,
                metric_data=metric_data,
                max_samples=max_samples,
                batch_size=batch_size,
                tqdm_func=tqdm_func,
            )

            tasks.append(task)

        parallel = Parallel(
            n_jobs=self.n_jobs,
            return_as="generator_unordered",
            batch_size=batch_size,
            backend="threading",
        )

        if tqdm_func is None:
            items = parallel(tasks)
        else:
            items = tqdm_func(parallel(tasks), desc="Forest Building", total=len(tasks))

        self.forest = []
        for decision_tree in items:
            self.forest.append(decision_tree)

    def predict_x(
            self,
            x: Dict,
        ) -> RandomForestPrediction:
        """
        Predict for a single sample.

        Parameters
        ----------
        x : dict
            A dictionary representing a single data point (feature: value).

        Returns
        -------
        prediction : RandomForestPrediction
            The prediction.
        """

        predictions = []
        for decision_tree in self.forest:
            prediction = decision_tree.predict_x(x)
            predictions.append(prediction)

        prediction = RandomForestPrediction(
            trees_predictions=predictions,
        )

        return prediction

    def predict_all(
            self,
            X: np.ndarray,
        ) -> List[RandomForestPrediction]:
        """
        Predict for multiple samples.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).

        Returns
        -------
        predictions : list of RandomForestPrediction
            List of prediction objects per sample.

        Raises
        ------
        ValueError
            If input X contains NaN values.
        """

        if np.any(np.isnan(X)):
            raise ValueError("Input X contains NaN values.")

        predictions = []
        for i in range(X.shape[0]):
            prediction = self.predict_x(X[i,:])
            predictions.append(prediction)

        return predictions

    def predict(
            self,
            X: np.ndarray,
        ) -> np.ndarray:
        """
        Return predited values for all samples.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        predictions : np.ndarray
            Array of shape (n_samples, n_classes) with predicted values.
        """

        rf_predictions = self.predict_all(X=X)

        predictions = []
        for rf_prediction in rf_predictions:
            x_predictions = np.mean(
                [p.value for p in rf_prediction.trees_predictions],
                axis=0
            )
            predictions.append(x_predictions)

        return np.array(predictions)

    def print_forest(
            self,
            n_estimators: int | None = None,
            max_depth: int = 1000,
            feature_names: List[str] | None = None,
            show_repartition: bool = True,
            show_metadata: bool = False,
            metric_name: str = "metric",
            digits: int = 100,
            digits_metric: int | None = None,
            digits_filter: int | None = None,
            x_to_predict: np.ndarray | None = None,
        ) -> None:
        """
        Print a summary of the forest's decision trees.

        Parameters
        ----------
        n_estimators : int or None, default=None
            Number of trees to print. Defaults to all trees.
        max_depth : int, default=1000
            Maximum depth to display for each tree.
        feature_names : list of str or None, default=None
            Names of the input features.
        show_repartition : bool, default=True
            Whether to show class counts per node.
        show_metadata : bool, default=False
            Whether to display metadata in tree nodes.
        metric_name : str, default="metric"
            Name of the metric used in the splits.
        digits : int, default=100
            Number of decimal places to show.
        digits_metric : int or None, default=None
            Decimal precision for metric values.
        digits_filter : int or None, default=None
            Decimal precision for filter thresholds.
        x_to_predict : np.ndarray or None, default=None
            Optional sample to visualize its path in each tree.
        """

        if n_estimators is None:
            n_estimators = self.n_estimators

        for id, decision_tree in enumerate(self.forest[:n_estimators]):

            print(f"TREE {id + 1}:")

            decision_tree.print_tree(
                max_depth=max_depth,
                feature_names=feature_names,
                show_repartition=show_repartition,
                show_metadata=show_metadata,
                metric_name=metric_name,
                digits=digits,
                digits_metric=digits_metric,
                digits_filter=digits_filter,
                x_to_predict=x_to_predict,
            )

            print("")

    def plot_forest(
            self,
            n_estimators: int | None = None,
            max_depth: int = 1000,
            feature_names: List[str] | None = None,
            show_repartition: bool = True,
            show_metadata: bool = False,
            metric_name: str = "metric",
            digits: int = 100,
            digits_metric: int | None = None,
            digits_filter: int | None = None,
            x_to_predict: np.ndarray | None = None,
        ) -> None:
        """
        Plot graphical representations of the decision trees in the forest.

        Parameters
        ----------
        n_estimators : int or None, default=None
            Number of trees to plot. Defaults to all trees.
        max_depth : int, default=1000
            Maximum depth to visualize per tree.
        feature_names : list of str or None, default=None
            Feature names for plotting.
        show_repartition : bool, default=True
            Whether to show class counts at each node.
        show_metadata : bool, default=False
            Whether to show metadata at nodes.
        metric_name : str, default="metric"
            Name of the metric used for splits.
        digits : int, default=100
            Decimal precision for displayed values.
        digits_metric : int or None, default=None
            Precision for metric values.
        digits_filter : int or None, default=None
            Precision for thresholds.
        x_to_predict : np.ndarray or None, default=None
            Optional sample to visualize path in trees.
        """

        if n_estimators is None:
            n_estimators = self.n_estimators

        for id, decision_tree in enumerate(self.forest[:n_estimators]):

            decision_tree.plot_tree(
                max_depth=max_depth,
                feature_names=feature_names,
                show_repartition=show_repartition,
                show_metadata=show_metadata,
                metric_name=metric_name,
                digits=digits,
                digits_metric=digits_metric,
                digits_filter=digits_filter,
                x_to_predict=x_to_predict,
                title=f"Decision Tree {id + 1}",
            )
