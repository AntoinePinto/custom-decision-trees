import math
import random
from itertools import combinations
from typing import Callable, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from custom_decision_trees.metrics import Gini, MetricBase
from custom_decision_trees.utils import (
    bold,
    get_node_coordinates,
    get_random_cardinalities,
)

from .schemas import Cut, DecisionTreePrediction, Partition, Split, SplitEvaluation


class DecisionTree:
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
        random_state : int or None, default=None
            Seed for reproducibility.
        n_jobs : int, default=1
            Number of parallel jobs to run.
        """

        if metric is None:
            metric = Gini()
        self.metric = metric

        self.max_depth = max_depth

        if isinstance(min_samples_split, float):
            if (min_samples_split <= 0) | (min_samples_split >= 1):
                raise ValueError(
                    "`min_samples_split`, when defined as float, correspond to a"
                    " fraction of the data set, i.e. it must be defined between"
                    " 0 and 1."
                )
        self.min_samples_split = min_samples_split

        if isinstance(min_samples_leaf, float):
            if (min_samples_leaf <= 0) | (min_samples_leaf >= 1):
                raise ValueError(
                    "`min_samples_leaf`, when defined as float, correspond to a"
                    " fraction of the data set, i.e. it must be defined between 0"
                    " and 1."
                )
        self.min_samples_leaf = min_samples_leaf

        if isinstance(max_features, float):
            if (max_features <= 0) | (max_features >= 1):
                raise ValueError(
                    "`max_features`, when defined as float, correspond to a fraction of"
                    "the features, i.e. it must be defined between 0 and 1."
                )
        self.max_features = max_features

        self.nb_max_conditions_per_node = nb_max_conditions_per_node
        self.nb_max_cut_options_per_var = nb_max_cut_options_per_var
        self.nb_max_split_options_per_node = nb_max_split_options_per_node
        self.random_state = random_state
        np.random.seed(random_state)
        self.n_jobs = n_jobs

    def get_var_cut_options(
            self,
            var: np.ndarray,
            col_type: type,
        ) -> np.ndarray:
        """
        Compute candidate cut values for a variable.

        Parameters
        ----------
        var : np.ndarray
            The variable values for which to compute cut options.

        Returns
        -------
        values : np.ndarray
            The candidate cut values.
        """

        nb_max = self.nb_max_cut_options_per_var

        values = np.sort(np.unique(var))
        if col_type in [int, float, bool]:
            values = values[:-1]
            if len(values) > nb_max:
                values = np.quantile(values, [i/nb_max for i in range(nb_max)])
        elif col_type in [str]:
            if len(values) > nb_max:
                values = np.random.choice(np.unique(var), nb_max, replace=False)
        else:
            raise Exception(f"Unsupported column type: {col_type}")

        return values

    def get_available_cuts(
            self,
            X: np.ndarray,
            min_samples_leaf: int,
            max_features: int,
        ) -> Tuple[List[Cut], np.ndarray]:
        """
        Generate all valid cuts based on the data and constraints.

        Parameters
        ----------
        X : np.ndarray
            The input features at the node.
        min_samples_leaf : int
            Minimum required samples in a leaf.
        max_features : int
            Number of features to consider for splitting.

        Returns
        -------
        available_cuts : list of Cut
            The list of valid cuts.
        mask_matrix : dict
            A dictionary mapping cut IDs to boolean masks.
        """

        features_sample = np.random.choice(
            range(X.shape[1]),
            size=max_features,
            replace=False,
        )

        i = 0
        available_cuts: List[Cut] = []
        mask_list = []
        for id in features_sample:

            col_types = set(type(x) for x in X[:, id].tolist())
            if len(col_types) > 1:
                raise Exception(
                    "The type of a feature's values must be unique. "
                    "Feature {id} has mixed types: {col_types}"
                )

            col_type = list(col_types)[0]

            var_values = self.get_var_cut_options(
                var=X[:, id],
                col_type=col_type,
            )

            for value in var_values:

                if col_type in [int, float, bool]:
                    left_mask = X[:, id] <= value
                    side1_sign, side2_sign = "<=", ">"
                    value = float(value)
                elif col_type in [str]:
                    left_mask = X[:, id] == value
                    side1_sign, side2_sign = "==", "!="

                nb_obs_left = int(np.sum(left_mask))
                if nb_obs_left >= min_samples_leaf:

                    cut = Cut(
                        id=i,
                        var_id=int(id),
                        sign=side1_sign,
                        value=value,
                        nb_obs=nb_obs_left,
                    )

                    available_cuts.append(cut)
                    mask_list.append(left_mask)
                    i += 1

                nb_obs_right = int(np.sum(~left_mask))
                if nb_obs_right >= min_samples_leaf:

                    cut = Cut(
                        id=i,
                        var_id=int(id),
                        sign=side2_sign,
                        value=value,
                        nb_obs=nb_obs_right,
                    )

                    available_cuts.append(cut)
                    mask_list.append(~left_mask)
                    i += 1

        mask_matrix = np.stack(mask_list, axis=0)

        return available_cuts, mask_matrix

    def get_nb_max_combis(
            self,
            nb_available_cuts: int,
        ):

        nb_split_options = 0
        for k in range(1, self.nb_max_conditions_per_node + 1):
            nb_split_options += math.comb(nb_available_cuts, k)

        return nb_split_options

    def get_split_options(
            self,
            available_cuts: List[Cut],
        ) -> List[Split]:
        """
        Generate split combinations from available cuts.

        Parameters
        ----------
        available_cuts : list of Cut
            The available cut conditions.

        Returns
        -------
        split_options : list of Split
            The list of candidate split combinations.
        """

        max_size = self.nb_max_split_options_per_node

        if max_size is not None:
            if max_size > self.get_nb_max_combis(len(available_cuts)):
                max_size = None

        available_cuts_indices = list(range(len(available_cuts)))

        cut_combis = []
        if max_size is None:
            for k in range(1, self.nb_max_conditions_per_node + 1):
                cut_combis.extend(list(combinations(available_cuts_indices, k)))
        else:

            cardinalities = get_random_cardinalities(
                nb_options=len(available_cuts),
                max_cardinality=self.nb_max_conditions_per_node,
                size=max_size
            )

            for k in cardinalities:
                cut_combis.append(random.sample(available_cuts_indices, k=k))

            cut_combis = list(map(list, dict.fromkeys(tuple(i) for i in cut_combis)))

        split_options: List[Split] = []
        for cuts_indices in cut_combis:

            split = Split(
                cuts=[available_cuts[i] for i in cuts_indices],
                operator="&",
            )

            split_options.append(split)

        return split_options

    def get_split_evaluation(
            self,
            metric_data: np.ndarray,
            split: Split,
            mask_matrix: Dict[int, np.ndarray],
            min_samples_leaf: int,
        ) -> SplitEvaluation | None:
        """
        Evaluate a split option using the provided metric.

        Parameters
        ----------
        metric_data : np.ndarray
            Data used for metric computation.
        split : Split
            The split to evaluate.
        mask_matrix : dict
            A dictionary of masks for each cut.
        min_samples_leaf : int
            Minimum required samples in a leaf.

        Returns
        -------
        evaluation : SplitEvaluation or None
            The split evaluation or None if invalid.
        """

        mask = split.get_mask(mask_matrix=mask_matrix)

        nb_obs = int(np.sum(mask))

        if nb_obs < min_samples_leaf:
            return None

        if len(split.cuts) > 1:
            if any([nb_obs == cut.nb_obs for cut in split.cuts]):
                return None

        metric, metadata = self.metric.compute_metric(
            metric_data=metric_data,
            mask=mask,
        )

        split_evaluation = SplitEvaluation(
            split=split,
            nb_obs=nb_obs,
            metric=metric,
            metadata=metadata,
        )

        return split_evaluation

    def get_split_evaluations(
            self,
            split_options: List[Split],
            mask_matrix: np.ndarray,
            metric_data: np.ndarray,
            min_samples_leaf: int,
            batch_size: int = 1000,
            tqdm_func: Callable | None = None,
        ) -> List[SplitEvaluation]:
        """
        Evaluate multiple splits in parallel.

        Parameters
        ----------
        split_options : list of Split
            The candidate splits.
        mask_matrix : dict
            Masks for each cut.
        metric_data : np.ndarray
            Data used for metric computation.
        min_samples_leaf : int
            Minimum required samples in a leaf.
        batch_size : int, default=1000
            Number of splits to evaluate per batch.
        tqdm_func : callable or None, default=None
            Optional progress bar function.

        Returns
        -------
        split_evaluations : list of SplitEvaluation
            Evaluated splits.
        """

        tasks = []
        for split in split_options:

            task = delayed(self.get_split_evaluation)(
                metric_data=metric_data,
                split=split,
                mask_matrix=mask_matrix,
                min_samples_leaf=min_samples_leaf,
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
            items = tqdm_func(
                parallel(tasks),
                desc="Split Evalutation",
                total=len(tasks)
            )

        split_evaluations = []
        for future in items:
            if future is not None:
                split_evaluations.append(future)

        return split_evaluations

    def get_repartition(
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

    def get_min_samples(
            self,
            param: int | float,
            X: np.ndarray,
        ) -> int:
        """
        Convert min_samples parameter to integer value.

        Parameters
        ----------
        param : int or float
            Minimum sample parameter.

        X : np.ndarray
            The data.

        Returns
        -------
        min_samples : int
            Computed minimum sample count.
        """

        if isinstance(param, int):
            return param
        else:
            return max([1, math.ceil(param * len(X))])

    def get_max_features(
            self,
            X: np.ndarray,
        ) -> int:
        """
        Determine number of features to consider for splitting.

        Parameters
        ----------
        X : np.ndarray
            The input features.

        Returns
        -------
        max_features : int
            Number of features to consider.
        """

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return int(np.sqrt(X.shape[1]))
            else:
                return int(np.log2(X.shape[1]))
        elif self.max_features is None:
            return X.shape[1]
        elif isinstance(self.max_features, float):
            return max([1, math.ceil(self.max_features * len(X.shape[1]))])
        else:
            return self.max_features

    def init_partitions(
            self,
            X: np.ndarray,
            y: np.ndarray,
            metric_data: np.ndarray,
        ) -> Dict[int, Partition]:
        """
        Initialize the root partition of the tree.

        Parameters
        ----------
        X : np.ndarray
            Features.

        y : np.ndarray
            Target values.

        metric_data : np.ndarray
            Data for metric computation.

        Returns
        -------
        partitions : dict
            Initial partition.
        """

        mask = np.repeat(True, len(X))
        metric, metadata = self.metric.compute_metric(
            metric_data=metric_data,
            mask=mask,
        )

        partition = Partition(
            id=0,
            depth=0,
            mask=mask,
            repartition=self.get_repartition(y),
            metric=metric,
            metadata=metadata,
            historic_splits=[],
        )

        return {0: partition}

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            metric_data: np.ndarray,
            classes: np.ndarray | None = None,
            batch_size: int = 1000,
            tqdm_func: Callable | None = None,
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

        min_samples_split = self.get_min_samples(param=self.min_samples_split, X=X)
        min_samples_leaf = self.get_min_samples(param=self.min_samples_leaf, X=X)
        max_features = self.get_max_features(X=X)

        self.partitions = self.init_partitions(
            X=X,
            y=y,
            metric_data=metric_data,
        )

        if tqdm_func is None:
            items = range(self.max_depth)
        else:
            items = tqdm_func(range(self.max_depth), desc="Depth")

        for depth in items:
            for partition_id, partition in dict(self.partitions).items():

                # Skip iteration if the partition does not concerns the active depth
                if partition.depth != depth:
                    continue

                # Skip iteration if there are fewer than 2 observations to split
                if np.sum(partition.mask) < min_samples_split:
                    continue

                available_cuts, mask_matrix = self.get_available_cuts(
                    X=X[partition.mask, :],
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                )

                split_options = self.get_split_options(available_cuts)
                partition_metric_data = metric_data[partition.mask]

                split_evaluations = self.get_split_evaluations(
                    split_options=split_options,
                    mask_matrix=mask_matrix,
                    metric_data=partition_metric_data,
                    min_samples_leaf=min_samples_leaf,
                    batch_size=batch_size,
                    tqdm_func=tqdm_func,
                )

                # Skip iteration if no valid cuts has been found
                if len(split_evaluations) == 0:
                    continue

                partition.split_evaluations = sorted(
                    split_evaluations,
                    key=lambda e: e.metric,
                    reverse=True
                )

                partition.best_split = partition.split_evaluations[0].split
                partition.type = "branch"

                mask = partition.best_split.get_mask(mask_matrix=mask_matrix)
                mask_side1, mask_side2 = partition.get_mask_sides(mask=mask)

                if sum(mask_side1) >= min_samples_leaf:

                    split_side1 = partition.best_split

                    metric, metadata = self.metric.compute_metric(
                        metric_data=partition_metric_data,
                        mask=mask,
                    )

                    self.partitions[partition_id * 2 + 1] = Partition(
                        id=partition_id * 2 + 1,
                        depth=partition.depth + 1,
                        mask=mask_side1,
                        repartition=self.get_repartition(y[mask_side1]),
                        metric=metric,
                        metadata=metadata,
                        historic_splits=partition.historic_splits + [split_side1],
                    )

                if sum(mask_side2) >= min_samples_leaf:

                    split_side2 = partition.best_split.get_inverse_split()

                    metric, metadata = self.metric.compute_metric(
                        metric_data=partition_metric_data,
                        mask=~mask,
                    )

                    self.partitions[partition_id * 2 + 2] = Partition(
                        id=partition_id * 2 + 2,
                        depth=partition.depth + 1,
                        mask=mask_side2,
                        repartition=self.get_repartition(y[mask_side2]),
                        metric=metric,
                        metadata=metadata,
                        historic_splits=partition.historic_splits + [split_side2],
                    )

    def predict_x(
            self,
            x: np.ndarray,
        ) -> DecisionTreePrediction:
        """
        Predict the class probabilities for a single sample.

        Parameters
        ----------
        x : np.ndarray
            A single sample represented as a dictionary of feature values.

        Returns
        -------
        list
            The predicted probabilities for each class.
        """

        partition_id = 0
        while True:

            partition = self.partitions[partition_id]

            if partition.type == "leaf":
                return partition.prediction
            elif partition.best_split.check_x(x):
                partition_id = partition_id * 2 + 1
            elif (partition_id * 2 + 2) in self.partitions:
                partition_id = partition_id * 2 + 2
            else:
                return partition.prediction

    def predict(
            self,
            X: np.ndarray,
        ) -> List[DecisionTreePrediction]:
        """
        Predict class probabilities for a set of samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        list of DecisionTreePrediction
            List of prediction objects containing probabilities and metrics.
        """

        if np.any(np.isnan(X)):
            raise ValueError("Input X contains NaN values.")

        predictions = []
        for i in range(len(X)):
            prediction = self.predict_x(x=X[i,:])
            predictions.append(prediction)

        return predictions

    def predict_probas(
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

        predictions = self.predict(X=X)
        probas = np.array([p.probas for p in predictions])

        return probas

    def predict_metrics(
            self,
            X: np.ndarray,
        ) -> np.ndarray:
        """
        Predict the splitting metric values for the provided samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Array of metric values corresponding to the predictions.
        """

        predictions = self.predict(X=X)
        metrics = np.array([p.metric for p in predictions])

        return metrics

    def next_uncle(
            self,
            parents: List[int],
        ) -> int:
        """
        Find the next uncle partition for a given list of parent IDs.

        Parameters
        ----------
        parents : list
            The list of parent IDs.

        Returns
        -------
        int or None
            The next uncle partition ID, or None if none exists.
        """

        id = next(
            (
                id + 1
                for id in parents
                if (id + 1 in self.partitions) and (id % 2 == 1)
            )
        )

        return id

    def get_next_id(
            self,
            partition_id: int,
        ) -> int:
        """
        Get the ID of the next partition during tree traversal.

        Parameters
        ----------
        partition_id : int
            Current partition ID.

        Returns
        -------
        next_id : int
            Next partition ID.
        """

        partition = self.partitions[partition_id]

        if partition.type == "branch":
            partition_id = partition_id * 2 + 1
        elif partition_id % 2 == 1:
            partition_id = partition_id + 1
        else:
            partition_id = self.next_uncle(partition.parents)

        if partition_id not in self.partitions:
            partition_id = self.next_uncle(partition.parents)

        return partition_id

    def print_tree(
            self,
            max_depth: int = 1000,
            feature_names: List[str] | None = None,
            show_repartition: bool = True,
            show_metadata: bool = False,
            metric_name: str = "metric",
            digits: int = 5,
            digits_metric: int | None = None,
            digits_filter: int | None = None,
            x_to_predict: np.ndarray | None = None,
        ) -> None:
        """
        Print a textual representation of the decision tree.

        Parameters
        ----------
        max_depth : int, default=1000
            Maximum depth of the tree to display.
        feature_names : list of str or None, default=None
            Names of the features.
        show_repartition : bool, default=True
            Whether to display class repartition.
        show_metadata : bool, default=False
            Whether to display additional metadata.
        metric_name : str, default='metric'
            Name of the splitting metric.
        digits : int, default=5
            Number of decimal places to display.
        digits_metric : int or None, default=None
            Decimal places for metric values.
        digits_filter : int or None, default=None
            Decimal places for filter conditions.
        x_to_predict : np.ndarray or None, default=None
            Optional sample to highlight its path.
        """

        if digits_metric is None:
            digits_metric = digits

        if digits_filter is None:
            digits_filter = digits

        if x_to_predict is not None:
            x_prediction = self.predict_x(x=x_to_predict)

        print(f"classes: {self.classes}")

        partition_id = 0
        while True:

            partition = self.partitions[partition_id]

            if partition.depth <= max_depth:

                label = f"{'|   ' * partition.depth}[{partition_id}] "

                if partition_id > 0:

                    filter = partition.historic_splits[-1].get_filter(
                        feature_names=feature_names,
                        digits=digits_filter,
                        logical_operator=True,
                    )

                    label += f"{filter} | "

                label += "{nobs} obs -> {metric_name} = {metric_value}".format(
                    nobs=partition.nb_obs,
                    metric_name=metric_name,
                    metric_value=round(partition.metric, digits),
                )

                if show_repartition is True:
                    label += f" | repartition = {partition.repartition}"

                if show_metadata is True:
                    label += f" | {partition.metadata}"

                if x_to_predict is not None:
                    if partition_id in x_prediction.path:
                        label = bold(label)

                print(label)

            if partition_id == max(self.partitions):
                break

            partition_id = self.get_next_id(partition_id)

        if x_to_predict is not None:
            prediction_str = x_prediction.to_str(digits=digits)
            print(bold(prediction_str))

    def plot_tree(
            self,
            max_depth: int = 1000,
            feature_names: List[str] | None = None,
            show_repartition: bool = True,
            show_metadata: bool = False,
            metric_name: str = "metric",
            digits: int = 3,
            digits_metric: int | None = None,
            digits_filter: int | None = None,
            x_to_predict: np.ndarray | None = None,
            title: str = "Decision Tree",
            figsize: Tuple = (16, 8),
        ) -> None:
        """
        Plot a graphical representation of the decision tree.

        Parameters
        ----------
        max_depth : int, default=1000
            Maximum depth of the tree to display.
        feature_names : list of str or None, default=None
            Names of the features.
        show_repartition : bool, default=True
            Whether to display class repartition.
        show_metadata : bool, default=False
            Whether to display additional metadata.
        metric_name : str, default='metric'
            Name of the splitting metric.
        digits : int, default=3
            Number of decimal places for numeric display.
        digits_metric : int or None, default=None
            Decimal places for metric values.
        digits_filter : int or None, default=None
            Decimal places for filter conditions.
        x_to_predict : np.ndarray or None, default=None
            Optional sample to highlight its path.
        title : str, default='Decision Tree'
            Title of the plot.
        figsize : tuple, default=(16, 8)
            Size of the figure.
        """

        if digits_metric is None:
            digits_metric = digits

        if digits_filter is None:
            digits_filter = digits

        if x_to_predict is not None:
            x_prediction = self.predict_x(x=x_to_predict)

        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.axis("off")

        max_depth = min([self.partitions[max(self.partitions)].depth, max_depth])

        for partition_id, partition in self.partitions.items():

            label = f"{metric_name} = {partition.metric:.{digits_metric}g}"

            if show_repartition:
                label += f"\n{partition.nb_obs} obs {partition.repartition}"

            if show_metadata:
                label += "\n\nMETADATA:\n"

                parts = []
                for k, v in partition.metadata.items():
                    parts.append(f"{k}: {v:.{digits}g}")

                label += "\n ".join(parts)

            x, y = get_node_coordinates(id=partition_id, max_depth=max_depth)

            color, lw = "#555", 1.5
            if x_to_predict is not None:
                if partition_id in x_prediction.path:
                    color, lw = "#000000", 3

            ax.text(
                x=x,
                y=y,
                s=label,
                ha="center",
                va="top",
                fontsize=10,
                bbox={
                    "fc": "#e0e0e0",
                    "ec": color,
                    "lw": lw,
                },
            )

            if partition.depth > 0:

                parent_id = int((partition_id - 1 - (partition_id % 2 == 0)) / 2)

                x_parent, y_parent = get_node_coordinates(
                    id=parent_id,
                    max_depth=max_depth
                )

                filter = partition.historic_splits[-1].get_filter(
                    feature_names=feature_names,
                    digits=digits_filter,
                    logical_operator=True,
                )

                filter = filter.replace(" AND ", "\nAND ").replace(" OR ", "\nOR ")

                ax.text(
                    x=x,
                    y=y + 0.02,
                    s=filter,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#000000",
                    bbox={
                        "fc": "white",
                    },
                )

                ax.plot(
                    [x_parent, x],
                    [y_parent, y],
                    color=color,
                    lw=lw,
                )

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

        plt.title(f"{title} (classes: {self.classes})", fontsize=14, loc="left")
        plt.tight_layout()
        plt.show()
