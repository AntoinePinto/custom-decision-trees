import logging
import math
import random
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Callable, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from custom_decision_trees.metrics import MetricBase
from custom_decision_trees.utils import (
    bold,
    get_node_coordinates,
)

from .schemas import Condition, DecisionTreePrediction, Partition, Split, SplitEvaluation


class DecisionTree(ABC):
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
            max_thresholds_per_feature: int = 10,
            max_conditions_per_split: int = 1,
            max_splits_evaluated_per_node: int | None = None,
            splits_search_budget_factor: int = 100,
            random_state: int | None = None,
            n_jobs: int = 1,
        ) -> None:
        """
        Custom Decision Tree classifier with support for different splitting metrics.


        Parameters
        ----------
        metric : MetricBase or None, default=None
            Criterion used to evaluate the quality of a split (e.g. Gini impurity,
            entropy, variance). If None, the child class is expected to define a
            default metric. The metric must expose a `compute_metric(metric_data, mask)`
            method returning a scalar score (higher = better split) along with an
            optional metadata dictionary.

        max_depth : int, default=5
            Maximum depth of the tree. At each depth level, every active partition
            can be subdivided into two new child partitions. Increasing this value
            raises the model's capacity but increases the risk of overfitting.

        min_samples_split : int or float, default=2
            Minimum number of samples required in a partition for a split to be
            attempted. If int, refers to an absolute sample count. If float (strictly
            between 0 and 1), refers to a fraction of the total dataset size (rounded
            up, with a minimum of 1). This parameter prevents splitting nodes that are
            already too small, before even evaluating any candidate split.

        min_samples_leaf : int or float, default=2
            Minimum number of samples required in each of the two child partitions
            resulting from a split for that split to be considered valid. Follows the
            same int/float logic as `min_samples_split`. This parameter is more
            restrictive than `min_samples_split`: it filters out splits that would
            produce overly small leaves, regardless of the parent node size.

        max_features : {"sqrt", "log2"} or int or float or None, default="sqrt"
            Number of candidate features to consider when searching for the best split
            at each node. A random subset is drawn at every node, introducing diversity
            across nodes — particularly useful when used within a random forest.
            - "sqrt"  : sqrt(n_features), commonly recommended for classification.
            - "log2"  : log2(n_features).
            - int     : absolute number of features to consider.
            - float   : fraction of features (strictly between 0 and 1, rounded up).
            - None    : all features are considered (no subsampling).

        max_thresholds_per_feature : int, default=10
            Maximum number of threshold values to evaluate per feature when building
            split conditions. For numerical features, thresholds are evenly spaced
            quantiles of the unique observed values. For categorical features, they
            are drawn at random from the unique values. Lowering this value speeds up
            training at the cost of a less exhaustive search over optimal thresholds.

        max_conditions_per_split : int, default=1
            Maximum number of conditions that can compose a single split. With the
            default value of 1, each node applies a single rule (e.g. `x1 <= 3.5`),
            which corresponds to the behaviour of a standard decision tree. Increasing
            this value allows compound splits combining multiple conditions via a
            logical `&` operator (e.g. `x1 <= 3.5 AND x2 > 1.0`), enabling more
            complex decision boundaries. Note that the number of candidate combinations
            grows exponentially with this parameter.

        max_splits_evaluated_per_node : int or None, default=None
            Maximum number of candidate splits to evaluate per node. If None, all
            valid splits (i.e. all combinations of conditions up to
            `max_conditions_per_split`) are evaluated exhaustively, which can become
            computationally prohibitive when `max_conditions_per_split` > 1. If an
            integer is provided, splits are sampled randomly until this quota is
            reached or the search budget defined by `splits_search_budget_factor` is
            exhausted.

        splits_search_budget_factor : int, default=100
            Multiplicative factor controlling the random search budget, active only
            when `max_splits_evaluated_per_node` is set. The maximum number of random
            draws attempted is `max_splits_evaluated_per_node *
            splits_search_budget_factor`. If this budget is exhausted before enough
            valid splits are found, training continues with however many splits were
            collected. Increasing this factor improves the chances of finding
            high-quality splits at the cost of a longer search time.

        random_state : int or None, default=None
            Seed for NumPy's random number generator, ensuring reproducibility of
            results across runs. Affects feature subsampling, random threshold
            selection for categorical features, and random split sampling when
            `max_splits_evaluated_per_node` is set.

        n_jobs : int, default=1
            Number of parallel workers used when evaluating candidate splits (via
            joblib). Set to -1 to use all available CPU cores. Most beneficial when
            the number of candidate splits is large, i.e. on big datasets or when
            `max_conditions_per_split` > 1.
        """

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

        self.max_conditions_per_split = max_conditions_per_split
        self.max_thresholds_per_feature = max_thresholds_per_feature
    
        self.max_splits_evaluated_per_node = max_splits_evaluated_per_node
        if max_splits_evaluated_per_node is not None:
            self.max_split_options_search = max_splits_evaluated_per_node * splits_search_budget_factor
    
        self.random_state = random_state
        np.random.seed(random_state)
        self.n_jobs = n_jobs

    def set_logger(
            self,
            verbose: bool = False,
        ) -> None:
        """
        Set up the logger for the decision tree.

        Parameters
        ----------
        verbose : bool, default=False
            Whether to enable verbose logging.
        """

        self.verbose = verbose

        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(handler)

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    def get_var_condition_options(
            self,
            var: np.ndarray,
            col_type: type,
        ) -> np.ndarray:
        """
        Compute candidate condition values for a variable.

        Parameters
        ----------
        var : np.ndarray
            The variable values for which to compute condition options.

        Returns
        -------
        values : np.ndarray
            The candidate condition values.
        """

        max_options = self.max_thresholds_per_feature

        values = np.sort(np.unique(var))
        if col_type in [int, float, bool]:
            values = values[:-1]
            if len(values) > max_options:
                values = np.quantile(values, [i/max_options for i in range(max_options)])
        elif col_type in [str]:
            if len(values) > max_options:
                values = np.random.choice(np.unique(var), max_options, replace=False)
        else:
            raise Exception(f"Unsupported column type: {col_type}")

        return values

    def get_available_conditions(
            self,
            X: np.ndarray,
            min_samples_leaf: int,
            max_features: int,
            tqdm_func: Callable = tqdm,
        ) -> Tuple[List[Condition], np.ndarray]:
        """
        Generate all valid conditions based on the data and constraints.

        Parameters
        ----------
        X : np.ndarray
            The input features at the node.
        min_samples_leaf : int
            Minimum required samples in a leaf.
        max_features : int
            Number of features to consider for splitting.
        tqdm_func : callable, default=tqdm
            Progress bar function.

        Returns
        -------
        available_conditions : list of Condition
            The list of valid conditions.
        mask_matrix : dict
            A dictionary mapping condition IDs to boolean masks.
        """

        features_sample = np.random.choice(
            range(X.shape[1]),
            size=max_features,
            replace=False,
        )

        i = 0
        available_conditions: List[Condition] = []
        mask_list = []
        for id in tqdm_func(
            features_sample,
            desc="Finding Conditions - Looping over features",
            disable=not self.verbose,
        ):

            col_types = set(type(x) for x in X[:, id].tolist())
            if len(col_types) > 1:
                raise Exception(
                    "The type of a feature's values must be unique. "
                    "Feature {id} has mixed types: {col_types}"
                )

            col_type = list(col_types)[0]

            var_values = self.get_var_condition_options(
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

                    condition = Condition(
                        id=i,
                        var_id=int(id),
                        sign=side1_sign,
                        value=value,
                        nb_obs=nb_obs_left,
                    )

                    available_conditions.append(condition)
                    mask_list.append(left_mask)
                    i += 1

                nb_obs_right = int(np.sum(~left_mask))
                if nb_obs_right >= min_samples_leaf:

                    condition = Condition(
                        id=i,
                        var_id=int(id),
                        sign=side2_sign,
                        value=value,
                        nb_obs=nb_obs_right,
                    )

                    available_conditions.append(condition)
                    mask_list.append(~left_mask)
                    i += 1

        mask_matrix = np.stack(mask_list, axis=0)

        return available_conditions, mask_matrix

    def get_nb_max_combis(
            self,
            nb_available_conditions: int,
        ):
        """
        Compute the maximum number of possible split combinations given
        the available condition options and the maximum number of conditions
        allowed per node.

        Parameters
        ----------
        nb_available_conditions : int
            The number of available condition options for splitting.

        Returns
        -------
        int
            The total number of possible split combinations.
        """

        nb_split_options = 0
        for k in range(1, self.max_conditions_per_split + 1):
            nb_split_options += math.comb(nb_available_conditions, k)

        return nb_split_options

    def is_valid_split(
        self,
        split,
        mask_matrix,
        min_samples_leaf: int,
    ) -> bool:

        mask = split.get_mask(mask_matrix=mask_matrix)

        nb_obs = int(np.sum(mask))

        if nb_obs < min_samples_leaf:
            return False

        if len(split.conditions) > 1:
            if nb_obs in [condition.nb_obs for condition in split.conditions]:
                return False

        return True

    def get_split_options(
            self,
            X: np.ndarray,
            min_samples_leaf: int,
            max_features: int,
            tqdm_func: Callable = tqdm,
        ) -> Tuple[List[Split], np.ndarray]:
        """
        Generate split combinations from available conditions.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        min_samples_leaf : int
            Minimum required samples in a leaf.
        max_features : int
            Number of features to consider for splitting.
        tqdm_func : callable, default=tqdm
            Progress bar function.

        Returns
        -------
        split_options : list of Split
            The list of candidate split combinations.
        mask_matrix : dict
            A dictionary mapping condition IDs to boolean masks.
        """

        max_size = self.max_splits_evaluated_per_node

        available_conditions, mask_matrix = self.get_available_conditions(
            X=X,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            tqdm_func=tqdm_func,
        )

        if max_size is not None:
            if max_size > self.get_nb_max_combis(len(available_conditions)):
                max_size = None

        available_conditions_indices = list(range(len(available_conditions)))
        self.logger.info("Generating split options...")
        split_options: List[Split] = []
        if max_size is None:
            for k in range(1, self.max_conditions_per_split + 1):
                condition_combis = list(combinations(available_conditions_indices, k))
                for conditions_indices in condition_combis:

                    split = Split(
                        conditions=[available_conditions[i] for i in conditions_indices],
                        operator="&",
                    )

                    if self.is_valid_split(split, mask_matrix, min_samples_leaf):
                        split_options.append(split)
        else:

            k_options = list(range(1, self.max_conditions_per_split + 1))
            weights = [math.comb(len(available_conditions), k) for k in k_options]

            pbar = tqdm_func(
                desc="Finding valid splits",
                total=max_size,
                disable=not self.verbose,
            )
            
            nb_searchs = 0

            while len(split_options) < max_size:

                k = random.choices(k_options, weights=weights, k=1)[0] # noqa: S311
                conditions_indices = random.sample(available_conditions_indices, k=k)

                split = Split(
                    conditions=[available_conditions[i] for i in conditions_indices],
                    operator="&",
                )

                if self.is_valid_split(split, mask_matrix, min_samples_leaf):
                    split_options.append(split)
                    pbar.update(1)

                nb_searchs += 1
                if nb_searchs >= self.max_split_options_search:
                    self.logger.info(
                        f"Maximum number of {nb_searchs} searchs reached while"
                        f" looking for valid splits. Found {len(split_options)}"
                        f" valid splits out of {max_size}(`max_splits_evaluated_per_node`) required."
                        f" Consider increasing `splits_search_budget_factor`"
                    )
                    break

            pbar.close()

        return split_options, mask_matrix

    def get_split_evaluation(
            self,
            metric_data: np.ndarray,
            split: Split,
            mask: np.ndarray,
        ) -> SplitEvaluation | None:
        """
        Evaluate a split option using the provided metric.

        Parameters
        ----------
        metric_data : np.ndarray
            Data used for metric computation.
        split : Split
            The split to evaluate.
        mask : np.ndarray
            The mask for the split.

        Returns
        -------
        evaluation : SplitEvaluation or None
            The split evaluation or None if invalid.
        """


        metric, metadata = self.metric.compute_metric(
            metric_data=metric_data,
            mask=mask,
        )

        split_evaluation = SplitEvaluation(
            split=split,
            nb_obs=int(np.sum(mask)),
            metric=metric,
            metadata=metadata,
        )

        return split_evaluation

    def get_split_evaluations(
            self,
            split_options: List[Split],
            mask_matrix: np.ndarray,
            metric_data: np.ndarray,
            batch_size: int = 1000,
            tqdm_func: Callable = tqdm,
        ) -> List[SplitEvaluation]:
        """
        Evaluate multiple splits in parallel.

        Parameters
        ----------
        split_options : list of Split
            The candidate splits.
        mask_matrix : dict
            Masks for each condition.
        metric_data : np.ndarray
            Data used for metric computation.
        min_samples_leaf : int
            Minimum required samples in a leaf.
        batch_size : int, default=1000
            Number of splits to evaluate per batch.
        tqdm_func : callable, default=tqdm
            Progress bar function.

        Returns
        -------
        split_evaluations : list of SplitEvaluation
            Evaluated splits.
        """

        tasks = []
        for split in split_options:

            mask = split.get_mask(mask_matrix=mask_matrix)

            task = delayed(self.get_split_evaluation)(
                metric_data=metric_data,
                split=split,
                mask=mask,
            )

            tasks.append(task)

        parallel = Parallel(
            n_jobs=self.n_jobs,
            return_as="generator_unordered",
            batch_size=batch_size,
            backend="loky",
        )

        split_evaluations = []
        for future in tqdm_func(
            parallel(tasks),
            desc="Split Evalutation",
            total=len(tasks),
            disable=not self.verbose,
        ):
            split_evaluations.append(future)

        return split_evaluations

    @abstractmethod
    def get_partition_value(
            self,
            y: np.ndarray,
        ) -> List[int] | float:
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

        pass  # This method should be defined in child class

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
            param: int | float | str | None,
            X: np.ndarray,
        ) -> int:
        """
        Determine number of features to consider for splitting.

        Parameters
        ----------
        param : int, float, str, or None
            The max_features parameter.
        X : np.ndarray
            The input features.

        Returns
        -------
        param : int
            Number of features to consider.
        """

        if isinstance(param, str):
            if param == "sqrt":
                return int(np.sqrt(X.shape[1]))
            else:
                return int(np.log2(X.shape[1]))
        elif param is None:
            return X.shape[1]
        elif isinstance(param, float):
            return max([1, math.ceil(param * X.shape[1])])
        else:
            return param

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
            value=self.get_partition_value(y),
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
            batch_size: int = 1000,
            tqdm_func: Callable = tqdm,
            verbose: bool = False,
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
        tqdm_func : callable, default=tqdm
            Progress bar function.
        verbose : bool, default=False
            Whether to enable verbose logging.
        """

        self.set_logger(verbose=verbose)

        min_samples_split = self.get_min_samples(param=self.min_samples_split, X=X)
        min_samples_leaf = self.get_min_samples(param=self.min_samples_leaf, X=X)
        max_features = self.get_max_features(param=self.max_features, X=X)

        self.partitions = self.init_partitions(
            X=X,
            y=y,
            metric_data=metric_data,
        )

        for depth in tqdm_func(
            range(self.max_depth),
            desc="Depth",
            disable=not self.verbose
        ):
            for partition_id, partition in dict(self.partitions).items():

                # Skip iteration if the partition does not concerns the active depth
                if partition.depth != depth:
                    continue

                # Skip iteration if there are fewer than 2 observations to split
                if np.sum(partition.mask) < min_samples_split:
                    continue

                split_options, mask_matrix = self.get_split_options(
                    X=X[partition.mask, :],
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                )

                # Skip iteration if no valid split options has been found
                if len(split_options) == 0:
                    continue

                partition_metric_data = metric_data[partition.mask]

                split_evaluations = self.get_split_evaluations(
                    split_options=split_options,
                    mask_matrix=mask_matrix,
                    metric_data=partition_metric_data,
                    batch_size=batch_size,
                    tqdm_func=tqdm_func,
                )

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
                        value=self.get_partition_value(y[mask_side1]),
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
                        value=self.get_partition_value(y[mask_side2]),
                        metric=metric,
                        metadata=metadata,
                        historic_splits=partition.historic_splits + [split_side2],
                    )

    def predict_x(
            self,
            x: np.ndarray,
        ) -> DecisionTreePrediction:
        """
        Predict for a single sample.

        Parameters
        ----------
        x : np.ndarray
            A single sample represented as a dictionary of feature values.

        Returns
        -------
        list
            The prediction.
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

    def predict_all(
            self,
            X: np.ndarray,
        ) -> List[DecisionTreePrediction]:
        """
        Predict for a set of samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        list of DecisionTreePrediction
            List of predictions.
        """

        predictions = []
        for i in range(len(X)):
            prediction = self.predict_x(x=X[i,:])
            predictions.append(prediction)

        return predictions

    def predict(
            self,
            X: np.ndarray,
        ) -> np.ndarray:
        """
        Predict the values as an array.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted values.
        """

        tree_predictions = self.predict_all(X=X)
        predictions = np.array([p.value for p in tree_predictions])

        return predictions

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
                    if isinstance(partition.value, List):
                        label += f" | repartition = {partition.value}"
                    else:
                        label += f" | mean y = {partition.value:.{digits}f}"

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

            label = f"{metric_name} = {partition.metric:.{digits_metric}f}"

            if show_repartition:

                label += f"\n{partition.nb_obs} obs"
                if isinstance(partition.value, List):
                    label += f" {partition.value}"
                else:
                    label += f" | mean y = {partition.value:.{digits}f}"

            if show_metadata:
                label += "\n\nMETADATA:\n"

                parts = []
                for k, v in partition.metadata.items():
                    parts.append(f"{k}: {v:.{digits}f}")

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

        plt.title(title, fontsize=14, loc="left")
        plt.tight_layout()
        plt.show()
