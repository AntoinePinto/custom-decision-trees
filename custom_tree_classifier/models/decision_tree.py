from typing import Any, Tuple

import numpy as np

from custom_tree_classifier.metrics import Gini, MetricBase


class CustomDecisionTreeClassifier:
    """
    A custom implementation of a decision tree classifier with support for different
    splitting metrics.
    """

    def __init__(
            self,
            max_depth: int = 5,
            metric: MetricBase | None = None
        ) -> None:
        """
        Initialize the decision tree classifier.

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of the tree (default is 5).
        metric : MetricBase or None, optional
            The metric used for splitting. Defaults to `Gini` if not provided.
        """

        self.max_depth = max_depth

        if metric is not None:
            self.setup_metric(metric)
        else:
            self.setup_metric(Gini)

    def setup_metric(
            self,
            metric: MetricBase
        ) -> None:
        """
        Configure the metric used for splitting.

        Parameters
        ----------
        metric : MetricBase
            The metric used for splitting.
        """

        self.metric = metric

    def get_best_split(
            self,
            values: list,
            metric_data: np.ndarray
        ) -> Tuple:
        """
        Find the best split for a variable.

        Parameters
        ----------
        values : list
            The variable values to evaluate.
        metric_data : np.ndarray
            The metric data used to compute split quality.

        Returns
        -------
        Tuple
            The best split value and its corresponding metric delta.
        """

        splits = np.sort(np.unique(values))[:-1]

        if len(splits) == 0:
            return np.nan, np.nan

        if len(splits) > 200:
            splits = np.quantile(splits, [i/200 for i in range(200)])

        deltas = {}
        for split in splits:

            delta = self.metric.compute_delta(
                split=values > split,
                metric_data=metric_data
            )

            deltas[split] = delta

        if len(deltas) == 0:
            return None, None

        value = max(deltas, key=lambda x: deltas[x])
        delta = deltas[value]

        return value, delta

    def get_best_split_by_var(
            self,
            X: np.ndarray,
            metric_data: np.ndarray
        ) -> list[dict[str, Any]]:
        """
        Identify the best splits for all variables.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        metric_data : np.ndarray
            The metric data used for evaluating splits.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries containing the best split information for each
            variable.
        """

        best_splits = []
        id_vars = range(X.shape[1])
        for id in id_vars:

            value, delta = self.get_best_split(
                values=X[:, id],
                metric_data=metric_data
            )

            if np.isnan(delta) | (np.round(delta, 10) == 0):
                continue

            best_splits.append(
                {
                    "id_var": id,
                    "value": value,
                    "delta": delta
                }
            )

        return best_splits

    def get_repartition(
            self,
            y: np.ndarray
        ) -> list[int]:
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

        repartition = [int(np.sum(y == 0)), int(np.sum(y == 1))]

        return repartition

    def get_init_partitions(
            self,
            y: np.ndarray,
            metric_data: np.ndarray
        ) -> dict[int, dict[str, Any]]:
        """
        Create the initial partitions for the tree.

        Parameters
        ----------
        y : np.ndarray
            The target variable.
        metric_data : np.ndarray
            The metric data for evaluation.

        Returns
        -------
        dict[int, dict[str, Any]]
            A dictionary containing the initial partitions.
        """

        init_partitions = {
            1: {
                "type": "leaf",
                "depth": 0,
                "mask": np.repeat(True, len(y)),
                "metric": self.metric.compute_metric(metric_data),
                "repartition": self.get_repartition(y)
            }
        }

        return init_partitions

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            metric_data: np.ndarray
        ) -> None:
        """
        Train the decision tree classifier.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        y : np.ndarray
            The target variable.
        metric_data : np.ndarray
            The metric data used for evaluation.

        Raises
        ------
        ValueError
            If any of the inputs contain NaN values.
        """

        if np.any(np.isnan(X)):
            raise ValueError("Input X contains NaN values.")

        if np.any(np.isnan(y)):
            raise ValueError("Input y contains NaN values.")

        if np.any(np.isnan(metric_data)):
            raise ValueError("Input metric_data contains NaN values.")

        self.partitions = self.get_init_partitions(
            y=y,
            metric_data=metric_data
        )

        for depth in range(self.max_depth):
            for partition_id, partition in dict(self.partitions).items():

                # Skip iteration if the partition does not concerns the active depth
                if partition["depth"] != depth:
                    continue

                # Skip iteration if there are fewer than 2 observations to split
                if np.sum(partition["repartition"]) < 2:
                    continue

                best_splits = self.get_best_split_by_var(
                    X=X[partition["mask"], :],
                    metric_data=metric_data[partition["mask"]]
                )

                # Skip iteration if no valid split has been found
                if len(best_splits) == 0:
                    continue

                best_split = max(best_splits, key=lambda split: split["delta"])

                partition["type"] = "branch"
                partition["splitting"] = best_split

                id_var, value = best_split["id_var"], best_split["value"]
                mask_side1 = partition["mask"] * (X[:, id_var] <= value)
                mask_side2 = partition["mask"] * (X[:, id_var] > value)

                if sum(mask_side1) > 0:

                    self.partitions[partition_id * 2] = {
                        "type": "leaf",
                        "cut": "<=",
                        "depth": partition["depth"] + 1,
                        "mask": mask_side1,
                        "metric": self.metric.compute_metric(metric_data[mask_side1]),
                        "repartition": self.get_repartition(y[mask_side1])
                    }

                if sum(mask_side2) > 0:

                    self.partitions[partition_id * 2 + 1] = {
                        "type": "leaf",
                        "cut": ">",
                        "depth": partition["depth"] + 1,
                        "mask": mask_side2,
                        "metric": self.metric.compute_metric(metric_data[mask_side2]),
                        "repartition": self.get_repartition(y[mask_side2])
                    }

    def get_proba(
            self,
            x: dict
        ) -> float:
        """
        Predict the class probabilities for a single sample.

        Parameters
        ----------
        x : dict
            A single sample represented as a dictionary of feature values.

        Returns
        -------
        float
            The predicted probabilities for each class.
        """

        partition_id = 1
        while True:

            partition = self.partitions[partition_id]

            if partition["type"] == "leaf":
                probas = partition["repartition"] / np.sum(partition["repartition"])
                return probas
            else:
                splitting = partition["splitting"]
                if x[splitting["id_var"]] <= splitting["value"]:
                    partition_id = partition_id * 2
                else:
                    partition_id = partition_id * 2 + 1

    def predict_proba(
            self,
            X: np.ndarray
        ) -> np.ndarray:
        """
        Predict the class probabilities for multiple samples.

        Parameters
        ----------
        X : np.ndarray
            The input features.

        Returns
        -------
        np.ndarray
            The predicted probabilities for each sample.

        Raises
        ------
        ValueError
            If the input contains NaN values.
        """

        if np.any(np.isnan(X)):
            raise ValueError("Input X contains NaN values.")

        probas = []
        for i in range(len(X)):
            proba = self.get_proba(x=X[i,:])
            probas.append(proba)

        return np.array(probas)

    def print_tree(
            self,
            max_depth: int = 1000,
            features_names: dict[str, str] | None = None,
            show_delta: bool = True,
            show_metric: bool = True,
            show_repartition: bool = True,
            show_metadata: bool = False,
            digits: int = 100,
            metric_name: str = "metric"
        ) -> None:
        """
        Print the decision tree structure.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to display (default is 1000).
        features_names : dict[str, str] or None, optional
            Mapping of feature indices to feature names (default is None).
        show_delta : bool, optional
            Whether to display metric delta for splits (default is True).
        show_metric : bool, optional
            Whether to display the metric value for each partition
            (default is True).
        show_repartition : bool, optional
            Whether to display the class distribution for each partition
            (default is True).
        show_metadata : bool, optional
            Whether to display additional metadata (default is False).
        digits : int, optional
            Number of digits for rounding (default is 100).
        metric_name : str, optional
            Name of the metric used for display (default is "metric").
        """

        partition_id = 1
        while True:

            if partition_id not in self.partitions:
                parents = self.list_parents(partition_id)
                partition_id = self.next_uncle(parents)
                if partition_id is None:
                    break

            partition = self.partitions[partition_id]

            if partition["depth"] <= max_depth:

                print("|   " * (partition["depth"]) + "", end="")
                print(f"[{partition_id}]", end = "")

                if partition_id > 1:
                    parent_id = int((partition_id - (partition_id % 2 == 1)) / 2)
                    splitting = self.partitions[parent_id]["splitting"]

                    if features_names is not None:
                        var = features_names[splitting["id_var"]]
                    else:
                        var = "feature " + str(splitting["id_var"])

                    cut_print = " {var} {sign} {value}".format(
                        var=var,
                        sign=partition["cut"],
                        value=round(splitting["value"], digits)
                    )

                    print(cut_print, end="")

                if show_metric is True:

                    metric_print = " -> {metric_name} = {metric}".format(
                        metric_name=metric_name,
                        metric=round(partition["metric"], digits)
                    )

                    print(metric_print, end="")

                if show_repartition is True:

                    repartition_print = " | repartition = {repartition}".format(
                        repartition=partition["repartition"]
                    )

                    print(repartition_print, end="")

                if show_metadata is True:

                    metadata_print = " | {metadata}".format(
                        metadata=partition["metadata"]
                    )

                    print(metadata_print, end="")

                print("")

            if partition["type"] == "branch":

                if show_delta is True:
                    print("|   " * (1 + partition["depth"]) + "", end="")

                    delta = round(partition["splitting"]["delta"], digits)

                    delta_print = " Δ {metric_name} = {delta}".format(
                        metric_name=metric_name,
                        delta=f"+{delta}" if delta > 0 else delta
                    )

                    print(delta_print)

                partition_id = partition_id * 2
            elif partition["cut"] == "<=":
                partition_id = partition_id + 1
            else:
                parents = self.list_parents(partition_id)
                partition_id = self.next_uncle(parents)
                if partition_id is None:
                    break

    def list_parents(
            self,
            id: int
        ) -> list[int]:
        """
        Retrieve the list of parent partitions for a given partition ID.

        Parameters
        ----------
        id : int
            The partition ID.

        Returns
        -------
        list[int]
            A list of parent partition IDs.
        """

        parents = []
        while id != 1:
            id = int((id - (id % 2 == 1)) / 2)
            parents.append(id)

        return parents

    def next_uncle(
            self,
            parents: list
        ) -> int | None:
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
                if (id + 1 in self.partitions) and (id % 2 == 0)
            ),
            None
        )

        return id
