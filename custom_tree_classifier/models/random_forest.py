from typing import Tuple

import numpy as np
from tqdm import tqdm

from custom_tree_classifier.metrics import Gini, MetricBase
from custom_tree_classifier.models.decision_tree import CustomDecisionTreeClassifier


class CustomRandomForestClassifier:
    """
    A custom implementation of a random forest classifier with configurable estimators
    and splitting metrics.
    """

    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 5,
            metric: MetricBase | None = None
        ) -> None:
        """
        Initialize the random forest classifier.

        Parameters
        ----------
        n_estimators : int, optional
            The number of trees in the forest (default is 100).
        max_depth : int, optional
            The maximum depth of each tree (default is 5).
        metric : MetricBase or None, optional
            The metric used for splitting. Defaults to `Gini` if not provided.
        """

        self.n_estimators = n_estimators
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

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            metric_data: np.ndarray
        ) -> None:
        """
        Train the random forest classifier.

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

        forest = {}
        for id_estimator in tqdm(range(self.n_estimators)):

            sub_var = np.random.choice(
                range(X.shape[1]),
                size=int(np.sqrt(X.shape[1])),
                replace=False
            )

            sub_obs = np.random.choice(
                range(X.shape[0]),
                size=X.shape[0],
                replace=True
            )

            model = CustomDecisionTreeClassifier(
                max_depth=self.max_depth,
                metric=self.metric
            )

            model.fit(
                X=X[sub_obs,:][:,sub_var],
                y=y[sub_obs],
                metric_data=metric_data[sub_obs]
            )

            for _, partition in model.partitions.items():
                if "splitting" in partition:
                    partition["splitting"].update(
                        {
                            "id_var": sub_var[partition["splitting"]["id_var"]]
                        }
                    )

            forest[id_estimator] = {
                "sub_obs": sub_obs,
                "model": model
            }

        self.forest = forest

    def predict_proba_x(
            self,
            x: dict
        ) -> list[float] | Tuple[list[float], str]:
        """
        Predict the class probabilities for a single sample using the ensemble.

        Parameters
        ----------
        x : dict
            A single sample represented as a dictionary of feature values.

        Returns
        -------
        list[float] or Tuple[list[float], str]
            The predicted probabilities for each class. Optionally includes additional
            metadata.
        """

        estimators_probas = []
        for id_estimator in self.forest:
            m = self.forest[id_estimator]["model"]
            proba = m.predict_proba(x)
            estimators_probas.append(proba)

        probas = list(np.mean(np.array(estimators_probas), axis=0))

        return probas

    def predict_proba(
            self,
            X: np.ndarray
        ) -> np.ndarray:
        """
        Predict the class probabilities for multiple samples using the ensemble.

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
        for i in range(X.shape[0]):
            proba = self.predict_proba_x(np.array([X[i,:]]))
            probas.append(proba[0])

        return np.array(probas)
