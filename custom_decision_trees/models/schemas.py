from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


@dataclass
class Cut:
    """
    Represents a single binary condition (cut) on a feature for a decision node.

    Attributes
    ----------
    id : int
        Unique identifier for the cut.
    var_id : str
        Identifier of the feature on which the cut is applied.
    sign : str
        Comparison operator (e.g., '>', '<=', etc.).
    value : float
        The threshold value for the split.
    nb_obs : int
        Number of observations satisfying this condition.
    """

    id: int
    var_id: str
    sign: str
    value: float | str
    nb_obs: int

@dataclass
class Split:
    """
    Represents a composite split made of multiple `Cut` objects.

    Attributes
    ----------
    cuts : list[Cut]
        List of binary conditions.
    operator : Literal["&", "|"], optional
        Logical operator for combining conditions, default is '&'.

    Properties
    ----------
    filter : str
        A string representation of the condition.
    check_x : Callable
        A lambda function to evaluate the condition for a single sample.
    """

    cuts: List[Cut]
    operator: Literal["&", "|"] = "&"

    def __post_init__(self) -> None:

        self.filter = self.get_filter()
        self.check_x = eval(f"lambda x: {self.filter}") # noqa: S307

    def get_filter(
            self,
            feature_names: List[str] | None = None,
            digits: int = 100,
            logical_operator: bool = False,
        ) -> str:
        """
        Build a string representation of the filter condition.

        Parameters
        ----------
        feature_names : list[str], optional
            List of feature names corresponding to `var_id`.
        digits : int
            Number of significant digits in formatting.
        logical_operator : bool
            Use logical names ('AND', 'OR') instead of symbols.

        Returns
        -------
        str
            A string representing the combined split conditions.
        """

        if logical_operator is True:
            operator = "AND" if self.operator == "&" else "OR"
        else:
            operator = self.operator

        condition_template = "(x[{var}] {sign} {value})"

        conditions = []
        for cut in self.cuts:

            if feature_names is not None:
                var = '"' + feature_names[int(cut.var_id)] + '"'
            else:
                var = cut.var_id

            value = cut.value
            if isinstance(value, float):
                value = round(value, digits)
            elif isinstance(value, str):
                value = f"'{value}'"

            condition = condition_template.format(
                var=var,
                sign=cut.sign,
                value=value,
            )

            conditions.append(condition)

        return f" {operator} ".join(conditions)

    def get_mask(
            self,
            mask_matrix: np.ndarray,
        ) -> np.ndarray:
        """
        Compute the boolean mask for the split.

        Parameters
        ----------
        mask_matrix : np.ndarray
            Precomputed mask matrix for each cut.

        Returns
        -------
        np.ndarray
            A boolean mask corresponding to the full split.
        """

        mask = np.logical_and.reduce([mask_matrix[cut.id] for cut in self.cuts])

        return mask

    def get_inverse_split(self):
        """
        Construct the inverse of the current split.

        Returns
        -------
        Split
            The inverse split with opposite signs and inverted logical operator.
        """

        cuts = []
        for cut_side1 in self.cuts:
            cut = replace(cut_side1, sign="<=" if cut_side1.sign == ">" else ">")
            cuts.append(cut)

        inverse_split = Split(
            cuts=cuts,
            operator="|" if self.operator == "&" else "&",
        )

        return inverse_split

@dataclass
class SplitEvaluation:
    """
    Stores evaluation data for a candidate split.

    Attributes
    ----------
    split : Split
        The split that was evaluated.
    nb_obs : int
        Number of observations falling into the split.
    metric : float
        Performance metric associated with this split.
    """

    split: Split
    nb_obs: int
    metric: float
    metadata: dict

@dataclass
class DecisionTreePrediction:
    """
    Prediction result from a decision tree for a single sample.

    Attributes
    ----------
    probas : list[float]
        Predicted class probabilities.
    metric : float
        Evaluation metric for the prediction.
    metadata : dict
        Additional information about the prediction.
    path : list[int]
        IDs of tree nodes traversed to reach the prediction.
    """

    probas: List[float]
    metric: float
    metadata: Dict
    path: List[int]

    def to_str(
            self,
            digits: int,
        ) -> str:
        """
        Format the prediction as a string.

        Parameters
        ----------
        digits : int
            Number of digits to round the output.

        Returns
        -------
        str
            Formatted prediction string.
        """

        prediction_str = "Prediction : probas={probas} ; metric={metric}".format(
            probas=[round(p, digits) for p in self.probas],
            metric=round(self.metric, digits),
        )

        return prediction_str

@dataclass
class RandomForestPrediction:
    """
    Aggregated prediction from all trees in the random forest.

    Attributes
    ----------
    probas : list[float]
        Averaged class probabilities from all trees.
    metric : float
        Averaged evaluation metric from all trees.
    trees_predictions : list[DecisionTreePrediction]
        Individual predictions from each tree.
    """

    probas: List[float]
    metric: float
    trees_predictions: List[DecisionTreePrediction]

@dataclass
class Partition:
    """
    Represents a node (partition) in a decision tree.

    Attributes
    ----------
    id : int
        Unique identifier of the partition.
    depth : int
        Depth level of the node in the tree.
    mask : np.ndarray
        Boolean array indicating which observations belong to this partition.
    metric : float
        Evaluation metric for this node.
    metadata : dict
        Additional data related to the node.
    repartition : list[int]
        Class distribution in the partition.
    historic_splits : list[Split]
        Sequence of splits that led to this partition.
    type : str, optional
        Type of the node (default is "leaf").
    split_evaluations : list[SplitEvaluation], optional
        Candidate splits and their evaluations.
    best_split : Split, optional
        The best split found for this partition.

    Properties
    ----------
    nb_obs : int
        Number of observations in the partition.
    parents : list[int]
        List of ancestor node IDs.
    prediction : DecisionTreePrediction
        Prediction object derived from the node statistics.
    filter : str
        Combined filter string from historic splits.
    """

    id: int
    depth: int
    mask: np.ndarray
    metric: float
    metadata: Dict
    repartition: List[int]
    historic_splits: List[Split]
    type: Optional[str] = "leaf"
    split_evaluations: Optional[List[SplitEvaluation]] = None
    best_split: Optional[Split] = None

    def __post_init__(self) -> None:

        self.nb_obs: int = int(sum(self.mask))
        self.parents = self.list_parents(id=self.id)
        self.prediction = self.get_prediction()

        conditions = [f"({split.filter})" for split in self.historic_splits]
        self.filter = " & ".join(conditions)

    def get_mask_sides(
            self,
            mask: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the current partition's mask into two child masks based on a condition.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask for the split condition.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple of (mask_side1, mask_side2) representing the two child partitions.
        """

        mask_side1 = self.mask.copy()
        mask_side1[self.mask] = mask

        mask_side2 = self.mask.copy()
        mask_side2[self.mask] = ~mask

        return mask_side1, mask_side2

    def list_parents(
            self,
            id: int,
        ) -> List[int]:
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
        while id != 0:
            id = int((id - 1 - (id % 2 == 0)) / 2)
            parents.append(id)

        return parents

    def get_prediction(self) -> DecisionTreePrediction:
        """
        Generate a prediction object for this partition.

        Returns
        -------
        DecisionTreePrediction
            The prediction containing class probabilities and metadata.
        """

        prediction = DecisionTreePrediction(
            probas=[p / self.nb_obs for p in self.repartition],
            metric=self.metric,
            metadata=self.metadata,
            path=[self.id] + self.parents,
        )

        return prediction
