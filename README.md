<div align="center">
<h1 align="center">
  <a><img src="https://github.com/AntoinePinto/custom-decision-trees/blob/master/media/logo.png?raw=true" width="80"></a>
  <br>
  <b>Custom Decision Trees</b>
  <br>
</h1>

![Static Badge](https://img.shields.io/badge/python->=3.10-blue)
![GitHub License](https://img.shields.io/github/license/AntoinePinto/StringPairFinder)
![PyPI - Downloads](https://img.shields.io/pypi/dm/custom-decision-trees)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

</div>

**Custom Decision Trees** is a Python package that lets you build machine learning models with advanced configuration :

## Main Features

### Splitting criteria customization

Define your own cutting criteria in Python language (documentation in the following sections). 

This feature is particularly useful in "cost-dependent" scenarios. Examples:

- **Trading Movements Classification:** When the goal is to maximize economic profit, the metric can be set to economic profit, optimizing tree splitting accordingly.
- **Churn Prediction:** To minimize false negatives, metrics like F1 score or recall can guide the splitting process.
- **Fraud Detection:** Splitting can be optimized based on the proportion of fraudulent transactions identified relative to the total, rather than overall classification accuracy.
- **Marketing Campaigns:** The splitting can focus on maximizing expected revenue from customer segments identified by the tree.

### Multi-conditional node splitting

Allow trees to split nodes with one or more simultaneous conditions.

Example of multi-condition splitting on the Titanic dataset:

![Multi Conditional Node Splitting](https://github.com/AntoinePinto/custom-decision-trees/blob/master/media/multi-condition-splitting.png?raw=true)

## Reminder on splitting criteria
Splitting in a decision tree is achieved by **optimizing a metric**. For example, Gini optimization consists in **maximizing** the $\Delta_{Gini}$ :

*   **The Gini Index** represents the impurity of a group of observations based on the observations of each class (0 and 1):

$$ I_{Gini} = 1 - p_0^2 - p_1^2 $$

*   The metric to be maximized is $\Delta_{Gini}$, the difference between **the Gini index on the parent node** and **the weighted average of the Gini index between the two child nodes** ($L$ and $R$).

$$ \Delta_{Gini} = I_{Gini} - \frac{N_L * I_{Gini_L}}{N} - \frac{N_R * I_{Gini_R}}{N} $$

At each node, the tree algorithm finds the split that minimizes $\Delta$ over all possible splits and over all features. Once the optimal split is selected, the tree is grown by recursively applying this splitting process to the resulting child nodes.

## Usage

> See `./notebooks/` folder for a complete examples.

### Installation

```
pip install custom-decision-trees
```

### Define your metric

To integrate a specific measure, the user must define a class containing the `compute_metric` and `compute_delta` methods, then insert this class into the classifier.

Example of a class with the Gini index :

```python
import numpy as np

from custom_decision_trees.metrics import MetricBase


def compute_gini(
        metric_data: np.ndarray
    ) -> float:

    y = metric_data[:, 0]

    prop0 = np.sum(y == 0) / len(y)
    prop1 = np.sum(y == 1) / len(y)

    metric = 1 - (prop0**2 + prop1**2)

    return float(metric)

class Gini(MetricBase):
    """
    A class that implements the Gini impurity metric for decision trees.
    """

    def __init__(
            self,
        ) -> None:
        pass

    def compute_metric(
            self,
            metric_data: np.ndarray,
            mask: np.ndarray
        ) -> tuple[float, dict]:

        gini_parent = compute_gini(metric_data)
        gini_side1 = compute_gini(metric_data[mask])
        gini_side2 = compute_gini(metric_data[~mask])

        delta = (
            gini_parent - 
            gini_side1 * np.mean(mask) - 
            gini_side2 * (1 - np.mean(mask))
        )

        metadata = {"gini": round(gini_side1, 3)}

        return float(delta), metadata
```

### Train and predict

Once you have instantiated the model with your custom metric, all you have to do is use the `.fit` and `.predict_proba` methods:

```python
from custom_decision_trees import DecisionTree

gini = Gini()

decision_tree = DecisionTree(
    metric=gini,
    max_depth=2,
    nb_max_conditions_per_node=2 # Set to 1 for a traditional decision tree
)

decision_tree.fit(
    X=X_train,
    y=y_train,
    metric_data=metric_data,
)

probas = model.predict_probas(
    X=X_test
)

probas[:5]
```

```
>>> array([[0.75308642, 0.24691358],
           [0.36206897, 0.63793103],
           [0.75308642, 0.24691358],
           [0.36206897, 0.63793103],
           [0.90243902, 0.09756098]])
```

## Print the tree

You can also display the decision tree, with the values of your metrics, using the `print_tree` method:

```python
decision_tree.print_tree(
    feature_names=features,
    metric_name="MyMetric",
)
```

```
>>> [0] 712 obs -> MyMetric = 0.0
    |   [1] (x["Sex"] <= 0.0) AND (x["Pclass"] <= 2.0) | 157 obs -> MyMetric = 0.16
    |   |   [3] (x["Age"] <= 2.0) AND (x["Fare"] > 26.55) | 1 obs -> MyMetric = 0.01
    |   |   [4] (x["Age"] > 2.0) OR (x["Fare"] <= 26.55) | 156 obs -> MyMetric = 0.01
    |   [2] (x["Sex"] > 0.0) OR (x["Pclass"] > 2.0) | 555 obs -> MyMetric = 0.16
    |   |   [5] (x["SibSp"] <= 2.0) AND (x["Age"] <= 8.75) | 27 obs -> MyMetric = 0.05
    |   |   [6] (x["SibSp"] > 2.0) OR (x["Age"] > 8.75) | 528 obs -> MyMetric = 0.05
```

## Plot the tree

```python
decision_tree.plot_tree(
    feature_names=features,
    metric_name="delta gini",
)
```

![Multi Conditional Node Splitting](https://github.com/AntoinePinto/custom-decision-trees/blob/master/media/multi-condition-splitting.png?raw=true)

### Random Forest

Same with Random Forest Classifier :

```python
from custom_decision_trees import RandomForest

random_forest = RandomForest(
    metric=gini,
    n_estimators=10,
    max_depth=2,
    nb_max_conditions_per_node=2,
)

random_forest.fit(
    X=X_train, 
    y=y_train, 
    metric_data=metric_data
)

probas = random_forest.predict_probas(
    X=X_test
)
```
