<div align="center">
<h1 align="center">
  <a><img src="https://github.com/AntoinePinto/custom-tree-classifier/blob/master/media/logo.png?raw=true" width="80"></a>
  <br>
  <b>Custom Tree Classifier</b>
  <br>
</h1>

![Static Badge](https://img.shields.io/badge/python->3.7-blue)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

</div>

**Custom Tree Classifier** is a Python package that allows building decision trees and random forests with custom splitting criteria, enabling optimization for specific problems. Users can define metrics like Gini, economic profit, or any custom cost function. 

This flexibility is particularly useful in "cost-dependent" scenarios.

<p align="center">
  <img src="https://github.com/AntoinePinto/custom-decision-tree/blob/master/media/illustration.jpg?raw=true" alt="drawing" width="500"/>
</p>

## Examples of use

Here are some examples of how custom splitting criteria can be beneficial:

- **Trading Movements Classification:** When the goal is to maximize economic profit, the metric can be set to economic profit, optimizing tree splitting accordingly.
- **Churn Prediction:** To minimize false negatives, metrics like F1 score or recall can guide the splitting process.
- **Fraud Detection:** Splitting can be optimized based on the proportion of fraudulent transactions identified relative to the total, rather than overall classification accuracy.
- **Marketing Campaigns:** The splitting can focus on maximizing expected revenue from customer segments identified by the tree.

## Usage

> See `./notebooks/Example.ipynb` for a complete example.

### Installation

```
pip install custom_tree_classifier
```

### Define your metric

To integrate a specific measure, the user must define a class containing the `compute_metric` and `compute_delta` methods, then insert this class into the classifier.

Example of a class with the Gini index :

```python
import numpy as np

from custom_tree.metrics import MetricBase

class Gini(MetricBase):

    @staticmethod
    def compute_metric(metric_data: np.ndarray) -> np.float64:

        y = metric_data[:, 0]

        prop0 = np.sum(y == 0) / len(y)
        prop1 = np.sum(y == 1) / len(y)

        metric = 1 - (prop0**2 + prop1**2)

        return metric

    @staticmethod
    def compute_delta(
            split: np.ndarray,
            metric_data: np.ndarray
        ) -> np.float64:

        delta = (
            Gini.compute_metric(metric_data) -
            Gini.compute_metric(metric_data[split]) * np.mean(split) -
            Gini.compute_metric(metric_data[np.invert(split)]) * (1 - np.mean(split))
        )

        return delta
```

### Train and predict

Once you have instantiated the model with your custom metric, all you have to do is use the `.fit` and `.predict_proba` methods:

```python
model = CustomDecisionTreeClassifier(
    max_depth=3,
    metric=Gini
)

model.fit(
    X=X_train, 
    y=y_train, 
    metric_data=metric_data
)

probas = model.predict_proba(
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
features_names = {
    0: "Pclass", 
    1: "Age"
}

model.print_tree(
    features_names=features_names,
    digits=2,
    metric_name="MyMetric"
)
```

```
>>> [1] -> MyMetric = 0.48 | repartition = [424, 290]
    |    Δ MyMetric = +0.05
    |   [2] Pclass <= 2.0 -> MyMetric = 0.49 | repartition = [154, 205]
    |   |    Δ MyMetric = +0.03
    |   |   |    Δ MyMetric = +0.01
    |   |   |    Δ MyMetric = +0.03
    |   [3] Pclass > 2.0 -> MyMetric = 0.36 | repartition = [270, 85]
    |   |    Δ MyMetric = +0.02
    |   |   |    Δ MyMetric = +0.04
    |   |   |    Δ MyMetric = +0.01
```

### Random Forest Classifier

Same with Random Forest Classifier :

```python
from custom_tree_classifier import CustomRandomForestClassifier

random_forest = CustomRandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    metric=Gini
)

random_forest.fit(
    X=X_train, 
    y=y_train, 
    metric_data=metric_data
)

probas = random_forest.predict_proba(
    X=X_test
)
```

## Reminder on splitting criteria

Typically, classification trees are constructed using a splitting criterion that is based on a measure of impurity or information gain. 

Let us consider a 2-class classification using the Gini index as metric. The Gini index represents the impurity of a group of observations based on the proportion of observations in each class 0 and 1 :

$$ I_{G} = 1 - p_0^2 - p_1^2 $$

Since the Gini index is an indicator of impurity, partitioning is done by minimising the weighted average of the index in the child nodes $L$ and $R$. This is equivalent to minimising $\Delta$ :

$$ \Delta = \frac{N_t}{N} \times (I_G - \frac{N_{t_L} * I_{G_L}}{N_t} - \frac{N_{t_R} * I_{G_R}}{N_t}) $$

At each node, the tree algorithm finds the split that minimizes $\Delta$ over all possible splits and over all features. Once the optimal split is selected, the tree is grown by recursively applying this splitting process to the resulting child nodes.