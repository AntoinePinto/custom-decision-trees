from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_decision_trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from custom_decision_trees.metrics import MSE, Gini


@pytest.fixture(scope="session")
def base_path() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def titanic_data(base_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    features = ["Pclass", "Age", "Fare", "Sex", "Embarked", "SibSp", "Parch"]

    df = (
        pd
        .read_csv(f"{base_path}/notebooks/data/titanic.csv")
        [["Survived", *features]]
        .dropna()
    )

    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    x = np.array(df[features], dtype=object)
    y = np.array(df["Survived"], dtype=int)
    metric_data = np.array(df[["Survived"]])

    return x, y, metric_data

@pytest.fixture(scope="session")
def boston_data(base_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    df = (
        pd
        .read_csv(f"{base_path}/notebooks/data/boston_housing.csv")
    )

    features = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
        "PTRATIO", "LSTAT", "CAT. MEDV"
    ]

    X = np.array(df[features])
    y = np.array(df["MEDV"])
    metric_data = np.array(df[["MEDV"]])

    return X, y, metric_data


@pytest.fixture(scope="session")
def wine_data(base_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    df = pd.read_csv(f"{base_path}/notebooks/data/wine.csv")

    features = [
        "Alcohol", "Malic.acid", "Ash", "Acl", "Mg", "Phenols",
        "Flavanoids", "Nonflavanoid.phenols", "Proanth", "Color.int", "Hue",
        "OD", "Proline"
    ]

    x = np.array(df[features])
    y = np.array(df["Wine"], dtype=int)
    metric_data = np.array(df[["Wine"]], dtype=int)

    return x, y, metric_data


@pytest.fixture(scope="module")
def fitted_decision_tree_classifier(
    titanic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[DecisionTreeClassifier, np.ndarray]:

    x, y, metric_data = titanic_data

    model = DecisionTreeClassifier(
        metric=Gini(),
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        nb_max_conditions_per_node=2,
        nb_max_cut_options_per_var=20,
        random_state=42,
        n_jobs=1,
    )

    model.fit(x, y, metric_data)

    return model, x


@pytest.fixture(scope="module")
def fitted_decision_tree_classifier_multiclass(
    wine_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[DecisionTreeClassifier, np.ndarray]:

    x, y, metric_data = wine_data

    model = DecisionTreeClassifier(
        metric=Gini(n_classes=3),
        max_depth=2,
        n_jobs=1,
    )

    model.fit(x, y, metric_data)

    return model, x


@pytest.fixture(scope="module")
def fitted_decision_tree_regressor(
    boston_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[DecisionTreeRegressor, np.ndarray]:

    x, y, metric_data = boston_data

    model = DecisionTreeRegressor(
        metric=MSE(),
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        nb_max_conditions_per_node=2,
        nb_max_cut_options_per_var=10,
        random_state=42,
        n_jobs=1,
    )

    model.fit(x, y, metric_data)

    return model, x


@pytest.fixture(scope="module")
def fitted_random_forest_classifier(
    titanic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[RandomForestClassifier, np.ndarray]:

    x, y, metric_data = titanic_data

    model = RandomForestClassifier(
        metric=Gini(),
        n_estimators=10,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        nb_max_conditions_per_node=2,
        nb_max_cut_options_per_var=10,
        bootstrap=True,
        max_samples=None,
        random_state=42,
        n_jobs=1,
    )

    model.fit(x, y, metric_data)

    return model, x


@pytest.fixture(scope="module")
def fitted_random_forest_regressor(
    boston_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[RandomForestRegressor, np.ndarray]:

    x, y, metric_data = boston_data

    model = RandomForestRegressor(
        metric=MSE(),
        n_estimators=10,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        nb_max_conditions_per_node=1,
        nb_max_cut_options_per_var=10,
        bootstrap=True,
        max_samples=None,
        random_state=42,
        n_jobs=1,
    )

    model.fit(x, y, metric_data)

    return model, x


@pytest.fixture
def plot_show_spy(monkeypatch: pytest.MonkeyPatch):
    calls: list[bool] = []

    def fake_show(*args: object, **kwargs: object) -> None:
        calls.append(True)

    monkeypatch.setattr(plt, "show", fake_show)
    yield calls
    plt.close("all")
