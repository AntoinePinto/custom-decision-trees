from __future__ import annotations

import numpy as np


def test_random_forest_regressor_fit_builds_forest(
    fitted_random_forest_regressor: tuple[object, np.ndarray],
) -> None:

    model, _ = fitted_random_forest_regressor

    assert hasattr(model, "forest")
    assert len(model.forest) == model.n_estimators


def test_random_forest_regressor_predict_returns_finite_values(
    fitted_random_forest_regressor: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_random_forest_regressor

    predictions = model.predict(x)

    assert predictions.shape == (x.shape[0],)
    assert np.isfinite(predictions).all()


def test_random_forest_regressor_predict_all_returns_one_prediction_per_row(
    fitted_random_forest_regressor: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_random_forest_regressor

    predictions_all = model.predict_all(x[:5])

    assert len(predictions_all) == 5


def test_random_forest_regressor_print_forest_writes_output(
    fitted_random_forest_regressor: tuple[object, np.ndarray],
    capsys,
) -> None:

    model, _ = fitted_random_forest_regressor

    model.print_forest(n_estimators=2, max_depth=1)

    printed = capsys.readouterr().out
    assert "TREE 1:" in printed
    assert "TREE 2:" in printed


def test_random_forest_regressor_plot_forest_calls_show_per_tree(
    fitted_random_forest_regressor: tuple[object, np.ndarray],
    plot_show_spy: list[bool],
) -> None:

    model, _ = fitted_random_forest_regressor

    model.plot_forest(n_estimators=2, max_depth=1)

    assert len(plot_show_spy) == 2
