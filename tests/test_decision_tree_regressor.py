from __future__ import annotations

import numpy as np


def test_decision_tree_regressor_fit_builds_tree(
    fitted_decision_tree_regressor: tuple[object, np.ndarray],
) -> None:

    model, _ = fitted_decision_tree_regressor

    assert hasattr(model, "partitions")
    assert len(model.partitions) > 0


def test_decision_tree_regressor_predict_returns_finite_values(
    fitted_decision_tree_regressor: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_decision_tree_regressor

    predictions = model.predict(x)

    assert predictions.shape == (x.shape[0],)
    assert np.isfinite(predictions).all()


def test_decision_tree_regressor_predict_all_returns_one_prediction_per_row(
    fitted_decision_tree_regressor: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_decision_tree_regressor

    predictions_all = model.predict_all(x[:10])

    assert len(predictions_all) == 10


def test_decision_tree_regressor_print_tree_writes_output(
    fitted_decision_tree_regressor: tuple[object, np.ndarray],
    capsys,
) -> None:

    model, _ = fitted_decision_tree_regressor

    model.print_tree(max_depth=1)

    printed = capsys.readouterr().out
    assert "obs ->" in printed


def test_decision_tree_regressor_plot_tree_calls_show(
    fitted_decision_tree_regressor: tuple[object, np.ndarray],
    plot_show_spy: list[bool],
) -> None:

    model, _ = fitted_decision_tree_regressor

    model.plot_tree(max_depth=1)

    assert len(plot_show_spy) == 1
