from __future__ import annotations

import numpy as np


def test_random_forest_classifier_fit_builds_forest(
    fitted_random_forest_classifier: tuple[object, np.ndarray],
) -> None:

    model, _ = fitted_random_forest_classifier

    assert hasattr(model, "forest")
    assert len(model.forest) == model.n_estimators


def test_random_forest_classifier_predict_returns_binary_labels(
    fitted_random_forest_classifier: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_random_forest_classifier

    predictions = model.predict(x)

    assert predictions.shape == (x.shape[0],)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_random_forest_classifier_predict_proba_returns_valid_probas(
    fitted_random_forest_classifier: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_random_forest_classifier

    probas = model.predict_proba(x)

    assert probas.shape == (x.shape[0], 2)
    assert np.allclose(np.sum(probas, axis=1), 1.0)


def test_random_forest_classifier_predict_all_returns_one_prediction_per_row(
    fitted_random_forest_classifier: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_random_forest_classifier

    predictions_all = model.predict_all(x[:5])

    assert len(predictions_all) == 5


def test_random_forest_classifier_print_forest_writes_output(
    fitted_random_forest_classifier: tuple[object, np.ndarray],
    capsys,
) -> None:

    model, _ = fitted_random_forest_classifier

    model.print_forest(n_estimators=2, max_depth=1)

    printed = capsys.readouterr().out
    assert "TREE 1:" in printed
    assert "TREE 2:" in printed


def test_random_forest_classifier_plot_forest_calls_show_per_tree(
    fitted_random_forest_classifier: tuple[object, np.ndarray],
    plot_show_spy: list[bool],
) -> None:

    model, _ = fitted_random_forest_classifier

    model.plot_forest(n_estimators=2, max_depth=1)

    assert len(plot_show_spy) == 2
