from __future__ import annotations

import numpy as np


def test_decision_tree_classifier_fit_builds_tree(
    fitted_decision_tree_classifier: tuple[object, np.ndarray],
) -> None:

    model, _ = fitted_decision_tree_classifier

    assert hasattr(model, "partitions")
    assert len(model.partitions) > 0


def test_decision_tree_classifier_predict_returns_binary_labels(
    fitted_decision_tree_classifier: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_decision_tree_classifier

    predictions = model.predict(x)

    assert predictions.shape == (x.shape[0],)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_decision_tree_classifier_predict_proba_returns_valid_probas(
    fitted_decision_tree_classifier: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_decision_tree_classifier

    probas = model.predict_proba(x)

    assert probas.shape == (x.shape[0], 2)
    assert np.allclose(np.sum(probas, axis=1), 1.0)


def test_decision_tree_classifier_predict_all_returns_one_prediction_per_row(
    fitted_decision_tree_classifier: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_decision_tree_classifier

    predictions_all = model.predict_all(x[:10])

    assert len(predictions_all) == 10


def test_decision_tree_classifier_print_tree_writes_output(
    fitted_decision_tree_classifier: tuple[object, np.ndarray],
    capsys,
) -> None:

    model, _ = fitted_decision_tree_classifier

    model.print_tree(max_depth=1)

    printed = capsys.readouterr().out
    assert "obs ->" in printed


def test_decision_tree_classifier_plot_tree_calls_show(
    fitted_decision_tree_classifier: tuple[object, np.ndarray],
    plot_show_spy: list[bool],
) -> None:

    model, _ = fitted_decision_tree_classifier

    model.plot_tree(max_depth=1)

    assert len(plot_show_spy) == 1


def test_decision_tree_classifier_multiclass_predict_returns_valid_labels(
    fitted_decision_tree_classifier_multiclass: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_decision_tree_classifier_multiclass

    predictions = model.predict(x)

    assert predictions.shape == (x.shape[0],)
    assert set(np.unique(predictions)).issubset(set(model.classes.tolist()))


def test_decision_tree_classifier_multiclass_predict_proba_returns_valid_probas(
    fitted_decision_tree_classifier_multiclass: tuple[object, np.ndarray],
) -> None:

    model, x = fitted_decision_tree_classifier_multiclass

    probas = model.predict_proba(x)

    assert probas.shape == (x.shape[0], 3)
    assert np.allclose(np.sum(probas, axis=1), 1.0)


def test_decision_tree_classifier_multiclass_print_tree_writes_output(
    fitted_decision_tree_classifier_multiclass: tuple[object, np.ndarray],
    capsys,
) -> None:

    model, _ = fitted_decision_tree_classifier_multiclass

    model.print_tree(max_depth=1)

    printed = capsys.readouterr().out
    assert "obs ->" in printed


def test_decision_tree_classifier_multiclass_plot_tree_calls_show(
    fitted_decision_tree_classifier_multiclass: tuple[object, np.ndarray],
    plot_show_spy: list[bool],
) -> None:

    model, _ = fitted_decision_tree_classifier_multiclass

    model.plot_tree(max_depth=1)

    assert len(plot_show_spy) == 1
