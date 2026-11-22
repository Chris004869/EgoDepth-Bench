import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from egodepth_eval.report.figure4 import panel_motion_contrast


@pytest.fixture
def axis():
    figure, axis = plt.subplots()
    yield axis
    plt.close(figure)


def plot_comparison(axis, **values):
    comparison = {
        "label": "method\nobject",
        "unweighted": None,
        "standardized": None,
        "unweighted_ci95": None,
        "standardized_ci95": None,
        **values,
    }
    result = panel_motion_contrast(axis, {"comparisons": [comparison]})
    axis.figure.canvas.draw()
    return result


def marker_values(axis, label):
    return next(line.get_ydata() for line in axis.lines if line.get_label() == label)


def interval_segments(axis):
    return [
        segment
        for container in axis.containers
        for collection in container.lines[2]
        for segment in collection.get_segments()
    ]


def test_raw_without_interval_has_only_a_marker_and_explicit_missing_standardized(axis):
    result = plot_comparison(axis, unweighted=5.0)

    np.testing.assert_array_equal(marker_values(axis, "unweighted"), [5.0])
    assert len(marker_values(axis, "standardized")) == 0
    assert interval_segments(axis) == []
    missing_label = next(text for text in axis.texts if text.get_text() == "std n/a")
    assert axis.get_xlim()[0] < missing_label.get_position()[0] < axis.get_xlim()[1]
    assert result["comparisons"][0]["standardized"] is None


def test_numeric_zero_is_a_supported_marker_and_zero_width_interval_is_preserved(axis):
    plot_comparison(axis, unweighted=0.0, standardized=0.0,
                    standardized_ci95=[0.0, 0.0])

    np.testing.assert_array_equal(marker_values(axis, "unweighted"), [0.0])
    np.testing.assert_array_equal(marker_values(axis, "standardized"), [0.0])
    assert all("n/a" not in text.get_text() for text in axis.texts)
    segments = interval_segments(axis)
    assert len(segments) == 1
    np.testing.assert_allclose(segments[0], [[0.16, 0.0], [0.16, 0.0]])


def test_raw_and_standardized_intervals_keep_their_own_endpoints(axis):
    plot_comparison(axis, unweighted=5.0, standardized=-2.0,
                    unweighted_ci95=[3.0, 7.0], standardized_ci95=[-4.0, -1.0])

    np.testing.assert_array_equal(marker_values(axis, "unweighted"), [5.0])
    np.testing.assert_array_equal(marker_values(axis, "standardized"), [-2.0])
    segments = interval_segments(axis)
    assert len(segments) == 2
    np.testing.assert_allclose(segments[0], [[-0.16, 3.0], [-0.16, 7.0]])
    np.testing.assert_allclose(segments[1], [[0.16, -4.0], [0.16, -1.0]])


def test_interval_outside_point_estimate_is_not_reflected_around_the_marker(axis):
    plot_comparison(axis, unweighted=5.0, unweighted_ci95=[6.0, 8.0])

    np.testing.assert_array_equal(marker_values(axis, "unweighted"), [5.0])
    np.testing.assert_allclose(interval_segments(axis)[0], [[-0.16, 6.0], [-0.16, 8.0]])


def test_both_undefined_estimates_have_labels_without_numeric_markers(axis):
    plot_comparison(axis)

    assert len(marker_values(axis, "unweighted")) == 0
    assert len(marker_values(axis, "standardized")) == 0
    assert interval_segments(axis) == []
    missing_labels = [text for text in axis.texts if text.get_text().endswith("n/a")]
    assert {text.get_text() for text in missing_labels} == {"raw n/a", "std n/a"}
    assert all(axis.get_xlim()[0] < text.get_position()[0] < axis.get_xlim()[1]
               for text in missing_labels)
