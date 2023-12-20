import pytest
import numpy as np
from colors import Color, COLOR_DICT

# Test find_closest_color with various realistic test values
@pytest.mark.parametrize("test_id, rgb_color, expected_color_name", [
    ("happy-1", (255, 0, 0), "red"),
    ("happy-2", (0, 255, 0), "lime"),
    ("happy-3", (0, 0, 255), "blue"),
    ("happy-4", (255, 255, 0), "yellow"),
    ("happy-5", (0, 255, 255), "aqua"),
])
def test_find_closest_color_happy_path(test_id, rgb_color, expected_color_name):
    # Act
    color = Color()
    closest_color = color.find_closest_color(rgb_color)

    # Assert
    assert closest_color == expected_color_name, f"Test {test_id} failed: {closest_color} is not the expected color {expected_color_name}"

# Test find_closest_color with edge cases (colors not in the dictionary)
@pytest.mark.parametrize("test_id, rgb_color, expected_color_name", [
    ("edge-1", (128, 128, 129), "gray"),
    ("edge-2", (255, 255, 254), "white"),
    ("edge-3", (0, 0, 1), "black"),
    ("edge-4", (254, 0, 0), "red"),
    ("edge-5", (1, 255, 0), "lime"),
])
def test_find_closest_color_edge_cases(test_id, rgb_color, expected_color_name):
    # Act
    color = Color()
    closest_color = color.find_closest_color(rgb_color)

    # Assert
    assert closest_color == expected_color_name, f"Test {test_id} failed: {closest_color} is not the expected color {expected_color_name}"

# Test find_closest_color with error cases (invalid input types)
@pytest.mark.parametrize("test_id, rgb_color, expected_exception", [
    ("error-1", "not a tuple", TypeError),
    ("error-2", (256, 255, 255), ValueError),
    ("error-3", (-1, 0, 0), ValueError),
    ("error-4", (0, 0, 0, 0), ValueError),
    ("error-5", [255, 0, 0], TypeError),
])
def test_find_closest_color_error_cases(test_id, rgb_color, expected_exception):
    # Act and Assert
    color = Color()
    with pytest.raises(expected_exception):
        color.find_closest_color(rgb_color)

# Test get_dominant_color with various realistic test values
@pytest.mark.parametrize("test_id, roi, num_clusters, expected_dominant_color", [
    ("happy-1", np.array([[255, 0, 0], [255, 0, 0], [255, 0, 0]]), 1, np.array([[255, 0, 0]])),
    ("happy-2", np.array([[0, 255, 0], [0, 255, 0], [0, 255, 0]]), 1, np.array([[0, 255, 0]])),
    ("happy-3", np.array([[0, 0, 255], [0, 0, 255], [0, 0, 255]]), 1, np.array([[0, 0, 255]])),
    ("happy-4", np.array([[255, 255, 0], [255, 255, 0], [255, 255, 0]]), 1, np.array([[255, 255, 0]])),
    ("happy-5", np.array([[0, 255, 255], [0, 255, 255], [0, 255, 255]]), 1, np.array([[0, 255, 255]])),
])
def test_get_dominant_color_happy_path(test_id, roi, num_clusters, expected_dominant_color):
    # Arrange
    color = Color(num_clusters=num_clusters)

    # Act
    dominant_color = color.get_dominant_color(roi)

    # Assert
    assert np.array_equal(dominant_color, expected_dominant_color), f"Test {test_id} failed: {dominant_color} is not the expected dominant color {expected_dominant_color}"

# Test get_dominant_color with edge cases (empty ROI or single color ROI)
@pytest.mark.parametrize("test_id, roi, num_clusters, expected_dominant_color", [
    ("edge-1", np.array([]).reshape(0, 3), 1, np.array([])),
    ("edge-2", np.array([[255, 255, 255]]), 1, np.array([[255, 255, 255]])),
])
def test_get_dominant_color_edge_cases(test_id, roi, num_clusters, expected_dominant_color):
    # Arrange
    color = Color(num_clusters=num_clusters)

    # Act
    dominant_color = color.get_dominant_color(roi)

    # Assert
    assert np.array_equal(dominant_color, expected_dominant_color), f"Test {test_id} failed: {dominant_color} is not the expected dominant color {expected_dominant_color}"

# Test get_dominant_color with error cases (invalid input types)
@pytest.mark.parametrize("test_id, roi, num_clusters, expected_exception", [
    ("error-1", "not an ndarray", 1, TypeError),
    ("error-2", np.array([[256, 255, 255]]), 1, ValueError),
    ("error-3", np.array([[-1, 0, 0]]), 1, ValueError),
    ("error-4", np.array([[0, 0]]), 1, ValueError),
    ("error-5", [[255, 0, 0]], 1, TypeError),
])
def test_get_dominant_color_error_cases(test_id, roi, num_clusters, expected_exception):
    # Arrange
    color = Color(num_clusters=num_clusters)

    # Act and Assert
    with pytest.raises(expected_exception):
        color.get_dominant_color(roi)
