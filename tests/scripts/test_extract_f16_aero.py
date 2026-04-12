import numpy as np
import pytest

from scripts.extract_f16_aero import (
    parse_matlab_assignment,
    parse_matlab_file,
)


def test_parse_simple_row_vector():
    src = "alpha1 = deg2rad([-20 -15 -10 0 5 10]);"
    out = parse_matlab_assignment(src, "alpha1")
    assert isinstance(out, np.ndarray)
    expected = np.deg2rad([-20, -15, -10, 0, 5, 10])
    np.testing.assert_allclose(out, expected)


def test_parse_column_vector_with_transpose():
    src = "Cywz1 = [-23.9 -29.5 -30.5]';"
    out = parse_matlab_assignment(src, "Cywz1")
    expected = np.array([-23.9, -29.5, -30.5])
    np.testing.assert_allclose(out, expected)


def test_parse_2d_matrix():
    src = """
    M = [1.0 2.0 3.0;
         4.0 5.0 6.0];
    """
    out = parse_matlab_assignment(src, "M")
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(out, expected)


def test_parse_3d_indexed_assignment():
    src = """
    Cy1(:,:,1) = [1.0 2.0;
                  3.0 4.0];
    Cy1(:,:,2) = [5.0 6.0;
                  7.0 8.0];
    """
    out = parse_matlab_assignment(src, "Cy1")
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out[:, :, 0], [[1, 2], [3, 4]])
    np.testing.assert_allclose(out[:, :, 1], [[5, 6], [7, 8]])


def test_parse_matlab_file_multiple_vars():
    src = """
    a = [1 2 3];
    b = [4; 5; 6];
    """
    out = parse_matlab_file(src, ["a", "b"])
    np.testing.assert_allclose(out["a"], [1, 2, 3])
    np.testing.assert_allclose(out["b"], [4, 5, 6])


def test_parse_handles_negation_followed_by_assignment():
    """Some matlab files do `Cy1 = -Cy1;` after defining Cy1. Parser must
    return the negated version when asked for the final value."""
    src = """
    Cy1 = [1 2; 3 4];
    Cy1 = -Cy1;
    """
    out = parse_matlab_assignment(src, "Cy1")
    np.testing.assert_allclose(out, [[-1, -2], [-3, -4]])
