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


def test_parse_single_row_vector_returns_1d():
    """Regression: `name = [1 2 3];` (no transpose) should return shape (3,),
    not (1, 3). The squeeze-on-singleton-dim path was added in cb1101a."""
    src = "name = [1 2 3];"
    out = parse_matlab_assignment(src, "name")
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [1, 2, 3])


def test_parse_indexed_pages_then_target_negation():
    """Regression: when a 3-D variable is built via `name(:,:,k) = ...` pages
    and then negated by `name = -name;`, the post-loop assembly must NOT
    overwrite the negated value. Fixed in cb1101a."""
    src = """
    Cy1(:,:,1) = [1 2; 3 4];
    Cy1(:,:,2) = [5 6; 7 8];
    Cy1 = -Cy1;
    """
    out = parse_matlab_assignment(src, "Cy1")
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out[:, :, 0], [[-1, -2], [-3, -4]])
    np.testing.assert_allclose(out[:, :, 1], [[-5, -6], [-7, -8]])


def test_side_variable_negation_propagates():
    """Regression: a side variable (`mxwy1 = [...]; mxwy1 = -mxwy1;`) that
    is later requested via parse_matlab_file must reflect the negation, not
    the original. This pattern appears in angular GetMx.m / GetMy.m."""
    src = """
    mxwy1 = [-1.0 -2.0 -3.0];
    mxwy1 = -mxwy1;
    """
    out = parse_matlab_assignment(src, "mxwy1")
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])
