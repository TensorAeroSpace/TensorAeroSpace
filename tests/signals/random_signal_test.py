import pytest
import numpy as np
from unittest.mock import patch
from io import StringIO
from tensoraerospace.signals.random import full_random_signal

def test_full_random_signal():
    # Проверка работы функции при нормальных входных данных
    t0 = 0.0
    dt = 0.1
    tn = 1.0
    sd = (0.5, 1.5)
    sv = (0.5, 1.5)
    result = full_random_signal(t0, dt, tn, sd, sv)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == int(np.floor((tn - t0) / dt) + 1)
    assert all(0.5 <= val <= 1.5 for val in result)

    # Проверка работы функции при пограничных входных данных
    sd = (0.0, 1.0)
    sv = (0.0, 1.0)
    result = full_random_signal(t0, dt, tn, sd, sv)
    assert all(0.0 <= val <= 1.0 for val in result)

    # Проверка работы функции при отрицательных входных данных
    sd = (-1.0, 1.0)
    sv = (-1.0, 1.0)
    result = full_random_signal(t0, dt, tn, sd, sv)
    assert all(-1.0 <= val <= 1.0 for val in result)

    # Проверка работы функции при очень больших входных данных
    sd = (100.0, 1000.0)
    sv = (100.0, 1000.0)
    result = full_random_signal(t0, dt, tn, sd, sv)
    assert all(100.0 <= val <= 1000.0 for val in result)

    # Проверка работы функции при нулевых входных данных
    sd = (0.0, 0.0)
    sv = (0.0, 0.0)
    result = full_random_signal(t0, dt, tn, sd, sv)
    assert all(0.0 <= val <= 0.0 for val in result)

    # Проверка работы функции при отрицательных входных данных
    sd = (-1.0, -1.0)
    sv = (-1.0, -1.0)
    result = full_random_signal(t0, dt, tn, sd, sv)
    assert all(-1.0 <= val <= -1.0 for val in result)

    # Проверка работы функции при очень больших входных данных
    sd = (1000.0, 1000.0)
    sv = (1000.0, 1000.0)
    result = full_random_signal(t0, dt, tn, sd, sv)
    assert all(1000.0 <= val <= 1000.0 for val in result)

    # Проверка работы функции при нулевых входных данных
    sd = (0.0, 0.0)
    sv = (0.0, 0.0)
    result = full_random_signal(t0, dt, tn, sd, sv)
    assert all(0.0 <= val <= 0.0 for val in result)
