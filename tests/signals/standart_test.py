from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest

from tensoraerospace.signals.standart import (
    constant_line,
    sinusoid,
    sinusoid_vertical_shift,
    unit_step,
)


def test_unit_step():
    # Проверка работы функции при нормальных входных данных
    tp = np.array([1.0, 2.0, 3.0, 4.0])
    degree = 90
    time_step = 2
    dt = 0.1
    output_rad = False
    result = unit_step(tp, degree, time_step, dt, output_rad)
    assert np.all(result == 90 * (tp >= time_step))

    # Проверка работы функции при пограничных входных данных
    tp = np.array([0.5, 1.5, 2.5, 3.5])
    degree = 180
    time_step = 1
    dt = 0.05
    output_rad = True
    result = unit_step(tp, degree, time_step, dt, output_rad)
    assert np.all(result == np.deg2rad(180) * (tp >= time_step))

    # Проверка работы функции при отрицательных входных данных
    tp = np.array([-1.0, -2.0, -3.0, -4.0])
    degree = -90
    time_step = -2
    dt = -0.1
    output_rad = False
    result = unit_step(tp, degree, time_step, dt, output_rad)
    assert np.all(result == -90 * (tp >= time_step))

    # Проверка работы функции при очень больших входных данных
    tp = np.array([100.0, 200.0, 300.0, 400.0])
    degree = 360
    time_step = 200
    dt = 0.2
    output_rad = True
    result = unit_step(tp, degree, time_step, dt, output_rad)
    assert np.all(result == np.deg2rad(360) * (tp >= time_step))

    # Проверка работы функции при отрицательных входных данных
    tp = np.array([-100.0, -200.0, -300.0, -400.0])
    degree = -360
    time_step = -200
    dt = -0.2
    output_rad = True
    result = unit_step(tp, degree, time_step, dt, output_rad)
    assert np.all(result == np.deg2rad(-360) * (tp >= time_step))

    # Проверка работы функции при очень больших входных данных
    tp = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    degree = 720
    time_step = 2000
    dt = 0.2
    output_rad = True
    result = unit_step(tp, degree, time_step, dt, output_rad)
    assert np.all(result == np.deg2rad(720) * (tp >= time_step))


from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest


def test_sinusoid():
    # Проверка работы функции при нормальных входных данных
    tp = np.array([1.0, 2.0, 3.0, 4.0])
    frequency = 1.0
    amplitude = 1.0
    result = sinusoid(tp, frequency, amplitude)
    assert np.all(result == np.sin(tp * amplitude) * frequency)

    # Проверка работы функции при пограничных входных данных
    tp = np.array([0.5, 1.5, 2.5, 3.5])
    frequency = 0.5
    amplitude = 0.5
    result = sinusoid(tp, frequency, amplitude)
    assert np.all(result == np.sin(tp * amplitude) * frequency)

    # Проверка работы функции при отрицательных входных данных
    tp = np.array([-1.0, -2.0, -3.0, -4.0])
    frequency = -1.0
    amplitude = -1.0
    result = sinusoid(tp, frequency, amplitude)
    assert np.all(result == np.sin(tp * amplitude) * frequency)

    # Проверка работы функции при очень больших входных данных
    tp = np.array([100.0, 200.0, 300.0, 400.0])
    frequency = 100.0
    amplitude = 100.0
    result = sinusoid(tp, frequency, amplitude)
    assert np.all(result == np.sin(tp * amplitude) * frequency)

    # Проверка работы функции при нулевых входных данных
    tp = np.array([0.0, 0.0, 0.0, 0.0])
    frequency = 0.0
    amplitude = 0.0
    result = sinusoid(tp, frequency, amplitude)
    assert np.all(result == np.sin(tp * amplitude) * frequency)

    # Проверка работы функции при отрицательных входных данных
    tp = np.array([-100.0, -200.0, -300.0, -400.0])
    frequency = -100.0
    amplitude = -100.0
    result = sinusoid(tp, frequency, amplitude)
    assert np.all(result == np.sin(tp * amplitude) * frequency)

    # Проверка работы функции при очень больших входных данных
    tp = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    frequency = 1000.0
    amplitude = 1000.0
    result = sinusoid(tp, frequency, amplitude)
    assert np.all(result == np.sin(tp * amplitude) * frequency)

    # Проверка работы функции при нулевых входных данных
    tp = np.array([0.0, 0.0, 0.0, 0.0])
    frequency = 0.0
    amplitude = 0.0
    result = sinusoid(tp, frequency, amplitude)
    assert np.all(result == np.sin(tp * amplitude) * frequency)


from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest


def test_constant_line():
    # Проверка работы функции при нормальных входных данных
    tp = np.array([1.0, 2.0, 3.0, 4.0])
    value_state = 2
    result = constant_line(tp, value_state)
    assert np.all(result == np.full_like(tp, value_state))

    # Проверка работы функции при пограничных входных данных
    tp = np.array([0.5, 1.5, 2.5, 3.5])
    value_state = 0.5
    result = constant_line(tp, value_state)
    assert np.all(result == np.full_like(tp, value_state))

    # Проверка работы функции при отрицательных входных данных
    tp = np.array([-1.0, -2.0, -3.0, -4.0])
    value_state = -1
    result = constant_line(tp, value_state)
    assert np.all(result == np.full_like(tp, value_state))

    # Проверка работы функции при очень больших входных данных
    tp = np.array([100.0, 200.0, 300.0, 400.0])
    value_state = 100
    result = constant_line(tp, value_state)
    assert np.all(result == np.full_like(tp, value_state))

    # Проверка работы функции при нулевых входных данных
    tp = np.array([0.0, 0.0, 0.0, 0.0])
    value_state = 0
    result = constant_line(tp, value_state)
    assert np.all(result == np.full_like(tp, value_state))

    # Проверка работы функции при отрицательных входных данных
    tp = np.array([-100.0, -200.0, -300.0, -400.0])
    value_state = -100
    result = constant_line(tp, value_state)
    assert np.all(result == np.full_like(tp, value_state))

    # Проверка работы функции при очень больших входных данных
    tp = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    value_state = 1000
    result = constant_line(tp, value_state)
    assert np.all(result == np.full_like(tp, value_state))

    # Проверка работы функции при нулевых входных данных
    tp = np.array([0.0, 0.0, 0.0, 0.0])
    value_state = 0
    result = constant_line(tp, value_state)
    assert np.all(result == np.full_like(tp, value_state))


from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest


def test_sinusoid_vertical_shift():
    # Проверка работы функции при нормальных входных данных
    tp = np.array([1.0, 2.0, 3.0, 4.0])
    frequency = 1.0
    amplitude = 1.0
    vertical_shift = 0.5
    result = sinusoid_vertical_shift(tp, frequency, amplitude, vertical_shift)
    assert np.all(
        result == amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift
    )

    # Проверка работы функции при пограничных входных данных
    tp = np.array([0.5, 1.5, 2.5, 3.5])
    frequency = 0.5
    amplitude = 0.5
    vertical_shift = 0.25
    result = sinusoid_vertical_shift(tp, frequency, amplitude, vertical_shift)
    assert np.all(
        result == amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift
    )

    # Проверка работы функции при отрицательных входных данных
    tp = np.array([-1.0, -2.0, -3.0, -4.0])
    frequency = -1.0
    amplitude = -1.0
    vertical_shift = -0.5
    result = sinusoid_vertical_shift(tp, frequency, amplitude, vertical_shift)
    assert np.all(
        result == amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift
    )

    # Проверка работы функции при очень больших входных данных
    tp = np.array([100.0, 200.0, 300.0, 400.0])
    frequency = 100.0
    amplitude = 100.0
    vertical_shift = 50.0
    result = sinusoid_vertical_shift(tp, frequency, amplitude, vertical_shift)
    assert np.all(
        result == amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift
    )

    # Проверка работы функции при нулевых входных данных
    tp = np.array([0.0, 0.0, 0.0, 0.0])
    frequency = 0.0
    amplitude = 0.0
    vertical_shift = 0.0
    result = sinusoid_vertical_shift(tp, frequency, amplitude, vertical_shift)
    assert np.all(
        result == amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift
    )

    # Проверка работы функции при отрицательных входных данных
    tp = np.array([-100.0, -200.0, -300.0, -400.0])
    frequency = -100.0
    amplitude = -100.0
    vertical_shift = -50.0
    result = sinusoid_vertical_shift(tp, frequency, amplitude, vertical_shift)
    assert np.all(
        result == amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift
    )

    # Проверка работы функции при очень больших входных данных
    tp = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    frequency = 1000.0
    amplitude = 1000.0
    vertical_shift = 500.0
    result = sinusoid_vertical_shift(tp, frequency, amplitude, vertical_shift)
    assert np.all(
        result == amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift
    )

    # Проверка работы функции при нулевых входных данных
    tp = np.array([0.0, 0.0, 0.0, 0.0])
    frequency = 0.0
    amplitude = 0.0
    vertical_shift = 0.0
    result = sinusoid_vertical_shift(tp, frequency, amplitude, vertical_shift)
    assert np.all(
        result == amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift
    )
