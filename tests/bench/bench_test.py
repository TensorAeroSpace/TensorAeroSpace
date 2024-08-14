import pytest
import numpy as np
from unittest.mock import patch
from io import StringIO
from tensoraerospace.benchmark.function import find_longest_repeating_series

import pytest
import numpy as np
from unittest.mock import patch
from io import StringIO

def test_find_longest_repeating_series():
    # Проверка работы функции при нормальных входных данных
    numbers = [1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    result = find_longest_repeating_series(numbers)
    assert result == (1, 3)

    # Проверка работы функции при пограничных входных данных
    numbers = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    result = find_longest_repeating_series(numbers)
    assert result == (0, 1)

