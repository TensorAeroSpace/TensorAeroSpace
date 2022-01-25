import numpy as np


def state2dict(state: list, list_state: list) -> dict:
    """
    Конвертирование массива состояний в словрь состояний

    :param state: Массив состояний

    :return: Словарь состояний
    """
    state = np.array(state).reshape([len(state), -1])
    return {
        st: state[:, list_state.index(st)]
        for i, st in enumerate(list_state)
    }


def control2dict(control: list, control_list: list) -> dict:
    """
    Конвертирование массива управления в словарь управления

    Args:
        control: Массив управления

    Returns:
        Словарь управления
    """
    control = np.array(control).reshape([len(control), -1])
    return {
        st: control[:, control_list.index(st)]
        for i, st in enumerate(control_list)
    }


def output2dict(output: np.ndarray, output_list: list) -> dict:
    """
    Конвертирование массива выхода state-space в словарь управления

    Args:
        control: Массив управления

    Returns:
        Словарь управления
    """
    return {st: output[i] for i, st in enumerate(output_list)}
