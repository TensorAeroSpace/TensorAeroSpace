import numpy as np


def state2dict(state: list, list_state: list) -> dict:
    """Convert state array to state dictionary.

    Args:
        state: State array.
        list_state: List of state names.

    Returns:
        dict: State dictionary.
    """
    state = np.array(state).reshape([len(state), -1])
    return {st: state[:, list_state.index(st)] for i, st in enumerate(list_state)}


def control2dict(control: list, control_list: list) -> dict:
    """Convert control array to control dictionary.

    Args:
        control: Control array.
        control_list: List of control names.

    Returns:
        dict: Control dictionary.
    """
    control = np.array(control).reshape([len(control), -1])
    return {st: control[:, control_list.index(st)] for i, st in enumerate(control_list)}


def output2dict(output: np.ndarray, output_list: list) -> dict:
    """Convert state-space output array to output dictionary.

    Args:
        output: Output array.
        output_list: List of output names.

    Returns:
        dict: Output dictionary.
    """
    return {st: output[i] for i, st in enumerate(output_list)}
