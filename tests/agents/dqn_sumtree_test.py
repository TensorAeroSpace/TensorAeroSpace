import numpy as np

from tensoraerospace.agent.dqn.model import SumTree


def test_sumtree_add_update_and_get_leaf():
    st = SumTree(capacity=4)
    # Add transitions with priorities
    for i in range(4):
        st.add(priority=float(i + 1), transition=(i, i, i, i, False))

    assert st.total_p > 0

    # Update a node's priority
    idx, _, _ = st.get_leaf(s=st.total_p * 0.25)
    st.update(idx, priority=10.0)
    assert st.total_p >= 10.0

    # Retrieve another leaf
    idx2, p2, t2 = st.get_leaf(s=st.total_p * 0.75)
    assert isinstance(idx2, (int, np.integer))
    assert p2 > 0
    assert len(t2) == 5
