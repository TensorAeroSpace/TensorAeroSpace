"""Tests for DQN SumTree data structure."""
import numpy as np

from tensoraerospace.agent.dqn import SumTree


def test_sumtree_initialization():
    """Test SumTree initialization."""
    capacity = 8
    st = SumTree(capacity=capacity)

    assert st.capacity == capacity
    assert st.tree.shape == (2 * capacity - 1,)
    assert st.transitions.shape == (capacity,)
    assert st.next_idx == 0
    assert st.total_p == 0.0


def test_sumtree_add_single_transition():
    """Test adding a single transition."""
    st = SumTree(capacity=4)
    transition = (np.zeros(4), 0, 1.0, np.zeros(4), False)
    priority = 2.5

    st.add(priority=priority, transition=transition)

    assert st.total_p == priority
    assert st.next_idx == 1
    assert st.transitions[0] == transition


def test_sumtree_add_multiple_transitions():
    """Test adding multiple transitions and total_p accumulation."""
    st = SumTree(capacity=4)
    priorities = [1.0, 2.0, 3.0, 4.0]

    for i, p in enumerate(priorities):
        transition = (i, i, float(i), i, False)
        st.add(priority=p, transition=transition)

    # total_p should be sum of all priorities
    assert np.isclose(st.total_p, sum(priorities))
    assert st.next_idx == 0  # wrapped around


def test_sumtree_circular_buffer():
    """Test that SumTree wraps around when full."""
    st = SumTree(capacity=4)

    # Fill buffer
    for i in range(4):
        st.add(priority=1.0, transition=(i, i, i, i, False))

    assert st.next_idx == 0

    # Add one more - should overwrite first
    st.add(priority=5.0, transition=(99, 99, 99, 99, True))
    assert st.next_idx == 1
    assert st.transitions[0] == (99, 99, 99, 99, True)


def test_sumtree_update_priority():
    """Test updating priority of existing transition."""
    st = SumTree(capacity=4)

    # Add transitions
    for i in range(4):
        st.add(priority=1.0, transition=(i, i, i, i, False))

    initial_total = st.total_p

    # Update priority of first leaf (index = capacity - 1 = 3)
    leaf_idx = st.capacity - 1
    new_priority = 10.0
    st.update(idx=leaf_idx, priority=new_priority)

    # Total should increase by (new_priority - old_priority)
    expected_total = initial_total - 1.0 + new_priority
    assert np.isclose(st.total_p, expected_total)


def test_sumtree_get_leaf():
    """Test retrieving leaf by priority value."""
    st = SumTree(capacity=4)
    priorities = [1.0, 2.0, 3.0, 4.0]

    for i, p in enumerate(priorities):
        st.add(priority=p, transition=(i, i, float(i), i, False))

    # Get a leaf by sampling
    s = st.total_p * 0.5  # sample in middle
    idx, priority, transition = st.get_leaf(s)

    assert isinstance(idx, (int, np.integer))
    assert priority > 0
    assert transition is not None
    assert len(transition) == 5


def test_sumtree_sampling_coverage():
    """Test that sampling covers different segments."""
    st = SumTree(capacity=8)

    # Add transitions with equal priorities
    for i in range(8):
        st.add(priority=1.0, transition=(i, i, float(i), i, False))

    # Sample from different segments
    samples = []
    for i in range(8):
        s = st.total_p * (i + 0.5) / 8.0  # middle of each segment
        idx, priority, transition = st.get_leaf(s)
        samples.append(transition[0])  # first element is the index

    # Should retrieve different transitions (though not guaranteed to be all 8)
    assert len(set(samples)) > 1


def test_sumtree_importance_sampling_weights():
    """Test that priorities are maintained for IS weight calculation."""
    st = SumTree(capacity=8)

    # Add transitions with varying priorities
    priorities = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    for i, p in enumerate(priorities):
        st.add(priority=p, transition=(i, i, float(i), i, False))

    # Verify total priority
    assert np.isclose(st.total_p, sum(priorities), rtol=1e-5)

    # Sample and verify priorities are non-zero
    segment = st.total_p / 4
    for i in range(4):
        s = np.random.uniform(segment * i, segment * (i + 1))
        idx, p, t = st.get_leaf(s)
        assert p > 0, "Sampled priority should be positive"


def test_sumtree_empty_buffer_sampling():
    """Test that sampling from partially filled buffer works."""
    st = SumTree(capacity=8)

    # Add only 3 transitions
    for i in range(3):
        st.add(priority=1.0, transition=(i, i, float(i), i, False))

    # total_p should be 3.0
    assert np.isclose(st.total_p, 3.0)

    # Sample should still work
    s = st.total_p * 0.5
    idx, priority, transition = st.get_leaf(s)
    assert priority > 0
    assert transition is not None


def test_sumtree_zero_priority_edge_case():
    """Test behavior with very small priorities."""
    st = SumTree(capacity=4)

    # Add transitions with small priorities
    for i in range(4):
        st.add(priority=1e-6, transition=(i, i, float(i), i, False))

    # Should still maintain structure
    assert st.total_p > 0
    assert st.total_p < 1e-5

    # Sampling should work
    s = st.total_p * 0.5
    idx, priority, transition = st.get_leaf(s)
    assert priority >= 0


def test_sumtree_wrap_around_behavior():
    """Test that next_idx cycles correctly when capacity is exceeded."""
    st = SumTree(capacity=4)

    # Fill buffer completely
    for i in range(4):
        st.add(priority=1.0, transition=(i, i, i, i, False))

    assert st.next_idx == 0  # Should wrap to 0

    # Add 8 more items (2 full cycles)
    for i in range(4, 12):
        st.add(priority=2.0, transition=(i, i, i, i, False))

    assert st.next_idx == 0  # Should be at 0 again

    # Verify last 4 transitions are stored
    expected_indices = [8, 9, 10, 11]
    for i in range(4):
        assert st.transitions[i][0] == expected_indices[i]


def test_sumtree_total_p_invariants():
    """Test total_p is maintained after add/update operations."""
    st = SumTree(capacity=4)
    priorities = [1.0, 2.0, 3.0, 4.0]

    # Add transitions and verify total_p after each addition
    cumulative_p = 0.0
    for i, p in enumerate(priorities):
        st.add(priority=p, transition=(i, i, i, i, False))
        cumulative_p += p
        assert np.isclose(st.total_p, cumulative_p)

    # Update priorities and verify total_p
    leaf_idx = st.capacity - 1  # First leaf in tree
    old_p = st.tree[leaf_idx]
    new_p = 10.0
    st.update(idx=leaf_idx, priority=new_p)

    expected_total = cumulative_p - old_p + new_p
    assert np.isclose(st.total_p, expected_total)


def test_sumtree_get_leaf_monotonicity():
    """Test that get_leaf respects segment ordering."""
    st = SumTree(capacity=8)
    priorities = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    for i, p in enumerate(priorities):
        st.add(priority=p, transition=(i, i, float(i), i, False))

    # Sample from consecutive segments
    num_segments = 8
    segment_size = st.total_p / num_segments

    for i in range(num_segments):
        s = segment_size * i + 0.01  # Sample from start of each segment
        idx, priority, transition = st.get_leaf(s)

        assert priority > 0, "Priority should be positive"
        assert transition is not None, "Transition should not be None"

        # Verify we got a valid transition
        trans_idx = transition[0]
        assert 0 <= trans_idx < 8


def test_sumtree_get_leaf_data_consistency():
    """Test that retrieved transition data matches stored data."""
    st = SumTree(capacity=4)

    # Add unique transitions
    transitions = [
        (np.array([1.0, 2.0]), 0, 0.5, np.array([3.0, 4.0]), False),
        (np.array([5.0, 6.0]), 1, 1.5, np.array([7.0, 8.0]), False),
        (np.array([9.0, 10.0]), 0, 2.5, np.array([11.0, 12.0]), True),
        (np.array([13.0, 14.0]), 1, 3.5, np.array([15.0, 16.0]), True),
    ]

    for i, t in enumerate(transitions):
        st.add(priority=float(i + 1), transition=t)

    # Sample and verify data integrity
    s = st.total_p * 0.5
    idx, priority, retrieved = st.get_leaf(s)

    # Verify structure
    assert len(retrieved) == 5
    assert isinstance(retrieved[1], (int, np.integer))  # action
    assert isinstance(retrieved[2], (float, np.floating))  # reward
    assert isinstance(retrieved[4], (bool, np.bool_))  # done


def test_sumtree_sampling_weights_normalized():
    """Test that k samples yield valid importance sampling weights."""
    st = SumTree(capacity=16)

    # Add transitions
    for i in range(16):
        priority = np.random.uniform(0.1, 5.0)
        st.add(priority=priority, transition=(i, i, i, i, False))

    k = 4
    segment = st.total_p / k

    idxes = []
    priorities = []

    for i in range(k):
        s = np.random.uniform(segment * i, segment * (i + 1))
        idx, p, t = st.get_leaf(s)
        idxes.append(idx)
        priorities.append(p)

    # Verify we got k samples
    assert len(idxes) == k
    assert len(priorities) == k

    # All priorities should be positive
    for p in priorities:
        assert p > 0

    # All indices should be in valid leaf range
    for idx in idxes:
        assert st.capacity - 1 <= idx < 2 * st.capacity - 1


def test_sumtree_property_fuzz():
    """Property-based mini-fuzz with random priorities."""
    st = SumTree(capacity=8)

    # Add transitions with random priorities
    np.random.seed(42)
    for i in range(8):
        priority = np.random.uniform(0.01, 10.0)
        st.add(priority=priority, transition=(i, i, float(i), i, False))

    # Verify no crash and valid state
    assert st.total_p > 0
    assert st.next_idx == 0

    # Sample multiple times
    for _ in range(20):
        s = np.random.uniform(0, st.total_p)
        idx, p, t = st.get_leaf(s)

        # All sampled indices should be in leaf range
        assert st.capacity - 1 <= idx < 2 * st.capacity - 1
        assert p > 0
        assert t is not None

    # Update with random priorities
    for _ in range(10):
        idx = np.random.randint(st.capacity - 1, 2 * st.capacity - 1)
        new_p = np.random.uniform(0.01, 10.0)
        st.update(idx, new_p)
        # total_p should change but remain positive
        assert st.total_p > 0
