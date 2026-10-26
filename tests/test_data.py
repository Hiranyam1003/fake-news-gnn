import numpy as np
import pytest

from data import parse_tree, remove_nodes, truncate
from features import structural_features

# Covers the quirks seen in the real files: a reply arriving "before" its parent,
# a self-loop, a node with a second parent, and an orphan with a negative delay.
LINES = [
    "['ROOT', 'ROOT', '0.0']->['u0', 'S', '0.0']",
    "['u0', 'S', '0.0']->['u1', 'S', '2.0']",
    "['u0', 'S', '0.0']->['u2', 'r2', '1.0']",
    "['u2', 'r2', '1.0']->['u3', 'S', '0.5']",
    "['u1', 'S', '2.0']->['u1', 'S', '2.0']",
    "['u2', 'r2', '1.0']->['u1', 'S', '3.0']",
    "['ux', 'missing', '1.0']->['u4', 'S', '-1.0']",
]


@pytest.fixture
def tree():
    return parse_tree(LINES, source_id="S", text="claim", label=1)


def test_parse_cleans_quirks(tree):
    # sorted by delay: root, u4 (orphan, clipped to 0), u2, u3 (delay raised to its parent's), u1
    assert list(tree.uid) == ["u0", "u4", "u2", "u3", "u1"]
    assert tree.parent.tolist() == [-1, 0, 0, 2, 0]
    assert tree.delay.tolist() == [0.0, 0.0, 1.0, 1.0, 2.0]
    assert tree.is_retweet.tolist() == [True, True, False, True, True]


def test_parents_come_before_children(tree):
    assert all(tree.parent[i] < i for i in range(1, tree.num_nodes))


def test_parse_infers_source_id_from_root():
    assert parse_tree(LINES).source_id == "S"


def test_parse_requires_root():
    with pytest.raises(ValueError):
        parse_tree(LINES[1:])


def test_truncate_by_count_and_time(tree):
    assert truncate(tree, k=3).parent.tolist() == [-1, 0, 0]
    assert truncate(tree, hours=1 / 60).num_nodes == 4
    assert truncate(tree, hours=0).num_nodes == 2
    assert truncate(tree, k=0).num_nodes == 1


def test_remove_nodes_reattaches_children(tree):
    smaller = remove_nodes(tree, [2])
    assert list(smaller.uid) == ["u0", "u4", "u3", "u1"]
    assert smaller.parent.tolist() == [-1, 0, 0, 0]


def test_remove_nodes_never_drops_root(tree):
    assert remove_nodes(tree, [0]).num_nodes == tree.num_nodes


def test_structural_features(tree):
    f = structural_features(tree)
    assert f.shape == (5, 7)
    assert f[0, 0] == 1 and f[1:, 0].sum() == 0  # is_root
    assert np.isclose(f[3, 1], np.log1p(2))  # u3 sits at depth 2
    assert f[4, 6] == 0 and f[:, 6].sum() == 0  # no repeated users here
