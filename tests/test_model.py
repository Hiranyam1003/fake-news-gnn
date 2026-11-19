import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from config import LABELS, STAGE_CONFIG
from data import parse_tree, truncate
from evaluate import compute_metrics
from explain import explain, text_evidence
from features import FeatureScaler, tree_to_data
from models import build_model

LINES = [
    "['ROOT', 'ROOT', '0.0']->['u0', 'S', '0.0']",
    "['u0', 'S', '0.0']->['u1', 'S', '1.0']",
    "['u0', 'S', '0.0']->['u2', 'r2', '2.0']",
    "['u2', 'r2', '2.0']->['u3', 'S', '3.0']",
]
TEXT_DIM = 8


@pytest.fixture
def batch():
    trees = [parse_tree(LINES, "S", "a", 0), truncate(parse_tree(LINES, "S", "b", 1), k=1)]
    scaler = FeatureScaler().fit(trees)
    data = [tree_to_data(t, np.random.rand(TEXT_DIM), scaler) for t in trees]
    return trees, next(iter(DataLoader(data, batch_size=2)))


@pytest.mark.parametrize("mode", ["content", "structure", "joint"])
@pytest.mark.parametrize("root_enhance", [True, False])
def test_forward_shapes_and_probabilities(batch, mode, root_enhance):
    _, b = batch
    cfg = {**STAGE_CONFIG["prototype"], "root_enhance": root_enhance}
    model = build_model(mode, b.x.size(1), TEXT_DIM, cfg).eval()
    logits, attention = model(b, return_attention=True)
    assert logits.shape == (2, len(LABELS))
    assert torch.allclose(torch.softmax(logits, 1).sum(1), torch.ones(2))
    if mode == "content":
        assert attention is None
    else:
        assert attention.shape == (b.num_nodes,)
        per_graph = torch.zeros(2).index_add_(0, b.batch, attention)
        assert torch.allclose(per_graph, torch.ones(2))  # attention is a distribution per tree


def test_explanation_never_empty(batch):
    trees, b = batch
    model = build_model("joint", b.x.size(1), TEXT_DIM, STAGE_CONFIG["prototype"]).eval()
    _, attention = model(b, return_attention=True)
    nodes, why = explain(trees[0], attention[b.batch == 0].detach().numpy())
    assert nodes and why
    assert explain(trees[0], None)[1]  # content-only model still gets a sentence


def test_metrics_perfect_predictions():
    y = np.array([0, 1, 2, 3])
    m = compute_metrics(y, np.eye(4))
    assert m["acc"] == 1 and m["macro_f1"] == 1 and m["verdict_acc"] == 1


def test_text_evidence_names_supporting_terms(batch):
    _, b = batch
    model = build_model("joint", b.x.size(1), TEXT_DIM, STAGE_CONFIG["beta"])
    data = b.get_example(0)
    data.batch = None
    names = [f"w{i}" for i in range(TEXT_DIM)]
    words = text_evidence(model, data, names, cls=0)
    assert all(w["term"] in names and w["weight"] > 0 for w in words)
    assert len(words) <= 5
