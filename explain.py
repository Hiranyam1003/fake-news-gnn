"""Reviewer-readable explanations: attention over the propagation tree, and the words in the source text."""
import numpy as np


def _fmt_minutes(m):
    if m < 1:
        return "a minute"
    if m < 120:
        return f"{m:.0f} min"
    if m < 48 * 60:
        return f"{m / 60:.1f} h"
    return f"{m / 1440:.1f} days"


def top_nodes(tree, attention, k=5):
    depth = np.zeros(tree.num_nodes, dtype=int)
    for i in range(1, tree.num_nodes):
        depth[i] = depth[tree.parent[i]] + 1
    children = np.bincount(tree.parent[1:], minlength=tree.num_nodes)
    order = np.argsort(-attention)[:k]
    return [
        dict(
            node=int(i),
            attention=float(attention[i]),
            kind="source post" if i == 0 else ("retweet" if tree.is_retweet[i] else "reply"),
            depth=int(depth[i]),
            delay_min=float(tree.delay[i]),
            children=int(children[i]),
        )
        for i in order
    ]


def rationale(nodes, num_nodes):
    if not nodes:
        return "No propagation evidence available; prediction based on the source text only."
    share = sum(n["attention"] for n in nodes)
    parts = []
    if any(n["kind"] == "source post" for n in nodes):
        parts.append("the source post itself")
    others = [n for n in nodes if n["kind"] != "source post"]
    if others:
        retweets = sum(n["kind"] == "retweet" for n in others)
        replies = len(others) - retweets
        kinds = " and ".join(
            f"{c} {w}{'s' if c != 1 else ''}" for c, w in [(retweets, "retweet"), (replies, "reply")] if c
        ).replace("replys", "replies")
        latest = max(n["delay_min"] for n in others)
        max_depth = max(n["depth"] for n in others)
        hubs = sum(n["children"] >= 3 for n in others)
        desc = f"{kinds} arriving within {_fmt_minutes(latest)} of posting (up to depth {max_depth})"
        if hubs:
            desc += f", {hubs} of which re-spread the claim further"
        parts.append(desc)
    return (
        f"The model focused on {' and '.join(parts)}; these {len(nodes)} of {num_nodes} nodes "
        f"carry {share:.0%} of its attention."
    )


def explain(tree, attention, k=5):
    nodes = top_nodes(tree, attention, k) if attention is not None else []
    return nodes, rationale(nodes, tree.num_nodes)


def text_evidence(model, data, feature_names, cls, k=5):
    """Gradient x input over the TF-IDF vector: the terms that pushed the model toward class `cls`."""
    data = data.clone()
    data.text = data.text.clone().requires_grad_(True)
    model.eval()
    model(data)[0, cls].backward()
    contrib = (data.text.grad * data.text)[0].detach().numpy()
    return [dict(term=str(feature_names[i]), weight=float(contrib[i])) for i in np.argsort(-contrib)[:k] if contrib[i] > 0]
