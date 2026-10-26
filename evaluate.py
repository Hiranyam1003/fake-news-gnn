"""Metrics, early-detection curves, explanation faithfulness, and the cross-run results table.

Run `python evaluate.py --stage prototype` to print/save the results table for a stage.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch

from config import DATASETS, LABELS, MODES, RUNS_DIR, SEED, VERDICT, run_dir
from data import remove_nodes, truncate
from features import tree_to_data


def predict_trees(model, trees, text_vecs, scaler, device="cpu", batch_size=64):
    """Return (probabilities [N, 4], list of per-node attention arrays or None)."""
    data = [tree_to_data(t, v, scaler) for t, v in zip(trees, text_vecs)]
    model.eval()
    probs, atts = [], []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=batch_size):
            batch = batch.to(device)
            logits, att = model(batch, return_attention=True)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            if att is not None:
                atts.extend(a.cpu().numpy() for a in unbatch(att, batch.batch))
    return np.concatenate(probs), (atts or None)


def compute_metrics(y, proba):
    pred = proba.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(len(LABELS)), zero_division=0)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", labels=list(range(len(LABELS))))
    except ValueError:
        auc = float("nan")
    verdict = np.array([VERDICT[l] for l in LABELS])
    m = dict(
        acc=accuracy_score(y, pred),
        macro_f1=float(f.mean()),
        auc_ovr=float(auc),
        verdict_acc=accuracy_score(verdict[y], verdict[pred]),
    )
    for i, label in enumerate(LABELS):
        m[f"f1_{label}"], m[f"precision_{label}"], m[f"recall_{label}"] = float(f[i]), float(p[i]), float(r[i])
    m["confusion"] = confusion_matrix(y, pred, labels=range(len(LABELS))).tolist()
    return m


def early_detection(model, trees, text_vecs, scaler, ks, hours, device="cpu"):
    y = np.array([t.label for t in trees])
    out = {}
    for k in ks:
        proba, _ = predict_trees(model, [truncate(t, k=k) for t in trees], text_vecs, scaler, device)
        m = compute_metrics(y, proba)
        out[f"k={k}" if k else "full"] = dict(acc=m["acc"], macro_f1=m["macro_f1"])
    for h in hours:
        proba, _ = predict_trees(model, [truncate(t, hours=h) for t in trees], text_vecs, scaler, device)
        m = compute_metrics(y, proba)
        out[f"{h}h"] = dict(acc=m["acc"], macro_f1=m["macro_f1"])
    return out


def faithfulness(model, trees, text_vecs, scaler, device="cpu", frac=0.1):
    """Drop in predicted-class probability when removing the top-attended nodes vs. random nodes.

    If attention reflects what the model actually uses, removing top nodes should hurt more.
    """
    rng = np.random.default_rng(SEED)
    proba, atts = predict_trees(model, trees, text_vecs, scaler, device)
    if atts is None:
        return None
    pred = proba.argmax(1)
    top_trees, rand_trees = [], []
    for t, a in zip(trees, atts):
        m = max(1, round(frac * (t.num_nodes - 1)))
        candidates = np.arange(1, t.num_nodes)
        top = candidates[np.argsort(-a[1:])[:m]]
        top_trees.append(remove_nodes(t, top))
        rand_trees.append(remove_nodes(t, rng.choice(candidates, size=min(m, len(candidates)), replace=False)))
    p_top, _ = predict_trees(model, top_trees, text_vecs, scaler, device)
    p_rand, _ = predict_trees(model, rand_trees, text_vecs, scaler, device)
    base = proba[np.arange(len(pred)), pred]
    drop_top = base - p_top[np.arange(len(pred)), pred]
    drop_rand = base - p_rand[np.arange(len(pred)), pred]
    return dict(
        removed_fraction=frac,
        mean_drop_top=float(drop_top.mean()),
        mean_drop_random=float(drop_rand.mean()),
        share_top_hurts_more=float((drop_top > drop_rand).mean()),
    )


def aggregate(fold_metrics):
    """Mean/std of every scalar across folds; confusion matrices are summed."""
    out = {}
    for key, value in fold_metrics[0].items():
        values = [m[key] for m in fold_metrics]
        if key == "confusion":
            out[key] = np.sum(values, axis=0).tolist()
        elif isinstance(value, dict):
            out[key] = aggregate(values)
        else:
            out[key] = dict(mean=float(np.mean(values)), std=float(np.std(values)))
    return out


def plot_early(early, title, path):
    keys = list(early)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    means = [early[k]["acc"]["mean"] for k in keys]
    stds = [early[k]["acc"]["std"] for k in keys]
    ax.errorbar(range(len(keys)), means, yerr=stds, marker="o", capsize=3)
    ax.set_xticks(range(len(keys)), keys)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(stage):
    rows = []
    for dataset in DATASETS:
        for mode in MODES:
            path = run_dir(stage, dataset, mode) / "metrics.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text())
            t = m["test"]
            row = dict(dataset=dataset, mode=mode)
            for key in ["acc", "macro_f1", "auc_ovr", "verdict_acc"] + [f"f1_{l}" for l in LABELS]:
                row[key] = f"{t[key]['mean']:.3f} ± {t[key]['std']:.3f}" if m["folds"] > 1 else f"{t[key]['mean']:.3f}"
            for cutoff, v in m.get("early", {}).items():
                row[f"acc@{cutoff}"] = f"{v['acc']['mean']:.3f}"
            rows.append(row)
    if not rows:
        raise SystemExit(f"No runs found for stage '{stage}'. Train first.")
    df = pd.DataFrame(rows)
    out = RUNS_DIR / stage / "results.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prototype", "beta"], required=True)
    summary(parser.parse_args().stage)
