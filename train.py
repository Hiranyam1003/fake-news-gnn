"""Train and evaluate one configuration.

Examples:
  python train.py --dataset twitter15 --mode joint --stage prototype
  python train.py --all --stage beta          # every dataset x mode
"""
import argparse
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from config import DATASETS, MODES, SEED, STAGE_CONFIG, run_dir
from data import cv_splits, holdout_split, load_trees, truncate
from evaluate import aggregate, compute_metrics, early_detection, faithfulness, plot_early, predict_trees
from features import FeatureScaler, TextEncoder, structural_features, tree_to_data
from models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_truncation(tree, rng):
    """Training-time augmentation for early detection: half the time keep only an early prefix."""
    if tree.num_nodes <= 3 or rng.random() < 0.5:
        return tree
    k = int(np.exp(rng.uniform(np.log(3), np.log(tree.num_nodes))))
    return truncate(tree, k=k)


def fit(model, train_trees, train_vecs, val_data, scaler, cfg, device, log=print):
    rng = np.random.default_rng(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    static_train = None if cfg["truncate_train"] else [tree_to_data(t, v, scaler) for t, v in zip(train_trees, train_vecs)]
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(cfg["epochs"]):
        train_data = static_train or [
            tree_to_data(random_truncation(t, rng), v, scaler) for t, v in zip(train_trees, train_vecs)
        ]
        model.train()
        for batch in DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True):
            batch = batch.to(device)
            opt.zero_grad()
            F.cross_entropy(model(batch), batch.y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                F.cross_entropy(model(b.to(device)), b.y, reduction="sum").item()
                for b in DataLoader(val_data, batch_size=128)
            ) / len(val_data)
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    log(f"    stopped after {epoch + 1} epochs, best val loss {best_loss:.4f}")
    return model


def run_split(trees, split, mode, cfg, device, log=print):
    train_idx, val_idx, test_idx = split
    pick = lambda idx: [trees[i] for i in idx]
    train, val, test = pick(train_idx), pick(val_idx), pick(test_idx)

    encoder = TextEncoder(cfg["text_encoder"]).fit([t.text for t in train])
    vec = lambda ts: encoder.transform([t.text for t in ts])
    train_vecs, val_vecs, test_vecs = vec(train), vec(val), vec(test)
    scaler = FeatureScaler().fit(train)

    node_dim = structural_features(train[0]).shape[1]
    model = build_model(mode, node_dim, encoder.dim, cfg).to(device)
    val_data = [tree_to_data(t, v, scaler) for t, v in zip(val, val_vecs)]
    fit(model, train, train_vecs, val_data, scaler, cfg, device, log)

    y = np.array([t.label for t in test])
    proba, _ = predict_trees(model, test, test_vecs, scaler, device)
    metrics = dict(test=compute_metrics(y, proba))
    metrics["early"] = early_detection(model, test, test_vecs, scaler, cfg["early_k"], cfg["early_hours"], device)
    if mode != "content":
        metrics["faithfulness"] = faithfulness(model, test, test_vecs, scaler, device)
    bundle = dict(model=model.cpu(), encoder=encoder, scaler=scaler, mode=mode, cfg=cfg,
                  node_dim=node_dim, text_dim=encoder.dim, test_ids=[t.source_id for t in test])
    return bundle, metrics


def measure_latency(bundle, trees, k=10):
    """Mean wall-clock ms to predict one claim from a 10-node partial tree (NFR1: < 2000 ms)."""
    times = []
    for t in trees[:50]:
        start = time.perf_counter()
        vec = bundle["encoder"].transform([t.text])
        predict_trees(bundle["model"], [truncate(t, k=k)], vec, bundle["scaler"])
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times))


def run(dataset, mode, stage, device):
    set_seed(SEED)
    cfg = STAGE_CONFIG[stage]
    trees = load_trees(dataset)
    labels = np.array([t.label for t in trees])
    out = run_dir(stage, dataset, mode)
    out.mkdir(parents=True, exist_ok=True)
    print(f"== {stage} | {dataset} | {mode} ({len(trees)} trees)")

    fold_metrics = []
    if cfg["folds"] > 1:
        for i, split in enumerate(cv_splits(labels, cfg["folds"])):
            print(f"  fold {i + 1}/{cfg['folds']}")
            _, m = run_split(trees, split, mode, cfg, device)
            fold_metrics.append(m)
            print(f"    acc {m['test']['acc']:.3f}  macro-F1 {m['test']['macro_f1']:.3f}")

    # The checkpoint used by predict.py / the app is always trained on the fixed 70/10/20 split,
    # so its test claims are never seen in training.
    print("  holdout split (saved model)")
    set_seed(SEED)
    bundle, holdout_metrics = run_split(trees, holdout_split(labels), mode, cfg, device)
    print(f"    acc {holdout_metrics['test']['acc']:.3f}  macro-F1 {holdout_metrics['test']['macro_f1']:.3f}")
    if not fold_metrics:
        fold_metrics = [holdout_metrics]

    result = dict(stage=stage, dataset=dataset, mode=mode, folds=len(fold_metrics), **aggregate(fold_metrics))
    result["holdout"] = holdout_metrics["test"]
    result["latency_ms"] = measure_latency(bundle, [trees[i] for i in holdout_split(labels)[2]])
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    checkpoint = {k: v for k, v in bundle.items() if k != "model"}
    torch.save({**checkpoint, "state_dict": bundle["model"].state_dict()}, out / "model.pt")
    plot_early(result["early"], f"{dataset} {mode} ({stage})", out / "early_detection.png")
    print(f"  saved {out}  (latency {result['latency_ms']:.0f} ms/claim)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--all", action="store_true", help="run every dataset x mode")
    parser.add_argument("--stage", choices=list(STAGE_CONFIG), default="prototype")
    parser.add_argument("--device", default="cpu", help="cpu or mps")
    args = parser.parse_args()
    if not args.all and not (args.dataset and args.mode):
        parser.error("give --dataset and --mode, or --all")
    jobs = [(d, m) for d in DATASETS for m in MODES] if args.all else [(args.dataset, args.mode)]
    for d, m in jobs:
        run(d, m, args.stage, args.device)
