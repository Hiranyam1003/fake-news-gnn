"""Load the Ma et al. (2017) Twitter15/16 release into clean propagation trees.

Each tree is stored with nodes sorted so that every parent comes before its
children (root is node 0). That makes early-detection truncation a simple prefix.
"""
import ast
import pickle
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import DATA_URL, DATASET_DIR, LABELS, PROCESSED_DIR, RAW_DIR, SEED


@dataclass
class Tree:
    source_id: str
    text: str
    label: int  # index into LABELS, -1 if unknown
    parent: np.ndarray  # int, -1 for the root
    delay: np.ndarray  # minutes after the source post
    uid: np.ndarray  # user id strings
    is_retweet: np.ndarray  # bool; False means a reply (it has its own tweet id)

    @property
    def num_nodes(self):
        return len(self.parent)


def ensure_raw():
    if DATASET_DIR.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "rumdetect2017.zip"
    try:
        print(f"Downloading dataset to {zip_path} ...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
    except Exception as e:
        raise SystemExit(
            f"Download failed ({e}). Download rumdetect2017.zip manually from\n  {DATA_URL}\n"
            f"and place it in {RAW_DIR}, then re-run."
        )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)


def parse_tree(lines, source_id=None, text="", label=-1):
    """Build a Tree from the lines of a tree file.

    Cleans known quirks: self-loops, nodes with several parents (first edge wins),
    negative delays, and nodes not reachable from the root (re-attached to it).
    """
    root = None
    parent_of, delay_of = {}, {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        p, c = (ast.literal_eval(part) for part in line.split("->"))
        c_key = (c[0], c[1])
        if p[0] == "ROOT":
            root = c_key
            continue
        p_key = (p[0], p[1])
        if p_key == c_key or c_key in parent_of or c_key == root:
            continue
        parent_of[c_key] = p_key
        delay_of[c_key] = max(0.0, float(c[2]))
    if root is None:
        raise ValueError("tree has no ROOT line")
    if source_id is None:
        source_id = root[1]
    parent_of.pop(root, None)

    children = {}
    for c, p in parent_of.items():
        children.setdefault(p, []).append(c)

    # BFS from the root; delays are made non-decreasing along each path so that
    # sorting by delay always puts a parent before its children.
    delay_of[root] = 0.0
    bfs, seen, queue = [root], {root}, deque([root])
    while queue:
        node = queue.popleft()
        for c in children.get(node, []):
            if c not in seen:
                seen.add(c)
                delay_of[c] = max(delay_of[c], delay_of[node])
                bfs.append(c)
                queue.append(c)
    for c in parent_of:
        if c not in seen:  # orphan or part of a cycle
            parent_of[c] = root
            seen.add(c)
            bfs.append(c)

    order = sorted(range(len(bfs)), key=lambda i: (delay_of[bfs[i]], i))
    nodes = [bfs[i] for i in order]
    index = {n: i for i, n in enumerate(nodes)}
    parent = np.array([-1] + [index[parent_of[n]] for n in nodes[1:]], dtype=np.int64)
    return Tree(
        source_id=str(source_id),
        text=text,
        label=label,
        parent=parent,
        delay=np.array([delay_of[n] for n in nodes], dtype=np.float32),
        uid=np.array([n[0] for n in nodes]),
        is_retweet=np.array([n[1] == source_id for n in nodes]),
    )


def truncate(tree, k=None, hours=None):
    """Keep only the earliest part of a propagation: first k nodes and/or nodes within `hours`."""
    n = tree.num_nodes
    if hours is not None:
        n = min(n, int(np.searchsorted(tree.delay, hours * 60.0, side="right")))
    if k is not None:
        n = min(n, k)
    n = max(n, 1)
    return Tree(tree.source_id, tree.text, tree.label, tree.parent[:n], tree.delay[:n],
                tree.uid[:n], tree.is_retweet[:n])


def remove_nodes(tree, drop):
    """Remove nodes (never the root); their children attach to the nearest kept ancestor."""
    keep = np.ones(tree.num_nodes, dtype=bool)
    keep[list(drop)] = False
    keep[0] = True
    new_index = np.cumsum(keep) - 1
    parent = []
    for i in np.flatnonzero(keep)[1:]:
        p = tree.parent[i]
        while not keep[p]:
            p = tree.parent[p]
        parent.append(new_index[p])
    return Tree(tree.source_id, tree.text, tree.label, np.array([-1] + parent, dtype=np.int64),
                tree.delay[keep], tree.uid[keep], tree.is_retweet[keep])


def load_trees(dataset):
    cache = PROCESSED_DIR / f"{dataset}_trees.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    ensure_raw()
    folder = DATASET_DIR / dataset
    labels = {}
    for line in open(folder / "label.txt"):
        if line.strip():
            label, sid = line.strip().split(":")
            labels[sid] = label
    texts = dict(line.rstrip("\n").split("\t", 1) for line in open(folder / "source_tweets.txt", encoding="utf-8"))
    trees = []
    for sid in sorted(labels):
        with open(folder / "tree" / f"{sid}.txt") as f:
            trees.append(parse_tree(f, sid, texts[sid], LABELS.index(labels[sid])))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(trees, f)
    return trees


def holdout_split(labels, seed=SEED):
    """Stratified 70/10/20 train/val/test index split."""
    idx = np.arange(len(labels))
    train_val, test = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.125, stratify=labels[train_val], random_state=seed)
    return train, val, test


def cv_splits(labels, folds, seed=SEED):
    """Stratified k-fold; 10% of each training fold is held out for early stopping."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_val, test in skf.split(np.zeros(len(labels)), labels):
        train, val = train_test_split(train_val, test_size=0.1, stratify=labels[train_val], random_state=seed)
        yield train, val, test
