"""Turn trees into model inputs: structural node features, source-text vectors, PyG graphs."""
import pickle

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data

from config import MINILM_MODEL, PROCESSED_DIR

NODE_FEATURES = ["is_root", "depth", "delay", "is_retweet", "children", "arrival_rank", "repeat_user"]


def structural_features(tree):
    n = tree.num_nodes
    depth = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        depth[i] = depth[tree.parent[i]] + 1
    children = np.bincount(tree.parent[1:], minlength=n).astype(np.float32)
    seen, repeat = set(), np.zeros(n, dtype=np.float32)
    for i, u in enumerate(tree.uid):
        repeat[i] = u in seen
        seen.add(u)
    is_root = np.zeros(n, dtype=np.float32)
    is_root[0] = 1
    is_retweet = tree.is_retweet.astype(np.float32)
    is_retweet[0] = 0
    return np.stack([
        is_root,
        np.log1p(depth),
        np.log1p(tree.delay),
        is_retweet,
        np.log1p(children),
        np.log1p(np.arange(n, dtype=np.float32)),
        repeat,
    ], axis=1)


class FeatureScaler:
    def fit(self, trees):
        feats = np.concatenate([structural_features(t) for t in trees])
        self.mean = feats.mean(0)
        self.std = feats.std(0) + 1e-6
        return self

    def transform(self, feats):
        return (feats - self.mean) / self.std


class TextEncoder:
    """'tfidf' (prototype, fitted on training texts) or 'minilm' (beta, frozen sentence encoder)."""

    def __init__(self, kind):
        self.kind = kind
        self._model = None

    def fit(self, texts):
        if self.kind == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
            self.vectorizer.fit(texts)
        return self

    @property
    def dim(self):
        return len(self.vectorizer.vocabulary_) if self.kind == "tfidf" else 384

    def transform(self, texts):
        if self.kind == "tfidf":
            return self.vectorizer.transform(texts).toarray().astype(np.float32)
        return self._minilm(texts)

    def _minilm(self, texts):
        cache_path = PROCESSED_DIR / "minilm_cache.pkl"
        cache = pickle.load(open(cache_path, "rb")) if cache_path.exists() else {}
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(MINILM_MODEL, device="cpu")
            vecs = self._model.encode(missing, batch_size=64, normalize_embeddings=True)
            cache.update(zip(missing, vecs.astype(np.float32)))
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            pickle.dump(cache, open(cache_path, "wb"))
        return np.stack([cache[t] for t in texts])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state


def tree_to_data(tree, text_vec, scaler):
    x = torch.from_numpy(scaler.transform(structural_features(tree)).astype(np.float32))
    child = torch.arange(1, tree.num_nodes)
    parent = torch.from_numpy(tree.parent[1:])
    edge_index = torch.stack([torch.cat([parent, child]), torch.cat([child, parent])])
    return Data(
        x=x,
        edge_index=edge_index,
        text=torch.as_tensor(text_vec, dtype=torch.float32).view(1, -1),
        y=torch.tensor([max(tree.label, 0)]),
        num_nodes=tree.num_nodes,
    )
