"""Predict one claim with a trained model.

Example:
  python predict.py --dataset twitter16 --tweet-id 656955120626880512 --k 10
(omit --tweet-id to use the first held-out test claim)
"""
import argparse
import time

import torch

from config import LABELS, MODES, STAGES, VERDICT, run_dir
from data import load_trees, truncate
from evaluate import predict_trees
from explain import explain, text_evidence
from features import tree_to_data
from models import build_model


class Predictor:
    def __init__(self, dataset, mode="joint", stage="beta"):
        path = run_dir(stage, dataset, mode) / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run: python train.py --dataset {dataset} "
                                    f"--mode {mode} --stage {stage}")
        bundle = torch.load(path, weights_only=False)  # our own file: holds the sklearn/encoder objects
        self.model = build_model(mode, bundle["node_dim"], bundle["text_dim"], bundle["cfg"])
        self.model.load_state_dict(bundle["state_dict"])
        self.encoder, self.scaler = bundle["encoder"], bundle["scaler"]
        self.mode = mode
        self.test_ids = bundle["test_ids"]

    def predict(self, tree, k=None, hours=None):
        start = time.perf_counter()
        tree = truncate(tree, k=k, hours=hours)
        vec = self.encoder.transform([tree.text])
        trees = [tree, truncate(tree, k=1)] if self.mode == "joint" else [tree]
        proba, atts = predict_trees(self.model, trees, vec.repeat(len(trees), 0), self.scaler)
        p = proba[0]
        label = LABELS[int(p.argmax())]
        nodes, why = explain(tree, atts[0] if atts else None)
        words = []
        if self.mode != "structure":
            words = text_evidence(self.model, tree_to_data(tree, vec[0], self.scaler),
                                  self.encoder.vectorizer.get_feature_names_out(), int(p.argmax()))
        text_only = None
        if self.mode == "joint":
            text_only = dict(label=LABELS[int(proba[1].argmax())], confidence=float(proba[1].max()))
        return dict(
            words=words,
            text_only=text_only,
            label=label,
            verdict=VERDICT[label],
            confidence=float(p.max()),
            probabilities=dict(zip(LABELS, map(float, p))),
            top_nodes=nodes,
            rationale=why,
            tree=tree,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["twitter15", "twitter16"], required=True)
    parser.add_argument("--tweet-id")
    parser.add_argument("--mode", choices=MODES, default="joint")
    parser.add_argument("--stage", choices=STAGES, default="beta")
    parser.add_argument("--k", type=int, help="use only the first k nodes of the propagation")
    parser.add_argument("--hours", type=float, help="use only nodes within this many hours of posting")
    args = parser.parse_args()

    predictor = Predictor(args.dataset, args.mode, args.stage)
    trees = {t.source_id: t for t in load_trees(args.dataset)}
    tweet_id = args.tweet_id or predictor.test_ids[0]
    if tweet_id not in trees:
        raise SystemExit(f"Tweet {tweet_id} is not in {args.dataset}.")
    if tweet_id not in predictor.test_ids:
        print("Note: this claim was in the model's training/validation data, so the result is optimistic.\n")

    tree = trees[tweet_id]
    r = predictor.predict(tree, k=args.k, hours=args.hours)
    print(f"Claim:      {tree.text}")
    print(f"Observed:   {r['tree'].num_nodes} of {tree.num_nodes} nodes")
    print(f"Verdict:    {r['verdict']}  (model label: {r['label']}, confidence {r['confidence']:.2f})")
    print(f"True label: {LABELS[tree.label]}")
    print("Probabilities: " + ", ".join(f"{l} {p:.2f}" for l, p in r["probabilities"].items()))
    if r["text_only"]:
        print(f"Text only:  {r['text_only']['label']} ({r['text_only']['confidence']:.2f}) before seeing the spread")
    if r["words"]:
        print("Key words:  " + ", ".join(w["term"] for w in r["words"]))
    print(f"Why:        {r['rationale']}")
    for n in r["top_nodes"]:
        print(f"  node {n['node']:>4}  {n['kind']:<11} depth {n['depth']}  +{n['delay_min']:.1f} min  "
              f"children {n['children']}  attention {n['attention']:.3f}")
    print(f"Latency:    {r['latency_ms']:.0f} ms")
    print("\nAdvisory only: a human reviewer must make the final call.")


if __name__ == "__main__":
    main()
