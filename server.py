"""Web app: JSON API over the trained models plus the static frontend in web/.

Run: python server.py            (serves http://localhost:8008)
"""
import argparse
import json
from functools import lru_cache

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from config import DATASETS, LABELS, MODES, ROOT, RUNS_DIR, VERDICT, run_dir
from data import load_trees, parse_tree, truncate
from evaluate import predict_trees
from explain import explain, text_evidence
from features import tree_to_data
from predict import Predictor

STAGE = "beta"  # the final stage serves the beta checkpoints
WEB_DIR = ROOT / "web"
MAX_CHECKPOINTS = 60


class StaticNoCache(StaticFiles):
    """Static files that the browser revalidates on every load, so edits show up immediately."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


@lru_cache
def predictor(dataset, mode):
    return Predictor(dataset, mode, STAGE)


@lru_cache
def trees(dataset):
    return {t.source_id: t for t in load_trees(dataset)}


def checkpoints(n):
    """Node counts at which to predict: every node early on, then log-spaced up to the full tree."""
    ks = set(range(1, min(n, 12) + 1)) | set(np.geomspace(1, n, MAX_CHECKPOINTS - 12).round().astype(int))
    return sorted(int(k) for k in ks if 1 <= k <= n)


def trajectory(p, tree):
    """Predictions as the spread grows, so the frontend can replay it without further requests."""
    ks = checkpoints(tree.num_nodes)
    vec = p.encoder.transform([tree.text])
    parts = [truncate(tree, k=k) for k in ks]
    proba, atts = predict_trees(p.model, parts, vec.repeat(len(parts), 0), p.scaler)
    names = p.encoder.vectorizer.get_feature_names_out() if p.mode != "structure" else None
    steps = []
    for k, part, pr, att in zip(ks, parts, proba, atts or [None] * len(ks)):
        label = LABELS[int(pr.argmax())]
        nodes, why = explain(part, att)
        words = text_evidence(p.model, tree_to_data(part, vec[0], p.scaler), names, int(pr.argmax())) if names is not None else []
        steps.append(dict(
            k=k, label=label, verdict=VERDICT[label], probabilities=[round(float(x), 4) for x in pr],
            top=[dict(node=n["node"], attention=round(n["attention"], 4), kind=n["kind"]) for n in nodes],
            attention=[round(float(a), 4) for a in att] if att is not None else None,
            rationale=why, words=words,
        ))
    return dict(
        text=tree.text, num_nodes=tree.num_nodes, true_label=LABELS[tree.label] if tree.label >= 0 else None,
        parent=tree.parent.tolist(), delay=[round(float(d), 2) for d in tree.delay],
        is_retweet=tree.is_retweet.tolist(), steps=steps, mode=p.mode,
    )


@lru_cache
def claim_list(dataset, mode):
    p, ts = predictor(dataset, mode), trees(dataset)
    test = [ts[i] for i in p.test_ids]
    vecs = p.encoder.transform([t.text for t in test])
    full = predict_trees(p.model, test, vecs, p.scaler)[0].argmax(1)
    alone = predict_trees(p.model, [truncate(t, k=1) for t in test], vecs, p.scaler)[0].argmax(1)
    return [dict(id=t.source_id, text=t.text, true_label=LABELS[t.label], predicted=LABELS[f],
                 text_alone=LABELS[a], nodes=t.num_nodes)
            for t, f, a in zip(test, full, alone)]


def params(request):
    dataset, mode = request.query_params.get("dataset", "twitter16"), request.query_params.get("mode", "joint")
    if dataset not in DATASETS or mode not in MODES:
        raise ValueError(f"Unknown dataset or model. Use one of {DATASETS} and {MODES}.")
    return dataset, mode


def error(message, status=400):
    return JSONResponse({"error": message}, status_code=status)


async def api_claims(request):
    try:
        dataset, mode = params(request)
        return JSONResponse(claim_list(dataset, mode))
    except (ValueError, FileNotFoundError) as e:
        return error(str(e))


async def api_claim(request):
    try:
        dataset, mode = params(request)
    except ValueError as e:
        return error(str(e))
    tree = trees(dataset).get(request.path_params["claim_id"])
    if tree is None:
        return error("That claim is not in this dataset.", 404)
    return JSONResponse(trajectory(predictor(dataset, mode), tree))


async def api_analyse(request):
    try:
        dataset, mode = params(request)
        body = await request.json()
        tree = parse_tree(str(body.get("tree", "")).splitlines(), text=str(body.get("text", "")))
    except (ValueError, SyntaxError) as e:
        return error("Could not read that tree. Each line should look like "
                     "['parent_user', 'parent_tweet', 'delay']->['user', 'tweet', 'delay'], starting with a ROOT line. "
                     f"({e})")
    return JSONResponse(trajectory(predictor(dataset, mode), tree))


async def api_results(request):
    metrics = {}
    for d in DATASETS:
        for m in MODES:
            path = run_dir(STAGE, d, m) / "metrics.json"
            if path.exists():
                r = json.loads(path.read_text())
                metrics[f"{d}/{m}"] = dict(test={k: r["test"][k] for k in ["acc", "macro_f1", "auc_ovr", "verdict_acc"]},
                                           early=r["early"], latency_ms=r["latency_ms"])
    report_path = RUNS_DIR / "final" / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return JSONResponse(dict(metrics=metrics, report=report, labels=LABELS))


async def index(request):
    return FileResponse(WEB_DIR / "index.html")


app = Starlette(routes=[
    Route("/", index),
    Route("/api/claims", api_claims),
    Route("/api/claims/{claim_id}", api_claim),
    Route("/api/analyse", api_analyse, methods=["POST"]),
    Route("/api/results", api_results),
    Mount("/static", StaticNoCache(directory=WEB_DIR)),
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8008)
    uvicorn.run(app, host="127.0.0.1", port=parser.parse_args().port)
