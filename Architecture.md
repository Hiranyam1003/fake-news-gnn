# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

GNN-based fake-news/rumor detection on Twitter15/16 (Ma et al., 2017) for a college course project (BTECH CAP 501, Project 13). Classifies a claim into 4 labels (`non-rumor / false / true / unverified`), also shown as a verdict (`Fake / Real / Needs review`, via `VERDICT` in `config.py`). Outputs are advisory. The parent directory holds the proposal, report, and reference papers, not code.

Stages: prototype, beta and final are built. The final stage adds explanations and analysis on top of the beta checkpoints (no new training stage); `server.py` serves `STAGE = "beta"` models. `README.md` has the stage table and the experiment results behind the design choices.

## Commands

Run everything from `code/` with the venv active (`source .venv/bin/activate`; set up with `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt`).

```bash
python -m pytest tests                                   # all tests (fast, synthetic trees, no dataset needed)
python -m pytest tests/test_model.py -k <name>           # single test
python train.py --dataset twitter15 --mode joint --stage beta   # one config; --device mps optional
python train.py --all --stage beta                       # 2 datasets x 3 modes, ~20-30 min CPU
python evaluate.py --stage beta                          # results table -> runs/<stage>/results.csv
python predict.py --dataset twitter16 --k 10             # CLI prediction on first k nodes
python evaluate.py --stage final                         # final report on beta joint checkpoints -> runs/final/report.json
python server.py                                         # web app on :8008 (Results view reads runs/final/report.json)
```

No linter or type checker is configured. `tests/conftest.py` puts `code/` on `sys.path`; modules import each other as top-level (`from config import ...`), so scripts must run from `code/`.

## Architecture

Pipeline: `data.py` (raw tree files -> `Tree`) -> `features.py` (`Tree` -> PyG `Data`) -> `models.py` -> `train.py` / `evaluate.py` -> `predict.py` (CLI) / `server.py` + `web/` (web app).

- **Node ordering invariant.** `parse_tree` sorts nodes so every parent precedes its children and the root is node 0 (delays are forced non-decreasing along paths). Early detection (`truncate`) depends on this: a partial tree is just a prefix. `remove_nodes` (used by the faithfulness test) re-attaches children to the nearest kept ancestor. Keep this invariant if you touch parsing.
- **Text is only on the source tweet.** The dataset has no reply/retweet text, so `content` = source text only, `structure` = tree with 7 structural node features only, `joint` = both. Text arrives as a single graph-level vector (`data.text`), not per-node features.
- **Stage config, not code branches.** `STAGE_CONFIG` in `config.py` drives behavior differences: `root_enhance` (prototype: text concatenated into every GraphSAGE layer; beta: late fusion, text joins only at the classifier), folds, `truncate_train` (random prefix augmentation each epoch), and early-detection checkpoints. Beta's late fusion and TF-IDF (over MiniLM) were chosen by experiment; don't revert them without new results.
- **Per-split fitting.** `run_split` fits the `TextEncoder` (TF-IDF) and `FeatureScaler` on that split's training trees only. With `folds > 1`, CV metrics are reported, but the saved `model.pt` always comes from a separate fixed 70/10/20 `holdout_split` so its test claims stay unseen; `predict.py`/`server.py` use `test_ids` from the checkpoint to pick claims.
- **Checkpoint format.** `runs/<stage>/<dataset>/<mode>/model.pt` is a dict with `state_dict` plus pickled `encoder`, `scaler`, `cfg`, dims, and `test_ids`; load with `torch.load(..., weights_only=False)` and rebuild via `build_model`. Changing `TextEncoder`/`FeatureScaler` or model constructor args breaks existing checkpoints — retrain.
- **Explanations.** `SAGEClassifier` returns its attention-pooling weights; `explain.py` turns the top-attended nodes into a plain-language rationale. `ContentMLP` returns `None` attention. `text_evidence` ranks TF-IDF terms by gradient × input on `data.text` (content/joint only). For joint, `Predictor.predict` also runs the root-only tree (`truncate(k=1)`) to report the text-alone call.
- **Web app.** `server.py` exposes `/api/claims`, `/api/claims/{id}`, `/api/analyse` (POST, custom tree) and `/api/results`. A claim response holds the whole tree plus predictions at ~60 node-count checkpoints (`checkpoints()`), so the frontend replays the spread without more requests. `web/app.js` is vanilla JS + D3 (CDN): the cascade puts nodes at an angle from a radial tree layout and a radius from log time since posting. Static files are served with `Cache-Control: no-cache`.
- **Caches.** `data/processed/<dataset>_trees.pkl` caches parsed trees and `minilm_cache.pkl` caches MiniLM embeddings. Delete the tree cache after changing `parse_tree`/`Tree`, or the old parse is silently reused. The raw dataset auto-downloads to `data/raw/` on first load.

`data/` and `runs/` are gitignored outputs.

## Findings and remaining limitations (final stage)

- Attention is faithful in magnitude: removing the top-attended 10% of nodes changes the predicted probability about 10× more (absolute) than removing random nodes. Beta's signed metric hid this. Attention does not give the direction a node pushes.
- At 5–50 nodes, joint performs about the same as content-only; the propagation gain needs most of the spread.
- The text side is explained by TF-IDF gradient × input terms; those are word-level cues, not fact-checking.
