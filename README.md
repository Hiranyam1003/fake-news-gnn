# Fake News Detection with Graph Neural Networks — code

Project 13, BTECH CAP 501. Classifies a Twitter claim as **non-rumor / false / true / unverified**
from its source text and its retweet/reply propagation tree, with a confidence score and an
attention-based explanation. The 4 labels are also shown as a reviewer verdict:
false → **Fake**, true / non-rumor → **Real**, unverified → **Needs review**.

Outputs are advisory only: the tool flags claims for human review.

## Stages

| Stage | Status | What it adds |
|---|---|---|
| 1. Prototype | built | End-to-end pipeline: TF-IDF text, GraphSAGE + attention pooling, content / structure / joint comparison on one 70/10/20 split, early detection at 5 and 10 nodes, CLI prediction with rationale |
| 2. Beta | built | Late fusion (graph sees structure only, text joins at the classifier), 5-fold cross-validation (mean ± std), training on truncated trees for early detection, curves by node count and by hours, explanation-faithfulness test, latency check, Streamlit reviewer app with feedback collection |
| 3. Final | built | Text explanation (TF-IDF words that drove the prediction), text-alone vs with-spread comparison, corrected faithfulness metric (absolute change), web app (replaces the beta Streamlit app) with an animated replay of each claim's spread, live verdict and confidence as the tree grows, a results view and a try-your-own form. Serves the beta checkpoints, so no retraining |

## Setup

```bash
cd code
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

The Twitter15/16 dataset (Ma et al., 2017) downloads automatically on first use into `data/raw/`.
If the download fails, get `rumdetect2017.zip` from the URL in `config.py` and put it in `data/raw/`.

## Running

```bash
# train one configuration, or all 3 modes x 2 datasets
python train.py --dataset twitter15 --mode joint --stage prototype
python train.py --all --stage prototype
python train.py --all --stage beta          # ~20-30 min on a laptop CPU

# results table (also saved to runs/<stage>/results.csv)
python evaluate.py --stage beta

# predict one claim, optionally from only the first k nodes of its spread
python predict.py --dataset twitter16 --k 10

# final-stage analysis on the saved beta checkpoints -> runs/final/report.json
python evaluate.py --stage final

# web app (Live demo, Results, Try your own)
python server.py                            # http://localhost:8008

# tests
python -m pytest tests
```

Each run writes to `runs/<stage>/<dataset>/<mode>/`:
- `metrics.json`: accuracy, macro-F1, AUC, verdict accuracy, per-class P/R/F1, confusion matrix, early-detection results, faithfulness, latency
- `model.pt`: model trained on the fixed 70/10/20 split, used by `predict.py` and the app
- `early_detection.png`

## How it works

| File | Role |
|---|---|
| `config.py` | paths, labels, verdict map, hyperparameters per stage |
| `data.py` | parses the tree files into clean trees with parents always before children; truncation for early detection; splits |
| `features.py` | 7 structural node features (depth, delay, retweet/reply, children, arrival rank, repeat user, is-root) and source-text vectors (TF-IDF; MiniLM available via `text_encoder="minilm"`) |
| `models.py` | `ContentMLP` (text only) and `SAGEClassifier` (GraphSAGE + attention pooling; structure-only or joint, with text fed into every layer or only at the classifier) |
| `train.py` | training with early stopping, CV, holdout model, latency |
| `evaluate.py` | metrics, early detection, faithfulness, results table, final-stage report |
| `explain.py` | turns attention weights into a plain-language rationale; ranks the TF-IDF terms behind a prediction (gradient × input) |
| `predict.py` | CLI prediction |
| `server.py` + `web/` | web app: Starlette JSON API over the saved models, and a hand-built HTML/JS/D3 frontend |

Only the source tweet has text in this dataset (Twitter's terms prevent sharing the rest), so:
- **content** = source text only
- **structure** = propagation tree only
- **joint** = both

## Design decisions from experiments (5-fold CV accuracy)

| Variant | Twitter15 | Twitter16 |
|---|---|---|
| content, TF-IDF | 0.827 | 0.835 |
| joint, text in every GraphSAGE layer (prototype design) | 0.803 | 0.784 |
| **joint, late fusion (beta design)** | **0.852** | **0.863** |
| content, MiniLM | 0.746 | 0.790 |
| joint, late fusion, MiniLM | 0.771 | 0.792 |

Feeding the text into every layer let the graph branch overfit before it learned to use the text;
late fusion fixed that. The frozen MiniLM encoder was worse than TF-IDF on these short,
claim-specific tweets, so beta keeps TF-IDF.

## Final-stage findings (held-out test claims, joint model)

| | Twitter15 | Twitter16 |
|---|---|---|
| accuracy from text alone (source post only) | 0.795 | 0.811 |
| accuracy with the full spread | 0.849 | 0.866 |
| verdicts changed by the spread | 10% (20 fixed, 4 broken) | 12% (13 fixed, 4 broken) |
| mean \|Δp\| when removing top-attended 10% of nodes | 0.065 | 0.077 |
| mean \|Δp\| when removing random 10% of nodes | 0.006 | 0.008 |

Beta reported attention as "weakly faithful" because it averaged *signed* probability drops, and
those cancel out: a removed node can push either way. Measured by absolute change, removing the
top-attended nodes moves the prediction about 10× more than removing random nodes. Attention does
find influential nodes; it does not say which direction they push.

At 5–50 nodes, joint still performs about the same as text only; the gain from propagation needs
more of the spread. This is reported as a limitation, not fixed.
