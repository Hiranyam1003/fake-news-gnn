"""Beta reviewer app.  Run: streamlit run app.py"""
import csv
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import DATASETS, LABELS, MODES, RUNS_DIR, STAGES, run_dir
from data import load_trees, parse_tree
from predict import Predictor

FEEDBACK_FILE = RUNS_DIR / "beta_feedback.csv"
MAX_DRAWN_NODES = 150
VERDICT_COLOR = {"Fake": "#c0392b", "Real": "#1e8449", "Needs review": "#b9770e"}


@st.cache_resource
def get_predictor(dataset, mode, stage):
    return Predictor(dataset, mode, stage)


@st.cache_data
def get_trees(dataset):
    return {t.source_id: t for t in load_trees(dataset)}


def draw_tree(tree, highlight):
    n = min(tree.num_nodes, MAX_DRAWN_NODES)
    depth = np.zeros(n, dtype=int)
    for i in range(1, n):
        depth[i] = depth[tree.parent[i]] + 1
    slot = {}
    x = np.zeros(n)
    for i in range(n):
        x[i] = slot.get(depth[i], 0)
        slot[depth[i]] = x[i] + 1
    for d, count in slot.items():
        x[depth == d] = (x[depth == d] + 0.5) / count
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i in range(1, n):
        p = tree.parent[i]
        ax.plot([x[p], x[i]], [-depth[p], -depth[i]], color="#bbb", lw=0.6, zorder=1)
    colors = ["#222" if i == 0 else ("#c0392b" if i in highlight else "#9fb3c8") for i in range(n)]
    sizes = [90 if i == 0 or i in highlight else 18 for i in range(n)]
    ax.scatter(x, -depth, c=colors, s=sizes, zorder=2)
    ax.set_yticks(-np.arange(depth.max() + 1), [f"depth {d}" for d in range(depth.max() + 1)])
    ax.set_xticks([])
    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)
    title = "Black = source post, red = most-attended nodes"
    if tree.num_nodes > n:
        title += f" (showing first {n} of {tree.num_nodes} nodes)"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def save_feedback(row):
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not FEEDBACK_FILE.exists()
    with open(FEEDBACK_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


st.set_page_config(page_title="Propagation Check (beta)", layout="wide")
st.title("Propagation-aware fake-news check")
st.caption("Beta build · Project 13, GNN fake-news detection")
st.warning("Advisory only. This tool flags claims for human review; it must not be used to remove "
           "content or penalise accounts on its own.")

with st.sidebar:
    st.header("Model")
    available = [s for s in STAGES if any((run_dir(s, d, m) / "model.pt").exists() for d in DATASETS for m in MODES)]
    if not available:
        st.error("No trained models found. Run `python train.py --all --stage beta` first.")
        st.stop()
    stage = st.selectbox("Stage", available, index=len(available) - 1)
    dataset = st.selectbox("Dataset", DATASETS)
    mode = st.selectbox("Model", MODES, index=MODES.index("joint"),
                        help="content = text only, structure = propagation only, joint = both")
    try:
        predictor = get_predictor(dataset, mode, stage)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

trees = get_trees(dataset)
tab_test, tab_upload = st.tabs(["Held-out test claim", "Upload a propagation tree"])
with tab_test:
    options = predictor.test_ids
    choice = st.selectbox("Claim (never seen during training)", options,
                          format_func=lambda sid: trees[sid].text[:110])
with tab_upload:
    uploaded = st.file_uploader("Tree file in Twitter15/16 format (parent -> child lines)", type="txt")
    pasted_text = st.text_area("Source tweet text", placeholder="Paste the original tweet text here")
    use_upload = st.checkbox("Analyse the uploaded tree instead", disabled=uploaded is None)

if use_upload and uploaded is not None:
    try:
        tree = parse_tree(uploaded.getvalue().decode("utf-8").splitlines(), text=pasted_text)
    except (ValueError, SyntaxError) as e:
        st.error(f"Could not read that tree file: {e}")
        st.stop()
    true_label = None
else:
    tree = trees[choice]
    true_label = LABELS[tree.label]

st.subheader("Claim")
st.write(f"> {tree.text or '(no text provided)'}")

k = st.slider("Early detection: observe only the first N nodes of the spread",
              min_value=1, max_value=tree.num_nodes, value=tree.num_nodes)
result = predictor.predict(tree, k=k)

c1, c2, c3 = st.columns(3)
color = VERDICT_COLOR[result["verdict"]]
c1.markdown(f"**Verdict**<br><span style='font-size:2rem;color:{color}'>{result['verdict']}</span>",
            unsafe_allow_html=True)
c2.metric("Confidence", f"{result['confidence']:.0%}", help="Probability of the predicted class")
c3.metric("Model label", result["label"], help=f"Observed {k} of {tree.num_nodes} nodes")
if true_label:
    st.caption(f"Dataset label (for beta testers): **{true_label}** · inference {result['latency_ms']:.0f} ms")

left, right = st.columns([1, 2])
with left:
    st.markdown("**Class probabilities**")
    st.bar_chart(pd.DataFrame({"probability": result["probabilities"]}), horizontal=True, height=200)
with right:
    st.markdown("**Why**")
    st.write(result["rationale"])
    if result["top_nodes"]:
        st.dataframe(pd.DataFrame(result["top_nodes"]).round(3), hide_index=True, use_container_width=True)

if mode != "content":
    st.pyplot(draw_tree(result["tree"], {n["node"] for n in result["top_nodes"]}))

st.divider()
st.subheader("Beta feedback")
with st.form("feedback", clear_on_submit=True):
    agree = st.radio("Do you agree with the verdict?", ["Agree", "Disagree", "Unsure"], horizontal=True)
    useful = st.radio("Was the explanation useful?", ["Yes", "Partly", "No"], horizontal=True)
    comment = st.text_area("Comments (optional)")
    if st.form_submit_button("Submit feedback"):
        save_feedback(dict(
            timestamp=datetime.now().isoformat(timespec="seconds"), stage=stage, dataset=dataset, mode=mode,
            source_id=tree.source_id, nodes_observed=k, predicted=result["label"],
            confidence=round(result["confidence"], 3), true_label=true_label or "", agree=agree,
            explanation_useful=useful, comment=comment,
        ))
        st.success("Thanks, your feedback was saved.")
