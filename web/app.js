"use strict";

const LABELS = ["non-rumor", "false", "true", "unverified"];
const LABEL_COLOR = { "non-rumor": "var(--c-nonrumor)", false: "var(--c-false)", true: "var(--c-true)", unverified: "var(--c-unverified)" };
const VERDICT_COLOR = { Fake: "var(--fake)", Real: "var(--real)", "Needs review": "var(--review)" };
const MODE_NAME = { content: "Text only", structure: "Spread only", joint: "Text + spread" };
const MAX_DRAWN = 3000;
const PLAY_MS = 9000;

const state = { dataset: "twitter16", mode: "joint", filter: "changed", query: "", claims: [], claimId: null, view: "demo" };
const tooltip = document.getElementById("tooltip");

const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = p => `${Math.round(p * 100)}%`;
const pct1 = p => `${(p * 100).toFixed(1)}%`;
const plural = (n, w) => `${n.toLocaleString()} ${w}${n === 1 ? "" : "s"}`;

function fmtDelay(m) {
  if (m < 1) return "under a minute";
  if (m < 60) return `${Math.round(m)} min`;
  if (m < 48 * 60) return `${(m / 60).toFixed(1)} h`;
  return `${(m / 1440).toFixed(1)} days`;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `The server answered ${response.status}.`);
  return body;
}

function showTip(event, html) {
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  const pad = 14, w = tooltip.offsetWidth, h = tooltip.offsetHeight;
  tooltip.style.left = `${Math.min(event.clientX + pad, innerWidth - w - 8)}px`;
  tooltip.style.top = `${Math.min(event.clientY + pad, innerHeight - h - 8)}px`;
}
const hideTip = () => { tooltip.hidden = true; };

/* ---------- Stage: one claim, its verdict, and the replayable cascade ---------- */

const ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4.5v15l13-7.5z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 4h4.5v16H6zM13.5 4H18v16h-4.5z"/></svg>';

class Stage {
  constructor(root) {
    this.root = root;
    this.raf = null;
  }

  show(data) {
    this.stop();
    this.data = data;
    this.n = data.num_nodes;
    this.k = 0;
    this.step = null;
    this.root.innerHTML = `
      <article class="stage">
        <p class="claim-text"></p>
        <div class="readout">
          <div aria-live="polite">
            <div class="verdict-word"></div>
            <div class="verdict-sub"></div>
            <div class="truth"></div>
            <p class="shift-line"></p>
          </div>
          <div class="probs">${LABELS.map(l => `
            <div class="prob" data-label="${l}"><span class="name">${l}</span>
              <span class="bar"><span class="fill" style="background:${LABEL_COLOR[l]}"></span></span><span class="v"></span></div>`).join("")}
          </div>
        </div>
        <section class="cascade">
          <div class="cascade-head">
            <h2>How it spread</h2>
            <p>Each dot is a retweet or reply. Distance from the centre is time since posting.
              Violet dots are where the model looked.</p>
          </div>
          <div class="cascade-body">
            <div>
              <svg class="tree" viewBox="-300 -300 600 600" role="img" aria-label="Propagation tree of the claim"></svg>
              <div class="controls">
                <button class="play" type="button" aria-label="Play the spread">${ICON_PLAY}</button>
                <input class="scrub" type="range" min="0" max="1000" step="1" aria-label="How much of the spread the model has seen">
                <div class="clock"><span class="count"></span><span class="time"></span></div>
              </div>
            </div>
            <div class="side">
              <h3>Confidence as the spread grows</h3>
              <div class="legend">${LABELS.map(l => `<span><i style="background:${LABEL_COLOR[l]}"></i>${l}</span>`).join("")}</div>
              <svg class="conf" viewBox="0 0 300 176" role="img" aria-label="Class probabilities by number of nodes seen"></svg>
              <p class="side-note muted"></p>
            </div>
          </div>
        </section>
        <section class="why">
          <div><h3>What in the text</h3><p class="words-intro"></p><ul class="words"></ul></div>
          <div><h3>What in the spread</h3><p class="rationale"></p><ul class="nodes"></ul></div>
        </section>
      </article>`;
    const $ = s => this.root.querySelector(s);
    this.el = {
      claim: $(".claim-text"), verdict: $(".verdict-word"), sub: $(".verdict-sub"), truth: $(".truth"),
      shift: $(".shift-line"), probs: [...this.root.querySelectorAll(".prob")], tree: $(".tree"), play: $(".play"),
      scrub: $(".scrub"), count: $(".count"), time: $(".time"), conf: $(".conf"), sideNote: $(".side-note"),
      wordsIntro: $(".words-intro"), words: $(".words"), rationale: $(".rationale"), nodes: $(".nodes"),
    };
    this.el.play.addEventListener("click", () => (this.raf ? this.stop() : this.play()));
    this.el.scrub.addEventListener("input", () => { this.stop(); this.setK(this.kFromScrub(+this.el.scrub.value), true); });
    this.el.scrub.disabled = this.n < 2;
    this.el.play.disabled = this.n < 2;
    this.el.sideNote.textContent = {
      content: "The text-only model never looks at the spread, so its lines stay flat.",
      structure: "The spread-only model ignores the text, so early on it has little to go on.",
      joint: "The first point is the text alone; later points add the spread.",
    }[data.mode];
    this.drawTree();
    this.drawConf();
    this.setK(this.n);
  }

  kFromScrub(v) { return this.n < 2 ? 1 : Math.max(1, Math.round(this.n ** (v / 1000))); }
  scrubFromK(k) { return this.n < 2 ? 1000 : Math.round((Math.log(k) / Math.log(this.n)) * 1000); }
  stepFor(k) {
    const steps = this.data.steps;
    let s = steps[0];
    for (const step of steps) if (step.k <= k) s = step;
    return s;
  }

  drawTree() {
    const { parent, delay, is_retweet } = this.data;
    const N = Math.min(this.n, MAX_DRAWN);
    this.N = N;
    const depth = new Array(N).fill(0);
    for (let i = 1; i < N; i++) depth[i] = depth[parent[i]] + 1;
    this.depth = depth;

    const root = d3.stratify().id(d => d.id).parentId(d => d.parent)(
      d3.range(N).map(i => ({ id: String(i), parent: i ? String(parent[i]) : null })));
    root.sort((a, b) => +a.id - +b.id);
    d3.tree().size([2 * Math.PI, 1]).separation((a, b) => (a.parent === b.parent ? 1 : 2) / a.depth)(root);
    const angle = new Array(N);
    root.each(d => { angle[+d.id] = d.x; });

    const maxDelay = Math.max(1, d3.max(delay.slice(0, N)));
    const R = 270, inner = 26;
    const t = d3.scaleLog().domain([1, maxDelay + 1]).range([inner, R]).clamp(true);
    const pos = d3.range(N).map(i => {
      const r = i === 0 ? 0 : t(delay[i] + 1);
      return [r * Math.sin(angle[i]), -r * Math.cos(angle[i])];
    });
    this.pos = pos;

    const svg = d3.select(this.el.tree);
    const rings = [[1, "1 min"], [10, "10 min"], [60, "1 h"], [360, "6 h"], [1440, "1 day"], [10080, "1 week"], [43200, "1 month"]]
      .filter(([m]) => m <= maxDelay);
    const ringG = svg.append("g");
    ringG.selectAll("circle").data(rings).join("circle").attr("class", "ring").attr("r", d => t(d[0] + 1));
    ringG.selectAll("text").data(rings).join("text").attr("class", "ring-label")
      .attr("x", 3).attr("y", d => -t(d[0] + 1) - 3).text(d => d[1]);

    const edgeG = svg.append("g");
    this.edgeEls = [null];
    for (let i = 1; i < N; i++) {
      const [x1, y1] = pos[parent[i]], [x2, y2] = pos[i];
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", "edge");
      line.setAttribute("x1", x1); line.setAttribute("y1", y1); line.setAttribute("x2", x2); line.setAttribute("y2", y2);
      edgeG.node().appendChild(line);
      this.edgeEls.push(line);
    }
    const nodeG = svg.append("g");
    this.baseR = N > 800 ? 1.3 : N > 200 ? 2 : 3;
    this.nodeEls = d3.range(N).map(i => {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("class", i === 0 ? "node root" : "node");
      c.setAttribute("cx", pos[i][0]); c.setAttribute("cy", pos[i][1]);
      c.setAttribute("r", i === 0 ? 7 : this.baseR);
      c.dataset.i = i;
      return c;
    });
    // Root last so it sits on top.
    this.nodeEls.slice(1).forEach(c => nodeG.node().appendChild(c));
    nodeG.node().appendChild(this.nodeEls[0]);
    for (let i = 1; i < N; i++) { this.nodeEls[i].style.display = "none"; this.edgeEls[i].style.display = "none"; }

    if (this.n > N) {
      svg.append("text").attr("class", "ring-label").attr("x", -295).attr("y", 292)
        .text(`Showing the first ${N.toLocaleString()} of ${this.n.toLocaleString()} nodes`);
    }

    this.el.tree.addEventListener("mousemove", e => {
      const i = e.target.dataset && e.target.dataset.i;
      if (i === undefined) return hideTip();
      showTip(e, this.nodeTip(+i));
    });
    this.el.tree.addEventListener("mouseleave", hideTip);
    this.isRetweet = is_retweet;
  }

  nodeTip(i) {
    const att = this.step && this.step.attention ? this.step.attention[i] : null;
    const kind = i === 0 ? "Source post" : this.isRetweet[i] ? "Retweet" : "Reply";
    const lines = [`<b>${kind}</b>`];
    if (i > 0) lines.push(`${fmtDelay(this.data.delay[i])} after posting, depth ${this.depth[i]}`);
    if (att !== null && att !== undefined) lines.push(`${pct1(att)} of the model's attention`);
    return lines.join("<br>");
  }

  drawConf() {
    const steps = this.data.steps;
    const svg = d3.select(this.el.conf);
    const x = (this.n < 2 ? d3.scaleLinear().domain([0, 1]) : d3.scaleLog().domain([1, this.n])).range([30, 292]);
    const y = d3.scaleLinear().domain([0, 1]).range([138, 8]);
    this.confX = x;
    const xTicks = [1, 10, 100, 1000, 10000].filter(v => v <= this.n);
    svg.append("g").attr("class", "axis").attr("transform", "translate(0,138)")
      .call(d3.axisBottom(x).tickValues(xTicks).tickFormat(d3.format("~s")).tickSizeOuter(0));
    svg.append("g").attr("class", "axis").attr("transform", "translate(30,0)")
      .call(d3.axisLeft(y).tickValues([0, 0.5, 1]).tickFormat(d3.format(".0%")).tickSizeOuter(0));
    svg.append("text").attr("x", 292).attr("y", 172).attr("text-anchor", "end").attr("class", "ring-label")
      .style("fill", "var(--ink-soft)").style("font-size", "10px").text("nodes seen, log scale");
    const line = d3.line().curve(d3.curveStepAfter);
    LABELS.forEach((l, li) => {
      const pts = steps.map(s => [x(s.k), y(s.probabilities[li])]);
      pts.push([x(this.n), pts[pts.length - 1][1]]);
      svg.append("path").attr("d", line(pts)).attr("fill", "none").style("stroke", LABEL_COLOR[l]).attr("stroke-width", 2);
    });
    this.cursor = svg.append("line").attr("class", "cursor").attr("y1", 8).attr("y2", 138);
  }

  setK(k, fromScrub = false) {
    k = Math.max(1, Math.min(this.n, k));
    const prev = this.k || 1;
    for (let i = Math.min(prev, k); i < Math.min(Math.max(prev, k), this.N); i++) {
      const show = i < k ? "" : "none";
      this.nodeEls[i].style.display = show;
      if (i > 0) this.edgeEls[i].style.display = show;
    }
    this.k = k;
    if (!fromScrub) this.el.scrub.value = this.scrubFromK(k);
    this.el.count.innerHTML = `<b>${k.toLocaleString()}</b> of ${plural(this.n, "node")} seen`;
    this.el.time.textContent = k === 1 ? "at the moment of posting" : `${fmtDelay(this.data.delay[k - 1])} after posting`;
    this.cursor.attr("x1", this.confX(k)).attr("x2", this.confX(k));
    const step = this.stepFor(k);
    if (step !== this.step) this.updateStep(step);
  }

  updateStep(step) {
    const first = this.data.steps[0];
    const prevVerdict = this.step && this.step.verdict;
    this.step = step;
    const p = Math.max(...step.probabilities);
    const v = this.el.verdict;
    v.textContent = step.verdict;
    v.style.color = VERDICT_COLOR[step.verdict];
    if (prevVerdict && prevVerdict !== step.verdict) { v.classList.remove("flip"); void v.offsetWidth; v.classList.add("flip"); }
    this.el.sub.innerHTML = `model label <b>${step.label}</b>, ${pct(p)} confident`;
    const truth = this.data.true_label;
    this.el.truth.textContent = truth
      ? `${truth === step.label ? "Matches" : "Differs from"} the dataset label (${truth})`
      : "";

    if (this.data.mode === "joint") {
      const alone = `From the text alone the model says <b>${first.label}</b> (${pct(Math.max(...first.probabilities))}).`;
      this.el.shift.innerHTML = step.k === 1
        ? `${alone} Press play to watch the spread arrive.`
        : `${alone} After ${(step.k - 1).toLocaleString()} ${step.k === 2 ? "retweet or reply" : "retweets and replies"} it says <b>${step.label}</b>${step.label === first.label ? ", " + (p >= Math.max(...first.probabilities) ? "more" : "less") + " sure" : ""}.`;
    } else {
      this.el.shift.textContent = this.data.mode === "content"
        ? "This model reads only the tweet text, so the spread never changes its answer."
        : "This model ignores the text and reads only how the claim spread.";
    }

    const top = step.probabilities.indexOf(p);
    this.el.probs.forEach((row, i) => {
      row.classList.toggle("top", i === top);
      row.querySelector(".fill").style.width = pct1(step.probabilities[i]);
      row.querySelector(".v").textContent = pct(step.probabilities[i]);
    });

    // Attention: every visible node sized by its share, top nodes in violet.
    const att = step.attention;
    const topSet = new Set(step.top.map(t => t.node));
    const maxAtt = att ? Math.max(...att.slice(1, this.N), 1e-9) : 1;
    for (let i = 1; i < this.N; i++) {
      const el = this.nodeEls[i];
      const a = att && i < att.length ? att[i] : 0;
      el.setAttribute("r", this.baseR + (att ? 6 * Math.sqrt(a / maxAtt) * (topSet.has(i) ? 1 : 0.5) : 0));
      el.classList.toggle("evidence", topSet.has(i));
    }

    this.renderWords(step.words);
    this.el.rationale.textContent = this.data.mode === "content"
      ? "This model does not look at the spread."
      : step.rationale;
    this.el.nodes.innerHTML = step.top.filter(t => t.node < this.N).map(t => `
      <li tabindex="0" data-i="${t.node}"><span>${t.node === 0 ? "Source post" : t.kind === "retweet" ? "Retweet" : "Reply"}
        <span class="muted">${t.node === 0 ? "" : fmtDelay(this.data.delay[t.node]) + " in"}</span></span>
        <span>${pct1(t.attention)}</span></li>`).join("");
    this.el.nodes.querySelectorAll("li").forEach(li => {
      const node = this.nodeEls[+li.dataset.i];
      const on = () => node.classList.add("lit");
      const off = () => node.classList.remove("lit");
      li.addEventListener("mouseenter", on); li.addEventListener("focus", on);
      li.addEventListener("mouseleave", off); li.addEventListener("blur", off);
    });
  }

  renderWords(words) {
    const text = this.data.text || "(no tweet text provided)";
    const weight = new Map();
    for (const w of words) for (const part of w.term.split(" ")) weight.set(part, Math.max(weight.get(part) || 0, w.weight));
    const max = Math.max(...words.map(w => w.weight), 1e-9);
    this.el.claim.innerHTML = esc(text).replace(/\b\w\w+\b/g, m => {
      const w = weight.get(m.toLowerCase());
      return w ? `<mark class="${w / max < 0.4 ? "weak" : ""}">${m}</mark>` : m;
    });
    if (!words.length) {
      this.el.wordsIntro.textContent = this.data.mode === "structure"
        ? "This model does not read the text."
        : "No single word pushed strongly toward this answer.";
      this.el.words.innerHTML = "";
      return;
    }
    this.el.wordsIntro.textContent = `Words that pushed the model toward “${this.step.label}”, underlined in the tweet.`;
    this.el.words.innerHTML = words.map(w => `
      <li><span>${esc(w.term)}</span><span class="wbar" style="width:${(100 * w.weight / max).toFixed(1)}%"></span></li>`).join("");
  }

  play() {
    if (this.n < 2) return;
    if (this.k >= this.n) this.setK(1);
    const start = performance.now() - (Math.log(this.k) / Math.log(this.n)) * PLAY_MS;
    this.el.play.innerHTML = ICON_PAUSE;
    this.el.play.setAttribute("aria-label", "Pause");
    const tick = now => {
      const t = (now - start) / PLAY_MS;
      if (t >= 1) { this.setK(this.n); this.stop(); return; }
      this.setK(Math.max(1, Math.round(this.n ** t)));
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  stop() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
    if (this.el) {
      this.el.play.innerHTML = ICON_PLAY;
      this.el.play.setAttribute("aria-label", "Play the spread");
    }
  }

  message(html, cls = "loading") {
    this.stop();
    this.root.innerHTML = `<div class="${cls}">${html}</div>`;
  }
}

/* ---------- Demo view: claim list ---------- */

const demoStage = new Stage(document.getElementById("demo-stage"));
const listEl = document.getElementById("claim-list");
const FILTERS = {
  changed: c => c.predicted !== c.text_alone,
  wrong: c => c.predicted !== c.true_label,
  all: () => true,
};

function filteredClaims() {
  const q = state.query.trim().toLowerCase();
  return state.claims.filter(FILTERS[state.filter]).filter(c => !q || c.text.toLowerCase().includes(q));
}

function renderChips() {
  document.querySelectorAll(".chips button").forEach(b => {
    const f = b.dataset.filter;
    b.setAttribute("aria-checked", String(f === state.filter));
    const n = state.claims.filter(FILTERS[f]).length;
    b.innerHTML = `${b.textContent.replace(/\s*\d+$/, "")}<span class="count">${n}</span>`;
  });
}

function renderList() {
  renderChips();
  const claims = filteredClaims();
  if (!claims.length) {
    listEl.innerHTML = `<li class="empty">${state.filter === "changed" && state.mode === "content"
      ? "The text-only model never changes its call as the spread grows. Pick All to browse every claim."
      : "No claims match. Clear the search or pick another filter."}</li>`;
    return;
  }
  listEl.innerHTML = claims.map(c => `
    <li><button type="button" data-id="${c.id}" aria-current="${c.id === state.claimId}">
      <span class="t">${esc(c.text)}</span>
      <span class="m">Labelled ${c.true_label}, ${plural(c.nodes, "node")}${c.predicted !== c.text_alone
        ? ` <span class="shift">spread changed the call</span>` : ""}</span>
    </button></li>`).join("");
}

listEl.addEventListener("click", e => {
  const b = e.target.closest("button[data-id]");
  if (b) selectClaim(b.dataset.id);
});

let claimToken = 0;
async function selectClaim(id) {
  state.claimId = id;
  listEl.querySelectorAll("button[data-id]").forEach(b => b.setAttribute("aria-current", String(b.dataset.id === id)));
  const token = ++claimToken;
  try {
    const data = await api(`/api/claims/${id}?dataset=${state.dataset}&mode=${state.mode}`);
    if (token === claimToken) demoStage.show(data);
  } catch (err) {
    if (token === claimToken) demoStage.message(esc(err.message), "error-box");
  }
}

async function loadClaims() {
  demoStage.message("Loading claims…");
  listEl.innerHTML = "";
  try {
    state.claims = await api(`/api/claims?dataset=${state.dataset}&mode=${state.mode}`);
  } catch (err) {
    demoStage.message(esc(err.message), "error-box");
    return;
  }
  if (!state.claims.some(FILTERS[state.filter])) state.filter = "all";
  renderList();
  const visible = filteredClaims();
  const keep = visible.find(c => c.id === state.claimId) || visible[0];
  if (keep) selectClaim(keep.id);
  else demoStage.message("Pick a claim from the list.");
}

document.querySelectorAll(".chips button").forEach(b => b.addEventListener("click", () => {
  state.filter = b.dataset.filter;
  renderList();
}));
document.getElementById("claim-search").addEventListener("input", e => { state.query = e.target.value; renderList(); });

/* ---------- Settings and routing ---------- */

function syncSettings() {
  document.querySelectorAll(".segmented").forEach(group => {
    const key = group.dataset.setting;
    group.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.value === state[key])));
  });
}

document.querySelectorAll(".segmented").forEach(group => group.addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b || state[group.dataset.setting] === b.dataset.value) return;
  state[group.dataset.setting] = b.dataset.value;
  syncSettings();
  loadClaims();
  if (tryStage.data) analyseTry();
}));

function route() {
  const view = (location.hash.slice(1) || "demo");
  state.view = ["demo", "results", "try"].includes(view) ? view : "demo";
  document.querySelectorAll(".view").forEach(v => { v.hidden = v.id !== `view-${state.view}`; });
  document.querySelectorAll(".views a").forEach(a => {
    if (a.dataset.view === state.view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  document.querySelector(".settings").hidden = state.view === "results";
  if (state.view !== "demo") demoStage.stop();
  if (state.view === "results") loadResults();
}
addEventListener("hashchange", route);

/* ---------- Results view ---------- */

let resultsLoaded = false;
async function loadResults() {
  if (resultsLoaded) return;
  const root = document.getElementById("view-results");
  root.innerHTML = `<p class="loading">Loading results…</p>`;
  try {
    renderResults(root, await api("/api/results"));
    resultsLoaded = true;
  } catch (err) {
    root.innerHTML = `<p class="error-box">${esc(err.message)}</p>`;
  }
}

const MODE_STYLE = {
  content: { color: "var(--ink-soft)", r: 5 },
  structure: { color: "var(--steel)", r: 5 },
  joint: { color: "var(--ink)", r: 7 },
};

function renderResults(root, { metrics, report }) {
  const acc = (d, m) => metrics[`${d}/${m}`].test.acc;
  const latency = metrics["twitter16/joint"].latency_ms;
  root.innerHTML = `
    <p class="lede">Reading how a claim spread lifts accuracy from ${pct1(acc("twitter15", "content").mean)} to
      ${pct1(acc("twitter15", "joint").mean)} on Twitter15, and from ${pct1(acc("twitter16", "content").mean)} to
      ${pct1(acc("twitter16", "joint").mean)} on Twitter16.</p>
    <p class="lede-note">Five-fold cross-validation over 1,490 and 818 claims with four labels: non-rumor, false,
      true and unverified. A prediction takes about ${Math.max(1, Math.round(latency))} ms per claim, well under the 2-second target.</p>
    <div class="figure"><div><h2>Text, spread, or both</h2>
      <p>The spread on its own is a weak signal. Added to the text, it is the best of the three. Bars show ± one standard deviation across folds.</p></div>
      <svg id="fig-ablation" viewBox="0 0 640 210"></svg></div>
    <div class="figure"><div><h2>How early can it tell?</h2>
      <p>Accuracy when the model sees only the start of the spread. With little spread, text + spread does about as well as text alone; the gain comes once more of the spread is in.</p>
      <div class="segmented toggle" role="radiogroup" aria-label="Measure early detection by">
        <button role="radio" data-axis="nodes" aria-checked="true">Nodes seen</button>
        <button role="radio" data-axis="hours" aria-checked="false">Hours since posting</button></div></div>
      <svg id="fig-early" viewBox="0 0 640 240"></svg></div>
    ${report ? `
    <div class="figure"><div><h2>When the spread changes the call</h2>
      <p>On held-out claims, the same model judged from the text alone, then with the full spread.</p></div>
      <svg id="fig-shift" viewBox="0 0 640 170"></svg></div>
    <div class="figure"><div><h2>Do the explanations point at what matters?</h2>
      <p>Average change in the model's confidence after removing 10% of nodes. Removing the nodes it attends to moves it about ten times more than removing random ones. Attention shows how much a node matters, not which way it pushes.</p></div>
      <svg id="fig-faith" viewBox="0 0 640 170"></svg></div>` : `
    <div class="figure"><div><h2>More analysis</h2><p>Run <code>python evaluate.py --stage final</code> to add the spread and faithfulness figures.</p></div></div>`}
    <div class="figure"><div><h2>Limitations</h2></div>
      <ul class="limits">
        <li>Only the source tweet has text; Twitter15/16 does not include reply text.</li>
        <li>Word-level evidence shows cues the model learned, not whether the claim is true.</li>
        <li>English Twitter data from 2015–16; not tested on other platforms or recent events.</li>
        <li>Advisory only: a human reviewer makes the final call.</li>
      </ul></div>`;

  drawAblation(metrics);
  drawEarly(metrics, "nodes");
  root.querySelectorAll(".toggle button").forEach(b => b.addEventListener("click", () => {
    root.querySelectorAll(".toggle button").forEach(x => x.setAttribute("aria-checked", String(x === b)));
    drawEarly(metrics, b.dataset.axis);
  }));
  if (report) { drawShift(report); drawFaith(report); }
}

function drawAblation(metrics) {
  const svg = d3.select("#fig-ablation");
  const x = d3.scaleLinear().domain([0.35, 0.92]).range([110, 500]);
  const rows = ["twitter15", "twitter16"];
  svg.append("g").attr("class", "grid").attr("transform", "translate(0,190)")
    .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(".0%")).tickSize(-180)).call(g => g.select(".domain").remove());
  rows.forEach((d, ri) => {
    const y0 = 30 + ri * 85;
    svg.append("text").attr("x", 0).attr("y", y0 + 22).attr("class", "label-strong").text(d === "twitter15" ? "Twitter15" : "Twitter16");
    ["structure", "content", "joint"].forEach((m, mi) => {
      const a = metrics[`${d}/${m}`].test.acc, y = y0 + mi * 18, st = MODE_STYLE[m];
      svg.append("line").attr("x1", x(a.mean - a.std)).attr("x2", x(a.mean + a.std)).attr("y1", y).attr("y2", y)
        .style("stroke", st.color).attr("stroke-width", 2);
      svg.append("circle").attr("cx", x(a.mean)).attr("cy", y).attr("r", st.r).style("fill", st.color);
      svg.append("text").attr("x", x(a.mean + a.std) + 8).attr("y", y + 4)
        .attr("class", m === "joint" ? "label-strong" : null).text(`${MODE_NAME[m]} ${pct1(a.mean)}`);
    });
  });
}

function drawEarly(metrics, axis) {
  const svg = d3.select("#fig-early");
  svg.selectAll("*").remove();
  const keys = axis === "nodes" ? ["k=5", "k=10", "k=20", "k=50", "full"] : ["0h", "1h", "4h", "12h", "24h"];
  const labels = axis === "nodes" ? ["5", "10", "20", "50", "all"] : ["0 h", "1 h", "4 h", "12 h", "24 h"];
  const y = d3.scaleLinear().domain([0.2, 0.9]).range([200, 20]);
  ["twitter15", "twitter16"].forEach((d, di) => {
    const x = d3.scalePoint().domain(labels).range([60 + di * 320, 290 + di * 320]).padding(0.2);
    const g = svg.append("g");
    g.append("text").attr("x", 60 + di * 320).attr("y", 10).attr("class", "label-strong").text(d === "twitter15" ? "Twitter15" : "Twitter16");
    g.append("g").attr("class", "axis").attr("transform", "translate(0,200)").call(d3.axisBottom(x).tickSizeOuter(0));
    g.append("g").attr("class", "grid").attr("transform", `translate(${60 + di * 320},0)`)
      .call(d3.axisLeft(y).ticks(4).tickFormat(di ? "" : d3.format(".0%")).tickSize(di ? -230 : -230))
      .call(s => s.select(".domain").remove());
    ["structure", "content", "joint"].forEach(m => {
      const early = metrics[`${d}/${m}`].early, st = MODE_STYLE[m];
      const pts = keys.map((k, i) => [x(labels[i]), y(early[k].acc.mean)]);
      g.append("path").attr("d", d3.line()(pts)).attr("fill", "none").style("stroke", st.color).attr("stroke-width", m === "joint" ? 2.5 : 1.5);
      g.selectAll(null).data(pts).join("circle").attr("cx", p => p[0]).attr("cy", p => p[1]).attr("r", m === "joint" ? 4 : 3).style("fill", st.color);
      g.append("text").attr("x", pts[4][0] + 8).attr("y", pts[4][1] + { joint: -6, content: 13, structure: 4 }[m])
        .attr("class", m === "joint" ? "label-strong" : null)
        .text(MODE_NAME[m].replace("Text + spread", "Both"));
    });
    g.append("text").attr("x", 175 + di * 320).attr("y", 234).attr("text-anchor", "middle")
      .text(axis === "nodes" ? "nodes of the spread seen" : "hours since posting");
  });
}

function drawShift(report) {
  const svg = d3.select("#fig-shift");
  const x = d3.scaleLinear().domain([0.75, 0.9]).range([110, 470]);
  svg.append("g").attr("class", "grid").attr("transform", "translate(0,150)")
    .call(d3.axisBottom(x).ticks(4).tickFormat(d3.format(".0%")).tickSize(-140)).call(g => g.select(".domain").remove());
  Object.entries(report).forEach(([d, r], i) => {
    const y = 40 + i * 60;
    svg.append("text").attr("x", 0).attr("y", y + 4).attr("class", "label-strong").text(d === "twitter15" ? "Twitter15" : "Twitter16");
    svg.append("line").attr("x1", x(r.acc_text_only)).attr("x2", x(r.acc_full)).attr("y1", y).attr("y2", y).style("stroke", "var(--ink)").attr("stroke-width", 2);
    svg.append("circle").attr("cx", x(r.acc_text_only)).attr("cy", y).attr("r", 6).style("fill", "var(--steel)");
    svg.append("circle").attr("cx", x(r.acc_full)).attr("cy", y).attr("r", 7).style("fill", "var(--ink)");
    svg.append("text").attr("x", x(r.acc_text_only)).attr("y", y - 12).attr("text-anchor", "middle").text(`text alone ${pct1(r.acc_text_only)}`);
    svg.append("text").attr("x", x(r.acc_full)).attr("y", y - 12).attr("text-anchor", "middle").attr("class", "label-strong").text(`with spread ${pct1(r.acc_full)}`);
    svg.append("text").attr("x", 490).attr("y", y + 4).text(`${r.changed_to_correct} calls fixed, ${r.changed_to_wrong} broken`);
  });
}

function drawFaith(report) {
  const svg = d3.select("#fig-faith");
  const entries = Object.entries(report);
  const max = d3.max(entries, ([, r]) => r.faithfulness.mean_abs_change_top);
  const x = d3.scaleLinear().domain([0, max * 1.1]).range([110, 560]);
  entries.forEach(([d, r], i) => {
    const y = 20 + i * 70, f = r.faithfulness;
    svg.append("text").attr("x", 0).attr("y", y + 16).attr("class", "label-strong").text(d === "twitter15" ? "Twitter15" : "Twitter16");
    [[f.mean_abs_change_top, "var(--evidence)", "nodes it attends to"], [f.mean_abs_change_random, "var(--steel)", "random nodes"]].forEach(([v, c, label], j) => {
      const yy = y + j * 22;
      svg.append("rect").attr("x", 110).attr("y", yy).attr("height", 16).attr("width", Math.max(1, x(v) - 110)).style("fill", c).attr("rx", 2);
      svg.append("text").attr("x", x(v) + 8).attr("y", yy + 12).attr("class", j ? null : "label-strong")
        .text(`${label}: ${(v * 100).toFixed(1)} pts`);
    });
  });
}

/* ---------- Try view ---------- */

const tryStage = new Stage(document.getElementById("try-stage"));
tryStage.message("Paste a claim and its spread, then choose Analyse claim. Or fill in an example to see the format.", "placeholder");
const tryText = document.getElementById("try-text");
const tryTree = document.getElementById("try-tree");
const tryError = document.getElementById("try-error");

async function analyseTry() {
  tryError.textContent = "";
  try {
    const data = await api(`/api/analyse?dataset=${state.dataset}&mode=${state.mode}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: tryText.value, tree: tryTree.value }),
    });
    tryStage.show(data);
  } catch (err) {
    tryError.textContent = err.message;
  }
}

document.getElementById("try-form").addEventListener("submit", e => { e.preventDefault(); analyseTry(); });
document.getElementById("try-file").addEventListener("change", async e => {
  const file = e.target.files[0];
  if (file) tryTree.value = await file.text();
});
document.getElementById("try-sample").addEventListener("click", () => {
  tryText.value = "breaking: officials confirm the city bridge has collapsed, dozens feared trapped";
  const lines = ["['ROOT', 'ROOT', '0.0']->['100', '9000', '0.0']"];
  const delays = [0.4, 0.9, 1.3, 2.1, 2.8, 3.5, 5, 7.2, 9, 12, 15, 21, 30, 44, 61, 90, 130, 200, 320, 600];
  delays.forEach((d, i) => {
    const parent = i < 6 ? ["100", "9000", "0.0"] : [String(101 + (i % 6)), "9000", String(delays[i % 6])];
    lines.push(`['${parent[0]}', '${parent[1]}', '${parent[2]}']->['${101 + i}', '9000', '${d}']`);
  });
  tryTree.value = lines.join("\n");
});

/* ---------- Start ---------- */

syncSettings();
route();
loadClaims();
