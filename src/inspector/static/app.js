/* Iron Inspector — view layer.
 *
 * Rules this file follows, because they are the point of the tool:
 *   - Nothing is rendered that did not come from /api. There is no sample
 *     payload and no default object; an absent artifact renders as the path it
 *     was looked for and the command that produces it.
 *   - Every number carries its source. Hovering a metric shows the file it was
 *     read from and the shas that identify the run.
 *   - Absence and refusal are rendered, never elided. An unmeasured metric
 *     says "unmeasured"; it never falls back to 0 or an empty cell.
 */

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

async function api(path) {
  const res = await fetch(path);
  const body = await res.json().catch(() => ({ error: `unreadable response from ${path}` }));
  return body;
}

/* --- shared fragments --------------------------------------------------- */

function absentBlock(payload) {
  const box = el("div", "absent");
  box.append(el("h3", null, `${payload.what} is not on disk`));
  const p = el("p", null, "This is a result, not an error. Nothing is shown because nothing has been produced.");
  box.append(p);
  const l1 = el("span", "label", "looked for");
  const path = el("code", "path", payload.looked_for);
  const l2 = el("span", "label", "produced by");
  const cmd = el("code", "cmd", payload.produced_by);
  box.append(l1, path, l2, cmd);
  return box;
}

function shaChip(label, value) {
  const span = el("span");
  span.append(el("span", "k", label + " "));
  span.append(el("span", "sha", value ? String(value).slice(0, 12) : "—"));
  return span;
}

let tooltipTimer = null;
function attachSource(node, lines) {
  const show = (ev) => {
    const tip = $("#tooltip");
    tip.textContent = "";
    lines.forEach(([k, v]) => {
      const row = el("div", null, `${k}: ${v}`);
      tip.append(row);
    });
    tip.hidden = false;
    const x = Math.min(ev.clientX + 14, window.innerWidth - 400);
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.min(ev.clientY + 16, window.innerHeight - 120)}px`;
  };
  const hide = () => { $("#tooltip").hidden = true; };
  node.addEventListener("mousemove", show);
  node.addEventListener("mouseleave", hide);
  node.addEventListener("focus", (e) => show({ clientX: 20, clientY: 80, ...e }));
  node.addEventListener("blur", hide);
  node.tabIndex = 0;
}

/* --- view: scorecard ---------------------------------------------------- */

const QUALIFIER = "coverage.observable_fraction";

async function viewScorecard(root) {
  root.textContent = "";
  const list = await api("/api/scorecards");
  const cards = list.scorecards || [];
  if (!cards.length) {
    root.append(absentBlock({
      what: "any scorecard",
      looked_for: "outputs/scorecards/*.json",
      produced_by: "make eval",
    }));
    return;
  }

  const picker = el("div", "panel");
  picker.append(el("h2", null, "Scorecard"));
  const sel = el("select");
  cards.forEach((c) => {
    const o = el("option", null, `${c.name}  (${c.clips_scored} clips)`);
    o.value = c.name;
    sel.append(o);
  });
  const cmpSel = el("select");
  cmpSel.append(Object.assign(el("option", null, "compare with… (none)"), { value: "" }));
  cards.forEach((c) => {
    const o = el("option", null, c.name);
    o.value = c.name;
    cmpSel.append(o);
  });
  const row = el("div", "scrub");
  row.append(sel, cmpSel);
  picker.append(row);
  root.append(picker);

  const body = el("div");
  root.append(body);

  const render = async () => {
    body.textContent = "";
    const card = await api(`/api/scorecard/${sel.value}`);
    if (card.absent) { body.append(absentBlock(card)); return; }
    body.append(scorecardPanel(card));
    if (cmpSel.value && cmpSel.value !== sel.value) {
      body.append(await comparePanel(sel.value, cmpSel.value));
    }
  };
  sel.addEventListener("change", render);
  cmpSel.addEventListener("change", render);
  await render();
}

function scorecardPanel(card) {
  const wrap = el("div");

  const ident = el("div", "panel");
  ident.append(el("h2", null, "Identity"));
  ident.append(el("p", "note",
    "A metric is only meaningful against the instrument that produced it. These are that instrument."));
  const dl = el("div", "readout");
  const grid = el("dl");
  [["golden set", card.golden_set_version],
   ["set_sha", card.golden_set_sha],
   ["envelope_sha", (card.envelope || {}).envelope_sha],
   ["measured stack", (card.envelope || {}).envelope_measured_stack],
   ["gate raster", (card.envelope || {}).envelope_gate_raster],
   ["clips scored", card.clips_scored],
   ["source file", card._source]].forEach(([k, v]) => {
    grid.append(el("dt", null, k));
    grid.append(el("dd", "sha", v === undefined || v === null ? "unmeasured" : String(v)));
  });
  dl.append(grid);
  ident.append(dl);
  wrap.append(ident);

  const metrics = el("div", "panel");
  metrics.append(el("h2", null, "Metrics"));
  metrics.append(el("p", "note", "Hover any metric for the file and run it came from."));
  const gridEl = el("div", "metric-grid");
  (card.metrics || []).forEach((m) => {
    const cell = el("div", "metric" + (m.name === QUALIFIER ? " qualifier" : ""));
    cell.append(el("span", "name", m.name));
    const v = el("div");
    if (m.value === null || m.value === undefined || Number.isNaN(m.value)) {
      v.append(el("span", "unmeasured", "unmeasured"));
    } else {
      v.append(el("span", "value", Number(m.value).toFixed(4)));
      v.append(el("span", "unit", m.unit || ""));
    }
    cell.append(v);
    attachSource(cell, [
      ["file", card._source],
      ["set_sha", String(card.golden_set_sha).slice(0, 16)],
      ["envelope_sha", String((card.envelope || {}).envelope_sha).slice(0, 16)],
      ["detail", m.detail || "(none)"],
    ]);
    gridEl.append(cell);
  });
  metrics.append(gridEl);
  wrap.append(metrics);

  const hard = new Set(card._hard_coverage || []);
  const perClip = card.per_clip || {};
  const names = Object.keys(perClip).sort();
  if (names.length) {
    const tp = el("div", "panel");
    tp.append(el("h2", null, "Per clip"));
    tp.append(el("p", "note",
      "Clips tagged hard_coverage are hatched. They are excluded from the mint-time observability floor and shown on their own, not averaged into the set."));
    const table = el("table");
    const head = el("tr");
    ["clip", "tp", "fn", "fp", "below env", "unobserv.", "scored", "observable"].forEach((h, i) => {
      const th = el("th", i ? "num" : null, h);
      head.append(th);
    });
    table.append(el("thead")).firstChild.append(head);
    const tb = el("tbody");
    const soft = [];
    const hardRows = [];
    names.forEach((n) => {
      const v = perClip[n];
      const total = (v.scored_frames || 0) + (v.below_envelope_frames || 0) + (v.unobservable_frames || 0);
      const obs = total ? (v.scored_frames / total) : null;
      const tr = el("tr", hard.has(n) ? "hard" : null);
      tr.append(el("td", null, n));
      [v.tp, v.fn, v.fp, v.below_envelope_frames, v.unobservable_frames, v.scored_frames]
        .forEach((x) => tr.append(el("td", "num", x === undefined ? "—" : String(Math.round(x)))));
      tr.append(el("td", "num", obs === null ? "unmeasured" : obs.toFixed(3)));
      (hard.has(n) ? hardRows : soft).push(tr);
    });
    soft.forEach((r) => tb.append(r));
    hardRows.forEach((r) => tb.append(r));
    table.append(tb);
    tp.append(table);
    wrap.append(tp);
  }
  return wrap;
}

async function comparePanel(left, right) {
  const res = await api(`/api/compare?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`);
  const panel = el("div", "panel");
  panel.append(el("h2", null, "Comparison"));
  if (res.refused) {
    const box = el("div", "refusal");
    box.append(el("h3", null, "Refused — these scorecards are not comparable"));
    box.append(el("p", null,
      "A delta between them would render as a change in the system and would be nothing of the kind. The library refuses this comparison; so does this view."));
    const ul = el("ul");
    (res.reasons || ["unavailable"]).forEach((r) => ul.append(el("li", null, r)));
    box.append(ul);
    panel.append(box);
    return panel;
  }
  const table = el("table");
  const head = el("tr");
  ["metric", left, right, "delta"].forEach((h, i) => head.append(el("th", i ? "num" : null, h)));
  table.append(el("thead")).firstChild.append(head);
  const tb = el("tbody");
  Object.entries(res.metrics).forEach(([name, pair]) => {
    const tr = el("tr");
    tr.append(el("td", null, name));
    tr.append(el("td", "num", pair.left === undefined ? "unmeasured" : pair.left.toFixed(4)));
    tr.append(el("td", "num", pair.right === undefined ? "unmeasured" : pair.right.toFixed(4)));
    const d = (pair.left !== undefined && pair.right !== undefined)
      ? (pair.right - pair.left).toFixed(4) : "—";
    tr.append(el("td", "num", d));
    tb.append(tr);
  });
  table.append(tb);
  panel.append(table);
  return panel;
}

/* --- view: clip --------------------------------------------------------- */

const BUCKET_CLASS = { 0: "b0", 1: "b1", 2: "b2", 3: "b3" };
const BUCKET_NAME = { 0: "no GT motion", 1: "not observable", 2: "below envelope", 3: "above envelope" };

async function viewClip(root) {
  root.textContent = "";
  const golden = await api("/api/golden");
  if (golden.absent) { root.append(absentBlock(golden)); return; }

  const picker = el("div", "panel");
  picker.append(el("h2", null, "Clip inspector"));
  picker.append(el("p", "note",
    "Where a miss is judged. The bands below run frame by frame: what the camera could observe, and what the gate decided."));
  const sel = el("select");
  golden.clips.forEach((c) => {
    const o = el("option", null, c.clip_id + (c.hard_coverage ? "  ◐ hard_coverage" : ""));
    o.value = c.clip_id;
    sel.append(o);
  });
  picker.append(sel);
  root.append(picker);

  const body = el("div");
  root.append(body);

  const render = async () => {
    body.textContent = "";
    const data = await api(`/api/clip/${sel.value}`);
    if (data.absent) { body.append(absentBlock(data)); return; }
    body.append(clipPanel(data));
  };
  sel.addEventListener("change", render);
  await render();
}

function clipPanel(d) {
  const wrap = el("div", "panel");
  const layout = el("div", "clip-layout");

  // left: frame + scrubber
  const left = el("div");
  const fw = el("div", "frame-wrap");
  const img = el("img");
  img.alt = `frame of ${d.clip_id}`;
  fw.append(img);
  left.append(fw);

  const scrub = el("div", "scrub");
  const range = el("input");
  range.type = "range"; range.min = "0"; range.max = String(d.frames - 1); range.value = "0";
  range.setAttribute("aria-label", "frame");
  const idx = el("span", "idx");
  scrub.append(range, idx);
  left.append(scrub);

  // right: readout
  const right = el("div", "readout");
  const dl = el("dl");
  right.append(dl);
  const verdict = el("div", "verdict");
  right.append(verdict);
  layout.append(left, right);
  wrap.append(layout);

  // bands
  const bands = el("div", "bands");
  const obsRow = el("div", "band-row");
  obsRow.append(el("span", "band-label", "observability"));
  const obsBand = el("div", "band");
  obsRow.append(obsBand);
  const wakeRow = el("div", "band-row");
  wakeRow.append(el("span", "band-label", "gate wake"));
  const wakeBand = el("div", "band");
  wakeRow.append(wakeBand);

  d.labels.forEach((lab, i) => {
    const c = el("div", `cell ${BUCKET_CLASS[lab]}` + (i < d.warmup_frames ? " warm" : ""));
    c.title = `frame ${i}: ${BUCKET_NAME[lab]}${i < d.warmup_frames ? " (warmup, not scored)" : ""}`;
    c.addEventListener("click", () => { range.value = String(i); update(); });
    obsBand.append(c);
  });
  d.wake.forEach((w, i) => {
    const c = el("div", `cell ${w ? "wake" : "sleep"}` + (i < d.warmup_frames ? " warm" : ""));
    c.title = `frame ${i}: gate ${w ? "awake" : "asleep"}`;
    wakeBand.append(c);
  });
  bands.append(obsRow, wakeRow);

  const legend = el("div", "legend");
  [["b1", "not observable — no pixels on this sensor"],
   ["b2", "below envelope — visible, not resolvable"],
   ["b3", "above envelope — the recall denominator"]].forEach(([k, t]) => {
    const s = el("span");
    s.append(Object.assign(el("span", `swatch ${k}`), {}));
    s.append(document.createTextNode(t));
    legend.append(s);
  });
  legend.append(el("span", null, "faded = warmup, not scored"));
  bands.append(legend);
  wrap.append(bands);

  const update = () => {
    const i = Number(range.value);
    idx.textContent = `${i} / ${d.frames - 1}`;
    img.src = `/api/clip/${d.clip_id}/frame/${i}`;
    obsBand.querySelectorAll(".cell").forEach((c, j) => c.classList.toggle("cursor", i === j));
    wakeBand.querySelectorAll(".cell").forEach((c, j) => c.classList.toggle("cursor", i === j));

    dl.textContent = "";
    const label = d.labels[i];
    const rows = [
      ["frame", `${i}${i < d.warmup_frames ? "  (warmup — not scored)" : ""}`],
      ["observability", BUCKET_NAME[label]],
      ["gate", d.wake[i] ? "AWAKE" : "asleep"],
    ];
    d.agents.forEach((a) => {
      const area = a.silhouette_gate_px[i];
      const thr = a.wake_threshold_px[i];
      rows.push([`agent ${a.agent} silhouette`, `${area.toFixed(1)} gate px`]);
      rows.push([`agent ${a.agent} speed`, `${a.speed_gate_px[i].toFixed(2)} gate px/frame`]);
      rows.push([`agent ${a.agent} needs`,
        thr === null ? "unreachable at this speed" : `${thr.toFixed(1)} gate px`]);
    });
    rows.push(["source", d._source]);
    rows.push(["envelope_sha", d.envelope_sha.slice(0, 12)]);
    rows.forEach(([k, v]) => { dl.append(el("dt", null, k)); dl.append(el("dd", null, String(v))); });

    verdict.textContent = "";
    const asleep = !d.wake[i];
    let text;
    if (label === 3 && asleep) {
      text = "A miss here is a GATE DEFECT. The mover is visible, unoccluded, and large enough at this speed for the gate to have woken.";
    } else if (label === 2 && asleep) {
      text = "A miss here is a PHYSICAL LIMIT, not a defect. The mover is visible but below what this gate can resolve at this speed — tuning the threshold down would buy it with wakes on sensor noise.";
    } else if (label === 1) {
      text = "Nothing reached this sensor. Scoring this frame would ask the camera to see through its own frustum.";
    } else if (label === 0) {
      text = d.wake[i]
        ? "The gate woke with no ground-truth motion. Either the stay-awake latch is holding, or the GT motion threshold disagrees with what the gate can see."
        : "No ground-truth motion, gate asleep.";
    } else {
      text = "Resolvable motion, and the gate woke. Counted as a true positive.";
    }
    verdict.textContent = text;
  };
  range.addEventListener("input", update);
  update();
  return wrap;
}

/* --- view: envelope ----------------------------------------------------- */

async function viewEnvelope(root) {
  root.textContent = "";
  const env = await api("/api/envelope");
  if (env.absent) { root.append(absentBlock(env)); return; }

  const panel = el("div", "panel");
  panel.append(el("h2", null, "Measured capability envelope"));
  panel.append(el("p", "note",
    "The argument that no scalar threshold exists. The struck-through line is the derived value the scorecard used to rely on; the points are what the gate actually does."));

  const samples = env.samples.slice().sort(
    (a, b) => a.displacement_gate_px_per_frame - b.displacement_gate_px_per_frame);
  panel.append(envelopeChart(samples, env.derived_foreground_threshold_px));

  const table = el("table");
  const head = el("tr");
  ["speed (gate px/frame)", "speed (native px/frame)", "wake threshold (silhouette gate px)", "vs derived"]
    .forEach((h, i) => head.append(el("th", i ? "num" : null, h)));
  table.append(el("thead")).firstChild.append(head);
  const tb = el("tbody");
  samples.forEach((s) => {
    const tr = el("tr");
    const t = s.wake_threshold_silhouette_gate_px;
    if (t === null) tr.className = "inferred";
    tr.append(el("td", null, s.displacement_gate_px_per_frame.toFixed(2)));
    tr.append(el("td", "num", String(s.displacement_native_px_per_frame)));
    tr.append(el("td", "num", t === null ? "never wakes at any size" : t.toFixed(1)));
    tr.append(el("td", "num", t === null ? "—" : (t / env.derived_foreground_threshold_px).toFixed(2) + "×"));
    tb.append(tr);
  });
  table.append(tb);
  panel.append(table);
  panel.append(el("p", "note", `source: ${env._source} · measured on ${env.measured_stack}`));
  root.append(panel);
}

function envelopeChart(samples, derived) {
  const W = 720, H = 320, P = { l: 62, r: 20, t: 18, b: 42 };
  const finite = samples.filter((s) => s.wake_threshold_silhouette_gate_px !== null);
  const maxY = Math.max(derived * 1.2, ...finite.map((s) => s.wake_threshold_silhouette_gate_px)) * 1.1;
  const maxX = Math.max(...samples.map((s) => s.displacement_gate_px_per_frame)) * 1.05;
  const x = (v) => P.l + (v / maxX) * (W - P.l - P.r);
  const y = (v) => H - P.b - (v / maxY) * (H - P.t - P.b);

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label",
    "Silhouette area required to wake the gate, against mover speed. The required area falls as speed rises and becomes unreachable at the slowest speeds.");
  const add = (tag, attrs, text) => {
    const n = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (text !== undefined) n.textContent = text;
    svg.append(n);
    return n;
  };

  for (let i = 0; i <= 4; i++) {
    const v = (maxY / 4) * i;
    add("line", { class: "grid", x1: P.l, x2: W - P.r, y1: y(v), y2: y(v) });
    add("text", { class: "tick", x: P.l - 8, y: y(v) + 3, "text-anchor": "end" }, v.toFixed(0));
  }
  add("line", { class: "axis", x1: P.l, x2: W - P.r, y1: y(0), y2: y(0) });
  add("line", { class: "axis", x1: P.l, x2: P.l, y1: P.t, y2: y(0) });

  samples.forEach((s) => {
    const px = x(s.displacement_gate_px_per_frame);
    add("text", { class: "tick", x: px, y: H - P.b + 14, "text-anchor": "middle" },
      s.displacement_gate_px_per_frame.toFixed(2));
  });
  add("text", { class: "lbl", x: (W) / 2, y: H - 6, "text-anchor": "middle" },
    "mover speed (gate px / frame)");
  add("text", { class: "lbl", x: 14, y: H / 2, transform: `rotate(-90 14 ${H / 2})`, "text-anchor": "middle" },
    "silhouette area to wake (gate px)");

  // The refuted single threshold, struck through.
  add("line", { class: "refuted", x1: P.l, x2: W - P.r, y1: y(derived), y2: y(derived) });
  add("text", { class: "refuted-label", x: W - P.r - 4, y: y(derived) - 7, "text-anchor": "end" },
    `derived single threshold ${derived.toFixed(1)} px — refuted`);

  // Unreachable region: speeds where the gate never woke at any size.
  const neverMax = Math.max(0, ...samples.filter((s) => s.wake_threshold_silhouette_gate_px === null)
    .map((s) => s.displacement_gate_px_per_frame));
  if (neverMax > 0) {
    add("line", { class: "never", x1: x(neverMax), x2: x(neverMax), y1: P.t, y2: y(0) });
    add("text", { class: "tick", x: x(neverMax) + 6, y: P.t + 12 },
      "← absorbed into background at any size");
  }

  const pts = finite.map((s) => [x(s.displacement_gate_px_per_frame), y(s.wake_threshold_silhouette_gate_px)]);
  if (pts.length > 1) {
    add("path", { class: "curve", d: pts.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ") });
  }
  finite.forEach((s) => {
    add("circle", {
      class: "pt", r: 4,
      cx: x(s.displacement_gate_px_per_frame),
      cy: y(s.wake_threshold_silhouette_gate_px),
    });
  });
  return svg;
}

/* --- view: events ------------------------------------------------------- */

async function viewEvents(root) {
  root.textContent = "";
  const data = await api("/api/events");
  if (data.absent) {
    const panel = el("div", "panel");
    panel.append(el("h2", null, "Events"));
    panel.append(absentBlock(data));
    root.append(panel);
    return;
  }
  const panel = el("div", "panel");
  panel.append(el("h2", null, "Event log"));
  panel.append(el("p", "note",
    "Rows with observed=false are dashed and italic with a ⌁ marker — the distinction must survive greyscale, so it never rests on colour."));
  const table = el("table");
  const cols = ["event_id", "subject", "verb", "object", "zone", "confidence", "importance", "observed"];
  const head = el("tr");
  cols.forEach((c, i) => head.append(el("th", i >= 5 ? "num" : null, c)));
  table.append(el("thead")).firstChild.append(head);
  const tb = el("tbody");
  (data.rows || []).forEach((r) => {
    const tr = el("tr", r.observed === false ? "inferred" : null);
    cols.forEach((c, i) => tr.append(el("td", i >= 5 ? "num" : null,
      r[c] === undefined || r[c] === null ? "—" : String(r[c]))));
    tb.append(tr);
  });
  table.append(tb);
  panel.append(table);
  panel.append(el("p", "note", `source: ${data._source}`));
  root.append(panel);
}

/* --- view: provenance --------------------------------------------------- */

async function viewProvenance(root) {
  root.textContent = "";
  const p = await api("/api/provenance");
  const panel = el("div", "panel");
  panel.append(el("h2", null, "Provenance"));
  if ((p.unresolved || []).length) {
    const warn = el("div", "warnband");
    warn.append(el("h3", null, "Unresolved provenance"));
    warn.append(el("p", null,
      `Could not resolve: ${p.unresolved.join(", ")}. Numbers shown elsewhere in this session rest on artifacts that cannot be fully identified.`));
    panel.append(warn);
  }
  const readout = el("div", "readout");
  const dl = el("dl");
  const rows = [
    ["git sha", (p.git || {}).sha || "unresolved"],
    ["branch", (p.git || {}).branch || "unresolved"],
    ["working tree", (p.git || {}).dirty ? "DIRTY — artifacts may not match this commit" : "clean"],
    ["config sha", p.config_sha || "unresolved"],
    ["config source", p.config_source],
    ["golden set", `${(p.golden || {}).version || "unresolved"}  ${(p.golden || {}).set_sha || ""}`],
    ["golden status", (p.golden || {}).status || "unresolved"],
    ["golden source", (p.golden || {}).source || "unresolved"],
    ["envelope sha", (p.envelope || {}).sha || "unresolved"],
    ["envelope stack", (p.envelope || {}).measured_stack || "unresolved"],
  ];
  rows.forEach(([k, v]) => { dl.append(el("dt", null, k)); dl.append(el("dd", null, String(v))); });
  readout.append(dl);
  panel.append(readout);
  root.append(panel);
}

/* --- shell -------------------------------------------------------------- */

const VIEWS = {
  scorecard: viewScorecard,
  clip: viewClip,
  envelope: viewEnvelope,
  events: viewEvents,
  provenance: viewProvenance,
};

async function provenanceStrip() {
  const p = await api("/api/provenance");
  const strip = $("#provenance-strip");
  strip.textContent = "";
  strip.append(shaChip("git", (p.git || {}).sha));
  if ((p.git || {}).dirty) strip.append(el("span", "dirty", "working tree DIRTY"));
  strip.append(shaChip("set", (p.golden || {}).set_sha));
  strip.append(shaChip("envelope", (p.envelope || {}).sha));
  strip.append(shaChip("config", p.config_sha));
}

function activate(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.view === name;
    t.setAttribute("aria-current", on ? "page" : "false");
  });
  Object.keys(VIEWS).forEach((v) => { $(`#view-${v}`).hidden = v !== name; });
  VIEWS[name]($(`#view-${name}`));
  history.replaceState(null, "", `#${name}`);
}

document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => activate(t.dataset.view));
});

provenanceStrip();
activate((location.hash || "#scorecard").slice(1) in VIEWS ? location.hash.slice(1) : "scorecard");
