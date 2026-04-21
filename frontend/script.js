/**
 * script.js
 * Frontend logic for ToxiClear — Explainable Toxic Text Normalization
 * Communicates with FastAPI backend at /analyze
 */

const API_BASE = "http://localhost:8000";

// DOM refs
const inputText     = document.getElementById("inputText");
const charCount     = document.getElementById("charCount");
const analyzeBtn    = document.getElementById("analyzeBtn");
const loadingState  = document.getElementById("loadingState");
const errorState    = document.getElementById("errorState");
const results       = document.getElementById("results");

// Result elements
const scoreValue          = document.getElementById("scoreValue");
const scoreRing           = document.getElementById("scoreRing");
const labelBadge          = document.getElementById("labelBadge");
const sentimentSummary    = document.getElementById("sentimentSummary");
const originalHighlighted = document.getElementById("originalHighlighted");
const cleanedText         = document.getElementById("cleanedText");
const simFill             = document.getElementById("simFill");
const simValue            = document.getElementById("simValue");
const summaryText         = document.getElementById("summaryText");
const toxicWordsList      = document.getElementById("toxicWordsList");
const patternsList        = document.getElementById("patternsList");
const structuralList      = document.getElementById("structuralList");
const featureChart        = document.getElementById("featureChart");
const changesList         = document.getElementById("changesList");

// ---- Character counter ----
inputText.addEventListener("input", () => {
  charCount.textContent = `${inputText.value.length} / 5000`;
});

// ---- Analyze button ----
analyzeBtn.addEventListener("click", runAnalysis);
inputText.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") runAnalysis();
});

async function runAnalysis() {
  const text = inputText.value.trim();
  if (!text) {
    showError("Please enter some text to analyze.");
    return;
  }

  setLoading(true);
  hideError();
  results.classList.add("hidden");

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || `Server error ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
    results.classList.remove("hidden");
    results.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    if (err.name === "TypeError") {
      showError("Cannot connect to backend. Make sure the FastAPI server is running on port 8000.");
    } else {
      showError(err.message);
    }
  } finally {
    setLoading(false);
  }
}

// ---- Render all results ----
function renderResults(data) {
  renderScore(data.toxicity_score, data.label);
  renderSentiment(data.sentiment_summary);
  renderTextComparison(data.original, data.cleaned_text, data.toxic_words);
  renderSimilarity(data.semantic_similarity);
  renderSummary(data.summary);
  renderTags(toxicWordsList, data.toxic_words, "danger");
  renderTags(patternsList, data.aggressive_patterns_found, "warning");
  renderTags(structuralList, data.structural_flags, "info");
  renderFeatureChart(data.explanation);
  renderChanges(data.rewrite_changes);
}

// ---- Score ring ----
function renderScore(score, label) {
  const pct = Math.round(score * 100);
  scoreValue.textContent = score.toFixed(2);

  // SVG ring: circumference = 2π×50 ≈ 314
  const circumference = 314;
  const offset = circumference - (pct / 100) * circumference;
  scoreRing.style.strokeDashoffset = offset;

  // Color based on score
  if (score >= 0.7) {
    scoreRing.style.stroke = "#ff4d6d";
  } else if (score >= 0.4) {
    scoreRing.style.stroke = "#ffb347";
  } else {
    scoreRing.style.stroke = "#00c896";
  }

  // Badge
  labelBadge.textContent = label.toUpperCase();
  labelBadge.className = `badge ${label === "toxic" ? "toxic" : "clean"}`;
}

function renderSentiment(text) {
  sentimentSummary.textContent = text;
}

// ---- Text comparison with toxic word highlighting ----
function renderTextComparison(original, cleaned, toxicWords) {
  originalHighlighted.innerHTML = highlightToxicWords(original, toxicWords);
  cleanedText.textContent = cleaned;
}

function highlightToxicWords(text, toxicWords) {
  if (!toxicWords || toxicWords.length === 0) return escapeHtml(text);

  // Sort by length descending to match longer phrases first
  const sorted = [...toxicWords].sort((a, b) => b.length - a.length);
  let result = escapeHtml(text);

  for (const word of sorted) {
    const escaped = escapeHtml(word);
    const regex = new RegExp(`\\b${escaped}\\b`, "gi");
    result = result.replace(regex, `<span class="toxic-word">${escaped}</span>`);
  }
  return result;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---- Similarity bar ----
function renderSimilarity(sim) {
  const pct = Math.round(sim * 100);
  simFill.style.width = `${pct}%`;
  simValue.textContent = `${pct}%`;
}

// ---- Summary ----
function renderSummary(text) {
  summaryText.textContent = text;
}

// ---- Tag lists ----
function renderTags(container, items, cls) {
  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.innerHTML = `<span class="tag empty">None detected</span>`;
    return;
  }
  for (const item of items) {
    const span = document.createElement("span");
    span.className = `tag ${cls}`;
    span.textContent = item;
    container.appendChild(span);
  }
}

// ---- Feature contribution chart ----
function renderFeatureChart(features) {
  featureChart.innerHTML = "";
  if (!features || features.length === 0) return;

  // Find max absolute contribution for scaling
  const maxAbs = Math.max(...features.map(f => Math.abs(f.contribution)), 0.001);

  for (const f of features) {
    const isPositive = f.contribution > 0;
    const pct = Math.min((Math.abs(f.contribution) / maxAbs) * 100, 100);

    const row = document.createElement("div");
    row.className = "feature-row";
    row.title = f.human_label;

    // Feature name (truncated)
    const nameEl = document.createElement("div");
    nameEl.className = "feature-name";
    nameEl.textContent = formatFeatureName(f.feature);

    // Bar
    const track = document.createElement("div");
    track.className = "feature-bar-track";
    const fill = document.createElement("div");
    fill.className = `feature-bar-fill ${isPositive ? "positive" : "negative"}`;
    fill.style.width = "0%";
    track.appendChild(fill);

    // Value
    const valEl = document.createElement("div");
    valEl.className = `feature-value ${isPositive ? "positive" : "negative"}`;
    valEl.textContent = (f.contribution > 0 ? "+" : "") + f.contribution.toFixed(3);

    row.appendChild(nameEl);
    row.appendChild(track);
    row.appendChild(valEl);
    featureChart.appendChild(row);

    // Animate bar after paint
    requestAnimationFrame(() => {
      setTimeout(() => { fill.style.width = `${pct}%`; }, 50);
    });
  }
}

function formatFeatureName(name) {
  if (name.startsWith("tfidf:")) return `"${name.replace("tfidf:", "")}"`;
  return name.replace(/_/g, " ");
}

// ---- Rewrite changes ----
function renderChanges(changes) {
  changesList.innerHTML = "";
  if (!changes || changes.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No transformations needed — text appears clean.";
    changesList.appendChild(li);
    return;
  }
  for (const change of changes) {
    const li = document.createElement("li");
    li.textContent = change;
    changesList.appendChild(li);
  }
}

// ---- UI helpers ----
function setLoading(on) {
  analyzeBtn.disabled = on;
  loadingState.classList.toggle("hidden", !on);
}

function showError(msg) {
  errorState.textContent = `Error: ${msg}`;
  errorState.classList.remove("hidden");
}

function hideError() {
  errorState.classList.add("hidden");
}
