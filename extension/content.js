/**
 * content.js — ToxiClear Chrome Extension
 * ----------------------------------------
 * DOM-based real-time toxic text detection and replacement.
 *
 * Strategy:
 * 1. Walk the DOM tree to find text nodes
 * 2. Extract meaningful text (skip scripts, styles, inputs)
 * 3. Send to backend API for analysis
 * 4. Replace toxic text nodes with cleaned versions
 * 5. Highlight modified nodes visually
 * 6. Use MutationObserver to handle dynamically added content
 * 7. Track processed nodes to avoid reprocessing
 */

const API_BASE = "http://localhost:8000";
const TOXICITY_THRESHOLD = 0.6;   // only rewrite if score >= this
const MIN_TEXT_LENGTH = 15;        // skip very short text nodes
const MAX_TEXT_LENGTH = 1000;      // skip very long blocks (articles)
const BATCH_DELAY_MS = 300;        // debounce for mutation observer
const PROCESSED_ATTR = "data-toxiclear-processed";
const MODIFIED_ATTR  = "data-toxiclear-modified";

let isEnabled = true;
let processedNodes = new WeakSet();
let pendingNodes = [];
let batchTimer = null;
let stats = { analyzed: 0, modified: 0 };

// ---- Initialization ----

chrome.storage.sync.get(["enabled"], (result) => {
  isEnabled = result.enabled !== false; // default ON
  if (isEnabled) startProcessing();
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "TOGGLE") {
    isEnabled = msg.enabled;
    if (isEnabled) {
      startProcessing();
    } else {
      stopProcessing();
    }
    sendResponse({ ok: true });
  }
  if (msg.type === "GET_STATS") {
    sendResponse(stats);
  }
});

// ---- DOM Walking ----

/**
 * Collect all eligible text nodes from the document.
 * Skips: script, style, noscript, code, pre, input, textarea, [data-toxiclear-processed]
 */
function collectTextNodes(root = document.body) {
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;

        const tag = parent.tagName.toLowerCase();
        const skipTags = new Set([
          "script", "style", "noscript", "code", "pre",
          "input", "textarea", "select", "button", "a",
          "head", "meta", "link", "title",
        ]);

        if (skipTags.has(tag)) return NodeFilter.FILTER_REJECT;
        if (parent.hasAttribute(PROCESSED_ATTR)) return NodeFilter.FILTER_REJECT;
        if (parent.hasAttribute(MODIFIED_ATTR)) return NodeFilter.FILTER_REJECT;
        if (parent.isContentEditable) return NodeFilter.FILTER_REJECT;

        const text = node.textContent.trim();
        if (text.length < MIN_TEXT_LENGTH) return NodeFilter.FILTER_REJECT;
        if (text.length > MAX_TEXT_LENGTH) return NodeFilter.FILTER_REJECT;

        // Skip if already processed
        if (processedNodes.has(node)) return NodeFilter.FILTER_REJECT;

        return NodeFilter.FILTER_ACCEPT;
      },
    }
  );

  const nodes = [];
  let node;
  while ((node = walker.nextNode())) {
    nodes.push(node);
  }
  return nodes;
}

// ---- API Call ----

async function analyzeText(text) {
  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null; // backend not available — fail silently
  }
}

// ---- Node Processing ----

async function processNode(textNode) {
  if (!isEnabled) return;
  if (processedNodes.has(textNode)) return;

  const text = textNode.textContent.trim();
  if (!text || text.length < MIN_TEXT_LENGTH) return;

  // Mark as processed immediately to prevent double-processing
  processedNodes.add(textNode);
  const parent = textNode.parentElement;
  if (parent) parent.setAttribute(PROCESSED_ATTR, "1");

  stats.analyzed++;

  const result = await analyzeText(text);
  if (!result) return;

  if (result.toxicity_score >= TOXICITY_THRESHOLD && result.cleaned_text !== text) {
    replaceNode(textNode, result);
    stats.modified++;
    notifyBackground();
  }
}

/**
 * Replace a text node with a highlighted span containing the cleaned text.
 * The original text is stored in a data attribute for reference.
 */
function replaceNode(textNode, result) {
  const parent = textNode.parentElement;
  if (!parent) return;

  const wrapper = document.createElement("span");
  wrapper.setAttribute(MODIFIED_ATTR, "1");
  wrapper.setAttribute("data-original", result.original);
  wrapper.setAttribute("data-score", result.toxicity_score);
  wrapper.setAttribute("title",
    `ToxiClear: Score ${result.toxicity_score.toFixed(2)} | ${result.summary}`
  );
  wrapper.textContent = result.cleaned_text;

  // Visual highlight style
  wrapper.style.cssText = `
    background: rgba(108, 99, 255, 0.12);
    border-bottom: 2px solid rgba(108, 99, 255, 0.6);
    border-radius: 3px;
    padding: 0 2px;
    cursor: help;
    transition: background 0.2s;
  `;

  wrapper.addEventListener("mouseenter", () => {
    wrapper.style.background = "rgba(108, 99, 255, 0.22)";
  });
  wrapper.addEventListener("mouseleave", () => {
    wrapper.style.background = "rgba(108, 99, 255, 0.12)";
  });

  parent.replaceChild(wrapper, textNode);
}

// ---- Batch Processing ----

function processBatch() {
  if (!isEnabled || pendingNodes.length === 0) return;

  const batch = pendingNodes.splice(0, 10); // process 10 at a time
  batch.forEach(node => processNode(node));

  if (pendingNodes.length > 0) {
    setTimeout(processBatch, 100); // continue after short delay
  }
}

function queueNodes(nodes) {
  pendingNodes.push(...nodes);
  if (batchTimer) clearTimeout(batchTimer);
  batchTimer = setTimeout(processBatch, BATCH_DELAY_MS);
}

// ---- MutationObserver (dynamic content) ----

let observer = null;

function startObserver() {
  if (observer) return;
  observer = new MutationObserver((mutations) => {
    if (!isEnabled) return;
    const newNodes = [];
    for (const mutation of mutations) {
      for (const added of mutation.addedNodes) {
        if (added.nodeType === Node.ELEMENT_NODE) {
          newNodes.push(...collectTextNodes(added));
        } else if (added.nodeType === Node.TEXT_NODE) {
          const text = added.textContent.trim();
          if (text.length >= MIN_TEXT_LENGTH && !processedNodes.has(added)) {
            newNodes.push(added);
          }
        }
      }
    }
    if (newNodes.length > 0) queueNodes(newNodes);
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
}

function stopObserver() {
  if (observer) {
    observer.disconnect();
    observer = null;
  }
}

// ---- Start / Stop ----

function startProcessing() {
  const nodes = collectTextNodes();
  queueNodes(nodes);
  startObserver();
}

function stopProcessing() {
  stopObserver();
  pendingNodes = [];
  if (batchTimer) clearTimeout(batchTimer);
}

// ---- Notify background of stats ----
function notifyBackground() {
  chrome.runtime.sendMessage({ type: "STATS_UPDATE", stats }).catch(() => {});
}
