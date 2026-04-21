/**
 * popup.js — ToxiClear Extension Popup
 */

const API_BASE = "http://localhost:8000";

const toggleSwitch  = document.getElementById("toggleSwitch");
const statAnalyzed  = document.getElementById("statAnalyzed");
const statModified  = document.getElementById("statModified");
const statusDot     = document.getElementById("statusDot");
const statusText    = document.getElementById("statusText");
const openUIBtn     = document.getElementById("openUI");

// ---- Load saved state ----
chrome.storage.sync.get(["enabled"], (result) => {
  toggleSwitch.checked = result.enabled !== false;
});

// ---- Toggle handler ----
toggleSwitch.addEventListener("change", () => {
  const enabled = toggleSwitch.checked;
  chrome.storage.sync.set({ enabled });

  // Notify active tab's content script
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, { type: "TOGGLE", enabled }).catch(() => {});
    }
  });
});

// ---- Load stats from background ----
chrome.runtime.sendMessage({ type: "GET_GLOBAL_STATS" }, (response) => {
  if (response) {
    statAnalyzed.textContent = response.analyzed || 0;
    statModified.textContent = response.modified || 0;
  }
});

// ---- Check backend health ----
async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    const data = await res.json();
    if (data.status === "ok") {
      statusDot.className = "dot online";
      statusText.textContent = data.model_loaded
        ? "Backend online · Model ready"
        : "Backend online · Model loading...";
    } else {
      throw new Error("not ok");
    }
  } catch {
    statusDot.className = "dot offline";
    statusText.textContent = "Backend offline (start FastAPI server)";
  }
}

checkBackend();

// ---- Open full UI ----
openUIBtn.addEventListener("click", () => {
  chrome.tabs.create({ url: chrome.runtime.getURL("../frontend/index.html") });
});
