/**
 * background.js — ToxiClear Service Worker
 * -----------------------------------------
 * Manages extension state, badge updates, and cross-tab communication.
 */

let globalStats = { analyzed: 0, modified: 0 };

// ---- Install / Startup ----

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({ enabled: true });
  updateBadge(true);
  console.log("ToxiClear installed.");
});

chrome.runtime.onStartup.addListener(() => {
  chrome.storage.sync.get(["enabled"], (result) => {
    updateBadge(result.enabled !== false);
  });
});

// ---- Message handling ----

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "STATS_UPDATE") {
    globalStats.analyzed += (msg.stats.analyzed || 0);
    globalStats.modified += (msg.stats.modified || 0);
    updateBadgeCount(globalStats.modified);
  }

  if (msg.type === "GET_GLOBAL_STATS") {
    sendResponse(globalStats);
  }

  if (msg.type === "RESET_STATS") {
    globalStats = { analyzed: 0, modified: 0 };
    updateBadgeCount(0);
    sendResponse({ ok: true });
  }
});

// ---- Badge helpers ----

function updateBadge(enabled) {
  chrome.action.setBadgeBackgroundColor({ color: enabled ? "#6c63ff" : "#555" });
  chrome.action.setBadgeText({ text: enabled ? "ON" : "OFF" });
}

function updateBadgeCount(count) {
  if (count > 0) {
    chrome.action.setBadgeText({ text: count > 99 ? "99+" : String(count) });
    chrome.action.setBadgeBackgroundColor({ color: "#ff4d6d" });
  }
}

// ---- Toggle from popup ----

chrome.storage.onChanged.addListener((changes) => {
  if (changes.enabled) {
    updateBadge(changes.enabled.newValue);
  }
});
