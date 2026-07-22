// EU Dual-Use Classifier — vanilla JS, no dependencies.
// Security note: model/dataset text is ALWAYS rendered via textContent, never
// innerHTML — nothing the model (or a prompt-injecting user) says can become
// markup in someone's browser.

(function () {
  "use strict";

  const cfg = window.CLASSIFIER_CONFIG || {};
  const $ = (id) => document.getElementById(id);

  // ---------- tabs ----------
  const panels = { chat: $("panel-chat"), browse: $("panel-browse") };
  const tabs = { chat: $("tab-chat"), browse: $("tab-browse") };
  function showTab(name) {
    for (const key of Object.keys(panels)) {
      panels[key].classList.toggle("hidden", key !== name);
      tabs[key].classList.toggle("active", key === name);
    }
    if (name === "browse") ensureAnnexLoaded();
  }
  tabs.chat.addEventListener("click", () => showTab("chat"));
  tabs.browse.addEventListener("click", () => showTab("browse"));

  // ---------- chat ----------
  let transcript = []; // opaque server-owned history, re-sent verbatim
  const messagesEl = $("messages");
  const form = $("chat-form");
  const input = $("chat-input");
  const sendBtn = $("send");
  const banner = $("budget-banner");

  function addBubble(cls, text) {
    const div = document.createElement("div");
    div.className = "msg " + cls;
    div.textContent = text;
    messagesEl.appendChild(div);
    div.scrollIntoView({ behavior: "smooth", block: "end" });
    return div;
  }

  function addVerdictCard(v) {
    const card = document.createElement("div");
    card.className = "verdict";
    const h = document.createElement("h3");
    h.textContent = "Classification result";
    card.appendChild(h);

    const status = document.createElement("p");
    const label = { listed: "Listed in Annex I", not_listed: "Not listed in Annex I", needs_expert: "Needs expert review" };
    const s = document.createElement("span");
    s.className = "status " + v.status;
    s.textContent = label[v.status] || v.status;
    status.appendChild(s);
    if (v.entry_codes && v.entry_codes.length) {
      status.appendChild(document.createTextNode(" — "));
      const codes = document.createElement("span");
      codes.className = "code";
      codes.textContent = v.entry_codes.join(", ");
      status.appendChild(codes);
    }
    card.appendChild(status);

    for (const r of v.reasoning || []) {
      const q = document.createElement("blockquote");
      const path = document.createElement("span");
      path.className = "code";
      path.textContent = r.dotted_path + ": ";
      q.appendChild(path);
      q.appendChild(document.createTextNode('"' + r.verbatim_quote + '"'));
      const expl = document.createElement("div");
      expl.textContent = r.explanation;
      q.appendChild(expl);
      card.appendChild(q);
    }

    const cav = document.createElement("div");
    cav.className = "caveats";
    const parts = [...(v.caveats || [])];
    if (v.disclaimer) parts.push(v.disclaimer);
    parts.push("Corpus version: " + v.corpus_version);
    cav.textContent = parts.join(" · ");
    card.appendChild(cav);

    messagesEl.appendChild(card);
    card.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  function showBudgetBanner(text, switchToBrowse) {
    banner.textContent = text;
    banner.classList.remove("hidden");
    if (switchToBrowse) setTimeout(() => showTab("browse"), 1800);
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    if (!cfg.WORKER_URL || cfg.WORKER_URL.startsWith("REPLACE")) {
      showBudgetBanner("The assistant backend is not deployed yet — Browse mode works fully.", true);
      return;
    }
    addBubble("user", text);
    transcript = transcript.concat([{ role: "user", content: text }]);
    input.value = "";
    sendBtn.disabled = true;
    const thinking = addBubble("thinking", "Consulting Annex I…");
    try {
      const resp = await fetch(cfg.WORKER_URL.replace(/\/$/, "") + "/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: transcript }),
      });
      const data = await resp.json();
      thinking.remove();
      if (data.type === "question") {
        transcript = data.messages;
        addBubble("assistant", data.text);
      } else if (data.type === "verdict") {
        transcript = data.messages;
        if (data.text) addBubble("assistant", data.text);
        addVerdictCard(data.verdict);
      } else {
        const reason = data.reason || "unknown";
        if (reason === "daily_budget_exhausted") {
          showBudgetBanner(
            "The AI assistant has used up today's demo budget. Meanwhile you can search Annex I directly in Browse mode — the assistant is back tomorrow.",
            true,
          );
        } else if (reason === "rate_limited") {
          showBudgetBanner("You've reached today's per-visitor conversation limit — Browse mode stays fully available.", true);
        } else if (reason === "conversation_too_long") {
          addBubble("assistant", "This conversation reached its length limit — please start a fresh one (reload the page).");
        } else {
          addBubble("assistant", "Something went wrong upstream — please try again in a moment.");
        }
      }
    } catch {
      thinking.remove();
      addBubble("assistant", "Network error — please try again.");
    } finally {
      sendBtn.disabled = false;
    }
  });

  // ---------- browse ----------
  let annex = null;
  const searchEl = $("search");
  const resultsEl = $("browse-results");
  const metaEl = $("browse-meta");

  async function ensureAnnexLoaded() {
    if (annex) return;
    metaEl.textContent = "Loading Annex I dataset…";
    try {
      const resp = await fetch(cfg.ANNEX_URL);
      annex = await resp.json();
      metaEl.textContent =
        annex.entry_count + " entries · consolidated version " + annex.corpus_version +
        (annex.valid_from ? " · in force since " + annex.valid_from : "");
      renderBrowse("");
    } catch {
      metaEl.textContent = "Could not load the dataset — please retry.";
    }
  }

  function renderBrowse(query) {
    resultsEl.textContent = "";
    if (!annex) return;
    const q = query.trim().toLowerCase();
    const hits = annex.entries.filter(
      (e) => !q || e.entry_code.toLowerCase().includes(q) || e.verbatim_text.toLowerCase().includes(q),
    );
    const shown = hits.slice(0, 60);
    for (const e of shown) {
      const det = document.createElement("details");
      det.className = "entry";
      const sum = document.createElement("summary");
      const firstLine = e.verbatim_text.split("\n", 1)[0];
      sum.textContent = firstLine.length > 120 ? firstLine.slice(0, 117) + "…" : firstLine;
      det.appendChild(sum);
      const pre = document.createElement("pre");
      pre.textContent = e.verbatim_text;
      det.appendChild(pre);
      resultsEl.appendChild(det);
    }
    if (hits.length > shown.length) {
      const more = document.createElement("p");
      more.className = "meta";
      more.textContent = "Showing " + shown.length + " of " + hits.length + " matches — refine the search.";
      resultsEl.appendChild(more);
    }
    if (hits.length === 0) {
      const none = document.createElement("p");
      none.className = "meta";
      none.textContent = "No entries match. Remember: catch-all controls can apply to unlisted items.";
      resultsEl.appendChild(none);
    }
  }

  let debounce = null;
  searchEl.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => renderBrowse(searchEl.value), 150);
  });
})();
