// EU Dual-Use Classifier — vanilla JS, no dependencies.
// Security note: model/dataset text is ALWAYS rendered via textContent /
// createTextNode, never innerHTML — nothing the model (or a prompt-injecting
// user) says can become markup in someone's browser.

(function () {
  "use strict";

  const cfg = window.CLASSIFIER_CONFIG || {};
  // owner/tester bypass of the per-visitor cap: visit once with ?tester=CODE
  const testerParam = new URLSearchParams(location.search).get("tester");
  if (testerParam) sessionStorage.setItem("tester_key", testerParam);
  const testerKey = sessionStorage.getItem("tester_key");
  const $ = (id) => document.getElementById(id);
  const workerReady = cfg.WORKER_URL && !cfg.WORKER_URL.startsWith("REPLACE");

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

  // ---------- health check ----------
  // Nothing is shown while the assistant is available; the only status the
  // page ever surfaces is the offline banner when the daily budget is spent.
  const OFFLINE_MESSAGE =
    "The assistant is at today's limit and will be back tomorrow — Browse mode " +
    "still searches the full Annex I, free and instant.";
  async function refreshStatus() {
    if (!workerReady) return;
    try {
      const resp = await fetch(cfg.WORKER_URL.replace(/\/$/, "") + "/api/health", {
        headers: testerKey ? { "x-tester-key": testerKey } : {},
      });
      const h = await resp.json();
      if (!h.assistant_available) {
        showBudgetBanner(OFFLINE_MESSAGE, false);
        input.disabled = true;
        input.placeholder = "The assistant is back tomorrow — Browse mode works fully.";
        sendBtn.disabled = true;
      }
    } catch {
      // health unavailable — stay silent; a failed chat POST reports itself
    }
  }
  refreshStatus();

  // ---------- chat ----------
  let transcript = []; // opaque server-owned history, re-sent verbatim
  const messagesEl = $("messages");
  const form = $("chat-form");
  const input = $("chat-input");
  const sendBtn = $("send");
  const banner = $("budget-banner");
  const examplesEl = $("examples");

  for (const chip of examplesEl.querySelectorAll(".chip")) {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.prompt || chip.textContent;
      input.focus();
    });
  }

  // Render assistant text with ONLY **bold** honoured — built via DOM nodes
  // (createTextNode/strong.textContent), so no markup of any kind can inject.
  function renderInline(parent, text) {
    const parts = String(text).split(/\*\*/);
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 1 && i < parts.length - (parts.length % 2 === 0 ? 1 : 0)) {
        const b = document.createElement("strong");
        b.textContent = parts[i];
        parent.appendChild(b);
      } else {
        parent.appendChild(document.createTextNode(parts[i]));
      }
    }
  }

  function addBubble(cls, text) {
    const div = document.createElement("div");
    div.className = "msg " + cls;
    if (cls === "assistant") {
      // heading/quote markers render as emphasis, never as raw # or >
      const lines = String(text).split("\n");
      for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        const heading = /^#{1,6}\s+(.*)$/.exec(line);
        if (heading) {
          const b = document.createElement("strong");
          renderInline(b, heading[1].replace(/\*\*/g, ""));
          div.appendChild(b);
        } else {
          renderInline(div, line.replace(/^>\s?/, ""));
        }
        if (i < lines.length - 1) div.appendChild(document.createTextNode("\n"));
      }
    } else {
      div.textContent = text;
    }
    messagesEl.appendChild(div);
    div.scrollIntoView({ behavior: "smooth", block: "end" });
    return div;
  }

  // Clipboard with fallback: navigator.clipboard is refused in some browsers /
  // permission states, so fall back to the classic textarea+execCommand path.
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        ta.remove();
        return ok;
      } catch {
        return false;
      }
    }
  }

  function verdictAsText(v) {
    const label = { listed: "LISTED IN ANNEX I", not_listed: "NOT LISTED IN ANNEX I", needs_expert: "NEEDS EXPERT REVIEW" };
    const lines = [
      "EU Dual-Use Classifier — indicative triage (not legal advice)",
      "Result: " + (label[v.status] || v.status) + (v.entry_codes.length ? " — " + v.entry_codes.join(", ") : ""),
    ];
    for (const r of v.reasoning || []) {
      const tag = r.met === false ? " (tested, ruled out)" : "";
      lines.push("- " + r.dotted_path + tag + ': "' + r.verbatim_quote + '" — ' + r.explanation);
    }
    for (const c of v.caveats || []) lines.push("Caveat: " + c);
    if (v.disclaimer) lines.push(v.disclaimer);
    lines.push("Corpus version: " + v.corpus_version + (v.corpus_sha256 ? " (sha256 " + v.corpus_sha256.slice(0, 12) + ")" : ""));
    lines.push("Generated at " + new Date().toISOString() + " — https://rikiosso.github.io/dualuse-classifier/");
    return lines.join("\n");
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
      const path = document.createElement("a");
      path.className = "code";
      path.textContent = r.dotted_path + (r.met === false ? " (tested, ruled out)" : "") + ": ";
      const entryCode = (r.entry_code || "").toUpperCase();
      path.href = "#" + entryCode;
      path.title = "Open " + entryCode + " in Browse mode";
      path.addEventListener("click", (ev) => {
        ev.preventDefault();
        openEntryInBrowse(entryCode);
      });
      q.appendChild(path);
      q.appendChild(document.createTextNode('"' + r.verbatim_quote + '"'));
      const expl = document.createElement("div");
      renderInline(expl, r.explanation);
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

    const copyBtn = document.createElement("button");
    copyBtn.className = "copy";
    copyBtn.type = "button";
    copyBtn.textContent = "Copy result";
    copyBtn.addEventListener("click", async () => {
      const ok = await copyText(verdictAsText(v));
      copyBtn.textContent = ok ? "Copied ✓" : "Copy failed — select the text manually";
      setTimeout(() => (copyBtn.textContent = "Copy result"), 2500);
    });
    card.appendChild(copyBtn);

    messagesEl.appendChild(card);
    lastVerdictCard = card; // the licensing pathway merges into this card
    card.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  // The licensing determination renders INSIDE the classification card when
  // one exists — one ask, one card. Falls back to a standalone card only if
  // no verdict card is on screen.
  let lastVerdictCard = null;

  function addPathwayCard(pw) {
    let card;
    if (lastVerdictCard) {
      card = lastVerdictCard;
      lastVerdictCard = null;
      card.classList.add("pathway", pw.outcome);
      const oldCopy = card.querySelector(".copy");
      if (oldCopy) oldCopy.remove();
      const divider = document.createElement("div");
      divider.className = "pathway-divider";
      card.appendChild(divider);
    } else {
      card = document.createElement("div");
      card.className = "verdict pathway " + pw.outcome;
    }
    const h = document.createElement("h3");
    h.textContent = "Licensing pathway";
    card.appendChild(h);

    const label = {
      gea_available: "EU General Export Authorisation available",
      individual_licence_required: "Individual authorisation required",
      sanctions_review_required: "⚠️ Sanctions regime — legal review required",
    };
    const status = document.createElement("p");
    const st = document.createElement("span");
    st.className = "status " + (pw.outcome === "gea_available" ? "not_listed" : pw.outcome === "individual_licence_required" ? "needs_expert" : "listed");
    st.textContent = label[pw.outcome] || pw.outcome;
    status.appendChild(st);
    if (pw.eligible_gea) {
      status.appendChild(document.createTextNode(" — "));
      const code = document.createElement("span");
      code.className = "code";
      code.textContent = pw.eligible_gea;
      status.appendChild(code);
    }
    status.appendChild(document.createTextNode(" · destination: " + pw.destination));
    card.appendChild(status);

    for (const c of pw.conditions_quoted || []) {
      const q = document.createElement("blockquote");
      const gid = document.createElement("span");
      gid.className = "code";
      gid.textContent = c.gea_id + ": ";
      q.appendChild(gid);
      q.appendChild(document.createTextNode('"' + c.verbatim_quote + '"'));
      const expl = document.createElement("div");
      renderInline(expl, c.explanation);
      q.appendChild(expl);
      card.appendChild(q);
    }

    const cav = document.createElement("div");
    cav.className = "caveats";
    const parts = [...(pw.caveats || [])];
    if (pw.disclaimer) parts.push(pw.disclaimer);
    parts.push("Corpus version: " + pw.corpus_version);
    cav.textContent = parts.join(" · ");
    card.appendChild(cav);

    const review = document.createElement("p");
    review.className = "review-cta";
    review.appendChild(
      document.createTextNode("This pathway is a draft determination — have it confirmed by the lawyer who built this tool. "),
    );
    const a = document.createElement("a");
    a.href = "https://www.linkedin.com/in/ricardo-ossorio";
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = "Request legal review →";
    review.appendChild(a);
    card.appendChild(review);

    if (!card.parentNode) messagesEl.appendChild(card);
    card.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  function showBudgetBanner(text, switchToBrowse) {
    banner.textContent = text;
    banner.classList.remove("hidden");
    if (switchToBrowse) setTimeout(() => showTab("browse"), 1800);
  }

  // After a LISTED verdict the licensing stage continues automatically —
  // the user should never be re-asked for facts already given. One silent
  // continuation per conversation; further turns are the user's.
  let autoContinued = false;
  const AUTO_CONTINUE =
    "Proceed to the licensing pathway using the facts already provided in this " +
    "conversation. Ask only for genuinely missing facts.";

  async function sendTurn(text, silent) {
    if (!workerReady) {
      showBudgetBanner("The assistant backend is not deployed yet — Browse mode works fully.", true);
      return;
    }
    if (!silent) addBubble("user", text);
    transcript = transcript.concat([{ role: "user", content: text }]);
    examplesEl.classList.add("hidden");
    sendBtn.disabled = true;
    const thinking = addBubble(
      "thinking",
      silent ? "Determining the licensing pathway…" : "Consulting Annex I…",
    );
    try {
      const resp = await fetch(cfg.WORKER_URL.replace(/\/$/, "") + "/api/chat", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...(testerKey ? { "x-tester-key": testerKey } : {}),
        },
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
        if (data.verdict.status === "listed" && !autoContinued) {
          autoContinued = true;
          sendBtn.disabled = false;
          return sendTurn(AUTO_CONTINUE, true);
        }
      } else if (data.type === "pathway") {
        transcript = data.messages;
        if (data.text) addBubble("assistant", data.text);
        addPathwayCard(data.pathway);
      } else {
        const reason = data.reason || "unknown";
        if (reason === "daily_budget_exhausted") {
          showBudgetBanner(OFFLINE_MESSAGE, true);
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
  }

  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    sendTurn(text, false);
  });

  // ---------- browse ----------
  let annex = null;
  const searchEl = $("search");
  const resultsEl = $("browse-results");
  const metaEl = $("browse-meta");

  async function ensureAnnexLoaded() {
    if (annex) return annex;
    metaEl.textContent = "Loading Annex I dataset…";
    try {
      const resp = await fetch(cfg.ANNEX_URL);
      annex = await resp.json();
      metaEl.textContent =
        annex.entry_count + " entries · consolidated version " + annex.corpus_version +
        (annex.valid_from ? " · in force since " + annex.valid_from : "");
      renderBrowse(searchEl.value || "");
    } catch {
      metaEl.textContent = "Could not load the dataset — please retry.";
    }
    return annex;
  }

  // append `text` to `parent`, wrapping case-insensitive matches of q in <mark>
  function appendHighlighted(parent, text, q) {
    if (!q) {
      parent.appendChild(document.createTextNode(text));
      return;
    }
    const lower = text.toLowerCase();
    let pos = 0;
    for (;;) {
      const hit = lower.indexOf(q, pos);
      if (hit === -1) break;
      parent.appendChild(document.createTextNode(text.slice(pos, hit)));
      const mark = document.createElement("mark");
      mark.textContent = text.slice(hit, hit + q.length);
      parent.appendChild(mark);
      pos = hit + q.length;
    }
    parent.appendChild(document.createTextNode(text.slice(pos)));
  }

  function entryDetails(e, q, open) {
    const det = document.createElement("details");
    det.className = "entry";
    det.id = "entry-" + e.entry_code;
    if (open) det.open = true;
    const sum = document.createElement("summary");
    const firstLine = e.verbatim_text.split("\n", 1)[0];
    appendHighlighted(sum, firstLine.length > 120 ? firstLine.slice(0, 117) + "…" : firstLine, q);
    det.appendChild(sum);
    const pre = document.createElement("pre");
    const paramSet = new Set(e.parameters || []);
    for (const line of e.verbatim_text.split("\n")) {
      const span = document.createElement("span");
      if (paramSet.has(line)) span.className = "param"; // threshold lines highlighted
      appendHighlighted(span, line, q);
      pre.appendChild(span);
      pre.appendChild(document.createTextNode("\n"));
    }
    det.appendChild(pre);
    det.addEventListener("toggle", () => {
      if (det.open) history.replaceState(null, "", "#" + e.entry_code);
    });
    return det;
  }

  function renderBrowse(query) {
    resultsEl.textContent = "";
    if (!annex) return;
    const q = query.trim().toLowerCase();
    const hits = annex.entries.filter(
      (e) => !q || e.entry_code.toLowerCase().includes(q) || e.verbatim_text.toLowerCase().includes(q),
    );
    const shown = hits.slice(0, 60);
    for (const e of shown) resultsEl.appendChild(entryDetails(e, q, false));
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

  async function openEntryInBrowse(code) {
    showTab("browse");
    const data = await ensureAnnexLoaded();
    if (!data) return;
    searchEl.value = code;
    resultsEl.textContent = "";
    const entry = data.entries.find((e) => e.entry_code === code);
    if (entry) {
      resultsEl.appendChild(entryDetails(entry, "", true));
      history.replaceState(null, "", "#" + code);
    } else {
      renderBrowse(code);
    }
  }

  let debounce = null;
  searchEl.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => renderBrowse(searchEl.value), 150);
  });

  // permalink: #3A001 opens that entry directly in Browse mode
  const hashCode = decodeURIComponent(location.hash.replace("#", "")).toUpperCase();
  if (/^\d[A-E]\d{3}$/.test(hashCode)) openEntryInBrowse(hashCode);
})();
