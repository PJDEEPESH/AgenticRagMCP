// When loaded from the FastAPI server (http://localhost:8000) use relative URLs.
// When opened directly as a file:// URL, fall back to the absolute backend URL.
const API = location.protocol === "file:" ? "http://localhost:8000" : "";
const SESSION_ID = "sess-" + Math.random().toString(36).slice(2, 10);

let uploadedDocs = [];
let messages = [];
let mcpTrace = [];
let isLoading = false;
let latestEvalScores = null;

// ── API helpers ─────────────────────────────────────────────────────────────

async function apiUpload(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(API + "/upload", { method: "POST", body: fd });
  if (!res.ok) {
    let detail = "Upload failed";
    try { const err = await res.json(); detail = err.detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

async function apiChat(question, sessionId) {
  const res = await fetch(API + "/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  if (!res.ok) {
    let detail = "Chat failed";
    try { const err = await res.json(); detail = err.detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

async function apiEvaluate(question, answer, contextChunks, sessionId) {
  const res = await fetch(API + "/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      answer,
      context_chunks: contextChunks,
      session_id: sessionId,
    }),
  });
  if (!res.ok) throw new Error("Evaluation failed");
  return res.json();
}

async function apiReset() {
  const res = await fetch(API + "/reset", { method: "DELETE" });
  return res.json();
}

// ── Utilities ────────────────────────────────────────────────────────────────

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch { return ""; }
}

function formatRelTime(iso) {
  try { return new Date(iso).toLocaleTimeString(); } catch { return ""; }
}

function getFormatBadge(filename) {
  const ext = (filename.split(".").pop() || "").toLowerCase();
  const map = {
    pdf: "fmt-pdf", csv: "fmt-csv", xlsx: "fmt-xlsx",
    pptx: "fmt-pptx", docx: "fmt-docx", txt: "fmt-txt", md: "fmt-md",
  };
  return `<span class="badge ${map[ext] || "badge-gray"}">${ext.toUpperCase() || "?"}</span>`;
}

function getMcpBadgeClass(type) {
  const t = String(type || "");
  if (t.includes("INGEST")) return "badge-green";
  if (t.includes("RETRIEVAL")) return "badge-cyan";
  if (t.includes("LLM")) return "badge-violet";
  if (t.includes("EVAL")) return "badge-amber";
  if (t.includes("ERROR")) return "badge-red";
  return "badge-gray";
}

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function scoreColor(v) {
  if (v >= 0.8) return "var(--accent-green)";
  if (v >= 0.5) return "var(--accent-amber)";
  return "var(--accent-red)";
}

// Minimal safe Markdown renderer
function renderMarkdown(src) {
  const text = String(src == null ? "" : src).replace(/\r\n/g, "\n");
  const lines = text.split("\n");
  const out = [];
  let i = 0;

  const inline = (s) => {
    let t = escapeHtml(s);
    t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    t = t.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?:;]|$)/g, "$1<em>$2</em>");
    return t;
  };

  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*$/.test(line)) { i++; continue; }

    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i]))
        items.push(lines[i++].replace(/^\s*[-*]\s+/, ""));
      out.push("<ul>" + items.map((it) => `<li>${inline(it)}</li>`).join("") + "</ul>");
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i]))
        items.push(lines[i++].replace(/^\s*\d+\.\s+/, ""));
      out.push("<ol>" + items.map((it) => `<li>${inline(it)}</li>`).join("") + "</ol>");
      continue;
    }
    const para = [];
    while (
      i < lines.length &&
      !/^\s*$/.test(lines[i]) &&
      !/^\s*[-*]\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i])
    ) para.push(lines[i++]);
    out.push("<p>" + para.map(inline).join("<br>") + "</p>");
  }
  return out.join("");
}

// ── Header stats ─────────────────────────────────────────────────────────────

function updateHeaderStats() {
  const el = document.getElementById("header-stats");
  if (!el) return;
  const docCount = uploadedDocs.length;
  const totalChunks = uploadedDocs.reduce((a, d) => a + (d.chunks_stored || d.chunk_count || 0), 0);
  el.innerHTML =
    `<span class="badge badge-cyan">${docCount} docs</span>` +
    `<span class="badge badge-green">${totalChunks} chunks</span>`;
}

// ── Document list ─────────────────────────────────────────────────────────────

function renderDocList() {
  const el = document.getElementById("doc-list");
  if (!el) return;
  if (uploadedDocs.length === 0) {
    el.innerHTML = `<div class="empty-text">No documents uploaded yet.</div>`;
    return;
  }
  el.innerHTML = uploadedDocs.map((d) => {
    const fname = escapeHtml(d.filename || "");
    const cnt = d.chunks_stored || d.chunk_count || 0;
    return `
      <div class="doc-item">
        <div class="filename" title="${fname}">${fname}</div>
        <div class="meta">${getFormatBadge(fname)}<span class="badge badge-green">${cnt} chunks</span></div>
      </div>`;
  }).join("");
}

// ── Source chunk renderer ─────────────────────────────────────────────────────

function buildSourceCards(sourceChunks) {
  if (!sourceChunks || sourceChunks.length === 0) return "";

  const cards = sourceChunks.map((c, idx) => {
    const txt = String(c.chunk || "");
    // Extract [source: file | location] tag
    const tagMatch = txt.match(/\[source:[^\]]+\]/);
    const tag = tagMatch ? tagMatch[0] : "";
    const body = (tag ? txt.replace(tag, "") : txt).trim();

    // Parse file and location from tag
    const tagInner = tag.replace(/^\[source:\s*/, "").replace(/\]$/, "");
    const [filePart, ...locParts] = tagInner.split("|").map((s) => s.trim());
    const locLabel = locParts.join(" | ") || "";

    const score = Math.max(0, Math.min(1, c.score || 0));
    const pct = (score * 100).toFixed(0);
    const barColor = scoreColor(score);

    return `
      <div class="src-chunk" id="src-chunk-${idx}">
        <div class="src-chunk-header">
          <div class="src-meta">
            <span class="src-file">${escapeHtml(filePart || "source")}</span>
            ${locLabel ? `<span class="src-loc">${escapeHtml(locLabel)}</span>` : ""}
          </div>
          <div class="src-score-wrap">
            <div class="src-score-bar" style="--fill:${barColor}; --pct:${pct}%"></div>
            <span class="src-score-pct">${pct}%</span>
          </div>
        </div>
        <div class="src-chunk-body">${escapeHtml(body.slice(0, 280))}${body.length > 280 ? "…" : ""}</div>
      </div>`;
  }).join("");

  return `
    <div class="src-section">
      <div class="src-section-header" onclick="this.parentElement.classList.toggle('open')">
        <span class="src-label">
          <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Sources (${sourceChunks.length})
        </span>
        <span class="src-chevron">▼</span>
      </div>
      <div class="src-cards">${cards}</div>
    </div>`;
}

// ── Evaluation scores renderer ───────────────────────────────────────────────

function buildEvalScores(scores) {
  if (!scores) return "";
  const metrics = [
    { key: "faithfulness",       label: "Faithfulness",       tip: "Are all answer claims grounded in the context?" },
    { key: "answer_relevancy",   label: "Answer Relevancy",   tip: "Does the answer address the question?" },
    { key: "context_precision",  label: "Context Precision",  tip: "Are the retrieved chunks relevant?" },
    { key: "overall",            label: "Overall",            tip: "Average of all three metrics" },
  ];
  const rows = metrics.map(({ key, label, tip }) => {
    const v = typeof scores[key] === "number" ? scores[key] : 0;
    const pct = (v * 100).toFixed(0);
    const color = scoreColor(v);
    return `
      <div class="eval-row" title="${escapeHtml(tip)}">
        <span class="eval-label">${escapeHtml(label)}</span>
        <div class="eval-bar-wrap">
          <div class="eval-bar-fill" style="width:${pct}%; background:${color}"></div>
        </div>
        <span class="eval-val" style="color:${color}">${pct}%</span>
      </div>`;
  }).join("");

  return `<div class="eval-card"><div class="eval-title">RAGAS Scores</div>${rows}</div>`;
}

// ── RAG Health panel ─────────────────────────────────────────────────────────

function renderHealthPanel() {
  const el = document.getElementById("health-panel");
  if (!el) return;
  if (!latestEvalScores) {
    el.innerHTML = `<div class="empty-text">Click "Evaluate" on any response to see RAGAS scores here.</div>`;
    return;
  }
  el.innerHTML = buildEvalScores(latestEvalScores);
}

// ── Right panel tab switching ─────────────────────────────────────────────────

function switchRightTab(tab) {
  document.querySelectorAll(".rtab-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === tab)
  );
  document.getElementById("trace-body").style.display = tab === "trace" ? "" : "none";
  document.getElementById("health-body").style.display = tab === "health" ? "" : "none";
}

// ── Messages renderer ─────────────────────────────────────────────────────────

function renderMessages() {
  const el = document.getElementById("chat-messages");
  if (!el) return;

  if (messages.length === 0 && !isLoading) {
    el.innerHTML = `
      <div class="empty-state">
        <div class="icon">🤖</div>
        <div class="title">Ask anything about your documents</div>
        <div class="subtitle">Upload PDFs, slides, spreadsheets or text files on the left, then start chatting. Sources are cited with every answer.</div>
      </div>`;
    document.getElementById("msg-count").textContent = "0";
    return;
  }

  let html = "";
  messages.forEach((m, i) => {
    if (m.role === "user") {
      html += `
        <div class="msg-wrap user">
          <div class="msg-bubble">${escapeHtml(m.text)}</div>
          <div class="msg-meta"><span>${formatTime(m.time)}</span></div>
        </div>`;
    } else {
      const trace8 = m.traceId ? m.traceId.slice(0, 8) : "";
      const evalHtml = m.evalScores ? buildEvalScores(m.evalScores) : "";
      const evalBtnHtml = !m.evalScores && m.sourceChunks && m.sourceChunks.length > 0
        ? `<button class="eval-btn" onclick="handleEvaluate(${i})" id="eval-btn-${i}">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            Evaluate
           </button>`
        : "";

      html += `
        <div class="msg-wrap assistant">
          <div class="msg-bubble md">${renderMarkdown(m.text)}</div>
          ${buildSourceCards(m.sourceChunks)}
          ${evalHtml}
          <div class="msg-meta">
            <span>${formatTime(m.time)}</span>
            ${trace8 ? `<span class="trace-id">trace=${escapeHtml(trace8)}</span>` : ""}
            ${evalBtnHtml}
          </div>
        </div>`;
    }
  });

  if (isLoading) {
    html += `
      <div class="thinking">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
        <span>Thinking</span>
      </div>`;
  }

  el.innerHTML = html;
  el.scrollTop = el.scrollHeight;
  document.getElementById("msg-count").textContent = String(messages.length);
}

// ── Evaluate handler ─────────────────────────────────────────────────────────

window.handleEvaluate = async function (msgIdx) {
  const msg = messages[msgIdx];
  if (!msg || msg.role !== "assistant") return;

  const btn = document.getElementById(`eval-btn-${msgIdx}`);
  if (btn) { btn.disabled = true; btn.textContent = "Evaluating…"; }

  // Find the user question that preceded this assistant message
  let question = "";
  for (let i = msgIdx - 1; i >= 0; i--) {
    if (messages[i].role === "user") { question = messages[i].text; break; }
  }

  const contextChunks = (msg.sourceChunks || []).map((c) => c.chunk || "");

  try {
    const data = await apiEvaluate(question, msg.text, contextChunks, SESSION_ID);
    const scores = data.scores || {};
    messages[msgIdx].evalScores = scores;
    latestEvalScores = scores;

    if (Array.isArray(data.mcp_trace)) mcpTrace = mcpTrace.concat(data.mcp_trace);

    renderMessages();
    renderMcpTrace();
    renderHealthPanel();
    switchRightTab("health");
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = "Evaluate"; }
    console.error("Evaluation error:", e);
  }
};

// ── MCP trace renderer ────────────────────────────────────────────────────────

function renderMcpTrace() {
  const el = document.getElementById("trace-list");
  const cnt = document.getElementById("trace-count");
  if (cnt) cnt.textContent = String(mcpTrace.length);
  if (!el) return;
  if (mcpTrace.length === 0) {
    el.innerHTML = `<div class="empty-text">No MCP messages yet. Upload a file or ask a question.</div>`;
    return;
  }
  el.innerHTML = mcpTrace.map((msg, i) => {
    const cls = getMcpBadgeClass(msg.type);
    const payloadStr = JSON.stringify(msg.payload, null, 2);
    return `
      <div class="trace-card">
        <div class="trace-header" onclick="togglePayload(${i})">
          <span class="sender">${escapeHtml(msg.sender)}</span>
          <span class="arrow">→</span>
          <span class="receiver">${escapeHtml(msg.receiver)}</span>
          <span class="badge ${cls}">${escapeHtml(msg.type)}</span>
          <span class="ts">${formatRelTime(msg.timestamp)}</span>
        </div>
        <div class="trace-payload" id="payload-${i}">
          <pre>${escapeHtml(payloadStr)}</pre>
        </div>
      </div>`;
  }).join("");
}

window.togglePayload = function (i) {
  const el = document.getElementById("payload-" + i);
  if (el) el.classList.toggle("open");
};

// ── Drop zone ────────────────────────────────────────────────────────────────

function initDropZone() {
  const zone = document.getElementById("drop-zone");
  const input = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  if (!zone || !input) return;

  zone.addEventListener("click", (e) => { if (!e.target.closest("#browse-btn")) input.click(); });
  if (browseBtn) browseBtn.addEventListener("click", (e) => { e.stopPropagation(); input.click(); });

  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length > 0) handleMultiUpload(files);
  });
  input.addEventListener("change", () => {
    const files = Array.from(input.files || []);
    if (files.length > 0) { handleMultiUpload(files); input.value = ""; }
  });
}

async function handleMultiUpload(files) {
  if (!files || files.length === 0) return;
  const status = document.getElementById("upload-status");
  const errEl = document.getElementById("upload-error");
  const zone = document.getElementById("drop-zone");

  if (errEl) { errEl.style.display = "none"; errEl.textContent = ""; }
  if (zone) zone.style.opacity = "0.6";

  const errors = [];
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (status) {
      status.style.display = "flex";
      status.classList.remove("success");
      status.innerHTML = `<span class="spinner"></span><span>Uploading ${i + 1}/${files.length}: ${escapeHtml(file.name)}…</span>`;
    }
    try {
      const data = await apiUpload(file);
      uploadedDocs.push({ filename: data.filename, chunks_stored: data.chunks_stored });
      if (Array.isArray(data.mcp_trace)) mcpTrace = mcpTrace.concat(data.mcp_trace);
      renderDocList();
      renderMcpTrace();
      updateHeaderStats();
    } catch (e) {
      errors.push(`${file.name}: ${e.message || "Upload failed"}`);
    }
  }

  if (zone) zone.style.opacity = "1";

  if (errors.length > 0) {
    if (errEl) { errEl.innerHTML = errors.map(escapeHtml).join("<br>"); errEl.style.display = "block"; }
    if (status) status.style.display = "none";
  } else {
    if (status) {
      status.classList.add("success");
      const totalChunks = uploadedDocs.slice(-files.length).reduce((a, d) => a + (d.chunks_stored || 0), 0);
      status.innerHTML = `<span>✓</span><span>${files.length} file${files.length > 1 ? "s" : ""} uploaded — ${totalChunks} chunks stored</span>`;
      setTimeout(() => { status.style.display = "none"; }, 3500);
    }
  }
}

async function handleUpload(file) {
  return handleMultiUpload([file]);
}

// ── Chat input ───────────────────────────────────────────────────────────────

function initChatInput() {
  const ta = document.getElementById("chat-input");
  const btn = document.getElementById("send-btn");
  if (!ta || !btn) return;

  ta.addEventListener("input", () => {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
    btn.disabled = ta.value.trim().length === 0;
  });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  });
  btn.addEventListener("click", () => handleSend());
}

async function handleSend() {
  const ta = document.getElementById("chat-input");
  const btn = document.getElementById("send-btn");
  if (!ta) return;
  const q = ta.value.trim();
  if (!q || isLoading) return;

  messages.push({ role: "user", text: q, time: new Date().toISOString() });
  ta.value = "";
  ta.style.height = "auto";
  isLoading = true;
  if (btn) btn.disabled = true;
  renderMessages();

  try {
    const data = await apiChat(q, SESSION_ID);
    messages.push({
      role: "assistant",
      text: data.answer || "",
      sourceChunks: data.source_chunks || [],
      traceId: data.trace_id || "",
      time: new Date().toISOString(),
      evalScores: null,
    });
    if (Array.isArray(data.mcp_trace)) mcpTrace = mcpTrace.concat(data.mcp_trace);
  } catch (e) {
    messages.push({ role: "assistant", text: "Error: " + (e.message || "Request failed"), time: new Date().toISOString() });
  } finally {
    isLoading = false;
    if (btn) btn.disabled = false;
    renderMessages();
    renderMcpTrace();
  }
}

// ── Reset ────────────────────────────────────────────────────────────────────

async function handleReset() {
  if (!confirm("Reset all uploads, chat history, and MCP trace?")) return;
  try { await apiReset(); } catch (_) {}
  uploadedDocs = [];
  messages = [];
  mcpTrace = [];
  latestEvalScores = null;
  renderDocList();
  renderMessages();
  renderMcpTrace();
  renderHealthPanel();
  updateHeaderStats();
}

// ── Bootstrap ────────────────────────────────────────────────────────────────

function bootstrap() {
  initDropZone();
  initChatInput();

  const resetBtn = document.getElementById("reset-btn");
  if (resetBtn) resetBtn.addEventListener("click", handleReset);

  // Right panel tab switching
  document.querySelectorAll(".rtab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchRightTab(btn.dataset.tab));
  });

  renderDocList();
  renderMessages();
  renderMcpTrace();
  renderHealthPanel();
  updateHeaderStats();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}
