"""
AI Chat Application
===================
Production-ready ASGI application configured for Vercel Serverless deployment and local execution.
Provides a modern chat interface and Gemini API backend.
"""

import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Load environment variables from .env if present
load_dotenv()

# Top-level ASGI entrypoint expected by Vercel
app = FastAPI(title="Gemini AI Chat Studio", version="1.0.0")

# Pricing per million tokens (USD)
PRICING_MAP = {
    "gemini-3.8-flash": {"in": 0.15, "out": 0.60},
    "gemini-3.5-flash-lite": {"in": 0.075, "out": 0.30},
    "gemini-4-argon": {"in": 1.25, "out": 5.00},
}


class MessageItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[MessageItem]
    system_instruction: Optional[str] = None
    model: str = "gemini-3.8-flash"
    temperature: float = 0.7
    top_p: float = 0.95
    max_output_tokens: int = 1024
    api_key: Optional[str] = None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def calculate_cost(model: str, prompt_toks: int, out_toks: int) -> float:
    rates = PRICING_MAP.get(model, {"in": 0.15, "out": 0.60})
    return (prompt_toks / 1_000_000 * rates["in"]) + (out_toks / 1_000_000 * rates["out"])


def get_server_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if key and key != "your_gemini_api_key_here":
        return key
    return ""


@app.get("/api/config")
async def get_config():
    server_key = get_server_key()
    return {
        "has_server_key": bool(server_key),
        "default_model": os.getenv("DEFAULT_MODEL", "gemini-3.8-flash"),
    }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    # Resolve API Key
    effective_key = req.api_key.strip() if req.api_key else ""
    if not effective_key or effective_key == "your_gemini_api_key_here":
        effective_key = get_server_key()

    # Calculate prompt tokens
    full_prompt_text = (req.system_instruction or "") + " " + " ".join(m.content for m in req.messages)
    p_toks = estimate_tokens(full_prompt_text)

    # 1. Live LLM execution if key available
    if effective_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=effective_key)

            config = types.GenerateContentConfig(
                system_instruction=req.system_instruction if req.system_instruction and req.system_instruction.strip() else None,
                temperature=req.temperature,
                top_p=req.top_p,
                max_output_tokens=req.max_output_tokens,
            )

            contents = [
                types.Content(
                    role="user" if m.role == "user" else "model",
                    parts=[types.Part.from_text(text=m.content)],
                )
                for m in req.messages
            ]

            response = client.models.generate_content(
                model=req.model,
                contents=contents,
                config=config,
            )

            reply_text = response.text or ""
            c_toks = estimate_tokens(reply_text)
            cost = calculate_cost(req.model, p_toks, c_toks)

            return {
                "reply": reply_text,
                "prompt_tokens": p_toks,
                "completion_tokens": c_toks,
                "cost_usd": cost,
                "is_simulated": False,
            }

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini API Error: {str(exc)}",
            )

    # 2. Simulated response fallback if no key is provided
    last_user_msg = req.messages[-1].content if req.messages else ""
    c_toks = 85
    cost = calculate_cost(req.model, p_toks, c_toks)

    simulated_reply = (
        f"**[Simulation Mode — {req.model}]**\n\n"
        f"I received your message: *\"{last_user_msg}\"*\n\n"
        f"- **Active Persona:** `{req.system_instruction[:60] if req.system_instruction else 'Default Assistant'}...`\n"
        f"- **Parameters:** Temperature `{req.temperature}` | Top-P `{req.top_p}` | Max Tokens `{req.max_output_tokens}`\n\n"
        f"💡 *To enable live AI generation, supply your Gemini API key in the sidebar settings or set `GEMINI_API_KEY` on Vercel.*"
    )

    return {
        "reply": simulated_reply,
        "prompt_tokens": p_toks,
        "completion_tokens": c_toks,
        "cost_usd": cost,
        "is_simulated": True,
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the complete, standalone AI Chat Studio interface."""
    return HTMLResponse(content=INDEX_HTML)


# Built-in Frontend Application
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gemini AI Chat Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <style>
    pre code { background: #1e293b !important; border-radius: 0.5rem; padding: 1rem; }
    .prose p { margin-bottom: 0.75rem; }
    .prose p:last-child { margin-bottom: 0; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 h-screen flex overflow-hidden font-sans">

  <!-- Sidebar -->
  <aside class="w-80 bg-slate-900 border-r border-slate-800 flex flex-col justify-between p-4 overflow-y-auto">
    <div class="space-y-4">
      <div class="flex items-center gap-2">
        <span class="text-2xl">✨</span>
        <h1 class="text-lg font-bold tracking-wide">AI Chat Studio</h1>
      </div>

      <!-- Authentication -->
      <div class="space-y-1">
        <label class="text-xs font-semibold text-slate-400">Gemini API Key</label>
        <input id="apiKeyInput" type="password" placeholder="Server key active (or paste here)"
               class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
        <p id="keyStatus" class="text-xs text-emerald-400 mt-1">Checking server key...</p>
      </div>

      <hr class="border-slate-800">

      <!-- Persona -->
      <div class="space-y-1">
        <label class="text-xs font-semibold text-slate-400">Persona Preset</label>
        <select id="personaPreset" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none">
          <option value="engineer">Senior Software Engineer</option>
          <option value="resume">ATS Resume Strategist</option>
          <option value="tutor">Socratic Tutor</option>
          <option value="concise">Concise Assistant</option>
          <option value="custom">Custom Instruction</option>
        </select>
        <textarea id="systemPrompt" rows="3" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-xs focus:outline-none mt-2"></textarea>
      </div>

      <hr class="border-slate-800">

      <!-- Model Parameters -->
      <div class="space-y-3">
        <div>
          <label class="text-xs font-semibold text-slate-400">Target Model</label>
          <select id="modelSelect" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none mt-1">
            <option value="gemini-3.8-flash">gemini-3.8-flash</option>
            <option value="gemini-3.5-flash-lite">gemini-3.5-flash-lite</option>
            <option value="gemini-4-argon">gemini-4-argon</option>
          </select>
        </div>

        <div>
          <div class="flex justify-between text-xs text-slate-400">
            <span>Temperature</span>
            <span id="tempVal">0.7</span>
          </div>
          <input id="tempSlider" type="range" min="0" max="1.5" step="0.05" value="0.7" class="w-full accent-indigo-500">
        </div>

        <div>
          <div class="flex justify-between text-xs text-slate-400">
            <span>Context Window (Turns)</span>
            <span id="windowVal">6</span>
          </div>
          <input id="windowSlider" type="range" min="1" max="15" step="1" value="6" class="w-full accent-indigo-500">
        </div>
      </div>

      <hr class="border-slate-800">

      <!-- Metrics -->
      <div class="bg-slate-950/60 p-3 rounded-lg border border-slate-800 space-y-1 text-xs">
        <div class="flex justify-between">
          <span class="text-slate-400">Prompt Tokens:</span>
          <span id="promptToks" class="font-mono text-slate-200">0</span>
        </div>
        <div class="flex justify-between">
          <span class="text-slate-400">Output Tokens:</span>
          <span id="outToks" class="font-mono text-slate-200">0</span>
        </div>
        <div class="flex justify-between pt-1 border-t border-slate-800">
          <span class="text-slate-400">Est. Cost (USD):</span>
          <span id="costVal" class="font-mono text-emerald-400">$0.000000</span>
        </div>
      </div>
    </div>

    <!-- Actions -->
    <div class="space-y-2 pt-4">
      <button onclick="clearChat()" class="w-full bg-slate-800 hover:bg-slate-700 text-xs py-2 rounded-lg font-medium transition">🗑️ Clear Chat</button>
      <div class="flex gap-2">
        <button onclick="exportHistory('json')" class="w-1/2 bg-slate-800 hover:bg-slate-700 text-xs py-1.5 rounded-lg transition">Export JSON</button>
        <button onclick="exportHistory('md')" class="w-1/2 bg-slate-800 hover:bg-slate-700 text-xs py-1.5 rounded-lg transition">Export MD</button>
      </div>
    </div>
  </aside>

  <!-- Main Chat -->
  <main class="flex-1 flex flex-col h-full bg-slate-950">
    <!-- Header -->
    <header class="h-14 border-b border-slate-800 px-6 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span class="text-lg">🤖</span>
        <div>
          <h2 class="text-sm font-semibold text-slate-200" id="headerTitle">Interactive Chat</h2>
          <p class="text-xs text-slate-400" id="headerSubtitle">Connected to Gemini</p>
        </div>
      </div>
      <div class="text-xs text-slate-400 font-mono bg-slate-900 border border-slate-800 px-3 py-1 rounded-full">
        Vercel Global Edge
      </div>
    </header>

    <!-- Message Stream -->
    <div id="chatContainer" class="flex-1 overflow-y-auto p-6 space-y-4">
      <div class="max-w-3xl mx-auto flex gap-3 text-slate-300">
        <div class="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center font-bold text-xs flex-shrink-0">AI</div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-sm leading-relaxed prose prose-invert">
          Hello! How can I assist you today?
        </div>
      </div>
    </div>

    <!-- Input Bar -->
    <div class="p-4 border-t border-slate-800 bg-slate-900/50">
      <div class="max-w-3xl mx-auto flex gap-2">
        <textarea id="userInput" rows="1" placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
                  class="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 resize-none"></textarea>
        <button id="sendBtn" onclick="sendMessage()" class="bg-indigo-600 hover:bg-indigo-500 px-5 rounded-xl font-medium text-sm flex items-center justify-center transition disabled:opacity-50">
          Send
        </button>
      </div>
    </div>
  </main>

  <script>
    const PERSONAS = {
      engineer: "You are an experienced senior software engineer and mentor. Provide clean, idiomatic, well-commented code, explain trade-offs, and suggest architecture best practices.",
      resume: "You are an expert executive recruiter and ATS resume specialist. Review achievements, quantify impact using the STAR/XYZ framework, and optimize keywords.",
      tutor: "You are a patient, insightful educator. Instead of giving answers immediately, guide the learner through thought-provoking questions, analogies, and conceptual hints.",
      concise: "You are a direct, helpful, and concise assistant. Answer all questions accurately and succinctly in under 3 sentences unless explicitly asked to elaborate.",
      custom: "You are an intelligent, empathetic, and knowledgeable AI assistant."
    };

    let messages = [];
    let totalPromptToks = 0;
    let totalOutToks = 0;
    let totalCostUsd = 0.0;

    // Elements
    const apiKeyInput = document.getElementById('apiKeyInput');
    const keyStatus = document.getElementById('keyStatus');
    const personaPreset = document.getElementById('personaPreset');
    const systemPrompt = document.getElementById('systemPrompt');
    const modelSelect = document.getElementById('modelSelect');
    const tempSlider = document.getElementById('tempSlider');
    const tempVal = document.getElementById('tempVal');
    const windowSlider = document.getElementById('windowSlider');
    const windowVal = document.getElementById('windowVal');
    const userInput = document.getElementById('userInput');
    const chatContainer = document.getElementById('chatContainer');
    const sendBtn = document.getElementById('sendBtn');

    // Init
    systemPrompt.value = PERSONAS.engineer;
    tempSlider.oninput = () => tempVal.innerText = tempSlider.value;
    windowSlider.oninput = () => windowVal.innerText = windowSlider.value;
    personaPreset.onchange = () => systemPrompt.value = PERSONAS[personaPreset.value];

    // Check config
    fetch('/api/config')
      .then(res => res.json())
      .then(data => {
        if (data.has_server_key) {
          keyStatus.innerText = "✅ Server GEMINI_API_KEY detected";
          keyStatus.className = "text-xs text-emerald-400 mt-1";
        } else {
          keyStatus.innerText = "ℹ️ Simulation mode (paste key above for live LLM)";
          keyStatus.className = "text-xs text-amber-400 mt-1";
        }
      });

    // Enter to send
    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    async function sendMessage() {
      const text = userInput.value.trim();
      if (!text) return;

      userInput.value = '';
      sendBtn.disabled = true;

      // Add user message
      messages.push({ role: "user", content: text });
      appendMessage("user", text);

      // Loading bubble
      const loadingId = appendLoading();
      chatContainer.scrollTop = chatContainer.scrollHeight;

      // Sliding window context
      const turnLimit = parseInt(windowSlider.value) * 2;
      const activeHistory = messages.slice(-turnLimit);

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: activeHistory,
            system_instruction: systemPrompt.value,
            model: modelSelect.value,
            temperature: parseFloat(tempSlider.value),
            api_key: apiKeyInput.value.trim() || undefined
          })
        });

        const data = await response.json();
        removeLoading(loadingId);

        if (!response.ok) {
          appendMessage("assistant", `⚠️ **Error:** ${data.detail || 'Failed to generate response'}`);
        } else {
          messages.push({ role: "assistant", content: data.reply });
          appendMessage("assistant", data.reply);

          totalPromptToks += data.prompt_tokens || 0;
          totalOutToks += data.completion_tokens || 0;
          totalCostUsd += data.cost_usd || 0.0;

          document.getElementById('promptToks').innerText = totalPromptToks;
          document.getElementById('outToks').innerText = totalOutToks;
          document.getElementById('costVal').innerText = `$${totalCostUsd.toFixed(6)}`;
        }
      } catch (err) {
        removeLoading(loadingId);
        appendMessage("assistant", `⚠️ **Connection Error:** ${err.message}`);
      } finally {
        sendBtn.disabled = false;
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }

    function appendMessage(role, text) {
      const isUser = role === "user";
      const div = document.createElement('div');
      div.className = "max-w-3xl mx-auto flex gap-3 text-slate-300";
      div.innerHTML = `
        <div class="w-8 h-8 rounded-full ${isUser ? 'bg-indigo-500' : 'bg-slate-700'} flex items-center justify-center font-bold text-xs flex-shrink-0">
          ${isUser ? 'U' : 'AI'}
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-sm leading-relaxed prose prose-invert overflow-x-auto flex-1">
          ${marked.parse(text)}
        </div>
      `;
      chatContainer.appendChild(div);
      div.querySelectorAll('pre code').forEach((block) => hljs.highlightElement(block));
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function appendLoading() {
      const id = 'loading-' + Date.now();
      const div = document.createElement('div');
      div.id = id;
      div.className = "max-w-3xl mx-auto flex gap-3 text-slate-300";
      div.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center font-bold text-xs flex-shrink-0 animate-pulse">AI</div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-sm text-slate-400 italic">
          Thinking...
        </div>
      `;
      chatContainer.appendChild(div);
      return id;
    }

    function removeLoading(id) {
      const el = document.getElementById(id);
      if (el) el.remove();
    }

    function clearChat() {
      messages = [];
      totalPromptToks = 0;
      totalOutToks = 0;
      totalCostUsd = 0.0;
      document.getElementById('promptToks').innerText = "0";
      document.getElementById('outToks').innerText = "0";
      document.getElementById('costVal').innerText = "$0.000000";
      chatContainer.innerHTML = `
        <div class="max-w-3xl mx-auto flex gap-3 text-slate-300">
          <div class="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center font-bold text-xs flex-shrink-0">AI</div>
          <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-sm leading-relaxed prose prose-invert">
            Chat cleared! How can I assist you now?
          </div>
        </div>
      `;
    }

    function exportHistory(format) {
      if (messages.length === 0) return alert("No messages to export!");
      let content = "";
      let filename = "";
      let type = "";

      if (format === 'json') {
        content = JSON.stringify(messages, null, 2);
        filename = "chat_history.json";
        type = "application/json";
      } else {
        content = "# Chat Conversation\\n\\n" + messages.map(m => `### ${m.role.toUpperCase()}\\n${m.content}\\n`).join('\\n---\\n');
        filename = "chat_history.md";
        type = "text/markdown";
      }

      const blob = new Blob([content], { type });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }
  </script>
</body>
</html>
"""

# Allow running directly via: python app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
