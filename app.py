"""
AI Chat Application
===================
A feature-packed, production-ready AI chat interface built with Streamlit and Google Gemini.

Features:
- Multi-tier API key resolution: Streamlit Secrets (.streamlit/secrets.toml) > Environment Variables (.env) > Sidebar UI input
- Real-time token streaming with Google GenAI SDK (`google-genai`)
- System Instruction & Persona customization (Technical Mentor, ATS Resume Strategist, Socratic Tutor, Custom)
- Multi-turn conversation management with a configurable sliding-window context buffer
- Live token estimation and per-session cost tracker (USD)
- Chat export (JSON & Markdown) and quick reset
- Graceful simulation fallback when no API key is provided
"""

import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Gemini AI Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Preset System Instructions / Personas
PERSONA_PRESETS = {
    "Senior Software Engineer": (
        "You are an experienced senior software engineer and mentor. Provide clean, idiomatic, "
        "well-commented code, explain trade-offs, and suggest architecture best practices."
    ),
    "ATS Resume Strategist": (
        "You are an expert executive recruiter and ATS resume specialist. Review achievements, "
        "quantify impact using the STAR/XYZ framework, and optimize keywords."
    ),
    "Socratic Tutor": (
        "You are a patient, insightful educator. Instead of giving answers immediately, guide the "
        "learner through thought-provoking questions, analogies, and conceptual hints."
    ),
    "Concise Assistant": (
        "You are a direct, helpful, and concise assistant. Answer all questions accurately and "
        "succinctly in under 3 sentences unless explicitly asked to elaborate."
    ),
    "Custom": ""
}

# Pricing per million tokens (USD)
PRICING_MAP = {
    "gemini-3.8-flash": {"in": 0.15, "out": 0.60},
    "gemini-3.5-flash-lite": {"in": 0.075, "out": 0.30},
    "gemini-4-argon": {"in": 1.25, "out": 5.00},
}


def estimate_tokens(text: str) -> int:
    """Heuristic approximation: ~4 characters per token."""
    return max(1, len(text) // 4) if text else 0


def calculate_cost(model: str, prompt_toks: int, out_toks: int) -> float:
    """Calculate query cost in USD based on model pricing."""
    rates = PRICING_MAP.get(model, {"in": 0.15, "out": 0.60})
    return (prompt_toks / 1_000_000 * rates["in"]) + (out_toks / 1_000_000 * rates["out"])


def resolve_api_key() -> str:
    """Retrieve API key from st.secrets, environment variables, or default empty string."""
    key = ""
    # 1. Try Streamlit Secrets
    try:
        if "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # 2. Try OS Environment
    if not key or key == "your_gemini_api_key_here":
        key = os.getenv("GEMINI_API_KEY", "")

    return key if key != "your_gemini_api_key_here" else ""


# =====================================================================
# Session State Initialization
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_prompt_tokens" not in st.session_state:
    st.session_state.total_prompt_tokens = 0

if "total_completion_tokens" not in st.session_state:
    st.session_state.total_completion_tokens = 0

if "total_cost_usd" not in st.session_state:
    st.session_state.total_cost_usd = 0.0


# =====================================================================
# Sidebar: Configuration & Controls
# =====================================================================
with st.sidebar:
    st.title("✨ App Settings")

    # 1. Authentication
    st.subheader("1. API Key")
    detected_key = resolve_api_key()

    user_api_key = st.text_input(
        "Gemini API Key",
        value=detected_key,
        type="password",
        placeholder="AIzaSy...",
        help="Reads automatically from .streamlit/secrets.toml or .env, or paste your key here.",
    )

    active_api_key = user_api_key.strip()
    is_live_key = bool(active_api_key and active_api_key != "your_gemini_api_key_here")

    if is_live_key:
        st.success("API Key Active (Ready for Live LLM)", icon="🟢")
    else:
        st.info("No active key detected. Running in educational simulation mode.", icon="ℹ️")

    st.divider()

    # 2. System Instructions / Personas
    st.subheader("2. Assistant Persona")
    selected_preset = st.selectbox("Persona Template", list(PERSONA_PRESETS.keys()))

    if selected_preset == "Custom":
        system_instruction = st.text_area(
            "Custom System Instruction",
            value="You are an intelligent, empathetic, and knowledgeable AI assistant.",
            height=90,
        )
    else:
        system_instruction = st.text_area(
            "Active System Instruction",
            value=PERSONA_PRESETS[selected_preset],
            height=90,
        )

    st.divider()

    # 3. Model Parameters
    st.subheader("3. Model & Hyperparameters")
    model_choice = st.selectbox(
        "Model Selection",
        ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-4-argon"],
        index=0,
    )

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, help="Higher values produce more creative output.")
    with col2:
        top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05, help="Nucleus sampling threshold.")

    max_tokens = st.slider("Max Output Tokens", 100, 2048, 1024, 64)
    history_limit = st.slider("Sliding Window Memory (Turns)", 1, 15, 6, help="Number of conversation turns passed as context.")

    st.divider()

    # 4. Token & Cost Telemetry
    st.subheader("4. Session Metrics")
    m1, m2 = st.columns(2)
    m1.metric("Prompt Tokens", st.session_state.total_prompt_tokens)
    m2.metric("Output Tokens", st.session_state.total_completion_tokens)
    st.metric("Estimated Cost", f"${st.session_state.total_cost_usd:.6f} USD")

    st.divider()

    # 5. Actions & Export
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_prompt_tokens = 0
        st.session_state.total_completion_tokens = 0
        st.session_state.total_cost_usd = 0.0
        st.rerun()

    if st.session_state.messages:
        # JSON Export
        export_json = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            "📥 Export History (JSON)",
            data=export_json,
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True,
        )

        # Markdown Export
        md_content = "# Chat Conversation Export\n\n"
        for m in st.session_state.messages:
            md_content += f"### {m['role'].capitalize()}\n{m['content']}\n\n---\n"
        st.download_button(
            "📄 Export History (Markdown)",
            data=md_content,
            file_name="chat_history.md",
            mime="text/markdown",
            use_container_width=True,
        )


# =====================================================================
# Main Chat Area
# =====================================================================
st.title("💬 Gemini AI Chat Studio")
st.caption(
    f"Active Model: **{model_choice}** | Temperature: `{temperature}` | Sliding Context Window: `{history_limit}` turns"
)

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
user_input = st.chat_input("Type your message here...")

if user_input:
    # 1. Append & render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Sliding window memory slice
    max_history_entries = history_limit * 2
    active_history = st.session_state.messages[-max_history_entries:]

    # 3. Stream or Generate Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        if is_live_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=active_api_key)

                config = types.GenerateContentConfig(
                    system_instruction=system_instruction if system_instruction.strip() else None,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_tokens,
                )

                # Format conversation history
                contents = []
                for entry in active_history:
                    contents.append(
                        types.Content(
                            role="user" if entry["role"] == "user" else "model",
                            parts=[types.Part.from_text(text=entry["content"])],
                        )
                    )

                # Stream response chunks
                response_stream = client.models.generate_content_stream(
                    model=model_choice,
                    contents=contents,
                    config=config,
                )

                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                # Telemetry updates
                p_toks = estimate_tokens(
                    (system_instruction or "") + " " + " ".join(m["content"] for m in active_history)
                )
                c_toks = estimate_tokens(full_response)
                cost = calculate_cost(model_choice, p_toks, c_toks)

                st.session_state.total_prompt_tokens += p_toks
                st.session_state.total_completion_tokens += c_toks
                st.session_state.total_cost_usd += cost

            except Exception as exc:
                full_response = (
                    f"**⚠️ API Error Encountered:**\n\n`{str(exc)}`\n\n"
                    "*Please verify your API key, target model selection, or network connectivity.*"
                )
                response_placeholder.markdown(full_response)
        else:
            # Interactive simulated response for instant zero-config testing
            simulated_chunks = [
                f"**[Simulated Response — {model_choice}]**\n\n",
                f"I received your query: *\"{user_input}\"*\n\n",
                f"- **Persona Applied:** `{system_instruction[:50]}...`\n",
                f"- **Hyperparameters:** Temperature `{temperature}`, Top-P `{top_p}`, Max Tokens `{max_tokens}`\n",
                f"- **Context Window:** Retaining last `{len(active_history)}` messages in sliding buffer.\n\n",
                "💡 *To switch to live Gemini generation, add your API key in `.streamlit/secrets.toml`, `.env`, or paste it in the sidebar.*"
            ]

            for chunk in simulated_chunks:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.06)

            response_placeholder.markdown(full_response)

            p_toks = estimate_tokens(user_input)
            c_toks = estimate_tokens(full_response)
            cost = calculate_cost(model_choice, p_toks, c_toks)

            st.session_state.total_prompt_tokens += p_toks
            st.session_state.total_completion_tokens += c_toks
            st.session_state.total_cost_usd += cost

    # 4. Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
