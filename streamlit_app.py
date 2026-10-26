"""
Streamlit Version of AI Chat Studio (Run locally with: streamlit run streamlit_app.py)
"""

import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Gemini AI Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

PRICING_MAP = {
    "gemini-3.8-flash": {"in": 0.15, "out": 0.60},
    "gemini-3.5-flash-lite": {"in": 0.075, "out": 0.30},
    "gemini-4-argon": {"in": 1.25, "out": 5.00},
}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def calculate_cost(model: str, prompt_toks: int, out_toks: int) -> float:
    rates = PRICING_MAP.get(model, {"in": 0.15, "out": 0.60})
    return (prompt_toks / 1_000_000 * rates["in"]) + (out_toks / 1_000_000 * rates["out"])


def resolve_api_key() -> str:
    key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    if not key or key == "your_gemini_api_key_here":
        key = os.getenv("GEMINI_API_KEY", "")

    return key if key != "your_gemini_api_key_here" else ""


if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_prompt_tokens" not in st.session_state:
    st.session_state.total_prompt_tokens = 0
if "total_completion_tokens" not in st.session_state:
    st.session_state.total_completion_tokens = 0
if "total_cost_usd" not in st.session_state:
    st.session_state.total_cost_usd = 0.0

with st.sidebar:
    st.title("✨ App Settings")
    detected_key = resolve_api_key()
    user_api_key = st.text_input(
        "Gemini API Key",
        value=detected_key,
        type="password",
        placeholder="AIzaSy...",
    )
    active_api_key = user_api_key.strip()
    is_live_key = bool(active_api_key and active_api_key != "your_gemini_api_key_here")

    if is_live_key:
        st.success("API Key Active", icon="🟢")
    else:
        st.info("Simulation mode", icon="ℹ️")

    st.divider()
    st.subheader("Persona")
    selected_preset = st.selectbox("Preset", list(PERSONA_PRESETS.keys()))
    if selected_preset == "Custom":
        system_instruction = st.text_area("Custom System Instruction", value="You are a helpful assistant.", height=90)
    else:
        system_instruction = st.text_area("Active Instruction", value=PERSONA_PRESETS[selected_preset], height=90)

    st.divider()
    st.subheader("Model")
    model_choice = st.selectbox("Model", ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-4-argon"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05)
    max_tokens = st.slider("Max Output Tokens", 100, 2048, 1024, 64)
    history_limit = st.slider("Memory (Turns)", 1, 15, 6)

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Prompt Tokens", st.session_state.total_prompt_tokens)
    m2.metric("Output Tokens", st.session_state.total_completion_tokens)
    st.metric("Cost", f"${st.session_state.total_cost_usd:.6f} USD")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_prompt_tokens = 0
        st.session_state.total_completion_tokens = 0
        st.session_state.total_cost_usd = 0.0
        st.rerun()

st.title("💬 Gemini AI Chat Studio")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    max_entries = history_limit * 2
    active_history = st.session_state.messages[-max_entries:]

    with st.chat_message("assistant"):
        placeholder = st.empty()
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
                contents = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=m["content"])],
                    )
                    for m in active_history
                ]
                stream = client.models.generate_content_stream(model=model_choice, contents=contents, config=config)
                for chunk in stream:
                    if chunk.text:
                        full_response += chunk.text
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                p_toks = estimate_tokens(system_instruction + " " + " ".join(m["content"] for m in active_history))
                c_toks = estimate_tokens(full_response)
                cost = calculate_cost(model_choice, p_toks, c_toks)
                st.session_state.total_prompt_tokens += p_toks
                st.session_state.total_completion_tokens += c_toks
                st.session_state.total_cost_usd += cost
            except Exception as e:
                full_response = f"**Error:** `{e}`"
                placeholder.markdown(full_response)
        else:
            full_response = f"**[Simulated Response - {model_choice}]** Hello! Add your Gemini API key in settings for live generation."
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
