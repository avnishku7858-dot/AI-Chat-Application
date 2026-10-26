# AI Chat Application

An interactive, production-ready AI Chat Application built with **Streamlit** and the official **Google GenAI SDK** (`google-genai`).

---

## 📁 Project Structure

```
AI-Chat-Application/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml
```

---

## ✨ Features

- **Multi-Source Authentication**:
  - Automatically loads API key from `.streamlit/secrets.toml`.
  - Falls back to `.env` or system environment variables (`GEMINI_API_KEY`).
  - Interactive sidebar password field for ad-hoc runtime key entry.
  - Zero-friction educational simulation mode if no API key is set.
- **Real-Time Token Streaming**: Streams generated tokens seamlessly using `client.models.generate_content_stream`.
- **Persona & Custom Instructions**: Presets for *Senior Software Engineer*, *ATS Resume Strategist*, *Socratic Tutor*, and *Concise Assistant*, plus full custom system instruction capability.
- **Model & Hyperparameter Control**: Easily switch between `gemini-3.8-flash`, `gemini-3.5-flash-lite`, and `gemini-4-argon`, adjusting Temperature, Top-P, and Max Tokens.
- **Context Sliding Window**: Intelligent conversation history pruning preserving recent dialog turns without overflowing token limits.
- **Token & Cost Telemetry**: Live heuristic token estimation and cost tracking in USD.
- **Export & Reset**: Download chat transcripts in **JSON** or formatted **Markdown**, or reset session state with one click.

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Installation
```bash
# Navigate to the project directory
cd AI-Chat-Application

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure API Key
Open `.streamlit/secrets.toml` and provide your Gemini API key:

```toml
GEMINI_API_KEY = "your_actual_gemini_api_key_here"
```

*(Alternatively, you can set `export GEMINI_API_KEY="..."` or create a `.env` file.)*

### 4. Run the Application
```bash
streamlit run app.py
```

Streamlit will launch local browser preview at `http://localhost:8501`.

---

## 🔒 Security Best Practices
- Never commit `.streamlit/secrets.toml` with real credentials to a public git repository. `.gitignore` is pre-configured to keep your credentials secure.
- For **Streamlit Community Cloud** deployment, add your `GEMINI_API_KEY` under **App Settings > Secrets**.
