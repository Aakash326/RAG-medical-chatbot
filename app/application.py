# app.py (OpenAI version)
from flask import Flask, render_template, request, session, redirect, url_for
from flask import Response, jsonify
from flask import stream_with_context
from typing import List, Dict, Generator
import time
from dotenv import load_dotenv
from markupsafe import Markup
import os

# ---- ENV / OpenAI client ----
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment/.env")

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Flask setup ----
app = Flask(__name__)
app.secret_key = os.urandom(24)

def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly and concisely. "
    "If the question depends on prior context, use the message history."
)


# ---- Runtime settings (simple "advanced" controls) ----
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

def build_messages(history: List[Dict], system_prompt: str = SYSTEM_PROMPT) -> List[Dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        messages.append({"role": role, "content": content})
    return messages

def call_openai_chat(
    history: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """
    Synchronous (non-streaming) call that returns the final assistant text.
    """
    messages = build_messages(history)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def stream_openai_chat(
    history: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Generator[str, None, None]:
    """
    Streaming generator yielding partial tokens. Use with Flask Response.
    """
    messages = build_messages(history)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        except Exception:
            # keep the stream resilient
            continue

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    # Advanced controls via query params or session defaults
    if "settings" not in session:
        session["settings"] = {
            "model": DEFAULT_MODEL,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
    # Allow overrides via query string, e.g., /?model=gpt-4o&temperature=0.2
    model_q = request.args.get("model")
    temp_q = request.args.get("temperature")
    max_t_q = request.args.get("max_tokens")
    if model_q:
        session["settings"]["model"] = model_q
    if temp_q:
        try:
            session["settings"]["temperature"] = float(temp_q)
        except ValueError:
            pass
    if max_t_q:
        try:
            session["settings"]["max_tokens"] = int(max_t_q)
        except ValueError:
            pass

    if request.method == "POST":
        user_input = request.form.get("prompt")
        if user_input:
            msgs = session["messages"]
            msgs.append({"role": "user", "content": user_input})
            session["messages"] = msgs

            try:
                s = session["settings"]
                result = call_openai_chat(
                    session["messages"],
                    model=s.get("model", DEFAULT_MODEL),
                    temperature=float(s.get("temperature", DEFAULT_TEMPERATURE)),
                    max_tokens=int(s.get("max_tokens", MAX_TOKENS)),
                )
                msgs.append({"role": "assistant", "content": result})
                session["messages"] = msgs
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return render_template("index.html", messages=session["messages"], error=error_msg)

        return redirect(url_for("index"))

    return render_template("index.html", messages=session.get("messages", []))

@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))

# ---- Advanced API endpoints ----

@app.post("/api/chat")
def api_chat():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("prompt") or "").strip()
    if "messages" not in session:
        session["messages"] = []
    if not user_input:
        return jsonify({"error": "prompt is required"}), 400

    session["messages"].append({"role": "user", "content": user_input})
    s = session.get("settings", {})
    try:
        answer = call_openai_chat(
            session["messages"],
            model=s.get("model", DEFAULT_MODEL),
            temperature=float(s.get("temperature", DEFAULT_TEMPERATURE)),
            max_tokens=int(s.get("max_tokens", MAX_TOKENS)),
        )
        session["messages"].append({"role": "assistant", "content": answer})
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/api/chat/stream")
def api_chat_stream():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("prompt") or "").strip()
    if "messages" not in session:
        session["messages"] = []
    if not user_input:
        return Response("prompt is required", status=400, mimetype="text/plain")

    session["messages"].append({"role": "user", "content": user_input})
    s = session.get("settings", {})
    gen = stream_openai_chat(
        session["messages"],
        model=s.get("model", DEFAULT_MODEL),
        temperature=float(s.get("temperature", DEFAULT_TEMPERATURE)),
        max_tokens=int(s.get("max_tokens", MAX_TOKENS)),
    )

    def event_stream():
        for token in gen:
            yield token
        # After stream ends, we also persist the final message by recomputing once quickly
        # (clients often send the concatenated text back; this is a fallback)
    return Response(stream_with_context(event_stream()), mimetype="text/plain")

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": session.get("settings", {}).get("model", DEFAULT_MODEL)})

if __name__ == "__main__":
    # Bind to all interfaces for container/K8s
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)