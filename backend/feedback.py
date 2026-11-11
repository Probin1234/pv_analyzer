"""
LLM part (Generative AI): uses a free LLM (Groq, Gemini, or Ollama on your own computer) for two jobs:
  1. evaluate_content()  - read the transcript of ONE video and score the content
  2. write_feedback()    - read the numbers of ALL a student's videos and write feedback

If no LLM key is set (or the LLM is down) the system still works: content is left empty
and the feedback is written by simple rules (rule_based_feedback).
"""
import json
import re
import time

import numpy as np
import pandas as pd
import requests

from . import config
from .config import METRICS


class LLMUnavailable(Exception):
    """Raised when there is no API key or the LLM can't be reached."""


# ------------------------------------------------------------------ talking to the LLM
def _post(url, headers, payload, timeout=None):
    """POST with a couple of retries for 'too many requests' / server hiccups."""
    last = None
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout or config.LLM_TIMEOUT_SEC)
        except requests.RequestException as e:
            last = e
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503):
            last = RuntimeError(f"LLM busy (HTTP {r.status_code})")
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise LLMUnavailable(f"LLM error HTTP {r.status_code}: {r.text[:200]}")
        return r.json()
    raise LLMUnavailable(f"Could not reach the LLM: {last}")


def call_llm(system, user, json_mode=False):
    """Send one prompt to the chosen free LLM and return its text answer."""
    provider = config.LLM_PROVIDER
    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise LLMUnavailable("GROQ_API_KEY is not set in .env")
        payload = {
            "model": config.GROQ_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = _post("https://api.groq.com/openai/v1/chat/completions",
                     {"Authorization": f"Bearer {config.GROQ_API_KEY}"}, payload)
        return data["choices"][0]["message"]["content"]

    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise LLMUnavailable("GEMINI_API_KEY is not set in .env")
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": config.LLM_TEMPERATURE},
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{config.GEMINI_MODEL}:generateContent")
        data = _post(url, {"x-goog-api-key": config.GEMINI_API_KEY}, payload)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise LLMUnavailable(f"Unexpected Gemini answer: {str(data)[:200]}")

    if provider == "ollama":
        payload = {
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "options": {"temperature": config.LLM_TEMPERATURE, "num_ctx": config.OLLAMA_CONTEXT},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if json_mode:
            payload["format"] = "json"
        try:
            data = _post(f"{config.OLLAMA_URL}/api/chat", {}, payload, timeout=config.OLLAMA_TIMEOUT_SEC)
        except LLMUnavailable as e:
            msg = str(e)
            if "Could not reach" in msg:
                msg += " - is Ollama running? Open the Ollama app, or run 'ollama serve'."
            elif "HTTP 404" in msg:
                msg += f" - the model is not installed. Run: ollama pull {config.OLLAMA_MODEL}"
            raise LLMUnavailable(msg)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError):
            raise LLMUnavailable(f"Unexpected Ollama answer: {str(data)[:200]}")

    raise LLMUnavailable(f"Unknown LLM_PROVIDER '{provider}' (use 'groq', 'gemini' or 'ollama')")


def _parse_json(text):
    """LLMs sometimes wrap JSON in ```json fences or chatter. Dig the JSON out."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


# ------------------------------------------------------------------ job 1: content score
CONTENT_SYSTEM = (
    "You are a fair, consistent university presentation coach. You grade the CONTENT of a "
    "student's short video talk from its transcript. The task given to students was: talk about "
    "what you have learned and about presentation skills, with only your face and voice. "
    "Ignore speech-to-text spelling mistakes. Reply with JSON only."
)

CONTENT_PROMPT = """Grade this transcript. Give each part a whole number from 0 to 25:
- relevance: does it talk about what the student learned and/or presentation skills?
- structure: clear opening, middle and ending?
- clarity: are the ideas easy to follow and explained simply?
- depth: specific points, examples or reflection instead of vague statements?

Return exactly this JSON:
{{"relevance": 0, "structure": 0, "clarity": 0, "depth": 0,
  "comment": "one or two plain sentences: what was good and what to improve"}}

TRANSCRIPT:
\"\"\"{transcript}\"\"\""""


def evaluate_content(transcript):
    """Return (content_score 0-100, comment). Raises LLMUnavailable if the LLM can't be used."""
    if len(transcript.split()) < 30:
        return float("nan"), "Too short to judge the content."
    raw = call_llm(CONTENT_SYSTEM, CONTENT_PROMPT.format(transcript=transcript[:12000]),
                   json_mode=True)
    data = _parse_json(raw)
    parts = [float(np.clip(float(data.get(k, 0)), 0, 25))
             for k in ("relevance", "structure", "clarity", "depth")]
    return float(sum(parts)), str(data.get("comment", "")).strip()


# ------------------------------------------------------------------ job 2: progress feedback
FEEDBACK_SYSTEM = (
    "You are a kind but honest presentation coach writing feedback to a university student. "
    "Use simple, plain English. Only use the numbers you are given; never invent numbers. "
    "Filler-word counts are only estimates, so never claim a student has zero or no filler words."
)

FEEDBACK_PROMPT = """Write feedback for {student} based on the data below (scores are out of 100).
Their first videos are compared with their latest videos.

SKILL SCORES (start -> now):
{skills}

MEASUREMENTS (start -> now):
{raw}

COMMENTS ON CONTENT (oldest to newest):
{comments}

IMPORTANT RULES:
- Filler word numbers are ESTIMATES (speech-to-text deletes many "um" and "uh"). Never say "zero" or "no" filler words. If the number is below 1 per minute say "few filler words were detected"; otherwise say "about N per minute".
- A comfortable speaking pace is 120-160 words per minute.

Write at most 180 words in this order:
1. One sentence on the overall improvement.
2. Two things they do well now.
3. Two things to work on, each with one concrete practice tip they can try in next week's videos.
Be encouraging and specific. No headings, just short paragraphs or a short list."""

RAW_FOR_PROMPT = ["words_per_min", "fillers_per_min", "long_pauses_per_min",
                  "eye_contact_pct", "pitch_std_st", "energy_std_db"]


def _fmt(x):
    return "n/a" if pd.isna(x) else f"{x:.0f}"


def write_feedback(student, table, videos):
    """LLM-written feedback. `table` = improvement table, `videos` = this student's results."""
    videos = videos.sort_values("date")
    k = max(1, min(3, len(videos) // 2))
    skills = "\n".join(
        f"- {r.Skill}: {_fmt(r.Start)} -> {_fmt(r.Now)}" for r in table.itertuples()
    )
    raw_lines = []
    for col in RAW_FOR_PROMPT:
        if col in videos:
            s = pd.to_numeric(videos[col], errors="coerce")
            raw_lines.append(f"- {config.RAW_LABELS[col]}: "
                             f"{s.head(k).mean():.1f} -> {s.tail(k).mean():.1f}")
    comments = [c for c in videos.get("content_comment", []) if isinstance(c, str) and c.strip()]
    comments = comments[:2] + comments[-2:] if len(comments) > 4 else comments
    prompt = FEEDBACK_PROMPT.format(
        student=student, skills=skills, raw="\n".join(raw_lines),
        comments="\n".join(f"- {c}" for c in comments) or "- (none)",
    )
    return call_llm(FEEDBACK_SYSTEM, prompt).strip()


# ------------------------------------------------------------------ fallback (no LLM needed)
def rule_based_feedback(student, table):
    skills = table[table["Skill"].isin(METRICS)].dropna(subset=["Now"])
    overall = table[table["Skill"] == "Overall"].iloc[0]
    if skills.empty:
        return "Not enough data to write feedback yet."
    weakest = skills.sort_values("Now").iloc[0]
    lines = []
    if pd.isna(overall["Change"]):
        lines.append(f"{student} has one analyzed video so far, so improvement can't be measured yet. "
                     f"Overall score: {overall['Now']:.0f}/100.")
    else:
        lines.append(f"{student}'s overall score went from {overall['Start']:.0f} to "
                     f"{overall['Now']:.0f} ({overall['Change']:+.0f} points).")
        changes = skills.dropna(subset=["Change"])
        if not changes.empty:
            best = changes.sort_values("Change", ascending=False).iloc[0]
            lines.append(f"Biggest improvement: {best['Skill']} ({best['Change']:+.0f} points).")
    lines.append(f"Needs the most work: {weakest['Skill']} (now {weakest['Now']:.0f}/100).")
    return "\n\n".join(lines)
