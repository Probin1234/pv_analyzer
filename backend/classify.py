"""
Video type part (NLP + Generative AI): read the transcript and say
  - what KIND of video it is (presentation, motivational talk, podcast, lecture ...)
  - what the person is talking ABOUT (a short topic + a one-sentence summary)

Main way:  ask the free LLM (the same one used for the content score) to pick one type
           from the list in config.VIDEO_TYPES.
Back-up:   simple keyword rules, so it still works when there is no LLM key or the LLM is down.

We only have the transcript, so this is a judgement from the WORDS (and how the talk is
organised), not from the picture or from hearing two different voices.
"""
import re
from collections import Counter

from . import config, feedback

# Words and phrases that hint at each type (used only by the back-up rules).
# A phrase matches from its first letter, so "motivat" also catches "motivation".
KEYWORDS = {
    "Presentation": ["my presentation", "presentation skills", "today i will", "my topic", "agenda",
                     "in conclusion", "to summarize", "i learned", "this week", "i will talk about",
                     "thank you for listening", "i would like to present"],
    "Motivational talk": ["believe", "never give up", "success", "dream", "motivat", "inspir",
                          "you can do", "passion", "failure", "hard work", "confidence", "mindset",
                          "keep going", "your goals"],
    "Podcast / conversation": ["podcast", "episode", "my guest", "today's guest", "thanks for having",
                               "listeners", "subscribe", "welcome back", "welcome to the show",
                               "let's dive in"],
    "Lecture / tutorial": ["tutorial", "step by step", "how to", "let's learn", "in this lesson",
                           "for example", "definition", "formula", "first step", "next step",
                           "we will learn"],
    "Storytelling": ["once upon", "one day", "when i was", "i remember", "back then", "story"],
    "Review / opinion": ["review", "in my opinion", "i recommend", "pros and cons", "worth it",
                         "rating", "compared to"],
    "Speech / announcement": ["ladies and gentlemen", "dear friends", "good morning everyone",
                              "announce", "welcome everyone", "honorable"],
}

STOPWORDS = set("""
that this with have from they their there what when which would about just like going think really
very your will been were them then than some also into more much here because things thing know want
does dont don't i'm it's that's can't i've i'll you're we're there's okay yeah so, actually maybe
could should many make made kind lots still every other only over after before where while those
these being doing getting something anything everything today week thank
""".split())

TYPE_SYSTEM = (
    "You recognize what KIND of spoken video a transcript comes from and what it is about. "
    "You only have the words (you cannot hear or see the video), so judge from the wording, the way "
    "the speaker talks to the audience, and how the talk is organised. "
    "Ignore speech-to-text spelling mistakes. Reply with JSON only."
)

TYPE_PROMPT = """Read this transcript of ONE short video by a university student.

Choose the single best video type from this list:
{types}

Notes:
- Choose "Podcast / conversation" only if there are signs of a podcast or chat (a host, a guest, listeners, an episode, questions to another person). One person simply speaking casually is NOT a podcast.
- If nothing fits well, choose "Other".

Return exactly this JSON:
{{"type": "one name from the list", "confidence": 0, "topic": "3 to 8 words", "summary": "one plain sentence about what the speaker talks about"}}
confidence is a whole number from 0 to 100.

TRANSCRIPT:
\"\"\"{transcript}\"\"\""""


def _result(video_type, topic, summary, confidence, method, note=""):
    return {"video_type": video_type, "topic": topic, "video_summary": summary,
            "type_confidence": confidence, "method": method, "note": note}


def _clean_type(name):
    """Make sure the type is one of the allowed names (the LLM may answer in a slightly different form)."""
    wanted = str(name).strip().lower()
    for allowed in config.VIDEO_TYPES:
        if allowed.lower() == wanted:
            return allowed
    return "Other"


def classify_with_llm(transcript):
    """Ask the LLM. Raises an error if the LLM can't be used (the caller then uses the rules)."""
    types = "\n".join(f"- {name}: {desc}" for name, desc in config.VIDEO_TYPES.items())
    raw = feedback.call_llm(TYPE_SYSTEM,
                            TYPE_PROMPT.format(types=types, transcript=transcript[:12000]),
                            json_mode=True)
    data = feedback._parse_json(raw)
    try:
        confidence = int(max(0, min(100, float(data.get("confidence", 0)))))
    except (TypeError, ValueError):
        confidence = 0
    return _result(_clean_type(data.get("type", "Other")),
                   str(data.get("topic", "")).strip()[:80],
                   str(data.get("summary", "")).strip()[:250],
                   confidence, "llm")


def classify_with_rules(transcript):
    """Back-up: count the hint words for each type and pick the type with the most hits."""
    lower = transcript.lower()
    scores = {name: sum(len(re.findall(r"\b" + re.escape(w), lower)) for w in words)
              for name, words in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    hits = scores[best]
    tokens = [t for t in re.findall(r"[a-z']{4,}", lower) if t not in STOPWORDS]
    topic = ", ".join(w for w, _ in Counter(tokens).most_common(3))
    if hits == 0:
        return _result("Other", topic, "", 0, "rules")
    return _result(best, topic, "", min(75, 30 + 10 * hits), "rules")


def classify_video(transcript):
    """
    Returns a dict: video_type, topic, video_summary, type_confidence (0-100),
    method ("llm", "rules" or "none") and note (a warning text, or "").
    Never raises: if the LLM fails it falls back to the keyword rules.
    """
    transcript = (transcript or "").strip()
    if len(transcript.split()) < 30:
        return _result("Other", "", "Too short to tell what kind of video this is.", 0, "none")
    try:
        return classify_with_llm(transcript)
    except Exception as e:
        info = classify_with_rules(transcript)
        info["note"] = f"video type guessed with simple keyword rules, not the LLM ({str(e)[:100]})"
        return info
