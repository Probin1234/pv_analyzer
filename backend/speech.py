"""
Speech part (NLP): turn the audio into text with Whisper, then measure HOW the
person spoke: speed, filler words, pauses, vocabulary variety, sentence length.
SpaCy is used for sentences and word roots when it is installed (optional).
"""
import re
from functools import lru_cache

import numpy as np

from . import config

# Hesitation sounds and "verbal crutches"
FILLER_PATTERNS = [
    r"\b(?:um+|uh+|er+m?|ah+|hmm+|mm+)\b",
    r"\byou know\b",
    r"\bi mean\b",
    r"\bbasically\b",
    r"\blike\b(?=,)",          # "so, like, I think" (Whisper puts a comma after filler "like")
]


# the sounds themselves ("um", "uhh" ...), used to avoid counting the same filler twice
FILLER_SOUND = r"um+|uh+|er+m?|ah+|hmm+|mm+"


@lru_cache(maxsize=1)
def _whisper_model():
    import whisper
    return whisper.load_model(config.WHISPER_MODEL)


@lru_cache(maxsize=1)
def _nlp():
    """SpaCy English model, or None if it isn't installed (the code still works without it)."""
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner"])
    except Exception:
        return None


def transcribe(audio_path):
    """Run Whisper. Returns its full result (text + segments + word timestamps)."""
    import torch
    return _whisper_model().transcribe(
        str(audio_path),
        language=config.WHISPER_LANGUAGE or None,
        word_timestamps=True,
        initial_prompt=config.FILLER_PROMPT,
        fp16=torch.cuda.is_available(),
    )


def _voiced_runs(mask, max_hole=2):
    """Find stretches of True in a True/False list. Tiny 1-2 frame holes are ignored."""
    runs, start, last, hole = [], None, None, 0
    for i, on in enumerate(mask):
        if on:
            if start is None:
                start = i
            last, hole = i, 0
        elif start is not None:
            hole += 1
            if hole > max_hole:
                runs.append((start, last + 1))
                start, hole = None, 0
    if start is not None:
        runs.append((start, last + 1))
    return runs


def detect_hesitations(wav_path, words):
    """
    Whisper deletes most "um/uh", so the transcript can say "zero fillers" even when there are
    many. Here we LISTEN to the audio instead. Two signs of a hidden hesitation sound:
      1. In a gap between two words the person is still making a steady, flat-pitched sound.
      2. One short word is stretched over a long time ("sooo", "and...") while the voice is on.
    Returns how many were found. It is an ESTIMATE (Whisper's word times are not perfect).
    """
    import librosa

    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    hop, frame = 512, 2048
    step = hop / sr                                   # seconds per audio frame (about 0.032)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    f0 = librosa.yin(y, fmin=70, fmax=380, sr=sr, frame_length=frame, hop_length=hop)
    n = min(len(rms), len(f0))
    rms, f0 = rms[:n], f0[:n]
    loud = rms > 0.15 * np.percentile(rms, 95)        # same "is someone making sound" rule as voice.py

    found = 0
    margin = 0.05                                     # stay away from the edges of the neighbour words
    for a, b in zip(words, words[1:]):                # sign 1: sound inside a gap between words
        g0, g1 = a["end"] + margin, b["start"] - margin
        if g1 - g0 < config.HESITATION_MIN_SEC:
            continue
        i0, i1 = int(g0 / step), int(g1 / step)
        for s, e in _voiced_runs(loud[i0:i1]):
            length = (e - s) * step
            if not config.HESITATION_MIN_SEC <= length <= config.HESITATION_MAX_SEC:
                continue
            pitch = f0[i0 + s:i0 + e]
            pitch = pitch[(pitch > 75) & (pitch < 370)]
            if len(pitch) < 0.7 * (e - s):            # no clear pitch = breath, click or noise
                continue
            semitones = 12 * np.log2(pitch / np.median(pitch))
            if np.std(semitones) <= config.HESITATION_PITCH_STEADY_ST:
                found += 1

    for w in words:                                   # sign 2: a short word stretched out
        token = re.sub(r"[^a-z']", "", w["word"].lower())
        if (token and len(token) <= config.STRETCHED_WORD_MAX_CHARS
                and not re.fullmatch(FILLER_SOUND, token)           # real fillers are counted from the text
                and w["end"] - w["start"] >= config.STRETCHED_WORD_SEC):
            i0, i1 = int(w["start"] / step), int(w["end"] / step)
            if i1 > i0 and loud[i0:i1].mean() >= 0.6:               # the voice is on, so it is not a silent pause
                found += 1
    return found


def _mattr(tokens, window=50):
    """Vocabulary variety that doesn't depend on how long the talk is (moving-average TTR)."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [len(set(tokens[i:i + window])) / window
              for i in range(0, len(tokens) - window + 1, 10)]
    return float(np.mean(ratios))


def analyze_speech(result, wav_path=None):
    """
    Take Whisper's result, return (features dict, transcript text).
    Features: words_per_min, fillers_per_min, hesitations_per_min, long_pauses_per_min, pause_ratio,
              lexical_diversity, avg_sentence_len, duration_sec
    Give wav_path too, and the audio is checked for "um/uh" that Whisper deleted.
    fillers_per_min = fillers found in the text + hesitation sounds found in the audio.
    """
    text = result.get("text", "").strip()
    segments = result.get("segments", [])
    words = [w for seg in segments for w in seg.get("words", [])]

    if words:
        start, end = words[0]["start"], words[-1]["end"]
        gaps = [b["start"] - a["end"] for a, b in zip(words, words[1:])]
    elif segments:                                   # no word times: use gaps between segments
        start, end = segments[0]["start"], segments[-1]["end"]
        gaps = [b["start"] - a["end"] for a, b in zip(segments, segments[1:])]
    else:
        raise RuntimeError("No speech was detected in this video.")

    duration = max(end - start, 1.0)
    minutes = duration / 60.0
    lower = text.lower()
    tokens = re.findall(r"[a-z']+", lower)
    if len(tokens) < 10:
        raise RuntimeError("Almost no speech was detected in this video.")

    text_fillers = sum(len(re.findall(p, lower)) for p in FILLER_PATTERNS)
    hesitations = float("nan")                       # nan = not measured from the audio
    if wav_path is not None and words:
        try:
            hesitations = float(detect_hesitations(wav_path, words))
        except Exception:
            pass                                     # audio check failed: keep the text-only count
    fillers = text_fillers + (0 if np.isnan(hesitations) else hesitations)
    long_pauses = sum(1 for g in gaps if g >= config.LONG_PAUSE_SEC)
    pause_time = sum(g for g in gaps if g >= config.PAUSE_SEC)

    nlp = _nlp()
    if nlp is not None:
        doc = nlp(text)
        n_sentences = max(1, len(list(doc.sents)))
        lemmas = [t.lemma_.lower() for t in doc if t.is_alpha]
        diversity = _mattr(lemmas or tokens)
    else:
        n_sentences = max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))
        diversity = _mattr(tokens)

    features = {
        "words_per_min": len(tokens) / minutes,
        "fillers_per_min": fillers / minutes,
        "hesitations_per_min": hesitations / minutes,
        "long_pauses_per_min": long_pauses / minutes,
        "pause_ratio": min(1.0, pause_time / duration),
        "lexical_diversity": diversity,
        "avg_sentence_len": len(tokens) / n_sentences,
        "duration_sec": duration,
    }
    return features, text
