"""
Scoring part: turn the raw measurements into 0-100 skill scores, and work out
how much a student has improved over their videos.

All the "good" and "bad" levels below are reasonable starting values (teaching rules of
thumb), not scientific truth. Tune them to fit your class's videos.
"""
import numpy as np
import pandas as pd

from .config import METRICS


def _scale(x, bad, good):
    """0 when x is at `bad`, 100 when x is at `good`, straight line between. Works either way round."""
    if x is None or pd.isna(x):
        return np.nan
    return float(np.clip((x - bad) / (good - bad) * 100.0, 0, 100))


def _weighted(parts):
    """Weighted average of (weight, score) pairs, ignoring scores that are missing."""
    parts = [(w, s) for w, s in parts if not pd.isna(s)]
    if not parts:
        return np.nan
    total = sum(w for w, _ in parts)
    return float(sum(w * s for w, s in parts) / total)


def fluency_score(f):
    wpm = f.get("words_per_min", np.nan)
    if pd.isna(wpm):
        pace = np.nan
    else:
        # 120-160 words/min is a comfortable speaking pace. Further away = lower score.
        dist = 0 if 120 <= wpm <= 160 else (120 - wpm if wpm < 120 else wpm - 160)
        pace = _scale(dist, 60, 0)
    fillers = _scale(f.get("fillers_per_min", np.nan), 8, 0)       # 0 fillers = 100, 8+/min = 0
    pauses = _scale(f.get("long_pauses_per_min", np.nan), 8, 2)    # <=2 long pauses/min = 100
    return _weighted([(0.35, pace), (0.40, fillers), (0.25, pauses)])


def eye_contact_score(f):
    if f.get("face_detected_pct", 0) < 30:          # face barely visible: can't judge
        return np.nan
    contact = _scale(f.get("eye_contact_pct", np.nan), 20, 75)     # 75%+ looking at camera = 100
    steady = _scale(f.get("head_movement", np.nan), 12, 3)         # steadier head = higher
    return _weighted([(0.85, contact), (0.15, steady)])


def voice_score(f):
    pitch = _scale(f.get("pitch_std_st", np.nan), 0.8, 3.5)        # monotone -> lively
    energy = _scale(f.get("energy_std_db", np.nan), 2, 8)
    return _weighted([(0.6, pitch), (0.4, energy)])


def score_video(features, content_score=np.nan):
    """Return {Fluency, Eye Contact, Voice, Content, Overall} for one video."""
    scores = {
        "Fluency": fluency_score(features),
        "Eye Contact": eye_contact_score(features),
        "Voice": voice_score(features),
        "Content": content_score if content_score is not None else np.nan,
    }
    valid = [v for v in scores.values() if not pd.isna(v)]
    scores["Overall"] = float(np.mean(valid)) if valid else np.nan
    return scores


def improvement_table(df, n=3):
    """
    Compare the student's first videos with their latest ones.
    df needs a 'date' column and the skill columns. Returns one row per skill:
      Start  = average of the first `n` videos      (n = 3, fewer if there aren't many)
      Now    = average of the last `n` videos
      Change = Now - Start        (empty when there's only 1 video)
      Trend (pts/week) = slope of a straight line through all the videos
    """
    cols = METRICS + ["Overall"]
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    k = max(1, min(n, len(df) // 2))
    weeks = ((df["date"] - df["date"].min()).dt.days / 7.0).to_numpy()

    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        start, now = s.head(k).mean(), s.tail(k).mean()
        change = now - start if len(df) >= 2 else np.nan
        ok = s.notna().to_numpy()
        trend = np.nan
        if ok.sum() >= 3 and np.ptp(weeks[ok]) > 0:
            trend = float(np.polyfit(weeks[ok], s.to_numpy()[ok], 1)[0])
        rows.append({"Skill": c, "Start": start, "Now": now,
                     "Change": change, "Trend (pts/week)": trend})
    return pd.DataFrame(rows).round(1)
