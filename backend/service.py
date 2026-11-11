"""
service.py - the ONLY file the screen (app.py) talks to.

It stores two small tables on disk (so nothing is lost when you close the app):
  results/roster.csv        who uploaded which video          (made by scan_drive)
  results/video_results.csv one row per analyzed video         (made by run_analysis)
"""
import hashlib
import importlib.util
import re
import shutil
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import classify, config, drive, eyes, feedback, media, scoring, speech, voice
from .config import METRICS

STUDENT_COLUMNS = ["Student", "Videos", "Analyzed", "Latest upload", "Overall score", "Improvement"]
ROSTER_COLUMNS = ["id", "file", "student", "date"]
MANUAL_COLUMNS = ROSTER_COLUMNS + ["path"]          # manual uploads also remember where the file is saved
MANUAL_ROSTER_FILE = config.RESULTS_DIR / "manual_roster.csv"   # videos added with "Upload videos"
MANUAL_VIDEOS_DIR = config.VIDEOS_DIR / "manual"
UPLOAD_TYPES = [e.lstrip(".") for e in config.VIDEO_EXTENSIONS]
RAW_COLUMNS = list(config.RAW_LABELS)
CLASSIFY_COLUMNS = ["video_type", "topic", "video_summary", "type_confidence"]
RESULT_COLUMNS = (["id", "student", "video", "date"] + RAW_COLUMNS
                  + METRICS + ["Overall", "content_comment"] + CLASSIFY_COLUMNS + ["analyzed_at"])


# ================================================================== small helpers
def _ensure_dirs():
    for d in (config.RESULTS_DIR, config.TRANSCRIPTS_DIR, config.FEEDBACK_DIR, config.TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _safe(text):
    return re.sub(r"\W+", "_", text).strip("_") or "student"


def _manual_roster():
    if not MANUAL_ROSTER_FILE.exists():
        return pd.DataFrame(columns=MANUAL_COLUMNS)
    return pd.read_csv(MANUAL_ROSTER_FILE, dtype=str).fillna("")


def _roster():
    """Every known video: the ones found on Drive plus the ones uploaded by hand."""
    if config.ROSTER_FILE.exists():
        from_drive = pd.read_csv(config.ROSTER_FILE, dtype=str).fillna("")
    else:
        from_drive = pd.DataFrame(columns=ROSTER_COLUMNS)
    from_drive["path"] = ""                          # Drive videos have no local path yet
    frames = [f for f in (from_drive, _manual_roster()) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=MANUAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)[MANUAL_COLUMNS]


def _results():
    if not config.RESULTS_FILE.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.read_csv(config.RESULTS_FILE, dtype={"id": str})
    for col in RESULT_COLUMNS:
        if col not in df:
            df[col] = np.nan
    return df


def _save_result(row, results):
    """Add (or replace) one video's row and write the file straight away."""
    results = results[results["id"] != row["id"]]
    new = pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS)
    results = new if results.empty else pd.concat([results, new], ignore_index=True)
    results.to_csv(config.RESULTS_FILE, index=False)
    return results


def _feedback_path(student):
    return config.FEEDBACK_DIR / f"{_safe(student)}.txt"


# ================================================================== Drive
def scan_drive(folder_link):
    """Look inside the Drive folder, remember every video and whose it is. Returns how many."""
    _ensure_dirs()
    client = drive.get_drive()
    videos = drive.scan_folder(client, drive.extract_folder_id(folder_link))
    pd.DataFrame(videos, columns=ROSTER_COLUMNS).to_csv(config.ROSTER_FILE, index=False)
    return len(videos)


# ================================================================== manual upload
def guess_date(file_name):
    """A date written in the file name (2026-08-12) if there is one, else today."""
    m = re.search(r"(\d{4})[-_.](\d{2})[-_.](\d{2})", file_name)
    if m:
        try:
            return str(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    return str(date.today())


def add_manual_videos(student, uploads):
    """
    Save videos picked on the Upload page and remember whose they are.
      uploads = [(file_name, file_or_bytes, "YYYY-MM-DD"), ...]
    Returns (student_name_as_stored, [video ids]). Uploading the same file again just replaces it.
    The name is tidied like Drive names ("probin" -> "Probin"), so it joins that student's report.
    """
    name = drive._clean_name(student or "")
    if name == "Unknown":
        raise ValueError("Please type the student's name.")
    if not uploads:
        raise ValueError("Please choose at least one video.")
    _ensure_dirs()
    folder = MANUAL_VIDEOS_DIR / _safe(name)
    folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for file_name, data, when in uploads:
        if isinstance(data, (bytes, bytearray, memoryview)):
            blob = data
        elif hasattr(data, "getbuffer"):
            blob = data.getbuffer()
        else:
            blob = data.read()
        vid_id = "local-" + hashlib.sha1(f"{name}|{file_name}|{len(blob)}".encode()).hexdigest()[:12]
        safe_name = re.sub(r"[^\w.\- ]", "_", file_name)
        path = folder / f"{vid_id[6:14]}_{safe_name}"
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(blob)
        partial.replace(path)                         # only now is it a finished file
        rows.append({"id": vid_id, "file": file_name, "student": name,
                     "date": str(pd.to_datetime(when).date()),
                     "path": path.relative_to(config.BASE_DIR).as_posix()})

    manual = _manual_roster()
    manual = manual[~manual["id"].isin([r["id"] for r in rows])]
    manual = pd.concat([manual, pd.DataFrame(rows, columns=MANUAL_COLUMNS)], ignore_index=True)
    manual.to_csv(MANUAL_ROSTER_FILE, index=False)
    return name, [r["id"] for r in rows]


def local_video_ids(student):
    """Ids of this student's manually uploaded videos."""
    mine = _manual_roster()
    return mine[mine["student"] == student]["id"].tolist()


# ================================================================== class overview
def list_students():
    """One row per student: video count, how many analyzed, latest score, improvement."""
    roster = _roster()
    if roster.empty:
        return pd.DataFrame(columns=STUDENT_COLUMNS)
    results = _results()
    rows = []
    for student, grp in roster.groupby("student"):
        res = results[results["student"] == student]
        now = change = np.nan
        if len(res):
            overall = scoring.improvement_table(res).query("Skill == 'Overall'").iloc[0]
            now, change = overall["Now"], overall["Change"]
        rows.append({"Student": student, "Videos": len(grp), "Analyzed": len(res),
                     "Latest upload": grp["date"].max(),
                     "Overall score": now, "Improvement": change})
    return pd.DataFrame(rows, columns=STUDENT_COLUMNS).sort_values("Student").reset_index(drop=True)


# ================================================================== student report
def get_report(student):
    """None if nothing analyzed yet, else {"videos", "table", "feedback"}."""
    res = _results()
    res = res[res["student"] == student]
    if res.empty:
        return None
    res = res.assign(date=pd.to_datetime(res["date"])).sort_values("date").reset_index(drop=True)
    table = scoring.improvement_table(res)

    path = _feedback_path(student)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = (feedback.rule_based_feedback(student, table)
                + "\n\n_(Written by simple rules. Add an LLM key in .env for AI-written feedback.)_")
    return {"videos": res, "table": table, "feedback": text}


def get_transcript(video_id):
    path = config.TRANSCRIPTS_DIR / f"{video_id}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def classify_saved_videos(student=None, force=False, on_progress=None):
    """
    Recognize the type and topic of videos that were analyzed BEFORE this feature existed.
    Uses the transcripts already saved on disk, so Whisper does not run again (fast).
    student=None means every student. Returns how many videos were updated.
    """
    on_progress = on_progress or (lambda p, t: None)
    results = _results()
    for col in CLASSIFY_COLUMNS:                      # empty columns load as numbers; we need text
        results[col] = results[col].astype(object)
    rows = results.index[results["student"] == student] if student else results.index
    todo = [i for i in rows
            if force or not (isinstance(results.at[i, "video_type"], str)
                             and results.at[i, "video_type"].strip())]
    updated = 0
    for n, i in enumerate(todo):
        on_progress(n / max(1, len(todo)), f"Video {n + 1} of {len(todo)}")
        transcript = get_transcript(str(results.at[i, "id"]))
        if not transcript.strip():
            continue
        info = classify.classify_video(transcript)
        for col in CLASSIFY_COLUMNS:
            results.at[i, col] = info[col]
        updated += 1
    if updated:
        results.to_csv(config.RESULTS_FILE, index=False)
    return updated


# ================================================================== the full pipeline
def _analyze_one(client, row, progress, warnings):
    """Download one video, run every analysis on it, return the row to save."""
    local = str(row.get("path", "") or "")
    if local:                                         # uploaded by hand: already on this computer
        progress(0.00, "reading the uploaded video")
        path = config.BASE_DIR / local
        if not path.exists():
            raise FileNotFoundError(f"the saved video is missing: {local}")
    else:
        progress(0.00, "downloading from Drive")
        path = drive.download_video(client, row["id"], row["file"], row["student"])

    wav = config.TMP_DIR / f"{row['id']}.wav"
    try:
        progress(0.10, "extracting audio")
        media.extract_audio(path, wav)

        progress(0.15, "transcribing speech (slowest step)")
        result = speech.transcribe(wav)
        speech_features, transcript = speech.analyze_speech(result, wav)   # wav: also listen for hidden "um/uh"
        if pd.isna(speech_features.get("hesitations_per_min")):
            warnings.append(f"{row['file']}: hidden hesitation sounds were not measured from the audio")

        progress(0.60, "measuring voice")
        try:
            voice_features = voice.analyze_voice(wav)
        except Exception as e:                       # keep going, just note it
            voice_features = {}
            warnings.append(f"{row['file']}: voice not measured ({e})")
    finally:
        wav.unlink(missing_ok=True)

    progress(0.70, "tracking eye contact")
    try:
        eye_features = eyes.analyze_eyes(path)
    except Exception as e:
        eye_features = {}
        warnings.append(f"{row['file']}: eye contact not measured ({e})")

    progress(0.88, "judging the content with the LLM")
    content, comment = np.nan, ""
    try:
        content, comment = feedback.evaluate_content(transcript)
    except Exception as e:
        comment = ""
        warnings.append(f"{row['file']}: content not scored ({e})")

    progress(0.94, "recognizing what kind of video it is")
    info = classify.classify_video(transcript)
    if info.get("note"):
        warnings.append(f"{row['file']}: {info['note']}")
    video_info = {k: info[k] for k in CLASSIFY_COLUMNS}

    features = {**speech_features, **voice_features, **eye_features}
    scores = scoring.score_video(features, content)
    (config.TRANSCRIPTS_DIR / f"{row['id']}.txt").write_text(transcript, encoding="utf-8")

    return {"id": row["id"], "student": row["student"], "video": row["file"],
            "date": row["date"], **features, **scores, **video_info,
            "content_comment": comment, "analyzed_at": datetime.now().isoformat(timespec="seconds")}


def run_analysis(student, on_progress=None, force=False, only_ids=None):
    """
    Analyze every NEW video of one student (videos done before are skipped, unless force=True),
    then write fresh feedback. only_ids=[...] limits it to those videos (e.g. just the new uploads). Calls on_progress(0.0-1.0, "message") along the way.
    Returns {"analyzed": n, "skipped": n, "failed": [(file, why)], "warnings": [...], "feedback": "..."}.
    """
    on_progress = on_progress or (lambda p, t: None)
    _ensure_dirs()

    roster = _roster()
    mine = roster[roster["student"] == student].sort_values("date")
    if mine.empty:
        raise ValueError(f"No videos found for {student}. Scan the Drive folder or upload videos first.")
    if only_ids is not None:
        mine = mine[mine["id"].isin(list(only_ids))]
        if mine.empty:
            raise ValueError(f"No matching videos found for {student}.")

    results = _results()
    done = set() if force else set(results["id"].astype(str))
    todo = mine[~mine["id"].isin(done)]
    summary = {"analyzed": 0, "skipped": len(mine) - len(todo),
               "failed": [], "warnings": [], "feedback": "unchanged"}

    if not todo.empty:
        client = None
        if (todo["path"] == "").any():                # log in to Drive only if a video needs it
            on_progress(0.01, "Logging in to Google Drive...")
            client = drive.get_drive()
        n = len(todo)
        for i, (_, row) in enumerate(todo.iterrows()):
            def step(frac, msg, i=i):
                on_progress(0.02 + 0.88 * (i + frac) / n, f"Video {i + 1} of {n}: {msg}...")
            try:
                new_row = _analyze_one(client, row, step, summary["warnings"])
            except Exception as e:                    # one bad video must not stop the rest
                summary["failed"].append((row["file"], str(e)))
                continue
            results = _save_result(new_row, results)
            summary["analyzed"] += 1

    # ---- feedback (rewrite when something changed, or when it's missing)
    mine_results = results[results["student"] == student]
    path = _feedback_path(student)
    if not mine_results.empty and (summary["analyzed"] > 0 or force or not path.exists()):
        on_progress(0.93, "Writing feedback with the LLM...")
        table = scoring.improvement_table(mine_results)
        try:
            path.write_text(feedback.write_feedback(student, table, mine_results), encoding="utf-8")
            summary["feedback"] = "written by the LLM"
        except Exception as e:
            path.unlink(missing_ok=True)              # don't leave out-of-date feedback behind
            summary["feedback"] = f"LLM feedback skipped ({e}); showing rule-based feedback"

    on_progress(1.0, "Done")
    return summary


# ================================================================== setup check (for the sidebar)
def system_check():
    """List of (name, ok, hint) so you can see at a glance what is missing."""
    def has(module):
        return importlib.util.find_spec(module) is not None

    def face_model_ready():
        if not has("mediapipe"):
            return False
        import mediapipe as mp
        return hasattr(mp, "solutions") or (config.MODELS_DIR / "face_landmarker.task").exists()

    def ollama_ready():
        """Is Ollama running, and is the chosen model downloaded?"""
        try:
            import requests
            tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).json()
            names = [m.get("name", "") for m in tags.get("models", [])]
        except Exception:
            return False
        want = config.OLLAMA_MODEL
        return any(n == want or n.startswith(want + ":") for n in names)

    if config.LLM_PROVIDER == "ollama":
        llm_check = (f"LLM (Ollama, model {config.OLLAMA_MODEL})", ollama_ready(),
                     f"start Ollama and run: ollama pull {config.OLLAMA_MODEL}")
    else:
        llm_key = config.GROQ_API_KEY if config.LLM_PROVIDER == "groq" else config.GEMINI_API_KEY
        llm_check = (f"LLM key ({config.LLM_PROVIDER})", bool(llm_key), "add it to the .env file")
    return [
        ("ffmpeg", shutil.which("ffmpeg") is not None, "install ffmpeg (README step 2)"),
        ("Whisper", has("whisper"), "pip install openai-whisper"),
        ("Librosa", has("librosa"), "pip install librosa"),
        ("MediaPipe", has("mediapipe"), "pip install mediapipe"),
        ("Face-tracking model file", face_model_ready(),
         "downloads by itself the first time you analyze (needs internet)"),
        ("SpaCy (optional)", has("spacy"), "pip install spacy, then download en_core_web_sm"),
        ("Google Drive login file (only for Drive links)", config.CREDENTIALS_FILE.exists(),
         "see README step 4, or use the Upload videos page instead"),
        llm_check,
    ]
