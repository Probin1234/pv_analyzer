"""
Settings for the whole system.
Change numbers here instead of hunting through the other files.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

try:  # optional: read keys from a .env file
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------- folders
VIDEOS_DIR = BASE_DIR / "videos"                  # downloaded videos
RESULTS_DIR = BASE_DIR / "results"                # everything the system saves
TRANSCRIPTS_DIR = RESULTS_DIR / "transcripts"
FEEDBACK_DIR = RESULTS_DIR / "feedback"
TMP_DIR = RESULTS_DIR / "tmp"
MODELS_DIR = BASE_DIR / "models"                  # face-tracking model file lives here
ROSTER_FILE = RESULTS_DIR / "roster.csv"          # who uploaded what (from Drive)
RESULTS_FILE = RESULTS_DIR / "video_results.csv"  # one row per analyzed video
CREDENTIALS_FILE = BASE_DIR / "credentials.json"  # Google OAuth file
TOKEN_FILE = BASE_DIR / "token.json"              # created after first login

# ---------------------------------------------------------------- Google Drive
DEFAULT_DRIVE_FOLDER = os.getenv("DRIVE_FOLDER", "")
# How to work out whose video it is:
#   "auto"     -> video inside a subfolder: subfolder name is the student.
#                 video sitting in the main folder: the start of the file name.
#   "filename" -> always use the file name (Probin_2026-08-12_video1.mp4 -> Probin)
#   "folder"   -> always use the name of the folder the video is in
NAME_SOURCE = os.getenv("NAME_SOURCE", "auto").lower()
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")

# ---------------------------------------------------------------- skills
METRICS = ["Fluency", "Eye Contact", "Voice", "Content"]

# Raw measurements saved for every video (column name -> friendly label)
RAW_LABELS = {
    "words_per_min": "Words per minute",
    "fillers_per_min": "Filler words per minute (estimated)",
    "hesitations_per_min": "Hidden hesitation sounds per minute (found in the audio)",
    "long_pauses_per_min": "Long pauses per minute",
    "pause_ratio": "Share of time spent in pauses",
    "lexical_diversity": "Vocabulary variety (0-1)",
    "avg_sentence_len": "Average sentence length (words)",
    "duration_sec": "Speaking time (seconds)",
    "pitch_std_st": "Pitch variation (semitones)",
    "energy_std_db": "Loudness variation (dB)",
    "eye_contact_pct": "Eye contact (%)",
    "face_detected_pct": "Face found in video (%)",
    "gaze_shifts_per_min": "Looking-away moments per minute",
    "head_movement": "Head movement",
}

# ---------------------------------------------------------------- speech (Whisper)
# tiny = fastest, base = good balance, small = more accurate but slow on a laptop
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
# Whisper tends to delete "um" and "uh". A prompt full of fillers makes it keep them.
FILLER_PROMPT = "Umm, let me think, like, hmm... Okay, so, uh, here is what I learned."
LONG_PAUSE_SEC = 1.0        # a gap this long between words counts as a long pause
PAUSE_SEC = 0.5             # gaps this long count towards "time spent in pauses"

# Whisper deletes most "um/uh", so we also LISTEN for them in the audio:
# a steady, flat-pitched sound in a gap where Whisper heard no word = a hidden hesitation sound.
HESITATION_MIN_SEC = 0.25        # the sound must last at least this long
HESITATION_MAX_SEC = 2.0         # longer than this is probably not an "uhh"
HESITATION_PITCH_STEADY_ST = 1.5 # the pitch must stay this steady (semitones): "uhh" is flat, speech moves
STRETCHED_WORD_SEC = 1.0         # a short word stretched this long ("sooo", "and...") counts too
STRETCHED_WORD_MAX_CHARS = 5

# ---------------------------------------------------------------- eyes (MediaPipe)
# New MediaPipe versions need this small model file (~4 MB). It downloads by itself the
# first time. If your internet blocks it, download it by hand and put it in the models/ folder.
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
EYE_SAMPLE_FPS = 5          # look at 5 frames per second (faster than every frame)
GAZE_H_TOL = 0.15           # how far the iris may sit from the eye centre (sideways)
GAZE_V_TOL = 0.18           # same, up/down (compared with the person's usual)
YAW_TOL = 0.12              # how far the head may turn left/right
PITCH_TOL = 0.10            # how far the head may tilt up/down (vs usual)

# ---------------------------------------------------------------- LLM (free)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()      # "groq", "gemini" or "ollama"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")   # Groq retires models now and then; change it in .env if you get a model_not_found error
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")   # Ollama runs on your own computer
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")   # small (3B) and runs on most laptops; try "llama3.1:8b" if you have 16 GB RAM
OLLAMA_TIMEOUT_SEC = 300   # a local model on a laptop CPU can take minutes, especially the first call
OLLAMA_CONTEXT = 8192      # how much text (tokens) the model can read at once; transcripts need more than Ollama's default
LLM_TEMPERATURE = 0.2       # low = more consistent scores for the same video
LLM_TIMEOUT_SEC = 60

# ---------------------------------------------------------------- video type (classify.py)
# The kinds of video the system can recognize. Add, remove or reword them here.
VIDEO_TYPES = {
    "Presentation": "a structured talk that presents or explains something to an audience (opening, main points, closing)",
    "Motivational talk": "inspires or encourages the listener: success, mindset, confidence, never giving up",
    "Podcast / conversation": "a podcast, interview or chat: a host or guest, listeners, an episode, talking with someone",
    "Lecture / tutorial": "teaches a concept or shows how to do something, often step by step with examples",
    "Storytelling": "tells a personal story or experience in time order",
    "Review / opinion": "gives an opinion, review or comparison of a product, film, book or idea",
    "Speech / announcement": "a formal speech, welcome, introduction or announcement",
    "Other": "does not fit any of the types above",
}
