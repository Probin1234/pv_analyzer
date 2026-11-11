# 🎤 Presentation Skills Analyzer

Analyzes students' weekly presentation videos (face + voice, no slides) from a Google Drive folder
and shows how much each student has improved from their first video to their latest.

Built for **MLA308 – NLP, Generative AI and GUI**.

```
FRONTEND (app.py, Streamlit)  ──►  backend/service.py  ──►  drive · speech · voice · eyes · scoring · feedback
   only screens and buttons          the middle layer          all the machine learning lives here
```

---

## 1. Folder layout

```
pv_analyzer/
├── app.py                 ← the whole frontend
├── analyze_all.py         ← analyze from the command line (run before the demo)
├── requirements.txt
├── .env.example           ← copy to ".env" and fill in
├── credentials.json       ← you create this (step 4)
└── backend/
    ├── config.py          ← all settings and thresholds
    ├── service.py         ← the ONLY file the screen talks to; runs the pipeline
    ├── drive.py           ← Google Drive: scan, find whose video, download
    ├── media.py           ← pull audio out of a video (ffmpeg)
    ├── speech.py          ← Whisper + SpaCy: transcript, speed, fillers, pauses, vocabulary
    ├── voice.py           ← Librosa: pitch and loudness variation
    ├── eyes.py            ← MediaPipe: eye contact, head movement
    ├── scoring.py         ← 0-100 skill scores + improvement calculation
    └── feedback.py        ← free LLM (Groq/Gemini): content score + written feedback
```

Folders created automatically while running: `videos/` (downloads), `results/` (saved analysis), `models/`.

---

## 2. Setup (do once)

**Python:** use 3.10, 3.11 or 3.12.

**ffmpeg** (needed to read audio from videos). Windows, in PowerShell:
```powershell
winget install Gyan.FFmpeg
```
Close and reopen VS Code, then check with `ffmpeg -version`.

**Project environment:**
```powershell
cd "PV analyzer"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
(The install is big because Whisper brings PyTorch. Use a good internet connection.)

---

## 3. Free LLM key (pick one)

- **Groq** (recommended, fast): create a free key at console.groq.com → API Keys
- **Gemini**: create a free key at aistudio.google.com → Get API key

Copy `.env.example` to `.env` and fill it in:
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```
Without a key the system still works: the Content score stays empty and feedback is written by simple rules.

---

## 4. Google Drive login (read-only)

1. Go to console.cloud.google.com and create a project.
2. Search **Google Drive API** → Enable.
3. **APIs & Services → OAuth consent screen** → External → add your own Gmail as a *test user*.
4. **Credentials → Create credentials → OAuth client ID → Desktop app** → download the JSON.
5. Rename it `credentials.json` and put it next to `app.py`.

The first scan opens a browser to log in with the Gmail that has access to sir's folder.
After that a `token.json` is saved and no browser opens again. **Never share `credentials.json` or `token.json`.**

---

## 5. Run

```powershell
streamlit run app.py
```
1. Paste the Drive folder link in the sidebar → **Connect & scan**.
2. **Class overview** shows every student who uploaded.
3. **Student report** → pick a student → **Analyze videos**.

**Before the demo**, analyze everyone once so the app opens instantly:
```powershell
python analyze_all.py --scan "<drive folder link>"
```
Results are saved in `results/`. Already-analyzed videos are skipped next time.

**Speed:** the first analysis of 20+ videos takes a while (Whisper is the slow part). On a laptop
without a GPU set `WHISPER_MODEL=tiny` in `.env` for speed, or `base` for better accuracy.

---

## 6. How a video becomes scores

| Skill | Measured from | Counts |
|---|---|---|
| **Fluency** | Whisper transcript + word times | speaking pace (120-160 words/min is ideal), filler words (um, uh, you know...), long pauses |
| **Eye Contact** | MediaPipe face + iris tracking | % of time looking at the camera, head steadiness |
| **Voice** | Librosa | pitch variation and loudness variation (monotone vs lively) |
| **Content** | LLM reads the transcript | relevance, structure, clarity, depth (4 × 25 points) |
| **Overall** | average of the above | skills that couldn't be measured are skipped, never counted as zero |

**Improvement** = average of the first 3 videos vs the average of the last 3 videos. The report also
shows a **trend** (points gained per week across all videos).

### Tuning
The "good/bad" levels are sensible starting values, not scientific truth. Change them in
`backend/scoring.py` (score ranges) and `backend/config.py` (eye-contact tolerances) after looking at
real results from your class.

---

## 7. Honest limits (good to mention to your teacher)

- **Eye contact is an estimate.** A normal webcam can't measure gaze precisely. The system checks the
  iris position, head turn and head tilt. It works best with a front-facing, well-lit face.
- **Filler words may be undercounted.** Speech-to-text tends to drop "um/uh". A prompt in `config.py`
  encourages Whisper to keep them, but it isn't perfect.
- **LLM scores can vary a little.** The temperature is set low (0.2) for consistency, and the content
  score is split into four sub-scores, but it is still a judgement.
- **Privacy:** transcripts are sent to the LLM provider. On Gemini's free tier, content may be used to
  improve Google's products. Tell students, or use Groq.
- **Name detection:** if student names come out wrong, change `NAME_SOURCE` in `.env`
  (`auto`, `filename` or `folder`) or the rule in `drive.py → student_from_filename`.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` | Install ffmpeg (step 2), restart VS Code |
| `pip` can't reach pypi.org | Internet/DNS problem: try another network or a phone hotspot |
| Sidebar **System check** shows ❌ | It tells you exactly what to install |
| Face-tracking model download fails | Download `face_landmarker.task` from the URL in `config.py`, save as `models/face_landmarker.task` |
| `numpy` version errors | `pip install "numpy<2"` |
| Student names look wrong | See "Name detection" above |
| Eye Contact shows `—` | Face was found in under 30% of the video (bad light, face too small, camera too low) |
| LLM errors (`HTTP 429`) | Free-tier limit: wait a minute and press Analyze again; finished videos are kept |
| Wrong LLM model name | Set `GROQ_MODEL` or `GEMINI_MODEL` in `.env` |

---

**Security note:** Never commit `.env`, `credentials.json`, or `token.json`. They are listed in `.gitignore`.
