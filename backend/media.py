"""Small helper: pull the audio track out of a video using ffmpeg."""
import shutil
import subprocess
from pathlib import Path


def extract_audio(video_path, wav_path):
    """Save the audio of a video as a 16 kHz mono .wav file (what Whisper and Librosa like)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed (or not on PATH). See README step 2.")
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(wav_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not wav_path.exists():
        raise RuntimeError(
            f"Could not read audio from {Path(video_path).name}. "
            f"Does the video have sound? ({result.stderr.strip()[-200:]})"
        )
    return wav_path
