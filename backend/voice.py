"""
Voice part: measure HOW the voice sounds (not what is said) using Librosa.
A monotone voice has little pitch and loudness variation; a lively one has more.
"""
import numpy as np


def analyze_voice(wav_path):
    """
    Returns: pitch_std_st   (how much the pitch moves, in semitones; ~3 is lively)
             energy_std_db  (how much the loudness moves, in dB)
    """
    import librosa

    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    if len(y) < sr * 3:
        raise RuntimeError("Audio is too short to analyze the voice.")

    hop, frame = 512, 2048
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    f0 = librosa.yin(y, fmin=70, fmax=380, sr=sr, frame_length=frame, hop_length=hop)

    n = min(len(rms), len(f0))
    rms, f0 = rms[:n], f0[:n]

    # Only look at frames where the person is actually speaking (not silence)
    speaking = rms > 0.15 * np.percentile(rms, 95)
    if speaking.sum() < 50:
        raise RuntimeError("Could not find enough speech to analyze the voice.")

    # Loudness variation
    db = 20 * np.log10(rms[speaking] + 1e-8)
    energy_std = float(np.std(db))

    # Pitch variation, measured in semitones around the person's own average pitch
    voiced_f0 = f0[speaking]
    voiced_f0 = voiced_f0[(voiced_f0 > 75) & (voiced_f0 < 370)]   # drop tracker errors
    if len(voiced_f0) < 50:
        pitch_std = float("nan")
    else:
        semitones = 12 * np.log2(voiced_f0 / np.median(voiced_f0))
        lo, hi = np.percentile(semitones, [5, 95])                # ignore wild outliers
        pitch_std = float(np.std(semitones[(semitones >= lo) & (semitones <= hi)]))

    return {"pitch_std_st": pitch_std, "energy_std_db": energy_std}
