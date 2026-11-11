"""
Eyes part: use MediaPipe Face Mesh to track the face, the iris (coloured part of the eye)
and the head, then work out how often the person is looking at the camera.

A frame counts as "eye contact" when ALL of these are true:
  1. the iris sits near the middle of the eye (not looking sideways)
  2. the head is not turned left/right
  3. the head is not tilted up/down compared with how they usually hold it
  4. the eyes are not looking up/down compared with how they usually look
This is an approximation (a normal webcam can't measure gaze exactly). You can tune
the tolerances in config.py.
"""
import numpy as np

from . import config

# MediaPipe Face Mesh landmark numbers
R_OUTER, R_INNER, R_IRIS, R_TOP, R_BOTTOM = 33, 133, 468, 159, 145
L_INNER, L_OUTER, L_IRIS, L_TOP, L_BOTTOM = 362, 263, 473, 386, 374
NOSE, CHIN, FACE_LEFT, FACE_RIGHT = 1, 152, 234, 454


def _measure(lm):
    """
    From one frame's landmarks return (h, v, yaw, pitch):
      h     iris position across the eye   (0.5 = centred)
      v     iris position up/down in eye   (compared later with the person's usual)
      yaw   nose position between the cheeks (0.5 = facing the camera)
      pitch nose height between eyes and chin
    Returns None when the eyes are closed (a blink).
    """
    def pt(i):
        return np.array([lm[i].x, lm[i].y])

    h_vals, v_vals, openness = [], [], []
    for a, b, iris, top, bottom in ((R_OUTER, R_INNER, R_IRIS, R_TOP, R_BOTTOM),
                                    (L_INNER, L_OUTER, L_IRIS, L_TOP, L_BOTTOM)):
        p1, p2, c = pt(a), pt(b), pt(iris)
        along = p2 - p1
        width = float(np.dot(along, along))
        if width < 1e-9:
            return None
        h_vals.append(float(np.dot(c - p1, along) / width))

        t, btm = pt(top), pt(bottom)
        opening = btm[1] - t[1]
        openness.append(opening / (np.sqrt(width) + 1e-9))
        if opening > 1e-6:
            v_vals.append(float((c[1] - t[1]) / opening))

    if np.mean(openness) < 0.10 or not v_vals:       # eyes (almost) closed
        return None

    nose, left, right, chin = pt(NOSE), pt(FACE_LEFT), pt(FACE_RIGHT), pt(CHIN)
    eye_mid_y = (pt(R_OUTER)[1] + pt(L_OUTER)[1]) / 2
    face_width = right[0] - left[0]
    face_height = chin[1] - eye_mid_y
    if abs(face_width) < 1e-6 or abs(face_height) < 1e-6:
        return None
    yaw = float((nose[0] - left[0]) / face_width)
    pitch = float((nose[1] - eye_mid_y) / face_height)
    return float(np.mean(h_vals)), float(np.mean(v_vals)), yaw, pitch


def _ensure_model():
    """Make sure the face-tracking model file exists (download it the first time)."""
    path = config.MODELS_DIR / "face_landmarker.task"
    if path.exists() and path.stat().st_size > 0:
        return path
    import requests
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(config.FACE_MODEL_URL, timeout=60)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Could not download the face-tracking model ({e}). Download it by hand from "
            f"{config.FACE_MODEL_URL} and save it as models/face_landmarker.task"
        )
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(path)
    return path


def _make_detector():
    """
    Returns (detect, close). detect(rgb_frame, time_ms) -> list of 478 landmarks, or None if no face.
    Works with both the new MediaPipe (Tasks API) and older versions (Solutions API).
    """
    import mediapipe as mp

    if hasattr(mp, "solutions"):                                   # older MediaPipe
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def detect(rgb, time_ms):
            res = mesh.process(rgb)
            return res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
        return detect, mesh.close

    from mediapipe.tasks import python as mp_python               # newer MediaPipe
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(rgb, time_ms):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = landmarker.detect_for_video(image, int(time_ms))
        return res.face_landmarks[0] if res.face_landmarks else None
    return detect, landmarker.close


def analyze_eyes(video_path):
    """
    Returns: eye_contact_pct       % of the time looking at the camera
             face_detected_pct     % of sampled frames where a face was found
             gaze_shifts_per_min   how often they look away from the camera
             head_movement         how much the head moves around (lower = steadier)
    """
    import cv2

    detect, close = _make_detector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        close()
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / config.EYE_SAMPLE_FPS)))

    samples = []                  # (time_sec, h, v, yaw, pitch) for frames we could measure
    sampled = faces = blinks = 0
    idx = 0
    try:
        while cap.grab():                              # grab = cheap, skips the decoding work
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    sampled += 1
                    height, width = frame.shape[:2]
                    if width > 640:
                        frame = cv2.resize(frame, (640, int(height * 640 / width)))
                    landmarks = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), idx / fps * 1000)
                    if landmarks is not None:
                        faces += 1
                        m = _measure(landmarks)
                        if m is None:
                            blinks += 1
                        else:
                            samples.append((idx / fps, *m))
            idx += 1
    finally:
        cap.release()
        close()

    if sampled == 0:
        raise RuntimeError("The video has no readable frames.")

    face_pct = 100.0 * faces / sampled
    if len(samples) < 20:
        return {"eye_contact_pct": float("nan"), "face_detected_pct": face_pct,
                "gaze_shifts_per_min": float("nan"), "head_movement": float("nan")}

    arr = np.array(samples)
    t, h, v, yaw, pitch = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    looking = (
        (np.abs(h - 0.5) < config.GAZE_H_TOL)
        & (np.abs(yaw - 0.5) < config.YAW_TOL)
        & (np.abs(pitch - np.median(pitch)) < config.PITCH_TOL)
        & (np.abs(v - np.median(v)) < config.GAZE_V_TOL)
    )
    # smooth out single-frame flickers (majority vote over 5 samples)
    looking = np.convolve(looking.astype(float), np.ones(5) / 5, mode="same") >= 0.5

    usable = max(1, sampled - blinks)              # blinks aren't "looking away"
    eye_contact = 100.0 * looking.sum() / usable
    minutes = max((t[-1] - t[0]) / 60.0, 1 / 60.0)
    shifts = int(np.sum(looking[:-1] & ~looking[1:]))          # contact -> away
    head_move = float((np.std(yaw) + np.std(pitch)) * 100)

    return {
        "eye_contact_pct": float(min(100.0, eye_contact)),
        "face_detected_pct": face_pct,
        "gaze_shifts_per_min": shifts / minutes,
        "head_movement": head_move,
    }
