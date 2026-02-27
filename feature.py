# feature_utils.py
import os
import shutil
import subprocess
import numpy as np
import librosa
import cv2

# Config (tweak if needed)
MAX_LEN = 200
SAMPLE_RATE = 16000
N_MFCC = 40
VIDEO_FEATURE_DIM = 2054  # keep this consistent with your model / checkpoint

# Try lazy import of mediapipe
HAS_MEDIAPIPE = True
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except Exception as e:
    HAS_MEDIAPIPE = False
    mp = None
    mp_face_mesh = None
    print("[WARN] mediapipe not available:", e)


# ---------------------------
# Helpers: ffmpeg audio extraction
# ---------------------------
def _ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or run: conda install -c conda-forge ffmpeg"
        )

def _extract_wav_with_ffmpeg(video_path, out_wav_path, sample_rate=SAMPLE_RATE):
    _ensure_ffmpeg()
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        out_wav_path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg failed to extract audio:\n{stderr}")


# ---------------------------
# Audio extraction + MFCC
# ---------------------------
def extract_audio_from_video(video_path, cleanup=True, sample_rate=SAMPLE_RATE):
    """
    Extract waveform (mono) from a video using ffmpeg and load with librosa.
    Returns waveform numpy array sampled at sample_rate.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    base, _ = os.path.splitext(video_path)
    audio_path = base + ".wav"
    _extract_wav_with_ffmpeg(video_path, audio_path, sample_rate=sample_rate)

    try:
        y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    finally:
        if cleanup and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass
    return y


def extract_audio_features(y, max_len=MAX_LEN, n_mfcc=N_MFCC, sample_rate=SAMPLE_RATE):
    """
    Compute MFCCs and pad/truncate to (max_len, n_mfcc).
    Returns float32 numpy array.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc).T  # [T, n_mfcc]
    if mfcc.ndim == 1:
        mfcc = mfcc.reshape(-1, n_mfcc)
    if mfcc.shape[0] < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc.astype(np.float32)


# ---------------------------
# Visual extraction (MediaPipe FaceMesh)
# ---------------------------
def _normalize_landmarks_array(lm_array):
    """
    lm_array: shape [num_landmarks*3], flattened [x1,y1,z1, x2,y2,z2, ...]
    Normalize by subtracting center and dividing by scale (max distance).
    Returns same shape float32.
    """
    coords = lm_array.reshape(-1, 3)
    # Use x,y only for centering/scale; keep z relative
    xy = coords[:, :2]
    center = np.mean(xy, axis=0)
    xy_centered = xy - center
    scale = np.max(np.linalg.norm(xy_centered, axis=1))
    if scale <= 0:
        scale = 1.0
    coords[:, :2] = xy_centered / scale
    return coords.flatten().astype(np.float32)


def extract_visual_features(video_path, max_len=MAX_LEN, target_dim=VIDEO_FEATURE_DIM, frame_skip=1):
    """
    Extract face-landmark features per frame using MediaPipe FaceMesh.
      - frame_skip: sample one every `frame_skip` frames (1 => all frames)
    Returns: numpy array (max_len, target_dim) float32
    If mediapipe isn't available, raises RuntimeError (or you can decide to return zeros).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not HAS_MEDIAPIPE:
        raise RuntimeError("MediaPipe not available in this environment. Install mediapipe or run in a supported Python (3.8-3.11).")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames_feats = []
    # build FaceMesh with conservative settings
    # Some mediapipe versions support refine_landmarks; this is optional
    face_mesh_kwargs = dict(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    try:
        # add refine flag if supported
        import inspect
        sig = inspect.signature(mp_face_mesh.FaceMesh.__init__)
        if "refine_landmarks" in sig.parameters:
            face_mesh_kwargs["refine_landmarks"] = False
    except Exception:
        pass

    with mp_face_mesh.FaceMesh(**face_mesh_kwargs) as fm:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_skip > 1 and (frame_idx % frame_skip != 0):
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)
            if results and getattr(results, "multi_face_landmarks", None):
                lm = results.multi_face_landmarks[0].landmark
                arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
                arr = _normalize_landmarks_array(arr)  # normalize per frame
                frames_feats.append(arr)
            else:
                # pad with zeros for that frame
                frames_feats.append(np.zeros(468 * 3, dtype=np.float32))

    cap.release()

    if len(frames_feats) == 0:
        # return zeros if no frames
        return np.zeros((max_len, target_dim), dtype=np.float32)

    features = np.stack(frames_feats, axis=0)  # [T, D]

    # pad/truncate time dimension
    if features.shape[0] < max_len:
        pad_t = max_len - features.shape[0]
        features = np.pad(features, ((0, pad_t), (0, 0)), mode="constant")
    else:
        features = features[:max_len, :]

    # pad/truncate feature dim
    if features.shape[1] < target_dim:
        pad_w = target_dim - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_w)), mode="constant")
    else:
        features = features[:, :target_dim]

    return features.astype(np.float32)


# ---------------------------
# Combined convenience function
# ---------------------------
def extract_audio_and_visual_from_video(video_path, save_npy=False, out_dir="extracted_features",
                                       max_len=MAX_LEN, n_mfcc=N_MFCC, target_dim=VIDEO_FEATURE_DIM,
                                       frame_skip=1, cleanup_audio=True):
    """
    Extract audio MFCC features and visual features from a video.
    Returns: (audio_mfcc [max_len, n_mfcc], visual_feats [max_len, target_dim])
    If save_npy=True -> saves returns (audio_path, visual_path) alongside arrays.
    """
    # audio
    y = extract_audio_from_video(video_path, cleanup=cleanup_audio)
    audio_feats = extract_audio_features(y, max_len=max_len, n_mfcc=n_mfcc)

    # visual (may raise RuntimeError if mediapipe missing)
    visual_feats = extract_visual_features(video_path, max_len=max_len, target_dim=target_dim, frame_skip=frame_skip)

    if save_npy:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        a_path = os.path.join(out_dir, f"{base}_audio_mfcc.npy")
        v_path = os.path.join(out_dir, f"{base}_visual.npy")
        np.save(a_path, audio_feats)
        np.save(v_path, visual_feats)
        return audio_feats, visual_feats, a_path, v_path

    return audio_feats, visual_feats
