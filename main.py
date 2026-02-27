import os
import uuid
import traceback
import cv2
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import your feature extraction functions
from feature import (
    extract_audio_from_video,
    extract_audio_features,
    extract_visual_features,
    MAX_LEN,
    N_MFCC,
    VIDEO_FEATURE_DIM
)

# -------------------------
# Model Definition
# -------------------------
class AudioVisualFusionModel(nn.Module):
    def __init__(self, audio_dim=40, video_dim=2054, d_model=256, nhead=4, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        self.fuse_proj = nn.Linear(2 * d_model, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attn_audio_to_video = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_video_to_audio = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        fusion_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, audio, video):
        audio = self.audio_proj(audio)
        video = self.video_proj(video)
        fused = torch.cat([audio, video], dim=-1)
        fused = self.fuse_proj(fused)
        B = fused.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        fused_seq = torch.cat([cls_tokens, fused], dim=1)
        enc = self.fusion_encoder(fused_seq)
        enc_audio = enc[:, 1:, :]
        enc_video = enc[:, 1:, :]
        a2v, _ = self.cross_attn_audio_to_video(enc_audio, enc_video, enc_video)
        v2a, _ = self.cross_attn_video_to_audio(enc_video, enc_audio, enc_audio)
        fused_seq = torch.cat([cls_tokens, a2v + v2a], dim=1)
        fused_out = self.fusion_transformer(fused_seq)
        cls_out = fused_out[:, 0, :]
        logits = self.classifier(cls_out)
        return logits



def is_smiling(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.5, 20)
        if len(smiles) > 0:
            return True
    return False

def check_smile_in_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    smile_detected = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        if is_smiling(frame):
            smile_detected = True
            break
        frame_count += 1

    cap.release()
    return smile_detected


# -------------------------
# Stress Detector Class
# -------------------------
class StressDetector:
    def __init__(self, model_path="transformer_model.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioVisualFusionModel(audio_dim=N_MFCC, video_dim=VIDEO_FEATURE_DIM).to(self.device)
        self.model_path = model_path

        if os.path.exists(model_path):
            try:
                sd = torch.load(model_path, map_location=self.device)
                if isinstance(sd, dict) and all(k in self.model.state_dict() for k in sd.keys()):
                    self.model.load_state_dict(sd)
                else:
                    self.model = sd.to(self.device)
                print(f"[StressDetector] Loaded model from {model_path}")
            except Exception as e:
                print(f"[StressDetector] Failed to load model: {e}. Using untrained model.")
        else:
            print(f"[StressDetector] Model file not found. Using untrained model.")

        self.model.eval()

    def _preprocess(self, video_path):
        y = extract_audio_from_video(video_path)
        audio_feats = extract_audio_features(y, max_len=MAX_LEN, n_mfcc=N_MFCC)
        visual_feats = extract_visual_features(video_path, max_len=MAX_LEN, target_dim=VIDEO_FEATURE_DIM)
        audio_tensor = torch.from_numpy(np.asarray(audio_feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        visual_tensor = torch.from_numpy(np.asarray(visual_feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        return audio_tensor, visual_tensor

    def predict(self, video_path):
        # ✅ Rule-based override: smiling face -> non-stress
        if check_smile_in_video(video_path):
            return {"prediction": "nonstress", "confidence": 1.0, "note": "Smiling detected!"}

        try:
            audio_t, visual_t = self._preprocess(video_path)
        except Exception as e:
            return {"error": f"Preprocessing failed: {str(e)}"}

        with torch.no_grad():
            try:
                logits = self.model(audio_t.float(), visual_t.float())
                probs = torch.softmax(logits, dim=-1)
                pred_idx = int(torch.argmax(probs, dim=1).cpu().item())
                label = "stress" if pred_idx == 1 else "nonstress"
                confidence = float(probs[0, pred_idx].cpu().item())
                return {"prediction": label, "confidence": round(confidence, 4)}
            except Exception as e:
                return {"error": f"Inference failed: {str(e)}"}


# -------------------------
# FastAPI App
# -------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Stress Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None

@app.on_event("startup")
def startup_event():
    global detector
    try:
        print("Loading StressDetector...")
        detector = StressDetector(model_path="transformer_model.pt")
        print("Model loaded successfully.")
    except Exception as e:
        print("Failed to load model:", e)
        traceback.print_exc()
        raise

@app.get("/")
def root():
    return {"status": "ok", "message": "Stress Detection API running"}

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    if detector is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    ext = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
    tmp_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, tmp_name)

    try:
        contents = await video.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        result = detector.predict(file_path)
        return result

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
