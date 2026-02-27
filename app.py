# app.py
import os
import uuid
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import your StressDetector from main.py
from main import StressDetector

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Stress Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None

@app.on_event("startup")
def startup_event():
    global detector
    try:
        print("Starting up: loading StressDetector...")
        detector = StressDetector(model_path="transformer_model.pt")
        print("Model loaded successfully.")
    except Exception as e:
        print("Failed to load model in startup_event:", e)
        traceback.print_exc()
        raise

@app.get("/")
def root():
    return {"status": "ok", "message": "Stress Detection API running"}

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """Accepts form field 'video' and returns JSON containing 'prediction' or 'error'."""
    if detector is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    # use unique filename to avoid overwriting / locking issues
    ext = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
    tmp_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, tmp_name)

    try:
        contents = await video.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        print(f"[predict] Saved uploaded file to: {file_path}")

        # Run prediction
        result = detector.predict(file_path)
        print(f"[predict] Raw detector.predict result: {repr(result)}")

        # Normalize to a JSON-serializable primitive
        if result is None:
            return JSONResponse(status_code=500, content={"error": "Detector returned None"})

        # If dict and contains 'prediction', use it
        if isinstance(result, dict):
            if "prediction" in result:
                prediction = result["prediction"]
            else:
                # return entire dict as string
                prediction = str(result)
        # Torch tensor or numpy scalar => convert to Python primitive
        elif hasattr(result, "item"):
            try:
                prediction = result.item()
            except Exception:
                prediction = str(result)
        else:
            prediction = result

        # make sure final is string (or primitive)
        if isinstance(prediction, (bytes, bytearray)):
            prediction = prediction.decode(errors="ignore")
        prediction = str(prediction)

        return {"prediction": prediction}

    except Exception as e:
        print("[predict] Error during prediction:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # cleanup uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[predict] Removed file: {file_path}")
        except Exception as cleanup_err:
            print("[predict] Error cleaning up file:", cleanup_err)

