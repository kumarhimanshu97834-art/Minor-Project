
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn as nn
import numpy as np
import cv2
import uuid
import io

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "resnet50_deepfake.pth"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD TRAINED MODEL ----------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

resnet50 = models.resnet50(weights=None)
resnet50.fc = nn.Linear(resnet50.fc.in_features, 1)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

resnet50.load_state_dict(state_dict)
resnet50 = resnet50.to(device)
resnet50.eval()

print("Trained ResNet50 model loaded successfully!")
print("Device:", device)
print("Prediction threshold:", THRESHOLD)

# ---------------- FACE DETECTOR ----------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError("Could not load Haar face detector")

# ---------------- PREDICTION HELPERS ----------------

def classify_probability(probability):
    return "Fake" if probability >= THRESHOLD else "Real"


def predict_face(face_image):
    image_tensor = preprocess(face_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = resnet50(image_tensor)
        probability = torch.sigmoid(output).item()

    return probability


def predict_image(image):
    image_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    detected_faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(detected_faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image"
        )

    x, y, w, h = max(
        detected_faces,
        key=lambda box: box[2] * box[3]
    )

    face = image_array[y:y+h, x:x+w]
    face_image = Image.fromarray(face)

    probability = predict_face(face_image)
    result = classify_probability(probability)

    return {
        "prediction": result,
        "fake_probability": round(probability * 100, 2),
        "real_probability": round((1 - probability) * 100, 2),
        "threshold": THRESHOLD,
        "faces_analyzed": 1
    }


def extract_faces(video_path, max_frames=10):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail="Could not open uploaded video"
        )

    faces = []
    frame_index = 0
    max_scan_frames = 100

    while cap.isOpened() and frame_index < max_scan_frames:
        success, frame = cap.read()

        if not success:
            break

        frame_index += 1

        if frame_index % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40)
        )

        if len(detected_faces) > 0:
            x, y, w, h = max(
                detected_faces,
                key=lambda box: box[2] * box[3]
            )

            face = frame[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            faces.append(Image.fromarray(face))

        if len(faces) >= max_frames:
            break

    cap.release()
    return faces


def predict_video(video_path):
    faces = extract_faces(video_path)

    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the video"
        )

    predictions = [predict_face(face) for face in faces]

    avg_probability = float(np.mean(predictions))
    result = classify_probability(avg_probability)

    return {
        "prediction": result,
        "fake_probability": round(avg_probability * 100, 2),
        "real_probability": round((1 - avg_probability) * 100, 2),
        "threshold": THRESHOLD,
        "frames_analyzed": len(predictions)
    }


# ---------------- FASTAPI APPLICATION ----------------

app = FastAPI(
    title="Deepfake Detection AI API",
    description="API for Deepfake Detection Project",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://minor-project-smoky-nine.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Deepfake Detection AI Backend is Running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Deepfake Detection AI",
        "model": "ResNet50",
        "threshold": THRESHOLD
    }


# ---------------- VIDEO UPLOAD ----------------

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    allowed_extensions = {".mp4", ".mov", ".avi"}
    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only MP4, MOV and AVI videos are allowed"
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty video")

    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Video size must be below 100 MB"
        )

    video_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{video_id}{extension}"

    file_path.write_bytes(content)

    return {
        "message": "Video uploaded successfully",
        "video_id": video_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "status": "uploaded"
    }


# ---------------- VIDEO PREDICTION ----------------

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    allowed_extensions = {".mp4", ".mov", ".avi"}
    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only MP4, MOV and AVI videos are allowed"
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty video")

    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video too large")

    video_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{video_id}{extension}"

    try:
        file_path.write_bytes(content)

        result = predict_video(file_path)

        return {
            "video_id": video_id,
            "filename": file.filename,
            **result
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if file_path.exists():
            file_path.unlink()


# ---------------- IMAGE PREDICTION ----------------

@app.post("/api/predict-image")
async def predict_image_api(file: UploadFile = File(...)):
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG and WEBP images are allowed"
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty image")

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must be below 10 MB"
        )

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        result = predict_image(image)

        return {
            "filename": file.filename,
            **result
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
