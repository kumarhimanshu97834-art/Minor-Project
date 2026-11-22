
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from PIL import Image
from torchvision import transforms
import cv2

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_faces(video_path, max_frames=10):

    cap = cv2.VideoCapture(str(video_path))

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = []
    frame_index = 0
    max_scan_frames = 100

    while cap.isOpened() and frame_index < max_scan_frames:

        success, frame = cap.read()

        if not success:
            break

        frame_index += 1

        # Process every 5th frame
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
    print("Faces detected:", len(faces), flush=True)

    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the video"
        )
    
    predictions = []
    

    for face in faces:

        image_tensor = preprocess(face).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        print("Running ResNet50 prediction...", flush=True)

        with torch.no_grad():
            output = resnet50(image_tensor)
            probability = torch.sigmoid(output).item()

        predictions.append(probability)

    avg_probability = sum(predictions) / len(predictions)

    result = "Fake" if avg_probability >= 0.5 else "Real"

    return {
        "prediction": result,
        "fake_probability": round(avg_probability * 100, 2),
        "real_probability": round((1 - avg_probability) * 100, 2),
        "frames_analyzed": len(predictions)
    }

def predict_image(image):

    # Convert image to OpenCV format
    image_array = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

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

    # Select the largest detected face
    x, y, w, h = max(
        detected_faces,
        key=lambda box: box[2] * box[3]
    )

    face = image_array[y:y+h, x:x+w]
    face_image = Image.fromarray(face)

    image_tensor = preprocess(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = resnet50(image_tensor)
        probability = torch.sigmoid(output).item()

    result = "Fake" if probability >= 0.5 else "Real"

    return {
        "prediction": result,
        "fake_probability": round(probability * 100, 2),
        "real_probability": round((1 - probability) * 100, 2),
        "faces_analyzed": 1
    }

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "resnet50_deepfake.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet50 = models.resnet50(weights=None)
resnet50.fc = nn.Linear(resnet50.fc.in_features, 1)

resnet50.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

resnet50 = resnet50.to(device)
resnet50.eval()

print("ResNet50 model loaded successfully!")


app = FastAPI(
    title="Deepfake Detection AI API",
    description="API for Deepfake Detection Project",
    version="1.0.0"
)

# Allow React frontend to communicate with Python API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def home():
    return {
        "message": "Deepfake Detection AI Backend is Running!"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Deepfake Detection AI"
    }

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):

    allowed_extensions = {".mp4", ".mov", ".avi"}

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only MP4, MOV and AVI videos are allowed"
        )

    video_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{video_id}{extension}"

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded video is empty"
        )

    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Video size must be below 100 MB"
        )

    file_path.write_bytes(content)

    return {
        "message": "Video uploaded successfully",
        "video_id": video_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "status": "uploaded"
    }
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):

    allowed_extensions = {".mp4", ".mov", ".avi"}
    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only MP4, MOV and AVI videos are allowed"
        )

    video_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{video_id}{extension}"

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty video")

    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video too large")

    file_path.write_bytes(content)

    try:
        result = predict_video(file_path)
        return {
            "video_id": video_id,
            "filename": file.filename,
            **result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if file_path.exists():
            file_path.unlink()


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
        raise HTTPException(
            status_code=400,
            detail="Empty image"
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must be below 10 MB"
        )

    try:
        image = Image.open(__import__("io").BytesIO(content)).convert("RGB")
        result = predict_image(image)

        return {
            "filename": file.filename,
            **result
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )