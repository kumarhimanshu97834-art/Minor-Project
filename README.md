
# Deepfake Detection System

An AI-powered Deepfake Detection System that analyzes images and videos to identify potentially manipulated media using a ResNet50 deep learning model.

## Project Overview

This project provides a web-based platform where users can upload images or videos and receive an AI-generated Real/Fake prediction with corresponding scores.

## Features

- Image Upload and Analysis
- Video Upload and Analysis
- Face Detection using OpenCV
- Deepfake Classification using ResNet50
- Real and Fake Score Visualization
- React-based User Interface
- FastAPI Backend

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React.js, Vite, CSS |
| Backend | Python, FastAPI |
| Deep Learning | PyTorch, ResNet50 |
| Image Processing | OpenCV, Pillow |
| Model | ResNet50 |
| API Testing | Swagger UI |

## Project Structure

```text
DEEP-FAKE-DETECTION/
│
├── notebooks/
├── results/
│
└── Deepfake-Detection-WebApp/
    ├── fronted/
    │   ├── src/
    │   ├── package.json
    │   └── vite.config.js
    │
    ├── backend/
    │
    └── ai-service/
        ├── main.py
        └── models/
            └── resnet50_deepfake.pth
```

## How It Works

1. User uploads an image or video.
2. FastAPI receives the uploaded media.
3. OpenCV detects faces in the media.
4. Extracted face images are resized and normalized.
5. ResNet50 analyzes the face images.
6. The API returns a Real/Fake prediction and scores.
7. React displays the result.

## Installation and Setup

### 1. Clone Repository

```bash
git clone https://github.com/kumarhimanshu97834-art/Minor-Project.git
cd Minor-Project
```

### 2. Start Backend

```bash
cd Deepfake-Detection-WebApp/ai-service

python -m venv venv
```

Activate virtual environment:

Windows:
```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install fastapi uvicorn python-multipart torch torchvision opencv-python==4.10.0.84 pillow
```

Start server:

```bash
python -m uvicorn main:app --port 8001
```

Backend API:
http://127.0.0.1:8001

Swagger Documentation:
http://127.0.0.1:8001/docs

### 3. Start Frontend

Open another terminal:

```bash
cd Deepfake-Detection-WebApp/fronted

npm install
npm run dev
```

Frontend:
http://localhost:5173

## Model Information

- Architecture: ResNet50
- Framework: PyTorch
- Input Size: 224 × 224
- Output: Real or Fake classification

## Limitations

- The current model is experimental.
- Initial training used a small 100-image subset.
- Predictions are not guaranteed to be accurate.
- Results should not be treated as verified evidence of manipulation.

## Future Improvements

- Improve model training with a larger dataset.
- Evaluate performance using proper validation and test datasets.
- Improve face detection and preprocessing.
- Deploy the application online.
- Improve model robustness against unseen deepfakes.

## Disclaimer

This project is developed for educational and research purposes. AI-generated predictions may be inaccurate and should not be considered definitive proof that media is real or fake.

## Author

Himanshu Kumar 

GitHub: https://github.com/kumarhimanshu97834-art
