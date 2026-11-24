# Deepfake Detection Web Application

An AI-powered web application that analyzes uploaded videos and predicts whether the video is **Real or Fake** using a trained ResNet50 deep learning model.

## 🚀 Live Demo

**[Click here to open and test the application](https://minor-project-smoky-nine.vercel.app/)**

Upload a video and check its Real/Fake prediction.

## ✨ Features

- Video upload and analysis
- Real or Fake classification
- Prediction confidence scores
- Frame-based video analysis
- Deep learning model using ResNet50
- Web-based user interface

## 🛠️ Tech Stack

**Frontend**
- React.js
- Vite
- CSS

**Backend**
- Python
- FastAPI
- PyTorch
- OpenCV

**Deep Learning**
- ResNet50
- Celeb-DF-v2 Dataset

**Deployment**
- Frontend: Vercel
- Backend: Render

## 📂 Project Structure

```text
Deepfake-Detection-WebApp/
│
├── ai-service/
│   ├── main.py
│   ├── models/
│   └── requirements.txt
│
├── fronted/
│   ├── src/
│   ├── public/
│   └── package.json
│
└── README.md
```

## 🧪 How to Use

1. Open the [Live Demo](https://minor-project-smoky-nine.vercel.app/).
2. Upload a supported video file.
3. Click the Predict button.
4. Wait for video analysis.
5. View the Real/Fake prediction and scores.

**Supported video formats:** MP4, MOV, AVI.

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/kumarhimanshu97834-art/Minor-Project.git
```

### 2. Start the Backend

```bash
cd ai-service
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
python -m uvicorn main:app --reload
```

Backend documentation:

http://127.0.0.1:8000/docs

### 3. Start the Frontend

Open another terminal:

```bash
cd fronted
npm install
npm run dev
```

Open the local URL provided by Vite.

## 📊 Model Information

- Architecture: ResNet50
- Dataset: Celeb-DF-v2
- Framework: PyTorch
- Task: Binary classification (Real / Fake)

## ⚠️ Disclaimer

This project is developed for academic and educational purposes. Predictions are model estimates and should not be treated as definitive proof that a video is authentic or manipulated.

## 👨‍💻 Author

**Himanshu Kumar**

GitHub: [kumarhimanshu97834-art](https://github.com/kumarhimanshu97834-art)

---

⭐ If you find this project interesting, consider giving the repository a star!