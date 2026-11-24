# Deepfake Detection System

An AI-powered web application that analyzes images and videos to identify potentially manipulated media using a ResNet50 deep learning model.

## 🚀 Live Project

**Try the application here:**

🌐 **[Open Live Deepfake Detection Website](https://minor-project-smoky-nine.vercel.app/)**

⚙️ **Backend API:** https://minor-project-1-7g7c.onrender.com

🩺 **API Health Check:** https://minor-project-1-7g7c.onrender.com/health

📚 **API Documentation:** https://minor-project-1-7g7c.onrender.com/docs

Upload an image or video and view the AI-generated Real/Fake prediction with corresponding scores.

> Note: The backend is hosted on Render's free service and may take some time to wake up after inactivity.

---

## 📌 Project Overview

This project provides a web-based platform where users can upload images or videos and receive an AI-generated Real/Fake prediction.

The system uses face detection, image preprocessing and a ResNet50-based deep learning model to analyze visual content.

## ✨ Features

- Image Upload and Analysis
- Video Upload and Analysis
- Face Detection using OpenCV
- Deepfake Classification using ResNet50
- Real and Fake Score Visualization
- Frame-based Video Analysis
- React-based User Interface
- FastAPI Backend
- REST API Integration

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Frontend | React.js, Vite, CSS |
| Backend | Python, FastAPI |
| Deep Learning | PyTorch, ResNet50 |
| Image Processing | OpenCV, Pillow |
| Dataset | Celeb-DF-v2 |
| Model | ResNet50 |
| API Testing | Swagger UI |
| Frontend Deployment | Vercel |
| Backend Deployment | Render |

## 📂 Project Structure

```text
Minor-Project/
│
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_face_extraction.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_model_training_and_comparison.ipynb
│   ├── 10_robustness_analysis.ipynb
│   ├── 12_model_explainability.ipynb
│   └── 13_model_error_analysis.ipynb
│
├── results/
│   ├── figures/
│   ├── face_samples/
│   ├── sample_frames/
│   └── model evaluation files
│
├── Deepfake-Detection-WebApp/
│   │
│   ├── ai-service/
│   │   ├── main.py
│   │   ├── models/
│   │   │   └── resnet50_deepfake.pth
│   │   └── requirements.txt
│   │
│   └── fronted/
│       ├── src/
│       ├── public/
│       ├── package.json
│       └── vite.config.js
│
├── README.md
└── .gitignore
```

## ⚙️ How It Works

1. The user uploads an image or video.
2. FastAPI receives and validates the uploaded media.
3. OpenCV processes video frames and detects faces.
4. Face images are resized and normalized.
5. ResNet50 analyzes the processed images.
6. The backend generates a Real/Fake prediction and corresponding scores.
7. React displays the prediction result to the user.

## 🧪 How to Use the Live Application

1. Open the [Live Demo](https://minor-project-smoky-nine.vercel.app/).
2. Select and upload an image or supported video.
3. Click the Predict button.
4. Wait for the analysis to complete.
5. View the Real/Fake prediction and scores.

**Supported video formats:** MP4, MOV, AVI.

**Supported image formats:** JPG, JPEG, PNG, WEBP.

## 💻 Installation and Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kumarhimanshu97834-art/Minor-Project.git

cd Minor-Project
```

### 2. Set Up the Backend

Open a terminal:

```bash
cd Deepfake-Detection-WebApp/ai-service
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
python -m uvicorn main:app --reload
```

**Local Backend:**

http://127.0.0.1:8000

**Swagger API Documentation:**

http://127.0.0.1:8000/docs

### 3. Set Up the Frontend

Open another terminal from the repository root:

```bash
cd Deepfake-Detection-WebApp/fronted
```

Install dependencies:

```bash
npm install
```

Start the React development server:

```bash
npm run dev
```

Open the local URL displayed by Vite, usually:

http://localhost:5173/

**Note:** For fully local frontend-to-backend testing, configure the frontend API base URL to `http://127.0.0.1:8000`. The deployed frontend uses the Render backend URL.

## 🧠 Model Information

- **Architecture:** ResNet50
- **Framework:** PyTorch
- **Dataset:** Celeb-DF-v2
- **Input Image Size:** 224 × 224 pixels
- **Classification:** Binary Classification
- **Output Classes:** Real and Fake

## 📊 Model Experiments

The project repository includes experiments covering:

- Dataset exploration and analysis
- Face extraction and preprocessing
- Custom CNN and ResNet50 comparison
- Model evaluation and threshold analysis
- Confusion matrix and performance metrics
- Grad-CAM model explainability
- Error analysis
- Robustness testing

Different notebooks represent different experimental stages and configurations. Their results should be interpreted separately rather than treated as measurements from one identical model configuration.

## ⚠️ Limitations

- The model is experimental and may produce incorrect predictions.
- Dataset class imbalance can affect performance.
- False positives and false negatives may occur.
- Performance may decrease on unseen videos or heavily distorted images.
- Prediction scores should not be interpreted as guaranteed probabilities of authenticity.

## 🔮 Future Improvements

- Improve real and fake classification performance.
- Improve detection on unseen datasets.
- Enhance face detection and preprocessing.
- Optimize video processing speed.
- Improve model robustness against different manipulation techniques.
- Explore temporal video-based deepfake detection.

## ⚠️ Disclaimer

This project is developed for academic and educational purposes. AI-generated predictions are estimates and should not be treated as definitive proof that any image or video is authentic or manipulated.

## 👨‍💻 Author

**Himanshu Kumar**

GitHub: [kumarhimanshu97834-art](https://github.com/kumarhimanshu97834-art)

**Project Repository:** [Minor-Project](https://github.com/kumarhimanshu97834-art/Minor-Project)

---

⭐ If you find this project interesting, consider giving the repository a star!
