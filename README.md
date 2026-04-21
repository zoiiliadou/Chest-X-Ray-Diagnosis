# Chest X-Ray Medical Diagnosis & Analysis AI

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A professional-grade, AI-powered medical diagnostic web application designed to assist radiologists and medical professionals in detecting Pneumonia from Chest X-Ray images (both standard image formats and DICOM). The system utilizes a fine-tuned MobileNetV2 architecture with Transfer Learning and provides Explainable AI (XAI) features via Grad-CAM heatmaps for high clinical transparency.

## 🚀 Key Features

* **Advanced Medical AI:** Fine-tuned `MobileNetV2` model optimized for binary classification (Normal vs. Pneumonia) with a heavily customized decision threshold for clinical safety.
* **Explainable AI (XAI):** Automatically generates **Grad-CAM Focus Maps** (Heatmaps) to highlight the lung regions the AI focused on, providing visual justifications for positive diagnoses.
* **DICOM Native Support:** Direct handling of `.dcm` files (Clinical standard) using `pydicom`. Extracts patient metadata and safely normalizes pixel arrays avoiding Hounsfield unit visual artifacts.
* **OOD Bouncer (Safety Net):** Dual-layer heuristic and AI-driven screening system that automatically rejects irrelevant images (e.g., passports, PDFs, natural images, CT scans) preventing false AI confidence.
* **Borderline Warnings:** Clinically driven low-confidence thresholds trigger a "Borderline Analysis Warning", strictly advising manual specialist review.
* **Admin Ecosystem & History:** Secure, PIN-protected admin area storing diagnostic history locally via `SQLite`, ensuring data persistence and tracking capabilities.
* **PDF Report Generation:** Automated, programmable `jsPDF` exportation formatting AI results and Patient metadata into industry-standard medical reports.

## 🧠 Model Training & Data Pipeline

The repository includes the complete Jupyter Notebook (`chest-x-ray-code.ipynb`) used for training the AI model. This provides full transparency into our data science pipeline, featuring:
* Data Preprocessing & Augmentation.
* Implementation of Transfer Learning natively on MobileNetV2.
* Training history, metric visualizations (Accuracy, Loss), and evaluation curves.

## 💻 Technology Stack

* **Backend:** FastAPI, Python 3.12, Uvicorn
* **Deep Learning:** TensorFlow (Keras), NumPy, Pillow
* **Medical Imaging Data:** PyDicom
* **Database:** SQLite3
* **Frontend:** Vanilla JavaScript, HTML5/CSS3 (Custom UI framework)
* **Deployment:** Docker, Hugging Face Spaces

## 🛠️ Local Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/Chest-X-Ray-Diagnosis.git
cd Chest-X-Ray-Diagnosis
```

2. **Install Dependencies:**
We recommend using a virtual environment.
```bash
pip install -r requirements.txt
```

3. **Run the Application:**
```bash
uvicorn main:app --reload
```
The application will be available at `http://localhost:8000`.

## 🐳 Docker Deployment

The application includes a `Dockerfile` pre-configured to run under the non-root `user` environment (UID 1000) for standard cloud deployments (like Hugging Face Spaces or AWS EC2).

```bash
docker build -t medical-xray-ai .
docker run -p 7860:7860 medical-xray-ai
```

## 🔒 Security & Medical Disclaimer

**Disclaimer:** This software is designed strictly as an academic/research diagnostic **assistant**. It does not replace professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or qualified health provider.

The local database (`xray_history.db`) stores analytics. In cloud environments (e.g., Hugging Face Free Tier), data is ephemeral and resets upon instance sleep for data-privacy adherence.
