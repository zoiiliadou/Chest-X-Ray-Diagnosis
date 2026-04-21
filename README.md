# Chest X-Ray Medical Diagnosis & Analysis AI

**Live Application:** [https://medicalaitech-chest-x-ray-diagnosis.hf.space](https://medicalaitech-chest-x-ray-diagnosis.hf.space)



A professional-grade, AI-powered medical diagnostic web application designed to assist radiologists and medical professionals in detecting Pneumonia from Chest X-Ray images, supporting both standard image formats and native DICOM files. The system utilizes a fine-tuned MobileNetV2 architecture with Transfer Learning and provides Explainable AI (XAI) features via Grad-CAM heatmaps for enhanced clinical transparency.

## Key Features

* **Advanced Medical Inference:** Fine-tuned `MobileNetV2` model optimized for binary classification (Normal vs. Pneumonia) with heavy custom decision thresholds prioritizing clinical safety over raw accuracy.
* **Explainable AI (XAI):** Automatically generates Grad-CAM Focus Maps (Heatmaps) to highlight the specific lung regions the neural network utilized for its inference, providing visual justification for positive diagnoses.
* **DICOM Native Support:** Direct handling of clinical standard `.dcm` files using `pydicom`. Extracts patient metadata and safely normalizes pixel arrays, avoiding Hounsfield unit visual artifacts.
* **Out-of-Distribution (OOD) Bouncer:** Dual-layer heuristic and AI-driven screening system that automatically rejects irrelevant images (e.g., passports, PDFs, natural images, CT scans) to prevent false-positive diagnostic confidence.
* **Borderline Warnings:** Clinically driven low-confidence thresholds trigger a strict "Borderline Analysis Warning", advising manual specialist review rather than forced automated categorization.
* **Administrative Audit History:** Secure, PIN-protected admin dashboard storing diagnostic history locally via `SQLite`, ensuring data persistence and tracking capabilities.
* **Automated Medical Reporting:** Programmable PDF exportation formatting both the AI results and Patient metadata into industry-standard printable medical reports.

## Model Training & Data Pipeline

The repository includes the complete Jupyter Notebook (`chest-x-ray-code.ipynb`) utilized for training the underlying artificial intelligence. This provides full transparency into the data science pipeline, featuring:
* Dataset procurement, cleaning, and preprocessing.
* Strategic implementation of Transfer Learning natively on the MobileNetV2 architecture.
* Training history, metric visualizations (Accuracy, Validation Loss), and rigorous model evaluation parameters.

## Technology Stack

* **Backend Development:** FastAPI, Python 3.12, Uvicorn
* **Model Inference:** TensorFlow (Keras), NumPy, Pillow
* **Medical Data Parsing:** PyDicom
* **Database Management:** SQLite3
* **Frontend Architecture:** Vanilla JavaScript, HTML5/CSS3 (Custom clinical aesthetic)
* **Deployment & Containerization:** Docker, Hugging Face Spaces

## Local Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/Chest-X-Ray-Diagnosis.git
cd Chest-X-Ray-Diagnosis
```

2. **Install Dependencies:**
It is highly recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

3. **Run the Application:**
```bash
uvicorn main:app --reload
```
The local server will instantiate at `http://localhost:8000`.

## Docker Deployment

The source code includes a `Dockerfile` pre-configured to execute under a non-root environment (UID 1000) for standard cloud provider compatibility (e.g., Hugging Face Spaces, Google Cloud CloudRun, AWS).

```bash
docker build -t medical-xray-ai .
docker run -p 7860:7860 medical-xray-ai
```

## Security & Medical Legal Disclaimer

**Disclaimer:** This software architecture is designed strictly as an academic and research diagnostic assistant. It is not intended to replace professional medical advice, clinical diagnosis, or human treatment execution. Always verify automated findings with a physician or independently qualified healthcare provider.

*Note on Data Storage:* The local database (`xray_history.db`) acts as an analytics repository. Within ephemeral cloud environments (e.g., Hugging Face Spaces Free Tier), instance sleeping will reset this database in adherence to standard data-privacy demonstrations.
