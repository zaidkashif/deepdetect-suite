# 🔍 DeepDetect Suite

**DeepDetect Suite** is a unified AI-powered toolkit designed for two core tasks:
1. **Urdu Deepfake Audio Detection**
2. **Multi-label Software Defect Prediction**

It offers a real-time, interactive **Streamlit-based web app** where users can:
- Upload audio clips and receive binary predictions (Deepfake or Bonafide).
- Input bug reports and receive predicted software defect labels.

---

## 🧠 Project Overview

This project integrates classical ML models with intuitive user interfaces to provide an end-to-end AI solution:

### 🔊 Part 1: Urdu Deepfake Audio Detection

**Objective:** Detect whether an uploaded audio clip is *Bonafide (real)* or *Deepfake (spoofed)*.

**Approach:**
- Dataset: Urdu-language audio clips from the `CSALT/deepfake_detection_dataset_urdu`.
- Preprocessing: MFCC features extracted using `librosa`, padded/truncated to a fixed duration.
- Models Used: 
  - Logistic Regression
  - SVM
  - Perceptron
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC.
- Feature Scaling: `StandardScaler` used for normalization.

---

### 🐞 Part 2: Multi-label Software Defect Prediction

**Objective:** Predict multiple defect labels from a given software bug report text.

**Approach:**
- Dataset: Custom dataset of bug report texts with 7 defect labels.
- Preprocessing: TF-IDF vectorization with top 3000 features.
- Models Used:
  - Logistic Regression (One-vs-Rest)
  - SVM (Linear)
  - Perceptron
- Labels Predicted:
  - `type_blocker`, `type_regression`, `type_bug`, `type_documentation`,
    `type_enhancement`, `type_task`, `type_dependency_upgrade`
- Evaluation Metrics: Hamming Loss, Micro/Macro F1 scores, Classification Report.

---

### 🌐 Part 3: Streamlit Web App

**Features:**
- Real-time interaction via clean UI.
- Users can select models (SVM, Logistic Regression, Perceptron).
- Upload audio files or enter text input.
- Confidence-based predictions and results display.
- Dynamic layout using tabs for audio and text-based tasks.

---

## 📁 Repository Structure

deepdetect-suite/
│
├── Part1/
│ ├── part1_script.py
│ └── part1_artifacts/
│ ├── scaler_p1.joblib
│ ├── svm_p1.joblib
│ ├── perceptron_p1.joblib
│ ├── logistic_regression_p1.joblib
│ └── part1_config.pkl
│
├── Part2/
│ ├── part2_script.py
│ └── part2_artifacts/
│ ├── tfidf_vectorizer.joblib
│ ├── svm_p2.joblib
│ ├── perceptron_p2.joblib
│ ├── logistic_regression_p2.joblib
│ └── part2_config.pkl
│
├── Part3/
│ └── app.py ← Streamlit App
│
├── dataset.csv ← Bug report data for Part 2 (sample/cleaned)
├── requirements.txt ← All dependencies
└── README.md

yaml
Copy
Edit

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/deepdetect-suite.git
cd deepdetect-suite
2. Create and Activate Environment (using Conda)
bash
Copy
Edit
conda create -n deepdetect python=3.10
conda activate deepdetect
3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
4. Launch the App
bash
Copy
Edit
cd Part3
streamlit run app.py
🧪 Sample Inputs
📄 Text Example (Part 2)
pgsql
Copy
Edit
The app crashes when saving a new project after installing the latest dependency.
🎵 Audio File (Part 1)
Format: .wav or .flac

Length: ≤ 5 seconds

Language: Urdu speech

✅ Future Enhancements
Add deep learning alternatives using DNN with TensorFlow Lite or ONNX.

Deploy via Docker or Streamlit Cloud for public access.

Enable batch file predictions.

vbnet
Copy
Edit

Would you like me to generate the `requirements.txt` file based on all your code next?
