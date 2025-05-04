# ==============================================================================
# Part 2: Multi-Label Defect Prediction - Sklearn-Only Script with Artifact Saving
# ==============================================================================

import os
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, hamming_loss, f1_score

# -----------------------------
# Load and Prepare the Dataset
# -----------------------------
df = pd.read_csv(r"D:\parh le bhai\Data Science Assignment 4\Part2\dataset.csv")
X_text = df['report']
y = df[['type_blocker', 'type_regression', 'type_bug', 'type_documentation',
        'type_enhancement', 'type_task', 'type_dependency_upgrade']]

# -----------------------------
# Vectorize the Text
# -----------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(X_text)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y.values, test_size=0.2, random_state=42)

# -----------------------------
# Create Output Directory
# -----------------------------
output_dir = r"D:\parh le bhai\Data Science Assignment 4\Part2\part2_artifacts"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))

# -----------------------------
# Model Evaluation Function
# -----------------------------
def evaluate_model(name, model, X_test, y_test, save=True):
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Hamming Loss:", hamming_loss(y_test, y_pred))
    print("Micro F1 Score:", f1_score(y_test, y_pred, average='micro'))
    print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))
    if save:
        joblib.dump(model, os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_p2.joblib"))

# -----------------------------
# Logistic Regression
# -----------------------------
lr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr.fit(X_train, y_train)
evaluate_model("Logistic Regression", lr, X_test, y_test)

# -----------------------------
# SVM
# -----------------------------
svm = OneVsRestClassifier(LinearSVC())
svm.fit(X_train, y_train)
evaluate_model("SVM", svm, X_test, y_test)

# -----------------------------
# Perceptron
# -----------------------------
perc = OneVsRestClassifier(Perceptron(max_iter=1000, eta0=1.0))
perc.fit(X_train, y_train)
evaluate_model("Perceptron", perc, X_test, y_test)

# -----------------------------
# Save Config
# -----------------------------
part2_config = {
    'labels': list(y.columns),
    'max_features': 3000
}
with open(os.path.join(output_dir, 'part2_config.pkl'), 'wb') as f:
    pickle.dump(part2_config, f)

print("\n✅ All Part 2 artifacts saved to:", output_dir)
