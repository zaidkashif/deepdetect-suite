import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss

# Load and prepare data
df = pd.read_csv("dataset.csv")
X_text = df['report']
y = df[['type_blocker', 'type_regression', 'type_bug', 'type_documentation',
        'type_enhancement', 'type_task', 'type_dependency_upgrade']]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2, random_state=42)

# Logistic Regression (One-vs-Rest)
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test)

# True labels
y_true = y_test

# Precision@k function
def precision_at_k(y_true, y_proba, k=3):
    total = 0
    correct = 0
    for i in range(len(y_true)):
        top_k = np.argsort(y_proba[i])[-k:]  # Get top-k label indices
        true_labels = np.where(y_true[i] == 1)[0]
        correct += len(set(top_k) & set(true_labels))
        total += k
    return correct / total

# Compute for k=1, 3, 5
for k in [1, 3, 5]:
    p_at_k = precision_at_k(y_true, y_proba, k=k)
    print(f"Precision@{k}: {p_at_k:.4f}")
