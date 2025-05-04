import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, f1_score, classification_report
import numpy as np

# Load dataset
df = pd.read_csv(r"D:\parh le bhai\Data Science Assignment 4\dataset.csv")
X_text = df['report']
y = df[['type_blocker', 'type_regression', 'type_bug', 'type_documentation',
        'type_enhancement', 'type_task', 'type_dependency_upgrade']]

# Vectorize text
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(X_text).toarray()
y = y.values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Online Perceptron for each label (manual one-vs-rest)
perceptrons = []
for i in range(y_train.shape[1]):
    p = Perceptron(max_iter=1, warm_start=True, eta0=1.0)
    # First call to partial_fit must include classes
    p.partial_fit(X_train, y_train[:, i], classes=np.array([0, 1]))
    for j in range(1, len(X_train)):
        p.partial_fit(X_train[j:j+1], y_train[j:j+1, i])
    perceptrons.append(p)

# Predictions
y_pred = np.zeros_like(y_test)
for i, model in enumerate(perceptrons):
    y_pred[:, i] = model.predict(X_test)

# Evaluation
print("\n=== Online Perceptron (Manual OvR) ===")
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Micro F1 Score:", f1_score(y_test, y_pred, average='micro'))
print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Classification Report:\n", classification_report(y_test, y_pred))
