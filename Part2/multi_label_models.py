import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, hamming_loss, f1_score
from sklearn.pipeline import make_pipeline

# Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv("dataset.csv")

# Separate features and labels
X_text = df['report']
y = df[['type_blocker', 'type_regression', 'type_bug', 'type_documentation',
        'type_enhancement', 'type_task', 'type_dependency_upgrade']]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n\n=== {name} ===")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Hamming Loss:", hamming_loss(y_test, y_pred))
    print("Micro F1 Score:", f1_score(y_test, y_pred, average='micro'))
    print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))

# 1. Logistic Regression
lr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
evaluate_model("Logistic Regression", lr)

# 2. SVM (Linear)
svm = OneVsRestClassifier(LinearSVC())
evaluate_model("SVM (Linear)", svm)

# 3. Perceptron
perc = OneVsRestClassifier(Perceptron(max_iter=1000, eta0=1.0, random_state=42))
evaluate_model("Perceptron", perc)

# 4. Deep Neural Network (TF-IDF → Dense Layers)
# Convert sparse matrix to dense for Keras
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

dnn = Sequential()
dnn.add(Dense(128, input_dim=X_train_dense.shape[1], activation='relu'))
dnn.add(Dense(64, activation='relu'))
dnn.add(Dense(y_train.shape[1], activation='sigmoid'))

dnn.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
dnn.fit(X_train_dense, y_train, epochs=10, batch_size=32, verbose=1)

# DNN Evaluation
y_pred_dnn = dnn.predict(X_test_dense)
y_pred_dnn_bin = (y_pred_dnn > 0.5).astype(int)

print("\n\n=== Deep Neural Network ===")
print("Classification Report:\n", classification_report(y_test, y_pred_dnn_bin))
print("Hamming Loss:", hamming_loss(y_test, y_pred_dnn_bin))
print("Micro F1 Score:", f1_score(y_test, y_pred_dnn_bin, average='micro'))
print("Macro F1 Score:", f1_score(y_test, y_pred_dnn_bin, average='macro'))
