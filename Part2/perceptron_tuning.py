import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

# Load dataset
df = pd.read_csv(r"D:\parh le bhai\Data Science Assignment 4\dataset.csv")
X_text = df['report']
y = df[['type_blocker', 'type_regression', 'type_bug', 'type_documentation',
        'type_enhancement', 'type_task', 'type_dependency_upgrade']]

# Vectorize
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(X_text)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model and parameter grid
model = OneVsRestClassifier(Perceptron(max_iter=1000))
param_grid = {'estimator__eta0': [0.1, 0.5, 1.0], 'estimator__penalty': [None, 'l2']}

# Grid search
grid = GridSearchCV(model, param_grid, scoring='f1_micro', cv=3)
grid.fit(X_train, y_train)

# Output
print("Best Params:", grid.best_params_)
y_pred = grid.predict(X_test)
print("Tuned Micro F1:", f1_score(y_test, y_pred, average='micro'))
print("Tuned Macro F1:", f1_score(y_test, y_pred, average='macro'))
