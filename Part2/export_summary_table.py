import pandas as pd

# Create summary data manually
data = {
    'Model': ['Logistic Regression', 'SVM', 'Perceptron', 'DNN', 'Online Perceptron'],
    'Hamming Loss': [0.1028, 0.0966, 0.1197, 0.1043, 0.1110],
    'Micro F1': [0.8155, 0.8285, 0.7804, 0.8149, 0.8068],
    'Macro F1': [0.3985, 0.5249, 0.4919, 0.5041, 0.5555],
    'Precision@1': [0.9101, '-', '-', '-', '-'],
    'Precision@3': [0.6235, '-', '-', '-', '-'],
    'Precision@5': [0.3964, '-', '-', '-', '-']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("model_evaluation_summary.csv", index=False)

print("✅ Summary table exported to model_evaluation_summary.csv")
