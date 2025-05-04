
import os
import sys
import gc
import time
import warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import joblib
import pickle  # For saving label mapping if needed
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.exceptions import ConvergenceWarning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns


# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.multiclass')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics._classification')
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ==============================================================================
# PART 1: URDU DEEPFAKE AUDIO DETECTION (80/10/10 Split)
# ==============================================================================
print("\n" + "="*20 + " PART 1: URDU DEEPFAKE AUDIO DETECTION " + "="*20)

# --- Configuration (Part 1) ---
FEATURE_TYPE = 'mfcc'
MAX_AUDIO_LEN_SECONDS = 5
N_MFCC = 40
RANDOM_STATE_P1 = 42
# --- CORRECTED Split Sizes for 80/10/10 ---
TEST_SPLIT_SIZE_P1 = 0.10 # 10% for Test set
VAL_SPLIT_SIZE_P1 = 0.10  # 10% for Validation set (relative to original)
# Training set size will be the remainder (80%)
# --------------------------
DNN_EPOCHS_P1 = 50
DNN_BATCH_SIZE_P1 = 32
DNN_PATIENCE_P1 = 10
NUM_WORKERS_MAP = 1 # Use 1 for stability

# --- 1. Load Dataset (Part 1) ---
print("\n--- 1. Loading Dataset (Part 1) ---")
ds_p1 = None
retries = 3
for attempt in range(retries):
    try:
        print(f"Attempt {attempt+1}/{retries} to load 'CSALT/deepfake_detection_dataset_urdu'...")
        ds_p1 = load_dataset("CSALT/deepfake_detection_dataset_urdu", trust_remote_code=True)
        print("Dataset loaded successfully.")
        break
    except Exception as e:
        print(f"Error loading dataset (Attempt {attempt+1}/{retries}): {e}")
        if "429" in str(e): time.sleep(30)
        elif attempt < retries - 1: time.sleep(10)
        else: ds_p1 = None

if not ds_p1:
    print("Exiting script because dataset loading failed.")
    sys.exit(1)

print("Dataset structure:")
print(ds_p1)

# Handle potential single-split dataset
if 'train' not in ds_p1 and len(ds_p1.keys()) == 1:
     default_key = list(ds_p1.keys())[0]
     print(f"Warning: Only one split ('{default_key}') found. Using it as the main dataset.")
     ds_p1['train'] = ds_p1.pop(default_key)
elif 'train' not in ds_p1:
     print("Error: 'train' split not found in the dataset. Exiting.")
     sys.exit(1)

if 'audio' not in ds_p1['train'].column_names:
    print("Error: 'audio' column not found in 'train' split. Exiting.")
    sys.exit(1)

print("Columns in loaded 'train' split:", ds_p1['train'].column_names)


# --- 2. Preprocessing (Part 1) ---
print("\n--- 2. Defining Preprocessing Functions (Part 1) ---")

def extract_features_p1(audio_data, sample_rate, feature_type='mfcc', n_mfcc=40, max_len_sec=5):
    if audio_data is None or sample_rate is None or audio_data.size == 0: return None
    target_len_samples = int(max_len_sec * sample_rate)
    current_len = len(audio_data)
    if current_len > target_len_samples: audio_data = audio_data[:target_len_samples]
    elif current_len < target_len_samples:
        pad_width = target_len_samples - current_len
        if pad_width >= 0: audio_data = np.pad(audio_data, (0, pad_width), 'constant')
    if audio_data.size == 0: return None
    try:
        if feature_type == 'mfcc':
            mfccs = librosa.feature.mfcc(y=audio_data.astype(np.float32), sr=sample_rate, n_mfcc=n_mfcc)
            if not np.isfinite(mfccs).all(): return None
            features = np.mean(mfccs.T, axis=0)
            return None if np.isnan(features).any() else features
        else: return None
    except Exception as e: print(f"Feature extraction error: {e}"); return None

def derive_label_from_path(file_path):
    if not file_path or not isinstance(file_path, str): return None
    normalized_path = os.path.normpath(file_path).lower()
    keywords_bonafide = ['bonafide', 'real']
    keywords_deepfake = ['deepfake', 'spoof', 'fake', 'logical_access', 'physical_access']
    basename = os.path.basename(normalized_path)
    if basename.startswith('la_t_') or basename.startswith('pa_t_'): return 1
    if basename.startswith('la_e_') or basename.startswith('pa_e_'): return 0
    if any(keyword in normalized_path for keyword in keywords_bonafide): return 0
    if any(keyword in normalized_path for keyword in keywords_deepfake): return 1
    return None

def preprocess_audio_data(example):
    audio_info = example.get('audio', {}); audio_array = audio_info.get('array'); sample_rate = audio_info.get('sampling_rate'); file_path = audio_info.get('path'); label = derive_label_from_path(file_path)
    if audio_array is None and file_path:
        try: audio_array, sr_from_file = sf.read(file_path); sample_rate = sr_from_file if sample_rate is None or sample_rate != sr_from_file else sample_rate
        except Exception as e: print(f"Error reading {file_path}: {e}"); audio_array = None
    elif audio_array is None or sample_rate is None: audio_array = None
    features = None
    if audio_array is not None and sample_rate is not None:
        if audio_array.ndim > 1: audio_array = np.mean(audio_array, axis=1) if audio_array.shape[1] > 0 else None
        if audio_array is not None and audio_array.size > 0: features = extract_features_p1(audio_array, sample_rate, FEATURE_TYPE, N_MFCC, MAX_AUDIO_LEN_SECONDS)
    return {'features': features, 'label': label}

print("\n--- 2b. Applying Preprocessing via .map() (Part 1) ---")
original_cols = ds_p1['train'].column_names; cols_to_remove = [col for col in ['audio', 'file'] if col in original_cols]
print(f"Columns present: {original_cols}. Removing: {cols_to_remove}.")
if 'label' not in original_cols: print("Info: 'label' column not found, will be derived.")
processed_ds_p1 = ds_p1.map(preprocess_audio_data, remove_columns=cols_to_remove, num_proc=NUM_WORKERS_MAP, load_from_cache_file=False)
print("Preprocessing finished. Columns after:", processed_ds_p1['train'].column_names)

print("\n--- 2c. Filtering Failed Samples (Part 1) ---")
original_counts = {split: len(processed_ds_p1[split]) for split in processed_ds_p1.keys()}
filtered_ds_p1 = processed_ds_p1.filter(lambda ex: ex['features'] is not None and ex['label'] is not None, num_proc=NUM_WORKERS_MAP)
final_counts = {split: len(filtered_ds_p1[split]) for split in filtered_ds_p1.keys()}
for split in original_counts: print(f"{split}: {original_counts[split] - final_counts.get(split, 0)} removed ({final_counts.get(split, 0)} remaining).")
if 'train' not in filtered_ds_p1 or len(filtered_ds_p1['train']) == 0: print("\nError: No valid data left. Exiting."); sys.exit(1)

# --- 3. Preparing Data for Models (Part 1) ---
print("\n--- 3. Preparing Data for Models (Part 1) ---")
all_features_p1 = np.array(filtered_ds_p1['train']['features'])
all_labels_p1 = np.array(filtered_ds_p1['train']['label'])
print(f"Label distribution before splitting: {dict(zip(*np.unique(all_labels_p1, return_counts=True)))}")

print(f"Splitting data into Train (~{100*(1-TEST_SPLIT_SIZE_P1-VAL_SPLIT_SIZE_P1):.0f}%)/Validation ({VAL_SPLIT_SIZE_P1:.0%})/Test ({TEST_SPLIT_SIZE_P1:.0%}) using sklearn...")
# First split: Create Train+Val and Test sets (90% / 10%)
X_train_val_p1, X_test_p1, y_train_val_p1, y_test_p1 = train_test_split(
    all_features_p1, all_labels_p1,
    test_size=TEST_SPLIT_SIZE_P1, # CORRECTED: Use 0.10
    random_state=RANDOM_STATE_P1,
    stratify=all_labels_p1
)

# Second split: Create Train and Validation sets from Train+Val (80% / 10% of original)
# Validation size relative to the Train+Val set = 0.10 / (1 - 0.10) = 0.10 / 0.90 = 1/9
val_split_fraction = VAL_SPLIT_SIZE_P1 / (1.0 - TEST_SPLIT_SIZE_P1) # CORRECTED: Calculation based on VAL_SPLIT_SIZE_P1 = 0.10
X_train_p1, X_val_p1, y_train_p1, y_val_p1 = train_test_split(
    X_train_val_p1, y_train_val_p1,
    test_size=val_split_fraction, # CORRECTED: Use the calculated 1/9
    random_state=RANDOM_STATE_P1,
    stratify=y_train_val_p1
)

if X_train_p1.shape[0] == 0 or X_val_p1.shape[0] == 0 or X_test_p1.shape[0] == 0: print("Error: One or more data splits are empty. Exiting."); sys.exit(1)
print("Label Distribution After Splitting:")
unique_train, counts_train = np.unique(y_train_p1, return_counts=True); print(f"Training:   {dict(zip(unique_train, counts_train))}")
unique_val, counts_val = np.unique(y_val_p1, return_counts=True); print(f"Validation: {dict(zip(unique_val, counts_val))}")
unique_test, counts_test = np.unique(y_test_p1, return_counts=True); print(f"Test:       {dict(zip(unique_test, counts_test))}")
print("Final Data Shapes:")
print(f"Training:   X={X_train_p1.shape}, y={y_train_p1.shape}")
print(f"Validation: X={X_val_p1.shape}, y={y_val_p1.shape}")
print(f"Test:       X={X_test_p1.shape}, y={y_test_p1.shape}")

print("Scaling features using StandardScaler...")
scaler_p1 = StandardScaler(); X_train_scaled_p1 = scaler_p1.fit_transform(X_train_p1)
X_val_scaled_p1 = scaler_p1.transform(X_val_p1); X_test_scaled_p1 = scaler_p1.transform(X_test_p1)
print("Feature scaling complete.")

# --- 4. Model Training and Evaluation (Part 1) ---
print("\n--- 4. Model Training and Evaluation (Part 1) ---")
results_p1 = {}; results_val_p1 = {}; results_train_p1 = {}

def evaluate_binary_model(model_name, model, X_train_sc, y_train, X_val_sc, y_val, X_test_sc, y_test):
    print(f"Training {model_name}..."); start_time = time.time()
    if not isinstance(model, tf.keras.Model): model.fit(X_train_sc, y_train)
    training_time = time.time() - start_time; print(f"{model_name} training completed in {training_time:.2f} seconds.")
    def get_preds_scores(data_scaled):
        if isinstance(model, tf.keras.Model): proba = model.predict(data_scaled).flatten(); pred = (proba > 0.5).astype(int); return pred, proba
        else: pred = model.predict(data_scaled); proba = model.predict_proba(data_scaled)[:, 1] if hasattr(model, "predict_proba") else None; return pred, proba
    print(f"Evaluating {model_name} on Test Set...")
    y_pred_test, y_pred_proba_test = get_preds_scores(X_test_sc)
    acc_test=accuracy_score(y_test, y_pred_test); prec_test=precision_score(y_test, y_pred_test, zero_division=0); rec_test=recall_score(y_test, y_pred_test, zero_division=0); f1_test=f1_score(y_test, y_pred_test, zero_division=0)
    auc_test = roc_auc_score(y_test, y_pred_proba_test) if y_pred_proba_test is not None else np.nan
    results_p1[model_name] = {"Accuracy": acc_test, "Precision": prec_test, "Recall": rec_test, "F1-Score": f1_test, "AUC-ROC": auc_test, "Pred_Proba": y_pred_proba_test}
    print(f"Test Results: Acc={acc_test:.4f}, P={prec_test:.4f}, R={rec_test:.4f}, F1={f1_test:.4f}, AUC={auc_test if not np.isnan(auc_test) else 'N/A'}")
    print(f"Evaluating {model_name} on Validation Set...")
    y_pred_val, y_pred_proba_val = get_preds_scores(X_val_sc)
    acc_val=accuracy_score(y_val, y_pred_val); prec_val=precision_score(y_val, y_pred_val, zero_division=0); rec_val=recall_score(y_val, y_pred_val, zero_division=0); f1_val=f1_score(y_val, y_pred_val, zero_division=0)
    auc_val = roc_auc_score(y_val, y_pred_proba_val) if y_pred_proba_val is not None else np.nan
    results_val_p1[model_name] = {"Accuracy": acc_val, "Precision": prec_val, "Recall": rec_val, "F1-Score": f1_val, "AUC-ROC": auc_val}
    print(f"Val Results:  Acc={acc_val:.4f}, P={prec_val:.4f}, R={rec_val:.4f}, F1={f1_val:.4f}, AUC={auc_val if not np.isnan(auc_val) else 'N/A'}")
    print(f"Evaluating {model_name} on Training Set...")
    y_pred_train, y_pred_proba_train = get_preds_scores(X_train_sc)
    acc_train=accuracy_score(y_train, y_pred_train); prec_train=precision_score(y_train, y_pred_train, zero_division=0); rec_train=recall_score(y_train, y_pred_train, zero_division=0); f1_train=f1_score(y_train, y_pred_train, zero_division=0)
    auc_train = roc_auc_score(y_train, y_pred_proba_train) if y_pred_proba_train is not None else np.nan
    results_train_p1[model_name] = {"Accuracy": acc_train, "Precision": prec_train, "Recall": rec_train, "F1-Score": f1_train, "AUC-ROC": auc_train}
    print(f"Train Results: Acc={acc_train:.4f}, P={prec_train:.4f}, R={rec_train:.4f}, F1={f1_train:.4f}, AUC={auc_train if not np.isnan(auc_train) else 'N/A'}")

print("\n--- Training & Evaluating Scikit-learn Models (Part 1) ---")
sklearn_models_p1 = {
    "SVM": SVC(probability=True, random_state=RANDOM_STATE_P1, C=1.0, class_weight='balanced'),
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE_P1, max_iter=1000, C=1.0, class_weight='balanced'),
    "Perceptron": Perceptron(random_state=RANDOM_STATE_P1, max_iter=1000, tol=1e-3, eta0=0.01, penalty='l2', alpha=0.0001, class_weight='balanced', early_stopping=True, n_iter_no_change=5, validation_fraction=0.1)
}
for name, model in sklearn_models_p1.items():
    evaluate_binary_model(name, model, X_train_scaled_p1, y_train_p1, X_val_scaled_p1, y_val_p1, X_test_scaled_p1, y_test_p1)

# --- DNN Model (Part 1) ---
print("\n--- Training DNN (Part 1) ---")
def build_dnn_p1(input_shape):
    model = Sequential([ Input(shape=(input_shape,), name="Input_Layer"), Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), name="Hidden_Layer_1"), BatchNormalization(name="BatchNorm_1"), Dropout(0.5, name="Dropout_1"), Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), name="Hidden_Layer_2"), BatchNormalization(name="BatchNorm_2"), Dropout(0.4, name="Dropout_2"), Dense(1, activation='sigmoid', name="Output_Layer")], name="Deepfake_Audio_DNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model
dnn_model_p1 = build_dnn_p1(X_train_scaled_p1.shape[1]); dnn_model_p1.summary()
early_stopping_p1 = EarlyStopping(monitor='val_loss', patience=DNN_PATIENCE_P1, restore_best_weights=True, verbose=1)
print("Fitting DNN model..."); start_time = time.time()
history_dnn_p1 = dnn_model_p1.fit(X_train_scaled_p1, y_train_p1, epochs=DNN_EPOCHS_P1, batch_size=DNN_BATCH_SIZE_P1, validation_data=(X_val_scaled_p1, y_val_p1), callbacks=[early_stopping_p1], verbose=1)
dnn_training_time = time.time() - start_time; print(f"DNN training completed in {dnn_training_time:.2f} seconds.")
evaluate_binary_model("DNN", dnn_model_p1, X_train_scaled_p1, y_train_p1, X_val_scaled_p1, y_val_p1, X_test_scaled_p1, y_test_p1)

# --- 5. Final Comparison (Part 1) ---
print("\n" + "="*20 + " FINAL MODEL COMPARISON (Part 1) " + "="*20)
results_df_train_p1 = pd.DataFrame(results_train_p1).T; results_df_val_p1 = pd.DataFrame(results_val_p1).T
results_df_test_p1_full = pd.DataFrame(results_p1).T; results_df_test_p1 = results_df_test_p1_full.drop(columns=['Pred_Proba'], errors='ignore')
metrics_to_display = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
float_formatter = "{:.4f}".format
print("\n--- TRAINING SET RESULTS (Part 1) ---"); df_train_display = results_df_train_p1.reindex(columns=metrics_to_display).sort_values(by="F1-Score", ascending=False, na_position='last'); print(df_train_display.to_string(float_format=float_formatter))
print("\n--- VALIDATION SET RESULTS (Part 1) ---"); df_val_display = results_df_val_p1.reindex(columns=metrics_to_display).sort_values(by="F1-Score", ascending=False, na_position='last'); print(df_val_display.to_string(float_format=float_formatter))
print("\n--- TEST SET RESULTS (Part 1) ---"); df_test_display = results_df_test_p1.reindex(columns=metrics_to_display).sort_values(by="F1-Score", ascending=False, na_position='last'); print(df_test_display.to_string(float_format=float_formatter))

# --- Overfitting/Underfitting Analysis (Part 1) ---
print("\n--- Overfitting/Underfitting Analysis (Part 1) ---"); print("Comparing Train vs Test F1-Scores:")
analysis = []; common_models_p1 = df_train_display.index.intersection(df_test_display.index)
for model_name in common_models_p1:
    if model_name in df_val_display.index:
        train_f1 = df_train_display.loc[model_name, "F1-Score"]; val_f1 = df_val_display.loc[model_name, "F1-Score"]; test_f1 = df_test_display.loc[model_name, "F1-Score"]
        status = "Good Fit / Balanced"; overfit_threshold = 0.05; underfit_threshold_test = 0.85
        valid_f1s = pd.notna([train_f1, val_f1, test_f1]).all()
        if not valid_f1s: status = "Evaluation Incomplete"
        elif (train_f1 > val_f1 + overfit_threshold) and (train_f1 > test_f1 + overfit_threshold): status = "Overfitting"
        elif test_f1 < underfit_threshold_test and (np.isnan(train_f1) or train_f1 < (underfit_threshold_test + 0.05)): status = "Underfitting"
        analysis.append({"Model": model_name, "Train F1": train_f1, "Val F1": val_f1, "Test F1": test_f1, "Train-Test F1 Diff": train_f1 - test_f1 if valid_f1s else np.nan, "Status": status})
    else: print(f"Warning: Results missing for {model_name} in validation set.")
if analysis: analysis_df = pd.DataFrame(analysis); print(analysis_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x))
else: print("Could not perform analysis.")

# --- Plot ROC Curves (Part 1) ---
print("\n--- Plotting ROC Curves (Part 1) ---"); plt.figure(figsize=(10, 8)); roc_plot_success = False
for name, metrics_dict in results_p1.items():
    proba = metrics_dict.get("Pred_Proba", None); auc_score = metrics_dict.get("AUC-ROC", np.nan)
    if proba is not None and not np.isnan(auc_score):
        try: fpr, tpr, _ = roc_curve(y_test_p1, proba); plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.4f})'); roc_plot_success = True
        except Exception as e_roc: print(f"Could not plot ROC for {name}: {e_roc}")
    elif name != "Perceptron" and (proba is None or np.isnan(auc_score)): print(f"Skipping ROC for {name}: No probabilities or valid AUC.")
if roc_plot_success: plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance level (AUC = 0.50)'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Part 1: ROC Curves for Deepfake Audio Detection Models (Test Set)'); plt.legend(loc='lower right'); plt.grid(True); plt.show()
else: print("No valid data to plot ROC curves.")

print("\n" + "="*25 + " PART 1 COMPLETE " + "="*25)
print("\n--- Saving Part 1 Artifacts ---")
output_dir_p1 = "part1_artifacts"
os.makedirs(output_dir_p1, exist_ok=True)

# Save scaler
joblib.dump(scaler_p1, os.path.join(output_dir_p1, "scaler_p1.joblib"))
print("Scaler saved.")

# Save sklearn models
for name, model in sklearn_models_p1.items():
    # Replace spaces for filename safety
    filename = name.lower().replace(" ", "_") + "_p1.joblib"
    joblib.dump(model, os.path.join(output_dir_p1, filename))
    print(f"{name} model saved.")

# Save DNN model
dnn_model_p1.save(os.path.join(output_dir_p1, "dnn_model_p1.keras"))
print("DNN model saved.")

# Optionally save N_MFCC constant
part1_config = {'N_MFCC': N_MFCC, 'MAX_AUDIO_LEN_SECONDS': MAX_AUDIO_LEN_SECONDS}
with open(os.path.join(output_dir_p1, 'part1_config.pkl'), 'wb') as f:
    pickle.dump(part1_config, f)
print("Part 1 config saved.")

print("Part 1 Artifact saving complete.")
