import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# ===============================
# KLASÖR HAZIRLIĞI
# ===============================
os.makedirs("figures", exist_ok=True)

# ===============================
# VERİYİ YÜKLE
# ===============================
DATA_PATH = "Loan_default.csv"
data = pd.read_csv(DATA_PATH)

print("Veri boyutu:", data.shape)

# ===============================
# HEDEF VE GİRİŞ AYRIMI
# ===============================
target_column = "Default"  # Kaggle veri setindeki hedef kolon
X = data.drop(columns=[target_column])
y = data[target_column]

# ===============================
# KATEGORİK KOLONLARI ENCODE ET
# ===============================
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# ÖLÇEKLEME
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# MODEL
# ===============================
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=50,
    random_state=42,
    verbose=True
)

# ===============================
# EĞİTİM
# ===============================
model.fit(X_train, y_train)

# ===============================
# TAHMİN
# ===============================
y_pred = model.predict(X_test)

# ===============================
# METRİKLER
# ===============================
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# ===============================
# LOSS GRAFİĞİ
# ===============================
plt.figure()
plt.plot(model.loss_curve_)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.grid(True)
plt.savefig("figures/training_loss.png")
plt.close()

# ===============================
# CONFUSION MATRIX GÖRSELİ
# ===============================
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("figures/confusion_matrix.png")
plt.close()

print("\nGrafikler 'figures/' klasörüne kaydedildi.")
