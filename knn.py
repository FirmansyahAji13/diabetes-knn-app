import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

# ==============================
# 1. PREPROCESSING DATA
# ==============================

# Load dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Mapping kolom kategorikal
df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})
df['smoking_history'] = df['smoking_history'].replace({
    'never': 0,
    'Passive smoker': 0,
    'former': 1,
    'current': 2,
    'ever': 1,
    'not current': 1
})

# Pisahkan fitur dan label
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

# ==============================
# 2. OVERSAMPLING MENGGUNAKAN SMOTE
# ==============================

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Jumlah data sebelum SMOTE: {np.bincount(y_train)}")
print(f"Jumlah data setelah SMOTE: {np.bincount(y_train_res)}")

# ==============================
# 3. TRAINING MODEL
# ==============================

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_res, y_train_res)

# Akurasi pada data latih setelah SMOTE
train_accuracy = accuracy_score(y_train_res, knn.predict(X_train_res))

# Simpan model, scaler, dan akurasi
with open('model.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('akurasi_model.pkl', 'wb') as f:
    pickle.dump(train_accuracy, f)

print("✅ Model, Scaler, dan Akurasi berhasil disimpan.")

# ==============================
# 4. EVALUASI MODEL
# ==============================

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n🎯 Akurasi (Test Set): {accuracy * 100:.2f}%")
print(f"📌 Precision : {precision:.2f}")
print(f"📌 Recall    : {recall:.2f}")
print(f"📌 F1 Score  : {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

# ==============================
# 5. VISUALISASI
# ==============================

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negatif', 'Positif'],
            yticklabels=['Negatif', 'Positif'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ==============================
# 6. FEATURE IMPORTANCE (CHI2)
# ==============================

selector = SelectKBest(score_func=chi2, k='all')
X_chi2 = selector.fit_transform(X, y)
scores = selector.scores_

feature_scores = pd.DataFrame({
    'Feature': df.drop('diabetes', axis=1).columns,
    'Score': scores
}).sort_values(by='Score', ascending=False)

print("\n📈 Feature Importance (Chi2):")
print(feature_scores)
