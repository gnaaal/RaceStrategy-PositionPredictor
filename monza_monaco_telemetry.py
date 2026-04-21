import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. Memuat Data
df = pd.read_excel('f1_Dataset_Monza_Monaco_2023.xlsx')

# 2. Transformasi Target Klasifikasi
def categorize_position(change):
    if change > 0:
        return 'Naik'
    elif change < 0:
        return 'Turun'
    else:
        return 'Tetap'

df['Target_Class'] = df['Position_Change'].apply(categorize_position)

# 3. Memilih Fitur yang akan digunakan
# Membuang kolom teks yang tidak relevan untuk prediksi seperti Driver atau Race
features = ['TyreLife', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'Compound']
X = df[features]
y = df['Target_Class']

# 4. Penanganan Nilai Kosong
# Mengisi nilai kosong pada fitur numerik dengan nilai rata-rata kolom tersebut
numerics = ['TyreLife', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress']
X[numerics] = X[numerics].fillna(X[numerics].mean())

# 5. Encoding Data Kategorikal (Compound Ban)
# Mengubah teks SOFT/MEDIUM/HARD menjadi angka 0 dan 1
X = pd.get_dummies(X, columns=['Compound'], drop_first=True)

# 6. Standarisasi Fitur Numerik
scaler = StandardScaler()
X_scaled = X.copy()
# Kita hanya menstandarisasi kolom numerik asli, hasil One-Hot Encoding tidak perlu distandarisasi
X_scaled[numerics] = scaler.fit_transform(X[numerics])

print("Data siap digunakan! Dimensi data saat ini:", X_scaled.shape)
print("\n5 Baris pertama data yang sudah diproses:\n", X_scaled.head())

# Memeriksa keseimbangan jumlah kelas (Class Imbalance)
class_counts = y.value_counts()

plt.figure(figsize=(8, 5))
plt.bar(class_counts.index, class_counts.values, color=['gray', 'blue', 'red'])
plt.title('Distribusi Kelas Perubahan Posisi Pembalap')
plt.xlabel('Kategori Posisi')
plt.ylabel('Jumlah Sampel')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 7. Pembagian Data (Train-Test Split)
# Membagi data: 80% untuk pelatihan (training) dan 20% untuk pengujian (testing)
# stratify=y memastikan proporsi kelas Naik/Tetap/Turun terbagi rata di kedua set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 8. Inisialisasi Model Machine Learning
# Parameter class_weight='balanced' sangat penting di sini untuk membantu model
# karena dari grafik sebelumnya, jumlah kelas 'Tetap' jauh lebih banyak (imbalance)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
svm_model = SVC(kernel='rbf', C=1.0, random_state=42, class_weight='balanced')

# 9. Pelatihan Model (Training)
print("\n[INFO] Sedang melatih model...")
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# 10. Pengujian Model (Prediction)
# Meminta model menebak jawaban (kelas) dari data uji
y_pred_rf = rf_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)

# 11. Evaluasi Kinerja Model
# Menampilkan laporan metrik lengkap (Akurasi, Precision, Recall, F1-Score)
print("\n=== Kinerja Random Forest Classifier ===")
print(f"Akurasi: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

print("\n=== Kinerja Support Vector Machine (SVM) ===")
print(f"Akurasi: {accuracy_score(y_test, y_pred_svm):.4f}")
print(classification_report(y_test, y_pred_svm))

# 12. Visualisasi Evaluasi (Confusion Matrix)
plt.figure(figsize=(12, 5))
labels = ['Naik', 'Tetap', 'Turun']

# Plot 1 untuk Random Forest
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_rf, labels=labels), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Prediksi Model')
plt.ylabel('Data Aktual (Kenyataan)')

# Plot 2 untuk SVM
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_svm, labels=labels), annot=True, fmt='d', cmap='Reds', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - SVM')
plt.xlabel('Prediksi Model')
plt.ylabel('Data Aktual (Kenyataan)')

plt.tight_layout()
plt.show()
