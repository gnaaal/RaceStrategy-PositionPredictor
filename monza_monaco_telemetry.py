import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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