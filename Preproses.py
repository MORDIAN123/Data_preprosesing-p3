# 1. Mengimpor Library yang Diperlukan
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 2. Membaca File CSV dengan Pemisah Titik Koma (;)
# Gantilah 'data.csv' dengan nama file CSV-mu yang sebenarnya
df = pd.read_csv('data.csv', sep=';')  # Tambahkan parameter sep=';' untuk membaca file yang menggunakan titik koma sebagai pemisah

# 3. Menampilkan Data Sebelum Preprocessing
print("Data Sebelum Preprocessing:")
print(df)

# 4. Mengatasi Nilai yang Hilang (Missing Values) dengan SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Menggunakan rata-rata untuk mengisi nilai yang hilang
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# 5. Encoding Data Kategori
# Label Encoding untuk kolom Purchased
label_encoder = LabelEncoder()
df['Purchased'] = label_encoder.fit_transform(df['Purchased'])

# One-Hot Encoding untuk kolom Country
df = pd.get_dummies(df, columns=['Country'])

# 6. Menampilkan Data Setelah Preprocessing
print("\nData Setelah Preprocessing:")
print(df)

# 7. Menyimpan Hasil Preprocessing ke dalam File CSV Baru
df.to_csv('preprocessing_output.csv', index=False)

print("\nData berhasil diproses dan disimpan ke 'preprocessing_output.csv'.")
