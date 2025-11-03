# ==============================
# LATIHAN PREPROCESSING DATA
# Dataset: Titanic (dari seaborn)
# ==============================

# 1. Import semua library yang dibutuhkan
import pandas as pd
import numpy as np
import seaborn as sns

# Modul dari scikit-learn
from sklearn.impute import SimpleImputer              # untuk mengisi data hilang
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import os

# ==============================
# 2. LOAD DATASET
# ==============================
# Kita gunakan dataset Titanic yang sudah tersedia di seaborn
df = sns.load_dataset('titanic')

# Tampilkan 5 baris pertama
print("==== 5 Data Awal ====")
print(df.head())

# ==============================
# 3. CEK INFORMASI DATA
# ==============================
print("\n==== Informasi Data ====")
print(df.info())

print("\n==== Jumlah Missing Value per Kolom ====")
print(df.isnull().sum())

# ==============================
# 4. PILIH FITUR YANG DIPAKAI
# ==============================
# Kita hanya ambil beberapa kolom penting saja
cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

data = df[cols + [target]].copy()

print("\n==== Contoh Data Terpilih ====")
print(data.head())

# ==============================
# 5. PISAHKAN FITUR NUMERIK & KATEGORIKAL
# ==============================
num_features = ['age', 'sibsp', 'parch', 'fare']     # kolom numerik
cat_features = ['pclass', 'sex', 'embarked']         # kolom kategorikal

# ==============================
# 6. BUAT PIPELINE UNTUK TIAP JENIS FITUR
# ==============================

# Pipeline numerik: isi missing dengan median, lalu scaling
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline kategorikal: isi missing dengan modus, lalu one-hot encoding
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

])

# Gabungkan keduanya dalam ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# ==============================
# 7. PISAHKAN FITUR (X) DAN TARGET (y)
# ==============================
X = data.drop(columns=[target])
y = data[target]

# Hapus baris jika target kosong
mask = ~y.isnull()
X = X[mask]
y = y[mask]

# Split menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nJumlah data training:", X_train.shape[0])
print("Jumlah data testing:", X_test.shape[0])

# ==============================
# 8. FIT DAN TRANSFORM DATA
# ==============================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==============================
# 9. BUAT NAMA KOLOM BARU SETELAH ENCODING
# ==============================
# Ambil nama kolom numerik
num_col_names = num_features

# Ambil nama hasil one-hot encoding
ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_col_names = []
for idx, feature in enumerate(cat_features):
    categories = ohe.categories_[idx]
    cat_col_names += [f"{feature}_{cat}" for cat in categories]

# Gabungkan semua nama kolom
all_col_names = num_col_names + cat_col_names

# ==============================
# 10. KONVERSI KE DATAFRAME UNTUK MELIHAT HASIL
# ==============================
X_train_df = pd.DataFrame(X_train_processed, columns=all_col_names)
X_test_df = pd.DataFrame(X_test_processed, columns=all_col_names)

print("\n==== Data Hasil Preprocessing (Training) ====")
print(X_train_df.head())

# ==============================
# 11. SIMPAN HASILNYA KE FILE CSV
# ==============================
os.makedirs("processed", exist_ok=True)
X_train_df.to_csv("processed/X_train_processed.csv", index=False)
X_test_df.to_csv("processed/X_test_processed.csv", index=False)
y_train.to_csv("processed/y_train.csv", index=False)
y_test.to_csv("processed/y_test.csv", index=False)

print("\nFile hasil preprocessing sudah disimpan di folder 'processed/'")

# ==============================
# 12. CEK HASIL AKHIR
# ==============================
print("\nShape data training (setelah preprocessing):", X_train_df.shape)
print("Shape data testing (setelah preprocessing):", X_test_df.shape)