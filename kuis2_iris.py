# ===== Cell 1: Imports =====
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib  # untuk menyimpan model

# ===== Cell 2: Load dataset (sklearn) =====
iris = load_iris()
X = iris.data            # fitur (4 kolom)
y = iris.target          # label (0,1,2)
feature_names = iris.feature_names
class_names = iris.target_names
print("Features:", feature_names)
print("Classes:", class_names)

# ===== Cell 3: Buat DataFrame (opsional, untuk inspect) =====
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [class_names[i] for i in y]
print(df.head())

# ===== Cell 4: Split train/test =====
# gunakan stratify supaya distribusi kelas seimbang
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ===== Cell 5: Definisikan model dan train =====
# coba criterion='entropy' supaya sesuai materi Decision Tree C4.5
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("Model terlatih.")

# ===== Cell 6: Prediksi & evaluasi =====
y_pred = model.predict(X_test)
print("Akurasi:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===== Cell 7: Visualisasi pohon keputusan =====
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
plt.title("Decision Tree - Iris (criterion='entropy', max_depth=3)")
plt.savefig("decision_tree_iris.png")
plt.show()

# ===== Cell 8: Simpan model (opsional) =====
joblib.dump(model, "decision_tree_iris.joblib")
print("Model disimpan di decision_tree_iris.joblib")

# ===== Cell 9: Contoh prediksi data baru =====
sample = np.array([[5.0, 3.4, 1.5, 0.2]])  # ubah sesuai mau test
pred = model.predict(sample)
print("Prediksi untuk sample:", sample, "->", class_names[pred[0]])
