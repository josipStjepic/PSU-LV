import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Učitavanje Iris skupa podataka
data = load_iris()
X = data.data  # Ulazne značajke
y = data.target  # Ciljna varijabla

# Podjela podataka na skup za učenje (80%) i testiranje (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Skaliranje podataka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Izgradnja modela KNN sa 5 susjeda
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predikcija na testnom skupu
y_pred = knn.predict(X_test_scaled)

# a) Matrica zabune
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Matrica zabune')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# b) Točnost klasifikacije
accuracy = accuracy_score(y_test, y_pred)
print(f"Točnost klasifikacije: {accuracy:.4f}")

# c) Preciznost i odziv
report = classification_report(y_test, y_pred, target_names=data.target_names)
print("Preciznost i odziv:\n", report)

# 5. Promjena broja susjeda
k_values = [1, 3, 5, 7, 9]
print("\nEvaluacija s različitim brojevima susjeda:")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Točnost za {k} susjeda: {accuracy:.4f}")

# 6. Evaluacija bez skaliranja podataka
knn_no_scaling = KNeighborsClassifier(n_neighbors=5)
knn_no_scaling.fit(X_train, y_train)
y_pred_no_scaling = knn_no_scaling.predict(X_test)

# Matrica zabune bez skaliranja
conf_matrix_no_scaling = confusion_matrix(y_test, y_pred_no_scaling)
sns.heatmap(conf_matrix_no_scaling, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Matrica zabune bez skaliranja')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Točnost bez skaliranja
accuracy_no_scaling = accuracy_score(y_test, y_pred_no_scaling)
print(f"Točnost klasifikacije bez skaliranja: {accuracy_no_scaling:.4f}")
