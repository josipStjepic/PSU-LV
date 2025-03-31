import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# 1. Izgradnja modela Stablo odlučivanja s maksimalnom dubinom 5
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)

# Predikcija na testnom skupu
y_pred = dt.predict(X_test_scaled)

# a) Vizualizacija stabla odlučivanja
plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True, feature_names=data.feature_names, class_names=data.target_names, rounded=True, fontsize=10)
plt.title('Vizualizacija stabla odlučivanja')
plt.show()

# b) Evaluacija modela
# i) Matrica zabune
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Matrica zabune')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ii) Točnost klasifikacije
accuracy = accuracy_score(y_test, y_pred)
print(f"Točnost klasifikacije: {accuracy:.4f}")

# iii) Preciznost i odziv
report = classification_report(y_test, y_pred, target_names=data.target_names)
print("Preciznost i odziv:\n", report)

# 2. Evaluacija s različitim vrijednostima max_depth
max_depth_values = [1, 3, 5, 7, 10]
print("\nEvaluacija s različitim max_depth vrijednostima:")
for depth in max_depth_values:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train_scaled, y_train)
    y_pred = dt.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Točnost za max_depth={depth}: {accuracy:.4f}")

# 3. Evaluacija bez skaliranja podataka
dt_no_scaling = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_no_scaling.fit(X_train, y_train)
y_pred_no_scaling = dt_no_scaling.predict(X_test)

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
