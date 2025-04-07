from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples, flagc):
    if flagc == 1:
        random_state = 365
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    elif flagc == 2:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples,
                                    centers=4,
                                    cluster_std=[1.0, 2.5, 0.5, 3.0],
                                    random_state=random_state)
    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    else:
        X = []
    return X

# Parametri
n_samples = 500  # broj uzoraka
flagc = 1  # Odabrani način generiranja podataka (1-5)

# Generiraj podatke
X = generate_data(n_samples, flagc)

# Primjena K-means algoritma
kmeans = KMeans(n_clusters=3, random_state=42)  # Odabrano 3 klastera
y_kmeans = kmeans.fit_predict(X)

# Prikaz rezultata
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', marker='o', s=50, edgecolors='k', alpha=0.6)

# Prikaz centara klastera
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroidi')
plt.title(f"Podaci i centri klastera (Metoda {flagc})")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
