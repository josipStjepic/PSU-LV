import numpy as np
import matplotlib.pyplot as plt
import os

if os.path.exists("mtcars.csv"):
    data = np.genfromtxt("mtcars.csv", delimiter=",", usecols=(1, 3, 6, 2), skip_header=1)

    mpg = data[:, 0]
    hp = data[:, 1]
    wt = data[:, 2]
    cyl = data[:, 3]

    plt.figure(figsize=(8, 6))
    plt.scatter(hp, mpg, s=wt*50, c='blue', alpha=0.5, edgecolors="black")

    plt.xlabel("Konjske snage (hp)")
    plt.ylabel("Potrošnja goriva (mpg)")
    plt.title("Ovisnost potrošnje goriva o konjskim snagama")
    plt.grid(True)

    plt.savefig('grafikon.png')
    plt.show()

    print("Minimalna potrošnja goriva (mpg):", np.min(mpg))
    print("Maksimalna potrošnja goriva (mpg):", np.max(mpg))
    print("Srednja potrošnja goriva (mpg):", np.mean(mpg))

    mpg_6cyl = mpg[cyl == 6]
    print("Minimalna potrošnja (mpg) za 6 cilindara:", np.min(mpg_6cyl))
    print("Maksimalna potrošnja (mpg) za 6 cilindara:", np.max(mpg_6cyl))
    print("Srednja potrošnja (mpg) za 6 cilindara:", np.mean(mpg_6cyl))

else:
    print("Fajl 'mtcars.csv' nije pronađen. Proverite putanju.")