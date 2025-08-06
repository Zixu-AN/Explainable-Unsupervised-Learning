import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler

X_moons, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
df_moons = pd.DataFrame(X_moons, columns=["Feature 1", "Feature 2"])
df_moons["Dataset"] = "Moons"
df_moons["ID"] = range(len(df_moons))
df_moons.to_csv("data_moons.csv", index=False)

# Blobs
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=[0.5, 1.0, 1.5], random_state=42)
df_blobs = pd.DataFrame(X_blobs, columns=["Feature 1", "Feature 2"])
df_blobs["Dataset"] = "Blobs"
df_blobs["ID"] = range(len(df_blobs))
df_blobs.to_csv("data_blobs.csv", index=False)


iris = load_iris()
X_iris = iris.data[:, :2]
df_iris = pd.DataFrame(X_iris, columns=["Feature 1", "Feature 2"])
df_iris["Dataset"] = "Iris"
df_iris["ID"] = range(len(df_iris))
df_iris.to_csv("data_iris.csv", index=False)


cancer = load_breast_cancer()
X_cancer_raw = cancer.data[:, :2]
scaler = StandardScaler()
X_cancer = scaler.fit_transform(X_cancer_raw)
df_cancer = pd.DataFrame(X_cancer, columns=["Feature 1", "Feature 2"])
df_cancer["Dataset"] = "Cancer"
df_cancer["ID"] = range(len(df_cancer))
df_cancer.to_csv("data_cancer.csv", index=False)


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
datasets = {
    "Moons": df_moons,
    "Blobs": df_blobs,
    "Iris": df_iris,
    "Cancer": df_cancer
}

for ax, (name, df) in zip(axes.flatten(), datasets.items()):
    ax.scatter(df["Feature 1"], df["Feature 2"], alpha=0.6)
    ax.set_title(name)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True)

plt.tight_layout()
plt.savefig("all_datasets_visual.png")
plt.show()

