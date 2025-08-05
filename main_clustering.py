import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN


datasets = {
    "Moons": pd.read_csv("data_moons.csv"),
    "Blobs": pd.read_csv("data_blobs.csv"),
    "Iris": pd.read_csv("data_iris.csv"),
    "Cancer": pd.read_csv("data_cancer.csv")
}

def run_clustering(df, dataset_name, kmeans_k=3, dbscan_eps=0.5, dbscan_min_samples=5):
    X = df[["Feature 1", "Feature 2"]].values


    km = KMeans(n_clusters=kmeans_k, random_state=0)
    labels_km = km.fit_predict(X)
    df["KMeans_Label"] = labels_km


    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_db = db.fit_predict(X)
    df["DBSCAN_Label"] = labels_db


    output_csv = f"clustered_{dataset_name.lower()}.csv"
    df.to_csv(output_csv, index=False)

    print(f"\n✅ {dataset_name} Clustering complete")
    print(f"- The clustering results are saved as：{output_csv}")

    return df, X, labels_km, labels_db


results = {}
results["Moons"] = run_clustering(datasets["Moons"], "Moons", dbscan_eps=0.2)
results["Blobs"] = run_clustering(datasets["Blobs"], "Blobs", dbscan_eps=1.2)
results["Iris"] = run_clustering(datasets["Iris"], "Iris", dbscan_eps=0.3)
results["Cancer"] = run_clustering(datasets["Cancer"], "Cancer", dbscan_eps=0.4)


synthetic = ["Moons", "Blobs"]
real = ["Iris", "Cancer"]


fig_km_synth, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axes, synthetic):
    _, X, labels_km, _ = results[name]
    ax.scatter(X[:, 0], X[:, 1], c=labels_km, cmap='tab10', s=30)
    ax.set_title(f"{name} - KMeans")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True)
plt.tight_layout()
plt.savefig("kmeans_synthetic.png")
plt.show()


fig_km_real, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axes, real):
    _, X, labels_km, _ = results[name]
    ax.scatter(X[:, 0], X[:, 1], c=labels_km, cmap='tab10', s=30)
    ax.set_title(f"{name} - KMeans")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True)
plt.tight_layout()
plt.savefig("kmeans_real.png")
plt.show()


fig_db_synth, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axes, synthetic):
    _, X, _, labels_db = results[name]
    ax.scatter(X[:, 0], X[:, 1], c=labels_db, cmap='tab10', s=30)
    ax.set_title(f"{name} - DBSCAN")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True)
plt.tight_layout()
plt.savefig("dbscan_synthetic.png")
plt.show()


fig_db_real, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axes, real):
    _, X, _, labels_db = results[name]
    ax.scatter(X[:, 0], X[:, 1], c=labels_db, cmap='tab10', s=30)
    ax.set_title(f"{name} - DBSCAN")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True)
plt.tight_layout()
plt.savefig("dbscan_real.png")
plt.show()

print("- kmeans_synthetic.png")
print("- kmeans_real.png")
print("- dbscan_synthetic.png")
print("- dbscan_real.png")
