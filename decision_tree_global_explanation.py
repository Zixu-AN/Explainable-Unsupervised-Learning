import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os


datasets = [
    ("clustered_moons.csv", "Moons"),
    ("clustered_blobs.csv", "Blobs"),
    ("clustered_iris.csv", "Iris"),
    ("clustered_cancer.csv", "Cancer")
]
labels = ["KMeans_Label", "DBSCAN_Label"]


output_dir = "decision_tree_plots"
os.makedirs(output_dir, exist_ok=True)


for file_name, dataset_name in datasets:
    df = pd.read_csv(file_name)

    for label_col in labels:
        print(f"\nDatasets：{dataset_name} | label：{label_col}")


        if label_col == "DBSCAN_Label":
            df_filtered = df[df[label_col] != -1].reset_index(drop=True)
        else:
            df_filtered = df.reset_index(drop=True)


        X = df_filtered[["Feature 1", "Feature 2"]]
        y = df_filtered[label_col]


        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X, y)

        plt.figure(figsize=(10, 6))
        plot_tree(
            clf,
            feature_names=["Feature 1", "Feature 2"],
            class_names=[str(cls) for cls in sorted(y.unique())],
            filled=True,
            rounded=True
        )
        plt.title(f"Decision Tree: {dataset_name} - {label_col}", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{dataset_name}_{label_col}_tree.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f" The image has been saved to：{save_path}")
