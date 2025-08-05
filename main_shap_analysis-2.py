import os, pathlib, pandas as pd, matplotlib.pyplot as plt, shap
from sklearn.tree import DecisionTreeRegressor

FEATS = ["Feature 1", "Feature 2"]
OUT_DIR = pathlib.Path("shap_summary_plots").resolve()
OUT_DIR.mkdir(exist_ok=True)
print("Image Output Catalogue：", OUT_DIR)


summary_data = {
    "Blobs_KM": ("clustered_blobs.csv", "KMeans_Label"),
    "Moons_DB": ("clustered_moons.csv", "DBSCAN_Label"),
    "Iris_KM":  ("clustered_iris.csv", "KMeans_Label"),
    "Cancer_KM":("clustered_cancer.csv", "KMeans_Label"),
}


for tag, (csv_path, label_col) in summary_data.items():
    print(f"\n Being processed：{tag}")
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        print("Missing label column, skip")
        continue

    X, y = df[FEATS].copy(), df[label_col].copy()
    if "DBSCAN" in label_col:
        m = y != -1
        X, y = X[m].reset_index(drop=True), y[m].reset_index(drop=True)

    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)


    plt.figure()
    shap.summary_plot(shap_values, features=X, feature_names=FEATS, show=False)
    out_path = OUT_DIR / f"{tag.lower()}_summary.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved：{out_path}")