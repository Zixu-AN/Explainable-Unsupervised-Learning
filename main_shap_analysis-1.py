
import os, pathlib, pandas as pd, matplotlib.pyplot as plt, shap
from sklearn.tree import DecisionTreeRegressor

print("：Current working directory", os.getcwd())

DATA = {
    "Moons_KM":  ("clustered_moons.csv" ,  1 , "KMeans_Label"),
    "Moons_DB":  ("clustered_moons.csv" ,  5 , "DBSCAN_Label"),
    "Blobs_KM":  ("clustered_blobs.csv" , 10 , "KMeans_Label"),
    "Blobs_DB":  ("clustered_blobs.csv" ,  5 , "DBSCAN_Label"),
    "Iris_KM":   ("clustered_iris.csv"  , 14 , "KMeans_Label"),
    "Cancer_KM": ("clustered_cancer.csv", 50 , "KMeans_Label"),
}
FEATS   = ["Feature 1", "Feature 2"]
OUT_DIR = pathlib.Path("shap_local_plots").resolve()
OUT_DIR.mkdir(exist_ok=True)
print("Image Output Catalog：", OUT_DIR)

for tag, (csv_path, idx, label_col) in DATA.items():
    print(f"\n deal with {tag}  sample{idx}")
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        print("Missing columns, skip");  continue

    X, y = df[FEATS].copy(), df[label_col].copy()
    if "DBSCAN" in label_col:
        m = y != -1
        X, y = X[m].reset_index(drop=True), y[m].reset_index(drop=True)

    if idx >= len(X): idx = 0
    sample = X.iloc[idx]

    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)      # (n_samples, n_features)
    sv, bv      = shap_values[idx], explainer.expected_value


    ret = shap.plots.waterfall(
        shap.Explanation(values=sv,
                         base_values=bv,
                         data=sample.values,
                         feature_names=sample.index.tolist()),
        show=False)


    fig = ret.figure if hasattr(ret, "figure") else plt.gcf()

    out_path = OUT_DIR / f"{tag.lower()}_{idx}_waterfall.png"
    fig.savefig(out_path, bbox_inches="tight");  plt.close(fig)

    assert out_path.exists(), "PNG not written, check path/permissions!"
    print(f"Saved{out_path}")