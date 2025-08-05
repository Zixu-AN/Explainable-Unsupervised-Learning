import os
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics.pairwise import cosine_similarity


FEATS = ["Feature 1", "Feature 2"]

DATA_CONFIG = {
    "Moons_DB": ("clustered_moons.csv", "DBSCAN_Label"),
    "Iris_KM": ("clustered_iris.csv", "KMeans_Label"),
}

OUT_DIR_CSV = pathlib.Path("shap_consistency_csv")
OUT_DIR_PLOT = pathlib.Path("shap_violin_consistency")
OUT_DIR_CSV.mkdir(exist_ok=True)
OUT_DIR_PLOT.mkdir(exist_ok=True)



def save_shap_cluster_csv(shap_values, cluster_labels, out_csv):
    df = pd.DataFrame()
    df['cluster'] = cluster_labels.values
    df['Feature 1 SHAP'] = shap_values[:, 0]
    df['Feature 2 SHAP'] = shap_values[:, 1]
    df.to_csv(out_csv, index=False)
    print(f"Saved SHAP CSV to {out_csv}")



def cluster_shap_consistency(shap_matrix):

    n_samples = shap_matrix.shape[0]
    if n_samples <= 1:
        return 1.0

    sim_matrix = cosine_similarity(shap_matrix)
    idx_upper = np.triu_indices(n_samples, k=1)
    sim_values = sim_matrix[idx_upper]

    if len(sim_values) == 0:
        return 1.0

    variance = np.var(sim_values)

    # 使用平滑函数归一化（避免极端值）
    norm_score = 1 / (1 + variance * 10)
    return round(norm_score, 4)



def plot_shap_consistency_violin(df, cluster_col, save_name, save_dir=OUT_DIR_PLOT):
    os.makedirs(save_dir, exist_ok=True)
    df_long = df.melt(
        id_vars=[cluster_col],
        value_vars=['Feature 1 SHAP', 'Feature 2 SHAP'],
        var_name='Feature',
        value_name='SHAP Value'
    )
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_long, x='Feature', y='SHAP Value', hue=cluster_col, split=True)
    plt.title(f'SHAP Consistency Within Clusters - {save_name}')
    plt.xlabel('Feature')
    plt.ylabel('SHAP Value')
    plt.legend(title='Cluster')
    plt.tight_layout()
    filepath = save_dir / f"{save_name.replace(' ', '_')}_consistency.png"
    plt.savefig(filepath)
    plt.close()
    print(f"Saved violin plot: {filepath}")



def plot_consistency_scores(consistency_scores, tag, save_dir=OUT_DIR_PLOT):
    os.makedirs(save_dir, exist_ok=True)
    clusters = list(consistency_scores.keys())
    scores = [consistency_scores[c] for c in clusters]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(clusters)), scores, color='skyblue')
    plt.ylim(0, 1.05)
    plt.xlabel('Cluster')
    plt.ylabel('SHAP Consistency Score')
    plt.title(f'SHAP Consistency Scores per Cluster - {tag}')
    plt.xticks(range(len(clusters)), labels=[f'Cluster {c}' for c in clusters])

    for i, v in enumerate(scores):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')

    plt.tight_layout()
    filepath = save_dir / f"{tag.replace(' ', '_')}_consistency_scores.png"
    plt.savefig(filepath)
    plt.close()
    print(f"Saved consistency scores bar plot: {filepath}")



for tag, (csv_path, label_col) in DATA_CONFIG.items():
    print(f"\nProcessing {tag}...")


    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        print(f"Label column {label_col} missing, skip {tag}")
        continue


    if "DBSCAN" in label_col:
        df = df[df[label_col] != -1].reset_index(drop=True)

    X = df[FEATS]
    y = df[label_col]


    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    model.fit(X, y)


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    out_csv = OUT_DIR_CSV / f"{tag.lower()}_shap.csv"
    save_shap_cluster_csv(shap_values, y, out_csv)


    df_shap = pd.read_csv(out_csv)

    print(f"\n Calculating SHAP consistency scores for {tag}...")
    grouped = df_shap.groupby('cluster')
    consistency_scores = {}

    for cluster_id, group in grouped:
        shap_matrix = group[['Feature 1 SHAP', 'Feature 2 SHAP']].values
        score = cluster_shap_consistency(shap_matrix)
        consistency_scores[cluster_id] = score

    print(f"\n SHAP consistency scores for {tag}:")
    for cid, score in consistency_scores.items():
        print(f"  Cluster {cid}: {score:.4f}")


    plot_shap_consistency_violin(df_shap, 'cluster', tag)

    plot_consistency_scores(consistency_scores, tag)

print("\nAll done!")
