# src/clustering_eval.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering

from .models.spherical_kmeans import SphericalKMeans


def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size


def evaluate_clustering(embeddings, y_true, k, n_init=20, random_seed=42):
    """Spherical K-means와 Hierarchical clustering 기반 평가 (ACC, ARI, NMI)"""
    results = []

    # 입력 데이터 검증
    if embeddings is None:
        print("[Error] embeddings is None")
        return results
    if y_true is None:
        print("[Error] y_true is None")
        return results
    if k is None:
        print("[Error] k is None")
        return results
    if len(embeddings) != len(y_true):
        print(
            f"[Error] Length mismatch: embeddings({len(embeddings)}) != y_true({len(y_true)})"
        )
        return results
    if k <= 1:
        print(f"[Error] Invalid k value: {k}")
        return results

    # 데이터 형태 검증
    if not isinstance(embeddings, np.ndarray):
        print(
            f"[Warning] Converting embeddings to numpy array (type: {type(embeddings)})"
        )
        embeddings = np.array(embeddings)
    if not isinstance(y_true, np.ndarray):
        print(f"[Warning] Converting y_true to numpy array (type: {type(y_true)})")
        y_true = np.array(y_true)

    # Spherical K-means
    try:
        print(f"[Info] Running Spherical K-means with k={k}, n_init={n_init}")
        skm = SphericalKMeans(n_clusters=k, n_init=n_init, random_state=random_seed)
        y_pred_skm = skm.fit_predict(embeddings)

        if len(np.unique(y_pred_skm)) != k:
            print(
                f"[Warning] Spherical K-means produced {len(np.unique(y_pred_skm))} clusters instead of {k}"
            )

        acc_skm = cluster_acc(y_true, y_pred_skm)
        ari_skm = adjusted_rand_score(y_true, y_pred_skm)
        nmi_skm = normalized_mutual_info_score(y_true, y_pred_skm)

        results.append(
            {
                "method": "Spherical K-means",
                "k": k,
                "acc": acc_skm,
                "ari": ari_skm,
                "nmi": nmi_skm,
            }
        )
        print(
            f"[Info] Spherical K-means results - ACC: {acc_skm:.4f}, ARI: {ari_skm:.4f}, NMI: {nmi_skm:.4f}"
        )
    except Exception as e:
        print(f"[Error] SphericalKMeans clustering failed: {str(e)}")
        import traceback

        traceback.print_exc()

    # Hierarchical Clustering with cosine affinity and average linkage
    try:
        print(f"[Info] Running Hierarchical Clustering with k={k}")
        hc = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        y_pred_hc = hc.fit_predict(embeddings)

        if len(np.unique(y_pred_hc)) != k:
            print(
                f"[Warning] Hierarchical Clustering produced {len(np.unique(y_pred_hc))} clusters instead of {k}"
            )

        acc_hc = cluster_acc(y_true, y_pred_hc)
        ari_hc = adjusted_rand_score(y_true, y_pred_hc)
        nmi_hc = normalized_mutual_info_score(y_true, y_pred_hc)

        results.append(
            {
                "method": "Hierarchical Clustering",
                "k": k,
                "acc": acc_hc,
                "ari": ari_hc,
                "nmi": nmi_hc,
            }
        )
        print(
            f"[Info] Hierarchical Clustering results - ACC: {acc_hc:.4f}, ARI: {ari_hc:.4f}, NMI: {nmi_hc:.4f}"
        )
    except Exception as e:
        print(f"[Error] Hierarchical clustering failed: {str(e)}")
        import traceback

        traceback.print_exc()

    return results
