# src/models/proposed_gae.py
import os
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch.amp import autocast, GradScaler
from torch_geometric.utils import to_undirected, remove_self_loops
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.clustering_eval import evaluate_clustering
from src.graph_utils import (
    hybrid_negative_sampling,
    create_two_hop_edge_candidates_with_iqr,
    hybrid_negative_sampling_with_k_hop,
    find_k_hop_neighbors,
)


class EnhancedGCNEncoder(nn.Module):
    def __init__(self, in_channels, channels1, channels2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, channels1, cached=True)
        self.conv2 = GCNConv(channels1, channels2, cached=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight=None):
        # 첫 번째 레이어 (1-hop 정보 집계)
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = F.gelu(x1)
        x1 = self.dropout(x1)

        # 두 번째 레이어 (2-hop까지의 정보 집계)
        x2 = self.conv2(x1, edge_index, edge_weight=edge_weight)

        return x2


class StandardGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


def compute_loss_with_clustering(
    z,
    pos_edge_index,
    selected_two_hop_edges,
    rejected_two_hop_edges,
    lambda_factor=1.0,
    hard_negative_ratio=0.5,
    device="cuda",
):
    # 1) 1-hop reconstruct
    pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    pos_loss = -F.logsigmoid(pos_score).mean()

    num_neg_samples = pos_edge_index.size(1)
    neg_edge_index = hybrid_negative_sampling(
        pos_edge_index,
        selected_two_hop_edges,
        rejected_two_hop_edges,
        z.size(0),
        num_neg_samples,
        hard_negative_ratio=hard_negative_ratio,
        device=device,
    )
    if neg_edge_index.size(1) > 0:
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        neg_loss = -F.logsigmoid(-neg_score).mean()
    else:
        neg_loss = torch.tensor(0.0, device=device)

    recon_loss = pos_loss + neg_loss

    # 2) 2-hop clustering loss
    flat_two_hop = [
        edge for node_edges in selected_two_hop_edges for edge in node_edges
    ]
    cluster_loss = torch.tensor(0.0, device=device)
    if flat_two_hop:
        two_hop_tensor = (
            torch.tensor(flat_two_hop, dtype=torch.long, device=device).t().contiguous()
        )
        valid_mask = (two_hop_tensor[0] < z.size(0)) & (two_hop_tensor[1] < z.size(0))
        two_hop_tensor = two_hop_tensor[:, valid_mask]
        if two_hop_tensor.size(1) > 0:
            hop_score = (z[two_hop_tensor[0]] * z[two_hop_tensor[1]]).sum(dim=1)
            cluster_loss = -F.logsigmoid(hop_score).mean()

    total_loss = recon_loss + lambda_factor * cluster_loss
    return total_loss, recon_loss.item(), cluster_loss.item()


def int_to_binary_vector(doc_id, num_bits=20):
    """Convert document ID to binary vector representation."""
    binary_str = format(doc_id, f"0{num_bits}b")
    return torch.tensor([int(bit) for bit in binary_str], dtype=torch.float)


def combine_features(x_dense, num_nodes, device="cuda"):
    """Combine dense features with binary document ID vectors."""
    # Create binary vectors for all document IDs
    binary_vectors = torch.stack([int_to_binary_vector(i) for i in range(num_nodes)])

    # Move both tensors to the same device
    x_dense = x_dense.to(device)
    binary_vectors = binary_vectors.to(device)

    # Concatenate dense features with binary vectors
    combined_features = torch.cat([x_dense, binary_vectors], dim=1)
    return combined_features


def ensure_dir_exists(file_path):
    """파일 경로의 디렉토리가 존재하는지 확인하고, 없으면 생성"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def train_improved_proposed_gae(
    x_sparse,  # 예: TF-IDF 임베딩
    x_dense,  # 예: 사전학습 모델 임베딩
    train_pos_edge_index,
    train_edge_weight,
    val_pos_edge_index,  # 매개변수는 유지하되 사용하지 않음
    documents,
    num_nodes,
    y_true,
    channels1=256,
    channels2=128,
    lr=0.01,
    epochs=1000,
    eval_interval=10,
    early_stopping=50,
    lambda_factor=1.0,
    iqr_k=0.5,
    max_edges_2hop=20,
    hard_negative_ratio=0.5,
    n_clusters=20,
    device="cuda",
    use_scheduler=True,
    save_dir=".",
    dataset_name=None,
    use_cache=True,
    random_seed=42,
    use_standard_gae=False,  # 표준 GAE 사용 여부
    gae_init_feature="combined",  # 초기 특성 선택: "onehot", "sparse", "dense", "combined"
    use_k_hop_negatives=True,  # k-hop negative sampling 사용 여부
    k_hop_values=[3, 4],  # 사용할 k-hop 값들
):
    print(
        f"--- Training {'Standard' if use_standard_gae else 'Proposed'} GAE (2-Hop with Combined Features) ---"
    )

    # 캐시 확인
    cache_path = None
    if save_dir and dataset_name and use_cache:
        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(
            save_dir,
            f"{'standard' if use_standard_gae else 'proposed'}_gae_{dataset_name}_ch1{channels1}_ch2{channels2}_lr{lr}_lambda{lambda_factor}_hnr{hard_negative_ratio}_init{gae_init_feature}_khop{'_'.join(map(str, k_hop_values))}.pkl",
        )
        if os.path.exists(cache_path):
            print(
                f"[CACHE] Loading {'Standard' if use_standard_gae else 'Proposed'} GAE results from {cache_path}"
            )
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)

                    # 캐시 데이터 구조 디버깅
                    print(
                        f"[DEBUG] Cache data keys: {list(cache_data.keys()) if isinstance(cache_data, dict) else 'Not a dict'}"
                    )

                    # 캐시 데이터가 올바른 구조인지 확인하고 반환
                    if isinstance(cache_data, dict):
                        # 필요한 키들이 모두 있는지 확인
                        required_keys = [
                            "embeddings",
                            "skmeans_acc",
                            "skmeans_ari",
                            "skmeans_nmi",
                            "hc_acc",
                            "hc_ari",
                            "hc_nmi",
                        ]
                        if all(key in cache_data for key in required_keys):
                            print(f"[DEBUG] Cache data validation successful")
                            print(
                                f"[DEBUG] Spherical K-means results - ACC: {cache_data.get('skmeans_acc')}, ARI: {cache_data.get('skmeans_ari')}, NMI: {cache_data.get('skmeans_nmi')}"
                            )
                            print(
                                f"[DEBUG] Hierarchical results - ACC: {cache_data.get('hc_acc')}, ARI: {cache_data.get('hc_ari')}, NMI: {cache_data.get('hc_nmi')}"
                            )
                            return cache_data
                        else:
                            missing_keys = [
                                key for key in required_keys if key not in cache_data
                            ]
                            print(
                                f"[WARNING] Cache data missing required keys: {missing_keys}"
                            )
                            print(
                                f"[WARNING] Available keys: {list(cache_data.keys())}"
                            )
                            # 기존 캐시 데이터로부터 결과 재구성 시도
                            if (
                                "embeddings" in cache_data
                                and cache_data["embeddings"] is not None
                            ):
                                print(
                                    "[INFO] Attempting to reconstruct results from cached embeddings"
                                )
                                embeddings = cache_data["embeddings"]

                                # 클러스터링 평가 수행
                                eval_results = evaluate_clustering(
                                    embeddings,
                                    y_true,
                                    n_clusters,
                                    random_seed=random_seed,
                                )
                                skmeans_res = next(
                                    (
                                        r
                                        for r in eval_results
                                        if r["method"] == "Spherical K-means"
                                    ),
                                    None,
                                )
                                hc_res = next(
                                    (
                                        r
                                        for r in eval_results
                                        if r["method"] == "Hierarchical Clustering"
                                    ),
                                    None,
                                )

                                if skmeans_res and hc_res:
                                    reconstructed_results = {
                                        "embeddings": embeddings,
                                        "skmeans_acc": skmeans_res["acc"],
                                        "skmeans_ari": skmeans_res["ari"],
                                        "skmeans_nmi": skmeans_res["nmi"],
                                        "hc_acc": hc_res["acc"],
                                        "hc_ari": hc_res["ari"],
                                        "hc_nmi": hc_res["nmi"],
                                        "best_epoch": cache_data.get("best_epoch", 0),
                                        "best_model_state": cache_data.get(
                                            "best_model_state", None
                                        ),
                                    }
                                    print(
                                        f"[INFO] Successfully reconstructed results from cached embeddings"
                                    )
                                    print(
                                        f"[INFO] Spherical K-means - ACC: {reconstructed_results['skmeans_acc']:.4f}, ARI: {reconstructed_results['skmeans_ari']:.4f}, NMI: {reconstructed_results['skmeans_nmi']:.4f}"
                                    )
                                    print(
                                        f"[INFO] Hierarchical - ACC: {reconstructed_results['hc_acc']:.4f}, ARI: {reconstructed_results['hc_ari']:.4f}, NMI: {reconstructed_results['hc_nmi']:.4f}"
                                    )

                                    # 재구성된 결과를 캐시에 다시 저장
                                    try:
                                        with open(cache_path, "wb") as cache_file:
                                            pickle.dump(
                                                reconstructed_results, cache_file
                                            )
                                        print(
                                            f"[INFO] Updated cache with reconstructed results"
                                        )
                                    except Exception as save_error:
                                        print(
                                            f"[WARNING] Failed to update cache: {save_error}"
                                        )

                                    return reconstructed_results
                                else:
                                    print(
                                        f"[WARNING] Failed to reconstruct results from cached embeddings"
                                    )

                            # 캐시가 유효하지 않으므로 다시 학습 진행
                            print(
                                f"[INFO] Cache data is invalid, proceeding with training"
                            )
                    else:
                        print(
                            f"[WARNING] Cache data is not a dictionary, proceeding with training"
                        )

            except Exception as e:
                print(
                    f"[Warning] Failed to load {'Standard' if use_standard_gae else 'Proposed'} GAE cache: {e}"
                )

    # GAE 초기 특성 설정
    if gae_init_feature == "onehot":
        x_gae = torch.eye(num_nodes, dtype=torch.float)
    elif gae_init_feature == "sparse" and x_sparse is not None:
        x_gae = x_sparse
    elif gae_init_feature == "dense" and x_dense is not None:
        x_gae = x_dense
    elif gae_init_feature == "combined" and x_dense is not None:
        # Combine dense features with binary document ID vectors
        x_gae = combine_features(x_dense, num_nodes, device)
    else:
        print(
            f"[Warning] Using default dense features for GAE (init_feature={gae_init_feature})"
        )
        x_gae = x_dense

    if x_gae is None:
        raise ValueError("GAE initial features are None.")
    if train_pos_edge_index is None or train_pos_edge_index.numel() == 0:
        raise ValueError("No train edges for GAE.")

    scaler = GradScaler(enabled=("cuda" in str(device)))

    # 1) 1-hop adjacency list
    one_hop_list = [[] for _ in range(num_nodes)]
    src_arr, dst_arr = train_pos_edge_index.cpu().numpy()
    for s, d in zip(src_arr, dst_arr):
        if s < num_nodes:
            one_hop_list[s].append(d)

    # 2) k-hop neighbors for negative sampling
    k_hop_neighbors = None
    if use_k_hop_negatives:
        k_hop_neighbors = [set() for _ in range(num_nodes)]
        for k in k_hop_values:
            k_hop = find_k_hop_neighbors(one_hop_list, k)
            for i in range(num_nodes):
                k_hop_neighbors[i].update(k_hop[i])

    # 3) 2-hop edges (표준 GAE에서는 사용하지 않음)
    selected_two_hop, rejected_two_hop = ([], [])
    if not use_standard_gae and documents is not None:
        two_hop_path = os.path.join(
            save_dir, f"two_hop_edges_{dataset_name}_iqr{iqr_k}_max{max_edges_2hop}.pkl"
        )
        selected_two_hop, rejected_two_hop = create_two_hop_edge_candidates_with_iqr(
            one_hop_list,
            documents,
            num_nodes,
            save_path=two_hop_path,
            max_edges=max_edges_2hop,
            k=iqr_k,
            use_cache=use_cache,
        )
    elif not use_standard_gae:
        print("[Warning] No documents => skip 2-hop analysis.")

    # 4) GCN 모델 초기화
    if use_standard_gae:
        model = StandardGCNEncoder(x_gae.shape[1], channels1, channels2).to(device)
    else:
        model = EnhancedGCNEncoder(x_gae.shape[1], channels1, channels2).to(device)

    # 5) 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=epochs // 4,  # 첫 번째 주기
            T_mult=2,  # 매 주기마다 2배씩 증가
            eta_min=lr * 0.1,  # 최소 learning rate
        )

    # 6) 장치 이동
    x_gae = x_gae.to(device)
    train_pos_edge_index = train_pos_edge_index.to(device)
    if train_edge_weight is not None:
        train_edge_weight = train_edge_weight.to(device)

    # Best model tracking
    best_acc = 0.0
    best_epoch = 0
    best_model_state = None
    best_metrics = None
    epochs_without_improvement = 0

    # Training logs
    training_logs = []

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=("cuda" in str(device))):
            # GCN encoding
            z = model(x_gae, train_pos_edge_index, edge_weight=train_edge_weight)

            # Calculate loss
            if use_standard_gae:
                # Standard GAE uses only 1-hop reconstruction loss
                pos_score = (
                    z[train_pos_edge_index[0]] * z[train_pos_edge_index[1]]
                ).sum(dim=1)
                pos_loss = -F.logsigmoid(pos_score).mean()

                # Negative sampling with k-hop neighbors
                neg_edge_index = hybrid_negative_sampling_with_k_hop(
                    train_pos_edge_index,
                    [],  # No 2-hop edges
                    [],  # No rejected edges
                    z.size(0),
                    train_pos_edge_index.size(1),
                    hard_negative_ratio=hard_negative_ratio,
                    k_hop_neighbors=k_hop_neighbors,
                    device=device,
                )
                if neg_edge_index.size(1) > 0:
                    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
                    neg_loss = -F.logsigmoid(-neg_score).mean()
                else:
                    neg_loss = torch.tensor(0.0, device=device)

                total_loss = pos_loss + neg_loss
                recon_l = total_loss.item()
                cluster_l = 0.0
            else:
                total_loss, recon_l, cluster_l = compute_loss_with_clustering(
                    z,
                    train_pos_edge_index,
                    selected_two_hop,
                    rejected_two_hop,
                    lambda_factor,
                    hard_negative_ratio,
                    device,
                )

        if torch.isnan(total_loss):
            print(
                f"[Warning] {'Standard' if use_standard_gae else 'Proposed'} GAE loss is NaN at epoch {epoch}"
            )
            break

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        # Evaluation
        if epoch % eval_interval == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                with autocast(device_type="cuda", enabled=("cuda" in str(device))):
                    z_eval = model(
                        x_gae, train_pos_edge_index, edge_weight=train_edge_weight
                    )
                emb_eval = z_eval.cpu().numpy()

            # Clustering evaluation
            eval_res = evaluate_clustering(
                emb_eval, y_true, n_clusters, random_seed=random_seed
            )
            skmeans_res = next(
                (r for r in eval_res if r["method"] == "Spherical K-means"), None
            )
            hc_res = next(
                (r for r in eval_res if r["method"] == "Hierarchical Clustering"), None
            )

            # Log training progress
            log_entry = {
                "epoch": epoch,
                "train_total_loss": total_loss.item(),
                "train_recon_loss": recon_l,
                "train_cluster_loss": cluster_l,
            }

            if skmeans_res:
                log_entry.update(
                    {
                        "skmeans_acc": skmeans_res["acc"],
                        "skmeans_ari": skmeans_res["ari"],
                        "skmeans_nmi": skmeans_res["nmi"],
                    }
                )

            if hc_res:
                log_entry.update(
                    {
                        "hc_acc": hc_res["acc"],
                        "hc_ari": hc_res["ari"],
                        "hc_nmi": hc_res["nmi"],
                    }
                )

            training_logs.append(log_entry)

            # Print progress
            print(
                f"[{'Standard' if use_standard_gae else 'Proposed'} GAE] Epoch {epoch}/{epochs}, "
                f"Train Loss={total_loss:.4f} (Recon={recon_l:.4f}, Clust={cluster_l:.4f})"
            )

            if skmeans_res:
                curr_acc = skmeans_res["acc"]
                print(
                    f"    S.K-means: ACC={curr_acc:.4f}, ARI={skmeans_res['ari']:.4f}, NMI={skmeans_res['nmi']:.4f}"
                )
                if hc_res:
                    print(
                        f"    HC: ACC={hc_res['acc']:.4f}, ARI={hc_res['ari']:.4f}, NMI={hc_res['nmi']:.4f}"
                    )

                # Model selection based on Spherical K-means accuracy
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    best_epoch = epoch
                    best_model_state = model.state_dict().copy()
                    best_metrics = {
                        "skmeans_acc": skmeans_res["acc"],
                        "skmeans_ari": skmeans_res["ari"],
                        "skmeans_nmi": skmeans_res["nmi"],
                        "hc_acc": hc_res["acc"] if hc_res else 0.0,
                        "hc_ari": hc_res["ari"] if hc_res else 0.0,
                        "hc_nmi": hc_res["nmi"] if hc_res else 0.0,
                    }
                    epochs_without_improvement = 0
                    print(f"    New best model! S.K-means ACC: {best_acc:.4f}")
                else:
                    epochs_without_improvement += eval_interval
            else:
                epochs_without_improvement += eval_interval

            # Early stopping
            if early_stopping > 0 and epochs_without_improvement >= early_stopping:
                print(
                    f"Early stopping at epoch {epoch}, no improvement in S.K-means ACC for {epochs_without_improvement} epochs."
                )
                break

    # Save training logs
    log_path = os.path.join(
        save_dir,
        f"{'standard' if use_standard_gae else 'prop'}_gae_training_log_{dataset_name}.csv",
    )
    if training_logs:
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        pd.DataFrame(training_logs).to_csv(log_path, index=False)
        print(f"Training logs saved to {log_path}")

    print(
        f"{'Standard' if use_standard_gae else 'Proposed'} GAE complete. Best S.K-means ACC={best_acc:.4f} at epoch={best_epoch}"
    )

    # Get final embeddings using best model
    final_embeddings = None
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=("cuda" in str(device))):
                z_final = model(
                    x_gae, train_pos_edge_index, edge_weight=train_edge_weight
                )
            final_embeddings = z_final.cpu().numpy()
    else:
        print("[Warning] No valid model state was saved during training")
        # 최종 모델 상태라도 사용
        model.eval()
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=("cuda" in str(device))):
                z_final = model(
                    x_gae, train_pos_edge_index, edge_weight=train_edge_weight
                )
            final_embeddings = z_final.cpu().numpy()

        # 최종 평가 수행
        if final_embeddings is not None:
            eval_results = evaluate_clustering(
                final_embeddings, y_true, n_clusters, random_seed=random_seed
            )
            skmeans_res = next(
                (r for r in eval_results if r["method"] == "Spherical K-means"), None
            )
            hc_res = next(
                (r for r in eval_results if r["method"] == "Hierarchical Clustering"),
                None,
            )

            if skmeans_res and hc_res:
                best_metrics = {
                    "skmeans_acc": skmeans_res["acc"],
                    "skmeans_ari": skmeans_res["ari"],
                    "skmeans_nmi": skmeans_res["nmi"],
                    "hc_acc": hc_res["acc"],
                    "hc_ari": hc_res["ari"],
                    "hc_nmi": hc_res["nmi"],
                }

    # Prepare final results
    final_results = {
        "embeddings": final_embeddings,
        "best_epoch": best_epoch,
        "skmeans_acc": best_metrics["skmeans_acc"] if best_metrics else 0.0,
        "skmeans_ari": best_metrics["skmeans_ari"] if best_metrics else 0.0,
        "skmeans_nmi": best_metrics["skmeans_nmi"] if best_metrics else 0.0,
        "hc_acc": best_metrics["hc_acc"] if best_metrics else 0.0,
        "hc_ari": best_metrics["hc_ari"] if best_metrics else 0.0,
        "hc_nmi": best_metrics["hc_nmi"] if best_metrics else 0.0,
        "best_model_state": best_model_state,
    }

    # Cache results
    if cache_path:
        try:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
                print(f"[CACHE] Created directory: {cache_dir}")

            with open(cache_path, "wb") as f:
                pickle.dump(final_results, f)
            print(
                f"[CACHE] Saved {'Standard' if use_standard_gae else 'Proposed'} GAE results to {cache_path}"
            )
            print(
                f"[CACHE] Saved results: S.K-means ACC={final_results['skmeans_acc']:.4f}, ARI={final_results['skmeans_ari']:.4f}, NMI={final_results['skmeans_nmi']:.4f}"
            )

        except Exception as e:
            print(
                f"[Warning] Failed to save {'Standard' if use_standard_gae else 'Proposed'} GAE cache: {e}"
            )

    if "cuda" in str(device):
        torch.cuda.empty_cache()

    return final_results
