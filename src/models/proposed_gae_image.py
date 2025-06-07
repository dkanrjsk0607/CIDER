# src/models/proposed_gae_image.py
# This file extends the proposed_gae.py to better handle image datasets

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
    hybrid_negative_sampling_with_k_hop,
    find_k_hop_neighbors,
)


class EnhancedGCNEncoder(nn.Module):
    def __init__(self, in_channels, channels1, channels2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, channels1, cached=True)
        self.conv2 = GCNConv(channels1, channels2, cached=True)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(channels1)
        self.batch_norm2 = nn.BatchNorm1d(channels2)

    def forward(self, x, edge_index, edge_weight=None):
        # 첫 번째 레이어 (1-hop 정보 집계)
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = self.batch_norm1(x1)
        x1 = F.gelu(x1)
        x1 = self.dropout(x1)

        # 두 번째 레이어 (2-hop까지의 정보 집계)
        x2 = self.conv2(x1, edge_index, edge_weight=edge_weight)
        x2 = self.batch_norm2(x2)

        return x2


class ImageGAEEncoder(nn.Module):
    """Enhanced encoder specifically designed for image datasets"""

    def __init__(self, in_channels, channels1, channels2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, channels1, cached=True)
        self.conv2 = GCNConv(channels1, channels2, cached=True)
        self.dropout = nn.Dropout(0.3)  # Slightly higher dropout for image data
        self.batch_norm1 = nn.BatchNorm1d(channels1)
        self.batch_norm2 = nn.BatchNorm1d(channels2)

        # Skip connection
        self.skip_proj = (
            nn.Linear(in_channels, channels2)
            if in_channels != channels2
            else nn.Identity()
        )

    def forward(self, x, edge_index, edge_weight=None):
        # Skip connection
        skip = self.skip_proj(x)

        # First layer
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = self.batch_norm1(x1)
        x1 = F.gelu(x1)
        x1 = self.dropout(x1)

        # Second layer
        x2 = self.conv2(x1, edge_index, edge_weight=edge_weight)
        x2 = self.batch_norm2(x2)

        # Add skip connection
        x2 = x2 + skip

        return x2


def compute_loss_for_image_data(
    z,
    pos_edge_index,
    k_hop_neighbors=None,
    lambda_factor=1.0,
    hard_negative_ratio=0.5,
    device="cuda",
):
    """Compute loss for image datasets using k-hop neighbors for negative sampling"""
    # 1) 1-hop reconstruction loss
    pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    pos_loss = -F.logsigmoid(pos_score).mean()

    # Negative sample 수를 positive sample 수와 동일하게 설정
    num_neg_samples = pos_edge_index.size(1)

    # large graph에서도 충분한 negative sample을 확보하기 위한 코드 -> 우선은 사용 안함.
    max_neg_samples = 100000  # 최대 negative sample 수를 늘림
    num_neg_samples = min(num_neg_samples, max_neg_samples)

    # print(f"Using {num_neg_samples} negative samples (from {pos_edge_index.size(1)} positive samples)")

    # hard_negative_ratio가 0이거나 k_hop_neighbors가 None이면 일반 negative sampling 사용
    if hard_negative_ratio <= 0 or k_hop_neighbors is None:
        # Regular negative sampling
        # print("[Info] Using standard random negative sampling")
        neg_edge_index = torch.stack(
            [
                torch.randint(0, z.size(0), (num_neg_samples,), device=device),
                torch.randint(0, z.size(0), (num_neg_samples,), device=device),
            ],
            dim=0,
        )
        # Self-loops 제거
        mask = neg_edge_index[0] != neg_edge_index[1]
        neg_edge_index = neg_edge_index[:, mask]

        # Positive edges와 겹치는 edge 제거
        pos_edge_set = set(
            zip(pos_edge_index[0].cpu().numpy(), pos_edge_index[1].cpu().numpy())
        )
        filtered_indices = []
        for i in range(neg_edge_index.size(1)):
            src, dst = neg_edge_index[0, i].item(), neg_edge_index[1, i].item()
            if (src, dst) not in pos_edge_set and (dst, src) not in pos_edge_set:
                filtered_indices.append(i)

        if filtered_indices:
            neg_edge_index = neg_edge_index[:, filtered_indices]

            # 필요한 만큼만 유지
            if neg_edge_index.size(1) > num_neg_samples:
                perm = torch.randperm(neg_edge_index.size(1))[:num_neg_samples]
                neg_edge_index = neg_edge_index[:, perm]
        else:
            print("[Warning] No valid negative edges found after filtering")
            neg_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        # Use k-hop neighbors for negative sampling
        neg_edge_index = hybrid_negative_sampling_with_k_hop(
            pos_edge_index,
            [],  # No 2-hop document edges for images
            [],  # No rejected edges for images
            z.size(0),
            num_neg_samples,
            hard_negative_ratio=hard_negative_ratio,
            k_hop_neighbors=k_hop_neighbors,
            device=device,
        )

    if neg_edge_index.size(1) > 0:
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        neg_loss = -F.logsigmoid(-neg_score).mean()
    else:
        print("[Warning] No negative samples generated. Using zero negative loss.")
        neg_loss = torch.tensor(0.0, device=device)

    # 2) Regularization loss (feature smoothness)
    reg_loss = torch.tensor(0.0, device=device)
    if lambda_factor > 0:
        # Calculate feature smoothness across graph
        src, dst = pos_edge_index
        # Sample edges if there are too many - regularization에만 샘플링 적용
        if pos_edge_index.size(1) > 10000:
            indices = torch.randperm(pos_edge_index.size(1), device=device)[:10000]
            src_reg, dst_reg = src[indices], dst[indices]
            feature_diff = (z[src_reg] - z[dst_reg]).pow(2).sum(1)
        else:
            feature_diff = (z[src] - z[dst]).pow(2).sum(1)

        reg_loss = feature_diff.mean()

    total_loss = pos_loss + neg_loss + lambda_factor * reg_loss
    return total_loss, pos_loss.item() + neg_loss.item(), reg_loss.item()


def train_image_gae(
    x_dense,  # 이미지 임베딩
    train_pos_edge_index,
    train_edge_weight,
    num_nodes,
    y_true,
    channels1=256,
    channels2=128,
    lr=0.01,
    epochs=1000,
    eval_interval=10,
    early_stopping=50,
    lambda_factor=1.0,
    hard_negative_ratio=0.5,
    n_clusters=20,
    device="cuda",
    use_scheduler=True,
    save_dir=".",
    dataset_name=None,
    use_cache=True,
    random_seed=42,
    use_k_hop_negatives=True,
    k_hop_values=[3, 4],
):
    """GAE training function specifically optimized for image datasets"""
    print(f"--- Training Image GAE (with improved negative sampling) ---")

    # 캐시 확인
    cache_path = None
    if save_dir and dataset_name and use_cache:
        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(
            save_dir,
            f"image_gae_{dataset_name}_ch1{channels1}_ch2{channels2}_lr{lr}_lambda{lambda_factor}_hnr{hard_negative_ratio}_khop{'_'.join(map(str, k_hop_values)) if use_k_hop_negatives and hard_negative_ratio > 0 else 'none'}.pkl",
        )
        if os.path.exists(cache_path):
            print(f"[CACHE] Loading Image GAE results from {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                    return cache_data
            except Exception as e:
                print(f"[Warning] Failed to load Image GAE cache: {e}")

    if x_dense is None:
        raise ValueError("Image embeddings (x_dense) are None.")
    if train_pos_edge_index is None or train_pos_edge_index.numel() == 0:
        raise ValueError("No train edges for GAE.")

    scaler = GradScaler(enabled=("cuda" in str(device)))

    # 1) 1-hop adjacency list for k-hop sampling
    one_hop_list = [[] for _ in range(num_nodes)]
    src_arr, dst_arr = train_pos_edge_index.cpu().numpy()
    for s, d in zip(src_arr, dst_arr):
        if s < num_nodes:
            one_hop_list[s].append(d)

    # 2) k-hop neighbors for negative sampling - hard_negative_ratio가 0 또는 None이면 건너뜀
    k_hop_neighbors = None
    if use_k_hop_negatives and (
        hard_negative_ratio is not None and hard_negative_ratio > 0
    ):
        print(f"Generating k-hop neighbors with values: {k_hop_values}")
        k_hop_neighbors = [set() for _ in range(num_nodes)]
        for k in k_hop_values:
            print(f"Finding {k}-hop neighbors...")
            k_hop = find_k_hop_neighbors(one_hop_list, k)
            for i in range(num_nodes):
                k_hop_neighbors[i].update(k_hop[i])

        # Verify k-hop neighbors
        total_k_hop = sum(len(neighbors) for neighbors in k_hop_neighbors)
        print(f"Total k-hop neighbors found: {total_k_hop}")
        if total_k_hop == 0:
            print("[Warning] No k-hop neighbors found. Check graph connectivity.")
            # k-hop 이웃이 없으면 k_hop_neighbors를 None으로 설정하여 일반 랜덤 샘플링 사용
            k_hop_neighbors = None
        else:
            print(f"Average k-hop neighbors per node: {total_k_hop/num_nodes:.2f}")
    else:
        print(
            f"Skipping k-hop neighbor calculation (hard_negative_ratio={hard_negative_ratio}, use_k_hop_negatives={use_k_hop_negatives})"
        )
        print("Using only random negative sampling")

    # 3) Image GAE model initialization
    model = ImageGAEEncoder(x_dense.shape[1], channels1, channels2).to(device)

    # 4) Optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=epochs // 4,
            T_mult=2,
            eta_min=lr * 0.1,
        )

    # 5) Move data to device
    x_dense = x_dense.to(device)
    train_pos_edge_index = train_pos_edge_index.to(device)
    if train_edge_weight is not None:
        train_edge_weight = train_edge_weight.to(device)

    # Best model tracking
    best_acc = 0.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    best_metrics = None

    # Training logs
    training_logs = []

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=("cuda" in str(device))):
            # GCN encoding
            z = model(x_dense, train_pos_edge_index, edge_weight=train_edge_weight)

            # Calculate loss optimized for image data
            total_loss, recon_l, reg_l = compute_loss_for_image_data(
                z,
                train_pos_edge_index,
                k_hop_neighbors=k_hop_neighbors,
                lambda_factor=lambda_factor,
                hard_negative_ratio=hard_negative_ratio,
                device=device,
            )

        if torch.isnan(total_loss):
            print(f"[Warning] Image GAE loss is NaN at epoch {epoch}")
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
                        x_dense, train_pos_edge_index, edge_weight=train_edge_weight
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
                "train_reg_loss": reg_l,
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

            # Log message
            print(
                f"[Image GAE] Epoch {epoch}/{epochs}, "
                f"Train Loss={total_loss:.4f} (Recon={recon_l:.4f}, Reg={reg_l:.4f})"
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
        f"image_gae_training_log_{dataset_name}.csv",
    )
    if training_logs:
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        pd.DataFrame(training_logs).to_csv(log_path, index=False)
        print(f"Training logs saved to {log_path}")

    print(
        f"Image GAE complete. Best S.K-means ACC={best_acc:.4f} at epoch={best_epoch}"
    )

    # Cache results
    if cache_path and best_model_state is not None:
        try:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            print(f"[CACHE] Created directory: {cache_dir}")

            # Get final embeddings using best model
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                with autocast(device_type="cuda", enabled=("cuda" in str(device))):
                    z_final = model(
                        x_dense, train_pos_edge_index, edge_weight=train_edge_weight
                    )
                final_embeddings = z_final.cpu().numpy()

            # 캐시에 결과 저장
            cache_data = {
                "embeddings": final_embeddings,
                "skmeans_acc": best_metrics.get("skmeans_acc", 0.0),
                "skmeans_ari": best_metrics.get("skmeans_ari", 0.0),
                "skmeans_nmi": best_metrics.get("skmeans_nmi", 0.0),
                "hc_acc": best_metrics.get("hc_acc", 0.0),
                "hc_ari": best_metrics.get("hc_ari", 0.0),
                "hc_nmi": best_metrics.get("hc_nmi", 0.0),
                "best_epoch": best_epoch,
                "best_model_state": best_model_state,
            }
            if use_cache:
                cache_path = os.path.join(save_dir, f"{dataset_name}.pkl")
                ensure_dir_exists(cache_path)
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                print(f"[CACHE] Saved Proposed GAE results to {cache_path}")

            # 최종 결과 반환
            return {
                "embeddings": final_embeddings,
                "skmeans_acc": best_metrics.get("skmeans_acc", 0.0),
                "skmeans_ari": best_metrics.get("skmeans_ari", 0.0),
                "skmeans_nmi": best_metrics.get("skmeans_nmi", 0.0),
                "hc_acc": best_metrics.get("hc_acc", 0.0),
                "hc_ari": best_metrics.get("hc_ari", 0.0),
                "hc_nmi": best_metrics.get("hc_nmi", 0.0),
                "best_epoch": best_epoch,
                "best_model_state": best_model_state,
            }
        except Exception as e:
            print(f"[Warning] Failed to save Image GAE cache: {e}")

    if "cuda" in str(device):
        torch.cuda.empty_cache()

    # Return final results
    if best_model_state is not None:
        # Get final embeddings using best model
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=("cuda" in str(device))):
                z_final = model(
                    x_dense, train_pos_edge_index, edge_weight=train_edge_weight
                )
            final_embeddings = z_final.cpu().numpy()

        return {
            "embeddings": final_embeddings,
            "best_epoch": best_epoch,
            "skmeans_acc": best_metrics["skmeans_acc"],
            "skmeans_ari": best_metrics["skmeans_ari"],
            "skmeans_nmi": best_metrics["skmeans_nmi"],
            "hc_acc": best_metrics["hc_acc"],
            "hc_ari": best_metrics["hc_ari"],
            "hc_nmi": best_metrics["hc_nmi"],
        }
    else:
        print("[Warning] No valid model state was saved during training")
        return {
            "embeddings": None,
            "best_epoch": 0,
            "skmeans_acc": 0.0,
            "skmeans_ari": 0.0,
            "skmeans_nmi": 0.0,
            "hc_acc": 0.0,
            "hc_ari": 0.0,
            "hc_nmi": 0.0,
        }
