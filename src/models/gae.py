# src/models/gae.py
import os
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GAE
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from ..clustering_eval import evaluate_clustering


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


def train_gae(
    x,
    edge_index,
    edge_weight,
    num_nodes,
    hidden_dim,
    out_dim,
    learning_rate,
    epochs,
    y_true,
    eval_interval=10,
    early_stopping=100,
    device="cuda",
    n_clusters=20,
    random_seed=42,
    save_dir=None,
    dataset_name=None,
    use_cache=True,
):
    print("--- Training Standard GAE ---")
    if x is None:
        print("[Error] GAE initial features are None.")
        return None, []
    if not edge_index or len(edge_index[0]) == 0:
        print("[Error] GAE edge index is empty.")
        return None, []

    # 캐시된 결과 불러오기
    cache_path = None
    if save_dir and dataset_name and use_cache:
        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(
            save_dir,
            f"gae_{dataset_name}_hd{hidden_dim}_od{out_dim}_lr{learning_rate}.pkl",
        )
        if os.path.exists(cache_path):
            print(f"[CACHE] Loading GAE results from {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                    return cache_data["embeddings"], cache_data["loss_history"]
            except Exception as e:
                print(f"[Warning] Failed to load GAE cache: {e}")

    scaler = GradScaler(enabled=("cuda" in str(device)))

    x = x.to(device)
    model = GAE(GCNEncoder(x.shape[1], hidden_dim, out_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 엣지 텐서 변환
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).to(device)
    if edge_weight is not None and len(edge_weight) == len(edge_index[0]):
        edge_weight_t = torch.tensor(edge_weight, dtype=torch.float).to(device)
    else:
        edge_weight_t = None

    best_acc = 0.0
    best_emb = None
    loss_history = []
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=("cuda" in str(device))):
            z = model.encode(x, edge_index_t, edge_weight=edge_weight_t)
            train_loss = model.recon_loss(z, edge_index_t)

        if torch.isnan(train_loss):
            print(f"[Warning] GAE loss is NaN at epoch {epoch}. Stop.")
            break

        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_history.append(train_loss.item())

        if epoch % eval_interval == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                with autocast(device_type="cuda", enabled=("cuda" in str(device))):
                    z_eval = model.encode(x, edge_index_t, edge_weight=edge_weight_t)
                    emb_eval = z_eval.cpu().numpy()

            # 클러스터링 평가 - 모델 선택의 유일한 기준
            eval_res = evaluate_clustering(
                emb_eval, y_true, k=n_clusters, random_seed=random_seed
            )

            skmeans_res = next(
                (r for r in eval_res if r["method"] == "Spherical K-means"), None
            )
            hc_res = next(
                (r for r in eval_res if r["method"] == "Hierarchical Clustering"), None
            )

            # 간결한 로깅 - 학습 손실과 클러스터링 성능만 표시
            log_msg = f"[GAE] Epoch {epoch}/{epochs}, Train Loss={train_loss:.4f}"

            if skmeans_res:
                curr_acc = skmeans_res["acc"]
                log_msg += f", S.K-means ACC={curr_acc:.4f}"

                # 모델 선택 기준: ACC만 고려
                is_best = curr_acc > best_acc
                if is_best:
                    best_acc = curr_acc
                    best_emb = emb_eval.copy()
                    best_epoch = epoch
                    epochs_no_improve = 0
                    log_msg += f" (Best)"
                else:
                    epochs_no_improve += eval_interval
            else:
                epochs_no_improve += eval_interval

            if hc_res:
                log_msg += f", HC ACC={hc_res['acc']:.4f}"

            print(log_msg)

            # Early stopping - ACC 기반
            if early_stopping > 0 and epochs_no_improve >= early_stopping:
                print(
                    f"Early stopping at epoch {epoch}, no improvement in ACC for {epochs_no_improve} epochs."
                )
                break

    print(f"GAE finished. Best ACC={best_acc:.4f} at epoch {best_epoch}")

    # 결과 캐싱
    if cache_path:
        try:
            # 디렉토리 확인 및 생성
            cache_dir = os.path.dirname(cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
                print(f"[CACHE] Created directory: {cache_dir}")

            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "embeddings": best_emb,
                        "loss_history": loss_history,
                        "best_epoch": best_epoch,
                        "best_acc": best_acc,
                    },
                    f,
                )
            print(f"[CACHE] Saved GAE results to {cache_path}")
        except Exception as e:
            print(f"[Warning] Failed to save GAE cache: {e}")

    if "cuda" in str(device):
        torch.cuda.empty_cache()
    return best_emb, loss_history
