# src/graph_utils.py
import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import csv
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, negative_sampling
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 그래프 구성
def simi_high(cand_list, s_second, dist_normalized, doc1, th, K_neighbors_indices):
    if not K_neighbors_indices.size:
        return None
    last_neighbor_idx = K_neighbors_indices[-1]
    if (last_neighbor_idx >= dist_normalized.shape[1]) or (
        doc1 >= dist_normalized.shape[0]
    ):
        return None

    threshold_similarity = dist_normalized[doc1, last_neighbor_idx] * th
    for alpha in cand_list:
        if alpha == s_second:
            continue
        if dist_normalized[s_second, alpha] > threshold_similarity:
            return s_second
    return None


def construct_graph_from_dist(
    dist_normalized,
    o_dist_original,
    K=20,
    alpha=0.2,
    cache_dir=None,
    dataset_name=None,
    use_cache=True,
):
    """
    KNN 그래프 구성 (상호 유사도 기반 필터링)
    """
    # 캐시 확인
    if cache_dir is not None and dataset_name is not None and use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"graph_{dataset_name}_K{K}_alpha{alpha}.pkl"
        )

        if os.path.exists(cache_path):
            print(f"[CACHE] Loading graph from {cache_path}")
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
                return cache_data["edge_index"], cache_data["edge_weight"]

    if dist_normalized is None or o_dist_original is None:
        print("[Error] Missing distance matrices.")
        return [[], []], []
    if dist_normalized.shape != o_dist_original.shape:
        print("[Error] dist shape mismatch.")
        return [[], []], []

    length = dist_normalized.shape[0]
    edge_ind_topk = []

    print(f"Constructing graph (K={K}, alpha={alpha})...")
    for i in tqdm(range(length), desc="Top K Neighbors"):
        actual_k = min(K, length - 1)
        if actual_k <= 0:
            p = np.array([], dtype=int)
        else:
            row_sim = dist_normalized[i]
            partitioned_indices = np.argpartition(row_sim, -actual_k)[-actual_k:]
            p = partitioned_indices[np.argsort(row_sim[partitioned_indices])[::-1]]
        edge_ind_topk.append(p)

    total_edge_index_filtered = [[] for _ in range(length)]
    for doc1 in tqdm(range(length), desc="Filtering Neighbors"):
        edge_doc1_candidate = []
        initial_neighbors = edge_ind_topk[doc1]

        if len(initial_neighbors) == 0:
            continue

        # 1) top-1
        edge_doc1_candidate.append(initial_neighbors[0])
        # 2) alpha 기준 필터
        current_cand = edge_doc1_candidate.copy()
        for idx in range(1, len(initial_neighbors)):
            s_second = initial_neighbors[idx]
            sec_index = simi_high(
                current_cand, s_second, dist_normalized, doc1, alpha, initial_neighbors
            )
            if sec_index is not None and sec_index == s_second:
                edge_doc1_candidate.append(s_second)
                current_cand.append(s_second)

        total_edge_index_filtered[doc1] = edge_doc1_candidate

    edge_index = [[], []]
    edge_weight = []
    added_edges = set()

    print("Building edge list...")
    for i in range(length):
        for j in total_edge_index_filtered[i]:
            if i == j:
                continue
            weight = o_dist_original[i, j]
            # Cosine이 [-1,1] 가능 → [0,1] 범위로 이동
            weight = (weight + 1.0) / 2.0
            weight = max(weight, 1e-6)
            edge_tuple = tuple(sorted((i, j)))
            if edge_tuple not in added_edges:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_weight.append(weight)
                # 양방향
                edge_index[0].append(j)
                edge_index[1].append(i)
                edge_weight.append(weight)
                added_edges.add(edge_tuple)

    print(f"Graph complete. {len(edge_index[0])//2} unique undirected edges.")

    # 결과 캐싱
    if cache_dir is not None and dataset_name is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"graph_{dataset_name}_K{K}_alpha{alpha}.pkl"
        )

        # 디렉토리 확인 및 생성
        cache_dir_path = os.path.dirname(cache_path)
        if cache_dir_path and not os.path.exists(cache_dir_path):
            os.makedirs(cache_dir_path, exist_ok=True)
            print(f"[CACHE] Created directory: {cache_dir_path}")

        with open(cache_path, "wb") as f:
            pickle.dump({"edge_index": edge_index, "edge_weight": edge_weight}, f)
        print(f"[CACHE] Saved graph to {cache_path}")

    return edge_index, edge_weight


# -------------------------------
# 2-Hop 관련 함수
# -------------------------------


def calculate_pairwise_tfidf_similarity(documents, doc_indices):
    if not documents or not doc_indices:
        return np.array([]), []
    sub_docs = [documents[i] for i in doc_indices if i < len(documents)]
    if not sub_docs:
        return np.array([]), []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    try:
        tfidf_matrix = vectorizer.fit_transform(sub_docs)
        tfidf_sim = cosine_similarity(tfidf_matrix)
    except ValueError:
        return np.zeros((len(sub_docs), len(sub_docs))), doc_indices

    return tfidf_sim, doc_indices


def calculate_iqr_threshold(similarities, k=0.5):
    if similarities is None or similarities.size == 0:
        return 0.0
    if similarities.size == 1:
        return similarities[0]
    Q1 = np.percentile(similarities, 25)
    Q3 = np.percentile(similarities, 75)
    IQR = Q3 - Q1
    threshold = Q3 + k * IQR
    threshold = np.clip(threshold, similarities.min(), similarities.max())
    return threshold


def create_two_hop_edge_candidates_with_iqr(
    one_hop_edges,
    documents,
    num_nodes,
    save_path=None,
    max_edges=20,
    k=0.5,
    debug=True,
    use_cache=True,
):
    """2-hop 후보군 및 필터링된 후보 생성 (캐싱 지원)"""
    # 캐시 파일 확인
    if save_path and use_cache and os.path.exists(save_path):
        try:
            print(f"[CACHE] Loading 2-hop edges from {save_path}")
            with open(save_path, "rb") as f:
                cache_data = pickle.load(f)
                two_hop_edges_selected = cache_data.get("selected_two_hop_edges", [])
                rejected_two_hop_edges = cache_data.get("rejected_two_hop_edges", [])
                return two_hop_edges_selected, rejected_two_hop_edges
        except Exception as e:
            print(f"[Error] Could not load cached 2-hop edges: {e}")
            # 캐시 로드 실패 시 계속 진행

    two_hop_edges_selected = [[] for _ in range(num_nodes)]
    rejected_two_hop_edges = []
    similarity_distributions = []

    one_hop_set = set()
    for node1, neighbors in enumerate(one_hop_edges):
        for node2 in neighbors:
            one_hop_set.add(tuple(sorted((node1, node2))))

    print("[2-Hop] Generating & filtering candidates...")
    for node1 in tqdm(range(num_nodes), desc="2-Hop"):
        one_hop_neighbors_of_node1 = set(one_hop_edges[node1])
        two_hop_candidates = set()

        for node2 in one_hop_neighbors_of_node1:
            if node2 < num_nodes:
                for node3 in one_hop_edges[node2]:
                    if node3 != node1 and node3 not in one_hop_neighbors_of_node1:
                        two_hop_candidates.add(node3)

        node_selected = []
        node_rejected = []
        if two_hop_candidates:
            candidate_doc_indices = [node1] + list(two_hop_candidates)
            tfidf_sim, _ = calculate_pairwise_tfidf_similarity(
                documents, candidate_doc_indices
            )
            index_map = {
                doc_idx: idx for idx, doc_idx in enumerate(candidate_doc_indices)
            }

            similarities = []
            candidate_edges_with_scores = []
            node1_sim_idx = index_map[node1] if node1 in index_map else None

            if node1_sim_idx is not None:
                for node3 in two_hop_candidates:
                    if node3 in index_map:
                        node3_sim_idx = index_map[node3]
                        if (node1_sim_idx < tfidf_sim.shape[0]) and (
                            node3_sim_idx < tfidf_sim.shape[1]
                        ):
                            sim_val = tfidf_sim[node1_sim_idx, node3_sim_idx]
                            similarities.append(sim_val)
                            candidate_edges_with_scores.append((node1, node3, sim_val))

            if similarities:
                sims_np = np.array(similarities)
                thresh = calculate_iqr_threshold(sims_np, k=k)
                filtered_above_thresh = [
                    (s, d, sco)
                    for (s, d, sco) in candidate_edges_with_scores
                    if sco >= thresh
                ]
                if len(filtered_above_thresh) > max_edges:
                    filtered_above_thresh.sort(key=lambda x: x[2], reverse=True)
                    filtered_above_thresh = filtered_above_thresh[:max_edges]

                # max_edges 이후는 rejected
                selected_indices = set(
                    (src, dest)
                    for (src, dest, sim) in filtered_above_thresh
                    if tuple(sorted((src, dest))) not in one_hop_set
                )
                for src, dest, sim in candidate_edges_with_scores:
                    if (src, dest) in selected_indices:
                        node_selected.append((src, dest))
                    else:
                        node_rejected.append((src, dest))

                hist, _ = np.histogram(sims_np, bins=np.arange(0, 1.05, 0.05))
                similarity_distributions.append(hist)
            else:
                similarity_distributions.append(
                    np.zeros(20, dtype=int)
                )  # 0~1 구간 0.05 step → 20개
                for c in two_hop_candidates:
                    node_rejected.append((node1, c))
        else:
            similarity_distributions.append(np.zeros(20, dtype=int))

        two_hop_edges_selected[node1] = node_selected
        rejected_two_hop_edges.extend(node_rejected)

    rejected_two_hop_edges = list(set(rejected_two_hop_edges))
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"[CACHE] Created directory: {save_dir}")

        # CSV 예시
        csv_file_path = save_path.replace(".pkl", "_similarity_distribution.csv")
        bin_labels = [f"{i*0.05:.2f}-{(i+1)*0.05:.2f}" for i in range(20)]
        try:
            csv_dir = os.path.dirname(csv_file_path)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)

            with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Node_ID"] + bin_labels)
                for node_id, hist in enumerate(similarity_distributions):
                    writer.writerow([node_id] + hist.tolist())
        except IOError as e:
            print(f"[Error] Could not write CSV: {e}")

        # Pickle로 2-hop 후보 저장
        try:
            with open(save_path, "wb") as f:
                pickle.dump(
                    {
                        "selected_two_hop_edges": two_hop_edges_selected,
                        "rejected_two_hop_edges": rejected_two_hop_edges,
                    },
                    f,
                )
            if debug:
                print(f"Saved 2-hop results to {save_path}")
        except IOError as e:
            print(f"[Error] Could not write Pickle: {e}")

    print(
        f"[2-Hop] Selected: {sum(len(s) for s in two_hop_edges_selected)}, Rejected: {len(rejected_two_hop_edges)}"
    )
    return two_hop_edges_selected, rejected_two_hop_edges


# 하드 네거티브 & 랜덤 네거티브 혼합 샘플링
def hybrid_negative_sampling(
    pos_edge_index,
    selected_two_hop_edges,
    rejected_two_hop_edges,
    num_nodes,
    num_neg_samples,
    hard_negative_ratio=0.5,
    device="cuda",
):
    if num_neg_samples == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    num_hard_target = int(hard_negative_ratio * num_neg_samples)
    num_random_target = num_neg_samples - num_hard_target

    # Hard negative 샘플링
    sampled_hard_neg = torch.empty((2, 0), dtype=torch.long, device=device)
    num_hard_sampled = 0
    if num_hard_target > 0 and rejected_two_hop_edges:
        hard_neg_candidates = (
            torch.tensor(rejected_two_hop_edges, dtype=torch.long).t().contiguous()
        )
        mask_non_self = hard_neg_candidates[0] != hard_neg_candidates[1]
        hard_neg_candidates = hard_neg_candidates[:, mask_non_self]
        mask_valid_idx = (hard_neg_candidates[0] < num_nodes) & (
            hard_neg_candidates[1] < num_nodes
        )
        hard_neg_candidates = hard_neg_candidates[:, mask_valid_idx]
        hard_neg_candidates = to_undirected(hard_neg_candidates, num_nodes=num_nodes)
        hard_neg_candidates, _ = remove_self_loops(hard_neg_candidates)

        num_hard_available = hard_neg_candidates.size(1)
        num_hard_to_sample = min(num_hard_target, num_hard_available)
        if num_hard_to_sample > 0:
            perm = torch.randperm(num_hard_available)[:num_hard_to_sample]
            sampled_hard_neg = hard_neg_candidates[:, perm].to(device)
            num_hard_sampled = sampled_hard_neg.size(1)

    # Random negative 샘플링
    num_random_needed = num_neg_samples - num_hard_sampled
    sampled_random_neg = torch.empty((2, 0), dtype=torch.long, device=device)
    if num_random_needed > 0:
        exclude_edges_list = []

        # 모든 텐서들이 같은 디바이스에 있는지 확인하고 필요하면 이동
        if pos_edge_index.device != device:
            pos_edge_index_device = pos_edge_index.to(device)
        else:
            pos_edge_index_device = pos_edge_index

        exclude_edges_list.append(pos_edge_index_device)

        flat_two_hop = [
            edge for node_edges in selected_two_hop_edges for edge in node_edges
        ]
        if flat_two_hop:
            selected_two_hop_tensor = (
                torch.tensor(flat_two_hop, dtype=torch.long).t().contiguous()
            )
            selected_two_hop_tensor = selected_two_hop_tensor.to(device)
            exclude_edges_list.append(selected_two_hop_tensor)

        if sampled_hard_neg.numel() > 0:
            exclude_edges_list.append(sampled_hard_neg)

        # 모든 텐서들이 같은 디바이스에 있으므로 concat 가능
        exclude_all = torch.cat(exclude_edges_list, dim=1)
        exclude_all_undir = to_undirected(exclude_all, num_nodes=num_nodes)
        exclude_all_undir, _ = remove_self_loops(exclude_all_undir)

        try:
            neg_edge_index_random = negative_sampling(
                edge_index=exclude_all_undir,
                num_nodes=num_nodes,
                num_neg_samples=num_random_needed,
            )
            sampled_random_neg = neg_edge_index_random.to(device)
        except RuntimeError as e:
            print(f"[Error] Random negative sampling failed: {e}")

    # 최종 결과 합치기
    neg_edge_results = []
    if sampled_hard_neg.numel() > 0:
        neg_edge_results.append(sampled_hard_neg)
    if sampled_random_neg.numel() > 0:
        neg_edge_results.append(sampled_random_neg)

    if neg_edge_results:
        final_neg = torch.cat(neg_edge_results, dim=1)
        # 필요한 것보다 많으면 잘라내기
        if final_neg.size(1) > num_neg_samples:
            perm = torch.randperm(final_neg.size(1))[:num_neg_samples]
            final_neg = final_neg[:, perm]
    else:
        final_neg = torch.empty((2, 0), dtype=torch.long, device=device)

    return final_neg


def find_k_hop_neighbors(one_hop_list, k=3):
    """Find k-hop neighbors for each node."""
    num_nodes = len(one_hop_list)
    k_hop_neighbors = [set() for _ in range(num_nodes)]

    for node in range(num_nodes):
        current_neighbors = set(one_hop_list[node])
        visited = set(one_hop_list[node])
        visited.add(node)

        for _ in range(k - 1):
            next_neighbors = set()
            for neighbor in current_neighbors:
                for next_hop in one_hop_list[neighbor]:
                    if next_hop not in visited:
                        next_neighbors.add(next_hop)
                        visited.add(next_hop)
            current_neighbors = next_neighbors
            k_hop_neighbors[node].update(current_neighbors)

    return k_hop_neighbors


def hybrid_negative_sampling_with_k_hop(
    pos_edge_index,
    selected_two_hop_edges,
    rejected_two_hop_edges,
    num_nodes,
    num_neg_samples,
    hard_negative_ratio=0.5,
    k_hop_neighbors=None,
    device="cuda",
):
    """Hybrid negative sampling with k-hop neighbors."""
    # 메모리 문제 방지를 위한 최대 negative sample 수 설정
    max_neg_samples = 100000  # 필요에 따라 조정
    if num_neg_samples > max_neg_samples:
        print(f"[Warning] Limiting negative samples from {num_neg_samples} to {max_neg_samples}")
        num_neg_samples = max_neg_samples
        
    # hard_negative_ratio가 0이거나 k_hop_neighbors가 None이면 일반 네거티브 샘플링만 수행
    if hard_negative_ratio <= 0 or k_hop_neighbors is None:
        return hybrid_negative_sampling(
            pos_edge_index,
            selected_two_hop_edges,
            rejected_two_hop_edges,
            num_nodes,
            num_neg_samples,
            hard_negative_ratio=0.0,  # 모든 샘플을 랜덤으로 생성
            device=device,
        )

    if num_neg_samples == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    num_hard_target = int(hard_negative_ratio * num_neg_samples)
    num_random_target = num_neg_samples - num_hard_target

    # Hard negative 샘플링
    sampled_hard_neg = torch.empty((2, 0), dtype=torch.long, device=device)
    num_hard_sampled = 0
    if num_hard_target > 0 and k_hop_neighbors:
        # Convert k_hop_neighbors to tensor format
        hard_neg_candidates = []
        for src in range(num_nodes):
            for dst in k_hop_neighbors[src]:
                if src != dst:  # Avoid self-loops
                    hard_neg_candidates.append([src, dst])

        if hard_neg_candidates:
            hard_neg_candidates = torch.tensor(
                hard_neg_candidates, dtype=torch.long
            ).t()
            num_hard_available = hard_neg_candidates.size(1)
            num_hard_to_sample = min(num_hard_target, num_hard_available)
            if num_hard_to_sample > 0:
                perm = torch.randperm(num_hard_available)[:num_hard_to_sample]
                sampled_hard_neg = hard_neg_candidates[:, perm].to(device)
                num_hard_sampled = sampled_hard_neg.size(1)
                print(f"Sampled {num_hard_sampled} hard negative edges from {num_hard_available} k-hop candidates")
        else:
            print("[Warning] No k-hop candidates found for hard negative sampling")

    # Random negative 샘플링
    num_random_needed = num_neg_samples - num_hard_sampled
    sampled_random_neg = torch.empty((2, 0), dtype=torch.long, device=device)
    if num_random_needed > 0:
        exclude_edges_list = []

        # 모든 텐서들이 같은 디바이스에 있는지 확인하고 필요하면 이동
        if pos_edge_index.device != device:
            pos_edge_index_device = pos_edge_index.to(device)
        else:
            pos_edge_index_device = pos_edge_index

        exclude_edges_list.append(pos_edge_index_device)

        flat_two_hop = [
            edge for node_edges in selected_two_hop_edges for edge in node_edges
        ]
        if flat_two_hop:
            selected_two_hop_tensor = (
                torch.tensor(flat_two_hop, dtype=torch.long).t().contiguous()
            )
            selected_two_hop_tensor = selected_two_hop_tensor.to(device)
            exclude_edges_list.append(selected_two_hop_tensor)

        if sampled_hard_neg.numel() > 0:
            exclude_edges_list.append(sampled_hard_neg)

        # 모든 텐서들이 같은 디바이스에 있으므로 concat 가능
        if exclude_edges_list:
            exclude_all = torch.cat(exclude_edges_list, dim=1)
            exclude_all_undir = to_undirected(exclude_all, num_nodes=num_nodes)
            exclude_all_undir, _ = remove_self_loops(exclude_all_undir)
        else:
            # Fallback for when there are no edges to exclude
            exclude_all_undir = torch.zeros((2, 0), dtype=torch.long, device=device)

        try:
            neg_edge_index_random = negative_sampling(
                edge_index=exclude_all_undir,
                num_nodes=num_nodes,
                num_neg_samples=num_random_needed,
            )
            sampled_random_neg = neg_edge_index_random.to(device)
            print(f"Sampled {sampled_random_neg.size(1)} random negative edges")
        except RuntimeError as e:
            print(f"[Error] Random negative sampling failed: {e}")
            # Fallback: 직접 랜덤 샘플링 생성
            all_edges = set()
            for i in range(exclude_all_undir.size(1)):
                all_edges.add((exclude_all_undir[0, i].item(), exclude_all_undir[1, i].item()))
            
            sampled_edges = []
            attempts = 0
            max_attempts = num_random_needed * 10  # 최대 시도 횟수 제한
            
            while len(sampled_edges) < num_random_needed and attempts < max_attempts:
                src = torch.randint(0, num_nodes, (1,)).item()
                dst = torch.randint(0, num_nodes, (1,)).item()
                
                if src != dst and (src, dst) not in all_edges and (dst, src) not in all_edges:
                    sampled_edges.append((src, dst))
                    all_edges.add((src, dst))
                    all_edges.add((dst, src))
                
                attempts += 1
            
            if sampled_edges:
                sampled_random_neg = torch.tensor(sampled_edges, dtype=torch.long).t().to(device)
                print(f"Fallback: Sampled {sampled_random_neg.size(1)} random negative edges manually")
            else:
                print("[Warning] Failed to sample random negative edges")

    # 최종 결과 합치기
    neg_edge_results = []
    if sampled_hard_neg.numel() > 0:
        neg_edge_results.append(sampled_hard_neg)
    if sampled_random_neg.numel() > 0:
        neg_edge_results.append(sampled_random_neg)

    if neg_edge_results:
        final_neg = torch.cat(neg_edge_results, dim=1)
        # 필요한 것보다 많으면 잘라내기
        if final_neg.size(1) > num_neg_samples:
            perm = torch.randperm(final_neg.size(1))[:num_neg_samples]
            final_neg = final_neg[:, perm]
        print(f"Final negative samples: {final_neg.size(1)} (target: {num_neg_samples})")
    else:
        print("[Warning] No negative samples generated")
        final_neg = torch.empty((2, 0), dtype=torch.long, device=device)

    return final_neg