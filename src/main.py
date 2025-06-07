# src/main.py
import os
import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.utils.data as data_utils
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union
from torch_geometric.utils import to_undirected
import time
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# 필요한 모듈 임포트
from src.config import (
    config,
    DATASET_PATHS,
    parse_args,
    get_experiment_params,
    is_image_dataset,
)
from src.data_utils import (
    set_seed,
    read_text,
    read_label,
    preprocess_text,
)
from src.embeddings import (
    get_tfidf_embeddings,
    get_pretrained_text_embeddings,
    get_pretrained_image_embeddings,
    calculate_cosine_similarity_matrix,
    prepare_normalized_dist_for_graph,
)
from src.graph_utils import construct_graph_from_dist
from src.clustering_eval import evaluate_clustering
from src.models.gae import train_gae
from src.models.proposed_gae import train_improved_proposed_gae
from src.experiment_manager import ExperimentManager

# 이미지 데이터셋 로더 임포트
try:
    from src.datasets.image_datasets import (
        load_stl10,
        load_cifar10,
        load_cifar100_20,
        load_imagenet10,
        load_imagenet_dogs,
        get_image_dataset_info,
    )

    HAS_IMAGE_DATASETS = True
except ImportError:
    HAS_IMAGE_DATASETS = False
    print(
        "[Warning] Image dataset module not found. Image datasets will not be available."
    )

# torchvision / PIL (이미지 데이터셋용)
try:
    import PIL

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[Warning] PIL not found. Image processing will be limited.")

################################################################################
# Helper functions
################################################################################


# 파일 저장 전 디렉토리 생성을 확인하는 헬퍼 함수 추가
def ensure_dir_exists(file_path):
    """파일 경로의 디렉토리가 존재하는지 확인하고, 없으면 생성"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"[Info] Created directory: {directory}")


def load_image_dataset(
    dataset_name: str, max_samples: Optional[int] = None
) -> Tuple[List, np.ndarray]:
    """이미지 데이터셋 로드"""
    if not HAS_IMAGE_DATASETS or not HAS_PIL:
        print("[Error] 이미지 데이터셋 로드에 필요한 모듈이 설치되지 않았습니다.")
        return None, None

    dataset_info = get_image_dataset_info(dataset_name)
    if not dataset_info:
        print(f"[Error] 알 수 없는 이미지 데이터셋: {dataset_name}")
        return None, None

    print(
        f"[Info] 이미지 데이터셋 로드 중: {dataset_name} ({dataset_info['description']})"
    )

    # 표준 이미지 변환 설정
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )

    # 데이터셋별 로더 호출
    loader_func = dataset_info["loader"]
    images, labels = loader_func(
        root="./data", transform=transform, max_samples=max_samples
    )

    if images is None or labels is None:
        print(f"[Error] 데이터셋 {dataset_name} 로드 실패")
        return None, None

    print(f"[Info] 로드 완료: {len(images)} 이미지, {len(np.unique(labels))} 클래스")
    return images, labels


################################################################################
# 메인 함수
################################################################################


def evaluate_clustering_performance(labels_true, labels_pred):
    """클러스터링 결과 평가"""
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    return nmi, ari


def plot_results(results, save_path):
    """실험 결과 시각화"""
    plt.figure(figsize=(12, 6))

    # NMI 비교
    plt.subplot(1, 2, 1)
    methods = list(results.keys())
    nmis = [results[m]["nmi"] for m in methods]
    plt.bar(methods, nmis)
    plt.title("NMI Comparison")
    plt.xticks(rotation=45)
    plt.ylabel("NMI Score")

    # ARI 비교
    plt.subplot(1, 2, 2)
    aris = [results[m]["ari"] for m in methods]
    plt.bar(methods, aris)
    plt.title("ARI Comparison")
    plt.xticks(rotation=45)
    plt.ylabel("ARI Score")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # 전체 클러스터링 시간 측정 시작
    total_start_time = time.time()

    args = parse_args()
    print("===== Experiment Parameters =====")
    print(vars(args))

    if "graph" not in config:
        config["graph"] = config["cache"]["stages"]["graph"]
    if "gae" not in config:
        config["gae"] = config["cache"]["stages"]["gae"]
    if "proposed" not in config:
        config["proposed"] = config["cache"]["stages"]["proposed"]
    if "ablation" not in config:
        config["ablation"] = config["cache"]["ablation"]
    if "clustering" not in config:
        config["clustering"] = config["cache"]["clustering"]

    # config & 디바이스 초기 설정
    config["knn_graph_method"] = args.knn_graph_method
    config["tfidf_method"] = args.tfidf_method
    config["pretrained_text_model_name"] = args.pretrained_text_model
    config["pretrained_text_model_type"] = args.pretrained_text_model_type
    config["pretrained_image_model_name"] = args.pretrained_image_model
    config["gae_init_feature"] = args.gae_init_feature
    if "proposed" not in config:
        config["proposed"] = config["cache"]["stages"]["proposed"]
    config["proposed"]["hard_negative_ratio"] = args.hard_negative_ratio
    config["ablation"]["skip_gae"] = args.skip_gae
    config["ablation"]["skip_proposed"] = args.skip_proposed

    # 캐싱 설정
    use_cache = args.use_cache

    # dataset 경로 설정
    if args.dataset in DATASET_PATHS:
        config["data_path"].update(DATASET_PATHS[args.dataset])
        config["save_dir"] = os.path.join("./results", args.dataset)
    else:
        config["save_dir"] = os.path.join("./results", args.dataset)

    # 캐시 디렉토리 생성
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["save_dir"], "cache", "graph"), exist_ok=True)
    os.makedirs(os.path.join(config["save_dir"], "cache", "gae"), exist_ok=True)
    os.makedirs(os.path.join(config["save_dir"], "cache", "proposed"), exist_ok=True)

    # Set cache directory paths
    config["graph"]["cache_dir"] = os.path.join(config["save_dir"], "cache", "graph")
    config["gae"]["cache_dir"] = os.path.join(config["save_dir"], "cache", "gae")
    config["proposed"]["cache_dir"] = os.path.join(
        config["save_dir"], "cache", "proposed"
    )
    config["graph"] = config["cache"]["stages"]["graph"]
    config["gae"] = config["cache"]["stages"]["gae"]
    config["proposed"] = config["cache"]["stages"]["proposed"]

    set_seed(config["random_seed"])

    # 디바이스 설정
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----- 1. 데이터 로드 (텍스트 / 이미지) -----
    is_img_dataset = is_image_dataset(args.dataset)
    texts = processed_texts = images = y_true = None

    if is_img_dataset:
        print(f"[Info] 이미지 데이터셋 로드: {args.dataset}")
        images, y_true = load_image_dataset(args.dataset, max_samples=args.max_samples)
        if images is None:
            print("[Error] Failed to load images.")
            return
        num_nodes = len(y_true)
        print(
            f"[Info] 이미지 데이터셋 로드 완료: {num_nodes} samples, {len(np.unique(y_true))} classes"
        )
    else:
        text_path = config["data_path"].get("text")
        label_path = config["data_path"].get("label")
        if not (text_path and label_path):
            print("[Error] Invalid text dataset paths.")
            return
        texts = read_text(text_path)
        y_true = read_label(label_path)
        if texts is None or y_true is None:
            print("[Error] Failed to load text or labels.")
            return
        if len(texts) != len(y_true):
            print("[Warning] # of texts != # of labels. Truncating.")
            min_len = min(len(texts), len(y_true))
            texts = texts[:min_len]
            y_true = y_true[:min_len]

        # Max samples 적용
        if args.max_samples and args.max_samples < len(texts):
            indices = np.random.choice(len(texts), args.max_samples, replace=False)
            texts = [texts[i] for i in indices]
            y_true = y_true[indices]
            print(f"[Info] Using {args.max_samples} samples from {args.dataset}")

        num_nodes = len(y_true)
        processed_texts = preprocess_text(texts)
        print(
            f"[Info] 텍스트 데이터셋 로드 완료: {num_nodes} samples, {len(np.unique(y_true))} classes"
        )

    if args.n_clusters is not None:
        k_clusters = args.n_clusters
    else:
        # 라벨로부터 자동 계산
        unique_classes = len(np.unique(y_true))
        print(f"Detected {unique_classes} unique classes in the dataset")
        k_clusters = unique_classes

    # Make sure k_clusters is always set correctly
    if is_img_dataset:
        # For known image datasets, double-check the class count
        dataset_info = get_image_dataset_info(args.dataset)
        if dataset_info and "n_clusters" in dataset_info:
            expected_clusters = dataset_info["n_clusters"]
            if k_clusters != expected_clusters:
                print(
                    f"[Warning] Detected {k_clusters} classes but {args.dataset} should have {expected_clusters} classes"
                )
                print(f"[Warning] Using the expected number: {expected_clusters}")
                k_clusters = expected_clusters

    # Set the cluster count in config
    config["clustering"]["k_values"] = k_clusters
    print(f"Number of clusters (k) = {k_clusters}")

    # 결과 저장용
    results_list = []

    # 캐시 매니저 초기화
    cache_manager = ExperimentManager(base_dir=config["cache"]["base_dir"])

    # 캐시 초기화
    if args.clear_cache:
        cache_manager.clear_cache()
    elif args.clear_stage:
        cache_manager.clear_stage(args.clear_stage)

    # 3) Direct Clustering만 진행?
    if args.direct_clustering:
        print("\n===== [Direct Clustering Mode] =====")
        # 직접 클러스터링을 위한 임베딩 준비
        if is_img_dataset:
            if images is None:
                print("[Error] No images => cannot do direct clustering.")
                return

            # 이미지 임베딩 추출
            image_embeddings = get_pretrained_image_embeddings(
                images,
                config["pretrained_image_model_name"],
                device,
                save_dir=config["save_dir"],
                dataset_name=args.dataset,
                use_cache=use_cache,
            )

            if image_embeddings is None:
                print("[Error] Failed to extract image embeddings.")
                return

            # 클러스터링 평가
            direct_res = evaluate_clustering(image_embeddings, y_true, k=k_clusters)
        else:
            if processed_texts is None:
                print(
                    "[Error] No processed_texts => cannot do direct clustering with TF-IDF."
                )
                return

            # TF-IDF 임베딩
            tfidf_features = get_tfidf_embeddings(
                processed_texts,
                tfidf_method=config["tfidf_method"],
                max_features=config["cache"]["stages"]["embeddings"]["tfidf"][
                    "max_features"
                ],
                threshold=config["cache"]["stages"]["embeddings"]["tfidf"]["threshold"],
                save_dir=config["save_dir"],
                dataset_name=args.dataset,
                use_cache=use_cache,
            )

            if tfidf_features is None:
                print("[Error] Failed to extract TF-IDF features.")
                return

            # 클러스터링 평가
            direct_res = evaluate_clustering(tfidf_features, y_true, k=k_clusters)

        # 결과 처리
        if direct_res:
            skmeans_res = next(
                (r for r in direct_res if r["method"] == "Spherical K-means"), None
            )
            hc_res = next(
                (r for r in direct_res if r["method"] == "Hierarchical Clustering"),
                None,
            )

            if skmeans_res:
                df_direct = pd.DataFrame(
                    {
                        "approach": f'DirectClustering({config["knn_graph_method"]})',
                        "acc": skmeans_res["acc"],
                        "ari": skmeans_res["ari"],
                        "nmi": skmeans_res["nmi"],
                    },
                    index=[0],
                )
                results_list.append(df_direct)
                print("\n--- Direct Clustering Results (Spherical K-means) ---")
                print(df_direct)

                # Hierarchical Clustering 결과도 추가
                if hc_res:
                    df_direct_hc = pd.DataFrame(
                        {
                            "approach": f'DirectClustering-HC({config["knn_graph_method"]})',
                            "acc": hc_res["acc"],
                            "ari": hc_res["ari"],
                            "nmi": hc_res["nmi"],
                        },
                        index=[0],
                    )
                    results_list.append(df_direct_hc)
                    print("\n--- Direct Clustering Results (Hierarchical) ---")
                    print(df_direct_hc)

                # 결과를 CSV로 저장
                save_path = os.path.join(
                    config["save_dir"], f"results_{args.dataset}_direct.csv"
                )
                ensure_dir_exists(save_path)
                df_direct.to_csv(save_path, index=False)
                print(f"[Info] Direct clustering results saved to: {save_path}")
            else:
                print(
                    "[Warning] No Spherical K-means result found in direct clustering."
                )
        else:
            print("[Warning] No direct clustering result.")
        return

    # 4) 그래프 생성을 위한 임베딩 준비
    # 4-1) Sparse 임베딩 (TF-IDF 또는 이미지 특성)
    sparse_embeddings = None
    if is_img_dataset:
        # 이미지의 경우 sparse embeddings는 사용하지 않음 (향후 확장 가능)
        pass
    elif processed_texts is not None:
        sparse_embeddings = get_tfidf_embeddings(
            processed_texts,
            tfidf_method=config["tfidf_method"],
            max_features=config["cache"]["stages"]["embeddings"]["tfidf"][
                "max_features"
            ],
            threshold=config["cache"]["stages"]["embeddings"]["tfidf"]["threshold"],
            save_dir=config["save_dir"],
            dataset_name=f"{args.dataset}_sparse",
            use_cache=use_cache,
        )

    # 4-2) Dense 임베딩 (사전학습 모델)
    dense_embeddings = None
    if is_img_dataset and images is not None:
        # 이미지 임베딩
        dense_embeddings = get_pretrained_image_embeddings(
            images,
            config["pretrained_image_model_name"],
            device,
            save_dir=config["save_dir"],
            dataset_name=f"{args.dataset}_dense",
            use_cache=use_cache,
        )
    elif texts is not None:
        # 텍스트 임베딩
        dense_embeddings = get_pretrained_text_embeddings(
            texts,
            config["pretrained_text_model_name"],
            config["pretrained_text_model_type"],
            device,
            save_dir=config["save_dir"],
            dataset_name=f"{args.dataset}_dense",
            use_cache=use_cache,
        )

    # 이미지 데이터셋의 경우 sparse embeddings 생성
    if is_img_dataset and dense_embeddings is not None:
        # 이미지는 sparse embedding 대신 dense embedding을 사용
        sparse_embeddings = dense_embeddings

    # 유사도 행렬 계산
    if sparse_embeddings is not None:
        sim_sparse = calculate_cosine_similarity_matrix(sparse_embeddings)
    else:
        sim_sparse = None

    if dense_embeddings is not None:
        sim_dense = calculate_cosine_similarity_matrix(dense_embeddings)
    else:
        sim_dense = None

    # 최종 유사도 행렬 결정 (KNN 그래프 구성용)
    if config["knn_graph_method"] == "tfidf" and sim_sparse is not None:
        o_dist = sim_sparse
    elif (
        config["knn_graph_method"] == "pretrained_text"
        and sim_dense is not None
        and not is_img_dataset
    ):
        o_dist = sim_dense
    elif (
        config["knn_graph_method"] == "pretrained_image"
        and sim_dense is not None
        and is_img_dataset
    ):
        o_dist = sim_dense
    elif (
        config["knn_graph_method"] == "reranked_tfidf_text"
        and sim_sparse is not None
        and sim_dense is not None
        and not is_img_dataset
    ):
        # TF-IDF와 사전학습 임베딩 결합
        alpha = config["graph"]["alpha"]
        o_dist = alpha * sim_sparse + (1 - alpha) * sim_dense
    elif (
        config["knn_graph_method"] == "reranked_tfidf_image"
        and sim_sparse is not None
        and sim_dense is not None
        and is_img_dataset
    ):
        # 이미지 임베딩 결합 (단, 이미지는 sparse/dense 모두 동일할 수 있음)
        alpha = config["graph"]["alpha"]
        o_dist = sim_dense  # 이미지의 경우 dense embeddings만 사용
    else:
        print(
            f"[Error] Cannot create similarity matrix for method: {config['knn_graph_method']} with is_image_dataset={is_img_dataset}"
        )
        return

    # 유사도 행렬 정규화
    np.clip(o_dist, -1.0, 1.0, out=o_dist)
    dist_norm = (o_dist - o_dist.min()) / (o_dist.max() - o_dist.min())

    # 5) KNN 그래프 구성
    graph_cache_dir = os.path.join(config["graph"]["cache_dir"], args.dataset)
    os.makedirs(graph_cache_dir, exist_ok=True)

    edge_index_list, edge_weight_list = construct_graph_from_dist(
        dist_norm,
        o_dist,
        K=config["graph"]["max_connections"],
        alpha=config["graph"]["alpha"],
        cache_dir=graph_cache_dir,
        dataset_name=f"{args.dataset}_{args.knn_graph_method}",
        use_cache=use_cache,
    )

    if edge_index_list is None or len(edge_index_list[0]) == 0:
        print("[Error] Failed to create graph. Edge list is empty.")
        return

    # 6) Torch 텐서 변환
    edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long)
    edge_weight_tensor = None
    if (
        edge_weight_list is not None
        and len(edge_weight_list) == edge_index_tensor.shape[1]
    ):
        edge_weight_tensor = torch.tensor(edge_weight_list, dtype=torch.float)

    # 7) 특성 텐서 변환
    x_sparse = (
        torch.tensor(sparse_embeddings, dtype=torch.float)
        if sparse_embeddings is not None
        else None
    )
    x_dense = (
        torch.tensor(dense_embeddings, dtype=torch.float)
        if dense_embeddings is not None
        else None
    )

    # 8) 전체 엣지를 학습에 사용 (validation loss는 제거했으므로 train-val 분할 불필요)
    # to_undirected 변환
    edge_index_undir, edge_weight_undir = to_undirected(
        edge_index_tensor, edge_weight_tensor, num_nodes
    )

    # 9) 표준 GAE 학습
    if not config["ablation"]["skip_gae"]:
        print("\n===== [Standard GAE Training] =====")

        # GAE 초기 특성 설정
        if config["gae_init_feature"] == "onehot":
            x_gae = torch.eye(num_nodes, dtype=torch.float)
        elif config["gae_init_feature"] == "sparse" and x_sparse is not None:
            x_gae = x_sparse
        elif config["gae_init_feature"] == "dense" and x_dense is not None:
            x_gae = x_dense
        elif config["gae_init_feature"] == "combined" and x_dense is not None:
            # 데이터 타입에 따라 combined 특성 설정
            if is_img_dataset:
                # 이미지의 경우 combined는 dense와 동일
                x_gae = x_dense
            else:
                # 텍스트의 경우 sparse와 dense 결합
                if x_sparse is not None:
                    # 차원 맞추기 (필요한 경우)
                    if x_sparse.shape[1] > 200:
                        # 차원 축소 또는 샘플링
                        indices = torch.randperm(x_sparse.shape[1])[:200]
                        sparse_reduced = x_sparse[:, indices]
                    else:
                        sparse_reduced = x_sparse

                    # 결합
                    x_gae = torch.cat([x_dense, sparse_reduced], dim=1)
                else:
                    x_gae = x_dense
        else:
            print("[Warning] Using default one-hot features for GAE")
            x_gae = torch.eye(num_nodes, dtype=torch.float)

        # GAE 학습 & 결과 추출
        gae_cache_dir = os.path.join(config["gae"]["cache_dir"], args.dataset)
        os.makedirs(gae_cache_dir, exist_ok=True)

        gae_emb, _ = train_gae(
            x=x_gae,
            edge_index=edge_index_list,
            edge_weight=edge_weight_list,
            num_nodes=num_nodes,
            hidden_dim=config["gae"]["hidden_dim"],
            out_dim=config["gae"]["out_dim"],
            learning_rate=config["gae"]["learning_rate"],
            epochs=config["gae"]["epochs"],
            eval_interval=config["gae"]["eval_interval"],
            early_stopping=config["gae"]["early_stopping"],
            y_true=y_true,
            device=device,
            n_clusters=k_clusters,
            save_dir=gae_cache_dir,
            dataset_name=f"{args.dataset}_{args.knn_graph_method}_{config['gae_init_feature']}",
            use_cache=use_cache,
            random_seed=config["random_seed"],
        )

        # GAE 결과 평가
        if gae_emb is not None:
            gae_res = evaluate_clustering(
                gae_emb,
                y_true,
                k=k_clusters,
            )
            if gae_res:
                # Spherical K-means 결과 추출
                skmeans_res = next(
                    (r for r in gae_res if r["method"] == "Spherical K-means"), None
                )
                hc_res = next(
                    (r for r in gae_res if r["method"] == "Hierarchical Clustering"),
                    None,
                )

                if skmeans_res:
                    df_gae = pd.DataFrame(
                        {
                            "approach": f"StandardGAE(graph={args.knn_graph_method}, feat={config['gae_init_feature']})",
                            "acc": skmeans_res["acc"],
                            "ari": skmeans_res["ari"],
                            "nmi": skmeans_res["nmi"],
                        },
                        index=[0],
                    )
                    results_list.append(df_gae)

                    # Hierarchical Clustering 결과도 추가
                    if hc_res:
                        df_gae_hc = pd.DataFrame(
                            {
                                "approach": f"StandardGAE-HC(graph={args.knn_graph_method}, feat={config['gae_init_feature']})",
                                "acc": hc_res["acc"],
                                "ari": hc_res["ari"],
                                "nmi": hc_res["nmi"],
                            },
                            index=[0],
                        )
                        results_list.append(df_gae_hc)

                    save_path = os.path.join(
                        config["save_dir"],
                        f"results_{args.dataset}_gae_{args.knn_graph_method}_{config['gae_init_feature']}.csv",
                    )
                    ensure_dir_exists(save_path)
                    df_gae.to_csv(save_path, index=False)

                    print("\n----- Standard GAE Results -----")
                    print(df_gae.to_string(index=False))
                    if hc_res:
                        print("\n----- Standard GAE Results (Hierarchical) -----")
                        print(df_gae_hc.to_string(index=False))
                else:
                    print(
                        "[Warning] No Spherical K-means result found in GAE evaluation."
                    )
            else:
                print("[Warning] GAE evaluation returned no results.")
        else:
            print("[Warning] GAE embedding is None.")
    else:
        print("\n[Info] Skipping Standard GAE training as configured.")

    # 10) 개선된 Proposed GAE 학습
    if not config["ablation"]["skip_proposed"]:
        print("\n===== [Proposed GAE with Combined Features] =====")

        # 이미지 데이터셋도 sparse/dense 임베딩이 필요
        if is_img_dataset:
            if x_dense is None:
                print(
                    "[Error] Dense embeddings are required for image datasets with Proposed GAE."
                )
                return
            # 이미지의 경우 sparse = dense 설정
            if x_sparse is None:
                x_sparse = x_dense
        elif x_sparse is None or x_dense is None:
            print(
                "[Error] Both sparse and dense features are required for Proposed GAE."
            )
            return

        # Proposed GAE 캐시 디렉토리
        prop_gae_cache_dir = os.path.join(config["proposed"]["cache_dir"], args.dataset)
        os.makedirs(prop_gae_cache_dir, exist_ok=True)

        # 이미지 데이터셋인 경우 specialized image GAE 사용
        if is_img_dataset:
            print("Using specialized Image GAE for image dataset...")
            from src.models.proposed_gae_image import train_image_gae

            # hard_negative_ratio 값 확인
            hard_negative_ratio = config["proposed"]["hard_negative_ratio"]
            use_k_hop_negatives = config["proposed"]["use_k_hop_negatives"]

            # hard_negative_ratio가 없거나 0이면 k-hop 계산 비활성화
            if hard_negative_ratio is None or hard_negative_ratio <= 0:
                print(
                    f"[Info] hard_negative_ratio is {hard_negative_ratio} - disabling k-hop negative sampling"
                )
                use_k_hop_negatives = False

            prop_results = train_image_gae(
                x_dense=x_dense,
                train_pos_edge_index=edge_index_undir,
                train_edge_weight=edge_weight_undir,
                num_nodes=num_nodes,
                y_true=y_true,
                channels1=config["proposed"]["channels1"],
                channels2=config["proposed"]["channels2"],
                lr=config["proposed"]["lr"],
                epochs=config["proposed"]["epochs"],
                eval_interval=config["proposed"]["eval_interval"],
                early_stopping=config["proposed"]["early_stopping"],
                lambda_factor=config["proposed"]["lambda_factor"],
                hard_negative_ratio=hard_negative_ratio,
                n_clusters=k_clusters,
                device=device,
                use_scheduler=config["proposed"]["use_scheduler"],
                save_dir=prop_gae_cache_dir,
                dataset_name=f"{args.dataset}_{args.knn_graph_method}_hnr{hard_negative_ratio}",
                use_cache=use_cache,
                random_seed=config["random_seed"],
                use_k_hop_negatives=use_k_hop_negatives,
                k_hop_values=config["proposed"]["k_hop_values"],
            )
        else:
            # 텍스트 데이터셋은 기존 ProposedGAE 사용
            prop_results = train_improved_proposed_gae(
                x_sparse=x_sparse,
                x_dense=x_dense,
                train_pos_edge_index=edge_index_undir,  # 전체 엣지 사용
                train_edge_weight=edge_weight_undir,  # 전체 엣지 가중치 사용
                val_pos_edge_index=None,  # 사용하지 않음
                documents=(
                    texts if not is_img_dataset else None
                ),  # 이미지 데이터셋은 documents=None
                num_nodes=num_nodes,
                y_true=y_true,
                channels1=config["proposed"]["channels1"],
                channels2=config["proposed"]["channels2"],
                lr=config["proposed"]["lr"],
                epochs=config["proposed"]["epochs"],
                eval_interval=config["proposed"]["eval_interval"],
                early_stopping=config["proposed"]["early_stopping"],
                lambda_factor=config["proposed"]["lambda_factor"],
                iqr_k=config["proposed"]["iqr_k"],
                max_edges_2hop=config["graph"]["max_connections"],
                hard_negative_ratio=config["proposed"]["hard_negative_ratio"],
                n_clusters=k_clusters,
                device=device,
                use_scheduler=config["proposed"]["use_scheduler"],
                save_dir=prop_gae_cache_dir,
                dataset_name=f"{args.dataset}_{args.knn_graph_method}_hnr{config['proposed']['hard_negative_ratio']}",
                use_cache=use_cache,
                random_seed=config["random_seed"],
                use_standard_gae=False,
                gae_init_feature=config["gae_init_feature"],  # GAE 초기 특성 설정 추가
                use_k_hop_negatives=config["proposed"]["use_k_hop_negatives"],
                k_hop_values=config["proposed"]["k_hop_values"],
            )

        # 11) 클러스터링 평가 및 결과 저장
        if (
            prop_results is not None
            and "embeddings" in prop_results
            and prop_results["embeddings"] is not None
        ):
            prop_emb = prop_results["embeddings"]
            # 결과 처리
            df_prop = pd.DataFrame(
                {
                    "approach": f"ProposedGAE(graph={args.knn_graph_method}, hnr={config['proposed']['hard_negative_ratio']})",
                    "acc": prop_results.get("skmeans_acc", 0.0),
                    "ari": prop_results.get("skmeans_ari", 0.0),
                    "nmi": prop_results.get("skmeans_nmi", 0.0),
                },
                index=[0],
            )

            results_list.append(df_prop)

            # Hierarchical Clustering 결과도 추가
            df_prop_hc = pd.DataFrame(
                {
                    "approach": f"ProposedGAE-HC(graph={args.knn_graph_method}, hnr={config['proposed']['hard_negative_ratio']})",
                    "acc": prop_results.get("hc_acc", 0.0),
                    "ari": prop_results.get("hc_ari", 0.0),
                    "nmi": prop_results.get("hc_nmi", 0.0),
                },
                index=[0],
            )
            results_list.append(df_prop_hc)

            # 결과 저장
            save_path = os.path.join(
                config["save_dir"],
                f"results_{args.dataset}_proposed_{args.knn_graph_method}_hnr{config['proposed']['hard_negative_ratio']}.csv",
            )
            ensure_dir_exists(save_path)
            df_prop.to_csv(save_path, index=False)

            print("\n===== Proposed GAE Results (Spherical K-means) =====")
            print(df_prop.to_string(index=False))
            print("\n===== Proposed GAE Results (Hierarchical Clustering) =====")
            print(df_prop_hc.to_string(index=False))
            print(f"[Info] Results saved to: {save_path}")
        else:
            print("[Error] Proposed GAE training failed or returned no embeddings.")
    else:
        print("\n[Info] Skipping Proposed GAE training as configured.")

    # 종합 결과 저장 (일자와 시간 포함)
    if results_list:
        all_results = pd.concat(results_list, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(
            config["save_dir"],
            f"summary_{args.dataset}_{args.knn_graph_method}_{timestamp}.csv",
        )
        ensure_dir_exists(summary_path)
        all_results.to_csv(summary_path, index=False)
        print(f"\n[Info] Summary results saved to: {summary_path}")

        # 실험 매니저에 결과 저장
        experiment_params = get_experiment_params(args)
        results_dict = {}
        for idx, row in all_results.iterrows():
            approach = row["approach"]
            if "StandardGAE" in approach:
                results_dict["gae_results"] = {
                    "acc": row["acc"],
                    "ari": row["ari"],
                    "nmi": row["nmi"],
                }
            elif "ProposedGAE" in approach:
                results_dict["proposed_results"] = {
                    "acc": row["acc"],
                    "ari": row["ari"],
                    "nmi": row["nmi"],
                }
            elif "DirectClustering" in approach:
                results_dict["direct_results"] = {
                    "acc": row["acc"],
                    "ari": row["ari"],
                    "nmi": row["nmi"],
                }

        cache_manager.save_experiment_results(experiment_params, results_dict)
    else:
        print("[Warning] No results to save.")

    # 전체 클러스터링 시간 출력
    total_end_time = time.time()
    total_clustering_time = total_end_time - total_start_time
    print(f"\nTotal Clustering Time: {total_clustering_time:.2f} seconds")
    print(f"Total Clustering Time: {total_clustering_time/60:.2f} minutes")

    # 결과 시각화
    if results_list:
        visualize_path = os.path.join(
            config["save_dir"], f"clustering_results_{args.dataset}_{timestamp}.png"
        )
        try:
            # 결과를 Dictionary 형태로 변환
            results_dict = {}
            for idx, row in all_results.iterrows():
                results_dict[row["approach"]] = {
                    "acc": row["acc"],
                    "ari": row["ari"],
                    "nmi": row["nmi"],
                }

            plot_results(results_dict, visualize_path)
            print(f"[Info] Visualization saved to: {visualize_path}")
        except Exception as e:
            print(f"[Warning] Failed to create visualization: {e}")

    return results_list


if __name__ == "__main__":
    main()
