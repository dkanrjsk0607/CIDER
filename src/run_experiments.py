import os
import argparse
import sys
from typing import Dict, Any, List, Tuple
import itertools
import torch
import numpy as np
from datetime import datetime
import yaml
import pickle
import gc
from tqdm import tqdm

from src.experiment_manager import ExperimentManager
from src.config import DATASET_PATHS
from src.data_utils import read_text, read_label, preprocess_text, set_seed
from src.embeddings import (
    get_tfidf_embeddings,
    get_pretrained_text_embeddings,
    calculate_cosine_similarity_matrix,
    prepare_normalized_dist_for_graph,
)
from src.graph_utils import construct_graph_from_dist
from src.models.gae import train_gae
from src.models.proposed_gae import train_improved_proposed_gae
from src.clustering_eval import evaluate_clustering


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """실험 설정 파일 로드"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_experiment_combinations(config_path: str) -> List[Dict[str, Any]]:
    """실험할 하이퍼파라미터 조합 생성"""
    print(f"Loading configuration from: {config_path}")
    config = load_experiment_config(config_path)
    base_params = config["base_params"]
    experiments = config["experiments"]

    experiment_combinations = []
    seen_params = set()  # 중복 파라미터 추적

    # 수정: GAE 및 Proposed GAE 파라미터 조합을 모두 생성
    # 이전 코드에서는 GAE와 Proposed GAE 파라미터를 다른 실험 조합으로 처리했으나,
    # test.yaml에서는 이들을 함께 처리해야 함

    # YAML 파일 디버깅 출력
    print("\nConfiguration loaded:")
    print(f"Base params: {base_params}")
    print(f"Experiments: {experiments}")

    # Proposed GAE 파라미터 조합 생성
    if experiments.get("proposed", {}).get("enabled", False):
        proposed_params_list = []

        # 파라미터 목록 출력
        print("\nProposed GAE parameters:")
        for key, values in experiments["proposed"]["params"].items():
            print(f"  {key}: {values}")

        # 모든 조합 생성
        keys = experiments["proposed"]["params"].keys()
        values = experiments["proposed"]["params"].values()

        # 확인: 모든 파라미터가 목록인지 확인
        for key, value in zip(keys, values):
            if not isinstance(value, list):
                print(f"Warning: Parameter '{key}' is not a list: {value}")
                if isinstance(value, (int, float, str, bool)):
                    # 단일 값을 리스트로 변환
                    experiments["proposed"]["params"][key] = [value]

        # 다시 가져오기 (수정된 경우를 위해)
        values = experiments["proposed"]["params"].values()

        # 조합 생성
        for params in itertools.product(*values):
            proposed_param = dict(zip(keys, params))
            # 고정 파라미터 추가
            for key, value in experiments["proposed"].get("fixed_params", {}).items():
                proposed_param[key] = value
            proposed_params_list.append(proposed_param)

        print(f"Generated {len(proposed_params_list)} proposed parameter combinations")

        # 그래프 파라미터 조합
        graph_params_list = []

        if experiments.get("graph", {}).get("enabled", False):
            graph_method = experiments["graph"].get("method", "reranked_tfidf_text")
            # 그래프 파라미터 출력
            print("\nGraph parameters:")
            for key, values in experiments["graph"]["params"].items():
                print(f"  {key}: {values}")

            # 알파, K, 오메가 파라미터 조합 생성
            alphas = experiments["graph"]["params"].get("alpha", [0.2])
            max_connections = experiments["graph"]["params"].get(
                "max_connections", [20]
            )
            omegas = experiments["graph"]["params"].get("omega", [0.5])

            # 확인: 리스트가 아닌 경우 변환
            if not isinstance(alphas, list):
                alphas = [alphas]
            if not isinstance(max_connections, list):
                max_connections = [max_connections]
            if not isinstance(omegas, list):
                omegas = [omegas]

            print(f"  Alphas: {alphas}")
            print(f"  Max connections: {max_connections}")
            print(f"  Omegas: {omegas}")

            # 그래프 파라미터 조합 생성
            for alpha, k, omega in itertools.product(alphas, max_connections, omegas):
                graph_params = {
                    "method": graph_method,
                    "alpha": alpha,
                    "max_connections": k,
                    "omega": omega,
                }
                graph_params_list.append(graph_params)

            print(f"Generated {len(graph_params_list)} graph parameter combinations")
        else:
            # 그래프 실험이 비활성화된 경우 base_params의 graph 사용
            graph_params_list.append(base_params.get("graph", {}))

        # 모든 조합 생성
        for graph_params in graph_params_list:
            for proposed_params in proposed_params_list:
                # 기본 파라미터 복사
                exp_params = base_params.copy()
                # 그래프 파라미터 추가
                exp_params["graph"] = graph_params
                # proposed 파라미터 추가
                exp_params["proposed"] = proposed_params
                # GAE 패스 설정
                exp_params["skip_gae"] = base_params.get("skip_gae", True)
                # Proposed GAE 활성화
                exp_params["skip_proposed"] = False

                # 파라미터 해시 생성 (중복 확인용)
                param_hash = hash(
                    str(sorted([(k, str(v)) for k, v in exp_params.items()]))
                )
                if param_hash not in seen_params:
                    seen_params.add(param_hash)
                    experiment_combinations.append(exp_params)

    # 실험 조합 확인
    print(f"\nGenerated {len(experiment_combinations)} total experiment combinations")

    # 상세 실험 조합 출력
    for i, exp in enumerate(experiment_combinations):
        print(f"\nExperiment {i+1}:")
        print(f"  Dataset: {exp.get('dataset')}")
        print(f"  Skip GAE: {exp.get('skip_gae', True)}")
        print(f"  Skip Proposed: {exp.get('skip_proposed', True)}")
        print(f"  Graph method: {exp.get('graph', {}).get('method')}")
        print(
            f"  Graph params: alpha={exp.get('graph', {}).get('alpha')}, "
            f"max_connections={exp.get('graph', {}).get('max_connections')}, "
            f"omega={exp.get('graph', {}).get('omega')}"
        )
        if "proposed" in exp:
            proposed = exp["proposed"]
            print(
                f"  Proposed: channels1={proposed.get('channels1')}, "
                f"channels2={proposed.get('channels2')}, "
                f"lr={proposed.get('lr')}, "
                f"lambda_factor={proposed.get('lambda_factor')}, "
                f"hard_negative_ratio={proposed.get('hard_negative_ratio')}"
            )

    # 메모리 정리
    seen_params.clear()
    gc.collect()

    return experiment_combinations


def prepare_directories(dataset_name: str) -> Dict[str, str]:
    """저장 디렉토리 구조 준비"""
    base_dir = os.path.join("results", dataset_name)
    stage_dirs = {
        "embeddings": os.path.join(base_dir, "embeddings"),
        "graph": os.path.join(base_dir, "graph"),
        "gae": os.path.join(base_dir, "gae"),
        "proposed": os.path.join(base_dir, "proposed"),
        "logs": os.path.join(base_dir, "logs"),
    }

    # 디렉토리 생성
    for dir_path in stage_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return stage_dirs


def load_dataset(dataset_name: str) -> Tuple[List[str], np.ndarray, int]:
    """데이터셋과 레이블 로드"""
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_paths = DATASET_PATHS[dataset_name]
    texts = read_text(dataset_paths["text"])
    labels = read_label(dataset_paths["label"])

    if len(texts) != len(labels):
        raise ValueError(
            f"Number of texts ({len(texts)}) does not match number of labels ({len(labels)})"
        )

    # 클러스터 수 결정
    n_clusters = len(np.unique(labels))
    print(f"Dataset loaded: {len(texts)} documents, {n_clusters} clusters")

    return texts, labels, n_clusters


def load_embeddings(
    texts: List[str],
    processed_texts: List[str],
    exp_params: Dict[str, Any],
    stage_dirs: Dict[str, str],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """임베딩 로드 또는 계산"""
    embeddings = {}

    # TF-IDF 임베딩
    if "tfidf" in exp_params["embeddings"]:
        print("Processing TF-IDF embeddings...")
        tfidf_params = exp_params["embeddings"]["tfidf"]
        embeddings["tfidf"] = get_tfidf_embeddings(
            processed_texts,
            tfidf_method=tfidf_params.get("method", "standard"),
            max_features=tfidf_params.get("max_features"),
            threshold=tfidf_params.get("threshold", 0.9),
            save_dir=stage_dirs["embeddings"],
            dataset_name=exp_params["dataset"],
            use_cache=True,
        )

    # 사전학습 임베딩
    if "pretrained" in exp_params["embeddings"]:
        print("Processing pretrained embeddings...")
        pretrained_params = exp_params["embeddings"]["pretrained"]
        if "text_model" in pretrained_params:
            embeddings["pretrained_text"] = get_pretrained_text_embeddings(
                texts,  # 원본 텍스트 사용
                model_name=pretrained_params["text_model"],
                model_type=pretrained_params.get(
                    "text_model_type", "sentence_transformer"
                ),
                device=device,
                save_dir=stage_dirs["embeddings"],
                dataset_name=exp_params["dataset"],
                use_cache=True,
            )

    return embeddings


def construct_graph(
    embeddings: Dict[str, np.ndarray],
    exp_params: Dict[str, Any],
    stage_dirs: Dict[str, str],
) -> Tuple[List, List]:
    """그래프 구성"""
    graph_method = exp_params["graph"]["method"]

    if graph_method != "reranked_tfidf_text":
        raise ValueError(f"Unknown graph method: {graph_method}")

    if "tfidf" not in embeddings or "pretrained_text" not in embeddings:
        raise ValueError(
            "Both TF-IDF and pretrained text embeddings are required for reranked_tfidf_text method"
        )

    # 그래프 구성 파라미터
    omega = exp_params["graph"].get("omega", 0.5)
    K = exp_params["graph"].get("max_connections", 20)
    alpha = exp_params["graph"].get("alpha", 0.2)

    # 캐시 경로 설정
    graph_cache_path = os.path.join(
        stage_dirs["graph"],
        f"graph_info_{exp_params['dataset']}_alpha{alpha}_k{K}_omega{omega}.pkl",
    )

    # 캐시된 그래프 사용
    if os.path.exists(graph_cache_path):
        print(f"Loading cached graph from {graph_cache_path}")
        with open(graph_cache_path, "rb") as f:
            graph_info = pickle.load(f)
            return (graph_info["edge_index"], graph_info["edge_weight"])

    # 새 그래프 구성
    print("Computing similarity matrices...")
    tfidf_sim = calculate_cosine_similarity_matrix(embeddings["tfidf"])

    pretrained_sim = calculate_cosine_similarity_matrix(embeddings["pretrained_text"])

    # 정규화된 거리 행렬 계산
    print(f"Preparing normalized distance matrix (omega={omega})...")
    dist_normalized, o_dist_original = prepare_normalized_dist_for_graph(
        tfidf_sim, pretrained_sim, omega=omega
    )

    # 유사도 행렬 메모리 정리
    del tfidf_sim, pretrained_sim
    gc.collect()

    # 그래프 구성
    print(f"Constructing graph (K={K}, alpha={alpha}, omega={omega})...")
    graph = construct_graph_from_dist(
        dist_normalized,
        o_dist_original,
        K=K,
        alpha=alpha,
        cache_dir=stage_dirs["graph"],
        dataset_name=f"{exp_params['dataset']}_alpha{alpha}_k{K}_omega{omega}",
        use_cache=True,
    )

    # 거리 행렬 메모리 정리
    del dist_normalized, o_dist_original
    gc.collect()

    # 그래프 정보 캐싱
    graph_info = {
        "edge_index": graph[0],
        "edge_weight": graph[1],
        "num_edges": len(graph[0][0]) if graph[0] else 0,
        "alpha": alpha,
        "max_connections": K,
        "omega": omega,
    }
    with open(graph_cache_path, "wb") as f:
        pickle.dump(graph_info, f)
    print(f"Graph info saved to {graph_cache_path}")

    return graph


def load_gae_features(
    labels: np.ndarray,
    gae_init_feature: str,
    stage_dirs: Dict[str, str],
    exp_params: Dict[str, Any],
) -> torch.Tensor:
    """GAE 초기 특성 로드"""
    x_gae = None

    if gae_init_feature == "onehot":
        x_gae = torch.eye(len(labels), dtype=torch.float)
    elif gae_init_feature == "tfidf":
        # TF-IDF 임베딩 로드
        tfidf_emb_path = os.path.join(
            stage_dirs["embeddings"],
            f"tfidf_emb_{exp_params['embeddings']['tfidf'].get('method', 'standard')}_"
            f"mf{exp_params['embeddings']['tfidf'].get('max_features', 'None')}_"
            f"th{exp_params['embeddings']['tfidf'].get('threshold', 0.9)}.pkl",
        )
        if os.path.exists(tfidf_emb_path):
            with open(tfidf_emb_path, "rb") as f:
                tfidf_features = pickle.load(f)
            x_gae = torch.tensor(tfidf_features, dtype=torch.float)
            del tfidf_features
            gc.collect()
    elif gae_init_feature == "pretrained_text" or gae_init_feature == "dense":
        # 사전학습 임베딩 로드
        pretrained_emb_path = os.path.join(
            stage_dirs["embeddings"],
            f"pretrained_text_emb_{exp_params['embeddings']['pretrained'].get('text_model_type', 'sentence_transformer')}_"
            f"{exp_params['embeddings']['pretrained'].get('text_model', '').replace('/', '_')}.pkl",
        )
        if os.path.exists(pretrained_emb_path):
            with open(pretrained_emb_path, "rb") as f:
                pretrained_features = pickle.load(f)
            x_gae = torch.tensor(pretrained_features, dtype=torch.float)
            del pretrained_features
            gc.collect()
    else:
        print(f"[Warning] Unknown GAE init feature: {gae_init_feature}, using one-hot.")
        x_gae = torch.eye(len(labels), dtype=torch.float)

    return x_gae


def run_standard_gae(
    graph: Tuple,
    labels: np.ndarray,
    n_clusters: int,
    exp_params: Dict[str, Any],
    stage_dirs: Dict[str, str],
    device: torch.device,
) -> Dict[str, Any]:
    """표준 GAE 학습 및 평가"""
    print("\n===== [Standard GAE Training] =====")

    # 초기 특성 로드
    gae_init_feature = exp_params.get("gae", {}).get("init_feature", "combined")
    print(f"Loading features for GAE (init_feature={gae_init_feature})...")
    x_gae = load_gae_features(labels, gae_init_feature, stage_dirs, exp_params)

    if x_gae is None:
        print("[Error] Failed to load features for GAE.")
        return None

    # GAE 학습 파라미터
    gae_params = exp_params.get("gae", {})
    gae_results = train_gae(
        x=x_gae,
        edge_index=graph[0],
        edge_weight=graph[1],
        num_nodes=len(labels),
        hidden_dim=gae_params.get("hidden_dim", 256),
        out_dim=gae_params.get("out_dim", 64),
        learning_rate=gae_params.get("learning_rate", 0.01),
        epochs=gae_params.get("epochs", 1000),
        eval_interval=gae_params.get("eval_interval", 10),
        early_stopping=gae_params.get("early_stopping", 100),
        y_true=labels,
        device=device,
        n_clusters=n_clusters,
        save_dir=stage_dirs["gae"],
        dataset_name=f"{exp_params['dataset']}_{exp_params['graph']['method']}_{gae_init_feature}",
        use_cache=True,
        random_seed=exp_params.get("random_seed", 42),
    )

    # 메모리 정리
    del x_gae
    gc.collect()

    # 결과 처리
    if gae_results and gae_results[0] is not None:
        gae_emb, loss_history = gae_results

        # 클러스터링 평가
        eval_results = evaluate_clustering(gae_emb, labels, n_clusters)

        # Spherical K-means 결과 추출
        skmeans_res = next(
            (r for r in eval_results if r["method"] == "Spherical K-means"), None
        )
        hc_res = next(
            (r for r in eval_results if r["method"] == "Hierarchical Clustering"), None
        )

        if skmeans_res:
            results = {
                "acc": skmeans_res.get("acc"),
                "ari": skmeans_res.get("ari"),
                "nmi": skmeans_res.get("nmi"),
            }

            # 결과 출력
            print("\n----- Standard GAE Results (Spherical K-means) -----")
            print(f"Accuracy: {results['acc']:.4f}")
            print(f"ARI: {results['ari']:.4f}")
            print(f"NMI: {results['nmi']:.4f}")

            # Hierarchical Clustering 결과도 추가
            if hc_res:
                hc_results = {
                    "acc": hc_res.get("acc"),
                    "ari": hc_res.get("ari"),
                    "nmi": hc_res.get("nmi"),
                }
                print("\n----- Standard GAE Results (Hierarchical) -----")
                print(f"Accuracy: {hc_results['acc']:.4f}")
                print(f"ARI: {hc_results['ari']:.4f}")
                print(f"NMI: {hc_results['nmi']:.4f}")

            return results

    return None


def run_proposed_gae(
    graph: Tuple,
    texts: List[str],
    labels: np.ndarray,
    n_clusters: int,
    exp_params: Dict[str, Any],
    stage_dirs: Dict[str, str],
    device: torch.device,
) -> Dict[str, Any]:
    """Proposed GAE 학습 및 평가"""
    print("\n===== [Proposed GAE Training] =====")

    # 현재 실험 파라미터 출력
    proposed_params = exp_params.get("proposed", {})
    print(f"Proposed GAE Parameters:")
    print(f"  channels1: {proposed_params.get('channels1')}")
    print(f"  channels2: {proposed_params.get('channels2')}")
    print(f"  lr: {proposed_params.get('lr')}")
    print(f"  lambda_factor: {proposed_params.get('lambda_factor')}")
    print(f"  hard_negative_ratio: {proposed_params.get('hard_negative_ratio')}")
    print(f"  k_hop_values: {proposed_params.get('k_hop_values')}")

    # 특성 텐서 로드
    print("Loading features for Proposed GAE...")

    # 경로 설정
    tfidf_emb_path = os.path.join(
        stage_dirs["embeddings"],
        f"tfidf_emb_{exp_params['embeddings']['tfidf'].get('method', 'standard')}_"
        f"mf{exp_params['embeddings']['tfidf'].get('max_features', 'None')}_"
        f"th{exp_params['embeddings']['tfidf'].get('threshold', 0.9)}.pkl",
    )

    pretrained_emb_path = os.path.join(
        stage_dirs["embeddings"],
        f"pretrained_text_emb_{exp_params['embeddings']['pretrained'].get('text_model_type', 'sentence_transformer')}_"
        f"{exp_params['embeddings']['pretrained'].get('text_model', '').replace('/', '_')}.pkl",
    )

    # 임베딩 로드
    x_sparse = x_dense = None

    if os.path.exists(tfidf_emb_path):
        with open(tfidf_emb_path, "rb") as f:
            tfidf_features = pickle.load(f)
        x_sparse = torch.tensor(tfidf_features, dtype=torch.float)
        del tfidf_features
        gc.collect()

    if os.path.exists(pretrained_emb_path):
        with open(pretrained_emb_path, "rb") as f:
            pretrained_features = pickle.load(f)
        x_dense = torch.tensor(pretrained_features, dtype=torch.float)
        del pretrained_features
        gc.collect()

    if x_sparse is None or x_dense is None:
        print("[Error] Could not load required embeddings for Proposed GAE.")
        return None

    # 텐서 변환
    edge_index_tensor = torch.tensor(graph[0], dtype=torch.long)
    edge_weight_tensor = None
    if graph[1] is not None and len(graph[1]) == len(graph[0][0]):
        edge_weight_tensor = torch.tensor(graph[1], dtype=torch.float)

    # ProposedGAE 파라미터 확인
    # channels1 타입 확인 및 변환
    channels1 = proposed_params.get("channels1", 256)
    if isinstance(channels1, list):
        print(f"Warning: channels1 is a list {channels1}, using first value")
        channels1 = channels1[0]

    # channels2 타입 확인 및 변환
    channels2 = proposed_params.get("channels2", 64)
    if isinstance(channels2, list):
        print(f"Warning: channels2 is a list {channels2}, using first value")
        channels2 = channels2[0]

    # lr 타입 확인 및 변환
    lr = proposed_params.get("lr", 0.01)
    if isinstance(lr, list):
        print(f"Warning: lr is a list {lr}, using first value")
        lr = lr[0]

    # lambda_factor 타입 확인 및 변환
    lambda_factor = proposed_params.get("lambda_factor", 1.0)
    if isinstance(lambda_factor, list):
        print(f"Warning: lambda_factor is a list {lambda_factor}, using first value")
        lambda_factor = lambda_factor[0]

    # hard_negative_ratio 타입 확인 및 변환
    hard_negative_ratio = proposed_params.get("hard_negative_ratio", 0.5)
    if isinstance(hard_negative_ratio, list):
        print(
            f"Warning: hard_negative_ratio is a list {hard_negative_ratio}, using first value"
        )
        hard_negative_ratio = hard_negative_ratio[0]

    # k_hop_values 타입 확인 및 변환
    k_hop_values = proposed_params.get("k_hop_values", [3, 4])
    if isinstance(k_hop_values, (int, float)):
        print(f"Warning: k_hop_values is not a list {k_hop_values}, converting to list")
        k_hop_values = [int(k_hop_values)]

    # 파라미터 로깅
    print(
        f"Using parameters - channels1: {channels1}, channels2: {channels2}, lr: {lr}, "
        f"lambda_factor: {lambda_factor}, hard_negative_ratio: {hard_negative_ratio}, "
        f"k_hop_values: {k_hop_values}"
    )

    # 캐시 키 계산 (고유한 파라미터 조합을 위한)
    cache_key = (
        f"{exp_params['dataset']}_"
        f"{exp_params['graph']['method']}_"
        f"alpha{exp_params['graph']['alpha']}_"
        f"k{exp_params['graph']['max_connections']}_"
        f"omega{exp_params['graph']['omega']}_"
        f"ch1{channels1}_"
        f"ch2{channels2}_"
        f"lr{lr}_"
        f"lambda{lambda_factor}_"
        f"hnr{hard_negative_ratio}"
    )

    # ProposedGAE 학습
    proposed_results_obj = train_improved_proposed_gae(
        x_sparse=x_sparse,
        x_dense=x_dense,
        train_pos_edge_index=edge_index_tensor,
        train_edge_weight=edge_weight_tensor,
        val_pos_edge_index=None,  # 사용하지 않음
        documents=texts,
        num_nodes=len(labels),
        y_true=labels,
        channels1=channels1,
        channels2=channels2,
        lr=lr,
        epochs=proposed_params.get("epochs", 1000),
        eval_interval=proposed_params.get("eval_interval", 10),
        early_stopping=proposed_params.get("early_stopping", 100),
        lambda_factor=lambda_factor,
        iqr_k=proposed_params.get("iqr_k", 0.5),
        max_edges_2hop=exp_params["graph"].get("max_connections", 20),
        hard_negative_ratio=hard_negative_ratio,
        n_clusters=n_clusters,
        device=device,
        use_scheduler=proposed_params.get("use_scheduler", True),
        save_dir=stage_dirs["proposed"],
        dataset_name=cache_key,
        use_cache=True,
        random_seed=exp_params.get("random_seed", 42),
        use_standard_gae=False,
        gae_init_feature=proposed_params.get("gae_init_feature", "combined"),
        use_k_hop_negatives=proposed_params.get("use_k_hop_negatives", True),
        k_hop_values=k_hop_values,
    )

    # 메모리 정리
    del x_sparse, x_dense, edge_index_tensor, edge_weight_tensor
    gc.collect()

    # 결과 처리 - 이 부분이 핵심
    if proposed_results_obj:
        print(f"[DEBUG] Proposed GAE results object type: {type(proposed_results_obj)}")
        print(
            f"[DEBUG] Proposed GAE results keys: {list(proposed_results_obj.keys()) if isinstance(proposed_results_obj, dict) else 'Not a dict'}"
        )

        # 결과 추출 - 키 이름을 정확히 매칭
        if isinstance(proposed_results_obj, dict):
            acc_value = proposed_results_obj.get("skmeans_acc")
            ari_value = proposed_results_obj.get("skmeans_ari")
            nmi_value = proposed_results_obj.get("skmeans_nmi")

            print(
                f"[DEBUG] Extracted values - ACC: {acc_value}, ARI: {ari_value}, NMI: {nmi_value}"
            )

            # NumPy 타입을 Python 기본 타입으로 변환하고 소수점 4자리로 반올림
            results = {
                "acc": float(acc_value) if acc_value is not None else None,
                "ari": float(ari_value) if ari_value is not None else None,
                "nmi": float(nmi_value) if nmi_value is not None else None,
            }

            # 결과 출력 (소수점 4자리로 포맷)
            print("\n----- Proposed GAE Results -----")
            print(
                f"Accuracy: {results.get('acc', 'N/A'):.4f}"
                if results.get("acc") is not None
                else "Accuracy: N/A"
            )
            print(
                f"ARI: {results.get('ari', 'N/A'):.4f}"
                if results.get("ari") is not None
                else "ARI: N/A"
            )
            print(
                f"NMI: {results.get('nmi', 'N/A'):.4f}"
                if results.get("nmi") is not None
                else "NMI: N/A"
            )

            return results
        else:
            print("[Error] Proposed GAE results object is not a dictionary")
            return None
    else:
        print("[Error] Proposed GAE returned None or empty results")
        return None


def run_experiment(
    exp_params: Dict[str, Any], exp_manager: ExperimentManager, device: torch.device
) -> None:
    """단일 실험 실행"""
    try:
        # 1. 디렉토리 준비
        dataset_name = exp_params["dataset"]
        stage_dirs = prepare_directories(dataset_name)

        # 2. 데이터셋 로드
        texts, labels, n_clusters = load_dataset(dataset_name)

        # 확정된 클러스터 수 (재정의된 경우)
        if exp_params.get("n_clusters") is not None:
            n_clusters = exp_params["n_clusters"]
            print(f"Using user-defined n_clusters: {n_clusters}")

        # 3. 텍스트 전처리
        print("Preprocessing text...")
        processed_texts = preprocess_text(texts)

        # 4. 임베딩 준비
        embeddings = load_embeddings(
            texts, processed_texts, exp_params, stage_dirs, device
        )

        # 5. 그래프 구성
        graph = construct_graph(embeddings, exp_params, stage_dirs)

        # 임베딩 메모리 정리 (더 이상 필요 없음)
        del embeddings, processed_texts
        gc.collect()

        # 그래프 유효성 확인
        if graph is None or (
            isinstance(graph, tuple) and (graph[0] is None or len(graph[0][0]) == 0)
        ):
            print("[Error] Failed to create graph. Edge list is empty.")
            return

        # 결과 저장 객체
        results = {}

        # 6. GAE 학습 (필요한 경우)
        if not exp_params.get("skip_gae", False):
            gae_results = run_standard_gae(
                graph, labels, n_clusters, exp_params, stage_dirs, device
            )
            if gae_results:
                results["gae_results"] = gae_results

        # 7. Proposed GAE 학습 (필요한 경우)
        if not exp_params.get("skip_proposed", True):
            proposed_results = run_proposed_gae(
                graph, texts, labels, n_clusters, exp_params, stage_dirs, device
            )
            if proposed_results:
                results["proposed_results"] = proposed_results

        # 8. 결과 저장
        exp_manager.save_experiment_results(exp_params, results)

        # 9. 메모리 정리
        del results, graph, texts, labels
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Run GAE experiments")
    parser.add_argument(
        "--clear_cache", action="store_true", help="Clear all cache before running"
    )
    parser.add_argument(
        "--clear_stage",
        type=str,
        choices=["embeddings", "graph", "gae", "proposed"],
        help="Clear cache for specific stage",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID to use. -1 for CPU."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/experiment_configs/default.yaml",
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # 랜덤 시드 설정
    set_seed(args.random_seed)

    # 실험 관리자 초기화
    exp_manager = ExperimentManager()

    # 캐시 삭제
    if args.clear_cache:
        exp_manager.clear_cache()
    elif args.clear_stage:
        exp_manager.clear_cache(args.clear_stage)

    # 디바이스 설정
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 실험 조합 생성
    print("Generating experiment combinations...")
    experiment_combinations = generate_experiment_combinations(args.config)

    # 실험 실행
    for idx, exp_params in enumerate(experiment_combinations):
        print(f"\n{'='*50}")
        print(f"Running experiment {idx+1}/{len(experiment_combinations)}")
        print(f"{'='*50}")

        try:
            run_experiment(exp_params, exp_manager, device)

            # 실험 사이에 메모리 정리
            exp_manager.clear_memory()
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during experiment {idx+1}: {e}")
            import traceback

            traceback.print_exc()

            # 오류가 있어도 계속 진행
            continue

    # 실험 결과 요약 출력
    summary_df = exp_manager.get_experiment_summary()
    print("\nExperiment Summary:")
    print(summary_df)

    # 결과를 CSV로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv_path = f"experiment_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nExperiment summary saved to {summary_csv_path}")


if __name__ == "__main__":
    main()
