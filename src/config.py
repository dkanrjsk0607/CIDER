# src/config.py
import argparse
import os
from typing import Dict, Any

# 전역 설정 딕셔너리
config = {
    "random_seed": 42,
    "data_path": {
        "text": "./data/20news.txt",
        "label": "./data/20news_label.txt",
        "image_dir": "./data/images",
    },
    "save_dir": "./results/20newsgroup_new",
    "cache": {
        "base_dir": "./cache",
        "stages": {
            "embeddings": {
                "tfidf": {
                    "method": "threshold",
                    "max_features": None,
                    "threshold": 0.9,
                },
                "pretrained": {
                    "text_model": "BAAI/bge-large-en-v1.5",
                    "text_model_type": "sentence_transformer",
                    "image_model": "google/vit-base-patch16-224-in21k",
                },
            },
            "graph": {
                "method": "tfidf",  # tfidf, pretrained_text, pretrained_image, reranked_tfidf_text
                "alpha": 0.2,
                "max_connections": 20,
            },
            "gae": {
                "hidden_dim": 256,
                "out_dim": 64,
                "learning_rate": 0.01,
                "epochs": 1000,
                "eval_interval": 10,
                "early_stopping": 100,
                "init_feature": "combined",  # onehot, sparse, dense, combined
            },
            "proposed": {
                "channels1": 256,
                "channels2": 64,
                "lr": 0.01,
                "epochs": 1000,
                "eval_interval": 10,
                "early_stopping": 100,
                "lambda_factor": 1.0,
                "iqr_k": 1.0,
                "use_scheduler": True,
                "hard_negative_ratio": 0.5,
                "use_k_hop_negatives": True,
                "k_hop_values": [3, 4],
            },
        },
        "clustering": {
            "k_values": 20,
            "n_init": 20,
        },
        "ablation": {
            "skip_threshold_tfidf": False,
            "skip_graph_construction": False,
            "skip_gae": True,
            "skip_proposed": False,
        },
    },
}

# 데이터셋 이름별 경로
DATASET_PATHS = {
    "20newsgroup": {
        "text": "./data/20newsgroup.txt",
        "label": "./data/20newsgroup_labels.txt",
    },
    "agnews": {
        "text": "./data/agnews.txt",
        "label": "./data/agnews_labels.txt",
    },
    "bbc": {
        "text": "./data/bbc.txt",
        "label": "./data/bbc_labels.txt",
    },
    "googlenews-t": {
        "text": "./data/googlenews-t.txt",
        "label": "./data/googlenews-t_labels.txt",
    },
    "reuters8": {
        "text": "./data/reuters8.txt",
        "label": "./data/reuters8_labels.txt",
    },
    "searchsnippets": {
        "text": "./data/searchsnippets.txt",
        "label": "./data/searchsnippets_labels.txt",
    },
    "stackoverflow": {
        "text": "./data/stackoverflow.txt",
        "label": "./data/stackoverflow_labels.txt",
    },
    "webkb": {
        "text": "./data/webkb.txt",
        "label": "./data/webkb_labels.txt",
    },
    # 이미지 데이터셋 추가
    "stl10": {
        "image_dir": "stl10",
        "n_clusters": 10,
    },
    "cifar10": {
        "image_dir": "cifar10",
        "n_clusters": 10,
    },
    "cifar100-20": {
        "image_dir": "cifar100-20",
        "n_clusters": 20,
    },
    "imagenet10": {
        "image_dir": "imagenet10",
        "n_clusters": 10,
    },
    "imagenet-dogs": {
        "image_dir": "imagenet-dogs",
        "n_clusters": 120,
    },
    "tiny-imagenet": {
        "image_dir": "tiny-imagenet",
        "n_clusters": 200,
    },
}

# 이미지 데이터셋 정보
IMAGE_DATASETS = {
    "stl10": {
        "n_clusters": 10,
        "description": "STL-10 (10 classes, 96x96 images)",
    },
    "cifar10": {
        "n_clusters": 10,
        "description": "CIFAR-10 (10 classes, 32x32 images)",
    },
    "cifar100-20": {
        "n_clusters": 20,
        "description": "CIFAR-100 with 20 superclasses",
    },
    "imagenet10": {
        "n_clusters": 10,
        "description": "ImageNet-10 (10 classes subset of ImageNet)",
    },
    "imagenet-dogs": {
        "n_clusters": 15,
        "description": "Subclasses of ImageNet-1000 (15 classes)",
    },
    "tiny-imagenet": {
        "n_clusters": 200,
        "description": "Tiny-ImageNet (200 classes, 64x64 images)",
    },
}


def is_image_dataset(dataset_name):
    """해당 데이터셋이 이미지 데이터셋인지 확인"""
    return dataset_name.lower() in IMAGE_DATASETS


def get_dataset_clusters(dataset_name):
    """데이터셋의 클러스터 수 반환"""
    # 이미지 데이터셋
    if is_image_dataset(dataset_name):
        return IMAGE_DATASETS[dataset_name.lower()]["n_clusters"]

    # 텍스트 데이터셋
    if dataset_name in DATASET_PATHS:
        return None  # 레이블에서 자동 추론

    return 20  # 기본값


def parse_args():
    """
    명령행 인자 파싱
    """
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
        "--dataset", type=str, default="searchsnippets", help="Dataset name"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters (if None, inferred from labels)",
    )
    parser.add_argument(
        "--knn_graph_method",
        type=str,
        default="reranked_tfidf_text",
        choices=[
            "tfidf",
            "pretrained_text",
            "pretrained_image",
            "reranked_tfidf_text",
            "reranked_tfidf_image",
        ],
        help="KNN graph construction method",
    )
    parser.add_argument(
        "--tfidf_method",
        type=str,
        default="threshold",
        choices=["standard", "threshold"],
        help="TF-IDF vectorization method",
    )
    parser.add_argument(
        "--pretrained_text_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Pretrained text model name",
    )
    parser.add_argument(
        "--pretrained_text_model_type",
        type=str,
        default="sentence_transformer",
        choices=["sentence_transformer", "transformers"],
        help="Pretrained text model type",
    )
    parser.add_argument(
        "--pretrained_image_model",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="Pretrained image model name",
    )
    parser.add_argument(
        "--gae_init_feature",
        type=str,
        default="combined",
        choices=["onehot", "sparse", "dense", "combined"],
        help="Initial feature type for GAE",
    )
    parser.add_argument(
        "--hard_negative_ratio",
        type=float,
        default=0.0,
        help="Ratio of hard negatives for Proposed GAE",
    )
    parser.add_argument(
        "--skip_gae",
        action="store_true",
        help="Skip standard GAE training",
    )
    parser.add_argument(
        "--skip_proposed",
        action="store_true",
        help="Skip proposed GAE training",
    )
    parser.add_argument(
        "--direct_clustering",
        action="store_true",
        help="Only perform direct clustering without GAE",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True,
        help="Use cached results if available",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from datasets",
    )
    return parser.parse_args()


def get_experiment_params(args) -> dict:
    """
    현재 실험의 파라미터를 딕셔너리로 반환

    Args:
        args: 명령행 인자

    Returns:
        실험 파라미터 딕셔너리
    """
    return {
        "dataset": args.dataset,
        "n_clusters": (
            args.n_clusters
            if args.n_clusters is not None
            else get_dataset_clusters(args.dataset)
        ),
        "embeddings": {
            "tfidf": {
                "method": args.tfidf_method,
                "max_features": config["cache"]["stages"]["embeddings"]["tfidf"][
                    "max_features"
                ],
                "threshold": config["cache"]["stages"]["embeddings"]["tfidf"][
                    "threshold"
                ],
            },
            "pretrained": {
                "text_model": args.pretrained_text_model,
                "text_model_type": args.pretrained_text_model_type,
                "image_model": args.pretrained_image_model,
            },
        },
        "graph": {
            "method": args.knn_graph_method,
            "alpha": config["cache"]["stages"]["graph"]["alpha"],
            "max_connections": config["cache"]["stages"]["graph"]["max_connections"],
        },
        "gae": {
            "hidden_dim": config["cache"]["stages"]["gae"]["hidden_dim"],
            "out_dim": config["cache"]["stages"]["gae"]["out_dim"],
            "learning_rate": config["cache"]["stages"]["gae"]["learning_rate"],
            "epochs": config["cache"]["stages"]["gae"]["epochs"],
            "eval_interval": config["cache"]["stages"]["gae"]["eval_interval"],
            "early_stopping": config["cache"]["stages"]["gae"]["early_stopping"],
            "init_feature": args.gae_init_feature,
        },
        "proposed": config["cache"]["stages"]["proposed"],
        "clustering": config["cache"]["clustering"],
        "ablation": {
            "skip_threshold_tfidf": config["cache"]["ablation"]["skip_threshold_tfidf"],
            "skip_graph_construction": config["cache"]["ablation"][
                "skip_graph_construction"
            ],
            "skip_gae": args.skip_gae,
            "skip_proposed": args.skip_proposed,
        },
        "max_samples": args.max_samples,
    }
