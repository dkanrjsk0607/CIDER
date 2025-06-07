# src/embeddings.py
import os
import pickle
import numpy as np
from typing import Tuple

try:
    from sentence_transformers import SentenceTransformer

    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("[Warning] sentence-transformers not found.")

try:
    from transformers import ViTFeatureExtractor, ViTModel, AutoTokenizer, AutoModel
    import PIL
    import torch

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[Warning] transformers or PIL not found.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
import torch


def get_tfidf_embeddings(
    texts,
    tfidf_method="standard",
    max_features=None,
    threshold=0.9,
    save_dir=".",
    dataset_name="dataset",
    use_cache=True,
):
    """TF-IDF 임베딩 계산 + 캐싱"""
    # 임베딩 저장 경로 설정
    cache_filename = f"tfidf_emb_{tfidf_method}_mf{max_features}_th{threshold}.pkl"
    cache_path = os.path.join(save_dir, cache_filename)

    if use_cache and os.path.exists(cache_path):
        print(f"[CACHE] Loading TF-IDF from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Calculating TF-IDF (method={tfidf_method})...")
    if not texts:
        print("[Error] No texts to compute TF-IDF.")
        return None

    if tfidf_method == "threshold":
        # 기존 threshold 방식
        vectorizer_full = TfidfVectorizer(max_features=max_features)
        tfidf_matrix_full = vectorizer_full.fit_transform(texts).toarray()
        length = tfidf_matrix_full.shape[0]

        docu_word_len = [
            len(tfidf_matrix_full[i][tfidf_matrix_full[i] > 0]) for i in range(length)
        ]
        w_x_tfidf = []
        for i in range(length):
            non_zero_indices = tfidf_matrix_full[i].nonzero()[0]
            sorted_indices = non_zero_indices[
                np.argsort(tfidf_matrix_full[i, non_zero_indices])[::-1]
            ]
            top_vals = tfidf_matrix_full[i, sorted_indices]
            w_x_tfidf.append(top_vals)

        for i in range(length):
            s = w_x_tfidf[i].sum()
            if s > 1e-9:
                w_x_tfidf[i] = w_x_tfidf[i] / s
            else:
                w_x_tfidf[i] = np.zeros_like(w_x_tfidf[i])

        countt = []
        for i in range(length):
            count = 0
            summ = 0
            for val in w_x_tfidf[i]:
                if summ < threshold:
                    summ += val
                    count += 1
                else:
                    break
            countt.append(max(1, count) if len(w_x_tfidf[i]) > 0 else 0)

        tt = []
        for i in range(length):
            non_zero_indices = tfidf_matrix_full[i].nonzero()[0]
            if len(non_zero_indices) == 0:
                tt.append("")
                continue
            sorted_original_indices = non_zero_indices[
                np.argsort(tfidf_matrix_full[i, non_zero_indices])[::-1]
            ]
            top_indices = sorted_original_indices[: countt[i]]
            ss = " ".join(str(idx) for idx in top_indices)
            tt.append(ss)

        vectorizer_threshold = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        try:
            features_tfidf = vectorizer_threshold.fit_transform(tt).toarray()
        except ValueError as e:
            print(f"[Warning] Fallback to standard TF-IDF due to error: {e}")
            vectorizer = TfidfVectorizer(max_features=max_features)
            features_tfidf = vectorizer.fit_transform(texts).toarray()

    else:
        # standard TF-IDF
        vectorizer = TfidfVectorizer(max_features=max_features)
        features_tfidf = vectorizer.fit_transform(texts).toarray()

    os.makedirs(save_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(features_tfidf, f)
    print(f"[CACHE] Saved TF-IDF to {cache_path}")
    return features_tfidf


def get_pretrained_text_embeddings(
    texts,
    model_name,
    model_type,
    device,
    save_dir=".",
    dataset_name="dataset",
    use_cache=True,
):
    """텍스트 임베딩 계산 + 캐싱 (SentenceTransformer 또는 Hugging Face Transformers)"""
    # 임베딩 저장 경로 설정
    model_tag = model_name.replace("/", "_")
    cache_filename = f"pretrained_text_emb_{model_type}_{model_tag}.pkl"
    cache_path = os.path.join(save_dir, cache_filename)

    if use_cache and os.path.exists(cache_path):
        print(f"[CACHE] Loading Text Embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Calculating text embeddings using {model_type} (model={model_name})...")
    if not texts:
        print("[Error] No texts for pretrained embeddings.")
        return None

    if model_type == "sentence_transformer":
        if not HAS_ST:
            print("[Error] SentenceTransformers not installed.")
            return None

        try:
            model = SentenceTransformer(model_name, device=device)
            # SentenceTransformer는 기본적으로 정규화 지원
            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
                device=device,
            )
        except Exception as e:
            print(
                f"[Error] Failed to encode with SentenceTransformer {model_name}: {e}"
            )
            return None

    elif model_type == "transformers":
        if not HAS_TRANSFORMERS:
            print("[Error] transformers not installed.")
            return None

        try:
            # RoBERTa 모델이나 다른 Hugging Face 모델 로드
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()

            # 배치 처리 및 평균 풀링으로 임베딩 계산
            batch_size = 16
            embeddings = []

            for i in tqdm(
                range(0, len(texts), batch_size), desc=f"Encoding with {model_name}"
            ):
                batch_texts = texts[i : i + batch_size]

                # 토큰화 및 패딩
                encoded_input = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)

                # 임베딩 계산
                with torch.no_grad():
                    model_output = model(**encoded_input)

                    # 마지막 히든 스테이트 사용 (RoBERTa 등은 CLS 토큰 또는 평균 풀링)
                    # 여기서는 평균 풀링 방식 사용
                    token_embeddings = model_output.last_hidden_state

                    # attention_mask를 활용한 평균 풀링
                    attention_mask = encoded_input["attention_mask"].unsqueeze(-1)
                    sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                    sum_mask = torch.sum(attention_mask, dim=1)
                    batch_embeddings = sum_embeddings / sum_mask

                    # CPU로 이동 및 변환
                    batch_embeddings = batch_embeddings.cpu().numpy()

                embeddings.append(batch_embeddings)

            # 전체 임베딩 병합
            embeddings = np.vstack(embeddings)

        except Exception as e:
            print(f"[Error] Failed to encode with transformers {model_name}: {e}")
            return None
    else:
        print(f"[Error] Unknown model type: {model_type}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"[CACHE] Saved text embeddings to {cache_path}")
    return embeddings


def get_pretrained_image_embeddings(
    images,
    model_name,
    device,
    save_dir=".",
    dataset_name="dataset",
    use_cache=True,
):
    """ViT나 ResNet 같은 사전 학습 모델을 사용한 이미지 임베딩 계산 + 캐싱"""
    if not HAS_TRANSFORMERS:
        print("[Error] transformers/PIL not available for image feature extraction.")
        return None

    # 임베딩 저장 경로 설정
    model_tag = model_name.replace("/", "_")
    cache_filename = f"pretrained_image_emb_{model_tag}.pkl"
    cache_path = os.path.join(save_dir, cache_filename)

    if use_cache and os.path.exists(cache_path):
        print(f"[CACHE] Loading Image Embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Calculating image embeddings using model={model_name}...")
    if not images:
        print("[Error] No images for pretrained embeddings.")
        return None

    try:
        # ViT 모델과 ResNet 모델을 분리하여 처리
        if "vit" in model_name.lower():
            # ViT 모델 처리
            feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            model = ViTModel.from_pretrained(model_name).to(device)
            model.eval()
            
            # ViT 모델의 임베딩 추출
            all_embs = []
            batch_size = 32
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size), desc="Processing images"):
                    batch_imgs = images[i : i + batch_size]
                    pil_batch = convert_batch_to_pil(batch_imgs)
                    
                    if not pil_batch:
                        continue

                    inputs = feature_extractor(images=pil_batch, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state[:, 0, :]  # CLS 토큰
                    all_embs.append(emb.cpu().numpy())
        
        elif "resnet" in model_name.lower():
            # ResNet 모델 처리
            from transformers import AutoFeatureExtractor, ResNetModel
            
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = ResNetModel.from_pretrained(model_name).to(device)
            model.eval()
            
            # ResNet 모델의 임베딩 추출
            all_embs = []
            batch_size = 32
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size), desc="Processing images with ResNet"):
                    batch_imgs = images[i : i + batch_size]
                    pil_batch = convert_batch_to_pil(batch_imgs)
                    
                    if not pil_batch:
                        continue
                    
                    inputs = feature_extractor(images=pil_batch, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    
                    # ResNet 출력 처리 개선
                    # last_hidden_state가 [batch_size, channels, height, width] 형태일 수 있음
                    if hasattr(outputs, 'last_hidden_state') and len(outputs.last_hidden_state.shape) == 4:
                        # 공간 차원에 대해 평균 풀링
                        emb = outputs.last_hidden_state.mean(dim=(2, 3))  # [batch_size, channels]
                    elif hasattr(outputs, 'pooler_output'):
                        emb = outputs.pooler_output
                    else:
                        # 다른 출력 형태
                        if hasattr(outputs, 'last_hidden_state'):
                            hidden = outputs.last_hidden_state
                            if len(hidden.shape) == 4:  # [batch_size, channels, height, width]
                                emb = hidden.mean(dim=(2, 3))
                            elif len(hidden.shape) == 3:  # [batch_size, seq_length, hidden_size]
                                emb = hidden.mean(dim=1)
                            else:
                                emb = hidden
                        else:
                            raise ValueError(f"Unknown output format for ResNet model. Output keys: {outputs.keys()}")
                    
                    all_embs.append(emb.cpu().numpy())
        
        else:
            # 그 외 모델은 일반적인 AutoModel 사용
            from transformers import AutoFeatureExtractor, AutoModel
            
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()
            
            # 임베딩 추출
            all_embs = []
            batch_size = 32
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size), desc=f"Processing images with {model_name}"):
                    batch_imgs = images[i : i + batch_size]
                    pil_batch = convert_batch_to_pil(batch_imgs)
                    
                    if not pil_batch:
                        continue
                    
                    inputs = feature_extractor(images=pil_batch, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    # 모델의 출력 형식에 따라 적절한 임베딩 추출
                    if hasattr(outputs, 'pooler_output'):
                        emb = outputs.pooler_output
                    elif hasattr(outputs, 'last_hidden_state'):
                        last_hidden = outputs.last_hidden_state
                        # 4D 텐서인 경우 공간 차원에 대해 평균 계산
                        if len(last_hidden.shape) == 4:  # [batch_size, channels, height, width]
                            emb = last_hidden.mean(dim=(2, 3))
                        else:  # [batch_size, seq_length, hidden_size]
                            emb = last_hidden.mean(dim=1)  # 평균 풀링
                    else:
                        raise ValueError(f"Unknown output format for model {model_name}")
                    all_embs.append(emb.cpu().numpy())
    
    except Exception as e:
        print(f"[Error] Failed to extract image embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None

    if not all_embs:
        print("[Error] No embeddings generated.")
        return None

    # 모든 임베딩 결합
    embeddings = np.concatenate(all_embs, axis=0)
    
    # 차원 확인 및 정규화
    print(f"Embeddings shape before normalization: {embeddings.shape}")
    
    # 2D 배열인지 확인
    if len(embeddings.shape) > 2:
        print(f"[Warning] Unexpected embedding shape: {embeddings.shape}. Reshaping...")
        # 첫 번째 차원은 배치 크기이므로 유지하고 나머지를 평탄화
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        print(f"Reshaped to: {embeddings.shape}")
    
    # L2 정규화
    from sklearn.preprocessing import normalize
    embeddings = normalize(embeddings, axis=1, norm="l2")
    print(f"Final embeddings shape: {embeddings.shape}")

    os.makedirs(save_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"[CACHE] Saved image embeddings to {cache_path}")
    return embeddings


def convert_batch_to_pil(batch_imgs):
    """텐서, NumPy 배열 또는 다른 형식의 이미지를 PIL Image로 변환"""
    pil_batch = []
    for img in batch_imgs:
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL Image
            if img.dim() == 3 and img.shape[0] in (1, 3):  # CHW format
                img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC
                if img.shape[0] == 1:  # Grayscale to RGB
                    img_np = np.repeat(img_np, 3, axis=2)
                # Ensure proper value range [0, 255]
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                pil_batch.append(PIL.Image.fromarray(img_np.astype(np.uint8)))
            else:
                print(
                    f"[Warning] Unexpected tensor shape: {img.shape}, skipping"
                )
                continue
        elif isinstance(img, PIL.Image.Image):
            pil_batch.append(img)
        elif isinstance(img, np.ndarray):
            # 이미지가 HWC 형식이라면 그대로 변환, CHW 형식이라면 변환
            if len(img.shape) == 3 and img.shape[0] == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))
            # 값 범위가 [0, 1]이라면 [0, 255]로 변환
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            pil_batch.append(PIL.Image.fromarray(img.astype(np.uint8)))
        else:
            print(f"[Warning] Unsupported image type: {type(img)}, skipping")
            continue
    
    return pil_batch


def calculate_cosine_similarity_matrix(feature_matrix):
    """문서 임베딩 간 코사인 유사도 행렬 계산"""
    if feature_matrix is None or feature_matrix.shape[0] == 0:
        return None
    sim = cosine_similarity(feature_matrix)
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


def prepare_normalized_dist_for_graph(
    tfidf_sim: np.ndarray, pretrained_sim: np.ndarray, omega: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TF-IDF와 Pretrained 임베딩의 유사도 행렬을 결합하여 정규화된 거리 행렬 생성

    Args:
        tfidf_sim: TF-IDF 유사도 행렬
        pretrained_sim: Pretrained 유사도 행렬
        omega: TF-IDF 가중치 (0~1)

    Returns:
        dist_normalized: 정규화된 거리 행렬
        o_dist_original: 원본 거리 행렬
    """

    # 가중 평균으로 결합
    o_dist_original = omega * tfidf_sim + (1 - omega) * pretrained_sim

    # 정규화
    dist_normalized = (o_dist_original - o_dist_original.min()) / (
        o_dist_original.max() - o_dist_original.min()
    )

    return dist_normalized, o_dist_original
