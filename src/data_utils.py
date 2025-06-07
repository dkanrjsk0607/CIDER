# src/data_utils.py
import os
import random
import numpy as np
import torch
import nltk

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK 다운로드 보장
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    print("[Info] Downloading NLTK data (punkt, stopwords)...")
    nltk.download("punkt")
    nltk.download("stopwords")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_text(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] Text file not found: {file_path}")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        docs = f.readlines()
    docs = [doc.strip() for doc in docs if doc.strip()]
    return docs


def read_label(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] Label file not found: {file_path}")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        docs = f.readlines()
    y_true = [int(doc.strip()) for doc in docs if doc.strip()]
    return np.array(y_true, dtype=np.int64)


def preprocess_text(texts):
    """간단한 전처리: 소문자화, 숫자 제거, 토큰화, stopwords/짧은 단어 제거"""
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        print("[Warning] NLTK stopwords not found. No stopword removal.")
        stop_words = set()

    processed_texts = []
    for text in tqdm(texts, desc="Preprocessing text"):
        text = text.lower()
        text = "".join([i for i in text if not i.isdigit()])  # 숫자 제거
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [
                w for w in tokens if w not in stop_words and len(w) > 3 and w.isalpha()
            ]
        except LookupError:
            tokens = text.split()
            filtered_tokens = [w for w in tokens if len(w) > 3 and w.isalpha()]
        processed_texts.append(" ".join(filtered_tokens))
    return processed_texts
