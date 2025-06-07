# src/datasets/image_datasets.py
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision
import torchvision.transforms as T
import torch.utils.data as data_utils
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import Dataset, Subset
import shutil
import urllib.request
import tarfile
import zipfile
import pickle


class ImageNet10(Dataset):
    """ImageNet-10 dataset (10 classes selected from ImageNet)"""

    def __init__(self, root, split="train", transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        self.class_names = [
            "dog",
            "bird",
            "car",
            "cat",
            "deer",
            "horse",
            "monkey",
            "plane",
            "ship",
            "truck",
        ]

        self.synsets = [
            "n02084071",  # dog
            "n01503061",  # bird
            "n02958343",  # car
            "n02121808",  # cat
            "n02431122",  # deer
            "n02374451",  # horse
            "n02484322",  # monkey
            "n02691156",  # airplane
            "n04194289",  # ship
            "n04490091",  # truck
        ]

        # 항상 고정된 경로 사용: ./data/imagenet_10
        custom_dir = os.path.abspath("./data/imagenet_10")

        # 경로 설정 및 알림
        self.data_dir = custom_dir
        print(f"Using fixed ImageNet-10 directory: {custom_dir}")

        # 데이터셋 확인
        if not os.path.exists(self.data_dir):
            msg = (
                f"ImageNet-10 dataset not found at {self.data_dir}. "
                "Please ensure that the dataset is located at ./data/imagenet_10"
            )
            raise RuntimeError(msg)

        # 이미지와 레이블 로드
        self.data = []
        self.targets = []

        # 사용자 지정 구조 (라벨 이름 기반 폴더)
        folders = [
            f
            for f in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, f))
        ]

        # 폴더 이름을 클래스 인덱스에 매핑
        self.folder_to_idx = {folder: idx for idx, folder in enumerate(sorted(folders))}

        for folder in folders:
            folder_path = os.path.join(self.data_dir, folder)
            class_id = self.folder_to_idx[folder]

            image_files = [
                f
                for f in os.listdir(folder_path)
                if f.endswith((".JPEG", ".jpeg", ".jpg", ".JPG", ".png", ".PNG"))
            ]

            for img_file in image_files:
                self.data.append(os.path.join(folder_path, img_file))
                self.targets.append(class_id)

        # 사용자 정의 클래스 이름 업데이트
        self.class_names = sorted(folders)

        print(
            f"Loaded ImageNet-10 dataset with {len(self.data)} images in {len(set(self.targets))} classes"
        )
        print(f"Class mapping: {self.folder_to_idx}")

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]

        # 이미지 로드
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageNetDogs(Dataset):
    """ImageNet-Dogs dataset with 15 dog breeds"""

    def __init__(self, root, split="train", transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        # 고정된 경로 사용: ./data/imagenet-dogs
        custom_dir = os.path.abspath("./data/imagenet-dogs")

        # 경로 설정 및 알림
        self.data_dir = custom_dir
        print(f"Using fixed ImageNet-Dogs directory: {custom_dir}")

        # 정확한 synset ID와 클래스 이름 목록 (imagenet_dog.txt에서 가져옴)
        self.synsets = [
            "n02085936",  # Maltese_dog
            "n02086646",  # Blenheim_spaniel
            "n02088238",  # basset
            "n02091467",  # Norwegian_elkhound
            "n02097130",  # giant_schnauzer
            "n02099601",  # golden_retriever
            "n02101388",  # Brittany_spaniel
            "n02101556",  # clumber
            "n02102177",  # Welsh_springer_spaniel
            "n02105056",  # groenendael
            "n02105412",  # kelpie
            "n02105855",  # Shetland_sheepdog
            "n02107142",  # Doberman
            "n02110958",  # pug
            "n02112137",  # chow
        ]

        self.class_names = [
            "Maltese_dog",
            "Blenheim_spaniel",
            "basset",
            "Norwegian_elkhound",
            "giant_schnauzer",
            "golden_retriever",
            "Brittany_spaniel",
            "clumber",
            "Welsh_springer_spaniel",
            "groenendael",
            "kelpie",
            "Shetland_sheepdog",
            "Doberman",
            "pug",
            "chow",
        ]

        # 데이터셋 확인
        if not os.path.exists(self.data_dir):
            msg = (
                f"ImageNet-Dogs dataset not found at {self.data_dir}. "
                "Please ensure that the dataset is located at ../data/imagenet-dogs"
            )
            raise RuntimeError(msg)

        # 이미지와 라벨 로드
        self.data = []
        self.targets = []

        try:
            # synset ID 기반 폴더 검색
            for class_id, synset in enumerate(self.synsets):
                class_dir = os.path.join(self.data_dir, synset)

                # 클래스 디렉토리가 존재하는지 확인
                if os.path.isdir(class_dir):
                    # 이미지 파일 찾기
                    image_files = [
                        f
                        for f in os.listdir(class_dir)
                        if f.lower().endswith((".jpeg", ".jpg", ".png", ".bmp"))
                    ]

                    for img_file in image_files:
                        self.data.append(os.path.join(class_dir, img_file))
                        self.targets.append(class_id)
                else:
                    print(f"Warning: Class directory {class_dir} not found")

            self.num_classes = len(self.synsets)

            print(
                f"Loaded ImageNet-Dogs dataset with {len(self.data)} images in {len(set(self.targets))} classes"
            )

        except Exception as e:
            print(f"[Error] Failed to load ImageNet-Dogs dataset: {e}")
            import traceback

            traceback.print_exc()
            self.data = []
            self.targets = []
            self.num_classes = 0

    def __getitem__(self, index):
        if index >= len(self.data):
            print(
                f"[Error] Index {index} out of range for dataset with length {len(self.data)}"
            )
            # 빈 더미 이미지와 0 라벨 반환
            dummy_img = torch.zeros((3, 224, 224))
            return dummy_img, 0

        img_path = self.data[index]
        target = self.targets[index]

        try:
            # 이미지 로드
            img = Image.open(img_path).convert("RGB")

            if self.transform is not None:
                img = self.transform(img)
        except Exception as e:
            print(f"[Error] Failed to load image {img_path}: {e}")
            # 오류 발생 시 더미 이미지 반환
            img = torch.zeros((3, 224, 224))

        return img, target

    def __len__(self):
        return len(self.data)


def load_cifar10(root="./data", transform=None, max_samples=None):
    """Load CIFAR-10 dataset for clustering"""
    if transform is None:
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)

    # Combine train and test sets
    full_dataset = data_utils.ConcatDataset([train_dataset, test_dataset])

    # Limit the number of samples if specified
    if max_samples is not None and max_samples < len(full_dataset):
        indices = list(range(len(full_dataset)))
        np.random.shuffle(indices)
        indices = indices[:max_samples]
        full_dataset = Subset(full_dataset, indices)

    # Extract images and labels
    images, labels = [], []
    for i in range(len(full_dataset)):
        img, label = full_dataset[i]
        images.append(img)
        labels.append(label)

    return images, np.array(labels, dtype=np.int64)


def load_cifar100_20(root="./data", transform=None, max_samples=None):
    """Load CIFAR-100 dataset with 20 superclasses for clustering"""
    if transform is None:
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    train_dataset = CIFAR100(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root=root, train=False, download=True, transform=transform)

    # 파인 레이블(100개 클래스)에서 코스 레이블(20개 슈퍼클래스)을 얻기 위한 매핑 로드
    meta_file = os.path.join(root, "cifar-100-python", "meta")

    # 메타 파일이 존재하지 않을 경우 대체 방법
    if not os.path.exists(meta_file):
        print("[Warning] Meta file not found. Creating mapping manually.")
        # CIFAR-100의 각 fine label이 어떤 coarse label에 속하는지 매핑
        # 이 매핑은 CIFAR-100 문서에 따른 것입니다
        fine_to_coarse = [
            4,
            1,
            14,
            8,
            0,
            6,
            7,
            7,
            18,
            3,
            3,
            14,
            9,
            18,
            7,
            11,
            3,
            9,
            7,
            11,
            6,
            11,
            5,
            10,
            7,
            6,
            13,
            15,
            3,
            15,
            0,
            11,
            1,
            10,
            12,
            14,
            16,
            9,
            11,
            5,
            5,
            19,
            8,
            8,
            15,
            13,
            14,
            17,
            18,
            10,
            16,
            4,
            17,
            4,
            2,
            0,
            17,
            4,
            18,
            17,
            10,
            3,
            2,
            12,
            12,
            16,
            12,
            1,
            9,
            19,
            2,
            10,
            0,
            1,
            16,
            12,
            9,
            13,
            15,
            13,
            16,
            19,
            2,
            4,
            6,
            19,
            5,
            5,
            8,
            19,
            18,
            1,
            2,
            15,
            6,
            0,
            17,
            8,
            14,
            13,
        ]
    else:
        # 기존 방식대로 메타 파일에서 로드
        with open(meta_file, "rb") as f:
            meta_data = pickle.load(f, encoding="latin1")
            fine_label_names = meta_data.get("fine_label_names", [])
            coarse_label_names = meta_data.get("coarse_label_names", [])

            # fine_to_coarse 매핑이 직접 제공되지 않는 경우
            if not hasattr(meta_data, "fine_to_coarse"):
                print(
                    "[Warning] Meta file does not contain fine_to_coarse mapping. Creating mapping manually."
                )
                # CIFAR-100의 표준 매핑
                fine_to_coarse = [
                    4,
                    1,
                    14,
                    8,
                    0,
                    6,
                    7,
                    7,
                    18,
                    3,
                    3,
                    14,
                    9,
                    18,
                    7,
                    11,
                    3,
                    9,
                    7,
                    11,
                    6,
                    11,
                    5,
                    10,
                    7,
                    6,
                    13,
                    15,
                    3,
                    15,
                    0,
                    11,
                    1,
                    10,
                    12,
                    14,
                    16,
                    9,
                    11,
                    5,
                    5,
                    19,
                    8,
                    8,
                    15,
                    13,
                    14,
                    17,
                    18,
                    10,
                    16,
                    4,
                    17,
                    4,
                    2,
                    0,
                    17,
                    4,
                    18,
                    17,
                    10,
                    3,
                    2,
                    12,
                    12,
                    16,
                    12,
                    1,
                    9,
                    19,
                    2,
                    10,
                    0,
                    1,
                    16,
                    12,
                    9,
                    13,
                    15,
                    13,
                    16,
                    19,
                    2,
                    4,
                    6,
                    19,
                    5,
                    5,
                    8,
                    19,
                    18,
                    1,
                    2,
                    15,
                    6,
                    0,
                    17,
                    8,
                    14,
                    13,
                ]
            else:
                fine_to_coarse = meta_data.fine_to_coarse

    # 전체 데이터셋 결합
    # Combine train and test sets
    full_dataset = data_utils.ConcatDataset([train_dataset, test_dataset])

    # 샘플 수 제한 (지정된 경우)
    if max_samples is not None and max_samples < len(full_dataset):
        indices = list(range(len(full_dataset)))
        np.random.shuffle(indices)
        indices = indices[:max_samples]
        full_dataset = Subset(full_dataset, indices)

    # 이미지와 코스 레이블 추출
    images, labels = [], []
    for i in range(len(full_dataset)):
        img, fine_label = full_dataset[i]

        # Subset인 경우 인덱스 접근 방식 변경
        if isinstance(full_dataset, Subset):
            dataset_idx = full_dataset.indices[i]

            # 원래 데이터셋에서의 인덱스 계산
            if dataset_idx < len(train_dataset):
                # 학습 데이터셋
                original_idx = dataset_idx
                original_dataset = train_dataset
            else:
                # 테스트 데이터셋
                original_idx = dataset_idx - len(train_dataset)
                original_dataset = test_dataset

            # fine_label을 사용해 coarse_label 계산
            fine_label = original_dataset.targets[original_idx]
            coarse_label = fine_to_coarse[fine_label]
        else:
            # 직접 데이터셋 접근
            if i < len(train_dataset):
                fine_label = train_dataset.targets[i]
            else:
                fine_label = test_dataset.targets[i - len(train_dataset)]

            # 파인 레이블에서 코스 레이블 매핑
            coarse_label = fine_to_coarse[fine_label]

        images.append(img)
        labels.append(coarse_label)

    print(
        f"Loaded CIFAR-100-20 with {len(images)} images and {len(np.unique(labels))} superclasses"
    )
    return images, np.array(labels, dtype=np.int64)


def load_stl10(root="./data", transform=None, max_samples=None):
    """Load STL-10 dataset for clustering"""
    if transform is None:
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    # Load both train and test splits
    train_dataset = STL10(root=root, split="train", download=True, transform=transform)
    test_dataset = STL10(root=root, split="test", download=True, transform=transform)

    # Combine train and test sets
    full_dataset = data_utils.ConcatDataset([train_dataset, test_dataset])

    # Limit the number of samples if specified
    if max_samples is not None and max_samples < len(full_dataset):
        indices = list(range(len(full_dataset)))
        np.random.shuffle(indices)
        indices = indices[:max_samples]
        full_dataset = Subset(full_dataset, indices)

    # Extract images and labels
    images, labels = [], []
    for i in range(len(full_dataset)):
        img, label = full_dataset[i]
        images.append(img)
        labels.append(label)

    return images, np.array(labels, dtype=np.int64)


def load_imagenet10(root="./data", transform=None, max_samples=None):
    """Load ImageNet-10 dataset for clustering"""
    if transform is None:
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    # 고정된 경로로 항상 로드 시도
    try:
        dataset = ImageNet10(root=root, transform=transform, download=False)
    except RuntimeError as e:
        print(f"Error loading ImageNet-10: {e}")
        return None, None

    # 샘플 수 제한 (지정된 경우)
    if max_samples is not None and max_samples < len(dataset):
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        indices = indices[:max_samples]
        dataset = Subset(dataset, indices)

    # 이미지와 라벨 추출
    images, labels = [], []
    for i in range(len(dataset)):
        img, label = dataset[i]
        images.append(img)
        labels.append(label)

    return images, np.array(labels, dtype=np.int64)


def load_imagenet_dogs(root="./data", transform=None, max_samples=None):
    """Load ImageNet-Dogs dataset for clustering"""
    if transform is None:
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # 항상 고정된 경로 사용
    try:
        dataset = ImageNetDogs(root=root, transform=transform, download=False)
    except RuntimeError as e:
        print(f"Error loading ImageNet-Dogs: {e}")
        return None, None

    # 샘플 수 제한 (지정된 경우)
    if max_samples is not None and max_samples < len(dataset):
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        indices = indices[:max_samples]
        dataset = Subset(dataset, indices)

    # 이미지와 라벨 추출
    images, labels = [], []
    for i in tqdm(range(len(dataset)), desc="Loading ImageNet-Dogs images"):
        img, label = dataset[i]
        images.append(img)
        labels.append(label)

    return images, np.array(labels, dtype=np.int64)


def get_image_dataset_info(dataset_name: str) -> dict:
    """Get information about an image dataset including its loader function"""
    dataset_loaders = {
        "stl10": {
            "description": "STL-10 (10 classes, 96x96 images)",
            "loader": load_stl10,
        },
        "cifar10": {
            "description": "CIFAR-10 (10 classes, 32x32 images)",
            "loader": load_cifar10,
        },
        "cifar100-20": {
            "description": "CIFAR-100 with 20 superclasses",
            "loader": load_cifar100_20,
        },
        "imagenet10": {
            "description": "ImageNet-10 (10 classes subset of ImageNet)",
            "loader": load_imagenet10,
        },
        "imagenet-dogs": {
            "description": "Subclasses of ImageNet-1000 (15 classes)",
            "loader": load_imagenet_dogs,
        },
    }

    return dataset_loaders.get(dataset_name.lower())
