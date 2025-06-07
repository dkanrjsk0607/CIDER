# prepare_imagenet_datasets.py
import os
import shutil
import pandas as pd
from tqdm import tqdm
import zipfile
import tarfile

def extract_dog_breeds(dog_data_dir, output_dir):
    """
    Kaggle Dog Breed Identification 데이터셋을 ImageNet-dogs 형식으로 변환
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 레이블 파일 로드
    labels_file = os.path.join(dog_data_dir, 'labels.csv')
    if not os.path.exists(labels_file):
        print(f"Error: labels.csv file not found in {dog_data_dir}")
        return False
    
    labels_df = pd.read_csv(labels_file)
    print(f"Found {len(labels_df)} dog images with {labels_df['breed'].nunique()} breeds")
    
    # 각 견종별 디렉토리 생성 및 이미지 복사
    train_dir = os.path.join(dog_data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"Error: train directory not found in {dog_data_dir}")
        return False
    
    # 견종(breed)을 ImageNet synset ID로 매핑 (임의 매핑)
    # ImageNet 견종 synset ID는 n02xxxxx 형식
    breeds = labels_df['breed'].unique()
    breed_to_synset = {}
    synset_base = 2085620  # ImageNet 견종 synset ID 시작점
    
    for i, breed in enumerate(breeds):
        # 120개 견종의 고유 synset ID 생성
        synset_id = synset_base + i
        synset = f"n{synset_id:08d}"
        breed_to_synset[breed] = synset
        
        # 견종별 디렉토리 생성
        breed_dir = os.path.join(output_dir, synset)
        os.makedirs(breed_dir, exist_ok=True)
    
    # 이미지 파일을 해당 디렉토리로 복사
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Copying dog images"):
        img_id = row['id']
        breed = row['breed']
        synset = breed_to_synset[breed]
        
        src_path = os.path.join(train_dir, f"{img_id}.jpg")
        dst_path = os.path.join(output_dir, synset, f"{synset}_{img_id}.JPEG")
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: Image {img_id}.jpg not found")
    
    # 견종 매핑 저장
    mapping_file = os.path.join(output_dir, "breed_synset_mapping.csv")
    mapping_df = pd.DataFrame(list(breed_to_synset.items()), columns=['breed', 'synset'])
    mapping_df.to_csv(mapping_file, index=False)
    
    print(f"Dog breeds dataset prepared in {output_dir}")
    print(f"Breed-synset mapping saved to {mapping_file}")
    return True

def prepare_imagenet10(imagenet10_dir, output_dir):
    """
    Kaggle ImageNet-10 데이터셋을 적절한 형식으로 준비
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # ImageNet-10 클래스와 synset 정의
    imagenet10_classes = {
        "dog": "n02084071",
        "bird": "n01503061",
        "car": "n02958343",
        "cat": "n02121808",
        "deer": "n02431122",
        "horse": "n02374451",
        "monkey": "n02484322",
        "airplane": "n02691156",
        "ship": "n04194289",
        "truck": "n04490091"
    }
    
    # 데이터 디렉토리 확인
    train_dir = os.path.join(imagenet10_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"Error: train directory not found in {imagenet10_dir}")
        return False
    
    # 각 클래스별 디렉토리 생성 및 이미지 복사
    for class_name, synset in imagenet10_classes.items():
        src_dir = os.path.join(train_dir, class_name)
        dst_dir = os.path.join(output_dir, synset)
        
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} not found")
            continue
        
        os.makedirs(dst_dir, exist_ok=True)
        
        # 모든 이미지 파일 복사
        img_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for i, img_file in enumerate(tqdm(img_files, desc=f"Copying {class_name} images")):
            src_path = os.path.join(src_dir, img_file)
            # ImageNet 형식에 맞게 파일명 변환
            dst_path = os.path.join(dst_dir, f"{synset}_{i:06d}.JPEG")
            
            shutil.copy2(src_path, dst_path)
    
    print(f"ImageNet-10 dataset prepared in {output_dir}")
    return True

def main():
    # 기본 경로 설정 (필요에 따라 수정)
    base_dir = "../../data"
    dog_data_dir = os.path.join(base_dir, "imagenet-dogs")
    imagenet10_dir = os.path.join(base_dir, "imagenet-10")
    
    # 출력 디렉토리 설정
    output_base = os.path.join(base_dir, "processed")
    imagenet_dogs_output = os.path.join(output_base, "imagenet-dogs")
    imagenet10_output = os.path.join(output_base, "imagenet-10")
    
    # 데이터 압축 해제 (필요한 경우)
    print("Checking for compressed files...")
    
    # Dog Breed 데이터셋 압축 해제
    dog_zip = os.path.join(base_dir, "imagenet-dogs.zip")
    if os.path.exists(dog_zip) and not os.path.exists(dog_data_dir):
        print(f"Extracting {dog_zip}...")
        with zipfile.ZipFile(dog_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
    
    # ImageNet-10 압축 해제
    imagenet10_zip = os.path.join(base_dir, "imagenet-10.zip")
    if os.path.exists(imagenet10_zip) and not os.path.exists(imagenet10_dir):
        print(f"Extracting {imagenet10_zip}...")
        with zipfile.ZipFile(imagenet10_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
    
    # 데이터셋 준비
    print("\nPreparing ImageNet-dogs dataset...")
    if os.path.exists(dog_data_dir):
        success = extract_dog_breeds(dog_data_dir, imagenet_dogs_output)
        if success:
            print("ImageNet-dogs preparation complete!")
        else:
            print("Failed to prepare ImageNet-dogs dataset")
    else:
        print(f"Error: Dog breed dataset not found at {dog_data_dir}")
    
    print("\nPreparing ImageNet-10 dataset...")
    if os.path.exists(imagenet10_dir):
        success = prepare_imagenet10(imagenet10_dir, imagenet10_output)
        if success:
            print("ImageNet-10 preparation complete!")
        else:
            print("Failed to prepare ImageNet-10 dataset")
    else:
        print(f"Error: ImageNet-10 dataset not found at {imagenet10_dir}")
    
    print("\nAll datasets prepared!")
    print(f"ImageNet-dogs available at: {imagenet_dogs_output}")
    print(f"ImageNet-10 available at: {imagenet10_output}")

if __name__ == "__main__":
    main()