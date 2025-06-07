#!/usr/bin/env python
# run_image_experiments.py
# Script to execute image clustering experiments across multiple datasets

import os
import subprocess
import argparse
import time
from datetime import datetime
import pandas as pd

# Define image datasets
IMAGE_DATASETS = [
    "stl10",  # STL-10 (10 classes)
    "cifar10",  # CIFAR-10 (10 classes)
    "cifar100-20",  # CIFAR-100 with 20 superclasses
    "imagenet10",  # ImageNet-10 (10 classes)
    "imagenet-dogs",  # ImageNet-Dogs (15 dog breeds)
]

# Default parameters for image datasets
DEFAULT_PARAMS = {
    "stl10": {
        "n_clusters": 10,
        "max_samples": 13000,
    },
    "cifar10": {
        "n_clusters": 10,
        "max_samples": 10000,  # Subsample from 60k images
    },
    "cifar100-20": {
        "n_clusters": 20,
        "max_samples": 10000,  # Subsample from 60k images
    },
    "imagenet10": {
        "n_clusters": 10,
        "max_samples": 13000,
    },
    "imagenet-dogs": {
        "n_clusters": 15,
        "max_samples": 19500,
    },
}

# Available pretrained image models
IMAGE_MODELS = [
    "microsoft/resnet-34",
    "google/vit-base-patch16-224-in21k",  # ViT base model
    "google/vit-large-patch16-224-in21k",  # ViT large model
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run image clustering experiments")

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=IMAGE_DATASETS + ["all"],
        default=["stl10"],
        help="Datasets to use for experiments (default: stl10)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=IMAGE_MODELS[0],
        help=f"Pretrained image model (default: {IMAGE_MODELS[0]})",
    )

    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID to use (-1 for CPU)"
    )

    parser.add_argument(
        "--run_gae", action="store_true", help="Run standard GAE (default: False)"
    )

    parser.add_argument(
        "--run_proposed",
        action="store_true",
        default=True,
        help="Run proposed GAE (default: True)",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Override default max samples per dataset",
    )

    parser.add_argument(
        "--clear_cache", action="store_true", help="Clear cache before running"
    )

    parser.add_argument(
        "--save_results",
        action="store_true",
        default=True,
        help="Save consolidated results to CSV",
    )

    return parser.parse_args()


def run_experiment(
    dataset, model, gpu_id, run_gae, run_proposed, max_samples, clear_cache
):
    """Run a single experiment with the given parameters"""

    # Get default parameters for this dataset
    params = DEFAULT_PARAMS.get(dataset, {})

    # Override max_samples if provided
    if max_samples is not None:
        samples = max_samples
    else:
        samples = params.get("max_samples", None)

    # Build command
    cmd = [
        "python",
        "-m",
        "src.main",
        "--dataset",
        dataset,
        "--pretrained_image_model",
        model,
        "--gpu_id",
        str(gpu_id),
        "--knn_graph_method",
        "pretrained_image",
    ]

    # 데이터셋이 imagenet10인 경우 처리 방법 지정
    if dataset == "imagenet10":
        # root 경로 지정
        custom_data_root = os.path.abspath(os.path.join("..", "data"))
        print(f"Setting custom data root for ImageNet-10: {custom_data_root}")

    # Add max samples if available
    if samples:
        cmd.extend(["--max_samples", str(samples)])

    # Add n_clusters if specified in params
    if "n_clusters" in params:
        cmd.extend(["--n_clusters", str(params["n_clusters"])])

    # Add GAE/Proposed GAE flags
    if not run_gae:
        cmd.append("--skip_gae")

    if not run_proposed:
        cmd.append("--skip_proposed")

    # Add cache clearing if requested
    if clear_cache:
        cmd.append("--clear_cache")

    # Generate a log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")

    # Print the command
    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")

    # Execute the command and capture output
    start_time = time.time()

    with open(log_file, "w") as f:
        # Write the command to the log file
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Started at: {timestamp}\n\n")

        # Run the process
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        # Stream output to console and log file
        for line in process.stdout:
            print(line, end="")  # Print to console
            f.write(line)  # Write to log file

        process.wait()

    end_time = time.time()
    run_time = end_time - start_time

    # Append run time to log file
    with open(log_file, "a") as f:
        f.write(
            f"\nExperiment completed in {run_time:.2f} seconds ({run_time/60:.2f} minutes)\n"
        )

    print(f"Experiment completed in {run_time:.2f} seconds ({run_time/60:.2f} minutes)")

    return {
        "dataset": dataset,
        "model": model,
        "run_time": run_time,
        "log_file": log_file,
        "exit_code": process.returncode,
    }


def extract_results(log_file):
    """Extract clustering results from log file with more robust parsing"""
    results = {}

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

            # 전체 로그를 검색하여 결과 찾기
            for i in range(len(lines)):
                line = lines[i].strip()

                # 직접 클러스터링 결과 확인
                if "Direct Clustering Results" in line:
                    # 이후 5-10줄 내에서 결과 찾기
                    for j in range(i + 1, min(i + 10, len(lines))):
                        # 결과 라인을 찾기 위한 여러 패턴 체크
                        result_line = lines[j].strip()
                        if "approach" in result_line and "acc" in result_line:
                            continue  # 헤더 라인 건너뛰기

                        # "DirectClustering" 텍스트가 포함된 라인 확인
                        if "DirectClustering" in result_line:
                            parts = result_line.split()
                            # 숫자 추출
                            numerics = [float(part) for part in parts if is_float(part)]
                            if len(numerics) >= 3:
                                results["direct"] = {
                                    "acc": numerics[0],
                                    "ari": numerics[1],
                                    "nmi": numerics[2],
                                }
                                break

                # 표준 GAE 결과 확인
                elif (
                    "Standard GAE Results" in line
                    or "----- Standard GAE Results -----" in line
                ):
                    # 이후 5-10줄 내에서 결과 찾기
                    for j in range(i + 1, min(i + 10, len(lines))):
                        result_line = lines[j].strip()
                        # "K-means: ACC=" 패턴 체크
                        if "K-means: ACC=" in result_line:
                            # 정규식을 사용해 숫자 추출
                            import re

                            nums = re.findall(r"[-+]?\d*\.\d+|\d+", result_line)
                            if len(nums) >= 3:
                                results["gae"] = {
                                    "acc": float(nums[0]),
                                    "ari": float(nums[1]),
                                    "nmi": float(nums[2]),
                                }
                                break
                        # 또는 다른 가능한 형식 체크
                        elif (
                            "acc" in result_line.lower()
                            and len(result_line.split()) >= 6
                        ):
                            parts = result_line.split()
                            numerics = [float(part) for part in parts if is_float(part)]
                            if len(numerics) >= 3:
                                results["gae"] = {
                                    "acc": numerics[0],
                                    "ari": numerics[1],
                                    "nmi": numerics[2],
                                }
                                break

                # Proposed GAE 결과 확인
                elif (
                    "Proposed GAE Results" in line
                    or "===== Proposed GAE Results =====" in line
                ):
                    # 이후 5-10줄 내에서 결과 찾기
                    for j in range(i + 1, min(i + 15, len(lines))):
                        result_line = lines[j].strip()
                        # 몇 가지 가능한 패턴 체크

                        # 패턴 1: "K-means: ACC=0.xxxx, ARI=0.xxxx, NMI=0.xxxx"
                        if "K-means: ACC=" in result_line:
                            import re

                            nums = re.findall(r"[-+]?\d*\.\d+|\d+", result_line)
                            if len(nums) >= 3:
                                results["proposed"] = {
                                    "acc": float(nums[0]),
                                    "ari": float(nums[1]),
                                    "nmi": float(nums[2]),
                                }
                                break

                        # 패턴 2: "Accuracy: 0.xxxx"와 같은 개별 라인
                        elif "Accuracy:" in result_line:
                            acc_val = None
                            ari_val = None
                            nmi_val = None

                            # Accuracy 추출
                            acc_match = re.search(
                                r"Accuracy:\s*([-+]?\d*\.\d+|\d+)", result_line
                            )
                            if acc_match:
                                acc_val = float(acc_match.group(1))

                            # 다음 라인들에서 ARI와 NMI 찾기
                            for k in range(j + 1, min(j + 5, len(lines))):
                                if "ARI:" in lines[k]:
                                    ari_match = re.search(
                                        r"ARI:\s*([-+]?\d*\.\d+|\d+)", lines[k]
                                    )
                                    if ari_match:
                                        ari_val = float(ari_match.group(1))
                                if "NMI:" in lines[k]:
                                    nmi_match = re.search(
                                        r"NMI:\s*([-+]?\d*\.\d+|\d+)", lines[k]
                                    )
                                    if nmi_match:
                                        nmi_val = float(nmi_match.group(1))

                            # 모든 값을 찾았다면 결과 저장
                            if (
                                acc_val is not None
                                and ari_val is not None
                                and nmi_val is not None
                            ):
                                results["proposed"] = {
                                    "acc": acc_val,
                                    "ari": ari_val,
                                    "nmi": nmi_val,
                                }
                                break

                        # 패턴 3: "ProposedGAE(...)" 형식의 직접 출력
                        elif (
                            "ProposedGAE" in result_line
                            and len(result_line.split()) >= 4
                        ):
                            parts = result_line.split()
                            numerics = [float(part) for part in parts if is_float(part)]
                            if len(numerics) >= 3:
                                results["proposed"] = {
                                    "acc": numerics[0],
                                    "ari": numerics[1],
                                    "nmi": numerics[2],
                                }
                                break

                # 일반적인 결과 라인 확인
                elif "ACC=" in line and "ARI=" in line and "NMI=" in line:
                    # 어떤 방법에 대한 결과인지 확인
                    method = None
                    if "Direct" in line or "direct" in line:
                        method = "direct"
                    elif (
                        "Standard GAE" in line
                        or "GAE" in line
                        and "Proposed" not in line
                    ):
                        method = "gae"
                    elif "Proposed" in line:
                        method = "proposed"

                    if method:
                        import re

                        nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                        if len(nums) >= 3:
                            results[method] = {
                                "acc": float(nums[0]),
                                "ari": float(nums[1]),
                                "nmi": float(nums[2]),
                            }

    except Exception as e:
        print(f"Error extracting results from {log_file}: {e}")
        import traceback

        traceback.print_exc()

    return results


# 숫자 확인 헬퍼 함수 추가
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def main():
    args = parse_arguments()

    # Determine which datasets to run
    if "all" in args.datasets:
        datasets = IMAGE_DATASETS
    else:
        datasets = args.datasets

    print(f"Starting experiments on datasets: {', '.join(datasets)}")
    print(f"Using model: {args.model}")

    # Store experiment run data
    experiments = []

    # Run all experiments
    for dataset in datasets:
        print(f"\n{'='*80}\nRunning experiment for {dataset}\n{'='*80}")

        exp_data = run_experiment(
            dataset=dataset,
            model=args.model,
            gpu_id=args.gpu_id,
            run_gae=args.run_gae,
            run_proposed=args.run_proposed,
            max_samples=args.max_samples,
            clear_cache=args.clear_cache,
        )

        # Extract results from log file
        results = extract_results(exp_data["log_file"])
        exp_data["results"] = results

        experiments.append(exp_data)

    # Save consolidated results if requested
    if args.save_results and experiments:
        results_rows = []

        for exp in experiments:
            # Add base info
            row = {
                "dataset": exp["dataset"],
                "model": exp["model"],
                "run_time_seconds": exp["run_time"],
                "exit_code": exp["exit_code"],
                "log_file": exp["log_file"],
            }

            # Add metrics for each method
            results = exp.get("results", {})

            # Add direct clustering results
            if "direct" in results:
                row["direct_acc"] = results["direct"]["acc"]
                row["direct_ari"] = results["direct"]["ari"]
                row["direct_nmi"] = results["direct"]["nmi"]
            else:
                row["direct_acc"] = None
                row["direct_ari"] = None
                row["direct_nmi"] = None

            # Add standard GAE results
            if "gae" in results:
                row["gae_acc"] = results["gae"]["acc"]
                row["gae_ari"] = results["gae"]["ari"]
                row["gae_nmi"] = results["gae"]["nmi"]
            else:
                row["gae_acc"] = None
                row["gae_ari"] = None
                row["gae_nmi"] = None

            # Add proposed GAE results
            if "proposed" in results:
                row["proposed_acc"] = results["proposed"]["acc"]
                row["proposed_ari"] = results["proposed"]["ari"]
                row["proposed_nmi"] = results["proposed"]["nmi"]
            else:
                row["proposed_acc"] = None
                row["proposed_ari"] = None
                row["proposed_nmi"] = None

            results_rows.append(row)

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(
            results_dir, f"image_clustering_results_{timestamp}.csv"
        )

        df = pd.DataFrame(results_rows)
        df.to_csv(results_file, index=False)

        print(f"\nConsolidated results saved to {results_file}")

        # Print summary table
        print("\nResults Summary:")
        print("-" * 80)
        print(f"{'Dataset':<15}{'ACC':<10}{'ARI':<10}{'NMI':<10}  Method")
        print("-" * 80)

        for exp in experiments:
            dataset = exp["dataset"]
            results = exp.get("results", {})

            # Print direct results
            if "direct" in results:
                print(
                    f"{dataset:<15}{results['direct']['acc']:<10.4f}{results['direct']['ari']:<10.4f}{results['direct']['nmi']:<10.4f}  Direct Clustering"
                )

            # Print GAE results
            if "gae" in results:
                print(
                    f"{dataset:<15}{results['gae']['acc']:<10.4f}{results['gae']['ari']:<10.4f}{results['gae']['nmi']:<10.4f}  Standard GAE"
                )

            # Print proposed results
            if "proposed" in results:
                print(
                    f"{dataset:<15}{results['proposed']['acc']:<10.4f}{results['proposed']['ari']:<10.4f}{results['proposed']['nmi']:<10.4f}  Proposed GAE"
                )

            print("-" * 80)


if __name__ == "__main__":
    main()
