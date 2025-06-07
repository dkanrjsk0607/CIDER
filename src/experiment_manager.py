import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import gc


class NumpyEncoder(json.JSONEncoder):
    """NumPy 배열을 JSON으로 직렬화하기 위한 인코더"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class ExperimentManager:
    def __init__(self, base_dir: str = "./results"):
        """
        실험 결과와 캐시를 관리하는 클래스

        Args:
            base_dir: 실험 결과를 저장할 기본 디렉토리
        """
        self.base_dir = base_dir
        self.experiments_dir = os.path.join(base_dir, "results")
        self.cache_dir = os.path.join(base_dir, "cache")
        self.metadata_dir = os.path.join(base_dir, "metadata")

        # 디렉토리 생성
        for dir_path in [self.experiments_dir, self.cache_dir, self.metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 기존 요약 데이터 로드 (있는 경우)
        self.summary_file = os.path.join(self.metadata_dir, "experiment_summary.csv")
        if os.path.exists(self.summary_file):
            try:
                self.summary_data = pd.read_csv(self.summary_file).to_dict("records")
                print(f"Loaded {len(self.summary_data)} previous experiment records")
            except Exception as e:
                print(f"Error loading summary data: {e}")
                self.summary_data = []
        else:
            self.summary_data = []

    def _generate_experiment_id(self, params: Dict[str, Any]) -> str:
        """
        실험 파라미터를 기반으로 고유한 실험 ID 생성

        Args:
            params: 실험 파라미터 딕셔너리

        Returns:
            실험 ID (해시값)
        """
        # 파라미터를 정렬된 문자열로 변환
        param_str = json.dumps(params, sort_keys=True, cls=NumpyEncoder)
        # SHA-256 해시 생성
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _get_cache_key(self, stage: str, params: Dict[str, Any]) -> str:
        """
        캐시 키 생성

        Args:
            stage: 캐시 단계 (embeddings, graph, gae, proposed)
            params: 실험 파라미터

        Returns:
            캐시 키
        """
        return f"{stage}_{self._generate_experiment_id(params)}"

    def save_experiment_results(
        self, params: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """실험 결과 저장"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 요약 데이터 구성
        summary_data = {
            "timestamp": timestamp,
            "dataset": params.get("dataset"),
            "graph_method": params.get("graph", {}).get("method"),
            "graph_alpha": params.get("graph", {}).get("alpha"),
            "graph_max_connections": params.get("graph", {}).get("max_connections"),
            "graph_omega": params.get("graph", {}).get("omega"),
        }

        # GAE 결과 추가
        if "gae_results" in results and results["gae_results"]:
            gae_results = results["gae_results"]
            if isinstance(gae_results, dict):
                print("Adding GAE results to summary:", gae_results)
                summary_data.update(
                    {
                        "gae_acc": gae_results.get("acc"),
                        "gae_ari": gae_results.get("ari"),
                        "gae_nmi": gae_results.get("nmi"),
                        "gae_hidden_dim": params.get("gae", {}).get("hidden_dim"),
                        "gae_out_dim": params.get("gae", {}).get("out_dim"),
                        "gae_lr": params.get("gae", {}).get("learning_rate"),
                    }
                )
            else:
                print(f"Warning: GAE results format unexpected: {type(gae_results)}")

        # Proposed GAE 결과 추가
        if "proposed_results" in results and results["proposed_results"]:
            proposed_results = results["proposed_results"]
            if isinstance(proposed_results, dict):
                print("Adding Proposed GAE results to summary:", proposed_results)

                # Proposed GAE 파라미터 처리
                proposed_params = params.get("proposed", {})

                # 리스트 파라미터는 첫 번째 값 사용
                channels1 = proposed_params.get("channels1")
                if isinstance(channels1, list) and channels1:
                    channels1 = channels1[0]

                channels2 = proposed_params.get("channels2")
                if isinstance(channels2, list) and channels2:
                    channels2 = channels2[0]

                lr = proposed_params.get("lr")
                if isinstance(lr, list) and lr:
                    lr = lr[0]

                lambda_factor = proposed_params.get("lambda_factor")
                if isinstance(lambda_factor, list) and lambda_factor:
                    lambda_factor = lambda_factor[0]

                hard_negative_ratio = proposed_params.get("hard_negative_ratio")
                if isinstance(hard_negative_ratio, list) and hard_negative_ratio:
                    hard_negative_ratio = hard_negative_ratio[0]

                k_hop_values = proposed_params.get("k_hop_values")

                summary_data.update(
                    {
                        "proposed_acc": proposed_results.get("acc"),
                        "proposed_ari": proposed_results.get("ari"),
                        "proposed_nmi": proposed_results.get("nmi"),
                        "channels1": channels1,
                        "channels2": channels2,
                        "lr": lr,
                        "lambda_factor": lambda_factor,
                        "hard_negative_ratio": hard_negative_ratio,
                        "k_hop_values": str(k_hop_values) if k_hop_values else None,
                    }
                )
            else:
                print(
                    f"Warning: Proposed GAE results format unexpected: {type(proposed_results)}"
                )

        # 결과 출력
        print("\nExperiment summary data to be added:")
        for key, value in summary_data.items():
            print(f"  {key}: {value}")

        # 요약 데이터 저장
        self.summary_data.append(summary_data)
        self.save_summary()

        # 저장 후 메모리 정리
        gc.collect()

    def get_cached_result(self, stage: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        캐시된 결과 로드

        Args:
            stage: 실험 단계 ("embeddings", "graph", "gae", "proposed")
            params: 실험 파라미터

        Returns:
            캐시된 결과 또는 None
        """
        cache_path = os.path.join(self.cache_dir, f"{stage}_cache.pkl")
        if os.path.exists(cache_path):
            try:
                # 단일 캐시 키만 로드하도록 최적화
                cache_key = self._get_cache_key(stage, params)

                # 필요한 키가 있는지 확인
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                    if cache_key in cache_data:
                        print(f"[CACHE] Found cached result for {stage}")
                        result = cache_data[cache_key]
                        del cache_data
                        gc.collect()
                        return result
            except Exception as e:
                print(f"[Warning] Failed to load cache for {stage}: {e}")
        return None

    def save_to_cache(self, stage: str, params: Dict[str, Any], result: Any):
        """
        결과를 캐시에 저장

        Args:
            stage: 실험 단계
            params: 실험 파라미터
            result: 저장할 결과
        """
        cache_path = os.path.join(self.cache_dir, f"{stage}_cache.pkl")
        try:
            # 기존 캐시 로드 또는 새로운 딕셔너리 생성
            cache_data = {}
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)

            # 결과 저장
            cache_key = self._get_cache_key(stage, params)
            cache_data[cache_key] = result

            # 캐시 파일 저장
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"[CACHE] Saved {stage} result to cache")

            # 메모리 정리
            del cache_data
            gc.collect()
        except Exception as e:
            print(f"[Warning] Failed to save cache for {stage}: {e}")

    def clear_cache(self, stage: Optional[str] = None):
        """
        캐시 삭제

        Args:
            stage: 삭제할 단계. None이면 모든 캐시 삭제
        """
        if stage:
            cache_path = os.path.join(self.cache_dir, f"{stage}_cache.pkl")
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"[CACHE] Cleared cache for {stage}")
        else:
            for file in os.listdir(self.cache_dir):
                if file.endswith("_cache.pkl"):
                    os.remove(os.path.join(self.cache_dir, file))
            print("[CACHE] Cleared all cache")

    def save_summary(self) -> None:
        """실험 요약을 CSV 파일로 저장"""
        if not self.summary_data:
            return

        try:
            # 결과가 None인 경우를 처리
            clean_data = []
            for item in self.summary_data:
                # None 값을 NaN으로 변환
                clean_item = {
                    k: (v if v is not None else float("nan")) for k, v in item.items()
                }
                clean_data.append(clean_item)

            # DataFrame으로 변환하고 저장
            df = pd.DataFrame(clean_data)

            # DataFrame 내용 확인
            print(f"\nSaving summary data ({len(df)} rows):")
            print(df.head())

            # 중복 행 제거 (모든 열 기준)
            df_dedup = df.drop_duplicates()
            if len(df_dedup) < len(df):
                print(f"Removed {len(df) - len(df_dedup)} duplicate rows")
                df = df_dedup

            # CSV로 저장
            df.to_csv(self.summary_file, index=False)
            print(f"Experiment summary saved to {self.summary_file}")

            # 추가로 타임스탬프가 있는 백업 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(
                self.metadata_dir, f"experiment_summary_{timestamp}.csv"
            )
            df.to_csv(backup_file, index=False)
            print(f"Backup saved to {backup_file}")
        except Exception as e:
            print(f"[Warning] Failed to save experiment summary: {e}")
            import traceback

            traceback.print_exc()

    def get_experiment_summary(self) -> pd.DataFrame:
        """실험 결과 요약을 DataFrame으로 반환"""
        # 결과가 None인 경우를 처리
        clean_data = []
        for item in self.summary_data:
            # None 값을 NaN으로 변환
            clean_item = {
                k: (v if v is not None else float("nan")) for k, v in item.items()
            }
            clean_data.append(clean_item)

        if not clean_data:
            return pd.DataFrame()

        return pd.DataFrame(clean_data)

    def clear_memory(self):
        """메모리 정리 함수"""
        # summary_data는 유지하고, 다른 메모리 정리
        gc.collect()
