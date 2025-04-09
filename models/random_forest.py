"""
비트코인 트레이딩 봇을 위한 Random Forest 모델

이 모듈은 분류(방향 예측) 작업을 위한 Random Forest 모델을 구현합니다.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple, TYPE_CHECKING
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import traceback
from abc import ABC, abstractmethod
import sys
import pickle
import json
from datetime import datetime
import logging
import joblib
import re
import time

from config import settings
from utils.logging import get_logger
from models.base import ModelOutput, TradingSignal, SignalType
from models.types import TrainingMetrics

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 파일 핸들러 추가
if not logger.handlers:
    fh = logging.FileHandler('logs/models.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# 순환 참조 방지를 위한 베이스 클래스 직접 정의
# ABC를 직접 상속해서 각 클래스를 별도로 정의
class _ModelBase(ABC):
    """로컬 참조용 기본 모델"""
    def __init__(self, name, model_type="base", version="1.0.0"):
        self.name = name
        self.model_type = model_type
        self.version = version
        self.is_trained = False
        
    def _preprocess_input(self, X):
        # 단순 구현
        return X, []

class _ClassificationModel(_ModelBase):
    """로컬 참조용 분류 모델"""
    def __init__(self, name, version="1.0.0"):
        super().__init__(name, "classification", version)

class _RegressionModel(_ModelBase):
    """로컬 참조용 회귀 모델"""
    def __init__(self, name, version="1.0.0"):
        super().__init__(name, "regression", version)

# 순환 참조 방지
if TYPE_CHECKING:
    from models.base import ClassificationModel, RegressionModel, ModelBase
else:
    # 런타임에는 실제 클래스 가져오기 시도
    # 로컬 클래스 정의는 그대로 보존 (fallback용)
    try:
        from models.base import ClassificationModel, RegressionModel, ModelBase
    except ImportError:
        # 가져오기 실패 시 로컬 클래스 사용
        ClassificationModel = _ClassificationModel
        RegressionModel = _RegressionModel
        ModelBase = _ModelBase

# 유틸리티 함수만 가져옴 (클래스 의존성 없는)
try:
    from models.base import get_feature_importance_sorted
except ImportError:
    # 가져오기 실패 시 간단한 버전 구현
    def get_feature_importance_sorted(model, top_n=None, threshold=None):
        if not hasattr(model, 'feature_importance') or not model.feature_importance:
            return {}
        
        sorted_importance = dict(sorted(
            model.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        if threshold is not None:
            sorted_importance = {k: v for k, v in sorted_importance.items() if v >= threshold}
        
        if top_n is not None:
            return dict(list(sorted_importance.items())[:top_n])
        
        return sorted_importance

# 로거 초기화
logger = get_logger(__name__)


class RandomForestDirectionModel(ClassificationModel):
    """시장 방향(상승/하락)을 예측하기 위한 Random Forest 모델"""
    
    def __init__(self, 
                name: str = "RandomForestDirection", 
                version: str = "1.0.0",
                n_estimators: int = 100,
                max_depth: int = 10,
                min_samples_split: int = 10,
                min_samples_leaf: int = 4,
                class_weight: str = 'balanced',
                strict_mode: bool = False,
                max_feature_count: Optional[int] = 20):
        """
        Random Forest 방향 예측 모델 초기화
        
        매개변수:
            name (str): 모델 이름
            version (str): 모델 버전
            n_estimators (int): 포레스트의 트리 개수
            max_depth (int): 트리의 최대 깊이
            min_samples_split (int): 노드 분할에 필요한 최소 샘플 수
            min_samples_leaf (int): 리프 노드에 필요한 최소 샘플 수
            class_weight (str): 클래스 불균형 처리 전략 ('balanced' 권장)
            strict_mode (bool): 엄격 모드 활성화 여부 (True일 경우 정규화 누락 등에서 예외 발생)
            max_feature_count (Optional[int]): 사용할 최대 특성 수 (특성 선택에 사용)
        """
        super().__init__(name, version)
        
        # 모델 초기화
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=42
        )
        
        # 모델 파라미터 저장
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight,
            'strict_mode': strict_mode,
            'max_feature_count': max_feature_count
        }
        
        self.feature_names = None
        self.selected_feature_indices = None  # selected_features에서 selected_feature_indices로 변경
        self.scaler = None  # 스케일러 객체 저장
        self.strict_mode = strict_mode  # strict_mode 저장
        self.max_feature_count = max_feature_count  # 최대 특성 수 저장
        self.metrics = {}  # 성능 메트릭을 저장할 빈 딕셔너리 초기화
        self.logger.info(f"{self.name} 모델을 {n_estimators}개의 트리, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, class_weight={class_weight}, strict_mode={strict_mode}로 초기화했습니다.")
    
    def _preprocess_input(self, X: Union[np.ndarray, pd.DataFrame], training: bool = False) -> Tuple[np.ndarray, List[int]]:
        """
        고급 데이터 전처리: 문자열 제거, 결측값 처리, 정규화/스케일링 수행
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): 입력 데이터
            training (bool): 훈련 데이터 여부, True이면 스케일러를 학습함
            
        Returns:
            Tuple[np.ndarray, List[int]]: 전처리된 데이터와 제거된 열 인덱스 목록
        
        Raises:
            ValueError: strict_mode가 True이고 스케일러가 없을 때 예측 단계에서 발생
        """
        # 부모 클래스의 기본 전처리 (문자열 제거, NaN 처리 등)
        X_cleaned, removed_indices = super()._preprocess_input(X)
        
        if X_cleaned.size == 0:
            self.logger.warning("전처리 후 데이터가 비어 있습니다.")
            return X_cleaned, removed_indices
        
        try:
            # 결측값 처리: NaN을 중앙값으로 대체
            # 이미 BaseModel에서 NaN을 0으로 대체했을 가능성이 있지만, 혹시 모르니 중앙값으로 더 보수적으로 처리
            for col in range(X_cleaned.shape[1]):
                col_data = X_cleaned[:, col]
                
                # 데이터 타입 확인 및 처리
                if not np.issubdtype(col_data.dtype, np.number):
                    self.logger.warning(f"컬럼 {col}: 숫자형이 아닌 데이터 타입({col_data.dtype}) 발견. 숫자형 변환 시도.")
                    try:
                        # pandas Series로 변환하여 to_numeric 적용 (벡터화된 연산 활용)
                        col_series = pd.Series(col_data)
                        numeric_col = pd.to_numeric(col_series, errors='coerce')
                        
                        # 변환 후 NaN 비율 확인 (너무 많은 값이 변환 실패했는지 체크)
                        nan_ratio = numeric_col.isna().mean()
                        if nan_ratio == 1.0:
                             self.logger.error(f"컬럼 {col}: 모든 값을 숫자형으로 변환 실패. 0으로 채웁니다.")
                             X_cleaned[:, col] = 0
                        else:
                            if nan_ratio > 0:
                                 self.logger.warning(f"컬럼 {col}: 숫자형 변환 중 일부 값({nan_ratio:.1%})이 NaN으로 처리됨.")
                            # 변환된 숫자형 데이터로 업데이트 (NaN 포함 가능)
                            X_cleaned[:, col] = numeric_col.values 
                            col_data = X_cleaned[:, col] # 업데이트된 col_data 사용
                            self.logger.info(f"컬럼 {col}: 숫자형 변환 완료.")
                            
                    except Exception as conversion_error:
                        self.logger.error(f"컬럼 {col}: 숫자형 변환 중 예상치 못한 오류 발생: {conversion_error}. 0으로 채웁니다.")
                        X_cleaned[:, col] = 0
                        continue # 오류 발생 시 해당 컬럼의 NaN 처리 건너뛰기
                
                # 이제 col_data는 숫자형이거나 숫자형으로 변환 시도 후의 데이터 (NaN 포함 가능)
                # np.isnan 처리는 숫자형 데이터에 대해서만 유효
                if np.issubdtype(col_data.dtype, np.number):
                    nan_mask = np.isnan(col_data.astype(float)) # astype(float) 추가하여 안전성 확보
                    if np.any(nan_mask):
                        # NaN이 있는 경우 중앙값으로 대체 (중앙값 계산이 불가능한 경우 0으로 대체)
                        valid_data = col_data[~nan_mask]
                        if valid_data.size > 0:
                            median_val = np.median(valid_data)
                            # np.where 사용 시 타입 일관성 유지 중요
                            X_cleaned[:, col] = np.where(nan_mask, median_val, col_data).astype(col_data.dtype)
                        else:
                            X_cleaned[:, col] = 0
            
            # 특성 수 조정 검사 (특히 예측 시)
            if not training and hasattr(self, 'expected_features_count'): # expected_features -> expected_features_count 로 변경
                current_features = X_cleaned.shape[1]
                if current_features != self.expected_features_count: # expected_features -> expected_features_count 로 변경
                    self.logger.warning(f"특성 수 불일치: 현재={current_features}, 모델 기대값={self.expected_features_count}") # expected_features -> expected_features_count 로 변경
                    X_cleaned = self._adjust_feature_count(X_cleaned, self.expected_features_count) # expected_features -> expected_features_count 로 변경
            
            # 스케일링 로직 수정 (train 메서드 내 _scale_features 호출 방식으로 변경됨)
            # 이 부분은 train 메서드에서 처리되므로 여기서는 제거하거나 주석 처리
            # self.logger.debug("_preprocess_input 단계에서는 스케일링을 수행하지 않습니다. train/predict 메서드에서 처리됩니다.")

            # 이상치 처리: 3 표준편차를 넘어가는 값을 클리핑
            # 스케일링은 train/predict에서 수행되므로, 여기서 이상치 처리는 스케일링된 데이터 기준이 아님
            # 필요하다면 원본 데이터 기준의 이상치 처리 로직을 추가하거나, 스케일링 후 처리하도록 train/predict 로직 검토
            # self.logger.debug("_preprocess_input 단계에서는 이상치 처리를 수행하지 않습니다.")

            return X_cleaned, removed_indices
            
        except Exception as e:
            self.logger.error(f"고급 전처리 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            # strict_mode가 활성화된 경우 예외를 다시 발생시킴
            if self.strict_mode:
                raise
            # 오류 발생 시 기본 전처리 결과 반환
            return X_cleaned, removed_indices
    
    def _adjust_feature_count(self, X: np.ndarray, expected_count: int) -> np.ndarray:
        """
        모델이 기대하는 특성 수에 맞게 입력 특성 배열을 조정합니다.
        
        Args:
            X (np.ndarray): 입력 특성 배열
            expected_count (int): 모델이 기대하는 특성 수
            
        Returns:
            np.ndarray: 조정된 특성 배열
        """
        current_count = X.shape[1]
        
        if current_count < expected_count:
            # 부족한 특성 추가 (0으로 채움)
            self.logger.warning(f"특성 수가 적음: {current_count} < {expected_count}, 부족한 특성을 0으로 채웁니다.")
            padding = np.zeros((X.shape[0], expected_count - current_count))
            return np.hstack((X, padding))
        elif current_count > expected_count:
            # 초과 특성 제거
            self.logger.warning(f"특성 수가 많음: {current_count} > {expected_count}, 처음 {expected_count}개만 사용합니다.")
            return X[:, :expected_count]
        else:
            # 특성 수가 일치하면 원본 반환
            return X
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, sample_weights=None, **kwargs) -> TrainingMetrics:
        """모델 훈련
        
        Args:
            X_train: 훈련 데이터
            y_train: 훈련 타겟
            X_val: 검증 데이터 (선택사항)
            y_val: 검증 타겟 (선택사항)
            feature_names: 특성 이름 목록 (선택사항)
            sample_weights: 샘플 가중치 (선택사항)
            **kwargs: 추가 매개변수
            
        Returns:
            TrainingMetrics: 훈련 지표 (정확도, F1 점수 등)
        """
        start_time = time.time()
        
        # 입력 데이터 형태 로깅
        self.logger.info(f"{self.name} 모델을 {len(X_train)}개의 샘플로 훈련합니다.")
        if hasattr(X_train, 'shape'):
            self.logger.debug(f"입력 데이터 형태: {X_train.shape}")
        
        # 3D 데이터인 경우 오류 처리
        if len(X_train.shape) == 3:
            error_msg = (f"RandomForest 모델은 3차원 데이터({X_train.shape})를 처리할 수 없습니다. "
                         "앙상블 모델의 train() 메서드에서 sequence_handling 옵션을 설정하세요.")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 데이터 검증 및 전처리
        try:
            y_train = self._validate_target(y_train)
            
            # --- 특성 선택 로직 --- 
            n_features_original = X_train.shape[1]
            X_train_processed = X_train.copy() # 원본 유지
            
            # 1. 특성 선택 (max_feature_count 사용 시)
            selected_indices = list(range(n_features_original))
            if self.max_feature_count is not None and n_features_original > self.max_feature_count:
                self.logger.info(f"특성 수 제한 적용: {n_features_original} -> {self.max_feature_count}")
                # 특성 중요도 계산을 위해 임시 모델 사용 (스케일링 전에 수행)
                temp_model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1, class_weight='balanced')
                try:
                     # 중요도 계산 시 NaN 값은 0으로 대체하여 사용
                     temp_model.fit(np.nan_to_num(X_train_processed, nan=0.0), y_train)
                     importances = temp_model.feature_importances_
                     if np.any(np.isnan(importances)) or np.any(importances < 0):
                          self.logger.warning("특성 중요도 계산 오류. 모든 특성을 사용합니다.")
                     else:
                          sorted_indices = np.argsort(importances)[::-1]
                          selected_indices = sorted_indices[:self.max_feature_count].tolist() # 리스트로 변환
                          self.logger.info(f"상위 {len(selected_indices)}개 특성 선택 완료.")
                except Exception as e:
                     self.logger.error(f"특성 중요도 계산 중 오류: {e}. 모든 특성을 사용합니다.")
                
            # 선택된 인덱스로 데이터 필터링
            X_train_selected = X_train_processed[:, selected_indices]
            self.selected_indices_temp = selected_indices # 훈련 성공 시 저장될 임시 변수
            
            # 선택된 특성 이름 업데이트
            if feature_names is not None and len(feature_names) == n_features_original:
                 self.selected_feature_names = [feature_names[i] for i in selected_indices]
            else:
                 self.selected_feature_names = [f"feature_{i}" for i in selected_indices] # 인덱스 기반 이름 생성
                 
            # --- 스케일링 로직 --- 
            # 2. 선택된 특성에 대해서만 스케일링 수행
            self._is_training = True # _scale_features 내부에서 훈련 상태 확인용
            X_train_scaled, scaler_fitted = self._scale_features(X_train_selected)
            self._is_training = False
            
            if not scaler_fitted:
                 self.logger.warning("스케일러가 학습되지 않았습니다. 스케일링 없이 진행될 수 있습니다.")

            # --- 최종 모델 훈련 --- 
            self.logger.info(f"최종 모델 훈련 시작. 입력 데이터 shape: {X_train_scaled.shape}")
            # ... (기존 모델 초기화 및 fit 부분)
            model_params = {
                'n_estimators': self.params['n_estimators'],
                'max_depth': self.params['max_depth'],
                'min_samples_split': self.params['min_samples_split'],
                'min_samples_leaf': self.params['min_samples_leaf'],
                'class_weight': self.params['class_weight'],
                'random_state': 42,
                'n_jobs': -1  # 병렬 처리 활용
            }
            self.model = RandomForestClassifier(**model_params)
            self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            
            # 예상 특성 수 저장 (선택/스케일링 후 최종 특성 수)
            self.expected_features_count = X_train_scaled.shape[1]
            self.logger.info(f"모델이 최종적으로 학습한 특성 수: {self.expected_features_count}")
            
            # 모델, 스케일러, 선택된 특성 간의 정합성 검증 강화
            model_features = getattr(self.model, 'n_features_in_', None)
            scaler_features = getattr(self.scaler, 'n_features_in_', None) if hasattr(self, 'scaler') else None
            selected_features_count = len(self.selected_indices_temp) # 임시 변수 사용
            
            if model_features is not None and scaler_features is not None:
                 if model_features != scaler_features or model_features != selected_features_count:
                      self.logger.error(f"훈련 후 특성 수 불일치: 모델={model_features}, 스케일러={scaler_features}, 선택된 특성={selected_features_count}")
                      if self.strict_mode:
                           raise ValueError("Strict Mode 활성화: 훈련 후 특성 수 불일치")
                 else:
                      self.logger.info("✓ 훈련 후 특성 수 정합성 검증 통과.")
            elif model_features is not None:
                 # 스케일러가 없는 경우 (예: _scale_features에서 학습 실패)
                 if model_features != selected_features_count:
                     self.logger.warning(f"훈련 후 특성 수 불일치 (스케일러 없음): 모델={model_features}, 선택된 특성={selected_features_count}")
            
            # 성능 지표 계산
            y_pred = self.model.predict(X_train_scaled) # 스케일링된 데이터 사용
            accuracy = accuracy_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred, average='weighted')
            
            # 소요 시간 계산
            elapsed_time = time.time() - start_time
            self.logger.info(f"모델 훈련 완료 ({elapsed_time:.2f}초)")
            self.logger.info(f"훈련 정확도: {accuracy:.4f}, F1 점수: {f1:.4f}")
            
            # 성능 임계값 확인 (kwargs에서 가져오거나 기본값 사용)
            min_accuracy_threshold = kwargs.get('min_accuracy_threshold', 0.55)
            min_f1_threshold = kwargs.get('min_f1_threshold', 0.5)
            
            # 성능이 임계값을 만족하는지 확인하고 is_trained 플래그 설정
            if accuracy < min_accuracy_threshold or f1 < min_f1_threshold:
                self.logger.warning(
                    f"{self.name} 모델 성능 기준 미달: 정확도={accuracy:.4f} (기준: {min_accuracy_threshold}), "
                    f"F1={f1:.4f} (기준: {min_f1_threshold})"
                )
                self.is_trained = False
            else:
                self.is_trained = True
                # 훈련 성공 시에만 선택된 특성 인덱스 저장
                if hasattr(self, 'selected_indices_temp'):
                    self.selected_feature_indices = self.selected_indices_temp
                    # selected_feature_names도 이때 확정
                    if not hasattr(self, 'selected_feature_names') or len(self.selected_feature_names) != len(self.selected_feature_indices):
                         self.selected_feature_names = [f"feature_{i}" for i in self.selected_feature_indices]
                    del self.selected_indices_temp # 임시 변수 삭제
                    self.logger.info(f"훈련 성공: 최종 선택된 특성 {len(self.selected_feature_indices)}개 저장됨 (인덱스/이름).")
                else:
                    # 특성 선택이 수행되지 않은 경우
                    self.selected_feature_indices = list(range(self.expected_features_count))
                    if feature_names is not None and len(feature_names) == self.expected_features_count:
                         self.selected_feature_names = feature_names
                    else:
                         self.selected_feature_names = [f"feature_{i+1}" for i in range(self.expected_features_count)]
                    self.logger.info(f"훈련 성공: 특성 선택 미수행, 모든 특성 {self.expected_features_count}개 사용됨.")
            
            # 훈련 지표 반환
            return TrainingMetrics(
                accuracy=accuracy,
                f1_score=f1,
                training_time=elapsed_time,
                feature_count=self.expected_features_count
            )
            
        except Exception as e:
            # 훈련 중 오류 발생 시 처리
            self.is_trained = False
            self.logger.error(f"모델 훈련 실패: {str(e)}")
            self.logger.error(traceback.format_exc())
            return TrainingMetrics(error='Model training failed')
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelOutput:
        """
        주어진 특성을 바탕으로 시장 방향 예측

        매개변수:
            X (np.ndarray): 입력 특성 (2D 배열)
            **kwargs: 추가 매개변수

        반환값:
            ModelOutput: 예측 결과를 포함하는 표준화된 모델 출력
            
        Raises:
            ValueError: strict_mode가 True이고 selected_features와 expected_features 간 심각한 불일치가 있을 때
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다.")
            # HOLD 신호와 함께 기본 ModelOutput 반환
            return ModelOutput(
                signal=TradingSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    reason="모델이 아직 훈련되지 않았습니다."
                ),
                confidence=0.0,
                metadata={"error": "Model not trained"}
            )

        try:
            # 입력 데이터 전처리 (training=False로 설정하여 저장된 스케일러 사용)
            # *** 중요: 전처리 전에 특성 수를 먼저 확인하고 조정해야 함 ***
            
            # 0. 원본 데이터 형태 저장 (디버깅용)
            original_shape = X.shape if hasattr(X, 'shape') else 'N/A'
            
            # 1. 모델이 학습 시 사용한 특성 인덱스(selected_feature_indices)로 필터링
            if hasattr(self, 'selected_feature_indices') and self.selected_feature_indices is not None:
                try:
                    # DataFrame인 경우 컬럼명 기반 필터링 시도
                    if isinstance(X, pd.DataFrame):
                         # 원본 컬럼명 저장 (나중에 비교용)
                         original_columns = X.columns.tolist()
                         # 선택된 특성 이름 목록 가져오기
                         expected_names = getattr(self, 'selected_feature_names', None)
                         if expected_names is not None:
                             # 예상 이름으로 정렬 및 누락 처리
                             X = X.reindex(columns=expected_names, fill_value=0)
                             self.logger.debug(f"DataFrame 입력: 선택된 특성 이름 {len(expected_names)}개로 정렬/필터링됨.")
                         else:
                             # 이름 정보가 없으면 인덱스 기반 필터링 (위험할 수 있음)
                             self.logger.warning("선택된 특성 이름 정보(selected_feature_names)가 없습니다. 인덱스 기반 필터링을 시도합니다.")
                             if X.shape[1] < max(self.selected_feature_indices) + 1:
                                 raise IndexError(f"NumPy 배열의 특성 수({X.shape[1]})가 선택된 최대 인덱스({max(self.selected_feature_indices)})보다 작습니다.")
                             X = X.iloc[:, self.selected_feature_indices] # DataFrame 인덱싱
                    # NumPy 배열인 경우 인덱스 기반 필터링
                    elif isinstance(X, np.ndarray):
                        if X.shape[1] < max(self.selected_feature_indices) + 1:
                             raise IndexError(f"NumPy 배열의 특성 수({X.shape[1]})가 선택된 최대 인덱스({max(self.selected_feature_indices)})보다 작습니다.")
                        X = X[:, self.selected_feature_indices]
                        self.logger.debug(f"NumPy 입력: 선택된 특성 인덱스 {len(self.selected_feature_indices)}개로 필터링됨.")
                    else:
                         self.logger.warning(f"지원하지 않는 입력 타입({type(X)}) для 특성 선택.")
                         
                except IndexError as ie:
                     self.logger.error(f"예측 시 특성 선택 오류 (IndexError): {ie}")
                     self.logger.error(f"원본 데이터 형태: {original_shape}, 필요한 최대 인덱스: {max(self.selected_feature_indices)}")
                     return ModelOutput(signal=TradingSignal(signal_type=SignalType.HOLD, reason=f"특성 선택 오류: {ie}"), metadata={'error': f'Feature selection IndexError: {ie}'})
                except Exception as fe:
                     self.logger.error(f"예측 시 특성 선택 중 예상치 못한 오류: {fe}")
                     self.logger.error(traceback.format_exc())
                     return ModelOutput(signal=TradingSignal(signal_type=SignalType.HOLD, reason=f"특성 선택 오류: {fe}"), metadata={'error': f'Feature selection error: {fe}'})
            else:
                self.logger.warning("선택된 특성 정보(selected_feature_indices)가 없습니다. 모든 입력 특성을 사용합니다.")

            # 2. 기본 전처리 수행 (문자열 제거, NaN 처리 등) - 선택된 특성에 대해서만
            X_cleaned, _ = self._preprocess_input(X, training=False)

            # 입력 데이터가 비어있는 경우
            if X_cleaned.size == 0:
                self.logger.warning("입력 데이터가 비어 있습니다.")
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason="입력 데이터가 비어 있습니다."
                    ),
                    confidence=0.0,
                    metadata={"error": "Empty input data"}
                )

            # 모델 존재 여부 확인
            if not hasattr(self, 'model') or self.model is None:
                self.logger.error("모델이 초기화되지 않았습니다.")
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason="모델이 초기화되지 않았습니다."
                    ),
                    confidence=0.0,
                    metadata={"error": "Model not initialized"}
                )

            # 2. 모델이 기대하는 특성 수 확인 (하드코딩된 76 대신 train에서 저장한 값 사용)
            expected_features = getattr(self, "expected_features", getattr(self.model, "n_features_in_", None))
            
            if expected_features is not None:
                n_features_input = X_cleaned.shape[1]

                # selected_features와 expected_features 간의 심각한 불일치 확인
                if self.selected_feature_indices is not None and expected_features != len(self.selected_feature_indices):
                    mismatch_msg = f"심각한 특성 불일치 위험: 선택된 특성 수({len(self.selected_feature_indices)})와 모델이 기대하는 특성 수({expected_features})가 다릅니다."
                    self.logger.warning(mismatch_msg)
                    
                    # 차이가 20% 이상인 경우 심각한 불일치로 간주
                    difference_ratio = abs(expected_features - len(self.selected_feature_indices)) / expected_features
                    if difference_ratio > 0.2:  # 20% 이상 차이
                        self.logger.error(f"특성 수 불일치가 심각합니다: {difference_ratio*100:.1f}% 차이")
                        if self.strict_mode:
                            raise ValueError(f"Strict Mode 활성화: {mismatch_msg} 예측을 중단합니다.")

                if expected_features != n_features_input:
                    self.logger.warning(
                        f"입력 특성 수({n_features_input})가 모델 특성 수({expected_features})와 일치하지 않습니다. "
                        "특성 수를 조정합니다."
                    )

                    if n_features_input < expected_features:
                        # 특성 수가 부족한 경우 0으로 패딩
                        padded_X = np.zeros((X_cleaned.shape[0], expected_features))
                        padded_X[:, :n_features_input] = X_cleaned
                        X_cleaned = padded_X
                        self.logger.info(f"특성 수를 {n_features_input}에서 {expected_features}로 패딩했습니다.")
                    else:
                        # 특성 수가 많은 경우 필요한 만큼만 사용
                        X_cleaned = X_cleaned[:, :expected_features]
                        self.logger.info(f"특성 수를 {n_features_input}에서 {expected_features}로 제한했습니다.")
                # else: # 특성 수가 일치하는 경우 조정 필요 없음

            # --- 예측 실행 ---
            # 특성 수 조정 로직이 끝난 후, 최종적으로 준비된 X_cleaned를 사용하여 예측
            raw_predictions = self.model.predict(X_cleaned)
            
            # 확률 값 추출 (가능한 경우)
            try:
                probabilities = self.model.predict_proba(X_cleaned)
                confidence = np.max(probabilities, axis=1)
            except (AttributeError, Exception) as e:
                self.logger.warning(f"확률 추출 실패: {str(e)}")
                confidence = np.ones_like(raw_predictions) * 0.5  # 기본 신뢰도

            # 결과를 ModelOutput으로 변환
            result = []
            for i, pred in enumerate(raw_predictions):
                # 방향에 따라 신호 유형 결정
                signal_type = SignalType.BUY if pred == 1 else SignalType.SELL
                conf_val = float(confidence[i]) if i < len(confidence) else 0.5
                
                # TradingSignal 생성
                signal = TradingSignal(
                    signal_type=signal_type,
                    confidence=conf_val,
                    reason=f"{self.name} 모델 예측",
                    metadata={
                        "model_name": self.name,
                        "model_type": self.model_type,
                        "model_version": self.version
                    }
                )
                
                # 개별 예측에 대한 ModelOutput 생성
                result.append(ModelOutput(
                    signal=signal,
                    raw_predictions=np.array([pred]),
                    confidence=conf_val,
                    metadata={
                        "model_name": self.name,
                        "model_type": self.model_type,
                        "prediction_time": datetime.now().isoformat()
                    }
                ))
            
            # 다중 예측인 경우 첫 번째 예측 반환, 단일 예측인 경우 해당 예측 반환
            if len(result) > 0:
                return result[0]
            else:
                # 빈 예측 결과인 경우 기본 HOLD 신호 반환
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason="예측 결과가 없습니다."
                    ),
                    confidence=0.0,
                    metadata={"error": "No prediction results"}
                )

        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # 오류 발생 시 HOLD 신호 반환
            return ModelOutput(
                signal=TradingSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    reason=f"예측 중 오류 발생: {str(e)}"
                ),
                confidence=0.0,
                metadata={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        모델을 사용하여 각 클래스의 확률 예측값을 반환합니다.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): 예측에 사용될 특성 데이터
            
        Returns:
            np.ndarray: 예측된 확률 배열. 각 행은 샘플을 나타내고, 각 열은 클래스를 나타냅니다.
            
        Raises:
            ValueError: strict_mode가 True이고 selected_features와 expected_features 간 심각한 불일치가 있을 때
        """
        if not self.is_trained:
            self.logger.error("모델이 학습되지 않았습니다. predict_proba를 호출하기 전에 train 함수를 호출하세요.")
            return np.zeros((X.shape[0] if hasattr(X, 'shape') else 1, 2))
        
        try:
            # 입력 데이터 전처리 (training=False로 설정하여 저장된 스케일러 사용)
            X_cleaned, _ = self._preprocess_input(X, training=False)
            
            # 모델의 예상 특성 수 확인 (하드코딩된 76 대신 학습 시 저장한 값 사용)
            expected_features = getattr(self, "expected_features", getattr(self.model, "n_features_in_", None))
            
            # 1. 먼저 selected_features가 있으면 적용 (특성 선택)
            if hasattr(self, 'selected_feature_indices') and self.selected_feature_indices is not None:
                selected_features_count = len(self.selected_feature_indices)
                
                # selected_features와 expected_features 간의 심각한 불일치 확인
                if expected_features is not None and expected_features != selected_features_count:
                    mismatch_msg = f"심각한 특성 불일치 위험: 선택된 특성 수({selected_features_count})와 모델이 기대하는 특성 수({expected_features})가 다릅니다."
                    self.logger.warning(mismatch_msg)
                    
                    # 차이가 20% 이상인 경우 심각한 불일치로 간주
                    difference_ratio = abs(expected_features - selected_features_count) / expected_features
                    if difference_ratio > 0.2:  # 20% 이상 차이
                        self.logger.error(f"특성 수 불일치가 심각합니다: {difference_ratio*100:.1f}% 차이")
                        if self.strict_mode:
                            raise ValueError(f"Strict Mode 활성화: {mismatch_msg} 예측을 중단합니다.")
                
                # 입력 데이터에 selected_features 적용 (선택된 특성만 사용)
                if X_cleaned.shape[1] >= max(self.selected_feature_indices) + 1:
                    X_cleaned = X_cleaned[:, self.selected_feature_indices]
                    self.logger.debug(f"선택된 특성 {selected_features_count}개를 사용하여 예측합니다.")
                else:
                    self.logger.warning(f"입력 데이터 특성 수({X_cleaned.shape[1]})가 선택된 특성 수({selected_features_count})보다 적습니다. 부분 적용 후 필요시 패딩합니다.")
                    # 사용 가능한 특성만 선택
                    valid_indices = [idx for idx in self.selected_feature_indices if idx < X_cleaned.shape[1]]
                    if valid_indices:
                        X_cleaned = X_cleaned[:, valid_indices]
            
            # 2. 그 다음 모델이 기대하는 특성 수에 맞게 조정
            if expected_features is not None and X_cleaned.shape[1] != expected_features:
                self.logger.warning(f"입력 특성 수({X_cleaned.shape[1]})가 모델 예상 특성 수({expected_features})와 일치하지 않습니다. 조정합니다.")
                
                if X_cleaned.shape[1] < expected_features:
                    # 특성 부족 시 0으로 패딩
                    padding = np.zeros((X_cleaned.shape[0], expected_features - X_cleaned.shape[1]))
                    X_cleaned = np.hstack((X_cleaned, padding))
                    self.logger.info(f"특성 수를 {expected_features}개로 패딩하였습니다.")
                else:
                    # 특성 초과 시 필요한 수만큼만 사용
                    X_cleaned = X_cleaned[:, :expected_features]
                    self.logger.info(f"특성 수를 {expected_features}개로 제한하였습니다.")
            
            # float 타입으로 변환 확인
            X_cleaned = X_cleaned.astype(np.float64)
            
            # 예측 수행
            proba = self.model.predict_proba(X_cleaned)
            return proba
            
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            if self.strict_mode:
                raise  # strict_mode일 경우 예외를 다시 발생시킴
            return np.zeros((X.shape[0] if hasattr(X, 'shape') else 1, 2))
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        테스트 데이터에 대한 모델 성능 평가

        매개변수:
            X_test (np.ndarray): 테스트 특성
            y_test (np.ndarray): 테스트 타겟
            **kwargs: 추가 매개변수

        반환값:
            Dict[str, Any]: 성능 지표
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다.")
            return {'error': 'Model not trained'}

        try:
            # 예측 수행
            y_pred = self.predict(X_test)

            # 예측 성공 여부 확인
            if y_pred is None or len(y_pred) == 0:
                self.logger.error("평가를 위한 예측 실패")
                return {'error': 'Prediction failed during evaluation'}

            # y_test와 y_pred 길이 확인 (예측이 X_test 크기와 다를 수 있으므로)
            if len(y_pred) != len(y_test):
                self.logger.warning(f"예측 결과 길이({len(y_pred)})와 실제 타겟 길이({len(y_test)})가 다릅니다. 평가를 위해 길이를 맞춥니다.")
                # 길이를 맞추는 로직 추가 (예: 짧은 쪽에 맞추거나, 오류 반환)
                # 여기서는 일단 짧은 쪽에 맞춘다고 가정
                min_len = min(len(y_pred), len(y_test))
                y_pred = y_pred[:min_len]
                y_test = y_test[:min_len]
                if min_len == 0:
                    self.logger.error("길이 조정 후 데이터가 없습니다. 평가 불가.")
                    return {'error': 'Length mismatch led to empty data for evaluation'}


            # 메트릭 계산
            metrics = {}
            # 배열 길이가 0이 아닌 경우에만 메트릭 계산 시도
            if len(y_test) > 0 and len(y_pred) > 0 :
                try:
                    metrics['accuracy'] = accuracy_score(y_test, y_pred)
                    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
                    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
                    metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
                except ValueError as ve:
                    self.logger.error(f"메트릭 계산 중 오류 발생 (데이터 타입 또는 형태 문제일 수 있음): {ve}")
                    return {'error': f'Metric calculation error: {ve}'}
            else:
                # 데이터가 없는 경우 기본값 설정
                metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                self.logger.warning("평가할 데이터가 없어 메트릭을 0으로 설정합니다.")

            # 확률 예측이 가능한 경우 신뢰도 점수 추가
            try:
                y_proba = self.predict_proba(X_test)
                # y_proba 길이도 확인 및 조정
                if y_proba is not None and len(y_proba) > 0:
                    if len(y_proba) != len(y_test): # y_test 길이는 위에서 조정되었을 수 있음
                        y_proba = y_proba[:len(y_test)]

                    if len(y_proba) > 0: # 조정 후 길이가 0이 아닌지 재확인
                        # 가장 높은 확률 클래스의 확률을 신뢰도로 사용
                        confidences = np.max(y_proba, axis=1)
                        metrics['mean_confidence'] = np.mean(confidences)
                        # 오류가 발생한 예측에 대한 신뢰도 계산 (y_pred와 y_test 길이는 위에서 맞춰짐)
                        error_mask = y_pred != y_test
                        if np.sum(error_mask) > 0:
                            metrics['confidence_at_error'] = np.mean(confidences[error_mask])
                        else:
                            metrics['confidence_at_error'] = None # 오류가 없는 경우
                    else:
                        self.logger.warning("신뢰도 계산 위한 확률 데이터 없음 (길이 조정 후).")

            except AttributeError:
                self.logger.debug("predict_proba 메서드가 없어 신뢰도 점수를 계산할 수 없습니다.")
            except Exception as e:
                self.logger.warning(f"신뢰도 점수 계산 중 오류: {str(e)}")

            # 메트릭 저장 및 반환
            # self.metrics가 딕셔너리라고 가정
            if hasattr(self, 'metrics') and isinstance(self.metrics, dict):
                self.metrics.update({f'test_{k}': v for k, v in metrics.items() if v is not None})
            else:
                self.logger.warning("self.metrics 속성이 없거나 딕셔너리가 아니어서 테스트 결과를 저장할 수 없습니다.")


            # 로그 출력 시 None 값 처리
            acc_log = f"{metrics.get('accuracy', 0.0):.4f}"
            f1_log = f"{metrics.get('f1', 0.0):.4f}"
            self.logger.info(f"테스트 정확도: {acc_log}, F1 점수: {f1_log}")

            return metrics

        except Exception as e:
            self.logger.error(f"모델 평가 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
            
    def save(self, custom_path: str = None) -> str:
        """
        모델 및 전처리 파이프라인 저장
        
        Args:
            custom_path (str, optional): 사용자 지정 경로
            
        Returns:
            str: 저장된 모델 파일 경로
        """
        if self.model is None:
            self.logger.error("저장할 모델이 없습니다. 먼저 모델을 학습해주세요.")
            return ""
        
        try:
            # 부모 클래스 save 메서드 호출
            save_path = super().save(custom_path)
            
            if not save_path:
                self.logger.error("기본 모델 저장 실패")
                return ""
            
            # 기본 경로에서 확장자 제거 후 디렉토리 이름 생성
            base_path = os.path.splitext(save_path)[0]
            model_dir = f"{base_path}_bundle"
            os.makedirs(model_dir, exist_ok=True)
            
            # 1. 모델 파일 저장
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # 2. 스케일러 저장
            if self.scaler is not None:
                scaler_path = os.path.join(model_dir, "scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            # 3. 특성 이름 목록 저장
            if hasattr(self, 'selected_feature_names') and self.selected_feature_names:
                features_path = os.path.join(model_dir, "features.json")
                with open(features_path, 'w') as f:
                    json.dump(self.selected_feature_names, f)
            elif hasattr(self, 'feature_names') and self.feature_names:
                features_path = os.path.join(model_dir, "features.json")
                with open(features_path, 'w') as f:
                    json.dump(self.feature_names, f)
            
            # 4. 메타데이터 저장 (YAML 형식)
            meta_data = {
                'name': self.name,
                'version': self.version,
                'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'expected_features': getattr(self, 'expected_features', None),
                'selected_features_count': len(self.selected_feature_indices) if hasattr(self, 'selected_feature_indices') and self.selected_feature_indices is not None else None,
                'strict_mode': getattr(self, 'strict_mode', False),
                'metrics': self.metrics,
                'params': self.params
            }
            
            import yaml
            meta_path = os.path.join(model_dir, "meta.yaml")
            with open(meta_path, 'w') as f:
                yaml.dump(meta_data, f, default_flow_style=False)
            
            # 5. 추가적인 구성 요소 저장 (기존 방식 호환성 유지)
            components_path = f"{base_path}_components.joblib"
            components = {
                'scaler': self.scaler,
                'expected_features': getattr(self, 'expected_features', None),
                'selected_features': self.selected_feature_indices,
                'feature_names': self.feature_names,
                'selected_feature_names': getattr(self, 'selected_feature_names', None),
                'strict_mode': getattr(self, 'strict_mode', False),
                'version': self.version
            }
            joblib.dump(components, components_path)
            
            self.logger.info(f"모델 번들 저장 완료: {model_dir}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류: {str(e)}")
            self.logger.error(traceback.format_exc())
            return ""
    
    def load(self, custom_path: str = None) -> bool:
        """
        모델 및 전처리 파이프라인 로드
        
        Args:
            custom_path (str, optional): 사용자 지정 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            # 부모 클래스 load 메서드 호출
            success = super().load(custom_path)
            
            if not success:
                self.logger.error("기본 모델 로드 실패")
                return False
            
            # 로드 경로 결정
            load_path = custom_path if custom_path else self.model_path
            base_path = os.path.splitext(load_path)[0]
            
            # 번들 디렉토리 확인
            model_dir = f"{base_path}_bundle"
            if os.path.isdir(model_dir):
                self.logger.info(f"모델 번들 로드: {model_dir}")
                
                # 1. 특성 목록 로드
                features_path = os.path.join(model_dir, "features.json")
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        self.feature_names = json.load(f)
                        self.selected_feature_names = self.feature_names
                    self.logger.info(f"특성 목록 로드됨: {len(self.feature_names)}개")
                else:
                    self.logger.warning("특성 목록 파일이 없습니다.")
                
                # 2. 스케일러 로드
                scaler_path = os.path.join(model_dir, "scaler.pkl")
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    self.logger.info("스케일러 로드됨")
                
                # 3. 메타데이터 로드
                meta_path = os.path.join(model_dir, "meta.yaml")
                if os.path.exists(meta_path):
                    import yaml
                    with open(meta_path, 'r') as f:
                        meta_data = yaml.safe_load(f)
                    
                    # 메타데이터에서 필요한 정보 추출
                    self.expected_features = meta_data.get('expected_features')
                    if 'strict_mode' in meta_data:
                        self.strict_mode = meta_data['strict_mode']
                        self.params['strict_mode'] = self.strict_mode
                    
                    # 버전 확인
                    loaded_version = meta_data.get('version', '0.0.0')
                    if loaded_version != self.version:
                        msg = f"저장된 모델 버전({loaded_version})과 현재 모델 버전({self.version})이 다릅니다. 호환성 문제가 발생할 수 있습니다."
                        self.logger.warning(msg)
                        
                    self.logger.info(f"메타데이터 로드됨: {meta_data.get('name')} v{loaded_version}")
                
            else:
                # 기존 방식의 컴포넌트 파일 확인 (하위 호환성)
                components_path = f"{base_path}_components.joblib"
                
                if os.path.exists(components_path):
                    # 구성 요소 로드
                    components = joblib.load(components_path)
                    
                    # 구성 요소 설정
                    if isinstance(components, dict):
                        self.scaler = components.get('scaler', None)
                        self.expected_features = components.get('expected_features', None)
                        self.selected_feature_indices = components.get('selected_features', None)
                        self.feature_names = components.get('feature_names', None)
                        self.selected_feature_names = components.get('selected_feature_names', self.feature_names)
                        
                        # strict_mode 로드 (저장된 값이 없으면 현재 값 유지)
                        if 'strict_mode' in components:
                            self.strict_mode = components['strict_mode']
                            self.params['strict_mode'] = self.strict_mode
                        
                        # 버전 확인 (호환성 확인용)
                        loaded_version = components.get('version', '0.0.0')
                        if loaded_version != self.version:
                            msg = f"저장된 모델 버전({loaded_version})과 현재 모델 버전({self.version})이 다릅니다. 호환성 문제가 발생할 수 있습니다."
                            self.logger.warning(msg)
                    
                    self.logger.info(f"레거시 모델 구성 요소 로드 완료: {components_path}")
                else:
                    msg = f"모델 구성 요소 파일이 없습니다. 기본 설정으로 계속합니다."
                    self.logger.warning(msg)
                    
                    # strict_mode가 활성화된 경우 컴포넌트 파일 누락에 대해 예외 발생
                    if self.strict_mode:
                        self.logger.error(f"Strict Mode 활성화: {msg}")
                        raise FileNotFoundError(f"Strict Mode 활성화: {msg}")
            
            # 4. 모델 검증
            self._validate_loaded_model()
            
            return True
                
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류: {str(e)}")
            self.logger.error(traceback.format_exc())
            if self.strict_mode:
                raise  # strict_mode일 경우 예외를 다시 발생시킴
            return False
    
    def _validate_loaded_model(self) -> bool:
        """
        로드된 모델의 유효성을 검사합니다.
        
        Returns:
            bool: 검증 성공 여부
        """
        if not hasattr(self, 'model') or self.model is None:
            self.logger.error("유효한 모델이 로드되지 않았습니다.")
            return False
        
        # 모델이 기대하는 특성 수 확인
        if hasattr(self.model, 'n_features_in_'):
            expected_count = self.model.n_features_in_
            
            # expected_features 설정되어 있지 않으면 설정
            if not hasattr(self, 'expected_features') or self.expected_features is None:
                self.expected_features = expected_count
                self.logger.info(f"모델이 기대하는 특성 수를 {expected_count}로 설정합니다.")
            
            # 특성 이름 검증
            if hasattr(self, 'feature_names') and self.feature_names:
                if len(self.feature_names) != expected_count:
                    msg = f"특성 이름 수({len(self.feature_names)})와 모델이 기대하는 특성 수({expected_count})가 일치하지 않습니다."
                    self.logger.warning(msg)
                    
                    if self.strict_mode:
                        raise ValueError(f"Strict Mode 활성화: {msg}")
            else:
                self.logger.warning("특성 이름이 없습니다. 예측 시 특성 순서가 중요합니다.")
        
        # 스케일러 검증
        if hasattr(self, 'scaler') and self.scaler is not None:
            if hasattr(self.scaler, 'n_features_in_'):
                scaler_features = self.scaler.n_features_in_
                model_features = getattr(self.model, 'n_features_in_', None)
                
                if model_features is not None and scaler_features != model_features:
                    msg = f"스케일러 특성 수({scaler_features})와 모델 특성 수({model_features})가 일치하지 않습니다."
                    self.logger.warning(msg)
                    
                    if self.strict_mode:
                        raise ValueError(f"Strict Mode 활성화: {msg}")
        
        return True

    def _validate_target(self, y: np.ndarray) -> np.ndarray:
        """
        타겟 데이터를 검증하고 필요한 경우 변환합니다.
        
        Args:
            y (np.ndarray): 검증할 타겟 데이터
            
        Returns:
            np.ndarray: 검증 및 변환된 타겟 데이터
        """
        # 입력이 None인 경우 처리
        if y is None:
            return None
        
        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        
        # 배열 형태로 변환
        y = np.asarray(y)
        
        # 차원 확인 및 1차원으로 변환
        if len(y.shape) > 1:
            y = y.flatten()
        
        # 분류 문제를 위한 라벨 검증
        unique_values = np.unique(y)
        if len(unique_values) > 2:
            self.logger.warning(f"이진 분류 모델에 {len(unique_values)}개의 고유 값이 있습니다. 수치가 0보다 큰지 여부에 따라 이진으로 변환합니다.")
            y = (y > 0).astype(int)
        
        # NaN 값 처리
        if np.any(np.isnan(y)):
            self.logger.warning("타겟 데이터에 NaN 값이 있습니다. 다수 클래스로 대체합니다.")
            most_common = np.bincount(y[~np.isnan(y)].astype(int)).argmax()
            y = np.where(np.isnan(y), most_common, y)
        
        return y.astype(int)
    
    def _scale_features(self, X: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        데이터 스케일링 (StandardScaler 사용)
        
        Args:
            X (np.ndarray): 스케일링할 데이터 (2D NumPy 배열)
            
        Returns:
            Tuple[np.ndarray, bool]: 스케일링된 데이터와 스케일러 학습 여부
        """
        scaler_fitted = False
        
        # 스케일러가 없으면 새로 생성 (훈련 시에만)
        if not hasattr(self, 'scaler') or self.scaler is None:
            # 훈련 시에만 스케일러 생성 및 학습
            if getattr(self, '_is_training', False): 
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                try:
                    X_scaled = self.scaler.fit_transform(X)
                    scaler_fitted = True
                    self.logger.info(f"StandardScaler가 {X.shape[1]}개 특성에 대해 학습되었습니다.")
                    # 스케일러의 특성 수 저장
                    setattr(self.scaler, 'n_features_in_', X.shape[1]) 
                except ValueError as e:
                    self.logger.error(f"스케일러 학습 오류: {e}")
                    # 오류 발생 시 원본 데이터 반환
                    X_scaled = X 
            else:
                 # 예측 시 스케일러가 없으면 경고 후 원본 반환
                 self.logger.warning("예측 시 스케일러가 없습니다. 스케일링 없이 진행합니다.")
                 X_scaled = X
        else:
            # 저장된 스케일러 사용 (예측 시)
            expected_features = getattr(self.scaler, 'n_features_in_', None)
            current_features = X.shape[1]
            
            if expected_features is not None and current_features != expected_features:
                msg = f"스케일러 특성 수 불일치: 입력={current_features}, 스케일러 기대값={expected_features}"
                self.logger.error(msg)
                if self.strict_mode:
                     raise ValueError(f"Strict Mode 활성화: {msg}")
                else:
                     self.logger.warning("특성 수를 스케일러에 맞게 자동 조정합니다.")
                     X = self._adjust_feature_count(X, expected_features)
            
            try:
                X_scaled = self.scaler.transform(X)
            except ValueError as e:
                 self.logger.error(f"저장된 스케일러 적용 오류: {e}")
                 # 오류 발생 시 원본 데이터 반환
                 X_scaled = X 

        return X_scaled, scaler_fitted