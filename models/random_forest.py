"""
비트코인 트레이딩 봇을 위한 Random Forest 모델

이 모듈은 분류(방향 예측)와 회귀(가격 예측) 작업을 위한 Random Forest 모델을 구현합니다.
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
        
    def _remove_string_columns(self, X):
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

from utils.logging import get_logger

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
                class_weight: str = 'balanced'):
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
        """
        super().__init__(name, version)
        
        # 로거 초기화
        self.logger = get_logger(__name__)
        
        # 모델 초기화
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=42
        )
        
        self.feature_names = None
        self.metrics = {}  # 성능 메트릭을 저장할 빈 딕셔너리 초기화
        self.logger.info(f"{self.name} 모델을 {n_estimators}개의 트리, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, class_weight={class_weight}로 초기화했습니다.")
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray, 
             feature_names: Optional[List[str]] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Random Forest 모델 훈련
        
        매개변수:
            X_train (np.ndarray): 학습 특성
            y_train (np.ndarray): 학습 타겟 (하락은 0, 상승은 1)
            feature_names (Optional[List[str]]): 특성 이름 목록
            **kwargs: 추가 매개변수
                - class_weight (str or dict): 클래스 가중치
                - sample_weight (np.ndarray): 샘플 가중치
                
        반환값:
            Dict[str, Any]: 학습 평가 지표
        """
        start_time = pd.Timestamp.now()
        self.logger.info(f"{self.name} 모델을 {X_train.shape[0]}개의 샘플로 훈련합니다.")
        
        # 제공된 경우 모델 매개변수 업데이트
        if 'class_weight' in kwargs:
            self.model.class_weight = kwargs['class_weight']
        
        # 문자열 열이 있는지 확인하고 제거
        original_shape = X_train.shape
        X_train_cleaned, removed_indices = self._remove_string_columns(X_train)
        
        # 데이터 형태가 변경되었는지 확인
        if original_shape[1] != X_train_cleaned.shape[1]:
            self.logger.info(f"X_train 형태가 {original_shape}에서 {X_train_cleaned.shape}로 변경되었습니다.")
            
            # feature_names가 제공된 경우, 제거된 열을 반영하여 업데이트
            if feature_names is not None:
                updated_feature_names = [
                    name for i, name in enumerate(feature_names) 
                    if i not in removed_indices
                ]
                self.logger.info(f"{len(feature_names) - len(updated_feature_names)}개의 feature 이름이 제거되었습니다.")
                feature_names = updated_feature_names
        else:
            X_train_cleaned = X_train
        
        # 특성 이름 저장
        self.feature_names = feature_names
        
        # 모델 훈련
        sample_weight = kwargs.get('sample_weight', None)
        try:
            self.model.fit(X_train_cleaned, y_train, sample_weight=sample_weight)
            
            # 특성 중요도 가져오기
            if self.feature_names:
                self.feature_importance = {
                    self.feature_names[i]: importance 
                    for i, importance in enumerate(self.model.feature_importances_)
                }
            else:
                self.feature_importance = {
                    f"feature_{i}": importance 
                    for i, importance in enumerate(self.model.feature_importances_)
                }
            
            # 학습 평가 지표 계산
            y_pred = self.model.predict(X_train_cleaned)
            train_accuracy = accuracy_score(y_train, y_pred)
            train_precision = precision_score(y_train, y_pred, average='weighted')
            train_recall = recall_score(y_train, y_pred, average='weighted')
            train_f1 = f1_score(y_train, y_pred, average='weighted')
            
            # 평가 지표 저장
            metrics = {
                'train_accuracy': train_accuracy,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'training_time': (pd.Timestamp.now() - start_time).total_seconds()
            }
            
            self.metrics.update(metrics)
            self.is_trained = True
            self.last_update = pd.Timestamp.now()
            self.classes_ = self.model.classes_
            
            self.logger.info(f"훈련 완료. 정확도: {train_accuracy:.4f}, F1: {train_f1:.4f}")
            return metrics
        except Exception as e:
            self.logger.error(f"모델 훈련 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def predict(self, 
               X: np.ndarray,
               **kwargs) -> np.ndarray:
        """
        모델을 사용하여 예측 수행
        
        매개변수:
            X (np.ndarray): 입력 특성
            **kwargs: 추가 매개변수
            
        반환값:
            np.ndarray: 예측된 클래스 레이블 (하락은 0, 상승은 1)
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다. 예측이 부정확할 수 있습니다.")
        
        # 문자열 열 제거 (훈련 시와 동일한 전처리)
        X_cleaned, _ = self._remove_string_columns(X)
        return self.model.predict(X_cleaned)
    
    def predict_proba(self, 
                    X: np.ndarray, 
                    **kwargs) -> np.ndarray:
        """
        클래스 확률 예측
        
        매개변수:
            X (np.ndarray): 입력 특성
            **kwargs: 추가 매개변수
            
        반환값:
            np.ndarray: 클래스 확률
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다. 예측이 부정확할 수 있습니다.")
        
        # 문자열 열 제거 (훈련 시와 동일한 전처리)
        X_cleaned, _ = self._remove_string_columns(X)
        return self.model.predict_proba(X_cleaned)
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        모델 평가
        
        매개변수:
            X_test (np.ndarray): 테스트 특성
            y_test (np.ndarray): 테스트 타겟
            **kwargs: 추가 매개변수
            
        반환값:
            Dict[str, Any]: 평가 지표
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다. 평가가 부정확할 수 있습니다.")
        
        # 문자열 열 제거 (훈련 시와 동일한 전처리)
        X_test_cleaned, _ = self._remove_string_columns(X_test)
        
        # 예측 수행
        y_pred = self.model.predict(X_test_cleaned)
        y_proba = self.model.predict_proba(X_test_cleaned)
        
        # 평가 지표 계산
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 평가 지표 저장
        metrics = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        
        self.metrics.update(metrics)
        self.logger.info(f"평가 완료. 정확도: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        return metrics
    
    def optimize_hyperparams(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray = None,
                           y_val: np.ndarray = None,
                           param_grid: Optional[Dict[str, List[Any]]] = None,
                           cv: int = 5,
                           n_iter: int = 30,
                           method: str = 'random',
                           **kwargs) -> Dict[str, Any]:
        """
        Hyperparameter optimization using grid search or random search
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            param_grid (Optional[Dict[str, List[Any]]]): Parameter grid to search
            cv (int): Number of cross-validation folds
            n_iter (int): Number of iterations for random search
            method (str): Search method ('grid' or 'random')
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Best parameters and evaluation metrics
        """
        self.logger.info(f"Hyperparameter optimization using {method} search")
        
        if param_grid is None:
            # Improved default parameter grid
            param_grid = {
                'n_estimators': [50, 75, 100],  # More efficient range of trees
                'max_depth': [5, 7, 10, 15],    # Fine-grained depth adjustment
                'min_samples_split': [10, 20, 30],  # Higher values
                'min_samples_leaf': [5, 10, 15]  # Higher values
            }
        
        # Check class distribution
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        self.logger.info(f"Class distribution: {class_distribution}")
        
        # Determine CV folds based on minimum class count
        min_class_count = min(class_counts)
        safe_cv = min(cv, min_class_count)
        if safe_cv < cv:
            self.logger.warning(f"Reducing CV folds from {cv} to {safe_cv} due to low sample count per class.")
            cv = max(2, safe_cv)  # Ensure at least 2 folds
        
        # Handle extreme imbalance (very few samples in one class)
        if min_class_count < 3:
            self.logger.warning(f"Extreme class imbalance detected: minimum class samples = {min_class_count}")
            
            # Method 1: Skip CV and train directly
            if min_class_count < 2:
                self.logger.warning("Skipping CV and training model directly.")
                # Create basic model with improved defaults
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    class_weight='balanced',
                    random_state=self.params['random_state'],
                    n_jobs=-1
                )
                
                # Train directly
                self.model.fit(X_train, y_train)
                self.is_trained = True
                self.last_update = pd.Timestamp.now()
                self.classes_ = self.model.classes_
                
                # Evaluate on validation set
                val_metrics = self.evaluate(X_val, y_val)
                
                # Return default parameters
                best_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10
                }
                self.params.update(best_params)
                
                self.logger.info(f"Model training complete. Using default parameters: {best_params}")
                
                return {
                    'best_params': best_params,
                    'best_score': val_metrics.get('test_f1', 0),
                    'val_metrics': val_metrics
                }
        
        # Set scoring method
        scoring = 'f1_weighted'  # Appropriate metric for imbalanced data
        
        # Use StratifiedKFold to maintain class ratio
        from sklearn.model_selection import StratifiedKFold
        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.params['random_state'])
        
        # Create base model with class_weight='balanced'
        base_model = RandomForestClassifier(
            random_state=self.params['random_state'], 
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Create search object
        if method == 'grid':
            search = GridSearchCV(
                base_model, 
                param_grid=param_grid, 
                cv=stratified_cv,  # Use StratifiedKFold
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        else:  # Random search
            search = RandomizedSearchCV(
                base_model, 
                param_distributions=param_grid, 
                n_iter=n_iter,
                cv=stratified_cv,  # Use StratifiedKFold
                scoring=scoring,
                n_jobs=-1,
                random_state=self.params['random_state'],
                verbose=1
            )
        
        try:
            # Run search
            search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.model = search.best_estimator_
            self.params.update(search.best_params_)
            self.is_trained = True
            self.last_update = pd.Timestamp.now()
            self.classes_ = self.model.classes_
            
            # Evaluate on validation set
            val_metrics = self.evaluate(X_val, y_val)
            
            self.logger.info(f"Hyperparameter optimization complete. Best parameters: {search.best_params_}")
            return {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'val_metrics': val_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter optimization: {str(e)}")
            self.logger.warning("Falling back to default model.")
            
            # Create basic model with improved defaults
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=self.params['random_state'],
                n_jobs=-1
            )
            
            # Train directly
            try:
                self.model.fit(X_train, y_train)
                self.is_trained = True
                self.last_update = pd.Timestamp.now()
                
                # Evaluate on validation set
                val_metrics = self.evaluate(X_val, y_val)
                
                # Return default parameters
                best_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10
                }
                self.params.update(best_params)
                
                self.logger.info(f"Default model training complete. Parameters: {best_params}")
                
                return {
                    'best_params': best_params,
                    'best_score': val_metrics.get('test_f1', 0),
                    'val_metrics': val_metrics
                }
            except Exception as e2:
                self.logger.error(f"Default model training failed: {str(e2)}")
                return {
                    'error': str(e2),
                    'best_params': {},
                    'best_score': 999999.0
                }

    def save(self, filepath: Optional[str] = None) -> str:
        """
        모델을 디스크에 저장
        
        매개변수:
            filepath (Optional[str]): 모델을 저장할 경로. None인 경우 기본 경로 사용
            
        반환값:
            str: 모델이 저장된 경로
        """
        if filepath is None:
            # 모델 이름, 유형 및 타임스탬프를 사용하여 기본 파일 이름 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{self.model_type}_{timestamp}.pkl"
            
            # 모델 디렉토리 확인 및 생성
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
            os.makedirs(model_dir, exist_ok=True)
            
            filepath = os.path.join(model_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            
            self.logger.info(f"모델이 {filepath}에 저장되었습니다.")
            
            # 쉽게 검사할 수 있도록 JSON 형식으로 메타데이터도 저장
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            
            # 직렬화 가능한 형태로 메타데이터 변환
            params_serializable = {}
            for k, v in self.params.items():
                if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                    params_serializable[k] = v
                else:
                    params_serializable[k] = str(v)
            
            metadata = {
                'name': self.name,
                'type': self.model_type,
                'version': self.version,
                'creation_date': self.creation_date.strftime("%Y-%m-%d %H:%M:%S") if hasattr(self, 'creation_date') else None,
                'last_update': self.last_update.strftime("%Y-%m-%d %H:%M:%S"),
                'is_trained': self.is_trained,
                'metrics': self.metrics,
                'params': params_serializable,
                'feature_importance': self.feature_importance if hasattr(self, 'feature_importance') else {}
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            return filepath
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'RandomForestDirectionModel':
        """
        디스크에서 모델 로드
        
        매개변수:
            filepath (str): 저장된 모델의 경로
            
        반환값:
            RandomForestDirectionModel: 로드된 모델
        """
        logger = get_logger(__name__)
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"모델이 {filepath}에서 로드되었습니다.")
            return model
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class RandomForestMarketStateModel(ClassificationModel):
    """시장 상태(강세/약세/횡보)를 분류하기 위한 Random Forest 모델"""
    
    def __init__(self, 
                name: str = "RandomForestMarketState", 
                version: str = "1.0.0",
                n_estimators: int = 100,
                max_depth: Optional[int] = 10,  # None에서 10으로 변경 (과적합 방지)
                min_samples_split: int = 20,    # 2에서 20으로 변경 (과적합 방지)
                min_samples_leaf: int = 10,     # 1에서 10으로 변경 (과적합 방지)
                random_state: int = 42,
                class_weight: str = 'balanced'):  # 기본값으로 'balanced' 설정 (클래스 불균형 처리)
        """
        Random Forest 시장 상태 모델 초기화
        
        매개변수:
            name (str): 모델 이름
            version (str): 모델 버전
            n_estimators (int): 포레스트의 트리 개수
            max_depth (Optional[int]): 트리의 최대 깊이 (10 권장)
            min_samples_split (int): 노드 분할에 필요한 최소 샘플 수 (20 권장)
            min_samples_leaf (int): 리프 노드에 필요한 최소 샘플 수 (10 권장)
            random_state (int): 재현성을 위한 랜덤 시드
            class_weight (str): 클래스 불균형 처리 전략 ('balanced' 권장)
        """
        super().__init__(name, version)
        
        # 로거 초기화
        self.logger = get_logger(__name__)
        
        # 모델 초기화
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,  # 클래스 불균형 처리 기본 활성화
            n_jobs=-1
        )
        
        # 매개변수 저장
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'class_weight': class_weight  # 클래스 불균형 설정 저장
        }
        
        self.feature_names = None
        self.metrics = {}  # 성능 메트릭을 저장할 빈 딕셔너리 초기화
        self.logger.info(f"{self.name} 모델을 {n_estimators}개의 트리, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, class_weight={class_weight}로 초기화했습니다.")
        
        # 시장 상태 정의 (0: 약세, 1: 횡보, 2: 강세)
        self.market_states = {
            0: 'bearish',
            1: 'sideways',
            2: 'bullish'
        }
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray, 
             feature_names: Optional[List[str]] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Random Forest 시장 상태 모델 훈련
        
        매개변수:
            X_train (np.ndarray): 학습 특성
            y_train (np.ndarray): 학습 타겟 (0: 약세, 1: 횡보, 2: 강세)
            feature_names (Optional[List[str]]): 특성 이름 목록
            **kwargs: 추가 매개변수
                - class_weight (str or dict): 클래스 가중치
                - sample_weight (np.ndarray): 샘플 가중치
                
        반환값:
            Dict[str, Any]: 학습 평가 지표
        """
        start_time = pd.Timestamp.now()
        self.logger.info(f"{self.name} 모델을 {X_train.shape[0]}개의 샘플로 훈련합니다.")
        
        # 제공된 경우 모델 매개변수 업데이트
        if 'class_weight' in kwargs:
            self.model.class_weight = kwargs['class_weight']
            self.params['class_weight'] = kwargs['class_weight']
        
        # 특성 이름 저장
        self.feature_names = feature_names
        
        # 모델 훈련
        sample_weight = kwargs.get('sample_weight', None)
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        
        # 특성 중요도 가져오기
        if self.feature_names:
            self.feature_importance = {
                self.feature_names[i]: importance 
                for i, importance in enumerate(self.model.feature_importances_)
            }
        else:
            self.feature_importance = {
                f"feature_{i}": importance 
                for i, importance in enumerate(self.model.feature_importances_)
            }
        
        # 학습 평가 지표 계산
        y_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        train_precision = precision_score(y_train, y_pred, average='weighted')
        train_recall = recall_score(y_train, y_pred, average='weighted')
        train_f1 = f1_score(y_train, y_pred, average='weighted')
        
        # 평가 지표 저장
        metrics = {
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'training_time': (pd.Timestamp.now() - start_time).total_seconds()
        }
        
        self.metrics.update(metrics)
        self.is_trained = True
        self.last_update = pd.Timestamp.now()
        self.classes_ = self.model.classes_
        
        self.logger.info(f"훈련 완료. 정확도: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        return metrics
    
    def predict(self, 
               X: np.ndarray,
               **kwargs) -> np.ndarray:
        """
        모델을 사용하여 예측 수행
        
        매개변수:
            X (np.ndarray): 입력 특성
            **kwargs: 추가 매개변수
            
        반환값:
            np.ndarray: 예측된 시장 상태 (0: 약세, 1: 횡보, 2: 강세)
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다. 예측이 부정확할 수 있습니다.")
        
        return self.model.predict(X)
    
    def predict_proba(self, 
                    X: np.ndarray, 
                    **kwargs) -> np.ndarray:
        """
        클래스 확률 예측
        
        매개변수:
            X (np.ndarray): 입력 특성
            **kwargs: 추가 매개변수
            
        반환값:
            np.ndarray: 클래스 확률
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다. 예측이 부정확할 수 있습니다.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        모델 평가
        
        매개변수:
            X_test (np.ndarray): 테스트 특성
            y_test (np.ndarray): 테스트 타겟
            **kwargs: 추가 매개변수
            
        반환값:
            Dict[str, Any]: 평가 지표
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다. 평가가 부정확할 수 있습니다.")
        
        # 예측 수행
        y_pred = self.predict(X_test)
        
        # 평가 지표 계산
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 평가 지표 저장
        metrics = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        
        self.metrics.update(metrics)
        self.logger.info(f"평가 완료. 정확도: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        return metrics
    
    def get_market_state(self, 
                       X: np.ndarray,
                       return_name: bool = True,
                       **kwargs) -> Union[np.ndarray, List[str]]:
        """
        예측된 시장 상태 가져오기
        
        매개변수:
            X (np.ndarray): 입력 특성
            return_name (bool): True인 경우 상태 이름 반환, False인 경우 인덱스 반환
            **kwargs: 추가 매개변수
            
        반환값:
            Union[np.ndarray, List[str]]: 예측된 시장 상태 인덱스 또는 이름
        """
        state_indices = self.predict(X)
        
        if return_name:
            return [self.market_states.get(idx, f"unknown_{idx}") for idx in state_indices]
        else:
            return state_indices

    def save(self, filepath: Optional[str] = None) -> str:
        """
        모델을 디스크에 저장
        
        매개변수:
            filepath (Optional[str]): 모델을 저장할 경로. None인 경우 기본 경로 사용
            
        반환값:
            str: 모델이 저장된 경로
        """
        if filepath is None:
            # 모델 이름, 유형 및 타임스탬프를 사용하여 기본 파일 이름 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{self.model_type}_{timestamp}.pkl"
            
            # 모델 디렉토리 확인 및 생성
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
            os.makedirs(model_dir, exist_ok=True)
            
            filepath = os.path.join(model_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            
            self.logger.info(f"모델이 {filepath}에 저장되었습니다.")
            
            # 쉽게 검사할 수 있도록 JSON 형식으로 메타데이터도 저장
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            
            # 직렬화 가능한 형태로 메타데이터 변환
            params_serializable = {}
            for k, v in self.params.items():
                if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                    params_serializable[k] = v
                else:
                    params_serializable[k] = str(v)
            
            metadata = {
                'name': self.name,
                'type': self.model_type,
                'version': self.version,
                'creation_date': self.creation_date.strftime("%Y-%m-%d %H:%M:%S") if hasattr(self, 'creation_date') else None,
                'last_update': self.last_update.strftime("%Y-%m-%d %H:%M:%S"),
                'is_trained': self.is_trained,
                'metrics': self.metrics,
                'params': params_serializable,
                'feature_importance': self.feature_importance if hasattr(self, 'feature_importance') else {}
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            return filepath
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'RandomForestMarketStateModel':
        """
        디스크에서 모델 로드
        
        매개변수:
            filepath (str): 저장된 모델의 경로
            
        반환값:
            RandomForestMarketStateModel: 로드된 모델
        """
        logger = get_logger(__name__)
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"모델이 {filepath}에서 로드되었습니다.")
            return model
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class RandomForestPriceModel(RegressionModel):
    """가격을 예측하기 위한 Random Forest 모델"""
    
    def __init__(self, 
                name: str = "RandomForestPrice", 
                version: str = "1.0.0",
                n_estimators: int = 75,  # 100에서 75로 최적화
                max_depth: Optional[int] = 10,   # None에서 10으로 변경
                min_samples_split: int = 20,     # 2에서 20으로 변경 
                min_samples_leaf: int = 10,      # 1에서 10으로 변경
                random_state: int = 42,
                bootstrap: bool = True,         # 데이터 부트스트래핑 활성화
                max_features: str = 'sqrt'):    # 분할에 고려할 최대 특성 수 제한
        """
        Random Forest 가격 예측 모델 초기화
        
        매개변수:
            name (str): 모델 이름
            version (str): 모델 버전
            n_estimators (int): 포레스트의 트리 개수
            max_depth (Optional[int]): 트리의 최대 깊이
            min_samples_split (int): 노드 분할에 필요한 최소 샘플 수
            min_samples_leaf (int): 리프 노드에 필요한 최소 샘플 수
            random_state (int): 재현성을 위한 랜덤 시드
            bootstrap (bool): 부트스트래핑 적용 여부
            max_features (str): 각 분할에 고려할 최대 특성 수
        """
        super().__init__(name, version)
        
        # 로거 초기화
        self.logger = get_logger(__name__)
        
        # 모델 초기화
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            bootstrap=bootstrap,
            max_features=max_features,
            n_jobs=-1
        )
        
        # 매개변수 저장
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'bootstrap': bootstrap,
            'max_features': max_features
        }
        
        # 예측 신뢰도를 계산하기 위한 가중치 정의
        self.confidence_weights = {
            'r2_score': 0.4,
            'model_depth': 0.3,
            'prediction_margin': 0.3
        }
        
        self.feature_names = None
        self.metrics = {}  # 성능 메트릭 저장 딕셔너리
        self.logger.info(f"{self.name} 모델을 {n_estimators}개의 트리, max_depth={max_depth}로 초기화했습니다.")
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray, 
             feature_names: Optional[List[str]] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Random Forest 가격 예측 모델 훈련
        
        매개변수:
            X_train (np.ndarray): 학습 특성
            y_train (np.ndarray): 학습 타겟 (가격)
            feature_names (Optional[List[str]]): 특성 이름 목록
            **kwargs: 추가 매개변수
                - sample_weight (np.ndarray): 샘플 가중치
                
        반환값:
            Dict[str, Any]: 학습 평가 지표
        """
        start_time = pd.Timestamp.now()
        self.logger.info(f"{self.name} 모델을 {X_train.shape[0]}개의 샘플로 훈련합니다.")
        
        # 문자열 데이터 감지 및 제거
        original_shape = X_train.shape
        X_train_cleaned, removed_indices = self._remove_string_columns(X_train)
        
        # 피처명 업데이트
        if original_shape[1] != X_train_cleaned.shape[1]:
            self.logger.info(f"X_train 형태가 {original_shape}에서 {X_train_cleaned.shape}로 변경되었습니다. 문자열 열 {len(removed_indices)}개 제거.")
            
            if feature_names is not None:
                updated_feature_names = [
                    name for i, name in enumerate(feature_names) 
                    if i not in removed_indices
                ]
                self.logger.info(f"{len(feature_names) - len(updated_feature_names)}개의 feature 이름이 제거되었습니다.")
                feature_names = updated_feature_names
        else:
            X_train_cleaned = X_train
        
        # 특성 이름 저장
        self.feature_names = feature_names
        
        # 모델 훈련
        sample_weight = kwargs.get('sample_weight', None)
        
        try:
            self.model.fit(X_train_cleaned, y_train, sample_weight=sample_weight)
            
            # 특성 중요도 가져오기
            if self.feature_names:
                self.feature_importance = {
                    self.feature_names[i]: importance 
                    for i, importance in enumerate(self.model.feature_importances_)
                }
            else:
                self.feature_importance = {
                    f"feature_{i}": importance 
                    for i, importance in enumerate(self.model.feature_importances_)
                }
            
            # 학습 평가 지표 계산
            y_pred = self.model.predict(X_train_cleaned)
            train_mse = mean_squared_error(y_train, y_pred)
            train_rmse = np.sqrt(train_mse)
            train_mae = mean_absolute_error(y_train, y_pred)
            train_r2 = r2_score(y_train, y_pred)
            
            # 평가 지표 저장
            metrics = {
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'training_time': (pd.Timestamp.now() - start_time).total_seconds()
            }
            
            self.metrics.update(metrics)
            self.is_trained = True
            self.last_update = pd.Timestamp.now()
            
            self.logger.info(f"훈련 완료. RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"모델 훈련 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
            
    def predict(self, 
               X: np.ndarray,
               **kwargs) -> np.ndarray:
        """
        모델을 사용하여 예측 수행
        
        매개변수:
            X (np.ndarray): 입력 특성
            **kwargs: 추가 매개변수
            
        반환값:
            np.ndarray: 예측된 가격
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다. 예측이 부정확할 수 있습니다.")
        
        # 문자열 열 제거 (훈련 시와 동일한 전처리)
        X_cleaned, _ = self._remove_string_columns(X)
        return self.model.predict(X_cleaned)
    
    def predict_interval(self, 
                        X: np.ndarray, 
                        confidence: float = 0.95, 
                        n_samples: int = 100,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        신뢰 구간을 포함한 예측 (부트스트래핑 사용)
        
        매개변수:
            X (np.ndarray): 입력 특성
            confidence (float): 신뢰 수준 (0.0-1.0)
            n_samples (int): 부트스트랩 샘플 수
            **kwargs: 추가 매개변수
            
        반환값:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 예측, 하한, 상한
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다. 예측이 부정확할 수 있습니다.")
        
        # 문자열 열 제거 (훈련 시와 동일한 전처리)
        X_cleaned, _ = self._remove_string_columns(X)
        
        # 랜덤 포레스트에서 추정기 가져오기
        estimators = self.model.estimators_
        
        # 각 트리로 예측 수행
        predictions = np.array([tree.predict(X_cleaned) for tree in estimators])
        
        # 평균 예측 계산
        y_pred = np.mean(predictions, axis=0)
        
        # 신뢰 구간 계산
        alpha = (1 - confidence) / 2
        lower_bound = np.percentile(predictions, 100 * alpha, axis=0)
        upper_bound = np.percentile(predictions, 100 * (1 - alpha), axis=0)
        
        return y_pred, lower_bound, upper_bound
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        모델 평가
        
        매개변수:
            X_test (np.ndarray): 테스트 특성
            y_test (np.ndarray): 테스트 타겟
            **kwargs: 추가 매개변수
            
        반환값:
            Dict[str, Any]: 평가 지표
        """
        if not self.is_trained:
            self.logger.warning("모델이 아직 훈련되지 않았습니다. 평가가 부정확할 수 있습니다.")
        
        # 문자열 열 제거 (훈련 시와 동일한 전처리)
        X_test_cleaned, _ = self._remove_string_columns(X_test)
        
        # 예측 수행
        y_pred = self.predict(X_test_cleaned)
        
        # 평가 지표 계산
        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        # 방향성 정확도 계산 (가격 변동용)
        if len(y_test) > 1:
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(actual_direction == pred_direction)
        else:
            directional_accuracy = None
        
        # 평가 지표 저장
        metrics = {
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'directional_accuracy': directional_accuracy
        }
        
        self.metrics.update(metrics)
        self.logger.info(f"평가 완료. RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        if directional_accuracy is not None:
            self.logger.info(f"방향성 정확도: {directional_accuracy:.4f}")
            
        return metrics

    def optimize_hyperparams(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           param_grid: Optional[Dict[str, List[Any]]] = None,
                           cv: int = 3,
                           n_iter: int = 10,
                           method: str = 'random',
                           **kwargs) -> Dict[str, Any]:
        """
        회귀 모델 하이퍼파라미터 최적화
        
        매개변수:
            X_train (np.ndarray): 학습 데이터
            y_train (np.ndarray): 학습 타겟
            X_val (np.ndarray): 검증 데이터
            y_val (np.ndarray): 검증 타겟
            param_grid (Optional[Dict[str, List[Any]]]): 하이퍼파라미터 그리드
            cv (int): 교차검증 폴드 수
            n_iter (int): 랜덤 서치 시 반복 횟수
            method (str): 최적화 방법 ('grid' 또는 'random')
            **kwargs: 추가 매개변수
            
        반환값:
            Dict[str, Any]: 최적 파라미터 및 성능 지표
        """
        self.logger.info(f"{method} 서치를 사용하여 하이퍼파라미터 최적화 시작")
        
        # 문자열 열 제거
        X_train_cleaned, _ = self._remove_string_columns(X_train)
        X_val_cleaned, _ = self._remove_string_columns(X_val)
        
        if param_grid is None:
            # 개선된 기본값 사용
            param_grid = {
                'n_estimators': [50, 75, 100],  # 더 효율적인 트리 개수 범위
                'max_depth': [5, 7, 10, 15],    # 더 세밀한 깊이 조정
                'min_samples_split': [10, 20, 30],  # 더 높은 값 사용
                'min_samples_leaf': [5, 10, 15],  # 더 높은 값 사용
                'max_features': ['sqrt', 'log2', 0.3]
            }
        
        try:
            # 기본 모델 생성
            base_model = RandomForestRegressor(random_state=self.params['random_state'], n_jobs=-1)
            
            # 최적화 객체 생성
            if method == 'grid':
                search = GridSearchCV(
                    base_model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
            else:  # 랜덤 서치
                search = RandomizedSearchCV(
                    base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=self.params['random_state'],
                    verbose=1
                )
            
            # 최적화 실행
            search.fit(X_train_cleaned, y_train)
            
            # 최적 모델로 업데이트
            self.model = search.best_estimator_
            self.params.update(search.best_params_)
            self.is_trained = True
            self.last_update = pd.Timestamp.now()
            
            # 검증 데이터에서 평가
            val_metrics = self.evaluate(X_val_cleaned, y_val)
            
            self.logger.info(f"하이퍼파라미터 최적화 완료. 최적 파라미터: {search.best_params_}")
            return {
                'best_params': search.best_params_,
                'best_score': -search.best_score_, # neg_mean_squared_error를 MSE로 변환
                'val_metrics': val_metrics
            }
            
        except Exception as e:
            self.logger.error(f"하이퍼파라미터 최적화 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.logger.warning("기본 모델로 대체")
            
            # 개선된 기본 모델 생성
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                bootstrap=True,
                random_state=self.params['random_state'],
                n_jobs=-1
            )
            
            # 직접 훈련
            try:
                self.model.fit(X_train_cleaned, y_train)
                self.is_trained = True
                self.last_update = pd.Timestamp.now()
                
                # 검증 세트에서 평가
                val_metrics = self.evaluate(X_val_cleaned, y_val)
                
                # 기본 파라미터
                best_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'max_features': 'sqrt',
                    'bootstrap': True
                }
                self.params.update(best_params)
                
                self.logger.info(f"기본 모델 훈련 완료. 파라미터: {best_params}")
                
                return {
                    'best_params': best_params,
                    'best_score': val_metrics.get('test_rmse', 999999.0),
                    'val_metrics': val_metrics
                }
            except Exception as e2:
                self.logger.error(f"기본 모델 훈련 중 오류 발생: {str(e2)}")
                return {
                    'error': str(e2),
                    'best_params': {},
                    'best_score': 999999.0
                }

    def save(self, filepath: Optional[str] = None) -> str:
        """
        모델을 디스크에 저장
        
        매개변수:
            filepath (Optional[str]): 모델을 저장할 경로. None인 경우 기본 경로 사용
            
        반환값:
            str: 모델이 저장된 경로
        """
        if filepath is None:
            # 모델 이름, 유형 및 타임스탬프를 사용하여 기본 파일 이름 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{self.model_type}_{timestamp}.pkl"
            
            # 모델 디렉토리 확인 및 생성
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
            os.makedirs(model_dir, exist_ok=True)
            
            filepath = os.path.join(model_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            
            self.logger.info(f"모델이 {filepath}에 저장되었습니다.")
            
            # 쉽게 검사할 수 있도록 JSON 형식으로 메타데이터도 저장
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            
            # 직렬화 가능한 형태로 메타데이터 변환
            params_serializable = {}
            for k, v in self.params.items():
                if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                    params_serializable[k] = v
                else:
                    params_serializable[k] = str(v)
            
            metadata = {
                'name': self.name,
                'type': self.model_type,
                'version': self.version,
                'creation_date': self.creation_date.strftime("%Y-%m-%d %H:%M:%S") if hasattr(self, 'creation_date') else None,
                'last_update': self.last_update.strftime("%Y-%m-%d %H:%M:%S"),
                'is_trained': self.is_trained,
                'metrics': self.metrics,
                'params': params_serializable,
                'feature_importance': self.feature_importance if hasattr(self, 'feature_importance') else {}
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            return filepath
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'RandomForestPriceModel':
        """
        디스크에서 모델 로드
        
        매개변수:
            filepath (str): 저장된 모델의 경로
            
        반환값:
            RandomForestPriceModel: 로드된 모델
        """
        logger = get_logger(__name__)
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"모델이 {filepath}에서 로드되었습니다.")
            return model
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise