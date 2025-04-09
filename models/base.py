"""
Model Base Module for Bitcoin Trading Bot

This module defines the abstract base classes for all models used in the trading system.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple, TYPE_CHECKING
from datetime import datetime
import time
import logging
import traceback

from config import settings
from utils.logging import get_logger
from utils.data_utils import clean_dataframe
from utils.constants import SignalType
from models.signal import ModelOutput, TradingSignal, standardize_model_output

# Initialize logger
logger = get_logger(__name__)

class ModelBase(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self, name: str, model_type: str = "base", version: str = "1.0.0"):
        """
        기본 모델 초기화
        
        Args:
            name (str): 모델 이름
            model_type (str): 모델 유형 ("classification", "regression", "base" 등)
            version (str): 모델 버전
        """
        self.name = name
        self.model_type = model_type
        self.version = version
        self.creation_date = datetime.now()
        self.last_update = self.creation_date
        self.is_trained = False
        self.metrics = {}
        self.params = {}
        self.feature_importance = {}
        self.logger = logger
        
        # settings 모듈에서 모델 디렉토리 경로 가져오기
        self.model_dir = settings.MODELS_DIR
        self.model_checkpoints_dir = settings.MODEL_CHECKPOINTS_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.model_checkpoints_dir, exist_ok=True)
        
        self.logger.info(f"Initialized {self.name} model of type {self.model_type}, version {self.version}")
        
        self.model = None
        # 모델 파일 경로를 settings에 정의된 경로 사용하도록 수정
        self.model_path = os.path.join(self.model_dir, f"{self.name}_{self.version}.joblib")
        
        # 특성 관리 기능 초기화
        try:
            from utils.feature_manager import FeatureManager
            self.feature_manager = FeatureManager(self.name, self.version)
        except ImportError:
            self.logger.warning("FeatureManager를 가져올 수 없습니다. 특성 관리 기능이 제한됩니다.")
            self.feature_manager = None
    
    @abstractmethod
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray, 
             **kwargs) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            **kwargs: Additional parameters for training
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, 
               X: np.ndarray, 
               **kwargs) -> ModelOutput:
        """
        Make predictions with the model
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional parameters for prediction
            
        Returns:
            ModelOutput: Standardized model output containing prediction results
        """
        pass
    
    @abstractmethod
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            **kwargs: Additional parameters for evaluation
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        pass
    
    def _preprocess_input(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, List[int]]:
        """
        데이터 전처리: DataFrame을 numpy 배열로 변환, 문자열 열 제거, NaN 처리 등
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): 입력 데이터
            
        Returns:
            Tuple[np.ndarray, List[int]]: 전처리된 데이터와 제거된 열 인덱스 목록
        """
        if X is None:
            self.logger.error("입력 데이터가 None입니다.")
            return np.array([]), []
        
        # 데이터프레임인 경우 clean_dataframe 함수로 전처리
        if isinstance(X, pd.DataFrame):
            self.logger.debug("DataFrame 전처리 시작")
            
            # clean_dataframe 함수 사용하여 데이터 정리
            X_clean = clean_dataframe(X, handle_missing='zero')
            
            # 전처리 후 numpy 배열로 변환
            X_array = X_clean.to_numpy()
            
            # 제거된 열의 인덱스 추적
            original_columns = set(X.columns)
            cleaned_columns = set(X_clean.columns)
            removed_columns = original_columns - cleaned_columns
            removed_indices = [list(X.columns).index(col) for col in removed_columns]
            
            self.logger.debug(f"DataFrame 전처리 완료: 원본 형태 {X.shape} -> 변환 형태 {X_array.shape}")
            return X_array, removed_indices
        
        # numpy 배열이 아닌 경우 변환 시도
        if not isinstance(X, np.ndarray):
            self.logger.warning(f"입력 데이터가 numpy 배열이 아닙니다. 타입: {type(X)}")
            try:
                X = np.array(X)
            except Exception as e:
                self.logger.error(f"입력 데이터를 numpy 배열로 변환할 수 없습니다: {str(e)}")
                return np.array([]), []
        
        # 배열 형태 확인 및 조정
        return self._remove_string_columns(X)
    
    def _remove_string_columns(self, X: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        데이터에서 문자열 열을 감지하고 제거
        
        Args:
            X (np.ndarray): 입력 데이터
            
        Returns:
            Tuple[np.ndarray, List[int]]: 문자열 열이 제거된 데이터와 제거된 열 인덱스 목록
        """
        # 1차원 배열인 경우 처리
        if len(X.shape) == 1:
            # 1차원 배열을 2차원으로 재구성
            X = X.reshape(1, -1)
            self.logger.debug(f"1차원 배열을 {X.shape} 형태로 재구성했습니다.")
        
        # 빈 배열인 경우 처리
        if X.size == 0:
            self.logger.warning("입력 데이터가 비어 있습니다.")
            return X, []
        
        # 각 열의 첫 번째 값 확인하여 문자열 열 탐지
        try:
            string_cols = []
            for i in range(X.shape[1]):
                # 열에 NaN 값만 있는지 확인
                if np.all(pd.isna(X[:, i])):
                    self.logger.warning(f"열 {i}에 NaN 값만 있습니다. NaN 값을 0으로 대체합니다.")
                    X[:, i] = 0
                    continue

                # 첫 번째 비-NaN 값 찾기
                for j in range(X.shape[0]):
                    if not pd.isna(X[j, i]):
                        try:
                            # 숫자로 변환 시도
                            float(X[j, i])
                            break  # 숫자로 변환 가능하면 다음 열로
                        except (ValueError, TypeError):
                            # 변환 실패 시 문자열 열로 간주
                            string_cols.append(i)
                            break
            
            if string_cols:
                self.logger.warning(f"문자열 열 {len(string_cols)}개가 감지되어 제거합니다: {string_cols}")
                # 문자열이 아닌 열만 선택
                mask = np.ones(X.shape[1], dtype=bool)
                for col in string_cols:
                    mask[col] = False
                
                # NaN 값을 0으로 대체하여 숫자 데이터만 포함하도록 함
                result = X[:, mask]
                result = np.nan_to_num(result, nan=0.0)
                return result, string_cols
            
            # NaN 값을 0으로 변환
            return np.nan_to_num(X, nan=0.0), []
        except Exception as e:
            self.logger.error(f"문자열 열 제거 중 오류 발생: {str(e)}")
            traceback.print_exc()
            # 안전하게 처리: 모든 NaN을 0으로 변환하고 원본 반환
            try:
                return np.nan_to_num(X, nan=0.0), []
            except:
                return X, []
    
    def save(self, custom_path: str = None) -> str:
        """
        모델 저장
        
        Args:
            custom_path (str, optional): 사용자 지정 경로. 기본값은 None.
            
        Returns:
            str: 저장된 모델 파일 경로
        """
        if self.model is None:
            self.logger.error("저장할 모델이 없습니다. 먼저 모델을 학습해주세요.")
            return ""
        
        # 저장 경로 결정
        save_path = custom_path if custom_path else self.model_path
        
        # 경로의 디렉토리 부분 확인 및 생성
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        try:
            # 모델 데이터 준비
            model_data = {
                'model': self.model,
                'metadata': {
                    'name': self.name,
                    'type': self.model_type,
                    'version': self.version,
                    'created': self.creation_date.isoformat(),
                    'updated': datetime.now().isoformat(),
                    'metrics': self.metrics,
                    'params': self.params,
                    'feature_importance': self.feature_importance
                }
            }
            
            # 모델 저장
            joblib.dump(model_data, save_path)
            self.logger.info(f"모델이 저장되었습니다: {save_path}")
            
            # 체크포인트 저장 (버전 관리)
            if not custom_path:  # 사용자 지정 경로가 아닌 경우에만 체크포인트 저장
                checkpoint_path = os.path.join(
                    self.model_checkpoints_dir,
                    f"{self.name}_{self.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                )
                joblib.dump(model_data, checkpoint_path)
                self.logger.debug(f"모델 체크포인트가 저장되었습니다: {checkpoint_path}")
                
            return save_path
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return ""
    
    def load(self, custom_path: str = None) -> bool:
        """
        모델 로드
        
        Args:
            custom_path (str, optional): 사용자 지정 경로. 기본값은 None.
            
        Returns:
            bool: 로드 성공 여부
        """
        # 로드 경로 결정
        load_path = custom_path if custom_path else self.model_path
        
        try:
            if not os.path.exists(load_path):
                self.logger.error(f"모델 파일을 찾을 수 없습니다: {load_path}")
                return False
            
            # 모델 로드
            model_data = joblib.load(load_path)
            
            # 모델 데이터 구조 확인
            if isinstance(model_data, dict) and 'model' in model_data and 'metadata' in model_data:
                # 새 형식 (메타데이터 포함)
                self.model = model_data['model']
                
                # 메타데이터 업데이트
                metadata = model_data['metadata']
                self.name = metadata.get('name', self.name)
                self.model_type = metadata.get('type', self.model_type)
                self.version = metadata.get('version', self.version)
                
                # 날짜 형식 처리
                try:
                    self.creation_date = datetime.fromisoformat(metadata.get('created', self.creation_date.isoformat()))
                except (ValueError, TypeError):
                    pass
                
                try:
                    self.last_update = datetime.fromisoformat(metadata.get('updated', self.last_update.isoformat()))
                except (ValueError, TypeError):
                    pass
                
                self.metrics = metadata.get('metrics', {})
                self.params = metadata.get('params', {})
                self.feature_importance = metadata.get('feature_importance', {})
            else:
                # 구 형식 (모델만 저장됨)
                self.model = model_data
            
            self.is_trained = True
            self.logger.info(f"모델을 로드했습니다: {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return False

    def save_features(self, feature_names: List[str], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        특성 이름 목록을 저장
        
        Args:
            feature_names (List[str]): 저장할 특성 이름 목록
            metadata (Optional[Dict[str, Any]]): 함께 저장할 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        # 특성 관리자 사용
        if hasattr(self, 'feature_manager') and self.feature_manager:
            return self.feature_manager.save_features(feature_names, metadata)
        
        # 역호환성을 위한 기본 구현
        self.feature_names = feature_names
        self.logger.info(f"특성 목록 저장됨: {len(feature_names)}개 (feature_manager 없음)")
        return True
    
    def align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        입력 데이터프레임의 특성을 모델에 맞게 정렬
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            
        Returns:
            pd.DataFrame: 정렬된 데이터프레임
        """
        # 특성 관리자 사용
        if hasattr(self, 'feature_manager') and self.feature_manager:
            return self.feature_manager.align_features(df)
            
        # 역호환성을 위한 기본 구현
        if hasattr(self, 'feature_names') and self.feature_names:
            self.logger.info(f"특성 정렬: {df.shape[1]}개 -> {len(self.feature_names)}개")
            return df.reindex(columns=self.feature_names, fill_value=0)
        else:
            self.logger.warning("특성 이름이 없어 정렬을 건너뜁니다.")
            return df
    
    def validate_features(self, features: Union[List[str], pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        입력 특성이 예상 특성과 일치하는지 검증
        
        Args:
            features (Union[List[str], pd.DataFrame, np.ndarray]): 검증할 특성 목록 또는 데이터
            
        Returns:
            Dict[str, Any]: 검증 결과
        """
        # 특성 관리자 사용
        if hasattr(self, 'feature_manager') and self.feature_manager:
            return self.feature_manager.validate_features(features)
            
        # 역호환성을 위한 기본 구현
        if hasattr(self, 'feature_names') and self.feature_names:
            current_features = []
            if isinstance(features, pd.DataFrame):
                current_features = features.columns.tolist()
            elif isinstance(features, list):
                current_features = features
            
            missing_features = [f for f in self.feature_names if f not in current_features]
            extra_features = [f for f in current_features if f not in self.feature_names]
            
            if missing_features:
                self.logger.warning(f"누락된 특성: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            
            return {
                'is_valid': len(missing_features) == 0,
                'missing_features': missing_features,
                'extra_features': extra_features
            }
        else:
            self.logger.warning("예상 특성 이름이 없어 검증을 건너뜁니다.")
            return {'is_valid': False, 'reason': 'No expected features'}


class ClassificationModel(ModelBase):
    """Base class for classification models"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize classification model
        
        Args:
            name (str): Model name
            version (str): Model version
        """
        super().__init__(name, "classification", version)
        self.classes_ = None
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional parameters
            
        Returns:
            np.ndarray: Class probabilities
        """
        if not self.is_trained:
            logger.warning("모델이 학습되지 않았습니다.")
            return np.array([])
            
        try:
            # 입력 데이터가 2D가 아닌 경우 reshape
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
                
            # 문자열 컬럼 제거
            X = self._remove_string_columns(X)[0]
            
            return self.model.predict_proba(X)
            
        except Exception as e:
            logger.error(f"확률 예측 중 오류 발생: {str(e)}")
            return np.array([])
            
    def _remove_string_columns(self, X: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """문자열 컬럼 제거"""
        try:
            # X가 None인 경우 예외 처리
            if X is None:
                logger.error("입력 데이터가 None입니다.")
                return np.array([]), []
            
            # 숫자형 컬럼만 선택
            if hasattr(X, 'dtype') and hasattr(X.dtype, 'fields') and X.dtype.fields is not None:
                numeric_mask = np.array([np.issubdtype(dtype, np.number) for dtype in X.dtype.fields.values()])
            else:
                # 구조화되지 않은 배열이거나 필드가 없는 경우, 모든 열을 유지
                if len(X.shape) > 1:
                    numeric_mask = np.ones(X.shape[1], dtype=bool)
                else:
                    # 1차원 배열이면 그대로 반환
                    return X, []
                
            return X[:, numeric_mask], []
            
        except Exception as e:
            logger.error(f"문자열 컬럼 제거 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 데이터 반환
            if X is not None and len(X.shape) > 1:
                return X, []
            return np.array([]), []


class Signal:
    """
    Trading Signal class
    
    Represents a trading signal with signal type, confidence level, reason, and metadata.
    """
    
    def __init__(self, 
                signal_type: str = "HOLD",
                confidence: float = 0.0,
                reason: str = "",
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new signal
        
        Args:
            signal_type (str): Signal type ("BUY", "SELL", "HOLD")
            confidence (float): Signal confidence level (0.0-1.0)
            reason (str): Reason for the signal
            metadata (Optional[Dict[str, Any]]): Additional signal metadata
        """
        self.signal_type = signal_type
        self.confidence = confidence
        self.reason = reason
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        
    def __str__(self) -> str:
        """String representation of the signal"""
        return f"Signal({self.signal_type}, confidence={self.confidence:.2f}, reason={self.reason})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the signal"""
        return (f"Signal(signal_type={self.signal_type}, confidence={self.confidence:.2f}, "
                f"reason={self.reason}, timestamp={self.timestamp}, metadata={self.metadata})")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert signal to dictionary
        
        Returns:
            Dict[str, Any]: Signal as dictionary
        """
        return {
            'signal': self.signal_type,
            'confidence': self.confidence,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """
        Create a signal from dictionary
        
        Args:
            data (Dict[str, Any]): Dictionary with signal data
            
        Returns:
            Signal: New signal instance
        """
        signal = cls(
            signal_type=data.get('signal', 'HOLD'),
            confidence=data.get('confidence', 0.0),
            reason=data.get('reason', ''),
            metadata=data.get('metadata', {})
        )
        
        # Parse timestamp if present
        if 'timestamp' in data:
            try:
                signal.timestamp = datetime.fromisoformat(data['timestamp'])
            except (ValueError, TypeError):
                signal.timestamp = datetime.now()
                
        return signal
    
    def is_valid(self) -> bool:
        """
        Check if the signal is valid
        
        Returns:
            bool: True if valid, False otherwise
        """
        # Signal must have a valid type and confidence
        if self.signal_type not in ["BUY", "SELL", "HOLD"]:
            return False
            
        if not (0.0 <= self.confidence <= 1.0):
            return False
            
        return True


class RegressionModel(ModelBase):
    """Base class for regression models"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize regression model
        
        Args:
            name (str): Model name
            version (str): Model version
        """
        super().__init__(name, "regression", version)
    
    def predict_interval(self, 
                        X: np.ndarray, 
                        confidence: float = 0.95, 
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals
        
        Args:
            X (np.ndarray): Input features
            confidence (float): Confidence level (0.0-1.0)
            **kwargs: Additional parameters
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Predictions, lower bounds, upper bounds
        """
        raise NotImplementedError("Subclasses must implement predict_interval")


class ReinforcementLearningModel(ModelBase):
    """Base class for reinforcement learning models"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize reinforcement learning model
        
        Args:
            name (str): Model name
            version (str): Model version
        """
        super().__init__(name, "reinforcement", version)
        self.action_space = None
        self.state_space = None
    
    @abstractmethod
    def act(self, state: np.ndarray, **kwargs) -> Tuple[int, Dict[str, Any]]:
        """
        Choose an action based on the current state
        
        Args:
            state (np.ndarray): Current state
            **kwargs: Additional parameters
            
        Returns:
            Tuple[int, Dict[str, Any]]: Action index and additional info
        """
        pass
    
    @abstractmethod
    def update(self, 
              state: np.ndarray, 
              action: int, 
              reward: float, 
              next_state: np.ndarray, 
              done: bool, 
              **kwargs) -> Dict[str, Any]:
        """
        Update the model with a step of experience
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode is done
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Update metrics
        """
        pass


class TimeSeriesModel(ModelBase):
    """Base class for time series models"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize time series model
        
        Args:
            name (str): Model name
            version (str): Model version
        """
        super().__init__(name, "time_series", version)
        self.sequence_length = None
        self.forecast_horizon = None
    
    @abstractmethod
    def forecast(self, 
                X: np.ndarray, 
                horizon: int, 
                **kwargs) -> np.ndarray:
        """
        Generate time series forecast
        
        Args:
            X (np.ndarray): Input sequences
            horizon (int): Forecast horizon
            **kwargs: Additional parameters
            
        Returns:
            np.ndarray: Forecasted values
        """
        pass


class EnsembleModel(ModelBase):
    """Base class for ensemble models"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize ensemble model
        
        Args:
            name (str): Model name
            version (str): Model version
        """
        super().__init__(name, "ensemble", version)
        self.models = []
        self.weights = []
    
    def add_model(self, model: ModelBase, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble
        
        Args:
            model (ModelBase): Model to add
            weight (float): Weight for this model
        """
        self.models.append(model)
        self.weights.append(weight)
        self.last_update = datetime.now()
        self.logger.info(f"Added {model.name} to ensemble with weight {weight}")
    
    def remove_model(self, model_index: int) -> None:
        """
        Remove a model from the ensemble
        
        Args:
            model_index (int): Index of model to remove
        """
        if 0 <= model_index < len(self.models):
            model_name = self.models[model_index].name
            self.models.pop(model_index)
            self.weights.pop(model_index)
            self.last_update = datetime.now()
            self.logger.info(f"Removed {model_name} from ensemble")
        else:
            self.logger.warning(f"Invalid model index: {model_index}")
    
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0"""
        if not self.weights:
            return
            
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
            self.logger.info(f"Normalized ensemble weights: {self.weights}")


class GPTAnalysisModel(ModelBase):
    """Base class for GPT-based analysis models"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize GPT analysis model
        
        Args:
            name (str): Model name
            version (str): Model version
        """
        super().__init__(name, "gpt_analysis", version)
        self.prompt_template = ""
    
    @abstractmethod
    def analyze(self, 
               market_data: Dict[str, Any], 
               **kwargs) -> Dict[str, Any]:
        """
        Analyze market data using GPT model
        
        Args:
            market_data (Dict[str, Any]): Market data to analyze
            **kwargs: Additional parameters for analysis
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        pass
    
    def set_prompt_template(self, template: str) -> None:
        """
        Set prompt template for GPT analysis
        
        Args:
            template (str): Prompt template
        """
        self.prompt_template = template
        self.logger.info("Prompt template set")
    
    def format_prompt(self, market_data: Dict[str, Any]) -> str:
        """
        Format prompt using template and market data
        
        Args:
            market_data (Dict[str, Any]): Market data to include in prompt
            
        Returns:
            str: Formatted prompt
        """
        # Basic implementation, should be overridden in specific models
        if not self.prompt_template:
            return str(market_data)
        return self.prompt_template


# 유틸리티 함수
def get_feature_importance_sorted(model: ModelBase,
                                 top_n: Optional[int] = None,
                                 threshold: Optional[float] = None) -> Dict[str, float]:
    """
    모델의 피처 중요도를 정렬하여 반환
    
    Args:
        model: 모델 객체 (feature_importance 속성을 가진)
        top_n (Optional[int]): 반환할 상위 n개 피처 수
        threshold (Optional[float]): 중요도가 threshold보다 큰 피처만 반환
        
    Returns:
        Dict[str, float]: 정렬된 피처 중요도
    """
    if not hasattr(model, 'feature_importance') or not model.feature_importance:
        logger.warning("피처 중요도 정보가 없습니다")
        return {}
    
    # 피처 중요도 정렬
    sorted_importance = dict(sorted(
        model.feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    # 임계값 적용 (지정된 경우)
    if threshold is not None:
        sorted_importance = {k: v for k, v in sorted_importance.items() if v >= threshold}
    
    # 상위 N개 반환 (지정된 경우)
    if top_n is not None:
        return dict(list(sorted_importance.items())[:top_n])
    
    return sorted_importance 