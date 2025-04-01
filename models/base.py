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

from utils.logging import get_logger

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 파일 핸들러 추가
if not logger.handlers:
    fh = logging.FileHandler('logs/models.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class ModelBase(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self, name: str, model_type: str, version: str = "1.0.0"):
        """
        Initialize the model base
        
        Args:
            name (str): Model name
            model_type (str): Type of model (e.g., 'classification', 'regression', 'reinforcement')
            version (str): Model version
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
        
        # Create model directory if it doesn't exist
        self.model_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.logger.info(f"Initialized {self.name} model of type {self.model_type}, version {self.version}")
        
        self.model = None
        self.model_path = os.path.join('models', 'saved', f"{self.name}_{self.version}.joblib")
    
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
               **kwargs) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional parameters for prediction
            
        Returns:
            np.ndarray: Predictions
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
    
    def _remove_string_columns(self, X: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        데이터에서 문자열 열을 감지하고 제거
        
        Args:
            X (np.ndarray): 입력 데이터
            
        Returns:
            Tuple[np.ndarray, List[int]]: 문자열 열이 제거된 데이터와 제거된 열 인덱스 목록
        """
        if X is None:
            self.logger.error("입력 데이터가 None입니다.")
            return np.array([]), []
        
        if not isinstance(X, np.ndarray):
            self.logger.warning(f"입력 데이터가 numpy 배열이 아닙니다. 타입: {type(X)}")
            return X, []
        
        # 1차원 배열인 경우 처리
        if len(X.shape) == 1:
            return X, []
        
        # 각 열의 첫 번째 값 확인하여 문자열 열 탐지
        try:
            string_cols = []
            for i in range(X.shape[1]):
                try:
                    # 열의 첫 번째 값을 float으로 변환 시도
                    float(X[0, i])
                except (ValueError, TypeError):
                    # 변환 실패 시 문자열 열로 간주
                    string_cols.append(i)
            
            if string_cols:
                self.logger.warning(f"문자열 열 {len(string_cols)}개가 감지되어 제거합니다.")
                # 문자열이 아닌 열만 선택
                mask = np.ones(X.shape[1], dtype=bool)
                for col in string_cols:
                    mask[col] = False
                return X[:, mask], string_cols
            
            return X, []
        except Exception as e:
            self.logger.error(f"문자열 열 제거 중 오류 발생: {str(e)}")
            return X, []
    
    def save(self) -> None:
        """모델 저장"""
        try:
            if not os.path.exists('models/saved'):
                os.makedirs('models/saved')
            
            joblib.dump(self.model, self.model_path)
            logger.info(f"모델 저장 완료: {self.model_path}")
            
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {str(e)}")
    
    def load(self) -> None:
        """모델 로드"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logger.info(f"모델 로드 완료: {self.model_path}")
            else:
                logger.warning(f"모델 파일이 존재하지 않음: {self.model_path}")
                
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        return self.params
    
    def set_params(self, **params) -> None:
        """
        Set model parameters
        
        Args:
            **params: Parameters to set
        """
        self.params.update(params)
        self.last_update = datetime.now()
        self.logger.info(f"Updated model parameters: {params}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance
        
        Returns:
            Dict[str, float]: Feature importance
        """
        return self.feature_importance
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata
        
        Returns:
            Dict[str, Any]: Model metadata
        """
        return {
            'name': self.name,
            'type': self.model_type,
            'version': self.version,
            'creation_date': self.creation_date,
            'last_update': self.last_update,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'params': self.params,
            'feature_importance': self.feature_importance
        }
    
    def summary(self) -> str:
        """
        Get a string summary of the model
        
        Returns:
            str: Model summary
        """
        summary = [
            f"Model: {self.name}",
            f"Type: {self.model_type}",
            f"Version: {self.version}",
            f"Trained: {self.is_trained}",
            "Parameters:"
        ]
        
        for k, v in self.params.items():
            summary.append(f"  - {k}: {v}")
        
        if self.metrics:
            summary.append("Metrics:")
            for k, v in self.metrics.items():
                summary.append(f"  - {k}: {v}")
        
        return "\n".join(summary)


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