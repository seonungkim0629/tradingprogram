"""
앙상블 모듈 (Ensemble Module)

이 모듈은 다양한 유형의 예측 모델을 통합하는 앙상블 시스템을 제공합니다.
GRU, RandomForest 등의 다양한 모델을 결합하여 더 나은 예측 결과를 생성합니다.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from datetime import datetime
import json
import traceback
import time

# 내부 모듈 임포트
from models.base import ModelBase, ClassificationModel, RegressionModel
from utils.logging import get_logger
from models.signal import TradingSignal, ModelOutput
from utils.constants import SignalType
from models.types import TrainingMetrics

# 앙상블 하위 모듈 임포트
from ensemble.ensemble_core import EnsembleBase, HybridEnsemble
from ensemble.scoring import calculate_model_performance, evaluate_ensemble_performance
from ensemble.weights import adjust_weights_based_on_performance, calculate_optimal_weights
from ensemble.market_context import adjust_weights_by_market_condition, extract_market_features
from ensemble.combiners import weighted_average_combiner, adaptive_weights_combiner

# 로거 초기화
logger = get_logger(__name__)


class TradingEnsemble(HybridEnsemble):
    """
    트레이딩용 하이브리드 앙상블 모델
    
    비트코인 거래 시스템을 위한 앙상블 모델로, 방향 예측(분류)과 가격 예측(회귀)을 결합합니다.
    시장 상황에 따라 적응적으로 가중치를 조정하고, 기술적 지표와 함께 최종 예측을 생성합니다.
    """
    
    def __init__(self, 
                name: str = "TradingEnsemble",
                version: str = "1.0.0",
                market: str = "KRW-BTC",
                timeframe: str = "day",
                direction_models: Optional[List[ClassificationModel]] = None,
                price_models: Optional[List[RegressionModel]] = None,
                confidence_threshold: float = 0.6,
                trend_weight: float = 0.3,
                ma_weight: float = 0.2,
                rsi_weight: float = 0.2,
                hourly_weight: float = 0.1,
                ml_weight: float = 0.2,
                use_ml_models: bool = True,
                use_market_context: bool = True,
                sequence_length: int = 10):
        """
        트레이딩 앙상블 초기화
        
        Args:
            name (str): 모델 이름
            version (str): 모델 버전
            market (str): 거래 시장 (예: "KRW-BTC")
            timeframe (str): 시간 프레임 (예: "day", "hour")
            direction_models (Optional[List[ClassificationModel]]): 방향 예측 모델 리스트
            price_models (Optional[List[RegressionModel]]): 가격 예측 모델 리스트
            confidence_threshold (float): 신호 생성 신뢰도 임계값
            trend_weight (float): 추세 지표 가중치
            ma_weight (float): 이동평균 지표 가중치
            rsi_weight (float): RSI 지표 가중치
            hourly_weight (float): 시간별 데이터 가중치
            ml_weight (float): 머신러닝 모델 가중치
            use_ml_models (bool): ML 모델 사용 여부
            use_market_context (bool): 시장 컨텍스트 사용 여부
            sequence_length (int): 시계열 데이터 처리를 위한 최소 필요 데이터 길이
        """
        super().__init__(name, version, direction_models, price_models)
        
        # 시장 정보 설정
        self.market = market
        self.timeframe = timeframe
        
        # 매개변수 설정
        self.confidence_threshold = confidence_threshold
        self.use_market_context = use_market_context
        self.sequence_length = sequence_length
        
        # [확장 예정] 기술적 지표 기반 전략 가중치
        # 향후 기술적 지표 기반 전략과 ML 모델을 통합할 때 사용될 예정
        # 현재는 ML 모델 앙상블에만 집중되어 있으나, 
        # 추세, 이동평균, RSI 등의 기술적 지표를 통합할 경우 핵심 역할 수행
        self.strategy_weights = {
            'trend': trend_weight,    # 추세 지표 가중치
            'ma': ma_weight,          # 이동평균 지표 가중치
            'rsi': rsi_weight,        # RSI 지표 가중치
            'hourly': hourly_weight,  # 시간별 데이터 가중치
            'ml': ml_weight           # ML 모델 가중치
        }
        
        # 사용 설정
        self.use_ml_models = use_ml_models
        
        # 성능 지표 및 매개변수 추가
        self.parameters = {
            'name': name,
            'version': version,
            'market': market,
            'timeframe': timeframe,
            'confidence_threshold': confidence_threshold,
            # [확장 예정] 기술적 지표 가중치 파라미터
            'trend_weight': trend_weight,
            'ma_weight': ma_weight,
            'rsi_weight': rsi_weight,
            'hourly_weight': hourly_weight,
            'ml_weight': ml_weight,
            'use_ml_models': use_ml_models,
            'use_market_context': use_market_context,
            'sequence_length': sequence_length
        }
        
        self.logger.info(f"{self.name} v{self.version} 트레이딩 앙상블 초기화 완료 (시장: {market}, 시간프레임: {timeframe})")
        
        # 추가 매개변수
        self.recent_performance = {
            'direction': [],
            'price': []
        }
        
        # 훈련과 검증을 위한 데이터 저장
        self.X_train = None
        self.validation_data = None
        
        # 성능 지표
        self.performance_metrics = {
            'direction': {},
            'price': {},
            'ensemble': {}
        }
        
        # 특성 이름 저장 경로
        self.features_path = os.path.join("data_storage", "models", f"{self.name}_features.json")
        self.expected_features = self._load_expected_features()
    
    def add_direction_model(self, model: ClassificationModel, weight: float = 1.0) -> None:
        """
        방향 예측 모델 추가
        
        Args:
            model (ClassificationModel): 추가할 모델
            weight (float): 모델 가중치
        """
        if not isinstance(model, ClassificationModel):
            raise TypeError("방향 모델은 ClassificationModel 타입이어야 합니다")
        
        self.direction_models.append(model)
        
        # 가중치 업데이트
        if self.direction_weights is None:
            self.direction_weights = [1.0] * len(self.direction_models)
        else:
            self.direction_weights.append(weight)
            self.direction_weights = self._normalize_weights(self.direction_weights)
        
        self.logger.info(f"방향 모델 {model.name}이 가중치 {weight}로 앙상블에 추가됨")
    
    def add_price_model(self, model: RegressionModel, weight: float = 1.0) -> None:
        """
        가격 예측 모델 추가
        
        Args:
            model (RegressionModel): 추가할 모델
            weight (float): 모델 가중치
        """
        if not isinstance(model, RegressionModel):
            raise TypeError("가격 모델은 RegressionModel 타입이어야 합니다")
        
        self.price_models.append(model)
        
        # 가중치 업데이트
        if self.price_weights is None:
            self.price_weights = [1.0] * len(self.price_models)
        else:
            self.price_weights.append(weight)
            self.price_weights = self._normalize_weights(self.price_weights)
        
        self.logger.info(f"가격 모델 {model.name}이 가중치 {weight}로 앙상블에 추가됨")
    
    def train(self, X_train, y_train_direction=None, y_train_price=None, 
              X_val=None, y_val_direction=None, y_val_price=None,
              feature_names=None, sample_weights=None, sequence_handling='last',
              min_accuracy_threshold=0.55, min_f1_threshold=0.5) -> TrainingMetrics:
        """
        Trains the ensemble model with the provided training data.
        
        Args:
            X_train: Training features.
            y_train_direction: Training labels for direction prediction.
            y_train_price: Training values for price prediction.
            X_val: Validation features.
            y_val_direction: Validation labels for direction prediction.
            y_val_price: Validation values for price prediction.
            feature_names: List of feature names.
            sample_weights: Weights for training samples.
            sequence_handling: How to handle 3D input data. Options:
                - 'last': Use only the last step in each sequence.
                - 'mean': Use the mean of each sequence.
                - 'flatten': Flatten the entire sequence (caution: creates many features).
            min_accuracy_threshold: Minimum accuracy threshold for a model to be considered trained.
            min_f1_threshold: Minimum F1 score threshold for a model to be considered trained.
            
        Returns:
            TrainingMetrics: 훈련 성과 지표를 포함하는 객체
        """
        start_time = time.time()
        
        # Handle 3D input data (sequence data)
        if len(X_train.shape) == 3:
            # Get original dimensions
            n_samples, n_steps, n_features = X_train.shape
            self.logger.info(f"3차원 입력 데이터 감지: {X_train.shape}")
            
            if sequence_handling == 'last':
                # Use only the last step of each sequence
                X_train = X_train[:, -1, :]
                if X_val is not None:
                    X_val = X_val[:, -1, :]
                self.logger.info(f"마지막 시퀀스만 사용하여 변환: {X_train.shape}")
                
            elif sequence_handling == 'mean':
                # Use the mean of each sequence
                X_train = np.mean(X_train, axis=1)
                if X_val is not None:
                    X_val = np.mean(X_val, axis=1)
                self.logger.info(f"시퀀스 평균값으로 변환: {X_train.shape}")
                
            elif sequence_handling == 'flatten':
                # Flatten each sequence (results in many features)
                X_train = X_train.reshape(n_samples, n_steps * n_features)
                if X_val is not None:
                    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])
                self.logger.info(f"시퀀스 평탄화로 변환: {X_train.shape}")
                
            else:
                raise ValueError(f"Unsupported sequence_handling option: {sequence_handling}. "
                                f"Use 'last', 'mean', or 'flatten'.")
        
        # Ensure y targets are 1D arrays if they are provided
        if y_train_direction is not None and len(y_train_direction.shape) > 1:
            y_train_direction = y_train_direction.flatten()
            
        if y_val_direction is not None and len(y_val_direction.shape) > 1:
            y_val_direction = y_val_direction.flatten()
            
        # Store the training data info
        training_data = {
            'X_shape': X_train.shape,
            'feature_count': X_train.shape[1],
            'sequence_handling': sequence_handling
        }
        
        direction_metrics = {}
        price_metrics = {}
        
        # Train direction models
        all_direction_models_trained = True
        for i, model in enumerate(self.direction_models, 1):
            model_name = getattr(model, 'name', f"model_{i}")
            self.logger.info(f"방향 모델 {i}/{len(self.direction_models)} 훈련 중: {model_name}")
            
            try:
                model_metrics = model.train(
                    X_train, y_train_direction, 
                    X_val, y_val_direction,
                    feature_names=feature_names, 
                    sample_weights=sample_weights,
                    min_accuracy_threshold=min_accuracy_threshold,
                    min_f1_threshold=min_f1_threshold
                )
                
                # Check if model meets minimum performance thresholds
                accuracy = model_metrics.get('accuracy', 0)
                f1_score = model_metrics.get('f1_score', 0)
                
                if accuracy < min_accuracy_threshold or f1_score < min_f1_threshold:
                    self.logger.warning(
                        f"모델 {model_name}의 성능이 기준 미달: 정확도={accuracy:.4f} (기준: {min_accuracy_threshold}), "
                        f"F1={f1_score:.4f} (기준: {min_f1_threshold})"
                    )
                    model.is_trained = False
                    all_direction_models_trained = False
                
                direction_metrics[model_name] = model_metrics
            except Exception as e:
                self.logger.error(f"방향 모델 {model_name} 훈련 오류: {str(e)}")
                model.is_trained = False
                all_direction_models_trained = False
                direction_metrics[model_name] = {'error': str(e)}
            
            if not model.is_trained:
                self.logger.warning(f"방향 모델 {model_name}이 훈련되지 않았습니다.")
                all_direction_models_trained = False
        
        # Train price models
        all_price_models_trained = True
        for i, model in enumerate(self.price_models, 1):
            model_name = getattr(model, 'name', f"price_model_{i}")
            if y_train_price is not None:
                self.logger.info(f"가격 모델 {i}/{len(self.price_models)} 훈련 중: {model_name}")
                
                try:
                    model_metrics = model.train(
                        X_train, y_train_price, 
                        X_val, y_val_price,
                        feature_names=feature_names, 
                        sample_weights=sample_weights,
                        min_r2_threshold=0.1  # 가격 모델의 최소 R² 임계값
                    )
                    
                    # Check if regression model meets minimum performance threshold
                    r2_score = model_metrics.get('r2_score', -float('inf'))
                    mae = model_metrics.get('mae', float('inf'))
                    
                    # For regression models, check different metrics
                    if r2_score < 0.1:  # Very low R² indicates poor fit
                        self.logger.warning(
                            f"가격 모델 {model_name}의 성능이 기준 미달: R²={r2_score:.4f} (기준: 0.1)"
                        )
                        model.is_trained = False
                        all_price_models_trained = False
                    
                    price_metrics[model_name] = model_metrics
                except Exception as e:
                    self.logger.error(f"가격 모델 {model_name} 훈련 오류: {str(e)}")
                    model.is_trained = False
                    all_price_models_trained = False
                    price_metrics[model_name] = {'error': str(e)}
                
                if not model.is_trained:
                    self.logger.warning(f"가격 모델 {model_name}이 훈련되지 않았습니다.")
                    all_price_models_trained = False
            else:
                # Skip price models if no price training data provided
                self.logger.warning(f"가격 모델 {model_name}이 훈련 데이터 없음으로 건너뜁니다.")
                model.is_trained = False
                all_price_models_trained = False
        
        # Save feature names for later validation
        if feature_names is not None and len(feature_names) > 0:
            self.save_features(feature_names)
        
        # 수정된 로직: 모든 모델이 성공적으로 훈련된 경우에만 앙상블을 훈련된 것으로 표시
        training_success = True
        
        # 방향 모델이 있는 경우, 모든 방향 모델이 훈련되어야 함
        if len(self.direction_models) > 0:
            training_success = training_success and all_direction_models_trained
        
        # 가격 모델이 있는 경우, 모든 가격 모델이 훈련되어야 함
        if len(self.price_models) > 0:
            training_success = training_success and all_price_models_trained
        
        self.is_trained = training_success
        
        if self.is_trained:
            self.logger.info("앙상블 모델 훈련 성공")
        else:
            self.logger.warning("앙상블 모델 훈련 실패")
        
        training_time = time.time() - start_time
        
        # 앙상블 모델의 종합 정확도와 F1 점수 계산
        avg_accuracy = 0.0
        avg_f1_score = 0.0
        count = 0
        
        for metrics in direction_metrics.values():
            if 'accuracy' in metrics and 'f1_score' in metrics:
                avg_accuracy += metrics['accuracy']
                avg_f1_score += metrics['f1_score']
                count += 1
        
        if count > 0:
            avg_accuracy /= count
            avg_f1_score /= count
        
        return TrainingMetrics(
            accuracy=avg_accuracy,
            f1_score=avg_f1_score,
            training_time=training_time,
            feature_count=X_train.shape[1],
            direction_metrics=direction_metrics,
            price_metrics=price_metrics,
            direction_models_trained=all_direction_models_trained,
            price_models_trained=all_price_models_trained
        )
    
    def predict(self, 
               X: np.ndarray,
               market_data: Optional[pd.DataFrame] = None,
               **kwargs) -> ModelOutput:
        """
        앙상블로 예측 수행
        
        Args:
            X (np.ndarray): 입력 특성
            market_data (Optional[pd.DataFrame]): 시장 데이터
            **kwargs: 추가 매개변수
            
        Returns:
            ModelOutput: 예측 결과
        """
        try:
            if not self.is_trained:
                self.logger.warning("앙상블이 아직 훈련되지 않았습니다")
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason="앙상블이 훈련되지 않음"
                    ),
                    confidence=0.0,
                    metadata={"error": "앙상블이 훈련되지 않음"}
                )
            
            # 입력 특성 수 검증 및 조정
            validated_X = self._validate_features(X)
            
            # _validate_features에서 오류가 반환된 경우 (딕셔너리 형태로 오류 반환)
            if isinstance(validated_X, dict) and 'error' in validated_X:
                error_msg = f"특성 검증 중 오류 발생: {validated_X['error']}"
                self.logger.error(error_msg)
                # 오류 메시지와 함께 HOLD 신호 반환
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason=f"오류로 인한 홀드: {error_msg}"
                    ),
                    confidence=0.0,
                    metadata={"error": error_msg}
                )
            
            X = validated_X  # 검증된 특성으로 업데이트
            
            # 모든 검증 후에도 형태가 맞지 않으면 에러 반환
            if not hasattr(self, 'expected_features_count'):
                error_msg = "기대 특성 수(expected_features_count)가 정의되지 않았습니다."
                self.logger.error(error_msg)
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason=f"오류로 인한 홀드: {error_msg}"
                    ),
                    confidence=0.0,
                    metadata={"error": error_msg}
                )
                
            if X.shape[1] != self.expected_features_count:
                error_msg = f"특성 수 불일치 오류: 예상={self.expected_features_count}, 실제={X.shape[1]}"
                self.logger.error(error_msg)
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason=f"특성 수 불일치: 예상={self.expected_features_count}, 실제={X.shape[1]}"
                    ),
                    confidence=0.0,
                    metadata={
                        "error": '특성 수 불일치 오류', 
                        "expected": self.expected_features_count, 
                        "actual": X.shape[1]
                    }
                )
            
            results = {
                'direction': None,
                'direction_proba': None,
                'price': None,
                'confidence': 0.0,
                'agreement': 0.0,
                'models_used': {
                    'direction': len(self.direction_models),
                    'price': len(self.price_models)
                }
            }
            
            # 시장 상황 분석 및 가중치 조정
            adjusted_direction_weights = self.direction_weights
            adjusted_price_weights = self.price_weights
            
            if self.use_market_context and market_data is not None:
                try:
                    market_features = extract_market_features(market_data)
                    market_volatility = market_features.get('volatility', 0.5)
                    trend_strength = market_features.get('trend_strength', 0.5)
                    
                    # 시장 상황에 따라 가중치 조정
                    adjusted_direction_weights = adjust_weights_by_market_condition(
                        self.direction_weights,
                        market_volatility,
                        trend_strength
                    )
                    
                    results['market_context'] = {
                        'volatility': market_volatility,
                        'trend_strength': trend_strength
                    }
                except Exception as e:
                    self.logger.error(f"시장 상황 분석 중 오류: {str(e)}")
            
            # 방향 예측
            if self.direction_models:
                direction_predictions = []
                direction_probas = []
                
                for model in self.direction_models:
                    try:
                        # 각 모델의 예측 및 확률 얻기
                        pred = model.predict(X)
                        proba = model.predict_proba(X)
                        direction_predictions.append(pred)
                        direction_probas.append(proba)
                    except Exception as e:
                        self.logger.error(f"모델 {model.name} 예측 중 오류: {str(e)}")
                
                if direction_probas:
                    # 가중 평균으로 확률 결합
                    ensemble_proba = weighted_average_combiner(
                        direction_probas, 
                        adjusted_direction_weights
                    )
                    
                    if ensemble_proba.shape[1] > 1:
                        # 다중 분류인 경우
                        direction_proba = ensemble_proba
                        direction = np.argmax(ensemble_proba, axis=1)
                        confidence = np.max(ensemble_proba, axis=1)
                    else:
                        # 이진 분류인 경우
                        direction_proba = ensemble_proba
                        direction = (ensemble_proba > 0.5).astype(int)
                        confidence = np.where(
                            ensemble_proba > 0.5,
                            ensemble_proba,
                            1 - ensemble_proba
                        )
                    
                    results['direction'] = direction
                    results['direction_proba'] = direction_proba
                    results['confidence'] = float(confidence[0])
                    
                    # 모델 간 합의율 계산
                    if len(direction_predictions) > 1:
                        agreement = np.mean([
                            np.mean(pred == direction) 
                            for pred in direction_predictions
                        ])
                        results['agreement'] = float(agreement)
            
            # 가격 예측 (가격 모델이 있는 경우)
            if self.price_models:
                price_predictions = []
                
                for model in self.price_models:
                    try:
                        pred = model.predict(X)
                        price_predictions.append(pred)
                    except Exception as e:
                        self.logger.error(f"모델 {model.name} 가격 예측 중 오류: {str(e)}")
                
                if price_predictions:
                    # 가중 평균으로 가격 결합
                    ensemble_price = np.zeros_like(price_predictions[0])
                    
                    for i, (pred, weight) in enumerate(zip(price_predictions, adjusted_price_weights)):
                        ensemble_price += pred * weight
                    
                    results['price'] = ensemble_price
            
            # 신뢰도가 임계값보다 낮으면 결과에 표시
            if results['confidence'] < self.confidence_threshold:
                results['low_confidence'] = True
            
            # 방향에 따른 신호 결정
            signal_type = SignalType.HOLD
            reason = "신호를 생성할 충분한 정보가 없습니다."
            
            if 'direction' in results and results['direction'] is not None:
                direction_value = results['direction'][0] if isinstance(results['direction'], np.ndarray) else results['direction']
                
                # 방향에 따른 신호 결정
                if direction_value == 1:  # 매수 신호
                    signal_type = SignalType.BUY
                    reason = "앙상블 모델이 상승 추세를 예측했습니다."
                elif direction_value == 0:  # 홀드 신호
                    signal_type = SignalType.HOLD
                    reason = "앙상블 모델이 횡보를 예측했습니다."
                elif direction_value == -1:  # 매도 신호
                    signal_type = SignalType.SELL
                    reason = "앙상블 모델이 하락 추세를 예측했습니다."
                
                # 낮은 신뢰도인 경우 신호를 홀드로 변경
                if results.get('low_confidence', False):
                    if signal_type != SignalType.HOLD:
                        reason += f" (신뢰도 낮음: {results['confidence']:.2f})"
                        
                        # 신뢰도가 매우 낮은 경우에만 신호를 HOLD로 변경
                        if results['confidence'] < self.confidence_threshold * 0.8:
                            signal_type = SignalType.HOLD
                            reason = f"앙상블 예측 신뢰도가 너무 낮습니다: {results['confidence']:.2f}"
            
            # 예측 가격 정보 추가
            predicted_price = None
            if 'price' in results and results['price'] is not None:
                predicted_price = float(results['price'][0]) if isinstance(results['price'], np.ndarray) else float(results['price'])
            
            # TradingSignal 객체 생성
            trading_signal = TradingSignal(
                signal_type=signal_type,
                confidence=results.get('confidence', 0.0),
                reason=reason,
                price=predicted_price,
                metadata={
                    "model_name": self.name,
                    "model_version": self.version,
                    "direction": results.get('direction'),
                    "direction_proba": results.get('direction_proba'),
                    "agreement": results.get('agreement'),
                    "models_used": results.get('models_used')
                }
            )
            
            # ModelOutput 객체 생성하여 반환
            return ModelOutput(
                signal=trading_signal,
                raw_predictions=results.get('direction_proba'),
                confidence=results.get('confidence', 0.0),
                metadata={
                    "model_name": self.name,
                    "model_version": self.version,
                    "model_type": "ensemble",
                    "prediction_time": datetime.now().isoformat(),
                    "market_data": results.get('market_context')
                }
            )
            
        except Exception as e:
            self.logger.error(f"앙상블 예측 중 오류: {str(e)}")
            self.logger.error(traceback.format_exc())
            return ModelOutput(
                signal=TradingSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    reason=f"예측 중 오류 발생: {str(e)}"
                ),
                confidence=0.0,
                metadata={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def _evaluate_on_validation(self) -> Dict[str, Any]:
        """
        검증 데이터로 앙상블 평가
        
        Returns:
            Dict[str, Any]: 검증 지표
        """
        # 구현 예정
        pass
    
    def _optimize_weights(self) -> None:
        """
        검증 데이터를 기반으로 앙상블 가중치 최적화
        """
        # 구현 예정
        pass
    
    def update_weights(self, 
                      direction_weights: Optional[List[float]] = None,
                      price_weights: Optional[List[float]] = None) -> None:
        """
        앙상블 가중치 수동 업데이트
        
        Args:
            direction_weights (Optional[List[float]]): 방향 모델 가중치
            price_weights (Optional[List[float]]): 가격 모델 가중치
        """
        if direction_weights is not None:
            if len(direction_weights) != len(self.direction_models):
                raise ValueError(f"방향 가중치 길이({len(direction_weights)})가 모델 수({len(self.direction_models)})와 일치하지 않습니다")
            self.direction_weights = self._normalize_weights(direction_weights)
            self.logger.info(f"방향 모델 가중치 업데이트됨: {self.direction_weights}")
        
        if price_weights is not None:
            if len(price_weights) != len(self.price_models):
                raise ValueError(f"가격 가중치 길이({len(price_weights)})가 모델 수({len(self.price_models)})와 일치하지 않습니다")
            self.price_weights = self._normalize_weights(price_weights)
            self.logger.info(f"가격 모델 가중치 업데이트됨: {self.price_weights}")
    
    def update_from_recent_performance(self, 
                                      direction_performance: Optional[List[float]] = None,
                                      price_performance: Optional[List[float]] = None,
                                      adapt_factor: float = 0.3) -> None:
        """
        최근 성능을 기반으로 가중치 업데이트
        
        Args:
            direction_performance (Optional[List[float]]): 방향 모델 성능 점수
            price_performance (Optional[List[float]]): 가격 모델 성능 점수
            adapt_factor (float): 적응 강도 (0-1 사이)
        """
        # 방향 예측 모델 가중치 업데이트
        if direction_performance and len(direction_performance) == len(self.direction_models):
            # 음수 성능 점수를 0.01로 조정 (최소 가중치 보장)
            adjusted_scores = [max(0.01, score + 0.1) for score in direction_performance]  # 기본 가중치 0.1 추가
            
            # 가중치 계산
            total_score = sum(adjusted_scores)
            if total_score > 0:
                # 새 가중치 계산
                new_weights = [score / total_score for score in adjusted_scores]
                
                # 현재 가중치와 새 가중치 혼합 (적응 강도 적용)
                updated_weights = []
                for i, current_weight in enumerate(self.direction_weights):
                    updated_weight = (1 - adapt_factor) * current_weight + adapt_factor * new_weights[i]
                    updated_weights.append(updated_weight)
                
                # 가중치 업데이트
                self.direction_weights = self._normalize_weights(updated_weights)
                self.logger.info(f"방향 모델 가중치 업데이트됨 (성능 기반): {self.direction_weights}")
        
        # 가격 예측 모델 가중치 업데이트
        if price_performance and len(price_performance) == len(self.price_models):
            # 음수 성능 점수를 0.01로 조정
            adjusted_scores = [max(0.01, score + 0.1) for score in price_performance]
            
            # 가중치 계산
            total_score = sum(adjusted_scores)
            if total_score > 0:
                # 새 가중치 계산
                new_weights = [score / total_score for score in adjusted_scores]
                
                # 현재 가중치와 새 가중치 혼합
                updated_weights = []
                for i, current_weight in enumerate(self.price_weights):
                    updated_weight = (1 - adapt_factor) * current_weight + adapt_factor * new_weights[i]
                    updated_weights.append(updated_weight)
                
                # 가중치 업데이트
                self.price_weights = self._normalize_weights(updated_weights)
                self.logger.info(f"가격 모델 가중치 업데이트됨 (성능 기반): {self.price_weights}")
        
        # 단일 모델에 대한 성능 업데이트
        if direction_performance and len(direction_performance) == 1 and len(self.direction_models) > 1:
            self.logger.warning("하나의 방향 모델에 대한 성능만 제공됨. 가중치 업데이트를 건너뜁니다.")
        
        if price_performance and len(price_performance) == 1 and len(self.price_models) > 1:
            self.logger.warning("하나의 가격 모델에 대한 성능만 제공됨. 가중치 업데이트를 건너뜁니다.")
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """거래 신호를 생성합니다.
        
        Args:
            data: 시장 데이터
            
        Returns:
            Dict[str, Any]: 거래 신호
        """
        try:
            self.logger.debug(f"generate_signal 호출됨. 입력 데이터 shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            
            # 0. 특성 준비 및 검증
            X = self._prepare_features(data)
            if X is None:
                return {'signal': SignalType.HOLD, 'confidence': 0.0, 'reason': '특성 준비 실패', 'timestamp': datetime.now()}
                
            # 특성 검증 (이 단계에서 predict 내부가 아닌, predict 호출 전에 검증)
            # validated_X = self._validate_features(X) # _validate_features는 현재 문제가 있으므로 predict 내부 검증에 의존
            # if isinstance(validated_X, dict) and 'error' in validated_X:
            #     reason = f"특성 검증 실패: {validated_X['error']}"
            #     self.logger.error(reason)
            #     return {'signal': SignalType.HOLD, 'confidence': 0.0, 'reason': reason, 'timestamp': datetime.now()}
            # X = validated_X # 검증된 특성 사용
                
            # 1. 개별 전략 신호 생성
            strategy_signals = {}
            
            # 1.1. 방향성 예측 모델 신호
            for model in self.direction_models:
                try:
                    # 모델 예측 (ModelOutput 객체 반환 예상)
                    output: ModelOutput = model.predict(X) 
                    
                    # ModelOutput에서 신뢰도 추출 (signal 객체 내부 또는 output 자체)
                    confidence = 0.0
                    predicted_direction = 0 # 기본값: HOLD (0), 상승(1), 하락(-1) - 모델 따라 조정 필요
                    
                    if hasattr(output, 'confidence'):
                        confidence = output.confidence
                    elif hasattr(output, 'signal') and hasattr(output.signal, 'confidence'):
                        confidence = output.signal.confidence
                        
                    # 방향 결정 로직 (모델의 출력 형식에 따라 달라짐)
                    # 예시 1: output.signal.signal_type 사용
                    if hasattr(output, 'signal') and hasattr(output.signal, 'signal_type'):
                        if output.signal.signal_type == SignalType.BUY:
                            predicted_direction = 1
                        elif output.signal.signal_type == SignalType.SELL:
                            predicted_direction = -1
                            
                    # 예시 2: output.raw_predictions 사용 (이진 분류 확률 등)
                    elif hasattr(output, 'raw_predictions') and output.raw_predictions is not None:
                         # raw_predictions가 (n_samples, n_classes) 형태의 확률 배열이라고 가정
                        if isinstance(output.raw_predictions, np.ndarray) and output.raw_predictions.ndim == 2:
                            # 이진 분류 (1 클래스 확률)
                            if output.raw_predictions.shape[1] == 1:
                                prob_buy = output.raw_predictions[0, 0]
                                if prob_buy > 0.5:
                                    predicted_direction = 1
                                    confidence = prob_buy # 신뢰도를 매수 확률로
                                else:
                                    predicted_direction = -1
                                    confidence = 1.0 - prob_buy # 신뢰도를 매도 확률로
                            # 다중 분류 (BUY=1, HOLD=0, SELL=-1 클래스 가정)
                            elif output.raw_predictions.shape[1] == 3:
                                class_index = np.argmax(output.raw_predictions[0])
                                confidence = output.raw_predictions[0, class_index]
                                if class_index == 1: # BUY
                                    predicted_direction = 1
                                elif class_index == 2: # SELL
                                    predicted_direction = -1
                                # else: HOLD (predicted_direction = 0)
                    
                    # 최종 신호 타입 결정
                    signal_type = SignalType.HOLD
                    if predicted_direction == 1:
                        signal_type = SignalType.BUY
                    elif predicted_direction == -1:
                        signal_type = SignalType.SELL
                        
                    strategy_signals[f"direction_{model.name}"] = {
                        'signal': signal_type,
                        'confidence': float(confidence), # float 타입 보장
                        'timestamp': datetime.now()
                    }
                    self.logger.debug(f"모델 {model.name} 방향 예측: signal={signal_type}, confidence={confidence:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"방향성 예측 모델 {model.name} 신호 생성 중 오류: {str(e)}")
                    self.logger.debug(traceback.format_exc()) # 디버깅을 위한 스택 트레이스 추가
                    # 오류 발생 시 HOLD 신호 기록
                    strategy_signals[f"direction_{model.name}"] = {
                        'signal': SignalType.HOLD, 'confidence': 0.0, 'timestamp': datetime.now(), 'error': str(e)
                    }
            
            # 1.2. 가격 예측 모델 신호 (RegressionModel)
            for model in self.price_models:
                try:
                    # 가격 모델은 예측 가격 (숫자) 반환 가정
                    predicted_price = model.predict(X) 
                    
                    # predict가 ModelOutput을 반환하는 경우 처리 (RegressionModelBase 확인 필요)
                    if isinstance(predicted_price, ModelOutput):
                         if hasattr(predicted_price, 'signal') and hasattr(predicted_price.signal, 'price'):
                             predicted_price = predicted_price.signal.price
                         elif hasattr(predicted_price, 'metadata') and 'predicted_value' in predicted_price.metadata:
                             predicted_price = predicted_price.metadata['predicted_value']
                         else:
                             # 예측값을 추출할 수 없는 경우
                             raise ValueError("가격 모델의 ModelOutput에서 예측 가격을 추출할 수 없습니다.")

                    # 예측값이 배열인 경우 첫번째 값 사용
                    if isinstance(predicted_price, np.ndarray):
                        predicted_price = predicted_price[0] 
                        
                    current_price = data['close'].iloc[-1]
                    signal_type = SignalType.BUY if predicted_price > current_price else SignalType.SELL
                    confidence = abs(predicted_price - current_price) / current_price # 정규화된 가격 차이
                    
                    strategy_signals[f"price_{model.name}"] = {
                        'signal': signal_type,
                        'confidence': float(confidence), # float 타입 보장
                        'timestamp': datetime.now()
                    }
                    self.logger.debug(f"모델 {model.name} 가격 예측: current={current_price:.2f}, predicted={predicted_price:.2f}, signal={signal_type}, confidence={confidence:.4f}")

                except Exception as e:
                    self.logger.error(f"가격 예측 모델 {model.name} 신호 생성 중 오류: {str(e)}")
                    self.logger.debug(traceback.format_exc())
                    # 오류 발생 시 HOLD 신호 기록
                    strategy_signals[f"price_{model.name}"] = {
                        'signal': SignalType.HOLD, 'confidence': 0.0, 'timestamp': datetime.now(), 'error': str(e)
                    }
            
            # 2. 신호 통합
            combined_signal = self._combine_signals(strategy_signals)
            
            # 3. 리스크 관리 적용 (generate_signal 내부에서는 호출하지 않음 - 백테스트 엔진/main에서 처리)
            # final_signal = self.apply_risk_management(combined_signal, portfolio) # 포트폴리오 정보 필요
            final_signal = combined_signal # 리스크 관리는 외부에서!
            
            self.logger.info(f"최종 생성 신호: {final_signal['signal']}, 신뢰도: {final_signal.get('confidence', 0.0):.4f}")
            return final_signal
            
        except KeyError as ke:
             self.logger.error(f"거래 신호 생성 중 KeyError 발생: {str(ke)} - 입력 데이터 컬럼 확인 필요")
             self.logger.debug(traceback.format_exc())
             return {'signal': SignalType.HOLD, 'confidence': 0.0, 'reason': f'KeyError: {str(ke)}', 'timestamp': datetime.now()}
        except Exception as e:
            self.logger.error(f"거래 신호 생성 중 예상치 못한 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'signal': SignalType.HOLD, 'confidence': 0.0, 'reason': f'오류: {str(e)}', 'timestamp': datetime.now()}

    def _combine_signals(self, strategy_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """개별 전략의 신호를 통합하여 최종 신호를 생성합니다. (개선된 로깅 추가)
        
        Args:
            strategy_signals: 개별 전략의 신호 정보 {'strategy_name': {'signal': SignalType, 'confidence': float, ...}}
            
        Returns:
            Dict[str, Any]: 통합된 거래 신호 {'signal': SignalType, 'confidence': float, 'reason': str, ...}
        """
        try:
            self.logger.debug(f"신호 통합 시작. 입력 신호 개수: {len(strategy_signals)}")
            
            if not strategy_signals:
                 self.logger.warning("통합할 개별 신호가 없습니다.")
                 return {'signal': SignalType.HOLD, 'confidence': 0.0, 'reason': '개별 신호 없음', 'timestamp': datetime.now(), 'strategy_signals': strategy_signals}

            # 1. 유효한 신호 필터링 및 가중치 준비
            valid_signals = []
            total_weight = 0
            signal_details = [] # 로깅용

            for strategy_name, signal_info in strategy_signals.items():
                # 오류가 있거나 신뢰도가 없는 신호는 제외
                if 'error' in signal_info or 'confidence' not in signal_info or 'signal' not in signal_info:
                    self.logger.warning(f"전략 '{strategy_name}'의 신호 건너뜀 (오류 또는 정보 부족): {signal_info.get('error', '정보 부족')}")
                    continue

                # 가중치 가져오기 (direction_, price_ 접두사 제거 시도)
                base_strategy_name = strategy_name.replace('direction_', '').replace('price_', '')
                weight = self.strategy_weights.get(base_strategy_name, self.strategy_weights.get('ml', 1.0)) # 기본 ML 가중치 또는 1.0
                
                valid_signals.append({
                    'name': strategy_name,
                    'signal': signal_info['signal'],
                    'confidence': signal_info['confidence'],
                    'weight': weight
                })
                total_weight += weight
                signal_details.append(f"{strategy_name}(W:{weight:.2f}, S:{signal_info['signal']}, C:{signal_info['confidence']:.3f})")

            self.logger.debug(f"통합 대상 유효 신호: {len(valid_signals)}개. 총 가중치: {total_weight:.2f}")
            self.logger.debug(f"개별 신호 상세: [{', '.join(signal_details)}]")

            if not valid_signals or total_weight <= 0:
                self.logger.warning("유효한 신호 또는 가중치가 없어 HOLD 신호 반환.")
                return {'signal': SignalType.HOLD, 'confidence': 0.0, 'reason': '유효 신호 없음', 'timestamp': datetime.now(), 'strategy_signals': strategy_signals}

            # 2. 가중 평균 계산
            weighted_signal_sum = 0.0
            # BUY = +1, SELL = -1, HOLD = 0 으로 변환하여 가중합 계산
            for signal in valid_signals:
                signal_value = 0
                if signal['signal'] == SignalType.BUY:
                    signal_value = 1
                elif signal['signal'] == SignalType.SELL:
                    signal_value = -1
                
                # 신뢰도와 가중치를 곱하여 가중합에 더함
                weighted_signal_sum += signal_value * signal['confidence'] * signal['weight']

            # 3. 최종 신호 결정
            # 가중합을 총 가중치로 나누어 평균 신호 강도 계산 (-1 ~ +1 범위)
            average_signal_strength = weighted_signal_sum / total_weight
            
            final_signal_type = SignalType.HOLD
            final_confidence = 0.0
            
            # 평균 신호 강도에 따라 최종 신호 결정
            # confidence_threshold의 절반을 기준으로 HOLD 영역 설정
            hold_threshold = self.confidence_threshold * 0.5 
            
            if average_signal_strength > hold_threshold:
                final_signal_type = SignalType.BUY
                # 신뢰도는 0~1 범위로 정규화 (BUY 신호 강도)
                final_confidence = min(1.0, max(0.0, (average_signal_strength - hold_threshold) / (1.0 - hold_threshold)))
            elif average_signal_strength < -hold_threshold:
                final_signal_type = SignalType.SELL
                # 신뢰도는 0~1 범위로 정규화 (SELL 신호 강도)
                final_confidence = min(1.0, max(0.0, (-average_signal_strength - hold_threshold) / (1.0 - hold_threshold)))
            # else: HOLD
                
            # 최종 신뢰도가 너무 낮으면 HOLD로 강제
            if final_confidence < self.confidence_threshold * 0.2: # 임계값의 20% 미만이면 무시
                 if final_signal_type != SignalType.HOLD:
                     self.logger.info(f"통합 신뢰도가 매우 낮아 ({final_confidence:.3f}) HOLD로 변경.")
                     final_signal_type = SignalType.HOLD
                     final_confidence = 0.0


            reason = f"신호 통합 결과: 강도={average_signal_strength:.3f}, 신뢰도={final_confidence:.3f}"
            self.logger.info(f"신호 통합 완료: 최종 신호={final_signal_type}, 신뢰도={final_confidence:.4f}")

            return {
                'signal': final_signal_type,
                'confidence': final_confidence,
                'reason': reason,
                'timestamp': datetime.now(),
                'average_signal_strength': average_signal_strength, # 디버깅용
                'strategy_signals': strategy_signals # 원본 신호 포함
            }
            
        except Exception as e:
            self.logger.error(f"신호 통합 중 예상치 못한 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'signal': SignalType.HOLD, 'confidence': 0.0, 'reason': f'신호 통합 오류: {str(e)}', 'timestamp': datetime.now()}

    def _prepare_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        모델 예측을 위한 특성 준비
        
        Args:
            data (Union[pd.DataFrame, Dict[str, Any]]): 원시 시장 데이터 (DataFrame 또는 단일 행 dict)
            
        Returns:
            Optional[np.ndarray]: 처리된 특성 배열, 오류 시 None
        """
        try:
            # 입력 데이터가 dict인 경우 DataFrame으로 변환
            if isinstance(data, dict):
                # 단일 행 DataFrame 생성
                data = pd.DataFrame([data]) 
                self.logger.debug(f"_prepare_features: dict 입력을 DataFrame으로 변환. Shape: {data.shape}")
            
            # DataFrame 타입 확인
            if not isinstance(data, pd.DataFrame):
                 raise ValueError(f"_prepare_features: 입력 데이터는 DataFrame 또는 dict여야 합니다. 실제 타입: {type(data)}")
                 
            # 빈 DataFrame 처리
            if data.empty:
                self.logger.warning("_prepare_features: 입력 데이터프레임이 비어 있습니다.")
                return None

            # 예상 특성 확인
            if self.expected_features is None or len(self.expected_features) == 0:
                self.logger.warning("_prepare_features: 예상 특성 정보가 없습니다. 사용 가능한 모든 숫자형 특성을 사용합니다.")
                # 데이터프레임에서 숫자형 컬럼만 선택
                numeric_data = data.select_dtypes(include=[np.number])
                if numeric_data.empty:
                     self.logger.warning("_prepare_features: 숫자형 특성이 없습니다.")
                     return None
                features = numeric_data.values
                # NaN/Inf 처리
                return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 사용 가능한 특성 필터링 및 순서 정렬
            aligned_df = pd.DataFrame(index=data.index) # 원본 인덱스 유지
            missing_features = []
            
            for feature in self.expected_features:
                if feature in data.columns:
                    aligned_df[feature] = data[feature]
                else:
                    # 누락된 특성은 0으로 채움
                    aligned_df[feature] = 0 
                    missing_features.append(feature)
            
            if missing_features:
                 self.logger.warning(f"_prepare_features: 예상 특성 중 다음이 누락되어 0으로 채웁니다: {missing_features}")
                 
            # NaN/Inf 값 처리 (0으로 대체)
            X = aligned_df.values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.logger.debug(f"_prepare_features 완료. 최종 특성 shape: {X.shape}")
            return X
            
        except Exception as e:
            self.logger.error(f"특성 준비 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _load_expected_features(self) -> List[str]:
        """
        예상 특성 이름 로드
            
        Returns:
            List[str]: 예상 특성 이름 목록
        """
        try:
            if os.path.exists(self.features_path):
                with open(self.features_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.logger.error(f"특성 이름 로드 중 오류: {str(e)}")
            return []

    def _validate_features(self, X: np.ndarray) -> np.ndarray:
        """
        입력 특성 수 검증 및 조정
        
        Args:
            X (np.ndarray): 입력 특성
            
        Returns:
            np.ndarray: 검증된 특성
        """
        # X에 컬럼 정보가 있을 경우 (DataFrame)
        if hasattr(X, 'columns'):
            # expected_features와 비교
            if hasattr(self, 'expected_features') and self.expected_features:
                missing_features = set(self.expected_features) - set(X.columns)
                if missing_features:
                    self.logger.warning(f"누락된 특성: {missing_features}")
                
                # 중복 컬럼 제거 (첫 번째 항목만 유지)
                if X.columns.duplicated().any():
                    self.logger.warning("중복된 컬럼명이 감지되었습니다. 첫 번째 항목만 유지하고 나머지는 제거합니다.")
                    X = X.loc[:, ~X.columns.duplicated()]
                    self.logger.info(f"중복 제거 후 특성 수: {X.shape[1]}개")
                
                # 특성 순서 맞추기
                X = X.reindex(columns=self.expected_features, fill_value=0)
                
                # strict mode에서는 특성 순서와 개수가 정확히 일치해야 함
                if getattr(self, 'strict_mode', False):
                    missing_features = set(self.expected_features) - set(X.columns)
                    extra_features = set(X.columns) - set(self.expected_features)
                    
                    if missing_features:
                        missing_msg = f"누락된 특성이 있습니다: {missing_features}"
                        self.logger.warning(missing_msg)
                    
                    if extra_features:
                        extra_msg = f"추가된 특성이 있습니다: {extra_features}"
                        self.logger.warning(extra_msg)
                    
                    # 특성 순서 검증
                    if list(X.columns) != self.expected_features:
                        msg = "특성 순서가 예상과 다릅니다. 모델 예측이 부정확할 수 있습니다."
                        self.logger.error(msg)
                        raise ValueError(f"Strict Mode 활성화: {msg}")
            
            # NumPy 배열로 변환
            X = X.values
        
        # 일반 NumPy 배열인 경우 특성 수만 검증
        if hasattr(self, 'expected_features_count') and X.shape[1] != self.expected_features_count:
            self.logger.error(f"입력 특성 수가 예상과 일치하지 않습니다. 예상={self.expected_features_count}, 실제={X.shape[1]}")
            
            # 스케일러와 모델 특성 수 일관성 검증
            if hasattr(self, 'scaler') and hasattr(self.scaler, 'n_features_in_'):
                scaler_features = self.scaler.n_features_in_
                model_features = self.expected_features_count
                
                if scaler_features != model_features:
                    self.logger.error(f"심각한 불일치: 스케일러 특성 수({scaler_features})와 모델 특성 수({model_features})가 다릅니다!")
                    return {'error': f'특성 수 불일치: 스케일러={scaler_features}, 모델={model_features}, 입력={X.shape[1]}'}
            
            # 특성 수 조정을 시도하지 않고 오류 반환
            return {'error': f'특성 수 불일치: 입력 {X.shape[1]} != 기대 {self.expected_features_count}. 예측 중단됨.'}
        
        return X
    
    def save_features(self, feature_names: List[str]) -> bool:
        """
        특성 이름 목록을 파일로 저장합니다.
        
        Args:
            feature_names (List[str]): 저장할 특성 이름 목록
            
        Returns:
            bool: 저장 성공 여부
        """
        if not feature_names:
            self.logger.warning("저장할 특성 이름이 비어 있습니다.")
            return False
        
        try:
            if not hasattr(self, 'features_path'):
                # 모델 번들 내에 features.json 파일 경로 설정
                model_dir = os.path.join("data_storage", "models")
                self.features_path = os.path.join(model_dir, f"{self.name}_features.json")
            
            # 디렉토리 경로 생성
            directory = os.path.dirname(self.features_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"특성 저장용 디렉토리 생성됨: {directory}")
            
            # 특성 이름 저장
            with open(self.features_path, 'w', encoding='utf-8') as f:
                json.dump(feature_names, f, indent=2, ensure_ascii=False)
            
            self.expected_features = feature_names
            self.expected_features_count = len(feature_names)
            
            # 디버그 로그로 첫 몇 개 특성 출력 (최대 5개)
            preview = ', '.join(feature_names[:5])
            if len(feature_names) > 5:
                preview += f'... (총 {len(feature_names)}개)'
            self.logger.info(f"특성 목록 저장 완료: {self.features_path} ({preview})")
            
            return True
        except Exception as e:
            error_msg = f"특성 목록 저장 중 오류: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # 개발/디버깅 모드에서는 파일 시스템 정보 추가 로깅
            try:
                self.logger.debug(f"저장 경로 상태: 존재={os.path.exists(os.path.dirname(self.features_path))}, 쓰기 가능={os.access(os.path.dirname(self.features_path), os.W_OK)}")
            except:
                pass
            
            return False
            
    def align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        입력 데이터프레임의 특성을 모델에 맞게 정렬합니다.
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            
        Returns:
            pd.DataFrame: 정렬된 데이터프레임
        """
        # 예상 특성 목록이 없으면 로드
        if not hasattr(self, 'expected_features') or not self.expected_features:
            self.expected_features = self._load_expected_features()
            
            # 로드 실패 시 현재 특성 사용
            if not self.expected_features:
                self.logger.warning("특성 목록을 불러오지 못했습니다. 현재 특성을 저장합니다.")
                self.save_features(df.columns.tolist())
                return df
        
        # 특성 정렬
        self.logger.info(f"특성 정렬: {df.shape[1]}개 -> {len(self.expected_features)}개")
        aligned_df = df.reindex(columns=self.expected_features, fill_value=0)
        
        # strict mode에서는 특성 순서와 개수가 정확히 일치해야 함
        if getattr(self, 'strict_mode', False):
            missing_features = set(self.expected_features) - set(df.columns)
            if missing_features:
                missing_msg = f"누락된 특성이 있습니다: {missing_features}"
                self.logger.warning(missing_msg)
                
                if len(missing_features) > len(self.expected_features) * 0.1:  # 10% 이상 누락 시 경고
                    msg = f"Strict Mode 활성화: 너무 많은 특성({len(missing_features)}/{len(self.expected_features)})이 누락되었습니다."
                    self.logger.error(msg)
                    raise ValueError(msg)
        
        return aligned_df 

    def apply_risk_management(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        위험 관리 규칙을 적용하여 거래 신호를 조정합니다. (포트폴리오 정보 사용)
        
        Args:
            signal (Dict): 거래 신호 정보 {'signal': SignalType, 'confidence': float, ...}
            portfolio (Dict): 현재 포트폴리오 정보 {'position': 'BUY'/'SELL'/'NONE', 'entry_price': float, ...}
            
        Returns:
            Dict: 위험 관리가 적용된 거래 신호
        """
        try:
            self.logger.debug(f"apply_risk_management 호출됨. 신호: {signal.get('signal')}, 포지션: {portfolio.get('position')}")
            
            # 신호 유효성 검사
            if not isinstance(signal, dict) or 'signal' not in signal:
                self.logger.error("apply_risk_management: 유효하지 않은 신호 형식")
                return {'signal': SignalType.HOLD, 'confidence': 0.0, 'reason': '유효하지 않은 신호', 'metadata': {}}
            
            # 포트폴리오 유효성 검사 (기본값 제공)
            if not isinstance(portfolio, dict):
                self.logger.warning("apply_risk_management: 포트폴리오 정보가 유효하지 않음, 기본값 사용")
                portfolio = {'position': 'NONE', 'entry_price': 0, 'quantity': 0}
            
            original_signal_type = signal.get('signal', SignalType.HOLD)
            current_position = portfolio.get('position', 'NONE')
            confidence = signal.get('confidence', 0.0)
            
            # 신호 조정 로직 (기존 main.py 바인딩 로직 참고)
            adjusted_signal = signal.copy() # 원본 복사하여 수정
            
            # 1. 이미 같은 포지션을 가지고 있는 경우 중복 신호 방지
            if (current_position == 'BUY' and original_signal_type == SignalType.BUY) or \
               (current_position == 'SELL' and original_signal_type == SignalType.SELL):
                reason = f"이미 {current_position} 포지션을 보유 중이므로 중복 신호 무시"
                self.logger.info(reason)
                adjusted_signal['signal'] = SignalType.HOLD
                adjusted_signal['reason'] = reason
                # 신뢰도도 0으로 설정하는 것이 명확할 수 있음
                # adjusted_signal['confidence'] = 0.0 
                return adjusted_signal
            
            # 2. 낮은 신뢰도 신호에 대한 처리 (포지션 진입/변경 시 더 높은 기준 적용 가능)
            min_confidence_entry = self.confidence_threshold # 포지션 진입/변경 최소 신뢰도 (기본값 사용)
            min_confidence_hold = self.confidence_threshold * 0.5 # 홀드 유지 최소 신뢰도 (더 낮게 설정 가능)

            if original_signal_type != SignalType.HOLD:
                # 포지션 진입(NONE -> BUY/SELL) 또는 변경(BUY->SELL, SELL->BUY) 시
                if current_position == 'NONE' or \
                   (current_position == 'BUY' and original_signal_type == SignalType.SELL) or \
                   (current_position == 'SELL' and original_signal_type == SignalType.BUY):
                    if confidence < min_confidence_entry:
                        reason = f"신규/변경 포지션 진입 신뢰도 부족 ({confidence:.3f} < {min_confidence_entry:.3f})"
                        self.logger.info(reason)
                        adjusted_signal['signal'] = SignalType.HOLD
                        adjusted_signal['reason'] = reason
                        return adjusted_signal
                # 기존 포지션 유지 신호 (BUY->BUY, SELL->SELL) - 이미 위에서 처리됨
            
            # 3. 추가적인 리스크 관리 규칙 (예: 최대 손실 제한, 변동성 기반 필터 등)
            #    - 필요에 따라 portfolio 정보 (진입 가격 등)와 외부 데이터 (시장 변동성 등)를 활용하여 구현
            #    - 예시: if portfolio.get('current_pnl_percent', 0) < -0.1: # 10% 이상 손실 시 강제 청산 등
            
            # 최종 로깅
            if adjusted_signal['signal'] != original_signal_type:
                self.logger.info(f"위험 관리 적용 결과: {original_signal_type} -> {adjusted_signal['signal']} (Reason: {adjusted_signal.get('reason', 'N/A')})")
            else:
                self.logger.debug(f"위험 관리 규칙 적용 후에도 신호 유지: {adjusted_signal['signal']}")
            
            return adjusted_signal
            
        except Exception as e:
            self.logger.error(f"apply_risk_management 적용 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            # 오류 발생 시 안전하게 HOLD 신호 반환
            return {'signal': SignalType.HOLD, 'confidence': 0.0, 'reason': f'리스크 관리 오류: {str(e)}', 'metadata': signal.get('metadata', {})} 