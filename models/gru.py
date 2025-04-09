"""
GRU based price prediction model.

이 모듈은 GRU(Gated Recurrent Unit) 신경망을 사용한 비트코인 가격 예측 모델을 구현합니다.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, GRU, Dropout, Input, Attention, LayerNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from keras.regularizers import l1_l2
import keras.backend as K
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
import json
import logging
import traceback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.base import TimeSeriesModel
from utils.constants import SignalType
from models.signal import TradingSignal, ModelOutput

# 로거 설정
logger = logging.getLogger(__name__)

# TensorFlow 로깅 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 정보 및 경고만 표시
tf.get_logger().setLevel(logging.ERROR)  # 오류만 표시

class GRUPriceModel(TimeSeriesModel):
    """비트코인 가격 예측을 위한 고급 GRU 모델"""
    
    def __init__(self, 
                name: str = "GRUPrice", 
                version: str = "1.0.0",
                sequence_length: int = 60,
                forecast_horizon: int = 1,
                units: List[int] = [96, 64, 32],  # 유닛 수 증가 및 레이어 추가
                dropout_rate: float = 0.35,       # 드롭아웃 비율 최적화
                learning_rate: float = 0.001,
                batch_size: int = 64,             # 배치 크기 증가
                epochs: int = 150,                # 에폭 수 증가
                use_attention: bool = True):      # 어텐션 메커니즘 추가
        """
        가격 예측 GRU 모델 초기화 - 최적화된 하이퍼파라미터 사용
        
        Args:
            name (str): 모델 이름
            version (str): 모델 버전
            sequence_length (int): 사용할 과거 시간 단계 수
            forecast_horizon (int): 예측할 미래 시간 단계 수
            units (List[int]): 각 GRU 레이어의 유닛 수 리스트
            dropout_rate (float): 각 GRU 레이어 이후의 Dropout 비율
            learning_rate (float): Adam 옵티마이저 학습률
            batch_size (int): 학습 배치 크기
            epochs (int): 최대 학습 에폭 수
            use_attention (bool): 어텐션 메커니즘 사용 여부
        """
        super().__init__(name, version)
        
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_attention = use_attention
        
        # 파라미터 저장
        self.params = {
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon,
            'units': units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'use_attention': use_attention
        }
        
        self.model = None
        self.feature_dim = None
        self.history = None
        
        self.logger.info(f"{self.name} 모델을 {len(units)}개 GRU 레이어로 초기화했습니다. 어텐션 메커니즘: {use_attention}")
    
    def _attention_block(self, inputs, hidden_size):
        """
        시간적 어텐션 메커니즘 구현
        
        Args:
            inputs: 어텐션을 적용할 입력 텐서
            hidden_size: 어텐션 메커니즘의 히든 유닛 수
            
        Returns:
            어텐션이 적용된 텐서
        """
        # 시간 차원에 걸친 어텐션 가중치 계산
        attention = tf.keras.layers.Dense(hidden_size, activation='tanh')(inputs)
        attention = tf.keras.layers.Dense(1, activation=None)(attention)
        attention_weights = tf.keras.layers.Softmax(axis=1)(attention)
        
        # 가중치와 입력을 곱하여 컨텍스트 벡터 생성
        context_vector = tf.keras.layers.Multiply()([inputs, attention_weights])
        context_vector = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
        
        return context_vector, attention_weights
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        향상된 GRU 모델 아키텍처 구축
        
        Args:
            input_shape (Tuple[int, int]): 입력 형태 (sequence_length, features)
        """
        self.feature_dim = input_shape[1]
        
        # 함수형 API를 사용하여 모델 구축
        inputs = tf.keras.Input(shape=input_shape)
        
        # 입력에 LayerNormalization 적용
        normalized = tf.keras.layers.LayerNormalization()(inputs)
        
        # GRU 레이어 추가
        x = normalized
        
        for i, units in enumerate(self.units):
            return_sequences = i < len(self.units) - 1 or self.use_attention
            
            # GRU 레이어
            gru_layer = tf.keras.layers.GRU(
                units, 
                return_sequences=return_sequences,
                recurrent_dropout=0.1,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_regularizer=l1_l2(l1=0.001, l2=0.001),
                stateful=False
            )(x)
            
            # LayerNormalization - 정규화를 통한 안정성 개선
            normalized_gru = tf.keras.layers.LayerNormalization()(gru_layer)
            
            # 잔차 연결 (Residual Connection) - 처음과 마지막 레이어는 제외
            if 0 < i < len(self.units) - 1 and input_shape[1] == units:
                x = tf.keras.layers.add([x, normalized_gru])
            else:
                x = normalized_gru
            
            # 드롭아웃
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        
        # 어텐션 메커니즘 적용
        if self.use_attention:
            context_vector, _ = self._attention_block(x, self.units[-1])
            x = context_vector
        
        # 출력 레이어 : 단일 값 (가격) 또는 여러 단계 예측
        outputs = tf.keras.layers.Dense(self.forecast_horizon)(x)
        
        # 모델 생성
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # 모델 요약 출력
        self.model = model
        model.summary(print_fn=self.logger.info)
        
        # 총 파라미터 수와 훈련 가능한 파라미터 수 로깅
        total_params = model.count_params()
        trainable_params = sum([K.count_params(w) for w in model.trainable_weights])
        self.logger.info(f"GRU 가격 예측 모델 총 파라미터 수: {total_params:,}")
        self.logger.info(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              early_stopping_patience: int = 15,
              reduce_lr_patience: int = 8,
              sample_weight: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        GRU 가격 예측 모델 훈련 - 향상된 버전
        
        Args:
            X_train (np.ndarray): 훈련 시퀀스, 형태 (samples, sequence_length, features)
            y_train (np.ndarray): 훈련 타겟 
            X_val (np.ndarray, optional): 검증 시퀀스
            y_val (np.ndarray, optional): 검증 타겟
            early_stopping_patience (int): 조기 종료 인내심
            reduce_lr_patience (int): 학습률 감소 인내심
            sample_weight (np.ndarray, optional): 샘플 가중치
            feature_names (List[str], optional): 특성 이름 목록
            
        Returns:
            Dict[str, float]: 훈련 결과 지표
        """
        start_time = datetime.now()
        self.logger.info(f"GRU 모델 훈련 시작: {X_train.shape[0]}개 샘플, 형태: {X_train.shape}")
        
        # 특성 이름 저장
        if feature_names is not None:
            self.feature_names = feature_names
            # feature_manager 사용
            if hasattr(self, 'feature_manager') and self.feature_manager:
                metadata = {
                    'model_type': 'gru',
                    'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'sequence_length': self.sequence_length,
                    'feature_count': X_train.shape[2]
                }
                self.feature_manager.save_features(feature_names, metadata)
        
        # 입력 형태 확인 및 모델 구축
        if len(X_train.shape) != 3:
            raise ValueError(f"입력은 3D 배열이어야 합니다. 형태: (samples, sequence_length, features), 현재: {X_train.shape}")
        
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, features)
            self.feature_dim = X_train.shape[2]
            self.build_model(input_shape)
        
        # 검증 데이터 설정
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # 콜백 설정
        callbacks = []
        
        # 조기 종료
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # 학습률 조정
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # 체크포인트
        checkpoint_dir = os.path.join("models", "checkpoints", self.name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}_best.h5")
        
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_loss' if validation_data else 'loss',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # TensorBoard 로깅
        log_dir = os.path.join("logs", "tensorboard", f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)
        
        # 모델 학습
        self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
                callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=1
        )
        
        # 학습 지표 저장
        self.metrics = {}
        
        # 훈련 손실
        if 'loss' in self.history.history:
            self.metrics['train_loss'] = float(self.history.history['loss'][-1])
            self.metrics['best_train_loss'] = float(min(self.history.history['loss']))
        
        # 검증 손실
        if 'val_loss' in self.history.history:
            self.metrics['val_loss'] = float(self.history.history['val_loss'][-1])
            self.metrics['best_val_loss'] = float(min(self.history.history['val_loss']))
        
        # MAE
        if 'mae' in self.history.history:
            self.metrics['train_mae'] = float(self.history.history['mae'][-1])
            self.metrics['best_train_mae'] = float(min(self.history.history['mae']))
        
        # 검증 MAE
        if 'val_mae' in self.history.history:
            self.metrics['val_mae'] = float(self.history.history['val_mae'][-1])
            self.metrics['best_val_mae'] = float(min(self.history.history['val_mae']))
        
        # 훈련 시간
        training_time = (datetime.now() - start_time).total_seconds()
        self.metrics['training_time'] = training_time
        self.metrics['epochs_trained'] = len(self.history.history['loss'])
        
        self.is_trained = True
        
        self.logger.info(f"GRU 모델 훈련 완료: {self.metrics['epochs_trained']}에폭, 소요 시간: {training_time:.2f}초")
        self.logger.info(f"최종 훈련 손실: {self.metrics.get('train_loss', 'N/A')}, 최종 검증 손실: {self.metrics.get('val_loss', 'N/A')}")
        
        return self.metrics
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        모델 평가를 수행합니다.
        
        Args:
            X_test (np.ndarray): 테스트용 특성 데이터
            y_test (np.ndarray): 테스트용 타겟 데이터
            **kwargs: 추가 파라미터
            
        Returns:
            Dict[str, Any]: 평가 지표
        """
        if not self.is_trained or self.model is None:
            self.logger.error("모델이 훈련되지 않았습니다. evaluate 메서드를 호출하기 전에 모델을 훈련하세요.")
            return {'error': 'Model not trained'}
        
        try:
            # 예측 수행
            y_pred = self.predict(X_test)
            
            # 성능 지표 계산
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-10))) * 100
            r2 = r2_score(y_test, y_pred)
            
            # 방향성 예측 정확도 (상승/하락)
            y_direction = np.sign(np.diff(np.append([y_test[0]], y_test), axis=0))
            pred_direction = np.sign(np.diff(np.append([y_test[0]], y_pred), axis=0))
            direction_accuracy = np.mean(y_direction == pred_direction)
            
            # 결과 저장
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'r2': float(r2),
                'direction_accuracy': float(direction_accuracy)
            }
            
            self.logger.info(f"평가 결과: RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, 방향 정확도: {direction_accuracy:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"모델 평가 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'mse': float('inf'),
                'rmse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf'),
                'r2': 0.0,
                'direction_accuracy': 0.0
            }
            
    def predict(self, X: np.ndarray) -> ModelOutput:
        """
        학습된 GRU 모델을 사용하여 예측 수행
        
        Args:
            X (np.ndarray): 입력 시퀀스, 형태 (samples, sequence_length, features)
            
        Returns:
            ModelOutput: 예측 결과를 포함하는 표준화된 모델 출력
        """
        if not self.is_trained or self.model is None:
            self.logger.error("모델이 학습되지 않았습니다. predict를 호출하기 전에 train 함수를 호출하세요.")
            return ModelOutput(
                signal=TradingSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    reason="모델이 학습되지 않았습니다."
                ),
                confidence=0.0,
                metadata={"error": "Model not trained"}
            )
            
        try:
            # 특성 검증 및 조정
            if hasattr(self, 'feature_manager') and self.feature_manager:
                # 데이터프레임 검사
                if hasattr(X, 'columns'):
                    # 데이터프레임인 경우 특성 정렬
                    X = self.feature_manager.align_features(X)
                elif len(X.shape) == 3:
                    # 3D 배열의 경우 특성 수 검증
                    expected_feature_count = getattr(self, 'feature_dim', None)
                    if expected_feature_count is not None and X.shape[2] != expected_feature_count:
                        self.logger.warning(f"특성 수 불일치: 예상={expected_feature_count}, 실제={X.shape[2]}")
                        # 각 시퀀스의 각 시점에 대해 특성 조정
                        adjusted_X = np.zeros((X.shape[0], X.shape[1], expected_feature_count))
                        for i in range(X.shape[0]):
                            for j in range(X.shape[1]):
                                adjusted_X[i, j] = self.feature_manager.adjust_feature_count(
                                    X[i, j].reshape(1, -1), expected_feature_count).flatten()
                        X = adjusted_X
            
            # 입력 형태 검증
            if len(X.shape) != 3:
                error_msg = f"입력은 3D 배열이어야 합니다. 형태: (samples, sequence_length, features), 현재: {X.shape}"
                self.logger.error(error_msg)
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason=error_msg
                    ),
                    confidence=0.0,
                    metadata={"error": error_msg}
                )
                
            if X.shape[1] != self.sequence_length:
                error_msg = f"입력 시퀀스 길이가 예상과 다릅니다. 예상: {self.sequence_length}, 실제: {X.shape[1]}"
                self.logger.error(error_msg)
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason=error_msg
                    ),
                    confidence=0.0,
                    metadata={"error": error_msg}
                )
                
            if self.feature_dim is not None and X.shape[2] != self.feature_dim:
                self.logger.warning(f"입력 특성 수가 예상과 다릅니다. 예상: {self.feature_dim}, 실제: {X.shape[2]}")
                
            # 예측 수행
            raw_predictions = self.model.predict(X, verbose=0)
            
            # 빈 예측 확인
            if len(raw_predictions) == 0:
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason="예측 결과가 없습니다."
                    ),
                    confidence=0.0,
                    metadata={"error": "Empty prediction result"}
                )
            
            # 각 예측에 대한 신호 생성
            results = []
            for i, pred in enumerate(raw_predictions):
                # 현재 가격과 비교
                current_price = X[i, -1, 0] if X.shape[0] > i and X.shape[1] > 0 and X.shape[2] > 0 else None
                
                # 방향을 기반으로 신호 결정
                if current_price is not None:
                    price_diff = float(pred) - current_price
                    pct_change = (price_diff / current_price) * 100 if current_price != 0 else 0
                    
                    # 임계값에 따라 신호 결정
                    threshold = 0.1  # 0.1% 이상의 변화가 있어야 방향성 신호
                    if pct_change > threshold:
                        signal_type = SignalType.BUY
                        reason = f"상승 예측: 현재가={current_price:.2f}, 예측가={float(pred):.2f}, 변화율={pct_change:.2f}%"
                    elif pct_change < -threshold:
                        signal_type = SignalType.SELL
                        reason = f"하락 예측: 현재가={current_price:.2f}, 예측가={float(pred):.2f}, 변화율={pct_change:.2f}%"
                    else:
                        signal_type = SignalType.HOLD
                        reason = f"유지 예측: 현재가={current_price:.2f}, 예측가={float(pred):.2f}, 변화율={pct_change:.2f}%"
                    
                    # 신뢰도 계산 (변화율의 절대값을 기준으로, 최대 3%에서 1.0으로 정규화)
                    confidence = min(abs(pct_change) / 3.0, 1.0)
                else:
                    # 현재 가격을 알 수 없는 경우
                    signal_type = SignalType.HOLD
                    reason = f"가격 비교 정보 없음, 예측가={float(pred):.2f}"
                    confidence = 0.3  # 기본 신뢰도
                
                # TradingSignal 생성
                signal = TradingSignal(
                    signal_type=signal_type,
                    confidence=confidence,
                    reason=reason,
                    price=float(pred),
                    metadata={
                        "model_name": self.name,
                        "model_type": self.model_type,
                        "model_version": self.version,
                        "current_price": current_price,
                        "predicted_price": float(pred),
                        "percent_change": pct_change if current_price is not None else None
                    }
                )
                
                # ModelOutput 생성
                results.append(ModelOutput(
                    signal=signal,
                    raw_predictions=np.array([pred]),
                    confidence=confidence,
                    metadata={
                        "model_name": self.name,
                        "model_type": self.model_type,
                        "prediction_time": datetime.now().isoformat(),
                        "sequence_length": self.sequence_length,
                        "feature_dim": self.feature_dim
                    }
                ))
            
            # 다중 예측인 경우 첫 번째 예측 반환, 단일 예측인 경우 해당 예측 반환
            if len(results) > 0:
                return results[0]
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
            return ModelOutput(
                signal=TradingSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    reason=f"예측 중 오류 발생: {str(e)}"
                ),
                confidence=0.0,
                metadata={"error": str(e), "traceback": traceback.format_exc()}
            )
            
    def predict_trend(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        가격 예측을 기반으로 트렌드 방향을 반환
        
        Args:
            X (np.ndarray): 입력 특성, 형태 [samples, sequence_length, features]
            horizon (int, optional): 예측 기간. 기본값은 1 (다음 기간)
            
        Returns:
            np.ndarray: 트렌드 방향 배열 (1: 상승, -1: 하락, 0: 중립)
        """
        if self.model is None:
            self.logger.error("모델이 초기화되지 않았습니다. predict_trend 실패.")
            return np.array([0])  # 기본값으로 중립 반환
        
        try:
            # 현재 가격 (입력 시퀀스의 마지막 종가)
            current_prices = X[:, -1, 0]  # 첫 번째 특성이 종가라고 가정
            
            # 가격 예측
            predicted_prices = self.predict(X)
            
            if len(predicted_prices) == 0:
                self.logger.error("가격 예측 실패")
                return np.array([0])
                
            # 예측된 가격과 현재 가격의 차이를 기반으로 트렌드 계산
            trends = np.zeros(len(predicted_prices))
            
            for i in range(len(predicted_prices)):
                price_diff = predicted_prices[i] - current_prices[i]
                
                # 변화율 계산 (%)
                pct_change = (price_diff / current_prices[i]) * 100
                
                # 트렌드 결정 (변화율이 임계값을 넘는 경우에만 방향 신호)
                threshold = 0.1  # 0.1% 이상의 변화가 있어야 트렌드로 인식
                
                if pct_change > threshold:
                    trends[i] = 1  # 상승 트렌드
                elif pct_change < -threshold:
                    trends[i] = -1  # 하락 트렌드
                else:
                    trends[i] = 0  # 중립 (변화 미미)
                    
                self.logger.info(f"트렌드 예측: 현재가={current_prices[i]:.2f}, 예측가={predicted_prices[i]:.2f}, "
                              f"변화율={pct_change:.2f}%, 트렌드={trends[i]}")
            
            return trends.astype(int)
            
        except Exception as e:
            self.logger.error(f"트렌드 예측 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return np.array([0])  # 오류 발생 시 중립 반환 
            
    def forecast(self, X: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        """
        시계열 예측 생성
        
        Args:
            X (np.ndarray): 입력 시퀀스
            horizon (int): 예측 기간
            **kwargs: 추가 파라미터
            
        Returns:
            np.ndarray: 예측된 값
        """
        if self.model is None:
            logger.error("모델이 초기화되지 않았습니다. forecast 실패.")
            return np.array([])
        
        try:
            # 현재 구현은 단일 스텝 예측만 지원
            if horizon != self.forecast_horizon:
                logger.warning(f"요청된 horizon({horizon})과 모델의 forecast_horizon({self.forecast_horizon})이 다릅니다. "
                            f"모델의 forecast_horizon을 사용합니다.")
            
            # 기본적으로 predict 메소드 활용
            forecasted_values = self.predict(X)
            
            logger.info(f"예측 완료: 형태={forecasted_values.shape}, 요청된 horizon={horizon}")
            return forecasted_values
            
        except Exception as e:
            logger.error(f"forecast 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return np.array([]) 

    def save(self, custom_path: str = None) -> bool:
        """
        학습된 GRU 모델 저장
        
        Args:
            custom_path (str, optional): 사용자 지정 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        if not self.is_trained or self.model is None:
            self.logger.error("저장할 학습된 모델이 없습니다.")
            return False
            
        try:
            # 기본 저장 경로 설정
            if custom_path is None:
                save_dir = os.path.join("models", "saved", self.name)
                os.makedirs(save_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                custom_path = os.path.join(save_dir, f"{self.name}_v{self.version}_{timestamp}.h5")
            
            # 모델 저장
            self.model.save(custom_path)
            self.model_path = custom_path
            
            self.logger.info(f"모델 저장 완료: {custom_path}")
            
            # 모델 번들 디렉토리 생성 (메타데이터 저장용)
            model_dir = os.path.splitext(custom_path)[0] + "_bundle"
            os.makedirs(model_dir, exist_ok=True)
            
            # 메타데이터 저장
            meta_path = os.path.join(model_dir, "meta.json")
            metadata = {
                "name": self.name,
                "version": self.version,
                "type": "gru",
                "sequence_length": self.sequence_length,
                "forecast_horizon": self.forecast_horizon,
                "feature_dim": self.feature_dim,
                "units": self.units,
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "use_attention": self.use_attention,
                "metrics": self.metrics,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 특성 목록 저장 (feature_manager 사용)
            if hasattr(self, 'feature_names') and self.feature_names:
                if hasattr(self, 'feature_manager') and self.feature_manager:
                    self.feature_manager.save_features(self.feature_names, metadata)
                else:
                    # 레거시 방식
                    features_path = os.path.join(model_dir, "features.json")
                    with open(features_path, 'w') as f:
                        json.dump(self.feature_names, f, indent=2)
            
            # 학습 이력 저장
            if self.history is not None:
                history_path = os.path.join(model_dir, "history.json")
                # Keras 히스토리를 일반 dict로 변환
                history_dict = {}
                for key, values in self.history.history.items():
                    history_dict[key] = [float(val) for val in values]  # numpy 값을 float로 변환
                
                with open(history_path, 'w') as f:
                    json.dump(history_dict, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def load(self, custom_path: str = None) -> bool:
        """
        학습된 GRU 모델 로드
        
        Args:
            custom_path (str, optional): 사용자 지정 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            # 로드할 경로 결정
            if custom_path is None:
                if self.model_path is None:
                    self.logger.error("로드할 모델 경로가 지정되지 않았습니다.")
                    return False
                custom_path = self.model_path
            
            # 모델 로드
            self.model = load_model(custom_path, compile=True)
            self.model_path = custom_path
            self.is_trained = True
            
            self.logger.info(f"모델 로드 완료: {custom_path}")
            
            # 모델 번들 디렉토리 확인
            model_dir = os.path.splitext(custom_path)[0] + "_bundle"
            if os.path.isdir(model_dir):
                # 메타데이터 로드
                meta_path = os.path.join(model_dir, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # 모델 파라미터 설정
                    if 'sequence_length' in metadata:
                        self.sequence_length = metadata['sequence_length']
                    if 'forecast_horizon' in metadata:
                        self.forecast_horizon = metadata['forecast_horizon']
                    if 'feature_dim' in metadata:
                        self.feature_dim = metadata['feature_dim']
                    if 'units' in metadata:
                        self.units = metadata['units']
                    if 'dropout_rate' in metadata:
                        self.dropout_rate = metadata['dropout_rate']
                    if 'learning_rate' in metadata:
                        self.learning_rate = metadata['learning_rate']
                    if 'use_attention' in metadata:
                        self.use_attention = metadata['use_attention']
                    if 'metrics' in metadata:
                        self.metrics = metadata['metrics']
                    
                    self.logger.info(f"메타데이터 로드됨: {meta_path}")
                
                # 특성 목록 로드
                features_path = os.path.join(model_dir, "features.json")
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        features_data = json.load(f)
                    
                    # 새로운 형식 (dict) 또는 레거시 형식 (list) 처리
                    if isinstance(features_data, dict) and 'features' in features_data:
                        self.feature_names = features_data['features']
                    elif isinstance(features_data, list):
                        self.feature_names = features_data
                    
                    self.logger.info(f"특성 목록 로드됨: {len(self.feature_names)}개")
                    
                    # feature_manager 사용
                    if hasattr(self, 'feature_manager') and self.feature_manager:
                        self.feature_manager.expected_features = self.feature_names
                        self.feature_manager.expected_features_count = len(self.feature_names)
            
            # 입력 형태 확인 및 feature_dim 설정
            if self.model is not None:
                input_shape = self.model.input_shape
                if input_shape is not None and len(input_shape) > 2:
                    self.feature_dim = input_shape[-1]
                    self.logger.info(f"모델 입력 형태: {input_shape}, 특성 수: {self.feature_dim}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

def create_sequence_data(X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    GRU 모델을 위한 시퀀스 데이터 생성
    
    Args:
        X (np.ndarray): 피처 데이터
        y (np.ndarray): 타겟 데이터
        sequence_length (int): 시퀀스 길이
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X_seq와 y_seq 데이터
    """
    try:
        # 데이터 유효성 검사
        if X.shape[0] != y.shape[0]:
            logger.error(f"피처와 타겟 데이터의 길이가 일치하지 않습니다: X={X.shape}, y={y.shape}")
            return np.array([]), np.array([])
            
        # 시퀀스 데이터용 빈 리스트
        X_seq = []
        y_seq = []
        
        # 시퀀스 데이터 생성
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        # 비어있는지 확인
        if not X_seq or not y_seq:
            logger.warning("생성된 시퀀스 데이터가 없습니다.")
            return np.array([]), np.array([])
        
        # 넘파이 배열로 변환
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # 차원이 누락된 경우 추가
        if len(y_seq.shape) == 1:
            y_seq = y_seq.reshape(-1, 1)
        
        logger.info(f"시퀀스 데이터 생성 완료: X 형태={X_seq.shape}, y 형태={y_seq.shape}")
        
        return X_seq, y_seq
        
    except Exception as e:
        logger.error(f"시퀀스 데이터 생성 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return np.array([]), np.array([]) 