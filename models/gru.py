"""
GRU 모델을 사용한 비트코인 방향 예측 모델

이 모듈은 시계열 분류를 위한 GRU 모델을 구현합니다.
최적화된 파라미터를 사용하여 모델을 구축하고 H5 형식으로 저장합니다.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from datetime import datetime
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import GRU, Dense, Dropout, LayerNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.regularizers import l1_l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import traceback
import json
from keras import backend as K

from models.base import ClassificationModel, TimeSeriesModel
from utils.logging import get_logger

# 로거 초기화
logger = get_logger(__name__)

# 재현성을 위한 랜덤 시드 설정
tf.random.set_seed(42)
np.random.seed(42)

# GPU 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"모델이 {len(gpus)} GPU(s)를 사용할 수 있습니다: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU 메모리 증가 활성화됨")
    except RuntimeError as e:
        logger.warning(f"GPU 메모리 증가 설정 실패: {e}")
else:
    logger.warning("사용 가능한 GPU가 없습니다. 훈련이 CPU에서 진행되어 느릴 수 있습니다.")

class GRUDirectionModel(ClassificationModel):
    """비트코인 가격 방향성 분류를 위한 GRU 모델"""
    
    def __init__(self, 
                name: str = "GRUDirection", 
                version: str = "1.0.0",
                sequence_length: int = 31,        # 최적화된 값
                units: List[int] = [62, 45],      # 최적화된 값
                dropout_rate: float = 0.4720,     # 최적화된 값
                learning_rate: float = 0.00253,   # 최적화된 값
                batch_size: int = 64,             # 최적화된 값
                epochs: int = 100):
        """
        방향성 분류 GRU 모델 초기화 - 최적화된 하이퍼파라미터 사용
        
        Args:
            name (str): 모델 이름
            version (str): 모델 버전
            sequence_length (int): 사용할 과거 시간 단계 수
            units (List[int]): 각 GRU 레이어의 유닛 수 리스트
            dropout_rate (float): 각 GRU 레이어 이후의 Dropout 비율
            learning_rate (float): Adam 옵티마이저 학습률
            batch_size (int): 학습 배치 크기
            epochs (int): 최대 학습 에폭 수
        """
        super().__init__(name, version)
        
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # 파라미터 저장
        self.params = {
            'sequence_length': sequence_length,
            'units': units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        }
        
        self.model = None
        self.feature_dim = None
        self.history = None
        self.classes_ = np.array([0, 1])  # 0: 하락, 1: 상승
        
        logger.info(f"{self.name} 모델을 {len(units)}개 GRU 레이어로 초기화했습니다")
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        간소화된 GRU 모델 아키텍처 구축 - 파라미터 수 감소
        
        Args:
            input_shape (Tuple[int, int]): 입력 형태 (sequence_length, features)
        """
        self.feature_dim = input_shape[1]
        
        model = Sequential()
        
        # 입력 데이터에 대해 배치 정규화 적용
        model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))
        
        # GRU 레이어 추가 (2층으로 축소)
        for i, units in enumerate(self.units):
            return_sequences = i < len(self.units) - 1
            
            # 첫 레이어에 입력 형태 지정
            if i == 0:
                model.add(GRU(units, 
                            return_sequences=return_sequences, 
                            recurrent_dropout=0.2,  # 순환 드롭아웃 증가
                            recurrent_regularizer=l1_l2(l1=0.001, l2=0.001)))  # L1 정규화 추가
            else:
                model.add(GRU(units, 
                            return_sequences=return_sequences,
                            recurrent_dropout=0.2,  # 순환 드롭아웃 증가
                            recurrent_regularizer=l1_l2(l1=0.001, l2=0.001)))  # L1 정규화 추가
            
            # 마지막 GRU 레이어 이후에만 배치 정규화 적용
            if i == len(self.units) - 1:
                model.add(tf.keras.layers.BatchNormalization())
            
            model.add(Dropout(self.dropout_rate))
        
        # 이진 분류를 위한 출력 레이어 (추가 밀집층 제거)
        model.add(Dense(1, activation='sigmoid'))
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # 모델 요약 정보 출력 및 파라미터 수 로깅
        total_params = model.count_params()
        logger.info(f"GRU 모델 총 파라미터 수: {total_params:,}")
        trainable_params = sum([K.count_params(w) for w in model.trainable_weights])
        logger.info(f"학습 가능한 파라미터 수: {trainable_params:,}")
        
        self.model = model
        model.summary(print_fn=logger.info)
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             class_weights: Optional[Dict[int, float]] = None,
             early_stopping_patience: int = 20,
             reduce_lr_patience: int = 10,
             reduce_lr_factor: float = 0.5,
             reduce_lr_min_lr: float = 0.00001) -> Dict[str, Any]:
        """
        GRU 방향성 모델 훈련 - 향상된 학습 프로세스
        
        Args:
            X_train (np.ndarray): 훈련 시퀀스, 형태 (samples, sequence_length, features)
            y_train (np.ndarray): 훈련 타겟 (0: 하락, 1: 상승)
            X_val (Optional[np.ndarray]): 검증 시퀀스
            y_val (Optional[np.ndarray]): 검증 타겟
            class_weights (Optional[Dict[int, float]]): 불균형 데이터용 클래스 가중치
            early_stopping_patience (int): Early stopping 인내심
            reduce_lr_patience (int): 학습률 감소 인내심
            reduce_lr_factor (float): 학습률 감소 비율
            reduce_lr_min_lr (float): 최소 학습률
            
        Returns:
            Dict[str, Any]: 훈련 메트릭
        """
        if self.model is None:
            logger.error("모델이 초기화되지 않았습니다. build_model()을 먼저 호출하세요.")
            return {"error": "Model not initialized"}
        
        start_time = pd.Timestamp.now()
        logger.info(f"Training {self.__class__.__name__} model on {len(X_train)} samples")
        
        # 콜백 준비
        callbacks = []
        
        # Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))
        
        # 학습률 감소 - 파라미터 사용자 정의 가능
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=reduce_lr_min_lr,
            verbose=1
        ))
        
        # 모델 체크포인트
        os.makedirs("saved_models", exist_ok=True)
        checkpoint_path = os.path.join("saved_models", f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
        callbacks.append(ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))
        
        # TensorBoard 콜백 추가
        log_dir = os.path.join("logs", f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        callbacks.append(TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        ))
        
        # 검증 데이터 준비
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # 입력 데이터 검증 및 전처리
        # NaN 체크 및 처리
        X_train_nan_count = np.isnan(X_train).sum()
        if X_train_nan_count > 0:
            logger.warning(f"X_train에서 {X_train_nan_count}개의 NaN 값이 발견되어 0으로 대체합니다.")
            X_train = np.nan_to_num(X_train, nan=0.0)
        
        # 무한값 처리
        X_train_inf_count = np.isinf(X_train).sum()
        if X_train_inf_count > 0:
            logger.warning(f"X_train에서 {X_train_inf_count}개의 무한값이 발견되어 0으로 대체합니다.")
            X_train = np.nan_to_num(X_train, posinf=0.0, neginf=0.0)
        
        # 검증 데이터도 동일하게 처리
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_nan_count = np.isnan(X_val).sum()
            if X_val_nan_count > 0:
                logger.warning(f"X_val에서 {X_val_nan_count}개의 NaN 값이 발견되어 0으로 대체합니다.")
                X_val = np.nan_to_num(X_val, nan=0.0)
            
            X_val_inf_count = np.isinf(X_val).sum()
            if X_val_inf_count > 0:
                logger.warning(f"X_val에서 {X_val_inf_count}개의 무한값이 발견되어 0으로 대체합니다.")
                X_val = np.nan_to_num(X_val, posinf=0.0, neginf=0.0)
            
            validation_data = (X_val, y_val)
        
        # 모델 훈련
        logger.info(f"모델 입력 형태: {X_train.shape}")
        
        # 실제 훈련 실행
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2
        )
        
        # 훈련 결과 로깅
        logger.info("=== 훈련 완료 ===")
        logger.info(f"총 에폭: {len(self.history.history['loss'])}")
        logger.info(f"최종 훈련 손실: {self.history.history['loss'][-1]:.4f}")
        if 'val_loss' in self.history.history:
            logger.info(f"최종 검증 손실: {self.history.history['val_loss'][-1]:.4f}")
        
        # 훈련 메트릭 계산
        y_pred_proba = self.model.predict(X_train)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        train_accuracy = accuracy_score(y_train, y_pred)
        train_precision = precision_score(y_train, y_pred, average='binary', zero_division=0)
        train_recall = recall_score(y_train, y_pred, average='binary', zero_division=0)
        train_f1 = f1_score(y_train, y_pred, average='binary', zero_division=0)
        
        # 메트릭 로깅
        logger.info(f"훈련 정확도: {train_accuracy:.4f}, 정밀도: {train_precision:.4f}, 재현율: {train_recall:.4f}, F1: {train_f1:.4f}")
        
        # 메트릭 저장
        metrics = {
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1': float(train_f1),
            'training_time': (pd.Timestamp.now() - start_time).total_seconds()
        }
        
        # 검증 메트릭 추가
        if validation_data is not None:
            y_val_pred_proba = self.model.predict(X_val)
            y_val_pred = (y_val_pred_proba > 0.5).astype(int)
            
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred, average='binary', zero_division=0)
            val_recall = recall_score(y_val, y_val_pred, average='binary', zero_division=0)
            val_f1 = f1_score(y_val, y_val_pred, average='binary', zero_division=0)
            
            # 검증 메트릭 로깅
            logger.info(f"검증 정확도: {val_accuracy:.4f}, 정밀도: {val_precision:.4f}, 재현율: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            metrics.update({
                'val_accuracy': float(val_accuracy),
                'val_precision': float(val_precision),
                'val_recall': float(val_recall),
                'val_f1': float(val_f1)
            })
        
        self.metrics.update(metrics)
        self.is_trained = True
        self.last_update = datetime.now()
        
        logger.info(f"훈련 완료. 소요 시간: {metrics['training_time']:.2f}초")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        방향성 예측 (0: 하락, 1: 상승)
        
        Args:
            X (np.ndarray): 입력 시퀀스, 형태 (samples, sequence_length, features)
            
        Returns:
            np.ndarray: 예측 레이블 (0 또는 1)
        """
        if not self.is_trained or self.model is None:
            logger.warning("모델이 훈련되지 않았습니다.")
            return np.array([])
        
        # 예측 확률
        probas = self.predict_proba(X)
        
        # 임계값 0.5를 사용하여 클래스 레이블 결정
        return (probas[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        확률 예측
        
        Args:
            X (np.ndarray): 입력 시퀀스, 형태 (samples, sequence_length, features)
            
        Returns:
            np.ndarray: 예측 확률, 형태 (samples, 2)
        """
        if not self.is_trained or self.model is None:
            logger.warning("모델이 훈련되지 않았습니다.")
            return np.array([])
        
        # NaN 값 처리
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 예측 (형태는 (samples, 1))
        y_pred = self.model.predict(X_clean)
        
        # 2열 형태로 변환 (0과 1 클래스에 대한 확률)
        # 첫 번째 열은 0 클래스 확률 (1 - y_pred)
        # 두 번째 열은 1 클래스 확률 (y_pred)
        return np.hstack([1 - y_pred, y_pred])
    
    def save_h5(self, filepath: Optional[str] = None) -> str:
        """
        모델을 H5 형식으로 저장 (옵티마이저 상태 제외)
        
        Args:
            filepath (Optional[str]): 저장할 파일 경로, None이면 기본 경로 사용
            
        Returns:
            str: 모델이 저장된 경로
        """
        if self.model is None:
            logger.error("저장할 모델이 없습니다")
            raise ValueError("No model to save")
        
        if filepath is None:
            # 기본 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.h5"
            filepath = os.path.join("saved_models", filename)
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            # 옵티마이저 상태 제외하고 저장
            self.model.save(filepath, include_optimizer=False)
            logger.info(f"모델이 {filepath}에 저장되었습니다 (옵티마이저 상태 제외)")
            
            # 설정 정보 별도 저장
            config_path = f"{filepath[:-3]}_config.json"
            config = {
                'name': self.name,
                'version': self.version,
                'sequence_length': self.sequence_length,
                'units': self.units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'tensorflow_version': tf.__version__,
                'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"모델 설정이 {config_path}에 저장되었습니다")
            
            return filepath
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    @classmethod
    def load_h5(cls, filepath: str) -> 'GRUDirectionModel':
        """
        H5 파일에서 모델 로드
        
        Args:
            filepath (str): 모델 파일 경로
            
        Returns:
            GRUDirectionModel: 로드된 모델
        """
        try:
            # 설정 파일 경로
            config_path = f"{filepath[:-3]}_config.json"
            
            # 설정 파일이 있으면 읽기
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 설정으로 모델 인스턴스 생성
                model_instance = cls(
                    name=config.get('name', 'GRUDirection'),
                    version=config.get('version', '1.0.0'),
                    sequence_length=config.get('sequence_length', 31),
                    units=config.get('units', [62, 45]),
                    dropout_rate=config.get('dropout_rate', 0.4720),
                    learning_rate=config.get('learning_rate', 0.00253),
                    batch_size=config.get('batch_size', 64),
                    epochs=config.get('epochs', 100)
                )
            else:
                # 설정 파일이 없으면 기본값으로 생성
                model_instance = cls()
                logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}. 기본 설정 사용")
            
            # 모델 로드
            keras_model = tf.keras.models.load_model(filepath, compile=False)
            
            # 모델 재컴파일
            keras_model.compile(
                optimizer=Adam(learning_rate=model_instance.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            # 모델 설정
            model_instance.model = keras_model
            model_instance.is_trained = True
            model_instance.last_update = datetime.now()
            
            logger.info(f"모델을 {filepath}에서 로드했습니다")
            return model_instance
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def plot_history(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        훈련 이력 시각화
        
        Args:
            figsize (Tuple[int, int]): 그림 크기
        """
        if self.history is None:
            logger.warning("훈련 이력이 없습니다")
            return
        
        plt.figure(figsize=figsize)
        
        # 손실 그래프
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 정확도 그래프
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

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
        if self.model is None:
            logger.error("모델이 초기화되지 않았습니다. build_model()을 먼저 호출하세요.")
            return {"error": "Model not initialized"}
        
        try:
            # 예측 수행
            y_pred_proba = self.predict_proba(X_test)
            
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                # 확률에서 클래스로 변환
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                # 이진 분류의 경우 임계값 적용
                y_pred = (y_pred_proba > 0.5).astype(int)
            
            # 평가 지표 계산
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # 결과 저장
            metrics = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1)
            }
            
            logger.info(f"평가 결과: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"모델 평가 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

class GRUPriceModel(TimeSeriesModel):
    """비트코인 가격 예측을 위한 GRU 모델"""
    
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
        가격 예측 GRU 모델 초기화 - 개선된 구조
        
        Args:
            name (str): 모델 이름
            version (str): 모델 버전
            sequence_length (int): 참고할 과거 시간 단계 수
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
        self.scaler = None
        
        logger.info(f"{self.name} 모델을 {len(units)}개 GRU 레이어로 초기화했습니다. 어텐션 사용: {use_attention}")
    
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
        model.summary(print_fn=logger.info)
        
        # 총 파라미터 수와 훈련 가능한 파라미터 수 로깅
        total_params = model.count_params()
        trainable_params = sum([K.count_params(w) for w in model.trainable_weights])
        logger.info(f"GRU 가격 예측 모델 총 파라미터 수: {total_params:,}")
        logger.info(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              early_stopping_patience: int = 15,     # 인내심 증가
              reduce_lr_patience: int = 8,
              sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        GRU 가격 예측 모델 훈련 - 향상된 버전
        
        Args:
            X_train (np.ndarray): 훈련 시퀀스, 형태 (samples, sequence_length, features)
            y_train (np.ndarray): 훈련 타겟 
            X_val (Optional[np.ndarray]): 검증 시퀀스
            y_val (Optional[np.ndarray]): 검증 타겟
            early_stopping_patience (int): Early stopping 인내심
            reduce_lr_patience (int): 학습률 감소 인내심
            sample_weight (Optional[np.ndarray]): 샘플 가중치
            
        Returns:
            Dict[str, float]: 훈련 메트릭
        """
        if self.model is None:
            logger.error("모델이 초기화되지 않았습니다. build_model()을 먼저 호출하세요.")
            return {"error": "Model not initialized"}
        
        start_time = pd.Timestamp.now()
        logger.info(f"Training {self.__class__.__name__} model on {len(X_train)} samples")
        
        # 콜백 준비
        callbacks = []
        
        # 체크포인트 저장 디렉토리
        checkpoint_dir = os.path.join('checkpoints', f"{self.name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 모델 체크포인트
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.h5")
        callbacks.append(ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))
        
        # Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))
        
        # 학습률 감소
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.3,  # 더 큰 감소 폭
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ))
        
        # 텐서보드 로깅
        log_dir = os.path.join('logs', 'tensorboard', f"{self.name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        callbacks.append(TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ))
        
        # 데이터 형태 검증
        if len(X_train.shape) != 3:
            logger.error(f"X_train은 3차원이어야 합니다. (samples, sequence_length, features). 현재 형태: {X_train.shape}")
            return {"error": f"Invalid X_train shape: {X_train.shape}"}
        
        # 교차 검증 데이터가 없는 경우 훈련 데이터에서 분할
        if X_val is None or y_val is None:
            val_split = 0.2
            val_size = int(len(X_train) * val_split)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
            if sample_weight is not None:
                sample_weight = sample_weight[:-val_size]
            logger.info(f"검증 데이터 분할: 훈련 {len(X_train)} 샘플, 검증 {len(X_val)} 샘플")
        
        # 모델 훈련
        try:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True,
                sample_weight=sample_weight
            )
            
            self.history = history.history
            
            # 훈련 메트릭
            train_metrics = {
                'train_loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None,
                'train_mae': history.history['mae'][-1],
                'val_mae': history.history['val_mae'][-1] if 'val_mae' in history.history else None,
                'training_time': (pd.Timestamp.now() - start_time).total_seconds()
            }
            
            # 모델 메트릭 저장
            self.metrics.update(train_metrics)
            self.is_trained = True
            self.last_update = pd.Timestamp.now()
            
            logger.info(f"GRU 가격 예측 모델 훈련 완료. Loss: {train_metrics['train_loss']:.4f}, MAE: {train_metrics['train_mae']:.4f}")
            
            # 검증 데이터로 테스트
            if X_val is not None:
                y_pred = self.model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_val, y_pred)
                mape = np.mean(np.abs((y_val - y_pred) / np.maximum(np.abs(y_val), 1e-10))) * 100
                r2 = r2_score(y_val, y_pred)
                
                validation_metrics = {
                    'validation_mse': mse,
                    'validation_rmse': rmse,
                    'validation_mae': mae,
                    'validation_mape': mape,
                    'validation_r2': r2
                }
                
                self.metrics.update(validation_metrics)
                logger.info(f"검증 성능: RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"모델 훈련 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
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
            logger.error("모델이 훈련되지 않았습니다. evaluate 메서드를 호출하기 전에 모델을 훈련하세요.")
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
            
            logger.info(f"평가 결과: RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, 방향 정확도: {direction_accuracy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"모델 평가 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'mse': float('inf'),
                'rmse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf'),
                'r2': 0.0,
                'direction_accuracy': 0.0
            }
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        모델을 사용하여 가격 예측
        
        Args:
            X (np.ndarray): 입력 특성, 형태 [samples, sequence_length, features]
            
        Returns:
            np.ndarray: 예측된 가격 배열
        """
        if self.model is None:
            self.logger.error("모델이 초기화되지 않았습니다. predict 실패.")
            return np.array([])
        
        try:
            # 입력 데이터 전처리 및 형태 확인
            if len(X.shape) != 3:
                self.logger.error(f"입력 형태가 잘못되었습니다: {X.shape}, 필요한 형태: [samples, sequence_length, features]")
                return np.array([])
            
            # 예측 실행
            predictions = self.model.predict(X, verbose=0)
            self.logger.info(f"예측 완료: 형태={predictions.shape}")
            
            return predictions
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return np.array([])
            
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