"""
앙상블 모델의 기본 클래스를 정의하는 모듈입니다.

다양한 앙상블 전략 구현을 위한 인터페이스와 기본 클래스를 제공합니다.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
import json
import logging
import traceback

from utils.logging import get_logger
from models.base import ModelBase, ClassificationModel, RegressionModel
from utils.constants import SignalType
from models.signal import TradingSignal, ModelOutput

logger = get_logger(__name__)

# 앙상블 코어 모듈 (Ensemble Core Module)
# 
# 이 모듈은 다양한 모델을 결합하는 앙상블의 핵심 클래스와 기능을 제공합니다.
# 모델 결합, 가중치 조정, 예측 생성 등의 기능을 담당합니다.

class EnsembleBase:
    """앙상블 모델의 기본 클래스"""
    
    def __init__(self, 
                name: str = "EnsembleBase", 
                version: str = "1.0.0"):
        """
        앙상블 기본 클래스 초기화
        
        Args:
            name (str): 모델 이름
            version (str): 모델 버전
        """
        self.name = name
        self.version = version
        self.is_trained = False
        self.models = []
        self.weights = None
        self.metrics = {}
        self.last_update = None
        self.logger = get_logger(f"{__name__}.{self.name}")
        
        self.logger.info(f"{self.name} v{self.version} 초기화됨")
    
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """
        가중치를 정규화하여 합이 1이 되도록 함
        
        Args:
            weights (List[float]): 정규화할 가중치 리스트
            
        Returns:
            List[float]: 정규화된 가중치 리스트
        """
        if not weights:
            return []
            
        total = sum(weights)
        if total == 0:
            # 모든 가중치가 0이면 균등 분배
            return [1.0 / len(weights)] * len(weights)
            
        return [w / total for w in weights]
    
    def save(self, directory: str) -> str:
        """
        앙상블 모델 저장
        
        Args:
            directory (str): 저장할 디렉토리 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 디렉토리가 없으면 생성
            os.makedirs(directory, exist_ok=True)
            
            # 메타데이터 생성
            metadata = {
                'name': self.name,
                'version': self.version,
                'is_trained': self.is_trained,
                'metrics': self.metrics,
                'last_update': datetime.now().isoformat() if self.last_update is None else self.last_update.isoformat(),
                'class': self.__class__.__name__
            }
            
            # 가중치 정보 추가
            if hasattr(self, 'weights') and self.weights is not None:
                metadata['weights'] = self.weights
            
            # 매개변수 정보 추가
            if hasattr(self, 'params'):
                metadata['params'] = self.params
            
            # 특성 정보 추가
            if hasattr(self, 'expected_features_count'):
                metadata['expected_features_count'] = self.expected_features_count
            
            # 메타데이터 저장
            metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # 개별 모델 저장 (하위 클래스에서 구현)
            
            self.logger.info(f"앙상블 메타데이터 저장됨: {metadata_path}")
            return metadata_path
            
        except Exception as e:
            self.logger.error(f"앙상블 저장 중 오류: {str(e)}")
            self.logger.error(traceback.format_exc())
            return ""
    
    @classmethod
    def load(cls, directory: str) -> Optional['EnsembleBase']:
        """
        앙상블 모델 로드
        
        Args:
            directory (str): 로드할 디렉토리 경로
            
        Returns:
            Optional[EnsembleBase]: 로드된 앙상블 모델
        """
        try:
            # 메타데이터 파일 경로 설정
            metadata_files = [f for f in os.listdir(directory) if f.endswith('_metadata.json')]
            
            if not metadata_files:
                logger.error(f"디렉토리에서 메타데이터 파일을 찾을 수 없음: {directory}")
                return None
            
            # 첫 번째 메타데이터 파일 사용
            metadata_path = os.path.join(directory, metadata_files[0])
            
            # 메타데이터 로드
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # 앙상블 객체 생성
            ensemble = cls(
                name=metadata.get('name', 'LoadedEnsemble'),
                version=metadata.get('version', '1.0.0')
            )
            
            # 기본 속성 설정
            ensemble.is_trained = metadata.get('is_trained', False)
            ensemble.metrics = metadata.get('metrics', {})
            if 'last_update' in metadata:
                ensemble.last_update = datetime.fromisoformat(metadata['last_update'])
            
            # 가중치 설정
            if 'weights' in metadata:
                ensemble.weights = metadata['weights']
            
            # 매개변수 설정
            if 'params' in metadata and hasattr(ensemble, 'params'):
                ensemble.params = metadata['params']
            
            # 특성 수 정보 설정
            if 'expected_features_count' in metadata:
                ensemble.expected_features_count = metadata['expected_features_count']
            
            logger.info(f"앙상블 메타데이터 로드됨: {metadata_path}")
            return ensemble
            
        except Exception as e:
            logger.error(f"앙상블 로드 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _validate_features(self, X: np.ndarray, expected_count: Optional[int] = None) -> np.ndarray:
        """
        입력 특성 검증 및 조정
        
        Args:
            X (np.ndarray): 입력 특성 배열
            expected_count (Optional[int]): 예상되는 특성 수 (None인 경우 self.expected_features_count 사용)
            
        Returns:
            np.ndarray: 조정된 특성 배열
        """
        # 예상 특성 수 설정
        if expected_count is None:
            if hasattr(self, 'expected_features_count'):
                expected_count = self.expected_features_count
            else:
                # 예상 특성 수가 없으면 원본 반환
                return X
        
        # 현재 특성 수 확인
        current_count = X.shape[1]
        self.logger.info(f"특성 수: 실제={current_count}, 예상={expected_count}")
        
        # 특성 수 조정
        if current_count > expected_count:
            self.logger.warning(f"특성 수가 많음: {current_count} > {expected_count}, 처음 {expected_count}개만 사용합니다.")
            adjusted_X = X[:, :expected_count]
            
            # 처리 후 다시 검증
            if adjusted_X.shape[1] != expected_count:
                self.logger.error(f"특성 수 불일치! 예상={expected_count}, 실제={adjusted_X.shape[1]}")
            
            return adjusted_X
        elif current_count < expected_count:
            self.logger.warning(f"특성 수가 적음: {current_count} < {expected_count}, 부족한 특성을 0으로 채웁니다.")
            
            # 부족한 특성 채우기
            padding = np.zeros((X.shape[0], expected_count - current_count))
            adjusted_X = np.concatenate([X, padding], axis=1)
            
            # 처리 후 다시 검증
            if adjusted_X.shape[1] != expected_count:
                self.logger.error(f"특성 수 불일치! 예상={expected_count}, 실제={adjusted_X.shape[1]}")
            
            return adjusted_X
        else:
            # 특성 수가 일치하면 원본 반환
            return X


class VotingEnsemble(EnsembleBase):
    """투표 기반 분류 앙상블 모델"""
    
    def __init__(self, 
                name: str = "VotingEnsemble", 
                version: str = "1.0.0",
                models: Optional[List[ClassificationModel]] = None,
                weights: Optional[List[float]] = None,
                voting: str = 'soft'):
        """
        투표 앙상블 초기화
        
        Args:
            name (str): 모델 이름
            version (str): 모델 버전
            models (Optional[List[ClassificationModel]]): 분류 모델 리스트
            weights (Optional[List[float]]): 각 모델의 가중치
            voting (str): 투표 전략 ('hard' 또는 'soft')
        """
        super().__init__(name, version)
        
        self.models = models or []
        self.weights = weights
        self.voting = voting
        
        # 가중치 정규화
        if self.weights is not None:
            self.weights = self._normalize_weights(self.weights)
        elif self.models:
            # 기본값: 균등 가중치
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        # 매개변수 저장
        self.params = {
            'voting': voting
        }
        
        self.logger.info(f"{self.name} 모델이 {len(self.models)}개의 서브 모델로 초기화됨")
    
    def add_model(self, 
                 model: ClassificationModel, 
                 weight: float = 1.0) -> None:
        """
        앙상블에 모델 추가
        
        Args:
            model (ClassificationModel): 추가할 모델
            weight (float): 모델 가중치
        """
        self.models.append(model)
        
        # 가중치 업데이트
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)
            self.weights = self._normalize_weights(self.weights)
        
        self.logger.info(f"모델 {model.name}이 가중치 {weight}로 앙상블에 추가됨")
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        앙상블로 예측 수행
        
        Args:
            X (np.ndarray): 입력 특성
            **kwargs: 추가 매개변수
            
        Returns:
            np.ndarray: 예측된 클래스 레이블
        """
        # 구현 예정
        pass
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        클래스 확률 예측
        
        Args:
            X (np.ndarray): 입력 특성
            **kwargs: 추가 매개변수
            
        Returns:
            np.ndarray: 예측 확률
        """
        # 구현 예정
        pass


class StackingEnsemble(EnsembleBase):
    """스태킹 앙상블 모델"""
    
    def __init__(self, 
                name: str = "StackingEnsemble",
                version: str = "1.0.0",
                base_models: Optional[List[ClassificationModel]] = None,
                meta_model: Optional[ClassificationModel] = None):
        """
        스태킹 앙상블 초기화
        
        Args:
            name (str): 모델 이름
            version (str): 모델 버전
            base_models (Optional[List[ClassificationModel]]): 기본 모델 리스트
            meta_model (Optional[ClassificationModel]): 메타 모델
        """
        super().__init__(name, version)
        
        self.base_models = base_models or []
        self.meta_model = meta_model
        
        self.logger.info(f"{self.name} 모델이 {len(self.base_models)}개의 기본 모델로 초기화됨")
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        스태킹 앙상블로 예측 수행
        
        Args:
            X (np.ndarray): 입력 특성
            **kwargs: 추가 매개변수
            
        Returns:
            np.ndarray: 예측된 클래스 레이블
        """
        # 구현 예정
        pass


class HybridEnsemble(EnsembleBase):
    """하이브리드 앙상블 모델 - 분류 모델과 회귀 모델 결합"""
    
    def __init__(self, 
                name: str = "HybridEnsemble",
                version: str = "1.0.0",
                direction_models: Optional[List[ClassificationModel]] = None,
                price_models: Optional[List[RegressionModel]] = None,
                direction_weights: Optional[List[float]] = None,
                price_weights: Optional[List[float]] = None):
        """
        하이브리드 앙상블 초기화
        
        Args:
            name (str): 모델 이름
            version (str): 모델 버전
            direction_models (Optional[List[ClassificationModel]]): 방향 예측 모델 리스트
            price_models (Optional[List[RegressionModel]]): 가격 예측 모델 리스트
            direction_weights (Optional[List[float]]): 방향 모델 가중치
            price_weights (Optional[List[float]]): 가격 모델 가중치
        """
        super().__init__(name, version)
        
        self.direction_models = direction_models or []
        self.price_models = price_models or []
        
        # 가중치 설정
        if direction_weights is not None:
            self.direction_weights = self._normalize_weights(direction_weights)
        elif self.direction_models:
            self.direction_weights = [1.0 / len(self.direction_models)] * len(self.direction_models)
        else:
            self.direction_weights = []
            
        if price_weights is not None:
            self.price_weights = self._normalize_weights(price_weights)
        elif self.price_models:
            self.price_weights = [1.0 / len(self.price_models)] * len(self.price_models)
        else:
            self.price_weights = []
        
        self.logger.info(f"{self.name} 모델이 {len(self.direction_models)}개의 방향 모델과 {len(self.price_models)}개의 가격 모델로 초기화됨")
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelOutput:
        """
        하이브리드 앙상블로 예측 수행
        
        Args:
            X (np.ndarray): 입력 특성
            **kwargs: 추가 매개변수
                market_data (pd.DataFrame): 시장 데이터
                
        Returns:
            ModelOutput: 표준화된 모델 출력 객체
        """
        # 방향 예측 모델들 실행
        direction_outputs = []
        direction_confidences = []
        
        for i, model in enumerate(self.direction_models):
            try:
                weight = self.direction_weights[i] if i < len(self.direction_weights) else 1.0
                direction_output = model.predict(X, **kwargs)
                direction_outputs.append(direction_output)
                direction_confidences.append(direction_output.confidence * weight)
            except Exception as e:
                self.logger.error(f"방향 모델 {model.name} 예측 중 오류: {str(e)}")
        
        # 가격 예측 모델들 실행
        price_outputs = []
        price_confidences = []
        
        for i, model in enumerate(self.price_models):
            try:
                weight = self.price_weights[i] if i < len(self.price_weights) else 1.0
                price_output = model.predict(X, **kwargs)
                price_outputs.append(price_output)
                price_confidences.append(price_output.confidence * weight)
            except Exception as e:
                self.logger.error(f"가격 모델 {model.name} 예측 중 오류: {str(e)}")
        
        # 결과가 없는 경우
        if not direction_outputs and not price_outputs:
            self.logger.error("모든 모델 예측이 실패했습니다.")
            return ModelOutput(
                signal=TradingSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    reason="모든 모델 예측이 실패했습니다."
                ),
                confidence=0.0,
                metadata={"error": "All model predictions failed"}
            )
        
        # 방향 모델들의 가중 평균 신호 결정
        direction_signal = None
        direction_confidence = 0.0
        
        if direction_outputs:
            # 각 방향에 대한 가중치 합계 계산
            buy_weight = sum(output.confidence * weight for output, weight in 
                          zip(direction_outputs, self.direction_weights) 
                          if output.signal.signal == SignalType.BUY)
            
            sell_weight = sum(output.confidence * weight for output, weight in 
                           zip(direction_outputs, self.direction_weights) 
                           if output.signal.signal == SignalType.SELL)
            
            hold_weight = sum(output.confidence * weight for output, weight in 
                           zip(direction_outputs, self.direction_weights) 
                           if output.signal.signal == SignalType.HOLD)
            
            # 가장 높은 가중치를 가진 방향 선택
            max_weight = max(buy_weight, sell_weight, hold_weight)
            
            if max_weight == buy_weight:
                direction_signal = SignalType.BUY
                direction_confidence = buy_weight / sum(self.direction_weights) if sum(self.direction_weights) > 0 else 0.0
            elif max_weight == sell_weight:
                direction_signal = SignalType.SELL
                direction_confidence = sell_weight / sum(self.direction_weights) if sum(self.direction_weights) > 0 else 0.0
            else:
                direction_signal = SignalType.HOLD
                direction_confidence = hold_weight / sum(self.direction_weights) if sum(self.direction_weights) > 0 else 0.0
        
        # 가격 모델들의 가중 평균 계산
        predicted_price = None
        price_confidence = 0.0
        
        if price_outputs:
            total_weight = sum(self.price_weights)
            if total_weight > 0:
                # 가중 평균 가격 계산
                predicted_price = sum(output.signal.price * weight for output, weight in 
                                   zip(price_outputs, self.price_weights) 
                                   if output.signal.price is not None) / total_weight
                
                # 가격 신뢰도 계산
                price_confidence = sum(output.confidence * weight for output, weight in 
                                    zip(price_outputs, self.price_weights)) / total_weight
        
        # 최종 신호 결정
        final_signal_type = direction_signal if direction_signal else SignalType.HOLD
        final_confidence = direction_confidence if direction_confidence > 0 else price_confidence
        
        # 이유 문자열 구성
        if not direction_outputs and price_outputs:
            reason = f"가격 모델만 사용: 예측 가격 {predicted_price:.2f}"
        elif direction_outputs and not price_outputs:
            reason = f"방향 모델만 사용: 예측 방향 {final_signal_type}"
        else:
            reason = f"하이브리드 예측: 방향 {final_signal_type}, 가격 {predicted_price:.2f if predicted_price else 'N/A'}"
        
        # 메타데이터 구성
        metadata = {
            "ensemble_name": self.name,
            "ensemble_version": self.version,
            "direction_models_count": len(self.direction_models),
            "price_models_count": len(self.price_models),
            "direction_confidence": direction_confidence,
            "price_confidence": price_confidence,
            "prediction_time": datetime.now().isoformat()
        }
        
        # 개별 모델 결과 추가
        for i, output in enumerate(direction_outputs):
            metadata[f"direction_model_{i}_name"] = self.direction_models[i].name
            metadata[f"direction_model_{i}_signal"] = output.signal.signal
            metadata[f"direction_model_{i}_confidence"] = output.confidence
        
        for i, output in enumerate(price_outputs):
            metadata[f"price_model_{i}_name"] = self.price_models[i].name
            metadata[f"price_model_{i}_price"] = output.signal.price
            metadata[f"price_model_{i}_confidence"] = output.confidence
        
        # 최종 ModelOutput 생성
        return ModelOutput(
            signal=TradingSignal(
                signal_type=final_signal_type,
                confidence=final_confidence,
                reason=reason,
                price=predicted_price,
                metadata=metadata
            ),
            confidence=final_confidence,
            metadata=metadata
        ) 