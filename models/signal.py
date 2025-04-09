"""
표준화된 거래 신호 모듈

이 모듈은 거래 신호의 표준 형식을 정의하여 다양한 전략과 모델에서 
일관된 형식의 신호를 생성하고 처리할 수 있게 합니다.
"""

from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import json
import numpy as np

from utils.constants import SignalType, TimeFrame, MetadataKey


class TradingSignal:
    """
    표준화된 거래 신호 클래스
    
    모든 거래 전략에서 이 클래스 형식의 신호를 생성하고 처리합니다.
    """
    
    def __init__(self, 
                signal_type: str = SignalType.HOLD, 
                confidence: float = 0.5,
                reason: str = "",
                metadata: Optional[Dict[str, Any]] = None,
                position_size: float = 0.0,
                price: Optional[float] = None,
                timestamp: Optional[datetime] = None):
        """
        거래 신호 초기화
        
        Args:
            signal_type (str): 신호 유형 ('BUY', 'SELL', 'HOLD')
            confidence (float): 신호의 확신도 (0.0-1.0)
            reason (str): 신호 생성 이유
            metadata (Dict[str, Any], optional): 추가 메타데이터
            position_size (float): 포지션 크기 비율 (0.0-1.0)
            price (float, optional): 거래 가격
            timestamp (datetime, optional): 신호 생성 시간
        """
        # 신호 유형 표준화
        self.signal = SignalType.standardize(signal_type)
        self.confidence = float(confidence)
        self.reason = reason
        self.metadata = metadata or {}
        self.position_size = float(position_size)
        self.price = price
        self.timestamp = timestamp or datetime.now()
        
        # 메타데이터에 타임스탬프 추가
        if MetadataKey.TIMESTAMP not in self.metadata:
            self.metadata[MetadataKey.TIMESTAMP] = self.timestamp.isoformat()
    
    @classmethod
    def from_dict(cls, signal_dict: Dict[str, Any]) -> 'TradingSignal':
        """
        딕셔너리에서 신호 객체 생성
        
        Args:
            signal_dict (Dict[str, Any]): 신호 데이터 딕셔너리
            
        Returns:
            TradingSignal: 생성된 신호 객체
        """
        # 필수 필드 추출
        signal_type = signal_dict.get('signal', SignalType.HOLD)
        confidence = signal_dict.get('confidence', 0.5)
        reason = signal_dict.get('reason', '')
        metadata = signal_dict.get('metadata', {})
        
        # 옵션 필드 추출
        position_size = signal_dict.get('position_size', 0.0)
        price = signal_dict.get('price')
        
        # 타임스탬프 처리
        timestamp = None
        ts_str = signal_dict.get('timestamp')
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                pass
        
        # 메타데이터에서 타임스탬프 찾기
        if not timestamp and metadata and MetadataKey.TIMESTAMP in metadata:
            try:
                timestamp = datetime.fromisoformat(metadata[MetadataKey.TIMESTAMP])
            except (ValueError, TypeError):
                pass
                
        return cls(
            signal_type=signal_type,
            confidence=confidence,
            reason=reason,
            metadata=metadata,
            position_size=position_size,
            price=price,
            timestamp=timestamp
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        신호 객체를 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 신호 데이터 딕셔너리
        """
        return {
            'signal': self.signal,
            'confidence': self.confidence,
            'reason': self.reason,
            'metadata': self.metadata,
            'position_size': self.position_size,
            'price': self.price,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """
        신호 객체를 JSON 문자열로 변환
        
        Returns:
            str: JSON 형식의 신호 데이터
        """
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TradingSignal':
        """
        JSON 문자열에서 신호 객체 생성
        
        Args:
            json_str (str): JSON 형식의 신호 데이터
            
        Returns:
            TradingSignal: 생성된 신호 객체
        """
        signal_dict = json.loads(json_str)
        return cls.from_dict(signal_dict)
    
    def is_actionable(self) -> bool:
        """
        액션이 필요한 신호인지 확인 (BUY 또는 SELL)
        
        Returns:
            bool: 액션이 필요한 신호이면 True
        """
        return self.signal in (SignalType.BUY, SignalType.SELL)
    
    def __str__(self) -> str:
        """문자열 표현"""
        return (f"TradingSignal({self.signal}, confidence={self.confidence:.2f}, "
                f"reason='{self.reason}', position_size={self.position_size:.2f})")
    
    def __repr__(self) -> str:
        """개발자용 표현"""
        return self.__str__()


class ModelOutput:
    """
    모델 출력 클래스
    
    모든 예측 모델의 출력이 이 클래스 형식을 가지도록 표준화합니다.
    내부에 TradingSignal을 포함하여 일관된 인터페이스를 제공합니다.
    """
    
    def __init__(self,
                signal: TradingSignal,
                raw_predictions: Optional[np.ndarray] = None,
                confidence: float = 0.0,
                metadata: Optional[Dict[str, Any]] = None):
        """
        모델 출력 초기화
        
        Args:
            signal (TradingSignal): 거래 신호 객체
            raw_predictions (np.ndarray, optional): 모델의 원시 예측값
            confidence (float): 모델의 전체 확신도 (0.0-1.0)
            metadata (Dict[str, Any], optional): 추가 메타데이터
        """
        self.signal = signal
        self.raw_predictions = raw_predictions
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        
        # 메타데이터에 타임스탬프 추가
        if MetadataKey.TIMESTAMP not in self.metadata:
            self.metadata[MetadataKey.TIMESTAMP] = self.timestamp.isoformat()
    
    @classmethod
    def from_signal(cls, signal: TradingSignal, **kwargs) -> 'ModelOutput':
        """
        TradingSignal로부터 ModelOutput 생성
        
        Args:
            signal (TradingSignal): 거래 신호 객체
            **kwargs: 추가 인자 (raw_predictions, metadata 등)
            
        Returns:
            ModelOutput: 생성된 모델 출력 객체
        """
        return cls(
            signal=signal,
            raw_predictions=kwargs.get('raw_predictions'),
            confidence=kwargs.get('confidence', signal.confidence),
            metadata=kwargs.get('metadata', signal.metadata.copy())
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelOutput':
        """
        딕셔너리에서 ModelOutput 객체 생성
        
        Args:
            data (Dict[str, Any]): 모델 출력 데이터 딕셔너리
            
        Returns:
            ModelOutput: 생성된 모델 출력 객체
        """
        # 신호 데이터 추출
        signal_data = data.get('signal', {})
        if isinstance(signal_data, dict):
            signal = TradingSignal.from_dict(signal_data)
        else:
            signal = standardize_signal(signal_data)
        
        # raw_predictions 처리
        raw_predictions = data.get('raw_predictions')
        if isinstance(raw_predictions, list):
            raw_predictions = np.array(raw_predictions)
            
        return cls(
            signal=signal,
            raw_predictions=raw_predictions,
            confidence=data.get('confidence', 0.0),
            metadata=data.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        모델 출력 객체를 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 모델 출력 데이터 딕셔너리
        """
        result = {
            'signal': self.signal.to_dict(),
            'confidence': self.confidence,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
        
        # raw_predictions가 있으면 리스트로 변환하여 추가
        if self.raw_predictions is not None:
            result['raw_predictions'] = self.raw_predictions.tolist() if hasattr(self.raw_predictions, 'tolist') else self.raw_predictions
            
        return result
    
    def to_json(self) -> str:
        """
        모델 출력 객체를 JSON 문자열로 변환
        
        Returns:
            str: JSON 형식의 모델 출력 데이터
        """
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelOutput':
        """
        JSON 문자열에서 모델 출력 객체 생성
        
        Args:
            json_str (str): JSON 형식의 모델 출력 데이터
            
        Returns:
            ModelOutput: 생성된 모델 출력 객체
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"ModelOutput(signal={self.signal}, confidence={self.confidence:.2f})"
    
    def __repr__(self) -> str:
        """개발자용 표현"""
        return (f"ModelOutput(signal={self.signal}, "
                f"confidence={self.confidence:.2f}, "
                f"metadata={self.metadata})")


def standardize_signal(signal: Union[Dict[str, Any], TradingSignal]) -> TradingSignal:
    """
    다양한 형식의 신호를 표준 TradingSignal 객체로 변환
    
    Args:
        signal (Union[Dict[str, Any], TradingSignal]): 변환할 신호
        
    Returns:
        TradingSignal: 표준화된 신호 객체
    """
    if isinstance(signal, TradingSignal):
        return signal
    
    if isinstance(signal, dict):
        return TradingSignal.from_dict(signal)
    
    # 기본 HOLD 신호 반환
    return TradingSignal()


def standardize_model_output(output: Union[Dict[str, Any], TradingSignal, 'ModelOutput']) -> 'ModelOutput':
    """
    다양한 형식의 모델 출력을 표준 ModelOutput 객체로 변환
    
    Args:
        output (Union[Dict[str, Any], TradingSignal, ModelOutput]): 변환할 모델 출력
        
    Returns:
        ModelOutput: 표준화된 모델 출력 객체
    """
    if isinstance(output, ModelOutput):
        return output
    
    if isinstance(output, TradingSignal):
        return ModelOutput.from_signal(output)
    
    if isinstance(output, dict):
        if 'signal' in output:
            # signal 키가 있으면 ModelOutput으로 변환
            return ModelOutput.from_dict(output)
        else:
            # signal 키가 없으면 TradingSignal로 변환 후 ModelOutput 생성
            signal = standardize_signal(output)
            return ModelOutput.from_signal(signal)
    
    # 기본 HOLD 신호와 함께 ModelOutput 반환
    return ModelOutput(signal=TradingSignal()) 