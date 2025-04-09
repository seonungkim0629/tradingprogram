"""
공통 상수 정의 모듈

이 모듈은 프로젝트 전체에서 일관되게 사용할 상수들을 정의합니다.
"""

# 타임프레임 표준화 상수
class TimeFrame:
    """타임프레임 상수 클래스"""
    # 표준 타임프레임 키
    MINUTE1 = "1m"
    MINUTE5 = "5m"
    MINUTE15 = "15m"
    MINUTE30 = "30m"
    HOUR1 = "1h"
    HOUR4 = "4h"
    DAY = "1d"
    WEEK = "1w"
    
    # 매핑 딕셔너리 (다양한 형식의 키를 표준 형식으로 변환)
    MAPPING = {
        # 표준 형식
        "1m": MINUTE1,
        "5m": MINUTE5,
        "15m": MINUTE15,
        "30m": MINUTE30,
        "1h": HOUR1,
        "4h": HOUR4,
        "1d": DAY,
        "1w": WEEK,
        
        # 다른 형식들
        "minute1": MINUTE1,
        "minute5": MINUTE5,
        "minute15": MINUTE15,
        "minute30": MINUTE30,
        "hour": HOUR1,
        "hour1": HOUR1,
        "hour4": HOUR4,
        "day": DAY,
        "daily": DAY,
        "week": WEEK,
        
        # 숫자 형식
        1: MINUTE1,
        5: MINUTE5,
        15: MINUTE15,
        30: MINUTE30,
        60: HOUR1,
        240: HOUR4,
        1440: DAY,
        10080: WEEK
    }
    
    @staticmethod
    def standardize(timeframe):
        """
        다양한 형식의 타임프레임 문자열이나 값을 표준 형식으로 변환
        
        Args:
            timeframe: 변환할 타임프레임 (문자열 또는 숫자)
            
        Returns:
            str: 표준화된 타임프레임 문자열
        """
        if timeframe in TimeFrame.MAPPING:
            return TimeFrame.MAPPING[timeframe]
        
        # 문자열 처리 (소문자로 변환)
        if isinstance(timeframe, str):
            timeframe_lower = timeframe.lower()
            if timeframe_lower in TimeFrame.MAPPING:
                return TimeFrame.MAPPING[timeframe_lower]
        
        # 변환할 수 없는 경우 원래 값 반환
        return timeframe


# 신호 유형 표준화 상수
class SignalType:
    """거래 신호 유형 상수 클래스"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    
    # 매핑 딕셔너리
    MAPPING = {
        # 대소문자 구분 없이 매핑
        "buy": BUY,
        "b": BUY,
        "long": BUY,
        "1": BUY,
        1: BUY,
        
        "sell": SELL,
        "s": SELL,
        "short": SELL,
        "-1": SELL,
        -1: SELL,
        
        "hold": HOLD,
        "h": HOLD,
        "neutral": HOLD,
        "0": HOLD,
        0: HOLD
    }
    
    @staticmethod
    def standardize(signal_type):
        """
        다양한 형식의 신호 유형을 표준 형식으로 변환
        
        Args:
            signal_type: 변환할 신호 유형
            
        Returns:
            str: 표준화된 신호 유형 문자열
        """
        if signal_type in SignalType.MAPPING:
            return SignalType.MAPPING[signal_type]
        
        # 문자열 처리 (소문자로 변환)
        if isinstance(signal_type, str):
            signal_lower = signal_type.lower()
            if signal_lower in SignalType.MAPPING:
                return SignalType.MAPPING[signal_lower]
        
        # 변환할 수 없는 경우 HOLD 반환 (기본값)
        return SignalType.HOLD


# 데이터 컬럼 표준화 상수
class DataColumn:
    """데이터 컬럼 상수 클래스"""
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    TIMESTAMP = "timestamp"
    
    # 필수 OHLCV 컬럼
    OHLCV_COLUMNS = [OPEN, HIGH, LOW, CLOSE, VOLUME]


# 메타데이터 키 표준화 상수
class MetadataKey:
    """메타데이터 키 상수 클래스"""
    STRATEGY = "strategy"
    STRATEGY_TYPE = "strategy_type" 
    TIMEFRAME = "timeframe"
    CONFIDENCE = "confidence"
    MODEL_TYPE = "model_type"
    TIMESTAMP = "timestamp" 