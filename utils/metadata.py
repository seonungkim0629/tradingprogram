"""
메타데이터 표준화 유틸리티

이 모듈은 전략, 거래, 시그널에 대한 메타데이터를 표준화하고 검증하는 기능을 제공합니다.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Union, Tuple

from utils.logging import get_logger

# 로거 초기화
logger = get_logger(__name__)

# 메타데이터 스키마 정의
TRADE_METADATA_SCHEMA = {
    'required': [
        'timestamp', 
        'strategy', 
        'strategy_type',
        'trade_count'
    ],
    'optional': [
        'timeframe',
        'commission',
        'balance_after',
        'profit_loss',
        'profit_loss_pct',
        'execution_price',
        'position_size',
        'original_reason',
        'original_confidence'
    ]
}

SIGNAL_METADATA_SCHEMA = {
    'required': [
        'timestamp', 
        'strategy', 
        'strategy_type',
        'timeframe'
    ],
    'optional': [
        'confidence',
        'trade_count',
        'gpt_analysis',
        'volatility',
        'market_condition',
        'hourly_signal',
        'daily_signal',
        'source'
    ]
}

PERFORMANCE_METADATA_SCHEMA = {
    'required': [
        'timestamp', 
        'strategy', 
        'period'
    ],
    'optional': [
        'metric_name',
        'metric_value',
        'start_date',
        'end_date',
        'trade_count',
        'win_count',
        'loss_count',
        'win_rate'
    ]
}

def create_standard_metadata(metadata_type: str, base_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    표준화된 메타데이터 생성
    
    Args:
        metadata_type (str): 메타데이터 유형 ('trade', 'signal', 'performance')
        base_data (Dict[str, Any]): 기본 데이터
        
    Returns:
        Dict[str, Any]: 표준화된 메타데이터
    """
    # 타입별 스키마 선택
    if metadata_type == 'trade':
        schema = TRADE_METADATA_SCHEMA
    elif metadata_type == 'signal':
        schema = SIGNAL_METADATA_SCHEMA
    elif metadata_type == 'performance':
        schema = PERFORMANCE_METADATA_SCHEMA
    else:
        logger.warning(f"Unknown metadata type: {metadata_type}")
        schema = {'required': [], 'optional': []}
    
    # 타임스탬프 추가 (없는 경우)
    if 'timestamp' not in base_data:
        base_data['timestamp'] = datetime.now().isoformat()
    
    # 필수 필드 검증
    for field in schema['required']:
        if field not in base_data:
            logger.warning(f"Missing required field '{field}' in {metadata_type} metadata")
    
    return base_data

def validate_metadata(metadata: Dict[str, Any], metadata_type: str) -> Tuple[bool, List[str]]:
    """
    메타데이터 유효성 검증
    
    Args:
        metadata (Dict[str, Any]): 검증할 메타데이터
        metadata_type (str): 메타데이터 유형 ('trade', 'signal', 'performance')
        
    Returns:
        Tuple[bool, List[str]]: (유효성 여부, 누락된 필드 목록)
    """
    # 타입별 스키마 선택
    if metadata_type == 'trade':
        schema = TRADE_METADATA_SCHEMA
    elif metadata_type == 'signal':
        schema = SIGNAL_METADATA_SCHEMA
    elif metadata_type == 'performance':
        schema = PERFORMANCE_METADATA_SCHEMA
    else:
        logger.warning(f"Unknown metadata type: {metadata_type}")
        return False, [f"Unknown metadata type: {metadata_type}"]
    
    # 필수 필드 검증
    missing_fields = []
    for field in schema['required']:
        if field not in metadata:
            missing_fields.append(field)
    
    # 모든 필수 필드가 있으면 유효함
    is_valid = len(missing_fields) == 0
    
    return is_valid, missing_fields

def normalize_metadata(metadata: Dict[str, Any], metadata_type: str) -> Dict[str, Any]:
    """
    메타데이터를 스키마에 맞게 정규화
    
    Args:
        metadata (Dict[str, Any]): 원본 메타데이터
        metadata_type (str): 메타데이터 유형 ('trade', 'signal', 'performance')
        
    Returns:
        Dict[str, Any]: 정규화된 메타데이터
    """
    # 타입별 스키마 선택
    if metadata_type == 'trade':
        schema = TRADE_METADATA_SCHEMA
    elif metadata_type == 'signal':
        schema = SIGNAL_METADATA_SCHEMA
    elif metadata_type == 'performance':
        schema = PERFORMANCE_METADATA_SCHEMA
    else:
        logger.warning(f"Unknown metadata type: {metadata_type}")
        return metadata
    
    # 허용된 필드만 포함한 새 메타데이터 생성
    allowed_fields = set(schema['required'] + schema['optional'])
    normalized = {}
    
    for key, value in metadata.items():
        if key in allowed_fields:
            normalized[key] = value
        else:
            logger.debug(f"Removing non-standard field '{key}' from {metadata_type} metadata")
    
    # 필수 필드 추가 (없는 경우)
    for field in schema['required']:
        if field not in normalized:
            if field == 'timestamp':
                normalized[field] = datetime.now().isoformat()
            else:
                normalized[field] = None
                logger.warning(f"Added missing required field '{field}' with null value")
    
    return normalized

def metadata_to_json(metadata: Dict[str, Any]) -> str:
    """
    메타데이터를 JSON 문자열로 변환
    
    Args:
        metadata (Dict[str, Any]): 변환할 메타데이터
        
    Returns:
        str: JSON 문자열
    """
    try:
        return json.dumps(metadata, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error converting metadata to JSON: {str(e)}")
        # 단순화된 버전 시도
        try:
            # 직렬화 불가능한 객체 제거
            simplified = {}
            for k, v in metadata.items():
                try:
                    json.dumps({k: v})
                    simplified[k] = v
                except:
                    simplified[k] = str(v)
            return json.dumps(simplified, ensure_ascii=False)
        except:
            return "{}"

def json_to_metadata(json_str: str) -> Dict[str, Any]:
    """
    JSON 문자열을 메타데이터로 변환
    
    Args:
        json_str (str): 변환할 JSON 문자열
        
    Returns:
        Dict[str, Any]: 변환된 메타데이터
    """
    try:
        if not json_str:
            return {}
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error parsing JSON metadata: {str(e)}")
        return {} 