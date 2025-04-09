"""
전략 상태 관리 유틸리티

이 모듈은 전략의 상태를 저장하고 로드하는 기능을 제공합니다.
특히 MixedTimeFrameStrategy의 trade_count와 같은 중요 상태를 세션 간에 유지합니다.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

from utils.logging import get_logger

# 로거 초기화
logger = get_logger(__name__)

# 상태 파일 디렉토리
STATE_DIR = os.path.join("data", "state")
os.makedirs(STATE_DIR, exist_ok=True)

def save_strategy_state(strategy_name: str, state: Dict[str, Any]) -> bool:
    """
    전략 상태를 파일로 저장
    
    Args:
        strategy_name (str): 전략 이름
        state (Dict[str, Any]): 저장할 상태 정보 (trade_count 등)
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 상태 파일 경로
        state_file = os.path.join(STATE_DIR, f"{strategy_name}_state.json")
        
        # 타임스탬프 추가
        state["last_updated"] = datetime.now().isoformat()
        
        # 파일에 상태 저장
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            
        logger.info(f"전략 상태 저장 완료: {strategy_name}, trade_count={state.get('trade_count', 'N/A')}")
        return True
    
    except Exception as e:
        logger.error(f"전략 상태 저장 중 오류 발생: {str(e)}")
        return False

def load_strategy_state(strategy_name: str) -> Dict[str, Any]:
    """
    저장된 전략 상태 로드
    
    Args:
        strategy_name (str): 전략 이름
        
    Returns:
        Dict[str, Any]: 로드된 상태 정보 또는 빈 딕셔너리
    """
    try:
        # 상태 파일 경로
        state_file = os.path.join(STATE_DIR, f"{strategy_name}_state.json")
        
        # 파일이 존재하면 로드
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                
            logger.info(f"전략 상태 로드 완료: {strategy_name}, trade_count={state.get('trade_count', 'N/A')}")
            return state
        else:
            logger.warning(f"전략 상태 파일이 존재하지 않음: {strategy_name}")
            return {}
    
    except Exception as e:
        logger.error(f"전략 상태 로드 중 오류 발생: {str(e)}")
        return {}

def save_trade_count(strategy_name: str, trade_count: int) -> bool:
    """
    trade_count를 저장하는 편의 함수
    
    Args:
        strategy_name (str): 전략 이름
        trade_count (int): 저장할 거래 카운트
        
    Returns:
        bool: 성공 여부
    """
    state = {"trade_count": trade_count}
    return save_strategy_state(strategy_name, state)

def load_trade_count(strategy_name: str) -> int:
    """
    저장된 trade_count를 로드하는 편의 함수
    
    Args:
        strategy_name (str): 전략 이름
        
    Returns:
        int: 로드된 trade_count 또는 0
    """
    state = load_strategy_state(strategy_name)
    return state.get("trade_count", 0)

def get_all_strategies_state() -> Dict[str, Dict[str, Any]]:
    """
    모든 전략의 상태 정보 로드
    
    Returns:
        Dict[str, Dict[str, Any]]: 전략 이름을 키로 하는 상태 정보 맵
    """
    all_states = {}
    
    try:
        if not os.path.exists(STATE_DIR):
            return all_states
        
        # 디렉토리 내 모든 JSON 파일 찾기
        for filename in os.listdir(STATE_DIR):
            if filename.endswith('.json'):
                strategy_name = filename.replace('_state.json', '')
                
                # 해당 전략의 상태 로드
                state = load_strategy_state(strategy_name)
                if state:
                    all_states[strategy_name] = state
        
        return all_states
    except Exception as e:
        logger.error(f"모든 전략 상태 로드 중 오류: {str(e)}")
        return all_states

def initialize_state_directory() -> None:
    """
    상태 저장 디렉토리 초기화
    """
    try:
        if not os.path.exists(STATE_DIR):
            os.makedirs(STATE_DIR)
            logger.info(f"상태 저장 디렉토리 생성: {STATE_DIR}")
        else:
            logger.info(f"상태 저장 디렉토리 확인: {STATE_DIR}")
    except Exception as e:
        logger.error(f"상태 저장 디렉토리 초기화 중 오류 발생: {str(e)}") 