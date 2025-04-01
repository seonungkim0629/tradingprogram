"""
모델 관련 유틸리티 함수들
"""

import os
import json
import shutil
from typing import Dict, Optional, Tuple
from utils.logging import get_logger

logger = get_logger(__name__)

def copy_optimized_models_to_saved(force_copy: bool = False) -> Dict[str, bool]:
    """
    최적화 결과 디렉토리에서 최신 모델을 찾아 models/saved 디렉토리로 복사합니다.
    
    Args:
        force_copy (bool): 기존 파일이 있어도 강제로 덮어쓸지 여부
        
    Returns:
        Dict[str, bool]: 모델 타입별 복사 성공 여부
    """
    result = {
        'rf_direction': False,
        'lstm_direction': False,
        'lstm_price': False
    }
    
    optimization_dir = "optimization_results"
    saved_models_dir = "models/saved"
    
    # saved_models_dir 디렉토리가 없으면 생성
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # 최적화 결과 디렉토리가 없으면 종료
    if not os.path.exists(optimization_dir):
        logger.warning(f"최적화 결과 디렉토리({optimization_dir})가 존재하지 않습니다.")
        return result
    
    # 최신 최적화 결과 디렉토리 찾기
    try:
        # 모든 서브 디렉토리 가져오기
        subdirs = [os.path.join(optimization_dir, d) for d in os.listdir(optimization_dir) 
                  if os.path.isdir(os.path.join(optimization_dir, d))]
        
        # 생성 날짜를 기준으로 최신 순으로 정렬
        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if not subdirs:
            logger.warning(f"{optimization_dir} 디렉토리에 최적화 결과가 없습니다.")
            return result
        
        latest_dir = subdirs[0]
        logger.info(f"최신 최적화 결과 디렉토리: {latest_dir}")
        
        # 모델 파일 복사 함수
        def copy_model_files(model_type: str, src_dir: str, dst_dir: str) -> bool:
            """특정 모델 파일들을 소스 디렉토리에서 대상 디렉토리로 복사"""
            if not os.path.exists(src_dir):
                logger.warning(f"소스 디렉토리({src_dir})가 존재하지 않습니다.")
                return False
            
            # 대상 디렉토리가 없으면 생성
            os.makedirs(dst_dir, exist_ok=True)
            
            # 이미 파일이 있고 force_copy가 False인 경우
            if os.path.exists(os.path.join(dst_dir, "model_info.json")) and not force_copy:
                logger.info(f"{model_type} 모델이 이미 {dst_dir}에 존재합니다. --force 옵션을 사용하여 덮어쓸 수 있습니다.")
                return False
            
            # 모든 파일 복사
            copied = False
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dst_file = os.path.join(dst_dir, filename)
                
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    copied = True
                    logger.info(f"{src_file}을(를) {dst_file}(으)로 복사했습니다.")
            
            return copied
        
        # 1. RandomForest 방향 예측 모델 복사
        rf_results_file = os.path.join(latest_dir, 'rf_direction_optimization.json')
        if os.path.exists(rf_results_file):
            src_dir = os.path.join(latest_dir, 'models', 'rf_direction')
            dst_dir = os.path.join(saved_models_dir, 'RF_Direction')
            result['rf_direction'] = copy_model_files("RandomForest", src_dir, dst_dir)
        
        # 2. GRU/LSTM 방향 예측 모델 복사
        gru_results_file = os.path.join(latest_dir, 'gru_direction_optimization.json')
        if os.path.exists(gru_results_file):
            src_dir = os.path.join(latest_dir, 'models', 'lstm_direction')
            dst_dir = os.path.join(saved_models_dir, 'LSTM_Direction')
            result['lstm_direction'] = copy_model_files("GRU", src_dir, dst_dir)
        
        # 3. GRU/LSTM 가격 예측 모델 복사
        gru_price_results_file = os.path.join(latest_dir, 'gru_price_optimization.json')
        if os.path.exists(gru_price_results_file):
            src_dir = os.path.join(latest_dir, 'models', 'lstm_price')
            dst_dir = os.path.join(saved_models_dir, 'LSTM_Price')
            result['lstm_price'] = copy_model_files("GRU Price", src_dir, dst_dir)
        
        # 복사 결과 요약
        if any(result.values()):
            logger.info("모델 복사 완료:")
            for model_type, success in result.items():
                logger.info(f"  - {model_type}: {'성공' if success else '실패 또는 필요 없음'}")
        else:
            logger.warning("복사된 모델이 없습니다.")
        
        return result
        
    except Exception as e:
        logger.error(f"모델 복사 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return result


def find_best_model_parameters(model_type: str) -> Optional[Dict]:
    """
    최적화 결과 디렉토리에서 특정 모델 타입의 최적 파라미터를 찾습니다.
    
    Args:
        model_type (str): 모델 타입 ('rf_direction', 'lstm_direction', 'lstm_price')
        
    Returns:
        Optional[Dict]: 최적 파라미터 딕셔너리 또는 None
    """
    optimization_dir = "optimization_results"
    
    # 최적화 결과 파일 이름 매핑
    file_mapping = {
        'rf_direction': 'rf_direction_optimization.json',
        'lstm_direction': 'gru_direction_optimization.json',
        'lstm_price': 'gru_price_optimization.json'
    }
    
    if model_type not in file_mapping:
        logger.error(f"지원되지 않는 모델 타입: {model_type}")
        return None
    
    # 최적화 결과 디렉토리가 없으면 종료
    if not os.path.exists(optimization_dir):
        logger.warning(f"최적화 결과 디렉토리({optimization_dir})가 존재하지 않습니다.")
        return None
    
    try:
        # 모든 서브 디렉토리 가져오기 (최신 순)
        subdirs = [os.path.join(optimization_dir, d) for d in os.listdir(optimization_dir) 
                  if os.path.isdir(os.path.join(optimization_dir, d))]
        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        for dir_path in subdirs:
            results_file = os.path.join(dir_path, file_mapping[model_type])
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                if 'best_params' in results:
                    logger.info(f"{model_type}의 최적 파라미터를 {results_file}에서 찾았습니다.")
                    return results['best_params']
        
        logger.warning(f"{model_type}의 최적화 결과를 찾지 못했습니다.")
        return None
        
    except Exception as e:
        logger.error(f"최적 파라미터 검색 중 오류 발생: {str(e)}")
        return None 