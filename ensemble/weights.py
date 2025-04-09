"""
가중치 관리 모듈 (Weights Management Module)

이 모듈은 모델 가중치 정규화 및 자동조정 함수를 제공합니다.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd

def normalize_weights(weights: Union[List[float], Dict[str, float]], 
                     min_weight: float = 0.0,
                     max_weight: float = 1.0) -> Union[List[float], Dict[str, float]]:
    """
    가중치 정규화 함수 (합이 1이 되도록)
    
    Args:
        weights (Union[List[float], Dict[str, float]]): 정규화할 가중치
        min_weight (float): 최소 가중치
        max_weight (float): 최대 가중치
        
    Returns:
        Union[List[float], Dict[str, float]]: 정규화된 가중치
    """
    # 딕셔너리 또는 리스트 형태 처리
    if isinstance(weights, dict):
        # 딕셔너리 가중치
        weight_values = np.array(list(weights.values()))
        weight_keys = list(weights.keys())
    else:
        # 리스트 가중치
        weight_values = np.array(weights)
        weight_keys = None
    
    # 최소/최대 가중치 범위 적용
    weight_values = np.clip(weight_values, min_weight, max_weight)
    
    # 합계로 나누어 정규화
    total = np.sum(weight_values)
    if total > 0:
        normalized_values = weight_values / total
    else:
        # 합계가 0이면 균등 가중치
        normalized_values = np.ones_like(weight_values) / len(weight_values)
    
    # 원래 입력 형태로 반환
    if weight_keys:
        return {k: float(v) for k, v in zip(weight_keys, normalized_values)}
    else:
        return normalized_values.tolist()


def adjust_weights_by_performance(performance_metrics: Dict[str, Dict[str, float]],
                                metric_name: str = 'accuracy',
                                alpha: float = 0.3,
                                min_weight: float = 0.05,
                                max_weight: float = 0.8) -> Dict[str, float]:
    """
    성능 기반 가중치 조정 함수
    
    Args:
        performance_metrics (Dict[str, Dict[str, float]]): 각 모델의 성능 지표
        metric_name (str): 사용할 성능 지표 이름
        alpha (float): 조정 강도 (0: 조정 없음, 1: 완전히 성능 기반)
        min_weight (float): 최소 가중치
        max_weight (float): 최대 가중치
        
    Returns:
        Dict[str, float]: 조정된 가중치
    """
    # 현재 가중치 및 성능 점수 추출
    current_weights = {}
    performance_scores = {}
    
    for model_name, metrics in performance_metrics.items():
        # 각 모델의 가중치와 성능 점수 가져오기
        current_weights[model_name] = metrics.get('weight', 1.0 / len(performance_metrics))
        
        # 지표 추출 (낮을수록 좋은 지표는 역수로 변환)
        score = metrics.get(metric_name, 0.5)
        if metric_name in ['mse', 'rmse', 'mae', 'mape', 'max_drawdown']:
            # 역수 변환 (더 낮은 값이 더 높은 가중치)
            score = 1.0 / max(score, 1e-10)
        
        performance_scores[model_name] = score
    
    # 성능 점수 기반 새 가중치 계산
    new_weights = normalize_weights(performance_scores, min_weight, max_weight)
    
    # 점진적 가중치 조정 (급격한 변화 방지)
    adjusted_weights = {}
    for model_name in current_weights:
        old_weight = current_weights[model_name]
        target_weight = new_weights[model_name] if model_name in new_weights else min_weight
        
        # 가중치 점진적 조정
        adjusted_weights[model_name] = (1 - alpha) * old_weight + alpha * target_weight
    
    # 최종 정규화
    return normalize_weights(adjusted_weights, min_weight, max_weight)


def hierarchical_normalize_weights(weights: Dict[str, float],
                                 model_groups: Dict[str, List[str]],
                                 group_weights: Optional[Dict[str, float]] = None,
                                 min_weight: float = 0.05,
                                 max_weight: float = 0.8) -> Dict[str, float]:
    """
    계층적 가중치 정규화 함수
    
    Args:
        weights (Dict[str, float]): 정규화할 모델 가중치
        model_groups (Dict[str, List[str]]): 모델 그룹 정의
        group_weights (Optional[Dict[str, float]]): 그룹별 가중치
        min_weight (float): 최소 가중치
        max_weight (float): 최대 가중치
        
    Returns:
        Dict[str, float]: 계층적 정규화된 가중치
    """
    normalized_weights = weights.copy()
    
    # 1. 그룹 내부 정규화
    for group_name, group_models in model_groups.items():
        # 그룹에 속한 모델 필터링
        group_model_weights = {m: normalized_weights[m] for m in group_models if m in normalized_weights}
        
        # 그룹 내 가중치 정규화
        if group_model_weights:
            normalized_group_weights = normalize_weights(group_model_weights)
            normalized_weights.update(normalized_group_weights)
    
    # 2. 그룹 간 가중치 적용
    if group_weights:
        # 각 그룹의 총 가중치 계산
        group_total_weights = {}
        
        for group_name, models in model_groups.items():
            models_in_group = [m for m in models if m in normalized_weights]
            if models_in_group:
                group_weight = group_weights.get(group_name, 1.0 / len(model_groups))
                
                # 그룹 내 각 모델에 그룹 가중치 적용
                for model in models_in_group:
                    normalized_weights[model] *= group_weight
    
    # 3. 최소/최대 가중치 제한 적용
    for model, weight in normalized_weights.items():
        normalized_weights[model] = max(min_weight, min(max_weight, weight))
    
    # 4. 최종 정규화
    return normalize_weights(normalized_weights)


def decay_weights_over_time(weights: Dict[str, float],
                          decay_factor: float = 0.95,
                          min_weight: float = 0.05) -> Dict[str, float]:
    """
    시간에 따른 가중치 감소 함수
    
    Args:
        weights (Dict[str, float]): 현재 가중치
        decay_factor (float): 감소 계수 (0-1)
        min_weight (float): 최소 가중치
        
    Returns:
        Dict[str, float]: 감소된 가중치
    """
    # 모든 가중치에 감소 계수 적용
    decayed_weights = {model: max(min_weight, weight * decay_factor) 
                      for model, weight in weights.items()}
    
    # 정규화하여 반환
    return normalize_weights(decayed_weights)


def adjust_weights_by_profit(weights: Dict[str, float],
                           profits: Dict[str, float],
                           alpha: float = 0.2,
                           min_weight: float = 0.05) -> Dict[str, float]:
    """
    수익률 기반 가중치 조정 함수
    
    Args:
        weights (Dict[str, float]): 현재 가중치
        profits (Dict[str, float]): 각 모델의 수익률
        alpha (float): 조정 강도 (0-1)
        min_weight (float): 최소 가중치
        
    Returns:
        Dict[str, float]: 조정된 가중치
    """
    # 수익률이 없는 경우 원래 가중치 반환
    if not profits:
        return weights
    
    # 수익률 기반 새 가중치 계산
    profit_weights = {}
    for model, profit in profits.items():
        if model in weights:
            # 수익률에 따른 가중치 상향/하향 조정
            if profit > 0:
                # 양수 수익은 가중치 증가
                profit_weights[model] = weights[model] * (1 + profit * alpha)
            else:
                # 음수 수익은 가중치 감소
                profit_weights[model] = weights[model] * (1 + profit * alpha)
    
    # 빠진 모델이 있으면 원래 가중치 사용
    for model in weights:
        if model not in profit_weights:
            profit_weights[model] = weights[model]
    
    # 정규화하여 반환 (최소 가중치 적용)
    return normalize_weights(profit_weights, min_weight)


def adjust_weights_based_on_performance(
        weights: List[float], 
        performance_scores: List[float], 
        alpha: float = 0.2,
        min_weight: float = 0.05
    ) -> List[float]:
    """
    모델 성능 점수에 기반하여 앙상블 가중치를 조정합니다.
    
    Args:
        weights (List[float]): 현재 모델 가중치 리스트
        performance_scores (List[float]): 모델별 성능 점수 리스트
        alpha (float): 조정 강도 (0-1)
        min_weight (float): 최소 가중치 값
        
    Returns:
        List[float]: 조정된 가중치 리스트
    """
    if not performance_scores or not weights or len(performance_scores) != len(weights):
        return weights
    
    # 성능 점수가 음수인 경우, 모두 양수로 변환 (최소값을 0으로)
    min_score = min(performance_scores)
    if min_score < 0:
        performance_scores = [score - min_score for score in performance_scores]
    
    # 모든 성능 점수가 0이면 원래 가중치 유지
    if sum(performance_scores) == 0:
        return weights
    
    # 성능 점수를 정규화
    total_score = sum(performance_scores)
    normalized_scores = [score / total_score for score in performance_scores]
    
    # 새 가중치 계산 (기존 가중치와 성능 점수의 가중 평균)
    new_weights = []
    for i in range(len(weights)):
        new_weight = weights[i] * (1 - alpha) + normalized_scores[i] * alpha
        new_weights.append(new_weight)
    
    # 최소 가중치 적용
    for i in range(len(new_weights)):
        new_weights[i] = max(min_weight, new_weights[i])
    
    # 정규화
    total = sum(new_weights)
    normalized_weights = [w / total for w in new_weights]
    
    return normalized_weights


def calculate_optimal_weights(
        model_predictions: List[np.ndarray], 
        true_values: np.ndarray,
        method: str = 'accuracy',
        min_weight: float = 0.05
    ) -> List[float]:
    """
    과거 예측 결과와 실제 값을 비교하여 최적의 앙상블 가중치를 계산합니다.
    
    Args:
        model_predictions (List[np.ndarray]): 각 모델의 예측 결과 리스트
        true_values (np.ndarray): 실제 값
        method (str): 최적화 방법 ('accuracy', 'f1', 'log_loss' 등)
        min_weight (float): 최소 가중치
        
    Returns:
        List[float]: 최적화된 가중치 리스트
    """
    if not model_predictions or len(model_predictions) == 0:
        return []
    
    num_models = len(model_predictions)
    
    # 단일 모델인 경우
    if num_models == 1:
        return [1.0]
    
    # 기본 가중치 (균등 배분)
    weights = [1.0 / num_models] * num_models
    
    # 각 모델의 정확도 계산
    accuracies = []
    for predictions in model_predictions:
        correct = np.sum(predictions == true_values)
        accuracy = correct / len(true_values) if len(true_values) > 0 else 0
        accuracies.append(accuracy)
    
    # 정확도 기반 가중치 계산
    total_accuracy = sum(accuracies)
    if total_accuracy > 0:
        weights = [acc / total_accuracy for acc in accuracies]
    
    # 최소 가중치 적용
    weights = [max(w, min_weight) for w in weights]
    
    # 정규화
    total = sum(weights)
    weights = [w / total for w in weights]
    
    return weights 