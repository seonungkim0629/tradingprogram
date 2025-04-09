"""
예측 결합 모듈 (Prediction Combiners Module)

이 모듈은 다양한 모델의 예측을 결합하는 방법을 제공합니다.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable


def weighted_average_combiner(predictions: List[np.ndarray], 
                           weights: List[float],
                           classes: int = 2) -> np.ndarray:
    """
    가중 평균 결합 함수 (예측 확률용)
    
    Args:
        predictions (List[np.ndarray]): 각 모델의 예측 확률
        weights (List[float]): 각 모델의 가중치
        classes (int): 클래스 수 (이진 분류의 경우 2)
        
    Returns:
        np.ndarray: 결합된 예측 확률
    """
    if not predictions or len(predictions) != len(weights):
        return np.array([])
    
    # 모든 예측 형태를 동일하게 변환
    formatted_predictions = []
    for pred in predictions:
        # 예측이 확률이 아닌 경우 (0 또는 1) 처리
        if pred.ndim == 1:
            if classes == 2:
                # 1차원 배열을 2차원 확률로 변환 (이진 분류)
                pred_proba = np.zeros((pred.shape[0], 2))
                pred_proba[:, 1] = pred
                pred_proba[:, 0] = 1 - pred
                formatted_predictions.append(pred_proba)
            else:
                # One-hot 인코딩으로 변환
                pred_one_hot = np.zeros((pred.shape[0], classes))
                for i, p in enumerate(pred):
                    pred_one_hot[i, int(p)] = 1
                formatted_predictions.append(pred_one_hot)
        else:
            formatted_predictions.append(pred)
    
    # 가중 평균 계산
    weighted_sum = np.zeros_like(formatted_predictions[0])
    weight_sum = 0
    
    for i, (pred, weight) in enumerate(zip(formatted_predictions, weights)):
        weighted_sum += pred * weight
        weight_sum += weight
    
    # 가중치 합이 0이면 균등 확률 반환
    if weight_sum == 0:
        return np.ones_like(weighted_sum) / classes
    
    # 가중치 합으로 나누어 정규화
    return weighted_sum / weight_sum


def majority_vote_combiner(predictions: List[np.ndarray], 
                         weights: Optional[List[float]] = None,
                         threshold: float = 0.5) -> np.ndarray:
    """
    다수결 투표 결합 함수 (클래스 레이블용)
    
    Args:
        predictions (List[np.ndarray]): 각 모델의 예측
        weights (Optional[List[float]]): 각 모델의 가중치
        threshold (float): 분류 임계값
        
    Returns:
        np.ndarray: 결합된 예측
    """
    if not predictions:
        return np.array([])
    
    # 가중치가 없는 경우 균등 가중치 사용
    if weights is None:
        weights = [1.0] * len(predictions)
    elif len(weights) != len(predictions):
        weights = [1.0] * len(predictions)
    
    # 예측 형태를 일관되게 만들기 (모두 0/1 클래스로)
    binary_predictions = []
    for pred in predictions:
        if pred.ndim > 1:
            # 확률에서 클래스로 변환
            binary_pred = (pred[:, 1] >= threshold).astype(int)
            binary_predictions.append(binary_pred)
        else:
            binary_predictions.append(pred)
    
    # 가중 투표 계산
    stacked_preds = np.column_stack(binary_predictions)
    weighted_votes = np.zeros((len(binary_predictions[0]), 2))
    
    for i, (pred, weight) in enumerate(zip(binary_predictions, weights)):
        for j, p in enumerate(pred):
            weighted_votes[j, int(p)] += weight
    
    # 최대 투표 클래스 선택
    return np.argmax(weighted_votes, axis=1)


def stacked_probabilities_combiner(probabilities: List[np.ndarray], 
                                meta_model: Callable) -> np.ndarray:
    """
    스태킹 결합 함수 (메타모델 사용)
    
    Args:
        probabilities (List[np.ndarray]): 각 모델의 예측 확률
        meta_model (Callable): 메타 모델 (스태킹에 사용될 모델)
        
    Returns:
        np.ndarray: 결합된 예측 확률
    """
    if not probabilities:
        return np.array([])
    
    # 메타 특성 생성 (모든 모델의 확률 결합)
    num_samples = probabilities[0].shape[0]
    
    # 이진 분류 경우 처리
    if probabilities[0].shape[1] == 2:
        # 각 모델의 양성 클래스 확률만 사용
        meta_features = np.column_stack([p[:, 1] for p in probabilities])
    else:
        # 다중 분류는 모든 확률 사용
        meta_features = np.hstack(probabilities)
    
    # 메타 모델 예측
    meta_predictions = meta_model(meta_features)
    return meta_predictions


def adaptive_weights_combiner(predictions: List[np.ndarray], 
                           weights: List[float],
                           recent_performance: Optional[List[float]] = None) -> np.ndarray:
    """
    적응형 가중치 결합 함수 (최근 성능 고려)
    
    Args:
        predictions (List[np.ndarray]): 각 모델의 예측 확률
        weights (List[float]): 각 모델의 기본 가중치
        recent_performance (Optional[List[float]]): 각 모델의 최근 성능 점수
        
    Returns:
        np.ndarray: 결합된 예측 확률
    """
    if not predictions:
        return np.array([])
    
    if len(predictions) == 1:
        return predictions[0]
    
    # 모델 수, 샘플 수, 클래스 수 확인
    num_models = len(predictions)
    n_samples = predictions[0].shape[0]
    n_classes = predictions[0].shape[1] if predictions[0].ndim > 1 else 2
    
    # 적응형 가중치 계산 (최근 성능 정보가 있는 경우)
    adaptive_weights = weights.copy()
    
    if recent_performance and len(recent_performance) == num_models:
        # 성능 점수 정규화
        total_perf = sum(max(0.01, perf) for perf in recent_performance)
        
        if total_perf > 0:
            # 성능 기반 가중치와 기본 가중치 결합 (70% 기본, 30% 성능)
            for i in range(num_models):
                perf_weight = max(0.01, recent_performance[i]) / total_perf
                adaptive_weights[i] = 0.7 * weights[i] + 0.3 * perf_weight
    
    # 정규화
    total_weight = sum(adaptive_weights)
    if total_weight > 0:
        adaptive_weights = [w / total_weight for w in adaptive_weights]
    else:
        # 가중치 합이 0이면 균등 분배
        adaptive_weights = [1.0 / num_models] * num_models
    
    # 예측 결합
    combined = np.zeros((n_samples, n_classes))
    
    for i, (pred, weight) in enumerate(zip(predictions, adaptive_weights)):
        if pred.ndim == 1:
            # 1차원 예측을 2차원으로 변환
            pred_2d = np.zeros((n_samples, n_classes))
            for j, p in enumerate(pred):
                pred_2d[j, int(p)] = 1.0
            combined += pred_2d * weight
        else:
            combined += pred * weight
    
    return combined


def confidence_based_combiner(predictions: List[np.ndarray],
                           confidences: List[np.ndarray],
                           min_confidence: float = 0.6) -> np.ndarray:
    """
    신뢰도 기반 결합 함수 (신뢰도가 높은 모델 우선)
    
    Args:
        predictions (List[np.ndarray]): 각 모델의 예측
        confidences (List[np.ndarray]): 각 모델의 예측 신뢰도
        min_confidence (float): 최소 신뢰도 임계값
        
    Returns:
        np.ndarray: 결합된 예측
    """
    if not predictions or len(predictions) != len(confidences):
        return np.array([])
    
    num_samples = len(predictions[0])
    num_models = len(predictions)
    combined = np.zeros(num_samples)
    
    # 각 샘플별로 처리
    for i in range(num_samples):
        sample_predictions = [predictions[m][i] for m in range(num_models)]
        sample_confidences = [confidences[m][i] for m in range(num_models)]
        
        # 신뢰도가 임계값 이상인 모델만 선택
        confident_models = [(pred, conf) for pred, conf in zip(sample_predictions, sample_confidences) 
                          if conf >= min_confidence]
        
        if confident_models:
            # 신뢰도가 높은 모델 중 가장 신뢰도가 높은 예측 선택
            max_conf_idx = np.argmax([conf for _, conf in confident_models])
            combined[i] = confident_models[max_conf_idx][0]
        else:
            # 신뢰도가 높은 모델이 없으면 가장 신뢰도가 높은 모델 선택
            max_conf_idx = np.argmax(sample_confidences)
            combined[i] = sample_predictions[max_conf_idx]
    
    return combined


def dynamic_selection_combiner(predictions: List[np.ndarray],
                            performance_history: List[List[float]],
                            window_size: int = 10) -> np.ndarray:
    """
    동적 모델 선택 결합 함수 (최근 성능이 가장 좋은 모델만 선택)
    
    Args:
        predictions (List[np.ndarray]): 각 모델의 예측
        performance_history (List[List[float]]): 각 모델의 과거 성능 기록
        window_size (int): 고려할 최근 성능 윈도우 크기
        
    Returns:
        np.ndarray: 결합된 예측
    """
    if not predictions or len(predictions) != len(performance_history):
        return np.array([])
    
    # 각 모델의 최근 성능 평균
    recent_performance = [np.mean(history[-window_size:]) if len(history) >= window_size else 0.5
                         for history in performance_history]
    
    # 최고 성능 모델 선택
    best_model_idx = np.argmax(recent_performance)
    
    # 최고 성능 모델의 예측 반환
    return predictions[best_model_idx] 