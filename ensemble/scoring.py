"""
성능 평가 모듈 (Scoring Module)

이 모듈은 다양한 모델 유형(분류/회귀)에 대한 성능 평가 함수를 제공합니다.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def evaluate_classification(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         probas: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    분류 모델 성능 평가 함수
    
    Args:
        y_true (np.ndarray): 실제 레이블
        y_pred (np.ndarray): 예측 레이블
        probas (Optional[np.ndarray]): 예측 확률 (신뢰도 계산용)
        
    Returns:
        Dict[str, float]: 성능 지표
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # 신뢰도 계산 (예측 확률이 제공된 경우)
    if probas is not None:
        # 각 예측의 최대 확률 추출
        confidence = np.max(probas, axis=1) if probas.ndim > 1 else probas
        metrics['confidence_mean'] = float(np.mean(confidence))
        metrics['confidence_std'] = float(np.std(confidence))
        
        # 정확한 예측과 부정확한 예측에 대한 평균 신뢰도
        if len(y_true) > 0:
            correct_mask = (y_pred == y_true)
            if np.any(correct_mask):
                metrics['confidence_correct'] = float(np.mean(confidence[correct_mask]))
            if np.any(~correct_mask):
                metrics['confidence_incorrect'] = float(np.mean(confidence[~correct_mask]))
    
    return metrics


def evaluate_regression(y_true: np.ndarray, 
                      y_pred: np.ndarray) -> Dict[str, float]:
    """
    회귀 모델 성능 평가 함수
    
    Args:
        y_true (np.ndarray): 실제 값
        y_pred (np.ndarray): 예측 값
        
    Returns:
        Dict[str, float]: 성능 지표
    """
    # MSE 및 RMSE
    mse = mean_squared_error(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    # 추가 지표: 평균 절대 백분율 오차(MAPE)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
        if not np.isnan(mape) and not np.isinf(mape):
            metrics['mape'] = mape
    
    return metrics


def evaluate_trading_performance(predictions: np.ndarray, 
                              y_true: np.ndarray,
                              returns: np.ndarray) -> Dict[str, float]:
    """
    트레이딩 성능 평가 함수
    
    Args:
        predictions (np.ndarray): 모델 예측 (1: 상승, 0: 하락, 2: 중립)
        y_true (np.ndarray): 실제 방향 (1: 상승, 0: 하락)
        returns (np.ndarray): 각 기간의 수익률
        
    Returns:
        Dict[str, float]: 트레이딩 성능 지표
    """
    if len(predictions) != len(y_true) or len(predictions) != len(returns):
        raise ValueError("predictions, y_true, returns 길이가 일치해야 합니다")
    
    # 기본 정확도 지표
    accuracy = accuracy_score(y_true[predictions != 2], predictions[predictions != 2])
    
    # 트레이딩 성능 지표
    metrics = {
        'accuracy': accuracy,
        'signal_ratio': np.mean(predictions != 2),  # 중립이 아닌 시그널 비율
    }
    
    # 상승/하락 예측 정확도
    if np.any(predictions == 1):
        metrics['accuracy_up'] = np.mean(y_true[predictions == 1] == 1)
    if np.any(predictions == 0):
        metrics['accuracy_down'] = np.mean(y_true[predictions == 0] == 0)
    
    # 수익성 지표
    traded_returns = returns.copy()
    traded_returns[predictions == 2] = 0  # 중립 시그널은 거래하지 않음
    traded_returns[predictions == 0] = -traded_returns[predictions == 0]  # 하락 예측은 수익 반전
    
    if len(traded_returns) > 0:
        metrics['mean_return'] = np.mean(traded_returns)
        metrics['total_return'] = np.sum(traded_returns)
        metrics['sharpe_ratio'] = np.mean(traded_returns) / (np.std(traded_returns) if np.std(traded_returns) > 0 else 1e-10)
        
        # 낙폭 분석
        cum_returns = np.cumprod(1 + traded_returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / peak
        metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0
    
    return metrics


def calculate_ensemble_weights(model_metrics: List[Dict[str, float]], 
                            metric_name: str = 'accuracy',
                            min_weight: float = 0.05) -> List[float]:
    """
    모델 성능 지표를 기반으로 앙상블 가중치 계산
    
    Args:
        model_metrics (List[Dict[str, float]]): 각 모델의 성능 지표
        metric_name (str): 사용할 성능 지표 이름
        min_weight (float): 최소 가중치
        
    Returns:
        List[float]: 계산된 가중치 리스트
    """
    # 성능 지표 추출
    metrics = [m.get(metric_name, 0) for m in model_metrics]
    
    # 성능 지표가 높을수록 가중치가 높도록 계산
    weights = np.array(metrics)
    
    # 음수 지표 처리 (낮을수록 좋은 지표인 경우)
    if metric_name in ['mse', 'rmse', 'mae', 'mape', 'max_drawdown']:
        # 역수 계산 (더 낮은 값이 더 높은 가중치)
        weights = 1.0 / np.maximum(weights, 1e-10)
    
    # 최소 가중치 적용
    weights = np.maximum(weights, min_weight)
    
    # 가중치 정규화
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        weights = np.ones_like(weights) / len(weights)
    
    return weights.tolist()


def calculate_model_performance(y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               y_proba: Optional[np.ndarray] = None,
                               task_type: str = 'classification') -> Dict[str, float]:
    """
    모델 성능 지표 계산
    
    Args:
        y_true (np.ndarray): 실제 레이블
        y_pred (np.ndarray): 예측 레이블
        y_proba (Optional[np.ndarray]): 예측 확률 (분류 모델용)
        task_type (str): 작업 유형 ('classification' 또는 'regression')
        
    Returns:
        Dict[str, float]: 성능 지표
    """
    metrics = {}
    
    if task_type == 'classification':
        # 분류 모델 지표
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        try:
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        except Exception:
            # 다중 분류 등에서 오류 발생 시 기본값
            metrics['f1'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
    
    elif task_type == 'regression':
        # 회귀 모델 지표
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        try:
            metrics['r2'] = r2_score(y_true, y_pred)
        except Exception:
            metrics['r2'] = 0.0
    
    return metrics


def calculate_performance_score(
        predictions: np.ndarray, 
        true_values: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        metric: str = 'accuracy'
    ) -> float:
    """
    특정 성능 지표에 따라 모델 성능을 계산합니다.
    
    Args:
        predictions (np.ndarray): 모델 예측값
        true_values (np.ndarray): 실제 값
        probabilities (Optional[np.ndarray]): 예측 확률 (log_loss 계산 시 필요)
        metric (str): 성능 지표 ('accuracy', 'precision', 'recall', 'f1', 'log_loss')
        
    Returns:
        float: 계산된 성능 점수
    """
    if len(predictions) == 0 or len(true_values) == 0:
        return 0.0
    
    try:
        if metric == 'accuracy':
            return accuracy_score(true_values, predictions)
        elif metric == 'precision':
            return precision_score(true_values, predictions, average='weighted', zero_division=0)
        elif metric == 'recall':
            return recall_score(true_values, predictions, average='weighted', zero_division=0)
        elif metric == 'f1':
            return f1_score(true_values, predictions, average='weighted', zero_division=0)
        elif metric == 'log_loss' and probabilities is not None:
            if probabilities.shape[1] == 1:  # 이진 분류
                return -log_loss(true_values, probabilities)
            else:  # 다중 분류
                return -log_loss(true_values, probabilities)
        else:
            # 기본값은 정확도
            return accuracy_score(true_values, predictions)
    except Exception as e:
        print(f"성능 계산 오류: {e}")
        return 0.0


def evaluate_ensemble_performance(ensemble, 
                                X_test: np.ndarray, 
                                y_test: np.ndarray,
                                market_data: Optional[pd.DataFrame] = None,
                                task_type: str = 'classification') -> Dict[str, Any]:
    """
    앙상블 모델의 성능 평가
    
    Args:
        ensemble: 앙상블 모델
        X_test (np.ndarray): 테스트 특성
        y_test (np.ndarray): 테스트 레이블
        market_data (Optional[pd.DataFrame]): 시장 데이터
        task_type (str): 작업 유형 ('classification' 또는 'regression')
        
    Returns:
        Dict[str, Any]: 성능 지표 및 개별 모델 성능
    """
    results = {
        'ensemble': {},
        'models': {}
    }
    
    # 앙상블 예측
    try:
        ensemble_result = ensemble.predict(X_test, market_data=market_data)
        
        if task_type == 'classification':
            if 'direction' in ensemble_result:
                ensemble_pred = ensemble_result['direction']
                ensemble_metrics = calculate_model_performance(
                    ensemble_pred, y_test, task_type='classification'
                )
                results['ensemble'] = ensemble_metrics
        
        elif task_type == 'regression':
            if 'price' in ensemble_result:
                ensemble_pred = ensemble_result['price']
                ensemble_metrics = calculate_model_performance(
                    ensemble_pred, y_test, task_type='regression'
                )
                results['ensemble'] = ensemble_metrics
    
    except Exception as e:
        results['error'] = str(e)
    
    # 개별 모델 평가 (방향 예측 모델)
    if hasattr(ensemble, 'direction_models'):
        for i, model in enumerate(ensemble.direction_models):
            try:
                model_pred = model.predict(X_test)
                model_metrics = calculate_model_performance(
                    model_pred, y_test, task_type='classification'
                )
                results['models'][f'direction_{i}_{model.name}'] = model_metrics
            except Exception as e:
                results['models'][f'direction_{i}_{model.name}'] = {'error': str(e)}
    
    # 개별 모델 평가 (가격 예측 모델)
    if hasattr(ensemble, 'price_models') and task_type == 'regression':
        for i, model in enumerate(ensemble.price_models):
            try:
                model_pred = model.predict(X_test)
                model_metrics = calculate_model_performance(
                    model_pred, y_test, task_type='regression'
                )
                results['models'][f'price_{i}_{model.name}'] = model_metrics
            except Exception as e:
                results['models'][f'price_{i}_{model.name}'] = {'error': str(e)}
    
    return results 