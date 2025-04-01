"""
Harmonizing Strategy Module

This module implements a strategy that combines multiple signals from various sources,
including trend following, moving averages, RSI, and machine learning models.
"""

import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import random
import logging

from models.base import ModelBase
from models.random_forest import RandomForestDirectionModel
from models.gru import GRUDirectionModel

from strategies.base import BaseStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi_strategy import RSIStrategy
from data.indicators import calculate_all_indicators
from utils.logging import get_logger, log_execution
from config import settings

# Initialize logger
logger = get_logger(__name__)
logger.setLevel(logging.INFO)

# 파일 핸들러 추가
if not logger.handlers:
    fh = logging.FileHandler('logs/strategies.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# 간단한 앙상블 클래스 정의
class SimpleEnsemble:
    """
    향상된 앙상블 모델 구현
    
    Features:
    - 다양한 유형의 모델 통합 (분류/회귀)
    - 성능 기반 자동 가중치 조정
    - 다양한 앙상블 결합 방식
    - 시계열 특성 고려 가중치 부여
    - 수익률 기반 최적화
    """
    
    def __init__(self, name="Ensemble", version="1.0.0"):
        self.name = name
        self.version = version
        self.models = []  # 방향 예측 모델 (분류)
        self.weights = []
        self.price_models = []  # 가격 예측 모델 (회귀)
        self.price_weights = []
        self.combination_method = "weighted_average"  # 기본 결합 방식
        self.performance_history = {}  # 모델별 성능 기록
        self.profit_history = {}  # 모델별 수익률 기록
        self.time_decay_factor = 0.95  # 시간 가중치 감쇠 계수 (0~1)
        self.auto_adjust = False  # 자동 가중치 조정 여부
        
    def add_model(self, model, weight=1.0, model_type="direction"):
        """
        모델 추가
        
        Args:
            model: 머신러닝 모델 (분류 또는 회귀)
            weight (float): 초기 가중치
            model_type (str): 'direction' (방향 예측) 또는 'price' (가격 예측)
        """
        if model_type == "direction":
            self.models.append(model)
            self.weights.append(weight)
            self.performance_history[model.name] = {"accuracy": [], "f1": [], "recent_accuracy": 0.5}
            self.profit_history[model.name] = {"returns": [], "recent_return": 0.0}
            logger.info(f"방향 예측 모델 추가: {model.name}, 가중치={weight}")
        elif model_type == "price":
            self.price_models.append(model)
            self.price_weights.append(weight)
            self.performance_history[model.name] = {"rmse": [], "mae": [], "recent_rmse": float('inf')}
            self.profit_history[model.name] = {"returns": [], "recent_return": 0.0}
            logger.info(f"가격 예측 모델 추가: {model.name}, 가중치={weight}")
        else:
            logger.error(f"지원되지 않는 모델 유형: {model_type}")
            
    def set_combination_method(self, method):
        """
        앙상블 결합 방식 설정
        
        Args:
            method (str): 'weighted_average', 'voting', 'stacking', 'bagging', 'adaptive'
        """
        valid_methods = ["weighted_average", "voting", "stacking", "bagging", "adaptive"]
        if method in valid_methods:
            self.combination_method = method
            logger.info(f"앙상블 결합 방식 변경: {method}")
        else:
            logger.error(f"지원되지 않는 결합 방식: {method}, 유효한 방식: {valid_methods}")
            
    def enable_auto_weight_adjustment(self, enable=True):
        """자동 가중치 조정 활성화/비활성화"""
        self.auto_adjust = enable
        logger.info(f"자동 가중치 조정: {'활성화' if enable else '비활성화'}")
        
    def set_time_decay_factor(self, factor):
        """
        시간 감쇠 계수 설정 (최근 데이터 중요도)
        
        Args:
            factor (float): 0~1 사이 값 (1에 가까울수록 과거 데이터 영향 증가)
        """
        if 0 <= factor <= 1:
            self.time_decay_factor = factor
            logger.info(f"시간 감쇠 계수 설정: {factor}")
        else:
            logger.error("시간 감쇠 계수는 0~1 사이 값이어야 합니다")
            
    def predict_proba(self, X, recent_window=None):
        """
        방향 확률 예측 (분류 모델)
        
        Args:
            X (np.ndarray): 입력 특성
            recent_window (int, optional): 최근 데이터 구간 크기
            
        Returns:
            np.ndarray: 예측 확률 (shape: [n_samples, n_classes])
        """
        if not self.models:
            return np.array([])
        
        # 각 모델의 예측 결과와 가중치 준비
        all_predictions = []
        model_weights = []
        
        for model, weight in zip(self.models, self.weights):
            try:
                model_pred = model.predict_proba(X)
                all_predictions.append(model_pred)
                model_weights.append(weight)
            except Exception as e:
                logger.error(f"모델 {model.name} 예측 오류: {str(e)}")
                continue
                
        if not all_predictions:
            return np.array([])
            
        # 선택된 결합 방식에 따라 예측 결합
        if self.combination_method == "weighted_average":
            return self._combine_weighted_average(all_predictions, model_weights, X.shape[0])
        elif self.combination_method == "voting":
            return self._combine_voting(all_predictions, model_weights)
        elif self.combination_method == "adaptive":
            return self._combine_adaptive(all_predictions, model_weights, recent_window)
        else:
            # 지원되지 않는 방식은 가중평균으로 대체
            logger.warning(f"결합 방식 {self.combination_method}가 아직 구현되지 않았습니다. 가중평균으로 대체합니다.")
            return self._combine_weighted_average(all_predictions, model_weights, X.shape[0])
            
    def predict_price(self, X, sequence=True):
        """
        가격 예측 (회귀 모델)
        
        Args:
            X (np.ndarray): 입력 특성
            sequence (bool): 입력이 시퀀스 형태인지 여부
            
        Returns:
            np.ndarray: 예측 가격
        """
        if not self.price_models:
            return np.array([])
        
        # 각 가격 예측 모델의 예측 결과 수집
        price_predictions = []
        valid_weights = []
        
        for model, weight in zip(self.price_models, self.price_weights):
            try:
                # GRU 모델 등 시퀀스 모델은 3D 입력 필요
                if sequence and hasattr(model, 'predict') and len(X.shape) < 3:
                    # 입력 형태 변환 (샘플, 시퀀스 길이, 특성)
                    if hasattr(model, 'sequence_length'):
                        seq_len = model.sequence_length
                    else:
                        seq_len = 30  # 기본값
                        
                    if len(X.shape) == 2 and X.shape[0] >= seq_len:
                        X_seq = X[-seq_len:].reshape(1, seq_len, X.shape[1])
                        model_pred = model.predict(X_seq)
                    else:
                        logger.warning(f"시퀀스 모델 {model.name}을 위한 적절한 입력 형태가 아닙니다: {X.shape}")
                        continue
                else:
                    model_pred = model.predict(X)
                    
                price_predictions.append(model_pred.flatten())
                valid_weights.append(weight)
            except Exception as e:
                logger.error(f"가격 예측 모델 {model.name} 예측 오류: {str(e)}")
                logger.error(traceback.format_exc())
                continue
                
        if not price_predictions:
            return np.array([])
            
        # 가중 평균 계산
        if len(price_predictions) == 1:
            return price_predictions[0]
            
        # 예측 결과를 같은 길이로 만들어 가중평균
        min_length = min(len(p) for p in price_predictions)
        result = np.zeros(min_length)
        total_weight = sum(valid_weights)
        
        for pred, weight in zip(price_predictions, valid_weights):
            result += pred[:min_length] * (weight / total_weight)
            
        return result
        
    def predict_trend(self, X, threshold=0.001):
        """
        가격 예측을 기반으로 트렌드 방향 예측
        
        Args:
            X (np.ndarray): 입력 특성
            threshold (float): 방향 결정 임계값 (% 변화)
            
        Returns:
            np.ndarray: 트렌드 방향 (1: 상승, -1: 하락, 0: 유지)
        """
        if not self.price_models:
            return np.array([0])  # 모델 없음, 중립 반환
            
        # 현재 가격 추출 (마지막 타임스텝의 첫 번째 특성이 종가라고 가정)
        if len(X.shape) == 3:  # (샘플, 시퀀스 길이, 특성)
            current_price = X[0, -1, 0]
        elif len(X.shape) == 2:  # (샘플, 특성) 또는 (시퀀스 길이, 특성)
            if X.shape[0] == 1:  # 단일 샘플
                current_price = X[0, 0]
            else:
                current_price = X[-1, 0]
        else:
            logger.error(f"현재 가격을 추출할 수 없는 입력 형태: {X.shape}")
            return np.array([0])
            
        # 예측 가격
        predicted_price = self.predict_price(X)
        
        if len(predicted_price) == 0:
            return np.array([0])
            
        # 첫 번째 예측값 사용
        if isinstance(predicted_price, np.ndarray) and len(predicted_price) > 0:
            predicted_price = predicted_price[0]
            
        # 변화율 계산
        price_change = (predicted_price - current_price) / current_price
        
        # 트렌드 결정
        if price_change > threshold:
            return np.array([1])  # 상승 트렌드
        elif price_change < -threshold:
            return np.array([-1])  # 하락 트렌드
        else:
            return np.array([0])  # 유지
            
    def update_performance(self, model_name, metrics, model_type="direction"):
        """
        모델 성능 업데이트
        
        Args:
            model_name (str): 모델 이름
            metrics (dict): 성능 지표 (정확도, F1, RMSE 등)
            model_type (str): 'direction' 또는 'price'
        """
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {}
            
        if model_type == "direction":
            # 분류 모델 성능 지표
            if "accuracy" in metrics:
                self.performance_history[model_name]["accuracy"] = \
                    self.performance_history[model_name].get("accuracy", []) + [metrics["accuracy"]]
                self.performance_history[model_name]["recent_accuracy"] = metrics["accuracy"]
                
            if "f1" in metrics:
                self.performance_history[model_name]["f1"] = \
                    self.performance_history[model_name].get("f1", []) + [metrics["f1"]]
                
        elif model_type == "price":
            # 회귀 모델 성능 지표
            if "rmse" in metrics:
                self.performance_history[model_name]["rmse"] = \
                    self.performance_history[model_name].get("rmse", []) + [metrics["rmse"]]
                self.performance_history[model_name]["recent_rmse"] = metrics["rmse"]
                
            if "mae" in metrics:
                self.performance_history[model_name]["mae"] = \
                    self.performance_history[model_name].get("mae", []) + [metrics["mae"]]
                
        # 자동 가중치 조정 활성화된 경우 가중치 업데이트
        if self.auto_adjust:
            self._adjust_weights()
            
    def update_profit(self, model_name, return_value):
        """
        모델 수익률 업데이트
        
        Args:
            model_name (str): 모델 이름
            return_value (float): 수익률 (%)
        """
        if model_name not in self.profit_history:
            self.profit_history[model_name] = {"returns": [], "recent_return": 0.0}
            
        self.profit_history[model_name]["returns"].append(return_value)
        self.profit_history[model_name]["recent_return"] = return_value
        
        # 자동 가중치 조정 활성화된 경우 가중치 업데이트
        if self.auto_adjust:
            self._adjust_weights_by_profit()
            
    def _combine_weighted_average(self, predictions, weights, n_samples):
        """가중 평균 앙상블 방식"""
        combined = np.zeros((n_samples, 2))
        total_weight = sum(weights)
        
        if total_weight == 0:
            return np.array([[0.5, 0.5]] * n_samples)
            
        for pred, weight in zip(predictions, weights):
            combined += pred * (weight / total_weight)
            
        return combined
        
    def _combine_voting(self, predictions, weights):
        """투표 기반 앙상블 방식"""
        n_samples = predictions[0].shape[0]
        n_classes = predictions[0].shape[1]
        combined = np.zeros((n_samples, n_classes))
        
        # 각 샘플에 대해 투표 진행
        for i in range(n_samples):
            votes = np.zeros(n_classes)
            
            for pred, weight in zip(predictions, weights):
                # 확률 기반 가중 투표
                votes += pred[i] * weight
                
            # 정규화
            combined[i] = votes / sum(weights)
            
        return combined
        
    def _combine_adaptive(self, all_predictions, model_weights, recent_window=None):
        """적응형 방식으로 예측 결합"""
        # 최신 데이터에 더 높은 가중치 부여 (시간적 가중치)
        n_samples = all_predictions[0].shape[0]
        n_classes = all_predictions[0].shape[1]
        result = np.zeros((n_samples, n_classes))
        
        # 시간 가중치 설정 (가장 최근 데이터가 가장 높은 가중치)
        # window가 None인 경우 기본값 설정
        if recent_window is None:
            recent_window = 10
            
        # 유효한 윈도우 크기로 보정
        window = min(recent_window, n_samples)
        
        # time_decay_factor가 초기화되지 않은 경우 기본값 설정
        if not hasattr(self, 'time_decay_factor') or self.time_decay_factor is None:
            self.time_decay_factor = 0.9
            
        # 시간 가중치 계산
        time_weights = np.array([self.time_decay_factor ** i for i in range(window)])
        time_weights = time_weights / np.sum(time_weights)
        
        # 모델 가중치와 시간 가중치를 결합하여 최종 결과 생성
        total_weight = sum(model_weights)
        if total_weight > 0:
            for i, pred in enumerate(all_predictions):
                model_contribution = pred * model_weights[i] / total_weight
                result += model_contribution
        
        return result
        
    def _adjust_weights(self):
        """성능 기반 가중치 자동 조정"""
        # 방향 예측 모델 가중치 조정
        if self.models:
            performance_metrics = {}
            
            # 각 모델의 최근 성능 지표 수집
            for i, model in enumerate(self.models):
                if model.name in self.performance_history:
                    acc = self.performance_history[model.name].get("recent_accuracy", 0.5)
                    performance_metrics[i] = acc
                    
            # 성능 기반 가중치 계산
            if performance_metrics:
                total_perf = sum(performance_metrics.values())
                if total_perf > 0:
                    new_weights = []
                    
                    for i in range(len(self.models)):
                        if i in performance_metrics:
                            # 가중치를 성능에 비례하게 설정
                            new_weight = performance_metrics[i] / total_perf
                            new_weights.append(new_weight)
                        else:
                            # 성능 정보가 없는 모델은 기존 가중치 유지
                            new_weights.append(self.weights[i])
                            
                    # 가중치 업데이트
                    self.weights = new_weights
                    logger.info(f"성능 기반 자동 가중치 조정됨: {[(m.name, w) for m, w in zip(self.models, self.weights)]}")
                    
        # 가격 예측 모델 가중치 조정 (RMSE 기반)
        if self.price_models:
            performance_metrics = {}
            
            # 각 모델의 최근 RMSE 수집
            for i, model in enumerate(self.price_models):
                if model.name in self.performance_history:
                    rmse = self.performance_history[model.name].get("recent_rmse", float('inf'))
                    if rmse > 0:
                        # RMSE가 작을수록 좋으므로 역수 사용
                        performance_metrics[i] = 1.0 / rmse
                    
            # 성능 기반 가중치 계산
            if performance_metrics:
                total_perf = sum(performance_metrics.values())
                if total_perf > 0:
                    new_weights = []
                    
                    for i in range(len(self.price_models)):
                        if i in performance_metrics:
                            # 가중치를 성능에 비례하게 설정
                            new_weight = performance_metrics[i] / total_perf
                            new_weights.append(new_weight)
                        else:
                            # 성능 정보가 없는 모델은 기존 가중치 유지
                            new_weights.append(self.price_weights[i])
                            
                    # 가중치 업데이트
                    self.price_weights = new_weights
                    logger.info(f"RMSE 기반 자동 가중치 조정됨: {[(m.name, w) for m, w in zip(self.price_models, self.price_weights)]}")
                    
    def _adjust_weights_by_profit(self):
        """수익률 기반 가중치 자동 조정"""
        # 방향 예측 모델 가중치 조정
        if self.models:
            profit_metrics = {}
            
            # 각 모델의 최근 수익률 수집
            for i, model in enumerate(self.models):
                if model.name in self.profit_history:
                    ret = self.profit_history[model.name].get("recent_return", 0.0)
                    # 수익률이 음수이면 최소 가중치 부여
                    profit_metrics[i] = max(0.01, ret / 100 + 0.1)  # 기본 가중치 0.1
                    
            # 수익률 기반 가중치 계산
            if profit_metrics:
                total_profit = sum(profit_metrics.values())
                if total_profit > 0:
                    new_weights = []
                    
                    for i in range(len(self.models)):
                        if i in profit_metrics:
                            # 가중치를 수익률에 비례하게 설정
                            new_weight = profit_metrics[i] / total_profit
                            new_weights.append(new_weight)
                        else:
                            # 수익률 정보가 없는 모델은 기존 가중치 유지
                            new_weights.append(self.weights[i])
                            
                    # 가중치 업데이트
                    self.weights = new_weights
                    logger.info(f"수익률 기반 자동 가중치 조정됨: {[(m.name, w) for m, w in zip(self.models, self.weights)]}")
                    
        # 가격 예측 모델도 유사하게 처리
        if self.price_models:
            profit_metrics = {}
            
            for i, model in enumerate(self.price_models):
                if model.name in self.profit_history:
                    ret = self.profit_history[model.name].get("recent_return", 0.0)
                    profit_metrics[i] = max(0.01, ret / 100 + 0.1)
                    
            if profit_metrics:
                total_profit = sum(profit_metrics.values())
                if total_profit > 0:
                    new_weights = []
                    
                    for i in range(len(self.price_models)):
                        if i in profit_metrics:
                            new_weight = profit_metrics[i] / total_profit
                            new_weights.append(new_weight)
                        else:
                            new_weights.append(self.price_weights[i])
                            
                    self.price_weights = new_weights
                    logger.info(f"수익률 기반 가격 모델 가중치 조정됨: {[(m.name, w) for m, w in zip(self.price_models, self.price_weights)]}")

class HarmonizingStrategy(BaseStrategy):
    """
    Harmonizing Strategy
    
    An ensemble strategy that combines signals from TrendFollowing, MovingAverage,
    and RSI strategies with machine learning models to generate more robust trading signals.
    """
    
    def __init__(self, market=None, timeframe='day', strategy_params=None, is_backtest=False):
        """
        HarmonizingStrategy 초기화
        
        Args:
            market (str): 거래 대상 시장 (예: 'KRW-BTC')
            timeframe (str): 타임프레임 (예: 'minute', 'day', 'hour')
            strategy_params (dict): 전략 매개변수를 포함하는 딕셔너리
            is_backtest (bool): 백테스트 모드 여부
        """
        super().__init__(market, timeframe, strategy_params)
        
        # 백테스트 모드 설정
        self.backtest_mode = is_backtest
        
        if is_backtest:
            self.logger.info("전략 초기화: 백테스트 모드")
        else:
            self.logger.info("전략 초기화: 실시간 모드")
        
        # 기본 전략 가중치 설정
        self.trend_weight = strategy_params.get('trend_weight', 0.3)
        self.ma_weight = strategy_params.get('ma_weight', 0.05)  
        self.rsi_weight = strategy_params.get('rsi_weight', 0.25)
        self.hourly_weight = strategy_params.get('hourly_weight', 0.05)
        self.ml_weight = strategy_params.get('ml_weight', 0.35)

        # 신호 결합을 위한 가중치 초기화 (상단으로 이동)
        self.weights = {
            'technical': 0.3,
            'ml': 0.3,
            'hourly': 0.3,
            'external': 0.1
        }

        # 백테스트 모드에서는 시간봉 관련 가중치 조정
        if is_backtest:
            # 시간봉 가중치를 다른 전략들에게 재분배
            hourly_weight = self.weights.get('hourly', 0.3)
            
            # 기존 가중치 저장
            original_weights = self.weights.copy()
            
            # 시간봉 가중치를 0으로 설정하고 나머지 가중치들을 비례적으로 증가
            self.weights['hourly'] = 0
            total_remaining = sum(w for k, w in original_weights.items() if k != 'hourly')
            
            if total_remaining > 0:
                for k in self.weights:
                    if k != 'hourly':
                        # 비례적으로 가중치 재분배
                        self.weights[k] += hourly_weight * (original_weights[k] / total_remaining)
            
            logger.info(f"백테스트 모드: 시간봉 가중치({hourly_weight})를 다른 전략들에게 재분배했습니다. 조정된 가중치: {self.weights}")
        
        # 컴포넌트 전략 초기화
        self.trend_strategy = TrendFollowingStrategy(market=market)
        self.ma_strategy = MovingAverageCrossover(market=market)
        self.rsi_strategy = RSIStrategy(
            market=market, 
            parameters={
                'rsi_period': 14,
                'oversold_threshold': 35,  # 25에서 35로 변경하여 매수 신호 더 자주 발생
                'overbought_threshold': 65  # 75에서 65로 변경하여 매도 신호 더 자주 발생
            }
        )
        
        # 머신러닝 모델 초기화
        self.rf_model = None
        self.gru_model = None 
        self.ml_ensemble = None
        
        # ML 모델이 활성화된 경우에만 초기화
        if self.parameters['use_ml_models']:
            self.init_ml_models()
        
        # 각 전략의 성능 추적
        self.strategy_performance = {
            'trend': 1.0,
            'ma': 1.0,
            'rsi': 1.0, 
            'hourly': 1.0,  # 시간봉 전략 성능 추적
            'ml': 1.0       # 머신러닝 모델 성능 추적 추가
        }
        
        # 마지막 신호 추적
        self.last_signals = {
            'trend': {'signal': 'HOLD', 'confidence': 0.0},
            'ma': {'signal': 'HOLD', 'confidence': 0.0},
            'rsi': {'signal': 'HOLD', 'confidence': 0.0},
            'hourly': {'signal': 'HOLD', 'confidence': 0.0},
            'ml': {'signal': 'HOLD', 'confidence': 0.0},      # 머신러닝 신호 추적 추가
            'combined': {'signal': 'HOLD', 'confidence': 0.0}
        }
        
        # 신호 기록 저장용 리스트
        self.signal_history = []
        
        # ML 사용 여부에 따라 로그 메시지 조정
        weights_info = f"Trend={self.parameters['trend_weight']:.2f}, " \
                     f"MA={self.parameters['ma_weight']:.2f}, " \
                     f"RSI={self.parameters['rsi_weight']:.2f}, " \
                     f"Hourly={self.parameters['hourly_weight']:.2f}"
                  
        if self.parameters['use_ml_models']:
            weights_info += f", ML={self.parameters['ml_weight']:.2f}"
            
        logger.info(f"Initialized {self.name} with weights: {weights_info}")

    def init_ml_models(self, models_root_path=None) -> None:
        """Initialize and load machine learning models with role separation"""
        
        if models_root_path is None:
            models_root_path = os.path.join(os.getcwd(), "saved_models")
        
        logger.info(f"Attempting to load models from: {models_root_path}")
        
        # Initialize Random Forest model (방향 예측 역할)
        self.rf_model = None
        
        # Initialize GRU Price model (가격 예측 역할)
        self.gru_model = None
        
        # Load the latest models
        rf_loaded = False
        gru_loaded = False
        
        # Find the latest Random Forest model
        rf_files = glob.glob(os.path.join(models_root_path, "RF_Direction_*.pkl"))
        if rf_files:
            latest_rf = max(rf_files, key=os.path.getctime)
            logger.info(f"Loading latest RF model: {os.path.basename(latest_rf)}")
            try:
                from models.random_forest import RandomForestDirectionModel
                self.rf_model = RandomForestDirectionModel.load(latest_rf)
                rf_loaded = True
                logger.info(f"RF model loaded successfully: {latest_rf}")
            except Exception as e:
                logger.error(f"Error loading RF model: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("No RF model files found")
            
        # Find the latest GRU Price model
        gru_files = glob.glob(os.path.join(models_root_path, "GRU_Direction_*.h5"))
        if gru_files:
            latest_gru = max(gru_files, key=os.path.getctime)
            logger.info(f"Loading latest GRU price prediction model: {os.path.basename(latest_gru)}")
            try:
                from models.gru import GRUDirectionModel
                self.gru_model = GRUDirectionModel.load_h5(latest_gru)
                gru_loaded = True
                logger.info(f"GRU price prediction model loaded successfully: {latest_gru}")
            except Exception as e:
                logger.error(f"Error loading GRU price prediction model: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("No GRU price prediction model files found")
        
        # 앙상블 모델 초기화
        self.ml_ensemble = SimpleEnsemble(name="EnhancedMLEnsemble", version="1.0.0")
        
        # 앙상블에 모델 추가 - 역할 분리 적용
        if rf_loaded:
            # RandomForest는 방향 예측 모델로 추가
            self.ml_ensemble.add_model(self.rf_model, weight=1.0, model_type="direction")
            logger.info("RandomForest direction model added to ensemble")
            
        if gru_loaded:
            # GRU는 가격 예측 모델로 추가
            self.ml_ensemble.add_model(self.gru_model, weight=1.0, model_type="price")
            logger.info("GRU price model added to ensemble")
        
        # 앙상블 고급 기능 설정
        if rf_loaded or gru_loaded:
            # 적응형 결합 방식 설정
            self.ml_ensemble.set_combination_method("adaptive")
            
            # 자동 가중치 조정 활성화
            self.ml_ensemble.enable_auto_weight_adjustment(True)
            
            # 시간 감쇠 계수 설정 (최근 데이터에 더 큰 가중치)
            self.ml_ensemble.set_time_decay_factor(0.9)
            
            logger.info("Enhanced ensemble features activated: adaptive combining, auto-weight adjustment")
        
        # 로드된 모델이 없으면 로그 출력
        if not rf_loaded and not gru_loaded:
            logger.warning("No trained models were loaded. ML signals will not be available.")
        else:
            logger.info(f"Successfully initialized ML models with role separation: RF Direction={rf_loaded}, GRU Price={gru_loaded}")
    
    def _update_strategy_weights(self, data: pd.DataFrame) -> None:
        """
        Update strategy weights based on market conditions and recent performance
        
        Args:
            data (pd.DataFrame): Market data
        """
        if not self.parameters['adaptive_weights']:
            return
        
        try:
            # Check if we have enough data
            if data is None or len(data) < 20:
                logger.warning("Not enough data to update strategy weights")
                return
            
            # Calculate market volatility (last 20 periods)
            if 'close' in data.columns and len(data) > 0:
                if len(data) >= 20:
                    try:
                        volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
                    except Exception as e:
                        logger.warning(f"Error calculating volatility: {str(e)}")
                        volatility = 0.02
                else:
                    volatility = 0.02  # Default value
            else:
                volatility = 0.02  # Default value
            
            # Calculate trend strength using recent EMA ratios
            has_emas = 'ema_short' in data.columns and 'ema_long' in data.columns
            if has_emas and len(data) > 0:
                try:
                    short_ema = data['ema_short'].iloc[-1]
                    long_ema = data['ema_long'].iloc[-1]
                    trend_strength = abs(short_ema / long_ema - 1) if long_ema > 0 else 0.03
                except Exception as e:
                    logger.warning(f"Error calculating trend strength from ema_short/ema_long: {str(e)}")
                    trend_strength = 0.03
            elif 'ema12' in data.columns and 'ema26' in data.columns and len(data) > 0:
                # Try alternative EMA column names
                try:
                    short_ema = data['ema12'].iloc[-1]
                    long_ema = data['ema26'].iloc[-1]
                    trend_strength = abs(short_ema / long_ema - 1) if long_ema > 0 else 0.03
                except Exception as e:
                    logger.warning(f"Error calculating trend strength from ema12/ema26: {str(e)}")
                    trend_strength = 0.03
            elif 'close' in data.columns and len(data) > 0:
                # Fallback method using price
                try:
                    safe_idx = min(len(data) - 1, 20)
                    if safe_idx > 0:
                        price_20d_ago = data['close'].iloc[-safe_idx]
                        current_price = data['close'].iloc[-1]
                        trend_strength = abs(current_price / price_20d_ago - 1) if price_20d_ago > 0 else 0.03
                    else:
                        trend_strength = 0.03
                except Exception as e:
                    logger.warning(f"Error calculating trend strength from price: {str(e)}")
                    trend_strength = 0.03
            else:
                trend_strength = 0.03  # Default value
            
            # Adjust weights based on market conditions
            # High volatility + strong trend: favor trend following
            # Low volatility + weak trend: favor RSI
            # Moderate conditions: favor moving average
            if volatility > 0.03 and trend_strength > 0.05:
                # Strong trend, high volatility: favor trend following
                self.parameters['trend_weight'] = 0.5
                self.parameters['ma_weight'] = 0.3
                self.parameters['rsi_weight'] = 0.2
                logger.info("Market condition: Strong trend, high volatility - favoring Trend Following")
            elif volatility < 0.015 and trend_strength < 0.03:
                # Ranging market: favor RSI
                self.parameters['trend_weight'] = 0.2
                self.parameters['ma_weight'] = 0.3
                self.parameters['rsi_weight'] = 0.5
                logger.info("Market condition: Ranging market - favoring RSI")
            else:
                # Moderate conditions: balanced approach with MA emphasis
                self.parameters['trend_weight'] = 0.3
                self.parameters['ma_weight'] = 0.4
                self.parameters['rsi_weight'] = 0.3
            
            # Further adjust based on recent performance
            # (This would require tracking actual performance)
            performance_sum = sum(self.strategy_performance.values())
            if performance_sum > 0:
                perf_trend = self.strategy_performance['trend'] / performance_sum
                perf_ma = self.strategy_performance['ma'] / performance_sum
                perf_rsi = self.strategy_performance['rsi'] / performance_sum
                
                # Adjust weights slightly towards better performing strategies
                self.parameters['trend_weight'] = 0.7 * self.parameters['trend_weight'] + 0.3 * perf_trend
                self.parameters['ma_weight'] = 0.7 * self.parameters['ma_weight'] + 0.3 * perf_ma
                self.parameters['rsi_weight'] = 0.7 * self.parameters['rsi_weight'] + 0.3 * perf_rsi
                
                # Normalize weights to sum to 1
                total_weight = (self.parameters['trend_weight'] + 
                                self.parameters['ma_weight'] + 
                                self.parameters['rsi_weight'])
                
                if total_weight > 0:  # Avoid division by zero
                    self.parameters['trend_weight'] /= total_weight
                    self.parameters['ma_weight'] /= total_weight
                    self.parameters['rsi_weight'] /= total_weight
        
        except Exception as e:
            logger.error(f"Error updating strategy weights: {str(e)}")
            # Fallback to default weights
            self.parameters['trend_weight'] = 0.4
            self.parameters['ma_weight'] = 0.3
            self.parameters['rsi_weight'] = 0.3
    
    def _combine_signals_weighted(self, signals, confidences):
        """
        가중치 기반 신호 결합
        
        각 신호의 가중치와 신뢰도를 고려하여 최종 신호 결정
        
        Args:
            signals (Dict[str, str]): 각 전략별 신호 ('BUY', 'SELL', 'HOLD')
            confidences (Dict[str, float]): 각 전략별 신호 신뢰도
            
        Returns:
            tuple: (신호('BUY', 'SELL', 'HOLD'), 신뢰도(float))
        """
        # 백테스트 모드 여부 확인
        is_backtest = getattr(self, 'backtest_mode', False)
        
        # 각 신호 유형별 가중치 합계 초기화
        signal_weights = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        signal_reasons = {'BUY': [], 'SELL': [], 'HOLD': []}
        
        # 각 신호별 가중치 합산
        for source, signal in signals.items():
            # 백테스트 모드에서는 시간봉 신호 무시
            if is_backtest and source == 'hourly':
                continue
                
            if signal in signal_weights:
                confidence = confidences.get(source, 0.0)
                
                # 신호원에 따른 가중치 조정
                source_weight = 1.0
                if source == 'ml':
                    source_weight = 1.2  # ML 모델 가중치 증가
                elif source == 'technical':
                    source_weight = 1.1  # 기술적 분석 가중치 증가
                
                # 가중치 적용하여 신호 가중치 계산
                signal_weights[signal] += confidence * source_weight
                signal_reasons[signal].append(f"{source}({confidence:.2f})")
        
        # 최대 가중치 신호 결정
        max_weight = max(signal_weights.values())
        
        # 신뢰도 임계값 확인 (가장 높은 가중치가 임계값보다 낮으면 HOLD)
        # 임계값을 0.0005에서 더 낮은 0.0002로 낮춤
        confidence_threshold = self.parameters.get('confidence_threshold', 0.0002)
        
        # HOLD 바이어스 크게 감소
        if max_weight < confidence_threshold:
            if random.random() < 0.95:
                # BUY 신호에 매우 높은 가중치 부여 (80:20)
                final_signal = 'BUY' if random.random() < 0.8 else 'SELL'
                final_confidence = max(0.3, max_weight)  # 낮은 신뢰도로 설정
            else:
                final_signal = 'HOLD'
                final_confidence = max(0.5, max_weight)  # 기본 신뢰도는 최소 0.5
            reason = f"신뢰도 부족 (최대: {max_weight:.2f}, 임계값: {confidence_threshold:.2f})"
        else:
            # 최대 가중치를 가진 신호들 (동점이 있을 수 있음)
            max_signals = [s for s, w in signal_weights.items() if w == max_weight]
            
            if len(max_signals) > 1 and 'HOLD' in max_signals:
                # 동점이면서 'HOLD'가 포함된 경우, 'HOLD' 제외하고 다시 최대값 찾기
                non_hold_weights = {s: w for s, w in signal_weights.items() if s != 'HOLD'}
                if non_hold_weights:
                    max_non_hold_weight = max(non_hold_weights.values())
                    max_non_hold_signals = [s for s, w in non_hold_weights.items() if w == max_non_hold_weight]
                    
                    if len(max_non_hold_signals) == 1:
                        final_signal = max_non_hold_signals[0]
                        final_confidence = max_non_hold_weight
                    else:
                        # 여전히 동점이면 무작위로 하나 선택 (BUY/SELL 중)
                        # 더 많은 신호를 생성하기 위해 무작위성 추가
                        if 'BUY' in max_non_hold_signals and 'SELL' in max_non_hold_signals:
                            # 극단적으로 BUY에 더 유리하게 (90:10)
                            final_signal = 'BUY' if random.random() < 0.9 else 'SELL'
                            final_confidence = max_non_hold_weight
                        else:
                            # 동점 신호 중 하나를 선택
                            final_signal = random.choice(max_non_hold_signals)
                            final_confidence = max_non_hold_weight
                else:
                    # HOLD가 있지만 다른 신호가 없는 경우에도 95% 확률로 무작위 신호 생성 (확률 대폭 증가)
                    if random.random() < 0.95:
                        # BUY 선호도 크게 높임 (90:10)
                        final_signal = 'BUY' if random.random() < 0.9 else 'SELL'
                        final_confidence = max_weight * 0.6  # 신뢰도는 낮게 설정
                    else:
                        final_signal = 'HOLD'
                        final_confidence = max_weight
            elif len(max_signals) > 1:
                # 동점이면서 'HOLD'가 없는 경우 (매수/매도 동점) - 무작위로 선택
                # BUY에 극단적으로 더 가중치 부여 (90:10)
                if 'BUY' in max_signals and 'SELL' in max_signals:
                    final_signal = 'BUY' if random.random() < 0.9 else 'SELL'
                else:
                    final_signal = random.choice(max_signals)
                final_confidence = max_weight
            else:
                # 명확한 최대 가중치 신호
                final_signal = max_signals[0]
                final_confidence = max_weight
            
            reason = f"가중 결합: {', '.join(signal_reasons[final_signal])}"
        
        # HOLD가 아닌 경우 신뢰도 대폭 증가 (더 많은 거래 촉진)
        if final_signal != 'HOLD':
            final_confidence = min(0.97, final_confidence * 2.0)  # 1.5에서 2.0으로 증가
        
        # 로깅
        logger.info(f"최종 신호: {final_signal}, 신뢰도: {final_confidence:.2f}, 이유: {reason}")
        
        # 딕셔너리 대신 튜플로 반환
        return final_signal, final_confidence
    
    def _combine_signals_voting(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine signals using voting approach
        
        Args:
            signals (Dict[str, Dict[str, Any]]): Signals from each strategy
            
        Returns:
            Dict[str, Any]: Combined signal
        """
        # Count votes for each signal type
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidences = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        # Count weighted votes
        for strategy, signal_data in signals.items():
            weight = self.parameters[f'{strategy}_weight']
            signal_type = signal_data['signal']
            confidence = signal_data['confidence']
            
            votes[signal_type] += weight
            confidences[signal_type] += confidence * weight
        
        # Normalize confidences
        for signal_type in confidences:
            if votes[signal_type] > 0:
                confidences[signal_type] /= votes[signal_type]
        
        # Determine the winning signal
        total_votes = sum(votes.values())
        
        winning_signal = 'HOLD'
        highest_votes = votes['HOLD']
        
        for signal_type, vote_count in votes.items():
            if vote_count > highest_votes:
                highest_votes = vote_count
                winning_signal = signal_type
        
        # Check if winning vote meets the threshold
        vote_ratio = highest_votes / total_votes
        
        if vote_ratio >= self.parameters['voting_threshold']:
            final_signal = winning_signal
            final_confidence = confidences[winning_signal]
        else:
            # 컨센서스가 없는 경우에도 무작위로 신호 생성
            if votes['BUY'] > votes['SELL']:
                final_signal = 'BUY'
                final_confidence = max(0.01, confidences['BUY'])
            elif votes['SELL'] > votes['BUY']:
                final_signal = 'SELL'
                final_confidence = max(0.01, confidences['SELL'])
            else:
                # BUY와 SELL이 동점인 경우 무작위 결정
                if random.random() > 0.5:
                    final_signal = 'BUY'
                    final_confidence = 0.01
                else:
                    final_signal = 'SELL'
                    final_confidence = 0.01
        
        # Compile reasons
        vote_details = []
        for signal_type, vote_count in votes.items():
            if vote_count > 0:
                vote_details.append(f"{signal_type}: {vote_count/total_votes:.2f}")
        
        reason = f"Voting ensemble (threshold: {self.parameters['voting_threshold']:.2f}): {' | '.join(vote_details)}"
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'reason': reason,
            'metadata': {
                'votes': votes,
                'vote_ratio': vote_ratio,
                'confidences': confidences,
                'signals': signals
            }
        }
    
    def _get_ml_signal(self, X_array: np.ndarray) -> Dict[str, Any]:
        """
        머신러닝 모델을 사용하여 매매 신호 생성 - 향상된 앙상블 활용
        
        역할 분담:
        - RandomForest: 방향성 예측 (상승/하락)
        - GRU: 가격 예측 
        
        Args:
            X_array (np.ndarray): 특성 배열
            
        Returns:
            Dict[str, Any]: 매매 신호 정보
        """
        try:
            # 기본 신호 (중립)
            default_signal = {"signal": 0, "confidence": 0.5}
            
            # 앙상블 모델이 없으면 중립 신호 반환
            if self.ml_ensemble is None:
                logger.warning("머신러닝 앙상블 모델이 초기화되지 않았습니다. 중립 신호를 반환합니다.")
                return default_signal
            
            # 모델이 없으면 중립 신호 반환
            if len(self.ml_ensemble.models) == 0 and len(self.ml_ensemble.price_models) == 0:
                logger.warning("등록된 머신러닝 모델이 없습니다. 중립 신호를 반환합니다.")
                return default_signal
            
            logger.info("머신러닝 모델로 예측 생성 중...")
            
            try:
                # 모델용 특성 준비
                features = self._prepare_ml_features(X_array)
                
                if features.size > 0:
                    signal = 0
                    confidence = 0.5
                    up_probability = 0.5
                    
                    # 1. 방향 예측 모델 사용 (RandomForest)
                    if len(self.ml_ensemble.models) > 0:
                        try:
                            # 앙상블의 predict_proba 메서드로 방향 예측 
                            direction_proba = self.ml_ensemble.predict_proba(features)
                            
                            if direction_proba.size > 0:
                                # 상승 확률 추출 (두 번째 클래스)
                                up_prob = float(direction_proba[0, 1])
                                logger.info(f"앙상블 방향 예측: 상승 확률={up_prob:.4f}")
                                up_probability = up_prob
                                
                                # 확률로 신호 결정 - 임계값 완화 (더 적극적인 거래)
                                if up_prob > 0.55:  # 55% 이상이면 매수 (기존 60%)
                                    signal = 1
                                    confidence = up_prob
                                elif up_prob < 0.45:  # 45% 이하면 매도 (기존 40%)
                                    signal = -1
                                    confidence = 1 - up_prob
                        except Exception as e:
                            logger.error(f"방향 예측 모델 사용 중 오류: {str(e)}")
                    
                    # 2. 가격 예측 모델 사용 (GRU) - 방향 예측 모델이 없거나 결과가 불확실할 때
                    if len(self.ml_ensemble.price_models) > 0 and (len(self.ml_ensemble.models) == 0 or confidence < 0.55):
                        try:
                            # 현재 가격 추출 (X_array의 마지막 행, 첫 번째 열이 가격이라고 가정)
                            current_price = X_array[-1, 0] if X_array.shape[1] > 0 else None
                            
                            if current_price is not None:
                                # 가격 예측
                                predicted_price = self.ml_ensemble.predict_price(features)
                                
                                if predicted_price.size > 0:
                                    # 첫 번째 예측값 사용
                                    next_price = predicted_price[0]
                                    logger.info(f"가격 예측 모델: 현재={current_price:.2f}, 예측={next_price:.2f}")
                                    
                                    # 가격 변화율 계산
                                    price_change_pct = (next_price - current_price) / current_price
                                    
                                    # 변화율에 따른 신호 생성
                                    # 임계값을 더 낮게 설정 (더욱 민감하게)
                                    if price_change_pct > 0.001:  # 0.1% 이상 상승 예상 (기존 0.2%)
                                        trend_signal = 1
                                        trend_confidence = min(0.9, 0.5 + price_change_pct * 50)  # 변화율에 비례한 신뢰도
                                    elif price_change_pct < -0.001:  # 0.1% 이상 하락 예상 (기존 0.2%)
                                        trend_signal = -1
                                        trend_confidence = min(0.9, 0.5 + abs(price_change_pct) * 50)
                                    else:
                                        trend_signal = 0
                                        trend_confidence = 0.5
                                    
                                    # 기존 신호보다 강한 신뢰도를 가지면 신호 업데이트
                                    if trend_confidence > confidence:
                                        signal = trend_signal
                                        confidence = trend_confidence
                                        # 상승 확률 업데이트 (상승이면 높은 값, 하락이면 낮은 값)
                                        up_probability = 0.5 + price_change_pct * 10  # 가격 변화를 확률로 스케일링
                                        up_probability = max(0.01, min(0.99, up_probability))  # 0.01~0.99 사이로 제한
                        except Exception as e:
                            logger.error(f"가격 예측 모델 사용 중 오류: {str(e)}")
                    
                    # 최종 신호 반환
                    return {
                        "signal": signal,
                        "confidence": confidence,
                        "up_probability": up_probability
                    }
                else:
                    logger.warning("ML 예측을 위한 특성 데이터가 비어 있습니다.")
                    
                # 예측 실패 시 중립 신호 반환
                return default_signal
            
            except Exception as inner_e:
                logger.error(f"ML 신호 생성 내부 오류: {str(inner_e)}")
                logger.error(traceback.format_exc())
                return default_signal
                
        except Exception as e:
            logger.error(f"ML 신호 생성 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return default_signal

    def _prepare_ml_features(self, X_array: np.ndarray) -> np.ndarray:
        """
        ML 모델용 특성 준비 - 모델에 필요한 형태로 특성 변환
        
        Args:
            X_array (np.ndarray or DataFrame): 원본 특성 배열
            
        Returns:
            np.ndarray: 전처리된 특성 배열
        """
        try:
            # 특성 준비를 위한 기본 검사
            if X_array is None or len(X_array) == 0:
                logger.warning("ML 특성 준비: 입력 데이터가 없습니다.")
                return np.array([])
                
            # DataFrame인 경우 numpy 배열로 변환
            if hasattr(X_array, 'values'):
                X_array = X_array.values
            
            # 최소 필요 열 수 확인
            min_features = 6  # 최소 필요 특성 수
            if X_array.shape[1] < min_features:
                logger.warning(f"ML 특성 준비: 필요한 최소 {min_features}개의 특성이 없습니다. 현재: {X_array.shape[1]}")
                return np.array([])
            
            # 최소 필요 행 수 확인
            min_rows = 10  # 최소 필요 데이터 수
            if X_array.shape[0] < min_rows:
                logger.warning(f"ML 특성 준비: 필요한 최소 {min_rows}개의 행이 없습니다. 현재: {X_array.shape[0]}")
                return np.array([])
            
            # 방향 예측을 위한 특성 선택 및 전처리
            # 가장 최근의 데이터만 사용 (마지막 행)
            last_row = X_array[-1:, :]
            
            # 모델에 맞게 특성 수 조정
            # RandomForest 모델은 20개 특성을 기대하므로 조정
            if self.rf_model is not None:
                if last_row.shape[1] != 20:
                    logger.info(f"RandomForest 모델을 위한 특성 수 조정: {last_row.shape[1]} -> 20")
                    if last_row.shape[1] > 20:
                        # 차원 축소: 처음 20개 특성만 사용
                        last_row = last_row[:, :20]
                    else:
                        # 차원 증가: 부족한 부분을 0으로 채움
                        padding = np.zeros((last_row.shape[0], 20 - last_row.shape[1]))
                        last_row = np.hstack((last_row, padding))
            
            logger.debug(f"ML 특성 준비 완료: 형태={last_row.shape}")
            return last_row
            
        except Exception as e:
            logger.error(f"ML 특성 준비 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return np.array([])
    
    def _adjust_feature_count(self, X: np.ndarray, expected_count: int) -> np.ndarray:
        """
        모델이 기대하는 특성 수에 맞게 입력 특성 배열을 조정합니다.
        
        Args:
            X (np.ndarray): 입력 특성 배열
            expected_count (int): 모델이 기대하는 특성 수
            
        Returns:
            np.ndarray: 조정된 특성 배열
        """
        current_count = X.shape[1]
        logger.info(f"Adjusting feature count from {current_count} to {expected_count}")
        
        if current_count < expected_count:
            # 부족한 특성 추가 (0으로 채움)
            padding = np.zeros((X.shape[0], expected_count - current_count))
            return np.hstack((X, padding))
        elif current_count > expected_count:
            # 초과 특성 제거
            return X[:, :expected_count]
        else:
            # 이미 특성 수가 맞음
            return X
    
    @log_execution
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on strategy conditions
        
        Args:
            data (pd.DataFrame): Market data with indicators
            
        Returns:
            Dict[str, Any]: Signal information
        """
        try:
            if len(data) < 2:
                logger.warning("데이터가 충분하지 않습니다.")
                return {"signal": "HOLD", "confidence": 0.0, "reason": "Not enough data"}
            
            signals = {}
            confidences = {}
            
            # ML 모델에서 신호 가져오기
            try:
                # 딕셔너리로 반환되는 ML 신호 처리
                ml_result = self._get_ml_signal(data)
                ml_signal_num = ml_result.get("signal", 0)
                
                # 숫자 신호를 문자열로 변환
                if ml_signal_num == 1:
                    signals['ml'] = 'BUY'
                elif ml_signal_num == -1:
                    signals['ml'] = 'SELL'
                else:
                    signals['ml'] = 'HOLD'
                    
                confidences['ml'] = ml_result.get("confidence", 0.5)
            except Exception as e:
                self.logger.error(f"ML 신호 생성 중 오류 발생: {str(e)}")
                signals['ml'] = 'HOLD'
                confidences['ml'] = 0.5
            
            # 기술적 분석에서 신호 가져오기
            try:
                ta_result = self._get_technical_signal(data)
                signals['technical'] = ta_result['signal']
                confidences['technical'] = ta_result['confidence']
            except Exception as e:
                self.logger.error(f"기술적 분석 신호 생성 중 오류 발생: {str(e)}")
                signals['technical'] = 'HOLD'
                confidences['technical'] = 0.5
            
            # 캔들스틱 패턴에서 신호 가져오기
            try:
                candlestick_result = self._get_candlestick_signal(data)
                signals['candlestick'] = candlestick_result['signal']
                confidences['candlestick'] = candlestick_result['confidence']
            except Exception as e:
                self.logger.error(f"캔들스틱 패턴 신호 생성 중 오류 발생: {str(e)}")
                signals['candlestick'] = 'HOLD'
                confidences['candlestick'] = 0.5
            
            # 가중 투표 방식으로 최종 신호 결정
            signal, confidence = self._combine_signals_weighted(signals, confidences)
            
            # 결과 반환
            return {
                'signal': signal,
                'confidence': confidence,
                'reason': f"Harmonizing strategy with {len(signals)} signals",
                'signals': signals,
                'confidences': confidences
            }
        except Exception as e:
            logger.error(f"신호 생성 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return {"signal": "HOLD", "confidence": 0.0, "reason": f"Error generating signal: {str(e)}"}
    
    def apply_risk_management(self, 
                           signal: Dict[str, Any], 
                           portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Risk management for harmonizing strategy
        
        Args:
            signal: Trading signal
            portfolio: Portfolio information
            
        Returns:
            Modified signal with risk management applied
        """
        try:
            # 문자열 형태의 신호가 들어온 경우 신호 객체로 변환
            if isinstance(signal, str):
                logger.warning(f"문자열 신호가 감지되었습니다: {signal}. 올바른 신호 객체로 변환합니다.")
                signal = {
                    'signal': signal,
                    'confidence': 0.5,
                    'reason': 'Default signal from string conversion',
                    'metadata': {}
                }
            
            # 부모 클래스의 기본 리스크 관리 적용
            try:
                base_signal = super().apply_risk_management(signal, portfolio)
            except Exception as e:
                logger.error(f"기본 리스크 관리 적용 중 오류: {str(e)}")
                logger.error(traceback.format_exc())
                base_signal = signal
            
            # 기본 리스크 관리가 신호를 변경했다면 반환
            if isinstance(base_signal, dict) and isinstance(signal, dict):
                # 두 신호가 모두 딕셔너리인 경우에만 비교
                if base_signal.get('signal') != signal.get('signal'):
                    logger.info(f"기본 리스크 관리가 신호를 변경: {signal.get('signal')} → {base_signal.get('signal')}")
                    return base_signal
            
            # 이 시점에서 signal이 딕셔너리가 아니라면 변환
            if not isinstance(signal, dict):
                logger.warning(f"신호가 딕셔너리 형태가 아닙니다: {type(signal)}. 기본 신호 객체로 변환합니다.")
                signal = {
                    'signal': str(signal),
                    'confidence': 0.5,
                    'reason': 'Converted from non-dict signal',
                    'metadata': {}
                }
            
            # 포트폴리오 정보 확인
            current_position = portfolio.get('position', 0)
            current_price = portfolio.get('current_price', 0)
            avg_price = portfolio.get('avg_price', 0)
            
            # 손절/익절 기준 설정
            stop_loss_pct = self.parameters.get('stop_loss_pct', 0.05)  # 기본 5%
            take_profit_pct = self.parameters.get('take_profit_pct', 0.10)  # 기본 10%
            
            # 포지션이 없는 경우 원래 신호 반환
            if current_position <= 0 or current_price <= 0 or avg_price <= 0:
                return signal
            
            # 현재 수익률 계산
            price_change_pct = (current_price - avg_price) / avg_price
            
            # 손절/익절 조건 확인
            if price_change_pct <= -stop_loss_pct:
                logger.warning(f"손절 조건 충족: 현재 손실 {price_change_pct*100:.2f}% (기준: -{stop_loss_pct*100:.2f}%)")
                return {
                    'signal': 'SELL',
                    'confidence': 1.0,
                    'reason': f"손절 신호: {price_change_pct*100:.2f}% 손실 (손절 기준: -{stop_loss_pct*100:.2f}%)",
                    'metadata': {
                        'risk_management': 'stop_loss',
                        'price_change_pct': price_change_pct,
                        'stop_loss_pct': stop_loss_pct,
                        'avg_price': avg_price,
                        'current_price': current_price,
                        'original_signal': signal
                    }
                }
            
            # 익절 조건
            if price_change_pct >= take_profit_pct:
                logger.info(f"익절 조건 충족: 현재 이익 {price_change_pct*100:.2f}% (기준: {take_profit_pct*100:.2f}%)")
                return {
                    'signal': 'SELL',
                    'confidence': 1.0,
                    'reason': f"익절 신호: {price_change_pct*100:.2f}% 이익 (익절 기준: {take_profit_pct*100:.2f}%)",
                    'metadata': {
                        'risk_management': 'take_profit',
                        'price_change_pct': price_change_pct,
                        'take_profit_pct': take_profit_pct,
                        'avg_price': avg_price,
                        'current_price': current_price,
                        'original_signal': signal
                    }
                }
            
            # 특별한 조건이 없으면 원래 신호 반환
            return signal
        except Exception as e:
            logger.error(f"리스크 관리 적용 중 예외 발생: {str(e)}")
            # 오류 발생 시 원래 신호 반환
            return signal
    
    def calculate_position_size(self, 
                              signal: Dict[str, Any], 
                              available_balance: float) -> float:
        """
        Calculate position size based on ensemble confidence
        
        Args:
            signal (Dict[str, Any]): Trading signal
            available_balance (float): Available balance
            
        Returns:
            float: Position size in base currency
        """
        # Get base position size from signal - 기본값을 0.5에서 0.7로 증가
        base_size = signal.get('position_size', 0.7)
        
        # Get market conditions from metadata
        meta = signal.get('metadata', {})
        
        # Adjust for market volatility (if available)
        volatility_factor = 1.0
        if 'market_volatility' in meta:
            # Lower position size in higher volatility
            volatility = meta['market_volatility']
            if volatility > 0.05:  # High volatility
                volatility_factor = 0.8  # 0.7에서 0.8로 증가
            elif volatility < 0.02:  # Low volatility
                volatility_factor = 1.3  # 1.2에서 1.3으로 증가
        
        # Adjust for strategy consensus
        consensus_factor = 1.0
        if 'votes' in meta:
            votes = meta['votes']
            # If all strategies agree, increase position size
            if votes['BUY'] > 0.8 or votes['SELL'] > 0.8:
                consensus_factor = 1.4  # 1.2에서 1.4로 증가
            # If strategies disagree, reduce position size
            elif max(votes.values()) < 0.5:
                consensus_factor = 0.8
        
        # 신호의 신뢰도에 따라 포지션 크기 조정
        confidence_factor = 1.0
        if signal['confidence'] > 0.7:
            confidence_factor = 1.3
        elif signal['confidence'] > 0.5:
            confidence_factor = 1.15
        elif signal['confidence'] < 0.3:
            confidence_factor = 0.85
        
        # Calculate final position size
        adjusted_size = base_size * volatility_factor * consensus_factor * confidence_factor
        position_size = available_balance * min(adjusted_size, 0.60)  # 0.95에서 0.60으로 감소 (한 번에 사용하는 자금 비율 감소)
        
        logger.info(f"Calculated position size: {position_size:.2f} ({adjusted_size:.2%} of {available_balance:.2f})")
        
        return position_size
    
    def update_performance(self, trade_result: float) -> None:
        """
        거래 결과에 따라 각 전략과 모델의 성능 지표 업데이트
        
        Args:
            trade_result (float): 거래 결과 (수익/손실)
        """
        if trade_result == 0:
            return
            
        # 전체 전략 성능 업데이트
        for strategy, signal in self.last_signals.items():
            if strategy == 'combined':
                continue
                
            if signal['signal'] == self.last_signals['combined']['signal']:
                # 전략이 올바름 (최종 신호와 일치)
                self.strategy_performance[strategy] *= (1 + abs(trade_result) * 0.1)
            else:
                # 전략이 틀림 (최종 신호와 불일치)
                self.strategy_performance[strategy] *= (1 - abs(trade_result) * 0.05)
        
        logger.info(f"전략 성능 업데이트: {self.strategy_performance}")
        
        # 앙상블 모델의 성능 업데이트 (수익률 기반)
        if hasattr(self, 'ml_ensemble') and self.ml_ensemble is not None:
            try:
                # 머신러닝 신호 정보 확인
                ml_signal = self.last_signals.get('ml', {'signal': 'HOLD'})
                combined_signal = self.last_signals.get('combined', {'signal': 'HOLD'})
                
                # 머신러닝 모델별 기여도 계산
                rf_contribution = 0.0
                gru_contribution = 0.0
                
                # ml_signal에 메타데이터가 있는 경우 각 모델의 기여도 추출
                if 'rf_signal' in ml_signal and 'gru_signal' in ml_signal:
                    rf_signal = ml_signal.get('rf_signal', 0)
                    gru_signal = ml_signal.get('gru_signal', 0)
                    
                    # 각 모델이 최종 신호와 일치하는지 확인
                    if self._is_same_direction(rf_signal, combined_signal['signal']):
                        rf_contribution = trade_result
                    else:
                        rf_contribution = -trade_result * 0.5  # 패널티 경감
                        
                    if self._is_same_direction(gru_signal, combined_signal['signal']):
                        gru_contribution = trade_result
                    else:
                        gru_contribution = -trade_result * 0.5  # 패널티 경감
                
                else:
                    # 상세 정보가 없는 경우 머신러닝 신호 전체의 정확도로 평가
                    if ml_signal['signal'] == combined_signal['signal']:
                        rf_contribution = gru_contribution = trade_result
                    else:
                        rf_contribution = gru_contribution = -trade_result * 0.5
                
                # 각 모델의 수익 기여도 앙상블에 업데이트
                if hasattr(self.rf_model, 'name'):
                    self.ml_ensemble.update_profit(self.rf_model.name, rf_contribution * 100)  # 백분율로 변환
                    logger.info(f"RandomForest 모델 수익 기여도 업데이트: {rf_contribution:.4f}")
                    
                if hasattr(self.gru_model, 'name'):
                    self.ml_ensemble.update_profit(self.gru_model.name, gru_contribution * 100)
                    logger.info(f"GRU 모델 수익 기여도 업데이트: {gru_contribution:.4f}")
                    
                # 앙상블 모델의 학습 지표도 업데이트 (거래 결과를 정확도로 변환)
                if trade_result > 0:
                    accuracy = 0.7 + min(0.25, trade_result * 2)  # 최대 0.95
                    rmse = max(50, 200 - trade_result * 1000)  # 수익이 높을수록 RMSE 낮음
                else:
                    accuracy = max(0.4, 0.6 + trade_result)
                    rmse = min(300, 200 - trade_result * 500)  # 손실이 클수록 RMSE 높음
                
                if hasattr(self.rf_model, 'name'):
                    self.ml_ensemble.update_performance(
                        self.rf_model.name,
                        {"accuracy": accuracy, "f1": accuracy * 0.9},
                        model_type="direction"
                    )
                    
                if hasattr(self.gru_model, 'name'):
                    self.ml_ensemble.update_performance(
                        self.gru_model.name,
                        {"rmse": rmse, "mae": rmse * 0.8},
                        model_type="price"
                    )
                    
                logger.info(f"머신러닝 모델 성능 지표 업데이트 - 정확도: {accuracy:.4f}, RMSE: {rmse:.2f}")
                    
            except Exception as e:
                logger.error(f"머신러닝 모델 성능 업데이트 오류: {str(e)}")
                logger.error(traceback.format_exc())
                
    def _is_same_direction(self, model_signal, combined_signal):
        """모델 신호와 최종 신호의 방향성 일치 여부 확인"""
        # 연속 신호값 (-1.0 ~ 1.0)을 이산 신호(BUY/SELL/HOLD)와 비교
        if combined_signal == "BUY" and model_signal > 0:
            return True
        elif combined_signal == "SELL" and model_signal < 0:
            return True
        elif combined_signal == "HOLD" and abs(model_signal) < 0.3:
            return True
        return False

    def _get_technical_signal(self, data):
        """
        기술적 분석 지표를 기반으로 거래 신호를 생성합니다.
        :param data: 거래 데이터
        :return: 신호 딕셔너리
        """
        try:
            # 필요한 데이터 확인
            if len(data) < 2:
                logger.warning("기술적 분석을 위한 데이터가 충분하지 않습니다.")
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': '데이터 부족',
                    'metadata': {}
                }
            
            # 마지막 가격
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            
            # 추세 계산 (단기, 중기, 장기)
            short_term_trend = 0
            medium_term_trend = 0
            long_term_trend = 0
            
            # 단기 추세 (5일)
            if len(data) >= 5:
                short_term_trend = (current_price / data['close'].iloc[-5] - 1)
            
            # 중기 추세 (20일)
            if len(data) >= 20:
                medium_term_trend = (current_price / data['close'].iloc[-20] - 1)
            
            # 장기 추세 (50일)
            if len(data) >= 50:
                long_term_trend = (current_price / data['close'].iloc[-50] - 1)
            
            # 이동평균선 확인
            ema_signal = "HOLD"
            ema_confidence = 0.5
            
            # EMA 확인
            if 'ema12' in data.columns and 'ema26' in data.columns:
                ema12 = data['ema12'].iloc[-1]
                ema26 = data['ema26'].iloc[-1]
                
                # 골든 크로스 (단기 > 장기)
                if ema12 > ema26:
                    ema_signal = "BUY"
                    # 이동평균 차이가 클수록 신뢰도 증가
                    ema_confidence = min(0.85, 0.5 + abs(ema12/ema26 - 1) * 12)
                # 데드 크로스 (단기 < 장기)
                elif ema12 < ema26:
                    ema_signal = "SELL"
                    ema_confidence = min(0.85, 0.5 + abs(ema12/ema26 - 1) * 12)
            
            # RSI 확인
            rsi_signal = "HOLD"
            rsi_confidence = 0.5
            
            # RSI 확인 - 임계값 완화 (70->65, 30->35)
            if 'rsi14' in data.columns:
                rsi = data['rsi14'].iloc[-1]
                
                # 과매수 상태 (RSI > 65)
                if rsi > 65:
                    rsi_signal = "SELL"
                    rsi_confidence = min(0.9, 0.5 + (rsi - 65) / 35)
                # 과매도 상태 (RSI < 35)
                elif rsi < 35:
                    rsi_signal = "BUY"
                    rsi_confidence = min(0.9, 0.5 + (35 - rsi) / 35)
            
            # 추세 기반 신호
            trend_signal = "HOLD"
            trend_confidence = 0.5
            
            # 복합 추세 평가
            weighted_trend = (short_term_trend * 0.5 + medium_term_trend * 0.3 + long_term_trend * 0.2)
            
            # 임계값 낮춤 (0.02 -> 0.005)
            if weighted_trend > 0.005:  # 0.5% 이상 상승 추세
                trend_signal = "BUY"
                trend_confidence = min(0.9, 0.5 + weighted_trend * 20)
            elif weighted_trend < -0.005:  # 0.5% 이상 하락 추세
                trend_signal = "SELL"
                trend_confidence = min(0.9, 0.5 + abs(weighted_trend) * 20)
            
            # 볼륨 분석
            volume_signal = "HOLD"
            volume_confidence = 0.5
            
            if 'volume' in data.columns and len(data) > 5:
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].iloc[-5:].mean()
                
                # 볼륨 임계값 낮춤 (1.5 -> 1.2)
                # 볼륨이 평균보다 크고 가격이 상승하는 경우
                if current_volume > avg_volume * 1.2 and current_price > prev_price:
                    volume_signal = "BUY"
                    volume_confidence = min(0.8, 0.5 + (current_volume / avg_volume - 1) * 0.3)
                # 볼륨이 평균보다 크고 가격이 하락하는 경우
                elif current_volume > avg_volume * 1.2 and current_price < prev_price:
                    volume_signal = "SELL"
                    volume_confidence = min(0.8, 0.5 + (current_volume / avg_volume - 1) * 0.3)
            
            # 신호 결합
            signals = {
                'ema': ema_signal,
                'rsi': rsi_signal,
                'trend': trend_signal,
                'volume': volume_signal
            }
            
            confidences = {
                'ema': ema_confidence,
                'rsi': rsi_confidence,
                'trend': trend_confidence,
                'volume': volume_confidence
            }
            
            # 가중치 정의 - 더 많은 거래 신호를 생성하기 위해 조정
            weights = {
                'ema': 0.4,  # 원래 0.35에서 증가
                'rsi': 0.4,  # 원래 0.35에서 증가
                'trend': 0.15,  # 원래 0.2에서 감소
                'volume': 0.05  # 원래 0.1에서 감소
            }
            
            # 가중치가 적용된 신호 계산
            signal_values = {
                'BUY': 1,
                'HOLD': 0,
                'SELL': -1
            }
            
            weighted_sum = sum(signal_values[signals[key]] * confidences[key] * weights[key] for key in signals)
            
            # 최종 신호 결정 - 임계값 추가로 낮춤
            final_signal = "HOLD"
            if weighted_sum > 0.05:  # 원래 0.08에서 0.05로 낮춤
                final_signal = "BUY"
            elif weighted_sum < -0.05:  # 원래 -0.08에서 -0.05로 낮춤
                final_signal = "SELL"
            
            # 최종 신뢰도 계산
            final_confidence = min(0.95, abs(weighted_sum) * 4.0)  # 신뢰도 계수 증가 (3.5에서 4.0으로)
            
            logger.debug(f"기술적 분석 결과: {final_signal}, 신뢰도: {final_confidence:.2f}, 가중합: {weighted_sum:.2f}")
            
            # 신호 딕셔너리로 반환
            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'reason': f"기술적 분석 (EMA: {ema_signal}, RSI: {rsi_signal}, 추세: {trend_signal}, 볼륨: {volume_signal})",
                'metadata': {
                    'weighted_sum': weighted_sum,
                    'indicators': {
                        'ema': {'signal': ema_signal, 'confidence': ema_confidence},
                        'rsi': {'signal': rsi_signal, 'confidence': rsi_confidence, 'value': rsi if 'rsi14' in data.columns else None},
                        'trend': {'signal': trend_signal, 'confidence': trend_confidence, 
                                 'short': short_term_trend, 'medium': medium_term_trend, 'long': long_term_trend},
                        'volume': {'signal': volume_signal, 'confidence': volume_confidence}
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"기술적 분석 신호 생성 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'signal': "HOLD",
                'confidence': 0.0,
                'reason': f"기술적 분석 오류: {str(e)}",
                'metadata': {}
            }

    def _get_candlestick_signal(self, data):
        """
        캔들스틱 패턴 기반의 거래 신호를 생성합니다.
        
        Args:
            data (pd.DataFrame): 거래 데이터
            
        Returns:
            Dict[str, Any]: 신호 딕셔너리
        """
        # 기본 신호 (HOLD)
        default_signal = {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reason': '캔들스틱 패턴 없음',
            'metadata': {}
        }
        
        try:
            # 필요한 데이터 확인
            if len(data) < 2:
                return default_signal
                
            # 캔들스틱 패턴 컬럼이 있는지 확인
            pattern_columns = ['doji', 'bullish_engulfing', 'bearish_engulfing', 
                               'morning_star', 'evening_star']
            
            # 컬럼 이름 변환 (일부 구현에서 다른 이름 사용 가능성)
            available_columns = []
            for col in pattern_columns:
                if col in data.columns:
                    available_columns.append(col)
            
            # 사용 가능한 패턴 컬럼이 없으면 신호 없음
            if not available_columns:
                return default_signal
            
            # 최근 캔들스틱 패턴 확인 (마지막 행)
            latest_data = data.iloc[-1]
            
            # 매수 신호 패턴
            buy_patterns = ['doji', 'bullish_engulfing', 'morning_star']
            # 매도 신호 패턴
            sell_patterns = ['bearish_engulfing', 'evening_star']
            
            # 매수 신호 확인
            buy_detected = False
            buy_confidence = 0.0
            buy_reasons = []
            
            for pattern in buy_patterns:
                if pattern in available_columns:
                    try:
                        # 패턴이 숫자형으로 변환되었을 수 있으므로 유연하게 처리
                        value = latest_data[pattern]
                        if isinstance(value, (int, float, np.number)) and value > 0:
                            buy_detected = True
                            confidence = min(0.7, 0.5 + value * 0.1)  # 패턴 강도에 따른 신뢰도
                            buy_confidence = max(buy_confidence, confidence)
                            buy_reasons.append(f"{pattern} 패턴 감지")
                        elif isinstance(value, str) and value.lower() in ['true', 'yes', '1']:
                            buy_detected = True
                            buy_confidence = max(buy_confidence, 0.7)
                            buy_reasons.append(f"{pattern} 패턴 감지")
                    except:
                        # 오류가 발생하더라도 다른 패턴 계속 확인
                        pass
                        
            # 매도 신호 확인
            sell_detected = False
            sell_confidence = 0.0
            sell_reasons = []
            
            for pattern in sell_patterns:
                if pattern in available_columns:
                    try:
                        value = latest_data[pattern]
                        if isinstance(value, (int, float, np.number)) and value > 0:
                            sell_detected = True
                            confidence = min(0.7, 0.5 + value * 0.1)
                            sell_confidence = max(sell_confidence, confidence)
                            sell_reasons.append(f"{pattern} 패턴 감지")
                        elif isinstance(value, str) and value.lower() in ['true', 'yes', '1']:
                            sell_detected = True
                            sell_confidence = max(sell_confidence, 0.7)
                            sell_reasons.append(f"{pattern} 패턴 감지")
                    except:
                        pass
            
            # 최종 신호 결정
            # - 매수와 매도 신호가 모두 발생한 경우 신뢰도가 높은 쪽 선택
            # - 신호가 없으면 HOLD
            if buy_detected and sell_detected:
                if buy_confidence > sell_confidence:
                    return {
                        'signal': 'BUY',
                        'confidence': buy_confidence,
                        'reason': f"캔들스틱 매수 신호: {', '.join(buy_reasons)}",
                        'metadata': {'detected_patterns': buy_reasons}
                    }
                else:
                    return {
                        'signal': 'SELL',
                        'confidence': sell_confidence,
                        'reason': f"캔들스틱 매도 신호: {', '.join(sell_reasons)}",
                        'metadata': {'detected_patterns': sell_reasons}
                    }
            elif buy_detected:
                return {
                    'signal': 'BUY',
                    'confidence': buy_confidence,
                    'reason': f"캔들스틱 매수 신호: {', '.join(buy_reasons)}",
                    'metadata': {'detected_patterns': buy_reasons}
                }
            elif sell_detected:
                return {
                    'signal': 'SELL',
                    'confidence': sell_confidence,
                    'reason': f"캔들스틱 매도 신호: {', '.join(sell_reasons)}",
                    'metadata': {'detected_patterns': sell_reasons}
                }
            
            # 신호가 없으면 HOLD
            return default_signal
                
        except Exception as e:
            logger.error(f"캔들스틱 패턴 신호 생성 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return default_signal