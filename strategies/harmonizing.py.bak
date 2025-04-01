"""
Harmonizing Strategy for Bitcoin Trading Bot

This module implements an ensemble strategy that combines multiple trading strategies
to generate more robust signals.
"""

import pandas as pd
import numpy as np
import random  # 무작위 모듈 추가
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import traceback
import os

from strategies.base import BaseStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi_strategy import RSIStrategy
from utils.logging import get_logger, log_execution

# EMA와 RSI 계산을 위한 커스텀 함수 추가
def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    지수이동평균(EMA)를 계산하는 함수
    
    Args:
        data (np.ndarray): 종가 데이터 배열
        period (int): EMA 기간
        
    Returns:
        np.ndarray: 계산된 EMA 값
    """
    ema = np.zeros_like(data)
    # 초기값 설정 (단순 평균으로)
    ema[0:period] = np.mean(data[0:period])
    
    # EMA 계산 (period+1부터 마지막까지)
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

def calculate_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """
    상대강도지수(RSI)를 계산하는 함수
    
    Args:
        data (np.ndarray): 종가 데이터 배열
        period (int): RSI 기간, 기본값 14
        
    Returns:
        np.ndarray: 계산된 RSI 값
    """
    # 가격 변화 계산
    delta = np.diff(data)
    delta = np.append(delta, 0)  # 배열 길이 유지를 위해 마지막에 0 추가
    
    # 상승과 하락 구분
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # 평균 상승/하락 계산 (초기값)
    avg_gain = np.zeros_like(data)
    avg_loss = np.zeros_like(data)
    
    # 첫 평균 계산
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    # 나머지 평균 계산 (지수평균)
    for i in range(period+1, len(data)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    # RSI 계산
    rs = np.zeros_like(data)
    rsi = np.zeros_like(data)
    
    for i in range(period, len(data)):
        if avg_loss[i] == 0:
            rs[i] = 100.0
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
        
        rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi

# 필요한 경우에만 머신러닝 모델 임포트
USE_ML_MODELS = True  # ML 모델 사용 여부 (True로 변경)

if USE_ML_MODELS:
    try:
        from models.random_forest import RandomForestDirectionModel
        from models.lstm import LSTMDirectionModel, LSTMPriceModel
        from models.ensemble import VotingEnsemble
        from data.processors import extract_features, split_historical_data
        from utils.data_utils import prepare_model_data
    except ImportError as e:
        get_logger(__name__).warning(f"머신러닝 모델 임포트 실패: {str(e)}")
        USE_ML_MODELS = False

# Initialize logger
logger = get_logger(__name__)


class HarmonizingStrategy(BaseStrategy):
    """
    Harmonizing Strategy
    
    An ensemble strategy that combines signals from TrendFollowing, MovingAverage,
    and RSI strategies with machine learning models to generate more robust trading signals.
    """
    
    def __init__(self, 
            market: str = "KRW-BTC",
            name: Optional[str] = None,
            parameters: Optional[Dict[str, Any]] = None,
            is_backtest: bool = False):  # is_backtest 파라미터 추가
        """
        Initialize Harmonizing strategy
        
        Args:
            market (str): Market to trade (e.g., KRW-BTC)
            name (Optional[str]): Strategy name
            parameters (Optional[Dict[str, Any]]): Strategy parameters
                - trend_weight (float): Weight for trend following strategy
                - ma_weight (float): Weight for moving average strategy
                - rsi_weight (float): Weight for RSI strategy
                - hourly_weight (float): Weight for hourly data analysis
                - ml_weight (float): Weight for machine learning models
                - use_ml_models (bool): Whether to use machine learning models
                - confidence_threshold (float): Minimum confidence threshold
                - voting_threshold (float): Threshold for voting-based decisions
                - adaptive_weights (bool): Whether to adapt weights based on market conditions
                - enable_risk_management (bool): Whether to enable risk management
                - stop_loss_pct (float): Stop loss percentage
                - take_profit_pct (float): Take profit percentage
                - disable_random_signals (bool): Whether to disable random signals
            is_backtest (bool): Whether the strategy is running in backtest mode
        """
        # 기본 매개변수 설정
        default_params = {
            'trend_weight': 0.30,       # 0.20에서 0.30으로 증가
            'ma_weight': 0.05,          # 0.15에서 0.05로 감소
            'rsi_weight': 0.25,         # 0.15에서 0.25로 증가
            'hourly_weight': 0.05,      # 0.10에서 0.05로 감소
            'ml_weight': 0.35,          # 0.40에서 0.35로 약간 감소
            'use_ml_models': USE_ML_MODELS,
            'confidence_threshold': 0.05,  # 신호 생성을 위한 신뢰도 임계값 (높을수록 더 확실한 신호만 생성)
            'voting_threshold': 0.10,     # 0.01에서 0.10으로 증가
            'adaptive_weights': True,
            'enable_risk_management': True,  # 위험 관리 활성화
            'stop_loss_pct': 0.05,          # 5% 손절 기준
            'take_profit_pct': 0.10,         # 10% 익절 기준
            'disable_random_signals': True   # 무작위 신호 생성 비활성화 (안정성 향상)
        }
        
        # 제공된 매개변수와 병합
        if parameters:
            default_params.update(parameters)
        
        super().__init__(market, name or "HarmonizingStrategy", default_params)
        
        # 백테스트 모드 설정
        self.is_backtest = is_backtest
        logger.info(f"전략 초기화: {'백테스트 모드' if is_backtest else '실시간 매매 모드'}")

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
        self.lstm_model = None 
        self.ml_ensemble = None
        
        # ML 모델이 활성화된 경우에만 초기화
        if self.parameters['use_ml_models'] and USE_ML_MODELS:
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
                  
        if self.parameters['use_ml_models'] and USE_ML_MODELS:
            weights_info += f", ML={self.parameters['ml_weight']:.2f}"
            
        logger.info(f"Initialized {self.name} with weights: {weights_info}")

    def init_ml_models(self):
        """
        머신러닝 모델 초기화 및 로드
        """
        if not self.parameters['use_ml_models']:
            logger.info("Machine learning models disabled in strategy configuration")
            return
            
        try:
            # 1. 모델 저장 위치 정의
            models_root_path = "data_storage/models"
            optimization_dir = "optimization_results"
            
            # 먼저 최적화 결과 디렉토리에서 최신 모델 확인
            best_rf_model_path = None
            best_lstm_model_path = None
            
            # 최적화 결과 디렉토리가 있는지 확인
            if os.path.exists(optimization_dir):
                logger.info(f"최적화 결과 디렉토리 발견: {optimization_dir}")
                # 최근 순으로 정렬된 서브 디렉토리 가져오기
                subdirs = [os.path.join(optimization_dir, d) for d in os.listdir(optimization_dir) 
                           if os.path.isdir(os.path.join(optimization_dir, d))]
                subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                # 최신 최적화 디렉토리에서 모델 파일 찾기
                if subdirs:
                    latest_dir = subdirs[0]
                    logger.info(f"최신 최적화 결과 디렉토리: {latest_dir}")
                    
                    # RF 모델 찾기
                    rf_results_file = os.path.join(latest_dir, 'rf_direction_optimization.json')
                    if os.path.exists(rf_results_file):
                        logger.info(f"RF 최적화 결과 파일 발견: {rf_results_file}")
                        best_rf_model_path = os.path.join(latest_dir, 'models', 'rf_direction')
                    
                    # GRU 모델 찾기
                    gru_results_file = os.path.join(latest_dir, 'gru_direction_optimization.json')
                    if os.path.exists(gru_results_file):
                        logger.info(f"GRU 최적화 결과 파일 발견: {gru_results_file}")
                        best_lstm_model_path = os.path.join(latest_dir, 'models', 'lstm_direction')
            
            # 랜덤 포레스트 모델 초기화
            self.rf_model = RandomForestDirectionModel(
                name="RF_Direction",
                version="1.0.0"
            )
            
            # LSTM 모델 초기화 
            self.lstm_model = LSTMDirectionModel(
                name="LSTM_Direction",
                version="1.0.0",
                sequence_length=30,
                units=[64, 32],     # [128, 64]에서 [64, 32]로 변경 (최적화)
                dropout_rate=0.4,   # 0.25에서 0.4로 증가 (과적합 방지)
                learning_rate=0.003,
                batch_size=32,
                epochs=100
            )
            
            # 모델이 이미 학습되었는지 확인 및 로드
            rf_loaded = False
            lstm_loaded = False
            
            try:
                # 최적화된 모델 로드 시도
                if best_rf_model_path and os.path.exists(best_rf_model_path):
                    logger.info(f"최적화된 RF 모델 로드 시도: {best_rf_model_path}")
                    self.rf_model = RandomForestDirectionModel.load(best_rf_model_path)
                    rf_loaded = self.rf_model.is_trained
                    logger.info(f"최적화된 RF 모델 로드 {'성공' if rf_loaded else '실패'}")
                
                if best_lstm_model_path and os.path.exists(best_lstm_model_path):
                    logger.info(f"최적화된 GRU 모델 로드 시도: {best_lstm_model_path}")
                    self.lstm_model = LSTMDirectionModel.load(best_lstm_model_path)
                    lstm_loaded = self.lstm_model.is_trained
                    logger.info(f"최적화된 GRU 모델 로드 {'성공' if lstm_loaded else '실패'}")
                
                # 기본 모델 로드 시도 (최적화 모델 로드 실패 시)
                if not rf_loaded:
                    logger.info("기본 RF 모델 로드 시도")
                    rf_path = os.path.join(models_root_path, "RF_Direction")
                    if os.path.exists(rf_path):
                        self.rf_model = RandomForestDirectionModel.load(rf_path)
                        rf_loaded = self.rf_model.is_trained
                        logger.info(f"기본 RF 모델 로드 {'성공' if rf_loaded else '실패'}")
                    else:
                        logger.warning(f"RF 모델 파일을 찾을 수 없습니다: {rf_path}")
                
                if not lstm_loaded:
                    logger.info("기본 GRU 모델 로드 시도")
                    lstm_path = os.path.join(models_root_path, "LSTM_Direction")
                    if os.path.exists(lstm_path):
                        self.lstm_model = LSTMDirectionModel.load(lstm_path)
                        lstm_loaded = self.lstm_model.is_trained
                        logger.info(f"기본 GRU 모델 로드 {'성공' if lstm_loaded else '실패'}")
                    else:
                        logger.warning(f"GRU 모델 파일을 찾을 수 없습니다: {lstm_path}")
                        
                        # LSTM 모델이 로드되지 않았으면 빌드라도 해둠
                        if not lstm_loaded:
                            logger.info("LSTM 모델 빌드 시도")
                            try:
                                # 입력 형태: (30, 10) - 일반적인 특성 수 가정
                                self.lstm_model.build_model(input_shape=(30, 10))
                                logger.info("LSTM 모델 빌드 성공")
                            except Exception as e:
                                logger.error(f"LSTM 모델 빌드 오류: {str(e)}")
            
            # 앙상블 모델 초기화
            self.ml_ensemble = VotingEnsemble(
                name="ML_Voting_Ensemble",
                version="1.0.0",
                voting='soft'
            )
            
            # 앙상블에 모델 추가
            if self.rf_model.is_trained:
                self.ml_ensemble.add_model(self.rf_model, weight=0.55)  # 0.6에서 0.55로 감소
                logger.info("Random Forest 모델이 앙상블에 추가되었습니다")
                
            if self.lstm_model.is_trained:
                # GRU+LayerNormalization 적용으로 LSTM 모델의 가중치를 증가시킴
                self.ml_ensemble.add_model(self.lstm_model, weight=0.45)  # 0.4에서 0.45로 증가
                logger.info("LSTM/GRU 모델이 앙상블에 추가되었습니다")
            
            if not self.ml_ensemble.models:
                logger.warning("앙상블에 추가된 모델이 없습니다. ML 신호 생성이 비활성화됩니다.")
            else:
                logger.info(f"머신러닝 앙상블이 {len(self.ml_ensemble.models)}개의 모델로 초기화되었습니다")
            
            except Exception as e:
                logger.warning(f"모델 로드 중 오류 발생: {str(e)}")
                logger.info("모델을 사용하기 전에 훈련이 필요합니다")
                
        except Exception as e:
            logger.error(f"머신러닝 모델 초기화 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            self.parameters['use_ml_models'] = False
            logger.warning("초기화 오류로 인해 머신러닝 모델이 비활성화되었습니다")

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
            if 'close' in data.columns:
                volatility = data['close'].pct_change().rolling(20).std().iloc[-1] if len(data) >= 20 else 0.02
            else:
                volatility = 0.02  # Default value
            
            # Calculate trend strength using recent EMA ratios
            has_emas = 'ema_short' in data.columns and 'ema_long' in data.columns
            if has_emas and len(data) > 0:
                short_ema = data['ema_short'].iloc[-1]
                long_ema = data['ema_long'].iloc[-1]
                trend_strength = abs(short_ema / long_ema - 1)
            elif 'ema12' in data.columns and 'ema26' in data.columns and len(data) > 0:
                # Try alternative EMA column names
                short_ema = data['ema12'].iloc[-1]
                long_ema = data['ema26'].iloc[-1]
                trend_strength = abs(short_ema / long_ema - 1)
            elif 'close' in data.columns and len(data) >= 20:
                # Fallback method using price
                safe_idx = min(len(data) - 1, 20)
                price_20d_ago = data['close'].iloc[-safe_idx] if safe_idx > 0 else data['close'].iloc[0]
                current_price = data['close'].iloc[-1]
                trend_strength = abs(current_price / price_20d_ago - 1) if price_20d_ago > 0 else 0.03
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
                logger.info("Market condition: Moderate - balanced approach with MA emphasis")
            
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
            Dict[str, Any]: 최종 신호 딕셔너리
        """
        # 백테스트 모드 여부 확인
        is_backtest = getattr(self, 'is_backtest', False)
        
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
                signal_weights[signal] += confidence
                signal_reasons[signal].append(f"{source}({confidence:.2f})")
        
        # 최대 가중치 신호 결정
        max_weight = max(signal_weights.values())
        
        # 신뢰도 임계값 확인 (가장 높은 가중치가 임계값보다 낮으면 HOLD)
        if max_weight < self.parameters.get('confidence_threshold', 0.05):
            final_signal = 'HOLD'
            final_confidence = max(0.5, max_weight)  # 기본 신뢰도는 최소 0.5
            reason = f"신뢰도 부족 (최대: {max_weight:.2f}, 임계값: {self.parameters.get('confidence_threshold', 0.05):.2f})"
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
                        # 여전히 동점이면 보수적으로 HOLD
                        final_signal = 'HOLD'
                        final_confidence = max_weight
                else:
                    final_signal = 'HOLD'
                    final_confidence = max_weight
            elif len(max_signals) > 1:
                # 동점이면서 'HOLD'가 없는 경우 (매수/매도 동점) - 보수적으로 HOLD
                final_signal = 'HOLD'
                final_confidence = max_weight
            else:
                # 명확한 최대 가중치 신호
                final_signal = max_signals[0]
                final_confidence = max_weight
            
            reason = f"가중 결합: {', '.join(signal_reasons[final_signal])}"
        
        # 결과 반환 (딕셔너리 형태로)
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'reason': reason
        }
    
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
    
    def _get_ml_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        머신러닝 모델을 사용하여 거래 신호 생성
        
        Args:
            data (pd.DataFrame): 일봉 마켓 데이터
            
        Returns:
            Dict[str, Any]: 머신러닝 모델의 신호 딕셔너리
        """
        # 기본 신호 설정
        default_signal = {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reason': '머신러닝 모델 사용 불가',
            'metadata': {}
        }
        
        # 전역 변수에서 ML 사용 불가능하거나 설정에서 비활성화된 경우
        if not USE_ML_MODELS or not self.parameters['use_ml_models']:
            logger.info("ML 모델이 비활성화되어 있습니다.")
            return default_signal
            
        # 앙상블 모델이 초기화되지 않았거나 모델이 없는 경우
        if self.ml_ensemble is None:
            logger.warning("ML 앙상블 모델이 초기화되지 않았습니다.")
            return default_signal
            
        if not hasattr(self.ml_ensemble, 'models') or not self.ml_ensemble.models:
            logger.warning("ML 앙상블에 모델이 없습니다.")
            return default_signal
            
        try:
            # 데이터 전처리 - 모델 입력 형태로 변환
            from utils.data_utils import prepare_model_data
            
            try:
            X, _ = prepare_model_data(data, n_steps=30, target_column='close')
            except Exception as e:
                logger.error(f"모델 데이터 준비 중 오류: {str(e)}")
                return default_signal
            
            # 모델이 충분히 많은 데이터를 가지고 있는지 확인
            if X is None or len(X) == 0:
                logger.warning("ML 모델 예측을 위한 데이터가 부족합니다.")
                return default_signal
                
            # 앙상블 모델로 예측
            latest_sample = X[-1:] if len(X.shape) == 2 else X[-1:].reshape(1, X.shape[1], X.shape[2])
            
            # 디버깅 정보 기록
            logger.info(f"ML 모델 입력 데이터 형태: {latest_sample.shape}")
            
            # 앙상블 모델의 클래스 확률 예측
            try:
                # predict_proba 메소드가 있는지 확인
                if not hasattr(self.ml_ensemble, 'predict_proba'):
                    logger.error("ML 앙상블 모델에 predict_proba 메소드가 없습니다.")
                    logger.error(f"사용 가능한 메소드: {[method for method in dir(self.ml_ensemble) if not method.startswith('_')]}")
                    return default_signal
                
                # 앙상블 모델에 포함된 모델 확인
                logger.info(f"앙상블 모델 내의 모델 수: {len(self.ml_ensemble.models) if hasattr(self.ml_ensemble, 'models') else 0}")
                for i, model in enumerate(self.ml_ensemble.models) if hasattr(self.ml_ensemble, 'models') else []:
                    logger.info(f"모델 {i+1}: {type(model).__name__}, 훈련됨: {getattr(model, 'is_trained', False)}")
                
            proba = self.ml_ensemble.predict_proba(latest_sample)
                
                # 확률 배열이 예상대로인지 확인
                if not isinstance(proba, np.ndarray) or proba.shape[0] == 0 or proba.shape[1] < 2:
                    logger.error(f"ML 모델의 예측 확률 형태가 잘못되었습니다: {proba.shape if isinstance(proba, np.ndarray) else type(proba)}")
                    return default_signal
                
                logger.info(f"앙상블 모델 예측 확률: {proba[0]}")
            except Exception as e:
                logger.error(f"ML 모델 예측 중 오류: {str(e)}")
                logger.error(traceback.format_exc())
                return default_signal
            
            # 방향 결정 (0: 하락, 1: 상승)
            direction = 1 if proba[0][1] > 0.5 else 0
            
            # 신뢰도 계산: 0.5에서 얼마나 멀리 떨어져 있는지
            confidence = abs(proba[0][1] - 0.5) * 2  # 0.5 -> 0, 1.0 -> 1.0
            
            # 신호 결정 - 신뢰도 임계값 상향 조정
            # LayerNormalization 적용으로 신호가 안정화되어 임계값 0.05 적용
            # (이전 BatchNormalization 사용 시보다 신호 변동성이 감소함)
            ml_confidence_threshold = self.parameters.get('confidence_threshold', 0.05)
            
            # 신호 결정 - 신뢰도가 임계값보다 높을 경우만 BUY/SELL 신호 생성
            if direction == 1 and confidence > ml_confidence_threshold:
                ml_signal = 'BUY'
                reason = f"ML 모델 상승 예측 (확률: {proba[0][1]:.4f}, 신뢰도: {confidence:.4f})"
            elif direction == 0 and confidence > ml_confidence_threshold:
                ml_signal = 'SELL'
                reason = f"ML 모델 하락 예측 (확률: {proba[0][0]:.4f}, 신뢰도: {confidence:.4f})"
            else:
                # 신뢰도가 낮은 경우 항상 HOLD 신호 생성
                ml_signal = 'HOLD'
                reason = f"ML 모델 신뢰도 부족 (상승확률: {proba[0][1]:.4f}, 신뢰도: {confidence:.4f} < {ml_confidence_threshold})"
                
            # 메타데이터 저장
            metadata = {
                'ensemble_proba': proba[0].tolist(),
                'confidence_threshold': ml_confidence_threshold
            }
            
            # 개별 모델 예측 확률 추가 (모델이 있는 경우만)
            if hasattr(self.rf_model, 'predict_proba') and hasattr(self.rf_model, 'is_trained') and self.rf_model.is_trained:
                try:
                    rf_proba = self.rf_model.predict_proba(latest_sample)
                    metadata['rf_confidence'] = rf_proba[0][1] if rf_proba.shape[1] > 1 else 0.5
                except Exception:
                    metadata['rf_confidence'] = 0.5
            else:
                metadata['rf_confidence'] = 0.5
                
            if hasattr(self.lstm_model, 'predict_proba') and hasattr(self.lstm_model, 'is_trained') and self.lstm_model.is_trained:
                try:
                    lstm_proba = self.lstm_model.predict_proba(latest_sample)
                    metadata['lstm_confidence'] = lstm_proba[0][1] if lstm_proba.shape[1] > 1 else 0.5
                except Exception:
                    metadata['lstm_confidence'] = 0.5
            else:
                metadata['lstm_confidence'] = 0.5
            
            logger.info(f"ML 신호: {ml_signal} (신뢰도: {confidence:.4f}). {reason}")
            
            return {
                'signal': ml_signal,
                'confidence': confidence,
                'reason': reason,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"ML 신호 생성 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return default_signal
    
    @log_execution
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        주어진 데이터를 분석하여 거래 신호를 생성합니다.
        :param data: 거래 데이터
        :return: 거래 신호 딕셔너리 (signal, confidence, reason, metadata)
        """
        try:
            if data is None or data.empty:
                logger.warning("데이터가 비어 있어 HOLD 신호를 반환합니다.")
                return {
                    'signal': 'HOLD',
                    'confidence': 0.5,
                    'reason': '데이터 없음',
                    'metadata': {}
                }
                
            current_price = data.iloc[-1]['close']
            
            # 각각의 신호 생성 시도
            signals = {}
            confidences = {}
            
            # 기술적 분석 신호 생성
            try:
                tech_result = self._get_technical_signal(data)
                if isinstance(tech_result, dict):
                    # 새로운 형식 (딕셔너리)
                    signal_tech = tech_result.get('signal', 'HOLD')
                    confidence_tech = tech_result.get('confidence', 0.5)
                    signals['technical'] = signal_tech
                    confidences['technical'] = confidence_tech * self.weights.get('technical', 0.3)  # 기본값 0.3 사용
                elif isinstance(tech_result, tuple) and len(tech_result) >= 2:
                    # 이전 형식 (튜플) 지원 - 하위 호환성
                    signal_tech, confidence_tech = tech_result
                    signals['technical'] = signal_tech
                    confidences['technical'] = confidence_tech * self.weights.get('technical', 0.3)  # 기본값 0.3 사용
                else:
                    logger.warning(f"기술적 분석 신호의 형식이 예상과 다릅니다: {type(tech_result)}")
                    signals['technical'] = "HOLD"
                    confidences['technical'] = 0
                
                logger.debug(f"기술적 분석 신호: {signal_tech}, 신뢰도: {confidence_tech}")
            except Exception as e:
                logger.warning(f"기술적 분석 신호 생성 오류: {str(e)}")
                logger.error(traceback.format_exc())
                signals['technical'] = "HOLD"
                confidences['technical'] = 0
            
            # ML 신호 생성
            try:
                ml_result = self._get_ml_signal(data)
                # _get_ml_signal이 딕셔너리나 튜플을 반환할 수 있으므로 처리
                if isinstance(ml_result, dict):
                    signal_ml = ml_result.get('signal', 'HOLD')
                    confidence_ml = ml_result.get('confidence', 0.5)
                elif isinstance(ml_result, tuple) and len(ml_result) >= 2:
                    signal_ml, confidence_ml = ml_result[0], ml_result[1]
                else:
                    signal_ml, confidence_ml = 'HOLD', 0.5
                    
                signals['ml'] = signal_ml
                confidences['ml'] = confidence_ml * self.weights.get('ml', 0.3)  # 기본값 0.3 사용
                logger.debug(f"ML 신호: {signal_ml}, 신뢰도: {confidence_ml}")
            except Exception as e:
                logger.warning(f"ML 신호 생성 오류: {str(e)}")
                signals['ml'] = "HOLD"
                confidences['ml'] = 0
            
            # 시계열 신호 생성
            try:
                # 백테스트 모드에서는 시간봉 데이터를 사용하지 않음
                is_backtest = getattr(self, 'is_backtest', False)
                # 백테스트 여부 확인을 위한 백업 메커니즘 (속성이 없는 경우)
                if not hasattr(self, 'is_backtest'):
                    # 백테스트 세션 ID가 있거나 mode가 'backtest'인 경우 백테스트로 간주
                    is_backtest = hasattr(self, 'backtest_session_id') or getattr(self, 'mode', '') == 'backtest'
                
                if is_backtest:
                    logger.debug("백테스트 모드에서는 시간봉 분석을 사용하지 않습니다.")
                    signal_hourly = 'HOLD'
                    confidence_hourly = 0.5
                    hourly_result = {
                        'signal': signal_hourly,
                        'confidence': confidence_hourly,
                        'reason': '백테스트 모드에서는 시간봉 분석 사용 안함',
            'metadata': {}
        }
                else:
                    # 실제 매매 모드에서만 시간봉 데이터 분석
                    hourly_result = self._analyze_hourly_data(data)
                
                # 결과 처리 (딕셔너리 또는 튜플 형태 모두 처리)
                if isinstance(hourly_result, dict):
                    signal_hourly = hourly_result.get('signal', 'HOLD')
                    confidence_hourly = hourly_result.get('confidence', 0.5)
                elif isinstance(hourly_result, tuple) and len(hourly_result) >= 2:
                    signal_hourly, confidence_hourly = hourly_result[0], hourly_result[1]
                else:
                    signal_hourly, confidence_hourly = 'HOLD', 0.5
                
                signals['hourly'] = signal_hourly
                
                # 백테스트 모드에서는 시간봉 데이터의 가중치를 0으로 설정
                if is_backtest:
                    confidences['hourly'] = 0
                else:
                    confidences['hourly'] = confidence_hourly * self.weights.get('hourly', 0.3)  # 기본값 0.3 사용
                
                logger.debug(f"시간별 데이터 신호: {signal_hourly}, 신뢰도: {confidence_hourly}")
            except Exception as e:
                logger.warning(f"시간별 데이터 분석 오류: {str(e)}")
                logger.error(traceback.format_exc())
                signals['hourly'] = "HOLD"
                confidences['hourly'] = 0
            
            # 외부 지표 신호 생성 (미구현된 경우 기본값 사용)
            try:
                if hasattr(self, '_get_external_signal'):
                    external_result = self._get_external_signal(data)
                    
                    # 결과 형식 확인 (딕셔너리 또는 튜플)
                    if isinstance(external_result, dict):
                        signal_ext = external_result.get('signal', 'HOLD')
                        confidence_ext = external_result.get('confidence', 0.5)
                    elif isinstance(external_result, tuple) and len(external_result) >= 2:
                        signal_ext, confidence_ext = external_result
                    else:
                        signal_ext, confidence_ext = 'HOLD', 0.5
                    
                    signals['external'] = signal_ext
                    # self.weights 딕셔너리에 'external' 키가 없을 경우를 대비
                    external_weight = self.weights.get('external', 0.1)  # 기본값 0.1 사용
                    confidences['external'] = confidence_ext * external_weight
                    logger.debug(f"외부 지표 신호: {signal_ext}, 신뢰도: {confidence_ext}")
                else:
                    signals['external'] = "HOLD"
                    confidences['external'] = 0
            except Exception as e:
                logger.warning(f"외부 지표 신호 생성 오류: {str(e)}")
                signals['external'] = "HOLD"
                confidences['external'] = 0
            
            # 신호 결합
            try:
                combined_result = self._combine_signals_weighted(signals, confidences)
                # _combine_signals_weighted가 튜플이나 딕셔너리를 반환할 수 있으므로 처리
                if isinstance(combined_result, tuple) and len(combined_result) >= 2:
                    final_signal, final_confidence = combined_result
                elif isinstance(combined_result, dict):
                    final_signal = combined_result.get('signal', 'HOLD')
                    final_confidence = combined_result.get('confidence', 0.5)
                else:
                    logger.warning(f"신호 결합 결과가 예상치 못한 형식입니다: {type(combined_result)}")
                    final_signal, final_confidence = "HOLD", 0.5
            except Exception as e:
                logger.error(f"신호 결합 중 오류: {str(e)}")
                final_signal, final_confidence = "HOLD", 0.5
            
            # 투표 결과 계산 (메타데이터용)
            vote_counts = {}
            for signal_type in ["BUY", "SELL", "HOLD"]:
                vote_counts[signal_type] = sum(1 for s in signals.values() if s == signal_type) / len(signals) if signals else 0
            
            # 최종 신호 구성
            signal_obj = {
                'signal': final_signal,
                'confidence': final_confidence,
                'reason': f"전략 결합: 기술적({signals['technical']}), ML({signals['ml']}), 시간봉({signals['hourly']})",
                'metadata': {
                    'prices': {'current': current_price},
                    'votes': vote_counts,
                    'individual_signals': signals,
                    'confidences': {k: round(v, 3) for k, v in confidences.items()}
                }
            }
            
            # 위험 관리 파라미터 추가
            signal_obj['metadata']['risk'] = {
                'stop_loss_pct': self.parameters.get('stop_loss_pct', 0.05),
                'take_profit_pct': self.parameters.get('take_profit_pct', 0.10),
            }
            
            # 데이터 기록
            try:
                timestamp = data.index[-1]
                self.signal_history.append({
                    "timestamp": timestamp,
                    "price": current_price,
                    "signal": final_signal,
                    "confidence": final_confidence,
                    "technical": signals.get('technical', "HOLD"),
                    "ml": signals.get('ml', "HOLD"),
                    "hourly": signals.get('hourly', "HOLD"),
                    "external": signals.get('external', "HOLD"),
                    "metadata": signal_obj['metadata']
                })
                
                # 신호 기록이 너무 길면 최근 100개만 유지
                if len(self.signal_history) > 100:
                    self.signal_history = self.signal_history[-100:]
            except Exception as e:
                logger.warning(f"신호 기록 저장 오류: {str(e)}")
            
            # 마지막으로 위험 관리 적용
            try:
                portfolio = {
                    'position': 0,  # 백테스트에서는 이 값이 나중에 채워질 것
                    'current_price': current_price,
                    'avg_price': 0  # 백테스트에서는 이 값이 나중에 채워질 것
                }
                signal_obj = self.apply_risk_management(signal_obj, portfolio)
            except Exception as e:
                logger.error(f"위험 관리 적용 오류: {str(e)}")
                logger.error(traceback.format_exc())
            
            return signal_obj
            
        except Exception as e:
            logger.error(f"신호 생성 중 치명적 오류: {str(e)}")
            logger.error(traceback.format_exc())
            # 오류 발생 시 기본 HOLD 신호 반환 (딕셔너리 형태)
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'reason': f'오류 발생: {str(e)}',
                'metadata': {}
            }

    def _analyze_hourly_data(self, daily_data: pd.DataFrame) -> Dict[str, Any]:
        """
        시간봉 데이터 분석 및 트레이딩 신호 생성
        
        Args:
            daily_data (pd.DataFrame): 일봉 마켓 데이터
            
        Returns:
            Dict[str, Any]: 시간봉 기반 신호 딕셔너리
        """
        # 기본 신호 설정
        default_signal = {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reason': '시간봉 데이터 없음',
            'metadata': {}
        }
        
        # 시간봉 데이터가 없으면 기본 신호 반환
        if self.hourly_data is None or self.hourly_data.empty:
            logger.warning("시간봉 데이터가 없어 시간봉 분석을 건너뜁니다.")
            return default_signal
        
        try:
            # 최소 데이터 요구사항 확인
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in self.hourly_data.columns]
            
            if missing_columns:
                logger.warning(f"시간봉 데이터에 필요한 컬럼이 없습니다: {missing_columns}")
                return default_signal
            
            # 현재 날짜(일봉 데이터의 마지막 날짜) 가져오기
            if daily_data is None or daily_data.empty:
                logger.warning("일봉 데이터가 없어 현재 날짜를 결정할 수 없습니다.")
                current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            else:
                try:
            current_date = daily_data.index[-1].strftime('%Y-%m-%d')
                except (IndexError, AttributeError) as e:
                    logger.warning(f"일봉 데이터에서 날짜 추출 오류: {e}")
                    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # 현재 날짜와 그 전날의 시간봉 데이터 가져오기 (최근 48시간)
            try:
            recent_hourly = self.hourly_data[self.hourly_data.index.strftime('%Y-%m-%d') >= (pd.to_datetime(current_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')]
            except Exception as e:
                logger.warning(f"최근 시간봉 데이터 추출 오류: {e}")
                # 일반적인 방법으로 실패한 경우 대체 방법 시도
                try:
                    # 마지막 48시간만 가져오기
                    recent_hourly = self.hourly_data.iloc[-48:]
                except Exception as e2:
                    logger.error(f"시간봉 데이터 백업 추출 실패: {e2}")
                    return default_signal
            
            if recent_hourly.empty:
                logger.warning("최근 시간봉 데이터가 없습니다.")
                return default_signal
            
            # --------- 시간봉 데이터 분석 로직 ---------
            
            # 최근 데이터 샘플 개수 결정 (최대 24시간, 최소 4시간)
            recent_hours = min(24, len(recent_hourly))
            recent_hours = max(4, recent_hours)  # 최소 4시간 데이터 필요
            
            # 마지막 시간들의 데이터 추출
            last_hours = recent_hourly.iloc[-recent_hours:]
            
            # 1. 시간별 움직임 계산 (상승 시간 수 - 하락 시간 수)
            try:
            price_changes = last_hours['close'].pct_change()
            up_hours = sum(1 for x in price_changes if x > 0)
            down_hours = sum(1 for x in price_changes if x < 0)
            hourly_trend = (up_hours - down_hours) / recent_hours  # -1 ~ 1 범위
            except Exception as e:
                logger.warning(f"시간봉 추세 계산 오류: {e}")
                hourly_trend = 0  # 기본값
            
            # 2. 시간별 거래량 분석 - 컬럼 존재 여부 확인 추가
            try:
                if 'volume' in last_hours.columns:
            avg_volume = last_hours['volume'].mean()
                    recent_volume = last_hours['volume'].iloc[-1] if not last_hours.empty else 0
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                else:
                    # volume 컬럼이 없는 경우 기본값 설정
                    logger.warning("시간봉 데이터에 거래량(volume) 정보가 없습니다.")
                    avg_volume = 0
                    recent_volume = 0
                    volume_ratio = 1.0
            except Exception as e:
                logger.warning(f"거래량 계산 오류: {e}")
                avg_volume = 0
                recent_volume = 0
                volume_ratio = 1.0
            
            # 3. 시간봉의 변동성 계산
            try:
                hourly_volatility = price_changes.std() if len(price_changes) > 0 else 0
            except Exception as e:
                logger.warning(f"변동성 계산 오류: {e}")
                hourly_volatility = 0.02  # 기본 변동성
            
            # 4. 추세 강도 계산 - 컬럼 존재 여부 확인 추가
            trend_strength = 0
            
            # EMA 컬럼 체크 및 계산
            try:
                if 'ema12' in last_hours.columns and 'ema26' in last_hours.columns:
                    short_ema = last_hours['ema12'].iloc[-1]
                    long_ema = last_hours['ema26'].iloc[-1]
                    trend_strength = (short_ema / long_ema - 1) if long_ema > 0 else 0
                elif 'ema_short' in last_hours.columns and 'ema_long' in last_hours.columns:
                    short_ema = last_hours['ema_short'].iloc[-1]
                    long_ema = last_hours['ema_long'].iloc[-1]
                    trend_strength = (short_ema / long_ema - 1) if long_ema > 0 else 0
            else:
                    # EMA 컬럼이 없는 경우 가격 기반으로 직접 계산
                    logger.info("EMA 컬럼이 없어 가격으로 추세 강도를 계산합니다.")
                    try:
                        # 데이터 길이 확인
                        if len(last_hours) >= 26:  # 장기 EMA(26) 계산에 필요한 최소 길이
                            # 커스텀 EMA 함수 사용
                            prices = last_hours['close'].values
                            short_ema = calculate_ema(prices, 12)[-1]
                            long_ema = calculate_ema(prices, 26)[-1]
                            trend_strength = (short_ema / long_ema - 1) if long_ema > 0 else 0
                        else:
                            # 데이터가 충분하지 않은 경우 단순 가격 비교로 대체
                            logger.warning(f"EMA 계산에 필요한 데이터가 부족합니다. 단순 가격 비교로 대체합니다. (데이터 길이: {len(last_hours)})")
                            if len(last_hours) >= 12:
                                # 12개 간격의 가격 비교
                                price_12h_ago = last_hours['close'].iloc[-min(12, len(last_hours))]
                                current_price = last_hours['close'].iloc[-1]
                                trend_strength = (current_price / price_12h_ago - 1) if price_12h_ago > 0 else 0
                            else:
                                # 가능한 가장 오래된 가격과 현재 가격 비교
                                first_price = last_hours['close'].iloc[0]
                                last_price = last_hours['close'].iloc[-1]
                                trend_strength = (last_price / first_price - 1) if first_price > 0 else 0
                    except Exception as e:
                        logger.warning(f"EMA 또는 추세 강도 계산 오류: {str(e)}")
                        trend_strength = 0  # 오류 발생 시 기본값
            except Exception as e:
                logger.warning(f"추세 강도 계산 오류: {str(e)}")
                trend_strength = 0  # 오류 발생 시 기본값
            
            # 5. RSI 계산 - 컬럼 존재 여부 확인 추가
            recent_rsi = 50  # 기본값 (중립)
            
            try:
                if 'rsi14' in last_hours.columns:
                    recent_rsi = last_hours['rsi14'].iloc[-1]
                    logger.info(f"last_hours['rsi14'] 컬럼에서 RSI 값을 추출했습니다: {recent_rsi}")
                elif 'rsi' in last_hours.columns:
                    recent_rsi = last_hours['rsi'].iloc[-1]
                    logger.info(f"last_hours['rsi'] 컬럼에서 RSI 값을 추출했습니다: {recent_rsi}")
                else:
                    # RSI 컬럼이 없는 경우 직접 계산
                    logger.info(f"RSI 컬럼이 없어 직접 계산합니다. 사용 가능한 컬럼: {last_hours.columns.tolist()}")
                    try:
                        # 커스텀 RSI 함수 사용
                        if len(last_hours) >= 14:  # RSI 계산에 필요한 최소 데이터 확인
                            close_values = last_hours['close'].values
                            logger.info(f"RSI 계산을 위한 종가 데이터: 길이={len(close_values)}, 처음 3개={close_values[:3]}, 마지막 3개={close_values[-3:]}")
                            rsi_values = calculate_rsi(close_values)
                            recent_rsi = rsi_values[-1]
                            logger.info(f"RSI 계산 성공: {recent_rsi}")
                        else:
                            logger.warning(f"RSI 계산에 필요한 데이터가 부족합니다. 기본값 50을 사용합니다. (데이터 길이: {len(last_hours)})")
                    except Exception as e:
                        logger.error(f"RSI 계산 중 오류 발생: {str(e)}")
                        logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"RSI 계산 및 추출 중 오류: {str(e)}")
                logger.error(traceback.format_exc())
                # 기본값 50 유지
            
            # ----- 신호 생성 로직 -----
            
            # 신호 임계값 설정
            hourly_signal = 'HOLD'
            confidence = 0.5
            
            # 강한 상승 신호
            if (hourly_trend > 0.3 and trend_strength > 0.01 and volume_ratio > 1.2 and recent_rsi < 70):
                hourly_signal = 'BUY'
                confidence = 0.7
                reason = f"시간봉 강한 상승세 (추세: {hourly_trend:.2f}, 볼륨: {volume_ratio:.2f}배, RSI: {recent_rsi:.2f})"
                
            # 강한 하락 신호
            elif (hourly_trend < -0.3 and trend_strength < -0.01 and volume_ratio > 1.2 and recent_rsi > 30):
                hourly_signal = 'SELL'
                confidence = 0.7
                reason = f"시간봉 강한 하락세 (추세: {hourly_trend:.2f}, 볼륨: {volume_ratio:.2f}배, RSI: {recent_rsi:.2f})"
            
            # 과매수 상태 (매도 신호)
            elif recent_rsi > 75 and volume_ratio > 1.1:
                hourly_signal = 'SELL'
                confidence = 0.65
                reason = f"시간봉 과매수 상태 (RSI: {recent_rsi:.2f}, 볼륨: {volume_ratio:.2f}배)"
            
            # 과매도 상태 (매수 신호)
            elif recent_rsi < 25 and volume_ratio > 1.1:
                hourly_signal = 'BUY'
                confidence = 0.65
                reason = f"시간봉 과매도 상태 (RSI: {recent_rsi:.2f}, 볼륨: {volume_ratio:.2f}배)"
            
            # 약한 신호들
            elif hourly_trend > 0.2 and recent_rsi < 60:
                hourly_signal = 'BUY'
                confidence = 0.55
                reason = f"시간봉 약한 상승 신호 (추세: {hourly_trend:.2f}, RSI: {recent_rsi:.2f})"
            
            elif hourly_trend < -0.2 and recent_rsi > 40:
                hourly_signal = 'SELL'
                confidence = 0.55
                reason = f"시간봉 약한 하락 신호 (추세: {hourly_trend:.2f}, RSI: {recent_rsi:.2f})"
                
            else:
                # 뚜렷한 신호 없음
                hourly_signal = 'HOLD'
                confidence = 0.5
                reason = f"시간봉 명확한 신호 없음 (추세: {hourly_trend:.2f}, 거래량: {volume_ratio:.2f}배, RSI: {recent_rsi:.2f})"
            
            # 메타데이터 저장
            metadata = {
                'hourly_trend': hourly_trend,
                'volume_ratio': volume_ratio,
                'hourly_volatility': hourly_volatility,
                'trend_strength': trend_strength,
                'recent_rsi': recent_rsi,
                'analyzed_hours': recent_hours,
                'available_indicators': {
                    'has_volume': 'volume' in last_hours.columns,
                    'has_ema12': 'ema12' in last_hours.columns,
                    'has_ema26': 'ema26' in last_hours.columns,
                    'has_rsi14': 'rsi14' in last_hours.columns
                }
            }
            
            logger.info(f"시간봉 분석 결과: {hourly_signal} (신뢰도: {confidence:.2f}) - {reason}")
            
            return {
                'signal': hourly_signal,
                'confidence': confidence,
                'reason': reason,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"시간봉 데이터 분석 오류: {str(e)}")
            self.logger.error(traceback.format_exc())  # 스택 트레이스 추가
            # 오류 발생 시 HOLD 신호 반환
            return default_signal
    
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
        Update performance tracking for strategy adaptation
        
        Args:
            trade_result (float): Result of the trade (profit/loss)
        """
        if trade_result == 0:
            return
            
        # Update overall strategy performance
        for strategy, signal in self.last_signals.items():
            if strategy == 'combined':
                continue
                
            if signal['signal'] == self.last_signals['combined']['signal']:
                # Strategy was correct (aligned with the combined signal)
                self.strategy_performance[strategy] *= (1 + abs(trade_result) * 0.1)
            else:
                # Strategy was incorrect (disagreed with the combined signal)
                self.strategy_performance[strategy] *= (1 - abs(trade_result) * 0.05)
        
        logger.info(f"Updated strategy performance: {self.strategy_performance}")

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
                    ema_confidence = min(0.8, 0.5 + abs(ema12/ema26 - 1) * 10)
                # 데드 크로스 (단기 < 장기)
                elif ema12 < ema26:
                    ema_signal = "SELL"
                    ema_confidence = min(0.8, 0.5 + abs(ema12/ema26 - 1) * 10)
            
            # RSI 확인
            rsi_signal = "HOLD"
            rsi_confidence = 0.5
            
            # RSI 확인
            if 'rsi14' in data.columns:
                rsi = data['rsi14'].iloc[-1]
                
                # 과매수 상태 (RSI > 70)
                if rsi > 70:
                    rsi_signal = "SELL"
                    rsi_confidence = min(0.9, 0.5 + (rsi - 70) / 30)
                # 과매도 상태 (RSI < 30)
                elif rsi < 30:
                    rsi_signal = "BUY"
                    rsi_confidence = min(0.9, 0.5 + (30 - rsi) / 30)
            
            # 추세 기반 신호
            trend_signal = "HOLD"
            trend_confidence = 0.5
            
            # 복합 추세 평가
            weighted_trend = (short_term_trend * 0.5 + medium_term_trend * 0.3 + long_term_trend * 0.2)
            
            if weighted_trend > 0.02:  # 2% 이상 상승 추세
                trend_signal = "BUY"
                trend_confidence = min(0.9, 0.5 + weighted_trend * 10)
            elif weighted_trend < -0.02:  # 2% 이상 하락 추세
                trend_signal = "SELL"
                trend_confidence = min(0.9, 0.5 + abs(weighted_trend) * 10)
            
            # 볼륨 분석
            volume_signal = "HOLD"
            volume_confidence = 0.5
            
            if 'volume' in data.columns and len(data) > 5:
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].iloc[-5:].mean()
                
                # 볼륨이 평균보다 크고 가격이 상승하는 경우
                if current_volume > avg_volume * 1.5 and current_price > prev_price:
                    volume_signal = "BUY"
                    volume_confidence = min(0.8, 0.5 + (current_volume / avg_volume - 1) * 0.2)
                # 볼륨이 평균보다 크고 가격이 하락하는 경우
                elif current_volume > avg_volume * 1.5 and current_price < prev_price:
                    volume_signal = "SELL"
                    volume_confidence = min(0.8, 0.5 + (current_volume / avg_volume - 1) * 0.2)
            
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
            
            # 가중치 정의
            weights = {
                'ema': 0.3,
                'rsi': 0.3,
                'trend': 0.3,
                'volume': 0.1
            }
            
            # 가중치가 적용된 신호 계산
            signal_values = {
                'BUY': 1,
                'HOLD': 0,
                'SELL': -1
            }
            
            weighted_sum = sum(signal_values[signals[key]] * confidences[key] * weights[key] for key in signals)
            
            # 최종 신호 결정
            final_signal = "HOLD"
            if weighted_sum > 0.15:
                final_signal = "BUY"
            elif weighted_sum < -0.15:
                final_signal = "SELL"
            
            # 최종 신뢰도 계산
            final_confidence = min(0.9, abs(weighted_sum) * 3)
            
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
            logger.error(f"기술적 분석 신호 생성 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'signal': "HOLD",
                'confidence': 0.0,
                'reason': f"기술적 분석 오류: {str(e)}",
                'metadata': {}
            }