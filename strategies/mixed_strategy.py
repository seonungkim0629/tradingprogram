"""
Mixed Time Frame Strategy Module

This module implements a strategy that switches between short-term and mid-term
strategies based on trade count.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import os
import pickle
from datetime import datetime

from strategies.base import BaseStrategy
from ensemble.ensemble import TradingEnsemble
from models import Signal, TradingSignal, standardize_signal, legacy_signal_to_trading_signal
from utils.logging import get_logger, log_execution
from utils.constants import TimeFrame, SignalType
from config import settings
from utils.state import save_trade_count, load_trade_count
from utils.metadata import create_standard_metadata, normalize_metadata
from utils.data_converters import prepare_data_for_strategy

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


class MixedTimeFrameStrategy(BaseStrategy):
    """Strategy that switches between short-term and mid-term strategies based on trade count"""
    
    def __init__(self, 
                market: str = "KRW-BTC", 
                name: Optional[str] = None, 
                parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy
        
        Args:
            market (str): Market to trade (e.g., KRW-BTC)
            name (Optional[str]): Strategy name
            parameters (Optional[Dict[str, Any]]): Strategy parameters
        """
        default_params = {
            'short_term_strategy': 'HarmonizingStrategy',
            'mid_term_strategy': 'HarmonizingStrategy',
            'short_term_params': {
                'timeframe': 'minute5',  # 단기 전략은 5분봉 기본 사용
                'trend_weight': 0.3,
                'ma_weight': 0.3,
                'rsi_weight': 0.4,
            },
            'mid_term_params': {
                'timeframe': 'day',  # 중기 전략은 일봉 사용
                'trend_weight': 0.5,
                'ma_weight': 0.3,
                'rsi_weight': 0.2,
            },
            'mid_term_frequency': 6,  # 6회 중 1회는 중기 전략 사용
            'trade_count': 0,
            'debug_mode': False
        }
        
        # Merge default parameters with provided parameters
        if parameters:
            for key, value in parameters.items():
                default_params[key] = value
        
        # 이름이 지정되지 않았으면 기본값 설정
        strategy_name = name or "MixedTimeFrameStrategy"
        
        super().__init__(market, strategy_name, default_params)
        
        # 데이터 저장
        self.data = {
            'day': None,
            'hour': None,
            'minute5': None,
            'minute1': None
        }
        
        # 저장된 trade_count 로드
        self._load_trade_count()
        
        # Initialize strategies
        self._init_strategies()
        
        self.logger.info(f"Initialized {self.name} strategy for {market} with trade_count={self.parameters['trade_count']}")
    
    def _load_trade_count(self) -> None:
        """트레이드 카운트 로드"""
        saved_count = load_trade_count(self.name)
        if saved_count is not None:
            self.parameters['trade_count'] = saved_count
            self.logger.info(f"[{self.name}] 저장된 거래 횟수 불러옴: {saved_count}")
        else:
            self.parameters['trade_count'] = 0
            self.logger.info(f"[{self.name}] 거래 횟수 초기화: 0")
    
    def _save_trade_count(self) -> None:
        """현재 트레이드 카운트 저장"""
        save_trade_count(self.name, self.parameters['trade_count'])
    
    def _init_strategies(self) -> None:
        """Initialize short-term and mid-term strategies"""
        try:
            # Short-term strategy
            short_term_params = self.parameters.get('short_term_params', {}).copy()
            short_term_params['is_backtest'] = True  # 백테스트 모드로 설정
            
            # Add market to parameters if not present
            if 'market' not in short_term_params:
                short_term_params['market'] = self.market
            
            self.short_term_strategy = TradingEnsemble(**short_term_params)
            self.short_term_strategy.name = "ShortTermStrategy"
            
            # Mid-term strategy
            mid_term_params = self.parameters.get('mid_term_params', {}).copy()
            mid_term_params['is_backtest'] = True  # 백테스트 모드로 설정
            
            # Add market to parameters if not present
            if 'market' not in mid_term_params:
                mid_term_params['market'] = self.market
            
            self.mid_term_strategy = TradingEnsemble(**mid_term_params)
            self.mid_term_strategy.name = "MidTermStrategy"
            
            self.logger.info("Successfully initialized short-term and mid-term strategies")
        except Exception as e:
            self.logger.error(f"Error initializing strategies: {str(e)}")
            raise
    
    def load_combined_data(self) -> bool:
        """
        최신 통합 데이터셋을 로드합니다.
        
        Returns:
            bool: 성공 여부
        """
        try:
            combined_dir = os.path.join(settings.DATA_DIR, 'combined')
            if not os.path.exists(combined_dir):
                self.logger.warning(f"통합 데이터 디렉토리가 존재하지 않습니다: {combined_dir}")
                return False
            
            # 가장 최신 파일 찾기
            combined_files = [f for f in os.listdir(combined_dir) if f.startswith(f"{self.market}_combined_") and f.endswith('.pkl')]
            if not combined_files:
                self.logger.warning(f"통합 데이터 파일을 찾을 수 없습니다: {combined_dir}")
                return False
            
            # 날짜 기준으로 정렬 (최신 순)
            combined_files.sort(reverse=True)
            latest_file = os.path.join(combined_dir, combined_files[0])
            
            # 파일 로드
            with open(latest_file, 'rb') as f:
                combined_data = pickle.load(f)
            
            # 데이터 확인 및 저장
            for timeframe, df in combined_data.items():
                if df is not None and not df.empty:
                    self.data[timeframe] = df
                    self.logger.info(f"{timeframe} 데이터 로드 완료: {len(df)}개 캔들")
                else:
                    self.logger.warning(f"{timeframe} 데이터가 비어있습니다")
            
            self.logger.info(f"통합 데이터셋 로드 완료: {latest_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"통합 데이터셋 로드 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def create_standard_metadata(self, strategy_type: str, timeframe: str) -> Dict[str, Any]:
        """
        표준화된 메타데이터 생성
        
        Args:
            strategy_type (str): 전략 타입 ('short_term' 또는 'mid_term')
            timeframe (str): 타임프레임
            
        Returns:
            Dict[str, Any]: 표준화된 메타데이터
        """
        metadata = {
            'strategy': self.name,
            'strategy_type': strategy_type,
            'timeframe': timeframe,
            'trade_count': self.parameters['trade_count']
        }
        
        return create_standard_metadata('signal', metadata)
    
    @log_execution
    def generate_signal(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Union[Dict[str, Any], TradingSignal]:
        """
        혼합 전략 신호 생성
        
        Args:
            data (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): 다양한 타임프레임 데이터 또는 단일 데이터프레임
            
        Returns:
            Union[Dict[str, Any], TradingSignal]: 생성된 신호
        """
        # trade_count 증가 및 저장
        self.parameters['trade_count'] += 1
        self._save_trade_count()
        
        # 이번 회차가 중장기 전략을 사용할 회차인지 확인
        mid_term_frequency = self.parameters.get('mid_term_frequency', 6)
        is_mid_term = (self.parameters['trade_count'] % mid_term_frequency == 0)
        
        if is_mid_term:
            self.logger.info(f"[{self.name}] Trade #{self.parameters['trade_count']}: 중장기 전략 사용")
        else:
            self.logger.info(f"[{self.name}] Trade #{self.parameters['trade_count']}: 단기 전략 사용")
        
        signal = None
        
        try:
            # 데이터 표준화
            data_dict = prepare_data_for_strategy(data)
            
            if is_mid_term and self.mid_term_strategy:
                # 중장기 전략 사용
                if TimeFrame.DAY in data_dict:
                    raw_signal = self.mid_term_strategy.generate_signal(data_dict[TimeFrame.DAY])
                    
                    if raw_signal:
                        # 기존 Signal 객체 처리
                        if isinstance(raw_signal, Signal):
                            signal = legacy_signal_to_trading_signal(raw_signal)
                        else:
                            # 딕셔너리 또는 TradingSignal 객체 처리
                            signal = standardize_signal(raw_signal)
                        
                        # 메타데이터 표준화 및 업데이트
                        strategy_metadata = self.create_standard_metadata('mid_term', TimeFrame.DAY)
                        signal.metadata.update(strategy_metadata)
                else:
                    self.logger.warning("일봉 데이터가 없어 중장기 전략을 실행할 수 없습니다")
            else:
                # 단기 전략 사용
                short_term_data = {}
                for tf in [TimeFrame.MINUTE1, TimeFrame.MINUTE5, TimeFrame.HOUR1]:
                    if tf in data_dict:
                        short_term_data[tf] = data_dict[tf]
                
                # 단기 전략에 단일 데이터프레임이 필요한 경우
                if short_term_data and self.short_term_strategy:
                    # 가능한 경우 단기 전략에 딕셔너리 전달
                    try:
                        raw_signal = self.short_term_strategy.generate_signal(short_term_data)
                    except (TypeError, ValueError):
                        # 실패한 경우, 첫 번째 데이터프레임만 전달
                        first_key = next(iter(short_term_data))
                        self.logger.info(f"단기 전략에 단일 데이터프레임 전달: {first_key}")
                        raw_signal = self.short_term_strategy.generate_signal(short_term_data[first_key])
                    
                    if raw_signal:
                        # 기존 Signal 객체 처리
                        if isinstance(raw_signal, Signal):
                            signal = legacy_signal_to_trading_signal(raw_signal)
                        else:
                            # 딕셔너리 또는 TradingSignal 객체 처리
                            signal = standardize_signal(raw_signal)
                        
                        # 메타데이터 표준화 및 업데이트
                        timeframes_str = ','.join(short_term_data.keys())
                        strategy_metadata = self.create_standard_metadata('short_term', timeframes_str)
                        signal.metadata.update(strategy_metadata)
                else:
                    self.logger.warning("단기 데이터가 없어 단기 전략을 실행할 수 없습니다")
        
        except Exception as e:
            self.logger.error(f"신호 생성 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return TradingSignal(signal_type=SignalType.HOLD, reason=f"오류 발생: {str(e)}")
        
        # 신호가 생성되지 않은 경우 HOLD 신호 반환
        if signal is None:
            return TradingSignal(signal_type=SignalType.HOLD, reason="신호 생성 실패")
            
        return signal
    
    def apply_risk_management(self, 
                           signal: Dict[str, Any], 
                           portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply risk management rules
        
        Args:
            signal (Dict[str, Any]): Trading signal
            portfolio (Dict[str, Any]): Portfolio state
            
        Returns:
            Dict[str, Any]: Modified signal with risk management applied
        """
        # Get the strategy type from metadata
        strategy_type = signal.get('metadata', {}).get('strategy_type', 'short_term')
        
        if strategy_type == 'mid_term':
            # Apply mid-term risk management
            return self.mid_term_strategy.apply_risk_management(signal, portfolio)
        else:
            # Apply short-term risk management
            return self.short_term_strategy.apply_risk_management(signal, portfolio)
    
    def calculate_position_size(self, 
                              signal: Dict[str, Any], 
                              available_balance: float) -> float:
        """
        Calculate position size based on the strategy type
        
        Args:
            signal (Dict[str, Any]): Trading signal
            available_balance (float): Available balance for trading
            
        Returns:
            float: Position size in base currency
        """
        # Get the strategy type from metadata
        strategy_type = signal.get('metadata', {}).get('strategy_type', 'short_term')
        
        if strategy_type == 'mid_term':
            # For mid-term strategy, use a larger position size (설정에 따라 조정)
            position_size = self.mid_term_strategy.calculate_position_size(signal, available_balance)
            self.logger.info(f"Mid-term position size: {position_size:.2f}")
            return position_size
        else:
            # For short-term strategy, use a smaller position size (설정에 따라 조정)
            position_size = self.short_term_strategy.calculate_position_size(signal, available_balance)
            self.logger.info(f"Short-term position size: {position_size:.2f}")
            return position_size
    
    def set_hourly_data(self, hourly_data: pd.DataFrame) -> None:
        """
        Set hourly data for both strategies
        
        Args:
            hourly_data (pd.DataFrame): Hourly market data
        """
        # Set hourly data for both strategies
        self.short_term_strategy.set_hourly_data(hourly_data)
        self.mid_term_strategy.set_hourly_data(hourly_data)
        
        # 시간봉 데이터 저장
        self.data['hour'] = hourly_data
        
        self.logger.info(f"Set hourly data for both strategies ({len(hourly_data)} rows)")
    
    def set_data(self, timeframe: str, data: pd.DataFrame) -> None:
        """
        특정 시간대 데이터 설정
        
        Args:
            timeframe (str): 시간대 ('day', 'hour', 'minute5', 'minute1')
            data (pd.DataFrame): 데이터
        """
        if data is not None and not data.empty:
            self.data[timeframe] = data
            self.logger.info(f"{timeframe} 데이터 설정 완료: {len(data)}개 캔들")
            
            # 단기 전략 사용 시간대와 동일하면 전략에도 설정
            if timeframe == self.parameters['short_term_params'].get('timeframe'):
                self.short_term_strategy.data = data
                self.logger.info(f"단기 전략에 {timeframe} 데이터 설정")
            
            # 중기 전략 사용 시간대와 동일하면 전략에도 설정
            if timeframe == self.parameters['mid_term_params'].get('timeframe'):
                self.mid_term_strategy.data = data
                self.logger.info(f"중기 전략에 {timeframe} 데이터 설정")
    
    def update_performance(self, trade_result: float) -> None:
        """
        Update strategy performance based on trade result
        
        Args:
            trade_result (float): Trade result (profit/loss)
        """
        # Get the current strategy type
        trade_count = self.parameters.get('trade_count', 0)
        mid_term_frequency = self.parameters.get('mid_term_frequency', 6)
        
        is_mid_term = ((trade_count - 1) % mid_term_frequency == 0)  # Use -1 because we already incremented
        
        # Update the appropriate strategy
        if is_mid_term:
            self.mid_term_strategy.update_performance(trade_result)
            self.logger.info(f"Updated mid-term strategy performance: {trade_result:.2f}")
        else:
            self.short_term_strategy.update_performance(trade_result)
            self.logger.info(f"Updated short-term strategy performance: {trade_result:.2f}")
            
        # Try to save performance to database
        try:
            from utils.database import save_performance_metric
            
            # 메타데이터 준비
            performance_metadata = {
                'trade_result': trade_result,
                'is_profit': trade_result >= 0
            }
            
            # 표준 메타데이터 생성
            metadata = self.create_standard_metadata('trade', 'day')
            
            # Save performance metrics
            save_performance_metric(
                strategy=self.name,
                metric_name="trade_profit" if trade_result >= 0 else "trade_loss",
                metric_value=abs(trade_result),
                period="trade"
            )
            
            # Save strategy-specific performance
            strategy_type = "mid_term" if is_mid_term else "short_term"
            save_performance_metric(
                strategy=f"{self.name}_{strategy_type}",
                metric_name="trade_profit" if trade_result >= 0 else "trade_loss",
                metric_value=abs(trade_result),
                period="trade"
            )
        except ImportError:
            self.logger.warning("데이터베이스 모듈을 불러올 수 없습니다.")
        except Exception as e:
            self.logger.error(f"성능 지표 저장 중 오류 발생: {str(e)}")
            
    def save(self, filepath: Optional[str] = None) -> str:
        """
        전략 상태 및 설정을 저장
        
        Args:
            filepath (Optional[str]): 저장할 파일 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        # 먼저 trade_count 저장
        self._save_trade_count()
        
        # 기본 저장 로직 호출
        return super().save(filepath) 