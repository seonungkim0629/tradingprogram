"""
Backtesting Engine for Bitcoin Trading Bot

This module provides backtesting functionality to evaluate trading strategies
using historical data without making actual trades.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import os
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from config import settings
from utils.logging import get_logger, log_execution
from utils.evaluation import calculate_metrics, perform_statistical_test, calculate_monte_carlo_confidence, analyze_market_correlation
from strategies.base import BaseStrategy
from data.collectors import get_historical_data
from data.indicators import calculate_all_indicators
from data.processors import extend_with_synthetic_data

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 파일 핸들러 추가
if not logger.handlers:
    fh = logging.FileHandler('logs/backtest.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

@dataclass
class BacktestResult:
    """Class to store backtesting results"""
    strategy_name: str
    market: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    returns: float
    trades: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate additional metrics after initialization"""
        if not self.metrics and len(self.equity_curve) > 0:
            self.calculate_metrics()
    
    def calculate_metrics(self) -> None:
        """Calculate performance metrics"""
        self.metrics = calculate_metrics(
            self.equity_curve, 
            self.trades, 
            self.daily_returns
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'strategy_name': self.strategy_name,
            'market': self.market,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'returns': self.returns,
            'trades': self.trades,
            'metrics': self.metrics,
            'parameters': self.parameters
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save results to JSON file"""
        if filepath is None:
            # Generate default filename based on strategy and date
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                settings.BACKTEST_RESULTS_DIR,
                f"backtest_{self.strategy_name}_{timestamp}.json"
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.debug(f"Saved backtest results to {filepath}")
        return filepath


class BacktestEngine:
    """
    백테스팅 엔진 클래스
    
    Arguments:
        config (Dict[str, Any], optional): 설정 딕셔너리
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        백테스팅 엔진 초기화
        
        Arguments:
            config (Dict[str, Any], optional): 설정 딕셔너리
        """
        self.config = config or {}
        
        # 기본 설정값 설정
        self.initial_balance = self.config.get('initial_balance', 1000000.0)
        self.commission_rate = self.config.get('fee', 0.0005)  # 'fee' 설정값을 가져오지만 변수명은 commission_rate로 유지
        self.slippage = self.config.get('slippage', 0.0)
        self.strict_mode = self.config.get('strict', False)  # 엄격 모드 설정
        
        # 시장 및 기간 설정
        self.market = self.config.get('market', 'KRW-BTC')
        self.timeframe = self.config.get('timeframe', 'day')
        
        # 기간 설정
        days = self.config.get('days', 365)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=days)
        
        # 시작일과 종료일이 명시적으로 제공된 경우 사용
        if 'start_date' in self.config:
            try:
                self.start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
            except ValueError:
                self.logger.warning(f"잘못된 시작일 형식: {self.config['start_date']}. 기본값 사용.")
        
        if 'end_date' in self.config:
            try:
                self.end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')
            except ValueError:
                self.logger.warning(f"잘못된 종료일 형식: {self.config['end_date']}. 기본값 사용.")
        
        # 데이터 수집 설정
        self.data_source = self.config.get('data_source', 'upbit')
        
        # 전략 초기화
        self.strategy_name = self.config.get('strategy', 'HarmonizingStrategy')
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized backtest engine with {self.config} KRW, {self.commission_rate*100:.2f}% commission, {self.slippage*100:.1f}% slippage")
    
    @log_execution
    def run(self, strategy: Optional[BaseStrategy] = None, 
            start_date: Optional[datetime] = None, 
            end_date: Optional[datetime] = None,
            data: Optional[pd.DataFrame] = None,
            use_daily_only: bool = True) -> BacktestResult:
        """
        백테스트 실행
        
        Arguments:
            strategy (BaseStrategy, optional): 백테스트할 전략 객체
            start_date (datetime, optional): 백테스트 시작일
            end_date (datetime, optional): 백테스트 종료일
            data (pd.DataFrame, optional): 사전 준비된 시장 데이터
            use_daily_only (bool): 일봉 데이터만 사용할지 여부
            
        Returns:
            BacktestResult: 백테스트 결과 객체
        """
        # 파라미터가 제공되지 않은 경우 self의 값 사용
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        
        # 전략이 제공되지 않은 경우 전략 초기화
        if strategy is None:
            try:
                from main import initialize_strategy
                strategy = initialize_strategy(self.config)
            except Exception as e:
                self.logger.error(f"전략 초기화 실패: {str(e)}")
                raise ValueError(f"전략 초기화 실패: {str(e)}")
        
        self.logger.info(f"Starting backtest for {strategy.name} from "
                        f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get data if not provided
        if data is None:
            if use_daily_only:
                # In backtest mode, use daily data only with synthetic extension if needed
                self.logger.info("Using daily data only for backtest")
                data = get_historical_data(
                    ticker=strategy.market,
                    days=1000,  # 1000일 데이터 요청
                    extend_with_synthetic=True  # 합성 데이터로 확장 활성화
                )
            else:
                # 시간봉 데이터도 포함하는 로직 (여기서는 사용하지 않음)
                pass
                
            # Filter data for backtest period
            data = data[data.index >= start_date]
            data = data[data.index <= end_date]
            
            # Add indicators
            data = calculate_all_indicators(data)
        
        # 데이터가 이미 필터링되어 있어야 하므로 다시 필터링하지 않음
        if len(data) == 0:
            self.logger.error(f"No data available for backtest period {start_date} to {end_date}")
            raise ValueError(f"No data available for backtest period {start_date} to {end_date}")
        
        # Initialize portfolio state
        portfolio = {
            'cash': self.initial_balance,  # 전체 자금을 현금으로 시작 (포지션 없음)
            'position': 0.0,  # 초기 포지션 없음
            'position_value': 0.0,  # 초기 포지션 가치 없음
            'total_value': self.initial_balance,
            'trades': [],
            'equity_curve': [self.initial_balance],
            'daily_returns': [],
            'last_signal': None,          # 마지막 신호 저장
            'last_signal_time': None,     # 마지막 신호 시간 저장
            'consecutive_signals': 0,      # 연속된 같은 신호 카운트
            'last_trade_time': None       # 마지막 거래 시간 저장
        }
        
        # 진행상황 표시 변수
        total_days = len(data)
        progress_interval = max(1, total_days // 10)  # 10% 단위로 진행상황 표시
        processed_days = 0
        
        # Iterate through each data point - 한 번만 실행되도록 수정
        prev_date = None
        for timestamp, row in data.iterrows():
            # 중요: 무한 반복 방지를 위해 현재 날짜가 end_date 이후인지 확인
            if timestamp > end_date:
                self.logger.warning(f"Skipping date {timestamp} as it is past the end date {end_date}")
                continue
                
            current_price = row['close']
            
            # 진행 상황 표시 (10% 단위)
            processed_days += 1
            if processed_days % progress_interval == 0:
                self.logger.info(f"Backtest progress: {processed_days * 100 // total_days}% ({processed_days}/{total_days} days)")
            
            # Fetch data up to this timestamp (Sliding Window)
            current_data_slice = data.loc[:timestamp]

            # --- 수정 시작 ---
            # TradingEnsemble이 사용하는 특성만 선택
            if hasattr(strategy, 'expected_features') and strategy.expected_features:
                # 마지막 행만 예측에 사용하고, 예상 특성만 선택
                try:
                    expected_features = getattr(strategy, 'expected_features', None)
                    if not expected_features:
                         self.logger.warning("전략 객체에 expected_features 속성이 없거나 비어 있습니다. 모든 특성을 사용합니다.")
                         # 중복 컬럼 제거는 수행
                         slice_for_pred = current_data_slice.iloc[-1:]
                         duplicates = slice_for_pred.columns[slice_for_pred.columns.duplicated()].tolist()
                         if duplicates:
                             self.logger.warning(f"원본 데이터 슬라이스에서 중복 컬럼 발견: {duplicates}. 첫 번째 항목만 유지합니다.")
                             features_for_prediction = slice_for_pred.loc[:, ~slice_for_pred.columns.duplicated()]
                         else:
                             features_for_prediction = slice_for_pred
                         
                    else:
                         # 1. 예상 특성 목록 정리 (중복 제거)
                         unique_expected_features = pd.Index(expected_features).drop_duplicates().tolist()
                         
                         # 2. 현재 데이터 슬라이스 준비 (마지막 행) 및 중복 제거
                         slice_for_pred = current_data_slice.iloc[-1:]
                         duplicates_in_slice = slice_for_pred.columns[slice_for_pred.columns.duplicated()].tolist()
                         if duplicates_in_slice:
                             self.logger.warning(f"원본 데이터 슬라이스(인덱스: {timestamp})에서 중복 컬럼 발견: {duplicates_in_slice}. 첫 번째 항목만 유지합니다.")
                             slice_no_duplicates = slice_for_pred.loc[:, ~slice_for_pred.columns.duplicated()]
                         else:
                             slice_no_duplicates = slice_for_pred
                         
                         actual_columns = slice_no_duplicates.columns.tolist()
                         
                         # 3. 예상 특성과 실제 특성 비교 및 로깅
                         set_expected = set(unique_expected_features)
                         set_actual = set(actual_columns)
                         
                         missing_expected = list(set_expected - set_actual)
                         extra_actual = list(set_actual - set_expected)
                         
                         if missing_expected:
                             self.logger.warning(f"데이터 슬라이스에 예상된 특성 중 일부가 누락됨: {missing_expected}")
                         if extra_actual:
                             # 너무 많은 예상 외 컬럼은 로그 길이를 위해 일부만 표시
                             log_extra = extra_actual[:10] + ['...'] if len(extra_actual) > 10 else extra_actual
                             self.logger.warning(f"데이터 슬라이스에 예상치 못한 특성이 포함됨: {log_extra}")
                             
                         # 4. 필요한 특성만 선택 (unique_expected_features에 있는 것만)
                         # slice_no_duplicates에서 unique_expected_features에 있는 컬럼만 선택
                         cols_to_select = [col for col in unique_expected_features if col in actual_columns]
                         if not cols_to_select:
                              self.logger.error("예상된 특성 중 사용 가능한 컬럼이 하나도 없습니다!")
                              features_selected = pd.DataFrame(columns=unique_expected_features, index=slice_no_duplicates.index) # 빈 DF 생성
                         else:
                              features_selected = slice_no_duplicates[cols_to_select]
                         
                         # 5. 최종 reindex (순서 맞추고, 누락된 예상 컬럼은 0으로 채움)
                         features_for_prediction = features_selected.reindex(columns=unique_expected_features, fill_value=0)
                         
                         # 6. 최종 형태 검증 (선택 사항이지만 디버깅에 유용)
                         final_shape = features_for_prediction.shape
                         if final_shape[1] != len(unique_expected_features):
                              self.logger.error(f"최종 특성 수 불일치 오류! 예상: {len(unique_expected_features)}, 실제: {final_shape[1]}. 로직 검토 필요.")
                         # else:
                         #      self.logger.debug(f"특성 준비 완료. 최종 형태: {final_shape}")

                except KeyError as e:
                     self.logger.error(f"expected_features 선택 중 KeyError 발생: {e}. 사용 가능한 컬럼: {current_data_slice.columns.tolist()}", exc_info=True)
                     features_for_prediction = pd.DataFrame() # 빈 데이터프레임 전달하여 오류 처리 유도
                except Exception as e:
                    self.logger.error(f"특성 선택 중 예상치 못한 오류 발생: {e}", exc_info=True)
                    features_for_prediction = pd.DataFrame() # 빈 데이터프레임 전달
            else:
                # expected_features가 없으면 일단 마지막 행 전체 사용 (경고 로깅)
                self.logger.warning("전략 객체에 expected_features 속성이 없거나 비어있습니다. 마지막 행의 모든 특성을 사용합니다.")
                features_for_prediction = current_data_slice.iloc[-1:]
            # --- 수정 끝 ---

            # Generate signal using the data slice
            try:
                # 필터링된 특성 데이터(마지막 행)를 전달
                # generate_signal 구현에 따라 딕셔너리 또는 DataFrame 직접 전달 필요
                # 현재 TradingEnsemble은 generate_signal에서 predict를 호출하며, predict는 DataFrame 입력을 처리함
                signal = strategy.generate_signal(features_for_prediction) # DataFrame 직접 전달
            except Exception as e: # generate_signal 내부 오류 처리 강화
                self.logger.error(f"신호 생성 중 예외 발생: {e}", exc_info=True)
                signal = {'signal': 'HOLD', 'reason': f"신호 생성 오류: {e}"} # 오류 발생 시 HOLD

            # 신호 유효성 검증 추가
            if not isinstance(signal, dict) or 'signal' not in signal:
                 self.logger.error(f"잘못된 신호 형식 수신: {signal}. HOLD로 처리합니다.")
                 signal = {'signal': 'HOLD', 'reason': '잘못된 신호 형식'}
                
                 # 현재 포지션 가치 업데이트만 수행하고 다음 데이터로 진행 (들여쓰기 수정)
                 portfolio['position_value'] = portfolio['position'] * current_price
                 portfolio['total_value'] = portfolio['cash'] + portfolio['position_value']
                 portfolio['equity_curve'].append(portfolio['total_value'])

                 # 진행 상황 표시 (들여쓰기 수정)
                 if processed_days % progress_interval == 0:
                     progress_pct = processed_days / total_days * 100
                     self.logger.info(f"백테스트 진행: {progress_pct:.1f}% ({processed_days}/{total_days})")
                 
                 # 하루 종료 후 일별 수익률 계산 (들여쓰기 수정)
                 if prev_date is not None and timestamp.date() != prev_date.date():
                     daily_return = (portfolio['total_value'] / prev_day_value) - 1
                     portfolio['daily_returns'].append(daily_return)
                     
                 prev_date = timestamp # (들여쓰기 수정)
                 prev_day_value = portfolio['total_value'] # (들여쓰기 수정)
                 
                 # 다음 데이터로 진행 (들여쓰기 수정)
                 continue
                
            # 연속된 같은 신호 추적
            if signal['signal'] == portfolio['last_signal']:
                portfolio['consecutive_signals'] += 1
            else:
                portfolio['consecutive_signals'] = 1  # 새로운 신호, 카운터 리셋
                
            portfolio['last_signal'] = signal['signal']
            portfolio['last_signal_time'] = timestamp
            
            # Apply risk management
            signal = strategy.apply_risk_management(signal, portfolio)
            
            # Calculate daily returns (if date changed)
            if prev_date is not None and timestamp.date() != prev_date:
                # Calculate daily return
                daily_return = (portfolio['total_value'] / portfolio['equity_curve'][-1]) - 1
                portfolio['daily_returns'].append(daily_return)
                prev_date = timestamp.date()
            elif prev_date is None:
                prev_date = timestamp.date()
            
            # Execute trade based on signal - 조건 완화
            if signal['signal'] == 'BUY' and portfolio['position'] < self.initial_balance * 0.5:  # 포지션이 초기 자금의 50% 미만일 때 매수 가능
                # 마지막 거래 이후 일정 시간이 지났는지 확인 (중복 거래 방지)
                can_trade = True
                if portfolio['last_trade_time'] is not None:
                    # 같은 날에는 중복 거래 방지
                    if portfolio['last_trade_time'].date() == timestamp.date():
                        if portfolio['consecutive_signals'] < 2:  # 최소 2번 연속 같은 신호가 와야 거래 (조건 완화)
                            can_trade = False
                            self.logger.debug(f"중복 거래 방지: 같은 날 이미 거래가 있고 연속 신호가 부족함 ({portfolio['consecutive_signals']})")
                
                if can_trade:
                    # Calculate position size - 보유 현금의 50%만 사용 (위험 분산)
                    position_size = min(
                        strategy.calculate_position_size(
                            signal=signal,
                            available_balance=portfolio['cash']
                        ),
                        portfolio['cash'] * 0.5 / current_price  # 현재 보유 현금의 50%만 사용
                    )
                    
                    # Apply slippage and commission
                    execution_price = current_price * (1 + self.slippage)
                    position_cost = position_size * execution_price
                    commission = position_cost * self.commission_rate
                    
                    if position_cost + commission <= portfolio['cash']:
                        # Execute buy
                        portfolio['position'] += position_size  # 기존 포지션에 추가
                        portfolio['cash'] -= (position_cost + commission)
                        portfolio['position_value'] = portfolio['position'] * current_price
                        portfolio['total_value'] = portfolio['cash'] + portfolio['position_value']
                        portfolio['last_trade_time'] = timestamp
                        
                        # 거래 로그 수준 향상 (DEBUG에서 INFO로)
                        self.logger.debug(f"BUY: {position_size:.8f} {strategy.market} @ {execution_price:.2f} = "
                                        f"{position_cost:.2f} (commission: {commission:.2f})")
                        
                        # 데이터베이스에 거래 기록 저장
                        try:
                            from utils.database import save_trade
                            
                            # 메타데이터 준비
                            metadata = {
                                'commission': commission,
                                'balance_after': portfolio['cash'],
                                'timestamp': timestamp.isoformat(),
                                'trade_count': len(portfolio['trades']) + 1
                            }
                            
                            # 데이터베이스에 저장
                            save_trade(
                                trade_type='BUY',
                                ticker=strategy.market,
                                price=execution_price,
                                amount=position_size,
                                total=position_cost,
                                reason=signal.get('reason', 'Strategy signal'),
                                strategy=strategy.name,
                                confidence=signal.get('confidence', 0.5),
                                metadata=metadata
                            )
                        except ImportError:
                            self.logger.warning("거래 데이터베이스 저장 실패: utils.database 모듈을 가져올 수 없습니다.")
                        except Exception as e:
                            self.logger.error(f"거래 데이터베이스 저장 오류: {str(e)}")
                        
                        # Add trade to history
                        trade = {
                            'timestamp': timestamp.isoformat(),
                            'type': 'BUY',
                            'price': execution_price,
                            'amount': position_size,
                            'cost': position_cost,
                            'commission': commission,
                            'balance_after': portfolio['cash'],
                            'position_size': position_size,
                            'reason': signal.get('reason', 'Strategy signal')
                        }
                        portfolio['trades'].append(trade)
                    else:
                        self.logger.debug(f"자금 부족으로 매수 실패: 필요 금액 {position_cost + commission:.2f}, 가용 자금 {portfolio['cash']:.2f}")
            
            elif signal['signal'] == 'SELL' and portfolio['position'] > 0:
                # 마지막 거래 이후 일정 시간이 지났는지 확인 (중복 거래 방지)
                can_trade = True
                if portfolio['last_trade_time'] is not None:
                    # 같은 날에는 중복 거래 방지
                    if portfolio['last_trade_time'].date() == timestamp.date():
                        if portfolio['consecutive_signals'] < 2:  # 최소 2번 연속 같은 신호가 와야 거래 (조건 완화)
                            can_trade = False
                            self.logger.debug(f"중복 거래 방지: 같은 날 이미 거래가 있고 연속 신호가 부족함 ({portfolio['consecutive_signals']})")
                
                if can_trade:
                    # 일부만 매도 (보유 포지션의 50%)
                    position_size = portfolio['position'] * 0.5  # 보유 포지션의 50%만 매도
                    
                    # Apply slippage and commission
                    execution_price = current_price * (1 - self.slippage)
                    position_value = position_size * execution_price
                    commission = position_value * self.commission_rate
                    
                    # position_cost 변수 정의 - 매입 단가를 계산
                    # 최근 BUY 거래의 가격을 찾거나, 없으면 현재 시장 가격으로 계산
                    position_cost = 0
                    for past_trade in reversed(portfolio['trades']):  # 가장 최근 거래부터 역순으로 확인
                        if past_trade['type'] == 'BUY':
                            position_cost = position_size * past_trade['price']  # 가장 최근 매수 가격으로 계산
                            break
                    
                    # 매수 기록이 없으면 현재 포지션의 평균 매입가를 추정
                    if position_cost == 0:
                        # 현재가의 95%를 매입가로 가정 (보수적인 수익 추정)
                        position_cost = position_size * (current_price * 0.95)
                        self.logger.warning(f"매수 기록을 찾을 수 없어 현재가 기준으로 매입가를 추정합니다: {current_price * 0.95:.2f}")
                    
                    # Execute sell
                    portfolio['cash'] += (position_value - commission)
                    portfolio['position'] -= position_size  # 매도한 만큼만 포지션 감소
                    portfolio['position_value'] = portfolio['position'] * current_price  # 남은 포지션 가치 업데이트
                    portfolio['total_value'] = portfolio['cash'] + portfolio['position_value']
                    portfolio['last_trade_time'] = timestamp
                    
                    # 거래 로그 수준 향상 (DEBUG에서 INFO로)
                    self.logger.debug(f"SELL: {position_size:.8f} {strategy.market} @ {execution_price:.2f} = "
                                    f"{position_value:.2f} (commission: {commission:.2f})")
                    
                    # 데이터베이스에 거래 기록 저장
                    try:
                        from utils.database import save_trade
                        
                        # 메타데이터 준비
                        metadata = {
                            'commission': commission,
                            'balance_after': portfolio['cash'],
                            'timestamp': timestamp.isoformat(),
                            'trade_count': len(portfolio['trades']) + 1,
                            'profit_loss': position_value - position_cost,
                            'profit_loss_pct': (position_value - position_cost) / position_cost
                        }
                        
                        # 데이터베이스에 저장
                        save_trade(
                            trade_type='SELL',
                            ticker=strategy.market,
                            price=execution_price,
                            amount=position_size,
                            total=position_value,
                            reason=signal.get('reason', 'Strategy signal'),
                            strategy=strategy.name,
                            confidence=signal.get('confidence', 0.5),
                            metadata=metadata
                        )
                    except ImportError:
                        self.logger.warning("거래 데이터베이스 저장 실패: utils.database 모듈을 가져올 수 없습니다.")
                    except Exception as e:
                        self.logger.error(f"거래 데이터베이스 저장 오류: {str(e)}")
                    
                    # Add trade to history
                    trade = {
                        'timestamp': timestamp.isoformat(),
                        'type': 'SELL',
                        'price': execution_price,
                        'amount': position_size,
                        'value': position_value,
                        'commission': commission,
                        'balance_after': portfolio['cash'],
                        'profit_loss': position_value - position_cost,  # 실제 수익/손실 금액
                        'returns': (position_value - position_cost) / position_cost,  # 수익률로 이름 통일 (기존 profit_loss_pct)
                        'reason': signal.get('reason', 'Strategy signal')
                    }
                    portfolio['trades'].append(trade)
            
            else:
                # Update position value
                portfolio['position_value'] = portfolio['position'] * current_price
                portfolio['total_value'] = portfolio['cash'] + portfolio['position_value']
            
            # Update equity curve
            portfolio['equity_curve'].append(portfolio['total_value'])
        
        # Final portfolio valuation
        final_balance = portfolio['total_value']
        returns = (final_balance / self.initial_balance) - 1  # 수익률
        
        # Create result object
        result = BacktestResult(
            strategy_name=strategy.name,
            market=strategy.market,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            returns=returns,  # 수익률
            trades=portfolio['trades'],
            daily_returns=portfolio['daily_returns'],
            equity_curve=portfolio['equity_curve'],
            parameters=strategy.parameters
        )
        
        # Calculate performance metrics
        result.calculate_metrics()
        
        # Print results summary
        monthly_return = result.metrics.get('monthly_return', 0)
        sharpe_ratio = result.metrics.get('sharpe_ratio', 0)
        max_drawdown = result.metrics.get('max_drawdown', 0)
        win_rate = result.metrics.get('win_rate', 0)
        
        # 거래 타입별 횟수 계산
        buy_count = sum(1 for trade in portfolio['trades'] if trade['type'] == 'BUY')
        sell_count = sum(1 for trade in portfolio['trades'] if trade['type'] == 'SELL')
        hold_count = len(data) - buy_count - sell_count
        
        self.logger.info(f"\n{'=' * 40}")
        self.logger.info(f"백테스트 결과: {strategy.name} - {strategy.market}")
        self.logger.info(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"{'=' * 40}")
        self.logger.info(f"초기 잔고: {self.initial_balance:,.0f} KRW")
        self.logger.info(f"최종 잔고: {final_balance:,.0f} KRW")
        self.logger.info(f"총 수익률: {returns:.2%}")
        self.logger.info(f"월 수익률: {monthly_return:.2%}")
        self.logger.info(f"연간 수익률: {result.metrics.get('annualized_return', 0):.2%}")
        self.logger.info(f"변동성: {result.metrics.get('volatility', 0):.2%}")
        self.logger.info(f"샤프 지수: {sharpe_ratio:.2f}")
        self.logger.info(f"최대 낙폭: {max_drawdown:.2%}")
        self.logger.info(f"승률: {win_rate:.2%}")
        self.logger.info(f"총 거래 횟수: {len(portfolio['trades'])} (매수: {buy_count}, 매도: {sell_count}, 홀드: {hold_count})")
        self.logger.info(f"완료된 거래 쌍: {len(portfolio['trades'])//2}")
        self.logger.info(f"{'=' * 40}\n")
        
        # 백테스트 결과를 검증
        self.validate_backtest_results(result, data)
        
        return result
    
    def validate_backtest_results(self, result: BacktestResult, data: pd.DataFrame = None) -> None:
        """
        백테스트 결과의 신뢰성 검증 및 경고
        
        Args:
            result (BacktestResult): 검증할 백테스트 결과
            data (pd.DataFrame): 사용된 시장 데이터
        """
        logger.info("백테스트 결과 검증 시작...")
        
        # 0. 거래 없이 수익이 발생한 경우 - 강제 수정
        if len(result.trades) == 0 and result.returns != 0:
            logger.error(f"❗ 심각한 오류: 거래가 없는데 수익률({result.returns:.2%})이 발생했습니다. 수익률을 0으로 강제 리셋합니다.")
            # 수익률 강제 리셋
            result.returns = 0.0
            result.final_balance = result.initial_balance
            
            # 주식 곡선 리셋
            result.equity_curve = [result.initial_balance for _ in result.equity_curve]
            
            # 지표 리셋
            if 'monthly_return' in result.metrics:
                result.metrics['monthly_return'] = 0.0
            if 'annualized_return' in result.metrics:
                result.metrics['annualized_return'] = 0.0
            if 'sharpe_ratio' in result.metrics:
                result.metrics['sharpe_ratio'] = 0.0
        
        # 1. 비정상적으로 높은 수익률 확인
        if result.returns > 1.0:  # 100% 이상
            logger.warning(f"경고: 비정상적으로 높은 수익률 ({result.returns:.2%})이 감지되었습니다. 데이터나 전략에 문제가 있을 수 있습니다.")
        
        # 2. 비정상적으로 낮은 변동성 확인
        if result.metrics.get('volatility', 0) < 0.0001:
            logger.warning(f"경고: 변동성이 비정상적으로 낮습니다 ({result.metrics.get('volatility', 0):.6f}). 데이터나 계산에 문제가 있을 수 있습니다.")
        
        # 3. 승률과 수익 비율이 일치하지 않는지 확인
        if result.metrics.get('win_rate', 0) > 0.5 and result.returns < 0:
            logger.warning(f"경고: 승률({result.metrics.get('win_rate', 0):.2%})이 높지만 총 수익률({result.returns:.2%})이 음수입니다. 손실 규모가 승리 규모보다 큽니다.")
        elif result.metrics.get('win_rate', 0) < 0.5 and result.returns > 0:
            logger.warning(f"경고: 승률({result.metrics.get('win_rate', 0):.2%})이 낮지만 총 수익률({result.returns:.2%})이 양수입니다. 큰 승리가 작은 손실을 보상하고 있습니다.")
        
        # 4. 거래 빈도 확인
        days_count = (result.end_date - result.start_date).days
        min_expected_trades = max(days_count // 30, 1)  # 최소 월 1회 거래 예상
        
        if len(result.trades) < min_expected_trades:
            logger.warning(f"경고: 거래 횟수가 예상보다 적습니다 ({len(result.trades)} < {min_expected_trades}). 백테스트 기간({days_count}일)에 비해 거래가 충분하지 않습니다.")
        
        # 5. 통계적 유의성 검사
        try:
            if len(result.daily_returns) >= 30:  # 충분한 샘플이 있을 때만
                stats_test = perform_statistical_test(result.daily_returns)
                
                if not stats_test.get('significant', False):
                    logger.warning(f"경고: 백테스트 결과가 통계적으로 유의하지 않습니다 (p-value: {stats_test.get('p_value', 'N/A')}). 결과가 우연에 의한 것일 수 있습니다.")
                else:
                    logger.info(f"통계적 유의성 검사: 결과가 유의합니다 (p-value: {stats_test.get('p_value', 'N/A')})")
        except Exception as e:
            logger.warning(f"통계적 유의성 검사 중 오류 발생: {str(e)}")
        
        # 6. 몬테카를로 시뮬레이션을 통한 신뢰 구간 검증
        try:
            mc_results = calculate_monte_carlo_confidence(result.daily_returns, num_simulations=1000)
            
            # 결과 확인
            return_ci = mc_results.get('return_ci', (0, 0))
            below_zero_prob = mc_results.get('return_below_zero_probability', 0)
            
            if below_zero_prob > 0.4:
                logger.warning(f"경고: 몬테카를로 시뮬레이션 결과, 손실 확률이 높습니다 ({below_zero_prob:.2%}).")
                
            logger.info(f"몬테카를로 시뮬레이션 95% 신뢰구간: {return_ci[0]:.2%} ~ {return_ci[1]:.2%}")
        except Exception as e:
            logger.warning(f"몬테카를로 시뮬레이션 중 오류 발생: {str(e)}")
        
        # 7. 시장 수익률과 전략 수익률 비교
        try:
            if data is not None:
                # 시장 수익률 추출 (일별 종가 기준)
                market_returns = data['close'].pct_change().dropna().tolist()
                
                # 길이 맞추기 (둘 중 더 짧은 길이로)
                min_length = min(len(result.daily_returns), len(market_returns))
                if min_length > 0:
                    strategy_returns = result.daily_returns[:min_length]
                    market_returns = market_returns[:min_length]
                    
                    # 상관관계 분석 수행
                    market_analysis = analyze_market_correlation(strategy_returns, market_returns)
                    
                    if market_analysis.get('correlation', 0) > 0.8:
                        logger.warning(f"경고: 전략이 시장과 높은 상관관계({market_analysis.get('correlation', 0):.2f})를 보입니다. 단순 추종 전략일 수 있습니다.")
                    
                    alpha = market_analysis.get('alpha', 0)
                    if alpha < 0:
                        logger.warning(f"경고: 알파가 음수({alpha:.4f})입니다. 시장 대비 초과 수익을 내지 못하고 있습니다.")
                else:
                    logger.warning("일별 수익률 데이터가 충분하지 않아 시장 상관관계 분석을 건너뜁니다.")
        except Exception as e:
            logger.warning(f"시장 상관관계 분석 중 오류 발생: {str(e)}")
        
        logger.info("백테스트 검증 완료")

    @log_execution
    def optimize(self, strategy: BaseStrategy,
                start_date: datetime,
                end_date: datetime,
                parameter_grid: Dict[str, List[Any]],
                metric: str = 'sharpe_ratio',
                data: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy (BaseStrategy): Strategy to optimize
            start_date (datetime): Backtest start date
            end_date (datetime): Backtest end date
            parameter_grid (Dict[str, List[Any]]): Parameters to test
            metric (str): Optimization metric
            data (pd.DataFrame, optional): Pre-loaded data for optimization
            
        Returns:
            Tuple[Dict[str, Any], BacktestResult]: Best parameters and result
        """
        self.logger.info(f"Starting parameter optimization for {strategy.name}")
        
        # Get data if not provided
        if data is None:
            self.logger.info("No data provided, fetching historical data...")
            # Get data once for all backtests
            data = get_historical_data(
                ticker=strategy.market,
                days=365,  # Use 365 days of data for backtest
                indicators=True,
                verbose=True,
                source="upbit",
                split=False,  # 데이터 분할 없이 전체 데이터 사용
                extend_with_synthetic=False  # 합성 데이터 사용하지 않음
            )
            
            # Filter data for backtest period
            data = data[data.index >= start_date]
            data = data[data.index <= end_date]
            
            # Add indicators if not already added
            if 'rsi_14' not in data.columns:
                data = calculate_all_indicators(data)
        else:
            self.logger.info(f"Using provided data with {len(data)} rows")
        
        # Generate all parameter combinations
        import itertools
        keys = parameter_grid.keys()
        values = parameter_grid.values()
        param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Track best result
        best_result = None
        best_params = None
        best_metric_value = float('-inf')
        
        # Test each parameter combination
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Set parameters
            strategy.set_parameters(params)
            
            # Run backtest
            result = self.run(strategy, start_date, end_date, data=data.copy())
            
            # Get metric value
            metric_value = result.metrics.get(metric, float('-inf'))
            
            # Update if better
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_params = params
                best_result = result
                
                self.logger.info(f"New best result: {metric}={metric_value}, params={params}")
        
        self.logger.info(f"Optimization completed. Best {metric}: {best_metric_value}")
        self.logger.info(f"Best parameters: {best_params}")
        
        # Reset strategy to best parameters
        strategy.set_parameters(best_params)
        
        return best_params, best_result

    @log_execution
    def analyze_market_conditions(self, strategy: BaseStrategy, market_data: pd.DataFrame, initial_balance: float = 10000000.0) -> Dict[str, Any]:
        """
        시장 조건별 전략 성과 분석 (상승장, 하락장, 횡보장)
        
        Args:
            strategy (BaseStrategy): 분석할 전략
            market_data (pd.DataFrame): 시장 데이터
            initial_balance (float): 초기 자금
            
        Returns:
            Dict[str, Any]: 시장 조건별 분석 결과
        """
        # 원본 데이터 복사하여 작업
        data = market_data.copy()
        
        # 날짜 정렬 (오름차순)
        data = data.sort_index()
        
        # 수익률 계산을 위한 컬럼 추가
        if 'return' not in data.columns:
            data['return'] = data['close'].pct_change()
            
        # 이동평균 추가 (20일, 50일, 200일)
        if 'ma_20' not in data.columns:
            data['ma_20'] = data['close'].rolling(window=20).mean()
        if 'ma_50' not in data.columns:
            data['ma_50'] = data['close'].rolling(window=50).mean()
        if 'ma_200' not in data.columns:
            data['ma_200'] = data['close'].rolling(window=200).mean()
        
        # NaN 값 제거 (이동평균 계산으로 인한 초기 NaN)
        data = data.dropna()
        
        if len(data) < 100:
            return {"error": "시장 조건 분석을 위한 충분한 데이터가 없습니다 (최소 100일 필요)"}
        
        try:
            # ===== 1. 시장 상태 분류 =====
            
            # 상승장 정의: 20일 이동평균 > 50일 이동평균 > 200일 이동평균
            # 하락장 정의: 20일 이동평균 < 50일 이동평균 < 200일 이동평균
            # 횡보장 정의: 그 외 (이동평균선이 얽혀있는 상태)
            
            data['market_condition'] = 'sideways'  # 기본값은 횡보장
            
            # 상승장 조건
            bull_condition = (data['ma_20'] > data['ma_50']) & (data['ma_50'] > data['ma_200'])
            data.loc[bull_condition, 'market_condition'] = 'bull'
            
            # 하락장 조건
            bear_condition = (data['ma_20'] < data['ma_50']) & (data['ma_50'] < data['ma_200'])
            data.loc[bear_condition, 'market_condition'] = 'bear'
            
            # ===== 2. 각 시장 조건별 백테스트 수행 =====
            
            # 무한 반복을 방지하기 위해 시장 조건별 테스트를 단순화함
            conditions = {
                'bull_market': data[data['market_condition'] == 'bull'],
                'bear_market': data[data['market_condition'] == 'bear'],
                'sideways_market': data[data['market_condition'] == 'sideways']
            }
            
            results = {}
            
            for condition_name, condition_data in conditions.items():
                # 데이터가 충분한지 확인 (최소 15일)
                if len(condition_data) < 15:
                    results[condition_name] = {
                        'count': len(condition_data),
                        'message': f"충분한 {condition_name} 데이터가 없습니다 (최소 15일 필요)"
                    }
                    continue
                
                # 날짜가 연속적이지 않아도 되도록 인덱스 재설정
                # condition_data = condition_data.reset_index()
                
                # 단일 기간 테스트만 수행 (조각 테스트 삭제)
                period_start = condition_data.index.min()
                period_end = condition_data.index.max()
                
                total_days = (period_end - period_start).days
                
                # 전략 테스트 시 새로운 백테스트를 수행하지 않고 간단히 통계만 계산
                returns = condition_data['return'].dropna()
                
                # 기본 통계 계산
                results[condition_name] = {
                    'count': len(condition_data),
                    'period_days': total_days,
                    'avg_return': returns.mean(),
                    'total_return': (1 + returns).prod() - 1,
                    'volatility': returns.std(),
                    'positive_days': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
                }
            
            # ===== 3. 전략의 일관성 평가 =====
            
            # 기본 일관성 계산: 각 시장 조건에서 양수 수익률을 내는지
            bull_return = results.get('bull_market', {}).get('total_return', -1)
            bear_return = results.get('bear_market', {}).get('total_return', -1)
            sideways_return = results.get('sideways_market', {}).get('total_return', -1)
            
            consistency_score = 0
            
            # 상승장에서 양수 수익률 (가중치 0.4)
            if bull_return > 0:
                consistency_score += 0.4
            
            # 하락장에서 양수 수익률 (가중치 0.4)
            if bear_return > 0:
                consistency_score += 0.4
            
            # 횡보장에서 양수 수익률 (가중치 0.2)
            if sideways_return > 0:
                consistency_score += 0.2
            
            results['overall_consistency'] = consistency_score
            
            # 일관성 점수에 따른 해석
            if consistency_score >= 0.8:
                results['consistency_interpretation'] = "우수: 대부분의 시장 조건에서 일관된 성과"
            elif consistency_score >= 0.6:
                results['consistency_interpretation'] = "양호: 다양한 시장 조건에서 어느 정도 일관됨"
            elif consistency_score >= 0.4:
                results['consistency_interpretation'] = "보통: 일부 시장 조건에서만 효과적"
            else:
                results['consistency_interpretation'] = "불량: 특정 시장 조건에 크게 의존"
                
            return results
            
        except Exception as e:
            self.logger.error(f"시장 조건 분석 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": f"시장 조건 분석 중 오류 발생: {str(e)}"}

    @log_execution
    def run_backtest(self) -> BacktestResult:
        """
        설정에 따라 백테스트 실행
        
        Returns:
            BacktestResult: 백테스트 결과 객체
        """
        # 전략 초기화
        try:
            from main import initialize_strategy
            strategy = initialize_strategy(self.config)
        except Exception as e:
            self.logger.error(f"전략 초기화 실패: {str(e)}")
            raise ValueError(f"전략 초기화 실패: {str(e)}")
        
        # 데이터 준비
        try:
            # 데이터 수집기 사용
            from data.collectors import get_historical_data
            
            # 데이터 수집
            data = get_historical_data(
                ticker=self.market,
                days=365,  # 백테스트 기간
                indicators=True,  # 인디케이터 추가
                verbose=True,
                extend_with_synthetic=False
            )
            
            if data is None or data.empty:
                raise ValueError(f"수집된 데이터가 없습니다: {self.market}, {self.timeframe}, {self.start_date} ~ {self.end_date}")
                
        except Exception as e:
            self.logger.error(f"데이터 수집 중 오류 발생: {str(e)}")
            raise ValueError(f"데이터 수집 실패: {str(e)}")
        
        # 백테스트 실행
        return self.run(
            strategy=strategy,
            start_date=self.start_date,
            end_date=self.end_date,
            data=data,
            use_daily_only=self.timeframe == 'day'
        ) 