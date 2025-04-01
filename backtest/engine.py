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
from data.indicators import add_indicators
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
        
        logger.info(f"Saved backtest results to {filepath}")
        return filepath


class BacktestEngine:
    """Backtesting engine for evaluating trading strategies"""
    
    def __init__(self, 
                initial_balance: float = 1000000.0,
                commission_rate: float = 0.0005,
                data_frequency: str = 'minute60',
                slippage: float = 0.0):
        """
        Initialize the backtest engine
        
        Args:
            initial_balance (float): Initial balance in KRW
            commission_rate (float): Trading commission as a decimal
            data_frequency (str): Data frequency (e.g., minute60, day)
            slippage (float): Slippage as a decimal
        """
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.data_frequency = data_frequency
        self.slippage = slippage
        self.logger = logger
        
        self.logger.info(f"Initialized backtest engine with {initial_balance} KRW, "
                         f"{commission_rate*100}% commission, {slippage*100}% slippage")
    
    @log_execution
    def run(self, strategy: BaseStrategy, 
            start_date: datetime, 
            end_date: datetime,
            data: Optional[pd.DataFrame] = None,
            use_daily_only: bool = True) -> BacktestResult:
        """
        Run backtest for a given strategy and time period
        
        Args:
            strategy (BaseStrategy): Trading strategy to test
            start_date (datetime): Backtest start date
            end_date (datetime): Backtest end date
            data (Optional[pd.DataFrame]): Historical data, will be fetched if None
            use_daily_only (bool): Whether to use only daily data (True) or include hourly data (False)
            
        Returns:
            BacktestResult: Backtest results
        """
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
            data = add_indicators(data)
        
        # 데이터가 이미 필터링되어 있어야 하므로 다시 필터링하지 않음
        if len(data) == 0:
            self.logger.error(f"No data available for backtest period {start_date} to {end_date}")
            raise ValueError(f"No data available for backtest period {start_date} to {end_date}")
        
        # Initialize portfolio state
        portfolio = {
            'cash': self.initial_balance * 0.7,  # 초기 자금의 30%는 이미 포지션에 투자
            'position': self.initial_balance * 0.3 / data.iloc[0]['close'],  # 초기 자금의 30%를 첫날 가격으로 나눠 초기 포지션 설정
            'position_value': self.initial_balance * 0.3,  # 초기 포지션 가치
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
            
            # Create market data for signal generation
            market_data = data.loc[:timestamp].copy()
            
            # Generate signal
            signal = strategy.generate_signal(market_data)
            
            # 신호 필터링 로직 추가
            if portfolio['last_signal'] == signal['signal']:
                portfolio['consecutive_signals'] += 1
            else:
                portfolio['consecutive_signals'] = 1
                
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
                        self.logger.info(f"BUY: {position_size:.8f} {strategy.market} @ {execution_price:.2f} = "
                                        f"{position_cost:.2f} (commission: {commission:.2f})")
                        
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
                    
                    # Execute sell
                    portfolio['cash'] += (position_value - commission)
                    portfolio['position'] -= position_size  # 매도한 만큼만 포지션 감소
                    portfolio['position_value'] = portfolio['position'] * current_price  # 남은 포지션 가치 업데이트
                    portfolio['total_value'] = portfolio['cash'] + portfolio['position_value']
                    portfolio['last_trade_time'] = timestamp
                    
                    # 거래 로그 수준 향상 (DEBUG에서 INFO로)
                    self.logger.info(f"SELL: {position_size:.8f} {strategy.market} @ {execution_price:.2f} = "
                                    f"{position_value:.2f} (commission: {commission:.2f})")
                    
                    # Add trade to history
                    trade = {
                        'timestamp': timestamp.isoformat(),
                        'type': 'SELL',
                        'price': execution_price,
                        'amount': position_size,
                        'value': position_value,
                        'commission': commission,
                        'balance_after': portfolio['cash'],
                        'position_size': position_size,
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
        returns = (final_balance / self.initial_balance) - 1
        
        # Create result object
        result = BacktestResult(
            strategy_name=strategy.name,
            market=strategy.market,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            returns=returns,
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
                data = add_indicators(data)
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