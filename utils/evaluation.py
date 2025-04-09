"""
Evaluation module for Bitcoin Trading Bot

This module provides functions to evaluate trading performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
import math

from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


def calculate_return(equity_curve: List[float]) -> float:
    """
    Calculate total return
    
    Args:
        equity_curve (List[float]): Equity curve values
    
    Returns:
        float: Total return
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    return (equity_curve[-1] / equity_curve[0]) - 1.0


def calculate_annualized_return(equity_curve: List[float], days: int) -> float:
    """
    Calculate annualized return
    
    Args:
        equity_curve (List[float]): Equity curve values
        days (int): Number of days in the period
    
    Returns:
        float: Annualized return
    """
    if not equity_curve or len(equity_curve) < 2 or days <= 0:
        return 0.0
    
    total_return = calculate_return(equity_curve)
    return ((1 + total_return) ** (365 / days)) - 1


def calculate_volatility(returns: List[float]) -> float:
    """
    Calculate volatility (standard deviation of returns)
    
    Args:
        returns (List[float]): List of period returns
    
    Returns:
        float: Volatility
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    return np.std(returns, ddof=1)


def calculate_annualized_volatility(daily_returns: List[float]) -> float:
    """
    Calculate annualized volatility
    
    Args:
        daily_returns (List[float]): List of daily returns
    
    Returns:
        float: Annualized volatility
    """
    if not daily_returns or len(daily_returns) < 2:
        return 0.0
    
    daily_volatility = calculate_volatility(daily_returns)
    return daily_volatility * math.sqrt(365)


def calculate_sharpe_ratio(returns: List[float],
                         risk_free_rate: float = 0.0,
                         trading_days: int = 365) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns (List[float]): List of period returns
        risk_free_rate (float): Risk-free rate (annualized)
        trading_days (int): Number of trading days in a year
    
    Returns:
        float: Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    volatility = calculate_volatility(returns)
    
    # 변동성이 매우 낮은 경우 (거의 0에 가까운 경우)
    if volatility < 0.0001:
        # 평균 수익이 무위험 수익률보다 높으면 양수, 아니면 음수 반환
        rfr_daily = (1 + risk_free_rate) ** (1 / trading_days) - 1
        return 1.0 if mean_return > rfr_daily else -1.0
    
    # Convert risk-free rate to daily
    rfr_daily = (1 + risk_free_rate) ** (1 / trading_days) - 1
    
    # Calculate Sharpe ratio
    sharpe = (mean_return - rfr_daily) / volatility
    
    # 극단적인 값 제한
    if sharpe < -5:
        sharpe = -5.0
    elif sharpe > 5:
        sharpe = 5.0
    
    # Annualize
    sharpe_annualized = sharpe * math.sqrt(trading_days)
    
    # 연율화된 결과도 제한
    if sharpe_annualized < -5:
        return -5.0
    elif sharpe_annualized > 5:
        return 5.0
    
    return sharpe_annualized


def calculate_sortino_ratio(returns: List[float],
                          risk_free_rate: float = 0.0,
                          trading_days: int = 365) -> float:
    """
    Calculate Sortino ratio (using only downside volatility)
    
    Args:
        returns (List[float]): List of period returns
        risk_free_rate (float): Risk-free rate (annualized)
        trading_days (int): Number of trading days in a year
    
    Returns:
        float: Sortino ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    
    # Convert risk-free rate to daily
    rfr_daily = (1 + risk_free_rate) ** (1 / trading_days) - 1
    
    # Calculate downside returns
    downside_returns = [r - rfr_daily for r in returns if r < rfr_daily]
    
    if not downside_returns:
        # 하락 위험이 없는 경우 (모든 수익이 양수)
        return 10.0  # 합리적인 상한값으로 설정
    
    # 하락 편차 계산
    downside_deviation = math.sqrt(sum((r - rfr_daily) ** 2 for r in downside_returns) / len(downside_returns))
    
    if downside_deviation == 0:
        # 하락 편차가 0이면 무한대가 나오므로 대신 0을 반환
        return 0.0
    
    # 소르티노 비율 계산
    sortino = (mean_return - rfr_daily) / downside_deviation
    
    # 너무 극단적인 값 제한
    if sortino < -10:
        sortino = -10.0
    elif sortino > 10:
        sortino = 10.0
    
    # 연율화
    sortino_annualized = sortino * math.sqrt(trading_days)
    
    # 최종 결과에도 극단값 제한 적용
    if sortino_annualized < -10:
        return -10.0
    elif sortino_annualized > 10:
        return 10.0
    
    return sortino_annualized


def calculate_max_drawdown(equity_curve: Union[List[float], np.ndarray]) -> float:
    """
    최대 낙폭 계산
    
    Args:
        equity_curve (Union[List[float], np.ndarray]): 자본 곡선
        
    Returns:
        float: 최대 낙폭 (0~1 사이 값)
    """
    # 배열로 변환
    if not isinstance(equity_curve, np.ndarray):
        equity_curve = np.array(equity_curve)
    
    # 빈 배열이거나 요소가 충분하지 않은지 확인
    if equity_curve.size == 0 or equity_curve.size < 2:
        return 0.0
    
    # 누적 최대값 계산
    running_max = np.maximum.accumulate(equity_curve)
    
    # 낙폭 비율 계산
    drawdown = (running_max - equity_curve) / running_max
    
    # 최대 낙폭 반환
    return np.max(drawdown) if drawdown.size > 0 else 0.0


def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate win rate (percentage of winning trades)
    
    Args:
        trades (List[Dict[str, Any]]): List of trades
    
    Returns:
        float: Win rate (0.0 to 1.0)
    """
    if not trades:
        return 0.0
    
    # 매수와 매도 거래 계산
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    
    # 매도 거래가 없으면 승률을 계산할 수 없음
    if not sell_trades:
        logger.warning("매도 거래가 없어 승률을 계산할 수 없습니다.")
        return 0.0
    
    winning_trades = 0
    matched_trades = []
    
    # 매수-매도 쌍 매칭
    for sell_trade in sell_trades:
        sell_price = sell_trade['price']
        sell_time = datetime.fromisoformat(sell_trade['timestamp'])
        
        # 가장 적합한 매수 거래 찾기 (아직 매칭되지 않은 것 중에서)
        best_buy_trade = None
        for buy_trade in buy_trades:
            if buy_trade not in matched_trades:
                buy_time = datetime.fromisoformat(buy_trade['timestamp'])
                if buy_time < sell_time:  # 매수가 매도보다 먼저 발생
                    best_buy_trade = buy_trade
                    break
        
        if best_buy_trade:
            matched_trades.append(best_buy_trade)
            buy_price = best_buy_trade['price']
            
            # 수익률 계산
            profit_pct = (sell_price / buy_price) - 1.0
            
            # 수수료 고려
            buy_commission = best_buy_trade['commission']
            sell_commission = sell_trade['commission']
            position_size = best_buy_trade.get('position_size', best_buy_trade.get('amount', 0))
            
            # 순수익 계산
            gross_profit = position_size * sell_price - position_size * buy_price
            net_profit = gross_profit - buy_commission - sell_commission
            
            if net_profit > 0:
                winning_trades += 1
    
    # 완료된 거래 쌍 수
    completed_trades = len(matched_trades)
    
    if completed_trades == 0:
        logger.warning("완료된 거래 쌍이 없어 승률을 계산할 수 없습니다.")
        return 0.0
    
    win_rate = winning_trades / completed_trades
    logger.info(f"승률 계산: {winning_trades}승 / {completed_trades}거래 = {win_rate:.2%}")
    
    return win_rate


def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate profit factor (gross profit / gross loss)
    
    Args:
        trades (List[Dict[str, Any]]): List of trades
    
    Returns:
        float: Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profit = 0.0
    gross_loss = 0.0
    buy_trades = {}
    
    # Process all trades
    for trade in trades:
        if trade['type'] == 'BUY':
            buy_trades[trade['timestamp']] = trade
        elif trade['type'] == 'SELL' and buy_trades:
            # Find matching buy trade (simple implementation)
            buy_timestamp = list(buy_trades.keys())[0]
            buy_trade = buy_trades.pop(buy_timestamp)
            
            # Calculate profit
            buy_price = buy_trade['price']
            sell_price = trade['price']
            profit = (sell_price / buy_price) - 1.0
            
            # Calculate net profit after commission
            buy_commission = buy_trade['commission']
            sell_commission = trade['commission']
            
            # 'position_size' 키가 없는 경우 'amount' 키 사용
            position_size = buy_trade.get('position_size', buy_trade.get('amount', 0))
            
            net_profit = (position_size * (1 + profit)) - position_size - buy_commission - sell_commission
            
            if net_profit > 0:
                gross_profit += net_profit
            else:
                gross_loss += abs(net_profit)
    
    # 손실이 매우 적거나 없는 경우
    if gross_loss < 0.0001:
        if gross_profit > 0:
            return 10.0  # 합리적인 상한값
        else:
            return 0.0
    
    # 손실이 훨씬 큰 경우 (매우 낮은 profit factor)
    if gross_profit < 0.0001 and gross_loss > 0:
        return 0.01  # 최소값
    
    profit_factor = gross_profit / gross_loss
    
    # 극단적인 값 제한
    if profit_factor > 10.0:
        profit_factor = 10.0
    
    return profit_factor


def calculate_average_trade(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate average trade metrics
    
    Args:
        trades (List[Dict[str, Any]]): List of trades
    
    Returns:
        Dict[str, float]: Dictionary with average trade metrics
    """
    if not trades:
        return {
            'avg_profit': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_hold_time': 0.0
        }
    
    total_profit = 0.0
    winning_profits = []
    losing_profits = []
    hold_times = []
    buy_trades = {}
    
    # Process all trades
    for trade in trades:
        if trade['type'] == 'BUY':
            buy_trades[trade['timestamp']] = trade
        elif trade['type'] == 'SELL' and buy_trades:
            # Find matching buy trade (simple implementation)
            buy_timestamp = list(buy_trades.keys())[0]
            buy_trade = buy_trades.pop(buy_timestamp)
            
            # Calculate profit
            buy_price = buy_trade['price']
            sell_price = trade['price']
            profit = (sell_price / buy_price) - 1.0
            
            # Calculate net profit after commission
            buy_commission = buy_trade['commission']
            sell_commission = trade['commission']
            
            # 'position_size' 키가 없는 경우 'amount' 키 사용
            position_size = buy_trade.get('position_size', buy_trade.get('amount', 0))
            
            net_profit = (position_size * (1 + profit)) - position_size - buy_commission - sell_commission
            
            # Track profits
            total_profit += net_profit
            if net_profit > 0:
                winning_profits.append(net_profit)
            else:
                losing_profits.append(net_profit)
            
            # Calculate hold time
            buy_time = datetime.fromisoformat(buy_trade['timestamp'])
            sell_time = datetime.fromisoformat(trade['timestamp'])
            hold_time_hours = (sell_time - buy_time).total_seconds() / 3600
            hold_times.append(hold_time_hours)
    
    # Calculate averages
    completed_trades = len(trades) // 2
    avg_profit = total_profit / completed_trades if completed_trades > 0 else 0.0
    avg_win = sum(winning_profits) / len(winning_profits) if winning_profits else 0.0
    avg_loss = sum(losing_profits) / len(losing_profits) if losing_profits else 0.0
    avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0.0
    
    return {
        'avg_profit': avg_profit,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_hold_time': avg_hold_time
    }


def calculate_metrics(equity_curve: List[float],
                    trades: List[Dict[str, Any]],
                    daily_returns: List[float],
                    risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate all performance metrics
    
    Args:
        equity_curve (List[float]): Equity curve values
        trades (List[Dict[str, Any]]): List of trades
        daily_returns (List[float]): List of daily returns
        risk_free_rate (float): Annual risk-free rate
    
    Returns:
        Dict[str, float]: Dictionary with all metrics
    """
    # Calculate days in backtest period
    days = len(daily_returns) if daily_returns else 1
    
    # 거래가 없는 경우에 대한 처리
    if len(trades) == 0:
        return {
            'total_return': 0.0,
            'monthly_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'annualized_volatility': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': -1.0,  # 거래가 없는 경우 약간 부정적인 값으로 설정
            'sortino_ratio': -1.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_profit': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_hold_time': 0.0,
            'total_trades': 0,
            'risk_reward_ratio': 0.0
        }
    
    # Calculate return metrics
    total_return = calculate_return(equity_curve)
    annualized_return = calculate_annualized_return(equity_curve, days)
    
    # Calculate risk metrics
    volatility = calculate_volatility(daily_returns) if daily_returns else 0.0
    annualized_volatility = calculate_annualized_volatility(daily_returns) if daily_returns else 0.0
    max_drawdown = calculate_max_drawdown(equity_curve)
    
    # Calculate risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate) if daily_returns else -1.0
    sortino = calculate_sortino_ratio(daily_returns, risk_free_rate) if daily_returns else -1.0
    
    # Calculate trade metrics
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    avg_trade_metrics = calculate_average_trade(trades)
    
    # Calculate monthly return (annualized_return to monthly)
    monthly_return = ((1 + annualized_return) ** (1/12)) - 1
    
    # 리스크 리워드 비율 계산 - 개선된 계산 로직
    risk_reward_ratio = 0.0
    avg_win = avg_trade_metrics['avg_win'] 
    avg_loss = avg_trade_metrics['avg_loss']
    
    # 평균 손실이 0보다 작은 경우만 계산
    if avg_loss < 0:
        risk_reward_ratio = abs(avg_win / avg_loss)
        
        # 평균 이익이 매우 작거나 0인 경우
        if avg_win < 0.0001:
            risk_reward_ratio = 0.01
        
        # 평균 손실이 매우 작은 경우 (분모가 매우 작음)
        if abs(avg_loss) < 0.0001 and avg_win > 0:
            risk_reward_ratio = 10.0
        
        # 극단적인 값 제한
        if risk_reward_ratio > 10.0:
            risk_reward_ratio = 10.0
    elif avg_loss == 0 and avg_win > 0:
        # 손실이 없는 경우
        risk_reward_ratio = 10.0
    
    # Combine all metrics
    metrics = {
        'total_return': total_return,
        'monthly_return': monthly_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'annualized_volatility': annualized_volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_profit': avg_trade_metrics['avg_profit'],
        'avg_win': avg_trade_metrics['avg_win'],
        'avg_loss': avg_trade_metrics['avg_loss'],
        'avg_hold_time': avg_trade_metrics['avg_hold_time'],
        'total_trades': len(trades) // 2,
        'risk_reward_ratio': risk_reward_ratio
    }
    
    logger.info(f"Calculated performance metrics: {metrics}")
    
    return metrics


def format_backtest_results(result) -> Dict[str, Any]:
    """
    Format backtest results to a human-readable dictionary
    
    Args:
        result: BacktestResult object from backtest engine
        
    Returns:
        Dict[str, Any]: Formatted results dictionary
    """
    formatted = {
        '전략': result.strategy_name,
        '시장': result.market,
        '기간': f"{result.start_date.strftime('%Y년 %m월 %d일')} ~ {result.end_date.strftime('%Y년 %m월 %d일')}",
        '초기 자금': f"{int(result.initial_balance):,} 원",
        '최종 자금': f"{int(result.final_balance):,} 원",
        '총 수익률': f"{result.returns * 100:.2f}%",
        '총 거래 횟수': f"{len(result.trades):,}회",
        
        # 중요 지표들 명확히 포맷팅
        '월 수익률': f"{result.metrics.get('monthly_return', 0) * 100:.2f}%",
        '최대 낙폭': f"{result.metrics.get('max_drawdown', 0) * 100:.2f}%",
        '샤프 지수': f"{result.metrics.get('sharpe_ratio', 0):.2f}",
        '소르티노 지수': f"{result.metrics.get('sortino_ratio', 0):.2f}",
        '승률': f"{result.metrics.get('win_rate', 0) * 100:.2f}%",
        '수익 요인': f"{result.metrics.get('profit_factor', 0):.2f}",
        '리스크 리워드 비율': f"{result.metrics.get('risk_reward_ratio', 0):.2f}",
    }
    
    # 거래 통계 추가
    if 'avg_profit' in result.metrics:
        formatted['평균 수익'] = f"{result.metrics['avg_profit']:.2f}"
    
    if 'avg_win' in result.metrics:
        formatted['평균 이익'] = f"{result.metrics['avg_win']:.2f}"
    
    if 'avg_loss' in result.metrics:
        formatted['평균 손실'] = f"{result.metrics['avg_loss']:.2f}"
    
    if 'avg_hold_time' in result.metrics:
        formatted['평균 보유 시간'] = f"{result.metrics['avg_hold_time']:.2f} 시간"
    
    # 이외의 지표들 처리
    metric_names = {
        'total_return': '총 수익률',
        'monthly_return': '월 수익률',
        'annualized_return': '연간 수익률',
        'volatility': '변동성',
        'annualized_volatility': '연간 변동성',
        'max_drawdown': '최대 낙폭',
        'sharpe_ratio': '샤프 지수',
        'sortino_ratio': '소르티노 지수',
        'win_rate': '승률',
        'profit_factor': '수익 요인',
        'risk_reward_ratio': '리스크 리워드 비율',
        'avg_profit': '평균 수익',
        'avg_win': '평균 이익',
        'avg_loss': '평균 손실',
        'avg_hold_time': '평균 보유 시간',
        'total_trades': '총 거래 횟수'
    }
    
    for key, value in result.metrics.items():
        # 이미 처리한 주요 지표들은 건너뜀
        if key in ['monthly_return', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 
                  'win_rate', 'profit_factor', 'risk_reward_ratio', 'avg_profit', 
                  'avg_win', 'avg_loss', 'avg_hold_time']:
            continue
            
        korean_key = metric_names.get(key, key)
        
        if isinstance(value, float):
            if 'return' in key or 'rate' in key or 'drawdown' in key:
                formatted[korean_key] = f"{value * 100:.2f}%"
            elif 'ratio' in key:
                formatted[korean_key] = f"{value:.2f}"
            else:
                formatted[korean_key] = f"{value:.2f}"
        else:
            formatted[korean_key] = value
    
    if result.parameters:
        formatted['매개변수'] = result.parameters
    
    return formatted 


def perform_statistical_test(daily_returns: List[float], confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    백테스트 결과의 통계적 유의성을 테스트
    
    Args:
        daily_returns (List[float]): 일별 수익률 목록
        confidence_level (float): 신뢰 수준 (기본값: 0.95)
        
    Returns:
        Dict[str, Any]: 통계적 테스트 결과
    """
    import scipy.stats as stats
    
    if not daily_returns or len(daily_returns) < 30:  # 최소 30개 샘플 필요
        return {
            'significant': False,
            'p_value': 1.0,
            'confidence_level': confidence_level,
            'test_type': 'insufficient_data'
        }
    
    # 일별 수익률이 0보다 큰지 테스트 (단측 t-검정)
    t_stat, p_value = stats.ttest_1samp(daily_returns, 0)
    
    # 단측 p-값 계산 (수익률이 0보다 큰지)
    if t_stat > 0:
        p_value = p_value / 2  # 단측 검정을 위해 p-값 조정
    else:
        p_value = 1 - (p_value / 2)  # 반대 방향 검정
    
    # 결과 해석
    significant = p_value < (1 - confidence_level)
    
    return {
        'significant': significant,
        'p_value': p_value,
        'confidence_level': confidence_level,
        'mean_return': np.mean(daily_returns),
        't_statistic': t_stat,
        'test_type': 't_test'
    }


def perform_walk_forward_analysis(
    strategy,
    data: pd.DataFrame, 
    initial_balance: float = 10000000,
    window_size: int = 30,  # 각 구간의 일 수
    step_size: int = 15,    # 구간 이동 간격
    commission_rate: float = 0.0005
) -> Dict[str, Any]:
    """
    워크포워드 분석을 수행하여 백테스트 결과의 견고성을 평가
    
    Args:
        strategy: 테스트할 전략 객체
        data (pd.DataFrame): 전체 가격 데이터
        initial_balance (float): 초기 잔고
        window_size (int): 각 테스트 창의 크기(일)
        step_size (int): 창 이동 간격(일)
        commission_rate (float): 거래 수수료율
        
    Returns:
        Dict[str, Any]: 워크포워드 분석 결과
    """
    from backtest.engine import BacktestEngine
    
    if len(data) <= window_size:
        return {
            'error': '데이터가 충분하지 않습니다.',
            'required_days': window_size,
            'available_days': len(data)
        }
    
    # 결과 저장 리스트
    period_results = []
    
    # 첫 날과 마지막 날
    start_idx = 0
    last_idx = len(data) - window_size
    
    # 각 구간에 대해 백테스트 실행
    while start_idx <= last_idx:
        end_idx = start_idx + window_size
        period_data = data.iloc[start_idx:end_idx].copy()
        
        # 이 구간의 시작일과 종료일
        period_start = period_data.index[0]
        period_end = period_data.index[-1]
        
        try:
            # 백테스트 엔진 초기화 및 실행
            engine = BacktestEngine(
                initial_balance=initial_balance,
                commission_rate=commission_rate
            )
            
            # 백테스트 실행
            result = engine.run(
                strategy=strategy,
                start_date=period_start,
                end_date=period_end,
                data=period_data
            )
            
            # 결과 저장
            period_results.append({
                'period': f"{period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
                'total_return': result.returns,
                'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                'win_rate': result.metrics.get('win_rate', 0),
                'trades': len(result.trades),
                'start_date': period_start,
                'end_date': period_end
            })
            
        except Exception as e:
            logger.error(f"구간 {period_start} ~ {period_end} 분석 중 오류: {str(e)}")
        
        # 다음 구간으로 이동
        start_idx += step_size
    
    # 결과 집계
    if not period_results:
        return {'error': '분석 가능한 구간이 없습니다.'}
    
    # 기간별 수익률 통계
    returns = [r['total_return'] for r in period_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in period_results]
    win_rates = [r['win_rate'] for r in period_results]
    
    # 결과 요약
    summary = {
        'periods': len(period_results),
        'profitable_periods': sum(1 for r in returns if r > 0),
        'avg_return': np.mean(returns),
        'median_return': np.median(returns),
        'std_return': np.std(returns),
        'avg_sharpe': np.mean(sharpe_ratios),
        'avg_win_rate': np.mean(win_rates),
        'period_results': period_results,
        'consistency_score': sum(1 for r in returns if r > 0) / len(returns) if returns else 0
    }
    
    return summary


def calculate_monte_carlo_confidence(strategy_returns: Union[List[float], np.ndarray], 
                                  num_simulations: int = 1000,
                                  confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    몬테카를로 시뮬레이션을 사용하여 수익률의 신뢰구간 계산
    
    Args:
        strategy_returns (Union[List[float], np.ndarray]): 일일 수익률 목록
        num_simulations (int): 시뮬레이션 횟수
        confidence_level (float): 신뢰 수준 (0~1 사이 값)
        
    Returns:
        Dict[str, Any]: 몬테카를로 시뮬레이션 결과
    """
    # 배열로 변환
    if not isinstance(strategy_returns, np.ndarray):
        strategy_returns = np.array(strategy_returns)
    
    # NaN 값 제거
    strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
    
    # 빈 배열이거나 요소가 충분하지 않은지 확인
    if strategy_returns.size == 0 or strategy_returns.size < 5:
        return {
            'original_return': 0.0,
            'return_ci': (0.0, 0.0),
            'return_below_zero_probability': 0.5,
            'number_of_simulations': num_simulations
        }
    
    # 원래 수익률 계산 (단순 누적 곱)
    original_return = np.prod(1 + strategy_returns) - 1
    
    # 일일 수익률의 평균 및 표준편차 계산
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)
    
    # 표준편차가 0인 경우 대비
    if std_return == 0 or np.isnan(std_return):
        std_return = 0.0001  # 매우 작은 값으로 설정
    
    # 몬테카를로 시뮬레이션 수행
    simulated_returns = []
    
    for _ in range(num_simulations):
        # 일일 수익률을 정규분포에서 무작위로 추출
        random_returns = np.random.normal(mean_return, std_return, len(strategy_returns))
        
        # 누적 수익률 계산
        sim_return = np.prod(1 + random_returns) - 1
        simulated_returns.append(sim_return)
    
    # 신뢰구간 계산
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    
    sorted_returns = np.sort(simulated_returns)
    lower_idx = int(lower_percentile * num_simulations)
    upper_idx = int(upper_percentile * num_simulations)
    
    lower_ci = sorted_returns[lower_idx]
    upper_ci = sorted_returns[upper_idx]
    
    # 손실 확률 계산 (0 미만 수익률의 비율)
    below_zero_count = sum(1 for r in simulated_returns if r < 0)
    below_zero_probability = below_zero_count / num_simulations
    
    # 시뮬레이션 경로로부터 자본 곡선 생성
    equity_curves = []
    for _ in range(min(100, num_simulations)):  # 100개 경로만 저장
        random_returns = np.random.normal(mean_return, std_return, len(strategy_returns))
        equity_curve = np.cumprod(1 + random_returns)
        equity_curves.append(equity_curve)
    
    # 평균 경로와 5%/95% 경로 계산
    equity_array = np.array(equity_curves)
    mean_curve = np.mean(equity_array, axis=0)
    lower_curve = np.percentile(equity_array, lower_percentile * 100, axis=0)
    upper_curve = np.percentile(equity_array, upper_percentile * 100, axis=0)
    
    # 최대 낙폭 계산
    try:
        max_drawdown = calculate_max_drawdown(mean_curve)
    except Exception as e:
        logger.warning(f"몬테카를로 최대 낙폭 계산 중 오류: {str(e)}")
        max_drawdown = 0.0
    
    return {
        'original_return': original_return,
        'return_ci': (lower_ci, upper_ci),
        'return_below_zero_probability': below_zero_probability,
        'number_of_simulations': num_simulations,
        'mean_curve': mean_curve.tolist(),
        'lower_curve': lower_curve.tolist(),
        'upper_curve': upper_curve.tolist(),
        'max_drawdown': max_drawdown
    }

def analyze_market_correlation(
    strategy_returns: List[float],
    market_returns: List[float]
) -> Dict[str, float]:
    """
    전략 수익률과 시장 수익률 간의 상관관계 및 베타 분석
    
    Args:
        strategy_returns (List[float]): 전략의 일별 수익률
        market_returns (List[float]): 시장의 일별 수익률
        
    Returns:
        Dict[str, float]: 상관관계 분석 결과
    """
    if len(strategy_returns) != len(market_returns) or not strategy_returns:
        return {
            'error': '전략 수익률과 시장 수익률의 데이터 길이가 일치하지 않거나 데이터가 없습니다.'
        }
    
    # 데이터 유효성 체크
    strategy_returns_clean = np.array(strategy_returns)
    market_returns_clean = np.array(market_returns)
    
    # NaN 값 제거
    valid_indices = ~(np.isnan(strategy_returns_clean) | np.isnan(market_returns_clean))
    strategy_returns_clean = strategy_returns_clean[valid_indices]
    market_returns_clean = market_returns_clean[valid_indices]
    
    # 데이터 값이 초기화되었는지 재확인
    if len(strategy_returns_clean) < 2 or len(market_returns_clean) < 2:
        return {
            'correlation': 0.0,
            'beta': 0.0,
            'alpha': 0.0,
            'excess_return': 0.0,
            'market_cumulative_return': 0.0,
            'strategy_cumulative_return': 0.0,
            'outperformance': 0.0
        }
    
    # 표준 편차 검사
    strategy_std = np.std(strategy_returns_clean)
    market_std = np.std(market_returns_clean)
    
    # 상관계수 계산 - 표준 편차가 0인 경우 포함한 주의 처리
    if strategy_std == 0 or market_std == 0:
        correlation = 0.0  # 표준 편차가 0이면 상관관계도 0
    else:
        try:
            # 수동 상관계수 계산
            mean_strategy = np.mean(strategy_returns_clean)
            mean_market = np.mean(market_returns_clean)
            
            numerator = np.sum((strategy_returns_clean - mean_strategy) * (market_returns_clean - mean_market))
            denominator = strategy_std * market_std * len(strategy_returns_clean)
            
            if denominator == 0:
                correlation = 0.0
            else:
                correlation = numerator / denominator
        except Exception as e:
            logger.warning(f"상관계수 계산 오류: {str(e)}")
            correlation = 0.0
    
    # 베타 계산 (시장 변동성 대비 전략 변동성)
    market_variance = np.var(market_returns_clean)
    if market_variance == 0:
        beta = 0
    else:
        beta = correlation * (strategy_std / market_std)
    
    # 알파 계산 (CAPM 모델 기반)
    risk_free_rate = 0.02 / 365  # 일별 무위험 수익률 (2% 연간)
    expected_return = risk_free_rate + beta * (np.mean(market_returns_clean) - risk_free_rate)
    alpha = np.mean(strategy_returns_clean) - expected_return
    
    # 시장 대비 초과 수익률
    excess_return = np.mean(strategy_returns_clean) - np.mean(market_returns_clean)
    
    # 지수화된 시장 및 전략 누적 수익률
    market_cumulative = (1 + np.array(market_returns_clean)).cumprod()[-1] - 1
    strategy_cumulative = (1 + np.array(strategy_returns_clean)).cumprod()[-1] - 1
    
    return {
        'correlation': correlation,
        'beta': beta,
        'alpha': alpha * 365,  # 연간 알파로 변환
        'excess_return': excess_return * 365,  # 연간 초과 수익률로 변환
        'market_cumulative_return': market_cumulative,
        'strategy_cumulative_return': strategy_cumulative,
        'outperformance': strategy_cumulative - market_cumulative
    }

def calculate_profit_metrics(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    거래 목록에서 수익률 지표 계산
    
    Args:
        trades (List[Dict[str, Any]]): 거래 목록
        
    Returns:
        Dict[str, float]: 수익률 지표
    """
    if not trades:
        return {
            'total_profit': 0,
            'total_returns': 0,  # profit_percentage에서 returns로 변경
            'win_rate': 0,
            'profit_factor': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'max_profit': 0,
            'max_loss': 0,
            'avg_hold_time': 0
        }
    
    # 수익 거래와 손실 거래 분리
    profit_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
    loss_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]
    
    # 총 수익 및 손실 계산
    total_profit = sum(t.get('profit_loss', 0) for t in profit_trades)
    total_loss = abs(sum(t.get('profit_loss', 0) for t in loss_trades))
    
    # 수익률 지표 계산
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    win_rate = len(profit_trades) / len(trades) if trades else 0
    
    # 평균 수익 및 손실
    avg_profit = total_profit / len(profit_trades) if profit_trades else 0
    avg_loss = total_loss / len(loss_trades) if loss_trades else 0
    
    # 최대 수익 및 손실
    max_profit = max([t.get('profit_loss', 0) for t in profit_trades], default=0)
    max_loss = min([t.get('profit_loss', 0) for t in loss_trades], default=0)
    
    # 평균 보유 기간 (거래 간 시간 간격)
    hold_times = []
    for i in range(len(trades)):
        if i > 0 and trades[i].get('type') == 'SELL' and trades[i-1].get('type') == 'BUY':
            try:
                buy_time = datetime.fromisoformat(trades[i-1].get('timestamp', ''))
                sell_time = datetime.fromisoformat(trades[i].get('timestamp', ''))
                hold_time = (sell_time - buy_time).total_seconds() / 86400  # 일 단위로 변환
                hold_times.append(hold_time)
            except (ValueError, TypeError):
                pass
    
    avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
    
    # 거래당 평균 수익률 계산
    avg_return_per_trade = sum(t.get('returns', 0) for t in trades) / len(trades) if trades else 0  # profit_pct에서 returns로 변경
    
    return {
        'total_profit': total_profit,
        'total_returns': total_profit / trades[0].get('balance_after', 1),  # profit_percentage에서 returns로 변경
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'avg_hold_time': avg_hold_time,
        'avg_return_per_trade': avg_return_per_trade  # profit_pct에서 returns_per_trade로 변경
    } 