"""
성능 지표 집계 유틸리티

이 모듈은 거래 성능 지표를 집계하고 분석하는 기능을 제공합니다.
주간, 월간 성능 지표 집계 및 스케줄링 기능을 포함합니다.
"""

import sqlite3
import threading
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd

from config import settings
from utils.logging import get_logger
from utils.database import get_db_connection, save_performance_metric
from utils.metadata import create_standard_metadata, metadata_to_json

# 로거 초기화
logger = get_logger(__name__)

def aggregate_performance_metrics(period: str = "weekly", strategy: Optional[str] = None) -> bool:
    """
    일간/거래별 성능 지표를 주간/월간으로 집계
    
    Args:
        period (str): 집계 기간 ('weekly' 또는 'monthly')
        strategy (str, optional): 특정 전략명 (None이면 모든 전략)
        
    Returns:
        bool: 성공 여부
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 현재 날짜와 집계 기준 날짜 계산
        now = datetime.now()
        
        if period == "weekly":
            # 지난 주 일요일부터 토요일까지
            end_date = now - timedelta(days=now.weekday() + 1)
            start_date = end_date - timedelta(days=6)
            period_str = f"{start_date.strftime('%Y-%m-%d')}~{end_date.strftime('%Y-%m-%d')}"
        elif period == "monthly":
            # 지난 달 1일부터 말일까지
            last_day_of_prev_month = date(now.year, now.month, 1) - timedelta(days=1)
            start_date = date(last_day_of_prev_month.year, last_day_of_prev_month.month, 1)
            end_date = last_day_of_prev_month
            period_str = f"{start_date.strftime('%Y-%m')}월"
        else:
            logger.error(f"지원하지 않는 집계 기간: {period}")
            return False
        
        # 전략 목록 조회
        if strategy:
            strategies = [strategy]
        else:
            # 모든 전략 가져오기
            cursor.execute("SELECT DISTINCT strategy FROM performance_metrics WHERE period IN ('daily', 'trade')")
            strategies = [row[0] for row in cursor.fetchall()]
        
        # 각 전략별로 집계
        for strategy_name in strategies:
            try:
                # 성능 지표 집계 대상 기간 조회
                query = """
                SELECT * FROM performance_metrics 
                WHERE strategy = ? 
                AND period IN ('daily', 'trade')
                AND timestamp BETWEEN ? AND ?
                """
                
                cursor.execute(query, (
                    strategy_name, 
                    start_date.strftime('%Y-%m-%d'), 
                    (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
                ))
                
                metrics = cursor.fetchall()
                
                if not metrics:
                    logger.info(f"전략 '{strategy_name}'의 {period} 집계를 위한 데이터가 없습니다.")
                    continue
                
                # 결과를 DataFrame으로 변환
                metrics_df = pd.DataFrame(metrics)
                
                # 성능 지표 집계
                total_profit = 0.0
                total_loss = 0.0
                trade_count = 0
                win_count = 0
                
                # 일일 수익 집계
                daily_profits = metrics_df[metrics_df['metric_name'] == 'daily_profit']
                if not daily_profits.empty:
                    daily_profit_sum = daily_profits['metric_value'].sum()
                    
                    # 일일 수익 저장
                    details = {
                        'strategy': strategy_name,
                        'period': period,
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'source': 'daily_profit'
                    }
                    save_performance_metric(
                        strategy_name, 
                        f"{period}_profit", 
                        daily_profit_sum,
                        period=period,
                        details=details
                    )
                
                # 거래별 수익 집계
                trade_profits = metrics_df[metrics_df['metric_name'] == 'trade_profit']
                if not trade_profits.empty:
                    for _, row in trade_profits.iterrows():
                        profit = row['metric_value']
                        trade_count += 1
                        
                        if profit > 0:
                            total_profit += profit
                            win_count += 1
                        else:
                            total_loss += abs(profit)
                    
                    # 순 수익 계산
                    net_profit = total_profit - total_loss
                    
                    # 승률 계산
                    win_rate = win_count / trade_count if trade_count > 0 else 0
                    
                    # 거래 성능 지표 저장
                    details = {
                        'strategy': strategy_name,
                        'period': period,
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'trade_count': trade_count,
                        'win_count': win_count,
                        'loss_count': trade_count - win_count,
                        'win_rate': win_rate,
                        'source': 'trade_profit'
                    }
                    
                    # 순 수익 저장
                    save_performance_metric(
                        strategy_name, 
                        f"{period}_net_profit", 
                        net_profit,
                        period=period,
                        details=details
                    )
                    
                    # 승률 저장
                    save_performance_metric(
                        strategy_name, 
                        f"{period}_win_rate", 
                        win_rate,
                        period=period,
                        details=details
                    )
                    
                    # 거래 횟수 저장
                    save_performance_metric(
                        strategy_name, 
                        f"{period}_trade_count", 
                        trade_count,
                        period=period,
                        details=details
                    )
                
                logger.info(f"전략 '{strategy_name}'의 {period} 성능 지표 집계 완료 ({period_str})")
                
            except Exception as e:
                logger.error(f"전략 '{strategy_name}'의 {period} 성능 지표 집계 중 오류 발생: {str(e)}")
        
        return True
    
    except Exception as e:
        logger.error(f"성능 지표 집계 중 오류 발생: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def get_performance_summary(strategy: str, period: str = "daily") -> Dict[str, Any]:
    """
    특정 전략의 성능 요약 정보 조회
    
    Args:
        strategy (str): 전략 이름
        period (str): 기간 ('daily', 'trade', 'weekly', 'monthly')
        
    Returns:
        Dict[str, Any]: 성능 요약 정보
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 최근 성능 지표 조회
        query = """
        SELECT * FROM performance_metrics 
        WHERE strategy = ? AND period = ?
        ORDER BY timestamp DESC
        LIMIT 10
        """
        
        cursor.execute(query, (strategy, period))
        metrics = cursor.fetchall()
        
        if not metrics:
            return {
                'strategy': strategy,
                'period': period,
                'message': f"성능 지표가 없습니다.",
                'metrics': {}
            }
        
        # 결과를 Dict로 변환
        metrics_dict = {}
        for metric in metrics:
            metric_name = metric['metric_name']
            if metric_name not in metrics_dict:
                metrics_dict[metric_name] = {
                    'value': metric['metric_value'],
                    'timestamp': metric['timestamp']
                }
        
        return {
            'strategy': strategy,
            'period': period,
            'last_updated': metrics[0]['timestamp'],
            'metrics': metrics_dict
        }
    
    except Exception as e:
        logger.error(f"성능 요약 정보 조회 중 오류: {str(e)}")
        return {'error': str(e)}
    finally:
        if conn:
            conn.close()

def compare_strategy_performance(strategies: List[str], period: str = "daily") -> Dict[str, Any]:
    """
    여러 전략의 성능 비교
    
    Args:
        strategies (List[str]): 전략 목록
        period (str): 기간 ('daily', 'trade', 'weekly', 'monthly')
        
    Returns:
        Dict[str, Any]: 전략별 성능 비교 결과
    """
    results = {}
    best_strategy = None
    best_net_profit = float('-inf')
    
    for strategy in strategies:
        summary = get_performance_summary(strategy, period)
        results[strategy] = summary
        
        # 최고 성능 전략 식별
        net_profit_metric = f"{period}_net_profit" if period in ["weekly", "monthly"] else "net_profit"
        if 'metrics' in summary and net_profit_metric in summary['metrics']:
            net_profit = summary['metrics'][net_profit_metric]['value']
            if net_profit > best_net_profit:
                best_net_profit = net_profit
                best_strategy = strategy
    
    return {
        'period': period,
        'strategies': results,
        'best_strategy': best_strategy,
        'best_net_profit': best_net_profit
    }

def schedule_regular_aggregation(interval: int = 3600) -> threading.Thread:
    """
    주기적인 성능 지표 집계 스케줄링
    
    Args:
        interval (int): 검사 주기 (초)
        
    Returns:
        threading.Thread: 스케줄링 스레드
    """
    def aggregation_task():
        logger.info(f"성능 지표 집계 스케줄러 시작 (검사 주기: {interval}초)")
        
        while True:
            try:
                now = datetime.now()
                
                # 자정에 가까울 때 집계 실행
                if now.hour == 0 and now.minute < 10:
                    # 오늘이 월요일이면 주간 집계 실행
                    if now.weekday() == 0:
                        logger.info("주간 성능 지표 집계 시작")
                        aggregate_performance_metrics(period="weekly")
                    
                    # 오늘이 월초이면 월간 집계 실행
                    if now.day == 1:
                        logger.info("월간 성능 지표 집계 시작")
                        aggregate_performance_metrics(period="monthly")
                
                # 지정된 간격으로 대기
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"성능 지표 집계 스케줄링 중 오류: {str(e)}")
                time.sleep(interval)  # 오류 발생해도 계속 실행
    
    # 스레드 생성 및 시작
    thread = threading.Thread(target=aggregation_task, daemon=True)
    thread.start()
    
    return thread 