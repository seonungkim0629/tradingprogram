"""
Database utility for trading message storage

This module provides functionality for saving and retrieving trading messages
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from utils.logging import get_logger
from utils.metadata import create_standard_metadata, normalize_metadata, metadata_to_json

# Initialize logger
logger = get_logger(__name__)

# Define database path
DB_DIR = os.path.join("data", "db")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "trading_data.db")

def get_db_connection():
    """
    데이터베이스 연결 반환
    
    Returns:
        sqlite3.Connection: 데이터베이스 연결 객체
    """
    try:
        # DB 디렉토리 확인
        db_dir = os.path.dirname(DB_PATH)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        # SQLite 연결
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"데이터베이스 연결 중 오류 발생: {str(e)}")
        return None

def initialize_database():
    """
    데이터베이스 초기화
    """
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("데이터베이스 연결 실패")
            return
            
        cursor = conn.cursor()
        
        # Create a table for trades
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            ticker TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            total REAL NOT NULL,
            reason TEXT,
            strategy TEXT,
            confidence REAL,
            metadata TEXT
        )
        ''')
        
        # Create a table for signals
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            strategy TEXT NOT NULL,
            ticker TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            confidence REAL,
            metadata TEXT
        )
        ''')
        
        # Create a table for performance metrics
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            strategy TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            period TEXT NOT NULL,
            details TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("데이터베이스 초기화 완료")
        
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")

# 모듈 임포트 시 데이터베이스 초기화
initialize_database()

def save_trade(trade_type, ticker, price, amount, total, reason=None, strategy=None, confidence=None, metadata=None):
    """
    트레이드 정보를 데이터베이스에 저장

    Args:
        trade_type (str): 트레이드 유형 ('buy' 또는 'sell')
        ticker (str): 거래 심볼
        price (float): 거래 가격
        amount (float): 거래 수량
        total (float): 총 거래 금액
        reason (str, optional): 거래 이유
        strategy (str, optional): 거래 전략
        confidence (float, optional): 신호 신뢰도 (0-1)
        metadata (dict, optional): 추가 상세 정보
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 메타데이터 표준화
        if metadata is None:
            metadata = {}
        
        # 기본 필드 추가
        if 'strategy' not in metadata and strategy:
            metadata['strategy'] = strategy
        if 'strategy_type' not in metadata:
            metadata['strategy_type'] = 'unknown'
        
        standard_metadata = normalize_metadata(metadata, 'trade')
        metadata_json = metadata_to_json(standard_metadata)
        
        cursor.execute(
            """
            INSERT INTO trades 
            (timestamp, type, ticker, price, amount, total, reason, strategy, confidence, metadata) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (datetime.now().isoformat(), trade_type, ticker, price, amount, total, reason, strategy, confidence, metadata_json)
        )
        
        conn.commit()
        logger.debug(f"저장된 거래: {trade_type} {ticker} {amount} @ {price}")
    except Exception as e:
        logger.error(f"거래 저장 중 오류 발생: {str(e)}")
    finally:
        if conn:
            conn.close()

def save_signal(strategy, ticker, signal_type, confidence=None, details=None):
    """
    전략 신호를 데이터베이스에 저장

    Args:
        strategy (str): 전략 이름
        ticker (str): 거래 심볼
        signal_type (str): 신호 유형('buy', 'sell', 'hold')
        confidence (float, optional): 신호 신뢰도 (0-1)
        details (dict, optional): 추가 상세 정보
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 메타데이터 표준화
        if details is None:
            details = {}
            
        # 기본 필드 추가
        if 'strategy' not in details:
            details['strategy'] = strategy
        if 'strategy_type' not in details:
            details['strategy_type'] = 'unknown'
            
        standard_metadata = normalize_metadata(details, 'signal')
        metadata_json = metadata_to_json(standard_metadata)
        
        cursor.execute(
            """
            INSERT INTO signals 
            (timestamp, strategy, ticker, signal_type, confidence, metadata) 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.now().isoformat(), strategy, ticker, signal_type, confidence, metadata_json)
        )
        
        conn.commit()
        logger.debug(f"저장된 신호: {strategy} {ticker} {signal_type} (신뢰도: {confidence})")
    except Exception as e:
        logger.error(f"신호 저장 중 오류 발생: {str(e)}")
    finally:
        if conn:
            conn.close()

def save_performance_metric(strategy: str, metric_name: str, metric_value: float, 
                           period: str = "daily", details: Dict[str, Any] = None) -> bool:
    """
    성능 지표를 데이터베이스에 저장

    Args:
        strategy (str): 전략 이름
        metric_name (str): 지표 이름
        metric_value (float): 지표 값
        period (str, optional): 기간 ('daily', 'trade', 'weekly', 'monthly')
        details (dict, optional): 추가 상세 정보
        
    Returns:
        bool: 성공 여부
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 메타데이터 표준화
        if details is None:
            details = {}
            
        # 기본 필드 추가
        details['strategy'] = strategy
        details['period'] = period
        details['metric_name'] = metric_name
        details['metric_value'] = metric_value
            
        standard_metadata = normalize_metadata(details, 'performance')
        metadata_json = metadata_to_json(standard_metadata)
        
        cursor.execute(
            """
            INSERT INTO performance_metrics 
            (timestamp, strategy, metric_name, metric_value, period, details) 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.now().isoformat(), strategy, metric_name, metric_value, period, metadata_json)
        )
        
        conn.commit()
        logger.debug(f"저장된 성능 지표: {strategy} {metric_name} = {metric_value} ({period})")
        return True
    except Exception as e:
        logger.error(f"성능 지표 저장 중 오류 발생: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def get_recent_trades(limit: int = 10, ticker: str = None) -> List[Dict[str, Any]]:
    """
    Get recent trades from database
    
    Args:
        limit (int, optional): Maximum number of trades to retrieve. Defaults to 10.
        ticker (str, optional): Filter by ticker symbol. Defaults to None.
        
    Returns:
        List[Dict[str, Any]]: List of trade records
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM trades"
        params = []
        
        if ticker:
            query += " WHERE ticker = ?"
            params.append(ticker)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        trades = []
        for row in rows:
            trade = dict(row)
            # Parse JSON metadata
            if trade['metadata']:
                trade['metadata'] = json.loads(trade['metadata'])
            trades.append(trade)
        
        conn.close()
        return trades
    except Exception as e:
        logger.error(f"Error retrieving trades: {str(e)}")
        return []

def get_performance_history(strategy: str, metric_name: str = None, 
                           period: str = None, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Get performance history for a strategy
    
    Args:
        strategy (str): Strategy name
        metric_name (str, optional): Filter by metric name. Defaults to None.
        period (str, optional): Filter by period. Defaults to None.
        limit (int, optional): Maximum number of records. Defaults to 30.
        
    Returns:
        List[Dict[str, Any]]: List of performance records
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM performance_metrics WHERE strategy = ?"
        params = [strategy]
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        if period:
            query += " AND period = ?"
            params.append(period)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        records = []
        for row in rows:
            records.append(dict(row))
        
        conn.close()
        return records
    except Exception as e:
        logger.error(f"Error retrieving performance history: {str(e)}")
        return [] 