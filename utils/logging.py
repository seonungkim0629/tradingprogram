"""
Logging module for Bitcoin Trading Bot

This module provides logging functionality for the trading system.
"""

import os
import sys
import logging
import logging.handlers
import functools
import time
import traceback
from datetime import datetime
from typing import Callable, Any, Dict, Optional, Union, List, Tuple
import threading

from config import settings

# Set up logging directory
os.makedirs(settings.LOG_DIR, exist_ok=True)

# Create logs for different components
SYSTEM_LOG = os.path.join(settings.LOG_DIR, 'system.log')
TRADING_LOG = os.path.join(settings.LOG_DIR, 'trading.log')
API_LOG = os.path.join(settings.LOG_DIR, 'api.log')
PERFORMANCE_LOG = os.path.join(settings.LOG_DIR, 'performance.log')
ERROR_LOG = os.path.join(settings.LOG_DIR, 'error.log')
BACKTEST_LOG = os.path.join(settings.LOG_DIR, 'backtest.log')

# Configure root logger
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 백테스트 관련 변수
BACKTEST_MODE = False

# Disable overly verbose loggers
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
# 추가 라이브러리 로그 레벨 조정
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('optuna').setLevel(logging.WARNING)
logging.getLogger('hyperopt').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)
logging.getLogger('py4j').setLevel(logging.WARNING)
logging.getLogger('h5py').setLevel(logging.WARNING)
logging.getLogger('nose').setLevel(logging.WARNING)


def get_file_handler(log_file: str, log_level: int = logging.INFO) -> logging.FileHandler:
    """
    Create a file handler for the specified log file
    
    Args:
        log_file (str): Path to the log file
        log_level (int, optional): Logging level. Defaults to logging.INFO.
        
    Returns:
        logging.FileHandler: Configured file handler
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
    return file_handler


def get_rotating_file_handler(log_file: str, log_level: int = logging.INFO) -> logging.handlers.TimedRotatingFileHandler:
    """
    Create a rotating file handler for the specified log file
    
    Args:
        log_file (str): Path to the log file
        log_level (int, optional): Logging level. Defaults to logging.INFO.
        
    Returns:
        logging.handlers.TimedRotatingFileHandler: Configured rotating file handler
    """
    try:
        handler = logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=30,  # Keep 30 days of logs
            delay=True  # Delay file creation until first log
        )
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
        
        # Safer rotation by overriding the existing rotate method
        original_rotate = handler.rotate
        
        def safe_rotate(source, dest):
            try:
                if os.path.exists(dest):
                    os.remove(dest)  # Remove destination if exists
                original_rotate(source, dest)
            except (OSError, PermissionError) as e:
                print(f"Could not rotate log file {source} to {dest}: {e}")
                # Attempt to create a new file with timestamp in the name
                import uuid
                new_dest = f"{dest}.{uuid.uuid4().hex[:8]}"
                try:
                    original_rotate(source, new_dest)
                except:
                    pass  # Last resort: just continue without rotation
        
        handler.rotate = safe_rotate
        return handler
    except Exception as e:
        print(f"Error creating log handler for {log_file}: {e}")
        # Fallback to a simple file handler that appends
        try:
            simple_handler = logging.FileHandler(log_file, mode='a')
            simple_handler.setLevel(log_level)
            simple_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
            return simple_handler
        except:
            # Last resort: just return a null handler
            return logging.NullHandler()


def set_backtest_mode(enabled: bool = True):
    """
    백테스트 모드 설정 - 백테스트 시 로깅 레벨 조정
    
    Args:
        enabled (bool): 백테스트 모드 활성화 여부
    """
    global BACKTEST_MODE
    BACKTEST_MODE = enabled
    
    if enabled:
        # 백테스트 모드에서는 중요한 로그만 출력하도록 조정
        logging.getLogger('strategies').setLevel(logging.WARNING)
        logging.getLogger('data').setLevel(logging.WARNING)
        logging.getLogger('models').setLevel(logging.WARNING)
        logging.getLogger('backtest').setLevel(logging.INFO)  # 백테스트 결과는 유지
    else:
        # 일반 모드에서는 기본 로그 레벨로 복원
        logging.getLogger('strategies').setLevel(settings.LOG_LEVEL)
        logging.getLogger('data').setLevel(settings.LOG_LEVEL)
        logging.getLogger('models').setLevel(settings.LOG_LEVEL)
        logging.getLogger('backtest').setLevel(settings.LOG_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """로거 생성"""
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 이미 핸들러가 있다면 추가하지 않음
    if logger.handlers:
        return logger
        
    # 로그 디렉토리 생성
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # 파일 핸들러 추가
    log_file = os.path.join('logs', 'trading.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    
    return logger


# Create a global logger for this module
logger = get_logger(__name__)


def log_execution(func: Callable) -> Callable:
    """
    Decorator to log function execution details
    
    Args:
        func (Callable): The function to decorate
        
    Returns:
        Callable: Decorated function
    """
    # 이미 처리 중인 함수 호출을 추적하기 위한 스레드별 세트
    _active_executions = threading.local()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        module_name = func.__module__
        func_id = f"{module_name}.{func_name}"
        
        # 재귀적 호출 감지 및 방지
        if not hasattr(_active_executions, 'funcs'):
            _active_executions.funcs = set()
            
        # 이미 실행 중인 경우 로깅 없이 원래 함수 실행
        if func_id in _active_executions.funcs:
            return func(*args, **kwargs)
            
        # 현재 실행 중인 함수 목록에 추가
        _active_executions.funcs.add(func_id)
        
        try:
            # 백테스트 모드에서는 디버그 로그를 남기지 않음
            if not BACKTEST_MODE or module_name.startswith('backtest'):
                logger.debug(f"Executing {func_id}")
            
            # 함수 실행
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 백테스트 모드에서 불필요한 로그는 출력하지 않음
            if BACKTEST_MODE and not module_name.startswith('backtest'):
                return result
                
            # 실행 시간에 따라 로그 레벨 결정
            if execution_time > 5.0:
                logger.debug(f"Slow execution of {func_id} in {execution_time:.2f}s")
            elif execution_time > 1.0:
                logger.info(f"Completed {func_id} in {execution_time:.2f}s")
            else:
                logger.debug(f"Completed {func_id} in {execution_time:.2f}s")
                
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error in {func_id} after {execution_time:.2f}s: {str(e)}"
            logger.error(error_msg)
            
            # 스택 트레이스 로깅 (verbose=False로 간결하게 설정)
            logger.error(traceback.format_exc())
            raise
            
        finally:
            # 항상 실행 목록에서 제거 (오류가 발생해도)
            _active_executions.funcs.remove(func_id)
    
    return wrapper


def log_trade(trade_type: str, ticker: str, price: float, amount: float, 
              total: float, reason: str = None) -> None:
    """
    Log a trade execution
    
    Args:
        trade_type (str): Type of trade ('BUY' or 'SELL')
        ticker (str): Ticker symbol
        price (float): Price per unit
        amount (float): Amount of units
        total (float): Total value of the trade
        reason (str, optional): Reason for the trade. Defaults to None.
    """
    # 데이터베이스에 저장
    try:
        from utils.database import save_trade
        
        # 추가 메타데이터 생성 (필요시 확장)
        metadata = {}
        if reason:
            metadata['reason_details'] = reason
        
        # 데이터베이스에 저장
        save_trade(
            trade_type=trade_type.upper(), 
            ticker=ticker, 
            price=price, 
            amount=amount, 
            total=total,
            reason=reason,
            metadata=metadata
        )
    except ImportError:
        # utils.database를 import할 수 없는 경우, 기존 로깅 방식 사용
        trade_logger = get_logger("trading.execution")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {trade_type} {ticker}: {amount:.8f} @ {price:,.0f} = {total:,.0f} KRW"
        
        if reason:
            log_message += f" | Reason: {reason}"
        
        trade_logger.debug(log_message)  # INFO에서 DEBUG로 변경


def log_strategy_signal(strategy_name: str, ticker: str, signal_type: str, 
                        confidence: float = None, details: Dict[str, Any] = None) -> None:
    """
    Log a strategy signal
    
    Args:
        strategy_name (str): Name of the strategy
        ticker (str): Ticker symbol
        signal_type (str): Type of signal (e.g., 'BUY', 'SELL', 'HOLD')
        confidence (float, optional): Signal confidence (0.0 to 1.0). Defaults to None.
        details (Dict[str, Any], optional): Additional signal details. Defaults to None.
    """
    # 데이터베이스에 저장
    try:
        from utils.database import save_signal
        
        # 데이터베이스에 저장
        save_signal(
            strategy=strategy_name,
            ticker=ticker,
            signal_type=signal_type.upper(),
            confidence=confidence,
            details=details
        )
    except ImportError:
        # utils.database를 import할 수 없는 경우, 기존 로깅 방식 사용 (DEBUG 레벨로 변경)
        strategy_logger = get_logger(f"strategy.{strategy_name}")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if confidence is not None:
            confidence_str = f"{confidence:.2f}"
            log_message = f"[{timestamp}] {strategy_name} - {signal_type} signal for {ticker} (confidence: {confidence_str})"
        else:
            log_message = f"[{timestamp}] {strategy_name} - {signal_type} signal for {ticker}"
        
        if details:
            details_str = " | " + " | ".join([f"{k}: {v}" for k, v in details.items()])
            log_message += details_str
        
        strategy_logger.debug(log_message)  # INFO에서 DEBUG로 변경


def log_portfolio_update(portfolio_value: float, cash: float, holdings: Dict[str, Dict[str, float]], 
                        daily_change: float = None, total_profit: float = None) -> None:
    """
    Log portfolio update
    
    Args:
        portfolio_value (float): Total portfolio value
        cash (float): Cash balance
        holdings (Dict[str, Dict[str, float]]): Holdings details
        daily_change (float, optional): Daily change percentage. Defaults to None.
        total_profit (float, optional): Total profit. Defaults to None.
    """
    portfolio_logger = get_logger("portfolio.updates")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Basic portfolio information
    log_message = f"[{timestamp}] Portfolio value: {portfolio_value:,.0f} KRW | Cash: {cash:,.0f} KRW"
    
    # Add daily change if provided
    if daily_change is not None:
        change_symbol = "↑" if daily_change >= 0 else "↓"
        log_message += f" | Daily change: {change_symbol} {abs(daily_change):.2f}%"
    
    # Add total profit if provided
    if total_profit is not None:
        profit_symbol = "+" if total_profit >= 0 else "-"
        log_message += f" | Total P/L: {profit_symbol}{abs(total_profit):,.0f} KRW"
    
    portfolio_logger.info(log_message)
    
    # Log holdings details
    if holdings:
        for ticker, details in holdings.items():
            amount = details.get('amount', 0)
            value = details.get('value', 0)
            avg_price = details.get('avg_price', 0)
            current_price = details.get('current_price', 0)
            profit_loss = details.get('profit_loss', 0)
            profit_loss_pct = details.get('profit_loss_pct', 0)
            
            pl_symbol = "+" if profit_loss >= 0 else "-"
            pl_pct_symbol = "+" if profit_loss_pct >= 0 else "-"
            
            holding_message = (
                f"[{timestamp}] {ticker}: {amount:.8f} | "
                f"Value: {value:,.0f} KRW | "
                f"Avg price: {avg_price:,.0f} KRW | "
                f"Current: {current_price:,.0f} KRW | "
                f"P/L: {pl_symbol}{abs(profit_loss):,.0f} KRW ({pl_pct_symbol}{abs(profit_loss_pct):.2f}%)"
            )
            
            portfolio_logger.info(holding_message)


def log_error(module: str, error_message: str, exception: Exception = None, 
             critical: bool = False, notify: bool = False) -> None:
    """
    Log an error with detailed information
    
    Args:
        module (str): Module where the error occurred
        error_message (str): Error message
        exception (Exception, optional): Exception object. Defaults to None.
        critical (bool, optional): Whether the error is critical. Defaults to False.
        notify (bool, optional): Whether to send notifications. Defaults to False.
    """
    error_logger = get_logger(f"error.{module}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine log level and prefix
    if critical:
        log_level = logging.CRITICAL
        prefix = "CRITICAL ERROR"
    else:
        log_level = logging.ERROR
        prefix = "ERROR"
    
    # Create error message
    log_message = f"[{timestamp}] {prefix} in {module}: {error_message}"
    
    # Add exception details if provided
    if exception:
        log_message += f"\nException: {str(exception)}\n{traceback.format_exc()}"
    
    # Log the error
    error_logger.log(log_level, log_message)
    
    # Handle notifications if needed
    if notify:
        # TODO: Implement notification system
        pass 