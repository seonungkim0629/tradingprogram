"""
Monitoring module for Bitcoin Trading Bot

This module provides functionality to monitor system resources, 
API usage, and trading bot performance.
"""

import os
import time
import psutil
import threading
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
import json
import pandas as pd
import numpy as np
from collections import deque

from config import settings
from utils.logging import get_logger, log_execution

# Initialize logger
logger = get_logger(__name__)

# Initialize monitoring data structures
api_call_counts = {
    'upbit': deque(maxlen=1440),  # Store 24 hours of data (per minute)
    'total_daily': 0,
    'total_since_start': 0,
    'last_reset': datetime.now()
}

performance_metrics = {
    'memory_usage': deque(maxlen=1440),  # 24 hours of data
    'cpu_usage': deque(maxlen=1440),
    'disk_usage': deque(maxlen=1440),
    'api_latency': deque(maxlen=1440),
    'last_updated': datetime.now()
}

trading_metrics = {
    'trades_today': 0,
    'successful_trades': 0,
    'failed_trades': 0,
    'profit_today': 0.0,
    'total_profit': 0.0,
    'win_rate': 0.0,
    'avg_profit_per_trade': 0.0,
    'max_drawdown': 0.0,
    'current_drawdown': 0.0,
    'peak_portfolio_value': 0.0,
    'last_updated': datetime.now()
}

# Lock for thread safety
metrics_lock = threading.Lock()


@log_execution
def track_api_call(api_name: str, endpoint: str = 'general', latency_ms: float = None) -> None:
    """
    Track API call to avoid rate limiting
    
    Args:
        api_name (str): Name of the API (e.g., 'upbit')
        endpoint (str, optional): Specific API endpoint. Defaults to 'general'.
        latency_ms (float, optional): API call latency in milliseconds. Defaults to None.
    """
    with metrics_lock:
        # Create current timestamp
        now = datetime.now()
        
        # Initialize if new API
        if api_name not in api_call_counts:
            api_call_counts[api_name] = deque(maxlen=1440)
        
        # Store the API call with timestamp
        api_call_counts[api_name].append({
            'timestamp': now,
            'endpoint': endpoint,
            'latency_ms': latency_ms
        })
        
        # Increment total counts
        api_call_counts['total_daily'] += 1
        api_call_counts['total_since_start'] += 1
        
        # Store latency if provided
        if latency_ms is not None:
            performance_metrics['api_latency'].append({
                'timestamp': now,
                'api': api_name,
                'endpoint': endpoint,
                'latency_ms': latency_ms
            })
        
        # Check if we should reset daily counter
        if (now - api_call_counts['last_reset']).days >= 1:
            api_call_counts['total_daily'] = 1  # Count the current call
            api_call_counts['last_reset'] = now
            logger.info(f"Reset daily API call counter. Total calls since start: {api_call_counts['total_since_start']}")


@log_execution
def get_api_call_count(api_name: str, minutes: int = 1) -> int:
    """
    Get API call count for the specified time window
    
    Args:
        api_name (str): Name of the API
        minutes (int, optional): Time window in minutes. Defaults to 1.
        
    Returns:
        int: Number of API calls in the specified window
    """
    with metrics_lock:
        if api_name not in api_call_counts:
            return 0
        
        # Calculate time threshold
        threshold = datetime.now() - timedelta(minutes=minutes)
        
        # Count calls after the threshold
        count = sum(1 for call in api_call_counts[api_name] if call['timestamp'] >= threshold)
        
        return count


@log_execution
def is_rate_limited(api_name: str, limit: int = None, minutes: int = 1) -> bool:
    """
    Check if rate limit is exceeded
    
    Args:
        api_name (str): Name of the API
        limit (int, optional): Rate limit. Defaults to settings.API_CALL_LIMIT.
        minutes (int, optional): Time window in minutes. Defaults to 1.
        
    Returns:
        bool: True if rate limited, False otherwise
    """
    if limit is None:
        limit = settings.API_CALL_LIMIT
    
    count = get_api_call_count(api_name, minutes)
    
    # Log warning if approaching limit
    if count >= limit * 0.8:
        logger.warning(f"Approaching API rate limit for {api_name}: {count}/{limit} calls in {minutes} minute(s)")
    
    return count >= limit


@log_execution
def log_resource_usage() -> Dict[str, float]:
    """
    Log system resource usage
    
    Returns:
        Dict[str, float]: Dictionary of resource usage metrics
    """
    try:
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Store metrics
        with metrics_lock:
            now = datetime.now()
            
            performance_metrics['memory_usage'].append({
                'timestamp': now,
                'percent': memory_percent
            })
            
            performance_metrics['cpu_usage'].append({
                'timestamp': now,
                'percent': cpu_percent
            })
            
            performance_metrics['disk_usage'].append({
                'timestamp': now,
                'percent': disk_percent
            })
            
            performance_metrics['last_updated'] = now
        
        # Log if resources are running low
        if memory_percent > 90:
            logger.warning(f"High memory usage: {memory_percent:.1f}%")
        if cpu_percent > 90:
            logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
        if disk_percent > 90:
            logger.warning(f"Low disk space: {disk_percent:.1f}%")
        
        # Return current metrics
        return {
            'memory_percent': memory_percent,
            'cpu_percent': cpu_percent,
            'disk_percent': disk_percent
        }
    
    except Exception as e:
        logger.error(f"Error logging resource usage: {str(e)}")
        return {}


@log_execution
def track_trade(trade_type: str, ticker: str, 
                amount: float, price: float, total: float, 
                is_successful: bool, profit_loss: float = 0.0) -> None:
    """
    Track trade performance metrics
    
    Args:
        trade_type (str): Type of trade ('BUY' or 'SELL')
        ticker (str): Ticker symbol
        amount (float): Amount of units
        price (float): Price per unit
        total (float): Total value of the trade
        is_successful (bool): Whether the trade was successful
        profit_loss (float, optional): Profit or loss from the trade. Defaults to 0.0.
    """
    with metrics_lock:
        # Update trades count
        trading_metrics['trades_today'] += 1
        
        if is_successful:
            trading_metrics['successful_trades'] += 1
        else:
            trading_metrics['failed_trades'] += 1
        
        # Update profit metrics for SELL trades
        if trade_type.upper() == 'SELL' and is_successful:
            trading_metrics['profit_today'] += profit_loss
            trading_metrics['total_profit'] += profit_loss
        
        # Calculate win rate
        total_trades = trading_metrics['successful_trades'] + trading_metrics['failed_trades']
        if total_trades > 0:
            trading_metrics['win_rate'] = trading_metrics['successful_trades'] / total_trades
        
        # Calculate average profit per trade
        if trading_metrics['successful_trades'] > 0:
            trading_metrics['avg_profit_per_trade'] = trading_metrics['total_profit'] / trading_metrics['successful_trades']
        
        # Update timestamp
        trading_metrics['last_updated'] = datetime.now()


@log_execution
def update_portfolio_metrics(current_value: float) -> None:
    """
    Update portfolio metrics including drawdown
    
    Args:
        current_value (float): Current portfolio value
    """
    with metrics_lock:
        # Update peak value if current value is higher
        if current_value > trading_metrics['peak_portfolio_value']:
            trading_metrics['peak_portfolio_value'] = current_value
        
        # Calculate current drawdown
        if trading_metrics['peak_portfolio_value'] > 0:
            current_drawdown = (trading_metrics['peak_portfolio_value'] - current_value) / trading_metrics['peak_portfolio_value']
            trading_metrics['current_drawdown'] = current_drawdown
            
            # Update max drawdown if current drawdown is greater
            if current_drawdown > trading_metrics['max_drawdown']:
                trading_metrics['max_drawdown'] = current_drawdown
        
        # Update timestamp
        trading_metrics['last_updated'] = datetime.now()


@log_execution
def reset_daily_metrics() -> None:
    """
    Reset daily metrics (called at market close or midnight)
    """
    with metrics_lock:
        trading_metrics['trades_today'] = 0
        trading_metrics['profit_today'] = 0.0
        api_call_counts['total_daily'] = 0
        api_call_counts['last_reset'] = datetime.now()
        logger.info("Daily metrics have been reset")


@log_execution
def get_system_info() -> Dict[str, str]:
    """
    Get system information
    
    Returns:
        Dict[str, str]: System information
    """
    try:
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'ram': f"{round(psutil.virtual_memory().total / (1024.0 ** 3))} GB",
            'python_version': platform.python_version()
        }
        return info
    
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {'error': str(e)}


@log_execution
def get_performance_summary() -> Dict[str, Any]:
    """
    Get performance summary
    
    Returns:
        Dict[str, Any]: Performance summary
    """
    with metrics_lock:
        # Calculate average resource usage over the last hour
        hour_ago = datetime.now() - timedelta(hours=1)
        
        memory_avg = 0.0
        if performance_metrics['memory_usage']:
            memory_hour = [m['percent'] for m in performance_metrics['memory_usage'] 
                          if m['timestamp'] >= hour_ago]
            memory_avg = sum(memory_hour) / len(memory_hour) if memory_hour else 0.0
        
        cpu_avg = 0.0
        if performance_metrics['cpu_usage']:
            cpu_hour = [c['percent'] for c in performance_metrics['cpu_usage'] 
                       if c['timestamp'] >= hour_ago]
            cpu_avg = sum(cpu_hour) / len(cpu_hour) if cpu_hour else 0.0
        
        api_latency_avg = 0.0
        if performance_metrics['api_latency']:
            latency_hour = [l['latency_ms'] for l in performance_metrics['api_latency'] 
                           if l['timestamp'] >= hour_ago and l['latency_ms'] is not None]
            api_latency_avg = sum(latency_hour) / len(latency_hour) if latency_hour else 0.0
        
        # Compile summary
        summary = {
            'system': {
                'memory_usage_avg_1h': memory_avg,
                'cpu_usage_avg_1h': cpu_avg,
                'disk_usage_current': performance_metrics['disk_usage'][-1]['percent'] if performance_metrics['disk_usage'] else 0.0
            },
            'api': {
                'calls_today': api_call_counts['total_daily'],
                'calls_total': api_call_counts['total_since_start'],
                'latency_avg_1h_ms': api_latency_avg
            },
            'trading': {
                'trades_today': trading_metrics['trades_today'],
                'profit_today': trading_metrics['profit_today'],
                'total_profit': trading_metrics['total_profit'],
                'win_rate': trading_metrics['win_rate'],
                'avg_profit_per_trade': trading_metrics['avg_profit_per_trade'],
                'current_drawdown': trading_metrics['current_drawdown'],
                'max_drawdown': trading_metrics['max_drawdown']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return summary


@log_execution
def save_performance_data(file_path: str = None) -> str:
    """
    Save performance data to file
    
    Args:
        file_path (str, optional): File path. Defaults to auto-generated path.
        
    Returns:
        str: Path to the saved file or empty string on error
    """
    try:
        # Generate file path if not provided
        if file_path is None:
            os.makedirs(os.path.join(settings.DATA_DIR, 'performance'), exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(settings.DATA_DIR, 'performance', f'performance_{timestamp}.json')
        
        # Get summary data
        summary = get_performance_summary()
        
        # Add system info
        summary['system_info'] = get_system_info()
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Performance data saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving performance data: {str(e)}")
        return ""


# Initialize monitoring thread
def start_monitoring(interval: int = 300) -> threading.Thread:
    """
    Start monitoring thread
    
    Args:
        interval (int, optional): Monitoring interval in seconds. Defaults to 300 (5 minutes).
        
    Returns:
        threading.Thread: Monitoring thread
    """
    def monitoring_task():
        logger.info(f"Starting system monitoring (interval: {interval}s)")
        while True:
            try:
                # Log resource usage
                log_resource_usage()
                
                # Check if we need to reset daily metrics
                now = datetime.now()
                last_reset = api_call_counts['last_reset']
                if now.date() > last_reset.date():
                    reset_daily_metrics()
                
                # Sleep for the specified interval
                time.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in monitoring task: {str(e)}")
                time.sleep(interval)  # Still sleep to avoid rapid error loops
    
    # Create and start the monitoring thread
    thread = threading.Thread(target=monitoring_task, daemon=True)
    thread.start()
    return thread


# Initialize monitoring data at module load time
if settings.MONITOR_RESOURCE_USAGE:
    # Set initial peak portfolio value to avoid division by zero
    trading_metrics['peak_portfolio_value'] = 1.0 