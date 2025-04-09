"""
Settings module for Bitcoin Trading Bot

This module contains all configuration settings for the trading system.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
DATA_DIR = os.path.join(BASE_DIR, 'data_storage')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

# 데이터 관련 하위 디렉토리 - 일관성 있는 경로 관리를 위해 추가
OHLCV_DIR = os.path.join(DATA_DIR, 'ohlcv')
INDICATORS_DIR = os.path.join(DATA_DIR, 'indicators')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
COMBINED_DIR = os.path.join(DATA_DIR, 'combined')
STACKED_DIR = os.path.join(DATA_DIR, 'stacked')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
STATE_DIR = os.path.join(DATA_DIR, 'state')

# 모델 관련 하위 디렉토리
MODEL_CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')
MODEL_EVALUATION_DIR = os.path.join(MODELS_DIR, 'evaluation')

# Create necessary directories
for directory in [DATA_DIR, LOG_DIR, MODELS_DIR, OHLCV_DIR, INDICATORS_DIR, 
                 FEATURES_DIR, RESULTS_DIR, COMBINED_DIR, STACKED_DIR, RAW_DIR, 
                 STATE_DIR, MODEL_CHECKPOINTS_DIR, MODEL_EVALUATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# API Keys - Load from environment variables or .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

UPBIT_ACCESS_KEY = os.environ.get('UPBIT_ACCESS_KEY', '')
UPBIT_SECRET_KEY = os.environ.get('UPBIT_SECRET_KEY', '')

# Default Market
DEFAULT_MARKET = "KRW-BTC"

# Trading Settings
TRADING_AMOUNT = float(os.environ.get('TRADING_AMOUNT', 10000000))  # 기본 거래 금액 (KRW)
FEE = 0.0005  # 거래 수수료 (0.05%)
TRADING_FEE = FEE  # 하위 호환성 유지
SLIPPAGE = 0.0002  # 슬리피지 (0.02%)

# Risk Management
MAX_TRADES_PER_DAY = 5
MAX_POSITION_SIZE = 0.2  # 최대 포지션 크기 (계좌의 비율)
STOP_LOSS_PERCENTAGE = 0.01  # 손절매 비율 (1%)
TAKE_PROFIT_PERCENTAGE = 0.025  # 이익실현 비율 (2.5%)
TRAILING_STOP_PERCENTAGE = 0.008  # 트레일링 스탑 비율 (0.8%)

# Backtest settings
from datetime import datetime, timedelta
BACKTEST_START_DATE = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
BACKTEST_END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
BACKTEST_COMMISSION = FEE  # BACKTEST_COMMISSION도 FEE로 통일하고 호환성 위해 유지
BACKTEST_SLIPPAGE = SLIPPAGE  # 슬리피지

# 데이터 관리 설정
DATA_MAX_CANDLES = {
    'day': 1000,
    'hour': 500,
    'minute5': 1000,
    'minute1': 500
}

DATA_UPDATE_INTERVALS = {
    'day': 86400,    # 24시간
    'hour': 3600,    # 1시간
    'minute5': 900,  # 15분
    'minute1': 300   # 5분
}

DATA_INITIAL_COUNTS = {
    'day': 100,
    'hour': 168,  # 1주일
    'minute5': 1000,
    'minute1': 500
}

DATA_UPDATE_COUNTS = {
    'day': 5,
    'hour': 24,
    'minute5': 100,
    'minute1': 100
}

# Model Settings
MODEL_VERSION = "1.0.0"
PREDICTION_INTERVAL = 60  # 예측 주기 (초)
RETRAINING_INTERVAL = 24  # 모델 재학습 주기 (시간)

# GPT Settings
USE_GPT = False  # GPT 사용 여부 (백테스트에서는 False, 실제 매매에서는 True)
GPT_MODEL = "gpt-4o"  # 사용할 GPT 모델
GPT_MAX_TOKENS = 1000  # 최대 토큰 수
GPT_TEMPERATURE = 0.2  # GPT 온도 설정
GPT_MARKET_CONTEXT_DAYS = 7  # 분석할 시장 데이터 기간
GPT_INCLUDE_TECHNICAL_INDICATORS = True  # 기술적 지표 포함 여부
GPT_INCLUDE_MARKET_SENTIMENT = True  # 시장 심리 포함 여부
GPT_INCLUDE_NEWS = False  # 뉴스 포함 여부

# Technical Indicator Settings
INDICATOR_SETTINGS = {
    'sma_periods': [7, 25, 99],
    'ema_periods': [12, 26, 200],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2.0,
    'atr_period': 14
}

# Strategy Settings
STRATEGY_SETTINGS = {
    'momentum': {
        'enabled': True,
        'lookback_period': 14,
        'entry_threshold': 0.01,
        'exit_threshold': -0.005
    },
    'mean_reversion': {
        'enabled': True,
        'lookback_period': 20,
        'entry_z_score': 2.0,
        'exit_z_score': 0.5
    },
    'trend_following': {
        'enabled': True,
        'fast_period': 10,
        'slow_period': 30,
        'signal_period': 9
    }
}

# Monitoring Settings
MONITOR_PERFORMANCE = True
MONITOR_RESOURCE_USAGE = True
API_CALL_LIMIT = 100  # API 호출 제한 (분당)
PERFORMANCE_ALERT_THRESHOLD = 0.1  # 성능 알림 임계값 (10% 하락)

# Notification Settings
ENABLE_NOTIFICATIONS = True
NOTIFICATION_METHODS = ['console']  # ['email', 'console', 'telegram']
CRITICAL_ALERT_METHODS = ['console']  # ['sms', 'email', 'console', 'telegram']

# Logging Settings
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_ROTATION = '1 day'
LOG_RETENTION = '30 days'

# Recovery Settings
RECOVERY_ENABLED = True
STATE_SAVE_INTERVAL = 300  # 상태 저장 주기 (초)
CHECKPOINT_INTERVAL = 1800  # 체크포인트 저장 주기 (초)
MAX_RETRY_ATTEMPTS = 3  # 최대 재시도 횟수

# Performance Evaluation
EVALUATION_METRICS = [
    'total_return',
    'annualized_return',
    'max_drawdown',
    'win_rate',
    'sharpe_ratio',
    'sortino_ratio',
    'calmar_ratio',
    'volatility'
]

# Target Monthly Return
TARGET_MONTHLY_RETURN = 0.05  # 목표 월간 수익률 (5%)

# Streaming settings
STREAM_DATA = False
STREAM_INTERVAL = 1  # 스트리밍 주기 (초)

# Scheduled job settings
SCHEDULE_JOBS = True
MARKET_DATA_UPDATE_INTERVAL = 3600  # 시장 데이터 업데이트 주기 (초)
MODEL_PREDICTION_INTERVAL = 900  # 모델 예측 주기 (초)
POSITION_CHECK_INTERVAL = 300  # 포지션 체크 주기 (초)

# User Interface
WEB_UI_ENABLED = False
WEB_UI_PORT = 8501
WEB_UI_THEME = 'dark'

# Load custom settings from YAML file if it exists
CUSTOM_SETTINGS_FILE = os.path.join(CONFIG_DIR, 'custom_settings.yaml')

if os.path.exists(CUSTOM_SETTINGS_FILE):
    try:
        with open(CUSTOM_SETTINGS_FILE, 'r') as file:
            custom_settings = yaml.safe_load(file)
            
            # Update settings from YAML
            for key, value in custom_settings.items():
                if key in globals():
                    globals()[key] = value
    except Exception as e:
        print(f"Error loading custom settings: {str(e)}")


def save_custom_settings(settings_dict: Dict[str, Any]) -> bool:
    """
    Save custom settings to YAML file
    
    Args:
        settings_dict (Dict[str, Any]): Dictionary of settings to save
        
    Returns:
        bool: Success status
    """
    try:
        # Backup existing settings file if it exists
        if os.path.exists(CUSTOM_SETTINGS_FILE):
            backup_file = f"{CUSTOM_SETTINGS_FILE}.{datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
            os.rename(CUSTOM_SETTINGS_FILE, backup_file)
        
        # Save new settings
        with open(CUSTOM_SETTINGS_FILE, 'w') as file:
            yaml.dump(settings_dict, file, default_flow_style=False)
        
        return True
    except Exception as e:
        print(f"Error saving custom settings: {str(e)}")
        return False


def get_all_settings() -> Dict[str, Any]:
    """
    Get all current settings as dictionary
    
    Returns:
        Dict[str, Any]: All current settings
    """
    # Get all uppercase variables (conventional for settings)
    settings = {key: value for key, value in globals().items() 
               if key.isupper() and not key.startswith('_')}
    
    # Convert non-serializable objects to strings
    for key, value in settings.items():
        if isinstance(value, Path):
            settings[key] = str(value)
    
    return settings


def print_settings(section: Optional[str] = None) -> None:
    """
    Print current settings
    
    Args:
        section (Optional[str], optional): Specific section to print. Defaults to None.
    """
    settings = get_all_settings()
    
    # Filter by section if specified
    if section:
        section = section.upper()
        filtered_settings = {k: v for k, v in settings.items() if k.startswith(section)}
        print(f"\n=== {section} Settings ===")
        for key, value in filtered_settings.items():
            print(f"{key}: {value}")
    else:
        # Group settings by prefix
        sections = {}
        for key, value in settings.items():
            prefix = key.split('_')[0]
            if prefix not in sections:
                sections[prefix] = {}
            sections[prefix][key] = value
        
        # Print each section
        for section_name, section_settings in sections.items():
            print(f"\n=== {section_name} Settings ===")
            for key, value in section_settings.items():
                print(f"{key}: {value}") 