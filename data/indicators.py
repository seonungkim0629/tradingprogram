"""
Technical indicators module for Bitcoin Trading Bot

This module provides functionality to calculate various technical indicators
from market data to assist in trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator, ADXIndicator, AroonIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator, WilliamsRIndicator, AwesomeOscillatorIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, AccDistIndexIndicator, VolumeWeightedAveragePrice
from ta.others import DailyReturnIndicator
from sklearn.preprocessing import StandardScaler

from config import settings
from utils.logging import get_logger, log_execution
import traceback

# Initialize logger
logger = get_logger(__name__)

# --- 모든 가능한 지표 컬럼 목록 정의 ---
ALL_POSSIBLE_INDICATOR_COLUMNS = [
    # Basic Indicators
    'sma7', 'sma25', 'sma99', 'ema12', 'ema26', 'ema200', 'macd', 'macd_signal', 'macd_diff',
    'rsi14', 'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d', 'atr', 'obv',
    'daily_return', 'volatility_14',
    # Advanced Indicators
    'cci', 'tsi', 'mfi', 'adi', 'roc', 'ichimoku_tenkan_sen',
    'ichimoku_kijun_sen', 'ichimoku_senkou_span_a', 'ichimoku_senkou_span_b', 'ichimoku_chikou_span',
    'keltner_mband', 'keltner_hband', 'keltner_lband', 'hma9', 'ao', 'williams_r', 'vwap',
    # Custom Indicators
    'volume_change', 'price_volume_corr', 'market_breadth', 'volatility_ratio', 'trend_strength',
    # Pattern Recognition (using actual implemented function names)
    'doji',           # Previously CDLDOJI
    'hammer',         # Previously CDLHAMMER (includes hanging man as -1)
    'bullish_engulfing', # Previously part of CDLENGULFING
    'bearish_engulfing', # Previously part of CDLENGULFING
    'morning_star',   # Previously CDLMORNINGSTAR
    'evening_star',   # Previously CDLEVENINGSTAR
    'three_line_strike' # Previously CDL3LINESTRIKE
    # Add other indicators calculated in add_* functions if any
]

@log_execution
def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    try:
        # Add simple moving averages
        result['sma7'] = SMAIndicator(close=result['close'], window=7).sma_indicator()
        result['sma25'] = SMAIndicator(close=result['close'], window=25).sma_indicator()
        result['sma99'] = SMAIndicator(close=result['close'], window=99).sma_indicator()
        
        # Add exponential moving averages
        result['ema12'] = EMAIndicator(close=result['close'], window=12).ema_indicator()
        result['ema26'] = EMAIndicator(close=result['close'], window=26).ema_indicator()
        result['ema200'] = EMAIndicator(close=result['close'], window=200).ema_indicator()
        
        # MACD (Moving Average Convergence Divergence)
        macd = MACD(close=result['close'], window_slow=26, window_fast=12, window_sign=9)
        result['macd'] = macd.macd()
        result['macd_signal'] = macd.macd_signal()
        result['macd_diff'] = macd.macd_diff()
        
        # RSI (Relative Strength Index)
        result['rsi14'] = RSIIndicator(close=result['close'], window=14).rsi()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=result['close'], window=20, window_dev=2)
        result['bb_upper'] = bollinger.bollinger_hband()
        result['bb_middle'] = bollinger.bollinger_mavg()
        result['bb_lower'] = bollinger.bollinger_lband()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=result['high'], low=result['low'], close=result['close'], window=14, smooth_window=3)
        result['stoch_k'] = stoch.stoch()
        result['stoch_d'] = stoch.stoch_signal()
        
        # Average True Range (ATR)
        result['atr'] = AverageTrueRange(high=result['high'], low=result['low'], close=result['close'], window=14).average_true_range()
        
        # On-balance Volume
        result['obv'] = OnBalanceVolumeIndicator(close=result['close'], volume=result['volume']).on_balance_volume()
        
        # Calculate daily returns
        result['daily_return'] = DailyReturnIndicator(close=result['close']).daily_return()
        
        # Volatility
        result['volatility_14'] = result['close'].pct_change().rolling(window=14).std()
        
        logger.info("Successfully added basic indicators")
        
        return result
    
    except Exception as e:
        logger.error(f"Error adding basic indicators: {str(e)}")
        return df  # Return original DataFrame if error


@log_execution
def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced technical indicators to DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # --- Input Data Validation and Type Conversion Start ---
    logger.debug(f"Entering add_advanced_indicators. Validating input df. Shape: {result.shape}")
    try:
        # 1. Ensure index is DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning(f"Input index is {type(result.index)}, attempting to convert to DatetimeIndex.")
            result.index = pd.to_datetime(result.index, errors='coerce')
            # Drop rows where index conversion failed
            original_len = len(result)
            result = result.dropna(axis=0, subset=[result.index.name] if result.index.name else None) 
            if len(result) < original_len:
                logger.warning(f"Dropped {original_len - len(result)} rows due to invalid index after conversion.")
        
        if result.empty or not isinstance(result.index, pd.DatetimeIndex):
             logger.error("Index conversion failed or resulted in empty DataFrame. Cannot proceed.")
             return df # Return original df

        # 2. Ensure required columns exist and are numeric
        required_cols = ['high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in result.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}. Cannot proceed.")
            return df # Return original df
            
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(f"Column '{col}' is {result[col].dtype}, attempting to convert to numeric.")
                original_type = result[col].dtype
                result[col] = pd.to_numeric(result[col], errors='coerce')
                # Check how many NaNs were introduced
                nan_count = result[col].isnull().sum()
                if nan_count > 0:
                    logger.warning(f"Conversion of '{col}' (from {original_type}) introduced {nan_count} NaNs.")
        
        # Optional: Handle NaNs introduced during conversion if necessary, though later steps might handle them.
        # For now, we assume subsequent indicator calculations or NaN handling steps will manage them.
        
        logger.debug(f"Input df validated. Index type: {type(result.index)}, Shape after validation: {result.shape}")
        logger.debug(f"  Column dtypes: high={result['high'].dtype}, low={result['low'].dtype}, close={result['close'].dtype}, volume={result['volume'].dtype}")
        
    except Exception as validation_e:
        logger.error(f"Error during input data validation/conversion: {validation_e}")
        logger.error(traceback.format_exc())
        return df # Return original df on validation error
    # --- Input Data Validation and Type Conversion End ---
    
    try:
        logger.debug("Calculating CCI...")
        result['cci'] = CCIIndicator(high=result['high'], low=result['low'], close=result['close'], window=20).cci()
        logger.debug("CCI calculated.")
        
        logger.debug("Calculating TSI...")
        result['tsi'] = TSIIndicator(close=result['close'], window_slow=25, window_fast=13).tsi()
        logger.debug("TSI calculated.")
        
        logger.debug("Calculating MFI...")
        result['mfi'] = MFIIndicator(high=result['high'], low=result['low'], close=result['close'], 
                                     volume=result['volume'], window=14).money_flow_index()
        logger.debug("MFI calculated.")
        
        logger.debug("Calculating ADI...")
        result['adi'] = AccDistIndexIndicator(high=result['high'], low=result['low'], close=result['close'], 
                                               volume=result['volume']).acc_dist_index()
        logger.debug("ADI calculated.")
        
        logger.debug("Calculating ROC (10)...") # ROC(10) 추가
        result['roc10'] = calculate_rate_of_change(result['close'], 10)
        logger.debug("ROC (10) calculated.")
        
        logger.debug("Calculating Ichimoku Cloud...")
        # Check dtypes before calling calculate_ichimoku
        logger.debug(f"  Ichimoku input dtypes: high={result['high'].dtype}, low={result['low'].dtype}, close={result['close'].dtype}")
        ichimoku = calculate_ichimoku(result)
        logger.debug("Ichimoku calculated. Merging...")
        result = pd.concat([result, ichimoku], axis=1)
        logger.debug("Ichimoku merged.")
        # Log dtypes after Ichimoku merge
        logger.debug(f"  Dtypes after Ichimoku:\n{result[['high', 'low', 'close']].dtypes}")

        logger.debug("Calculating Keltner Channel...")
        # Check dtypes before calling calculate_keltner_channel
        logger.debug(f"  Keltner input dtypes: high={result['high'].dtype}, low={result['low'].dtype}, close={result['close'].dtype}")
        keltner = calculate_keltner_channel(result)
        logger.debug("Keltner calculated. Merging...")
        result = pd.concat([result, keltner], axis=1)
        logger.debug("Keltner merged.")
        # Log dtypes after Keltner merge
        logger.debug(f"  Dtypes after Keltner:\n{result[['high', 'low', 'close']].dtypes}")

        logger.debug("Calculating Hull MA...")
        # Add try-except for HMA
        try:
            result['hma9'] = calculate_hull_ma(result['close'], 9)
            logger.debug("Hull MA calculated.")
            # Log dtypes after Hull MA calculation
            logger.debug(f"  Dtypes after Hull MA:\n{result[['high', 'low', 'close']].dtypes}")
        except Exception as e_hma:
            logger.warning(f"Error calculating Hull MA: {e_hma}")
            result['hma9'] = np.nan

        logger.debug("Calculating Awesome Oscillator...")
        # Check dtypes before calling calculate_awesome_oscillator
        logger.debug(f"  AO input dtypes: high={result['high'].dtype}, low={result['low'].dtype}")
        # Corrected try-except block for AO
        try:
            result['ao'] = calculate_awesome_oscillator(result)
            logger.debug("Awesome Oscillator calculated.")
            # Log dtypes after AO calculation
            logger.debug(f"  Dtypes after AO:\n{result[['high', 'low', 'close']].dtypes}")
        except Exception as e_ao:
            logger.warning(f"Error calculating Awesome Oscillator: {e_ao}")
            result['ao'] = np.nan # Set to NaN on error
        
        logger.debug("Calculating Williams %R...")
        # Check dtypes before calling calculate_williams_r
        logger.debug(f"  Williams %R input dtypes: high={result['high'].dtype}, low={result['low'].dtype}, close={result['close'].dtype}")
        # Add try-except for Williams %R as well
        try:
            result['williams_r'] = calculate_williams_r(result)
            logger.debug("Williams %R calculated.")
        except Exception as e_wr:
            logger.warning(f"Error calculating Williams %R: {e_wr}")
            result['williams_r'] = np.nan # Set to NaN on error
        
        logger.debug("Calculating VWAP...")
        # Check dtypes before calling calculate_vwap
        logger.debug(f"  VWAP input dtypes: high={result['high'].dtype}, low={result['low'].dtype}, close={result['close'].dtype}, volume={result['volume'].dtype}")
        result['vwap'] = calculate_vwap(result)
        logger.debug("VWAP calculated.")
        
        logger.debug("Calculating ADX (14)...") # ADX(14) 추가
        try: # ADX 계산 오류 처리 추가
            adx_indicator = ADXIndicator(high=result['high'], low=result['low'], close=result['close'], window=14, fillna=True)
            result['adx14'] = adx_indicator.adx()
            # Optionally add +DI and -DI if needed
            # result['adx_pos'] = adx_indicator.adx_pos()
            # result['adx_neg'] = adx_indicator.adx_neg()
            logger.debug("ADX (14) calculated.")
        except Exception as e_adx:
            logger.warning(f"Error calculating ADX (14): {e_adx}")
            result['adx14'] = np.nan

        logger.debug("Calculating Bollinger Bands %B (20)...") # %B(20) 추가
        result['bb_percent_b20'] = calculate_percent_b(high=result['high'], low=result['low'], close=result['close'], window=20, window_dev=2)
        logger.debug("Bollinger Bands %B (20) calculated.")

        logger.debug("Calculating Chaikin Money Flow (20)...") # CMF(20) 추가
        result['cmf20'] = calculate_cmf(high=result['high'], low=result['low'], close=result['close'], volume=result['volume'], window=20)
        logger.debug("Chaikin Money Flow (20) calculated.")

        logger.debug("Calculating EMA Distance Norm (12, 26)...") # EMA 거리 정규화 추가
        result['ema12_ema26_dist_norm'] = calculate_ema_dist_norm(close=result['close'], window_fast=12, window_slow=26)
        logger.debug("EMA Distance Norm (12, 26) calculated.")

        # ... (기존 HMA, AO, Williams %R 등 계산) ...

        # --- Add pct_change calculations ---
        logger.debug("Calculating Price and Volume Change (1 period)...")
        # Ensure close and volume are numeric before pct_change
        close_numeric = pd.to_numeric(result['close'], errors='coerce')
        volume_numeric = pd.to_numeric(result['volume'], errors='coerce')
        result['price_change_1'] = close_numeric.pct_change(periods=1)
        result['volume_change_1'] = volume_numeric.pct_change(periods=1)
        # Fill initial NaNs created by pct_change and potential division by zero
        result['price_change_1'] = result['price_change_1'].replace([np.inf, -np.inf], np.nan).fillna(0)
        result['volume_change_1'] = result['volume_change_1'].replace([np.inf, -np.inf], np.nan).fillna(0)
        logger.debug("Price and Volume Change (1 period) calculated.")
        # --- End pct_change calculations ---

        # Final NaN check and fill for all added indicators
        # ... (기존 NaN 처리 로직) ...
        # Consolidate NaN filling here after all indicators are added
        indicator_cols = result.columns.difference(df.columns) # Get newly added columns
        result[indicator_cols] = result[indicator_cols].fillna(method='ffill').fillna(method='bfill')
        # Fill any remaining NaNs (e.g., at the very beginning) with 0 or another appropriate value
        result[indicator_cols] = result[indicator_cols].fillna(0)

        logger.info("Successfully added advanced indicators")
        
        return result
    
    except Exception as e:
        # --- Enhanced Traceback Logging ---
        error_msg = f"Error adding advanced indicators: {str(e)}"
        logger.error(error_msg)
        try:
            tb_str = traceback.format_exc()
            logger.error("--- Traceback Start ---")
            logger.error(tb_str) # Log the full traceback
            logger.error("--- Traceback End ---")
        except Exception as tb_e:
            logger.error(f"Could not format or log traceback: {tb_e}")
        # --- End Enhanced Traceback Logging ---
        return df # Return original DataFrame if error


@log_execution
def calculate_rate_of_change(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Rate of Change indicator
    
    Args:
        series (pd.Series): Price series
        period (int): Period for calculation
        
    Returns:
        pd.Series: Rate of Change values
    """
    try:
        series_numeric = pd.to_numeric(series, errors='coerce')
        if series_numeric.isnull().all():
            logger.warning(f"ROC 계산 입력 데이터(period={period})에 문제가 있습니다.")
            return pd.Series(np.nan, index=series.index)

        shifted_series = series_numeric.shift(period)
        # Avoid division by zero
        shifted_series_safe = shifted_series.replace(0, np.nan)
        
        roc = ((series_numeric - shifted_series) / shifted_series_safe) * 100
        roc.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaNs (from division by zero or initial calculation)
        return roc.fillna(method='ffill').fillna(method='bfill').fillna(0) # Fill remaining NaNs with 0
    except Exception as e:
        logger.error(f"Error calculating ROC for period {period}: {e}")
        logger.debug(traceback.format_exc())
        return pd.Series(np.nan, index=series.index)


@log_execution
def calculate_hull_ma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Hull Moving Average
    
    Args:
        series (pd.Series): Price series
        period (int): Period for calculation
        
    Returns:
        pd.Series: Hull Moving Average values
    """
    logger.debug(f"Hull MA 계산 시작. 입력 Series 길이: {len(series)}, 기간: {period}")
    if len(series) < period:
        logger.warning(f"Hull MA 계산 위한 데이터 부족 ({len(series)} < {period}). NaN 반환.")
        return pd.Series(np.nan, index=series.index)

    try:
        # Ensure input is numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.isnull().all():
            logger.warning("Hull MA 입력 Series 전체가 NaN 또는 변환 불가. NaN 반환.")
            return pd.Series(np.nan, index=series.index)
            
        logger.debug(f"Hull MA 입력 데이터 타입: {numeric_series.dtype}")
        logger.debug(f"Hull MA 입력 데이터 head:\\n{numeric_series.head()}")

        half_period = period // 2
        sqrt_period = int(np.sqrt(period))

        # Ensure periods are valid
        if half_period < 1 or sqrt_period < 1:
             logger.warning(f"Hull MA 계산 기간 부적절 (period={period}). NaN 반환.")
             return pd.Series(np.nan, index=series.index)

        # Use min_periods=1 in rolling to handle initial NaNs better
        wma1 = numeric_series.rolling(window=half_period, min_periods=1).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
        wma2 = numeric_series.rolling(window=period, min_periods=1).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
        
        hull = 2 * wma1 - wma2
        hma = hull.rolling(window=sqrt_period, min_periods=1).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)

        # Replace inf/-inf with NaN
        hma.replace([np.inf, -np.inf], np.nan, inplace=True)

        logger.debug("Hull MA 계산 완료")
        return hma
        
    except Exception as e:
        logger.error(f"Hull MA 계산 중 오류 발생: {e}")
        # Log detailed info in except block
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Input data type causing error: {series.dtype}")
        logger.error(f"Input data head causing error:\\n{series.head()}")
        logger.error(traceback.format_exc()) # Log the full traceback
        # Return NaN series on failure
        return pd.Series(np.nan, index=series.index)


@log_execution
def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud indicator
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with Ichimoku components or NaN DataFrame on error
    """
    ichimoku_cols = [
        'ichimoku_tenkan_sen',
        'ichimoku_kijun_sen',
        'ichimoku_senkou_span_a',
        'ichimoku_senkou_span_b',
        'ichimoku_chikou_span'
    ]
    # Create a DataFrame with NaN values initially to return on error
    nan_df = pd.DataFrame(np.nan, index=df.index, columns=ichimoku_cols)
    
    logger.debug(f"Ichimoku 계산 시작. 입력 DF shape: {df.shape}")
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Ichimoku 계산에 필요한 컬럼({required_cols}) 없음. NaN DF 반환.")
        return nan_df
        
    # Check minimum required length (52 for Senkou Span B)
    if len(df) < 52:
        logger.warning(f"Ichimoku 계산 위한 데이터 부족 ({len(df)} < 52). NaN DF 반환.")
        return nan_df

    try:
        # Ensure inputs are numeric, coercing errors to NaN
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')

        if high.isnull().all() or low.isnull().all() or close.isnull().all():
             logger.warning("Ichimoku 입력 high/low/close 전체가 NaN 또는 변환 불가. NaN DF 반환.")
             return nan_df

        logger.debug(f"Ichimoku 입력 데이터 타입: High={high.dtype}, Low={low.dtype}, Close={close.dtype}")
        logger.debug(f"Ichimoku 입력 데이터 head:\\n{df[required_cols].head()}")

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2 (min_periods=1)
        tenkan_sen_high = high.rolling(window=9, min_periods=1).max()
        tenkan_sen_low = low.rolling(window=9, min_periods=1).min()
        tenkan_sen = (tenkan_sen_high + tenkan_sen_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2 (min_periods=1)
        kijun_sen_high = high.rolling(window=26, min_periods=1).max()
        kijun_sen_low = low.rolling(window=26, min_periods=1).min()
        kijun_sen = (kijun_sen_high + kijun_sen_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 (shifted 26 periods ahead)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 (shifted 26 periods ahead)
        senkou_span_b_high = high.rolling(window=52, min_periods=1).max()
        senkou_span_b_low = low.rolling(window=52, min_periods=1).min()
        senkou_span_b = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted 26 periods behind
        chikou_span = close.shift(-26)
        
        # Replace inf/-inf just in case (though unlikely for Ichimoku)
        result_df = pd.DataFrame({
            'ichimoku_tenkan_sen': tenkan_sen,
            'ichimoku_kijun_sen': kijun_sen,
            'ichimoku_senkou_span_a': senkou_span_a,
            'ichimoku_senkou_span_b': senkou_span_b,
            'ichimoku_chikou_span': chikou_span
        }, index=df.index)
        result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        logger.debug("Ichimoku 계산 완료")
        return result_df
        
    except Exception as e:
        logger.error(f"Ichimoku 계산 중 오류 발생: {e}")
        # Log detailed info in except block
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Input data types causing error: High={df['high'].dtype}, Low={df['low'].dtype}, Close={df['close'].dtype}")
        logger.error(f"Input data head causing error:\\n{df[required_cols].head()}")
        logger.error(traceback.format_exc()) # Log the full traceback
        # Return NaN DataFrame on failure
        return nan_df


@log_execution
def calculate_keltner_channel(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
    """
    Calculate Keltner Channel
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        ema_period (int, optional): Period for EMA calculation. Defaults to 20.
        atr_period (int, optional): Period for ATR calculation. Defaults to 10.
        multiplier (float, optional): Multiplier for ATR. Defaults to 2.0.
        
    Returns:
        pd.DataFrame: DataFrame with Keltner Channel components or NaN DataFrame on error
    """
    # 결과 컬럼 이름 정의
    keltner_cols = ['keltner_mband', 'keltner_hband', 'keltner_lband']
    nan_df = pd.DataFrame(np.nan, index=df.index, columns=keltner_cols)
    
    logger.debug(f"Keltner 채널 계산 시작. 입력 DF shape: {df.shape}, EMA={ema_period}, ATR={atr_period}, Mult={multiplier}")
    
    # 필요한 컬럼 체크
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Keltner 채널 계산에 필요한 컬럼({required_cols}) 없음. NaN DF 반환.")
        return nan_df
        
    # 최소 필요 길이 확인
    min_len = max(ema_period, atr_period)
    if len(df) < min_len:
        logger.warning(f"Keltner 채널 계산 위한 데이터 부족 ({len(df)} < {min_len}). NaN DF 반환.")
        return nan_df

    # 순서대로 시도할 두 가지 방법
    methods = [
        "direct_ta_library",   # ta 라이브러리의 KeltnerChannel 클래스 직접 사용
        "manual_calculation"   # 직접 EMA와 ATR 계산
    ]

    for method in methods:
        try:
            logger.debug(f"Keltner 채널 계산 시도: {method}")
            
            # 입력 데이터 변환
            high = pd.to_numeric(df['high'], errors='coerce')
            low = pd.to_numeric(df['low'], errors='coerce')
            close = pd.to_numeric(df['close'], errors='coerce')

            # 데이터 유효성 검사
            if high.isnull().all() or low.isnull().all() or close.isnull().all():
                logger.warning("Keltner 입력 high/low/close 전체가 NaN 또는 변환 불가. 다음 방법 시도.")
                continue
                
            # NaN 값 비율 확인
            nan_ratio = (high.isnull().sum() + low.isnull().sum() + close.isnull().sum()) / (len(df) * 3)
            if nan_ratio > 0.3:  # 30% 이상이 NaN이면 경고 로그
                logger.warning(f"Keltner 입력 데이터에 높은 NaN 비율 ({nan_ratio:.2%})이 포함됨.")

            # 결측치 사전 처리 (중요: 전방 채우기 후 후방 채우기)
            high = high.fillna(method='ffill').fillna(method='bfill')
            low = low.fillna(method='ffill').fillna(method='bfill')
            close = close.fillna(method='ffill').fillna(method='bfill')

            if method == "direct_ta_library":
                # ta 라이브러리의 KeltnerChannel 클래스 직접 사용
                keltner = KeltnerChannel(
                    high=high, 
                    low=low, 
                    close=close, 
                    window=ema_period, 
                    window_atr=atr_period, 
                    fillna=True,  # 결측치 자동 처리
                    original_version=False,  # 표준 버전 사용
                    multiplier=multiplier  # multiplier 파라미터 추가
                )
                
                # 각 밴드 계산 - API 변경 문제 처리
                try:
                    # 새로운 API에서는 multiplier를 생성자에 전달
                    mband = keltner.keltner_channel_mband()
                    hband = keltner.keltner_channel_hband()
                    lband = keltner.keltner_channel_lband()
                except TypeError as te:
                    # API 호환성 문제 - 기존 방식으로 시도
                    logger.warning(f"Keltner 채널 API 호환성 문제: {str(te)}. 다른 방법 시도.")
                    # 수동 계산 방법으로 대체
                    method = "manual_calculation"
                    continue
                
                # 값 유효성 검사: 모두 NaN이면 실패로 간주
                if mband.isna().all() or hband.isna().all() or lband.isna().all():
                    logger.warning("KeltnerChannel 클래스에서 반환된 밴드가 모두 NaN입니다. 다음 방법 시도.")
                    continue
                    
                # 결과 구성
                result_df = pd.DataFrame({
                    'keltner_mband': mband,
                    'keltner_hband': hband,
                    'keltner_lband': lband
                }, index=df.index)
                
            else:  # "manual_calculation"
                # EMA 계산 
                ema = EMAIndicator(close=close, window=ema_period, fillna=True)
                middle = ema.ema_indicator()
                
                # ATR 계산
                atr_ind = AverageTrueRange(high=high, low=low, close=close, window=atr_period, fillna=True)
                atr = atr_ind.average_true_range()
                
                # 무한대 또는 NaN 값 확인 및 처리
                if atr.isna().all() or np.isinf(atr).any():
                    logger.warning("ATR 계산 결과에 문제가 있습니다. 중간 값으로 대체.")
                    # 문제 해결: ATR을 단순 TR의 이동평균으로 계산
                    tr = pd.DataFrame()
                    tr['h-l'] = high - low
                    tr['h-pc'] = abs(high - close.shift(1))
                    tr['l-pc'] = abs(low - close.shift(1))
                    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
                    atr = tr['tr'].rolling(window=atr_period, min_periods=1).mean()
                
                # 밴드 계산
                upper = middle + (multiplier * atr)
                lower = middle - (multiplier * atr)
                
                # 결과 구성
                result_df = pd.DataFrame({
                    'keltner_mband': middle,
                    'keltner_hband': upper,
                    'keltner_lband': lower
                }, index=df.index)
        
            # 무한대 값을 NaN으로 대체
            result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
            # NaN 상태 확인 및 보고
            nan_stats = result_df.isna().sum()
            if nan_stats.sum() > 0:
                logger.warning(f"Keltner 채널 계산 후 NaN 상태: {nan_stats.to_dict()}")
                
                # NaN 비율이 너무 높으면 경고
                nan_ratio = result_df.isna().mean().mean()
                if nan_ratio > 0.3:  # 30% 이상이 NaN이면 경고
                    logger.warning(f"계산된 Keltner 채널에 높은 NaN 비율({nan_ratio:.2%})이 있습니다. 결과의 신뢰성이 떨어질 수 있습니다.")
                
                # NaN 처리: 전방 채우기 후 후방 채우기, 그래도 남아있으면 0으로 대체
                result_df = result_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
                logger.info("Keltner 채널의 NaN 값을 채움 처리했습니다.")
            
            logger.debug(f"Keltner 채널 계산 성공 (방법: {method})")
            return result_df
        
        except Exception as e:
            logger.error(f"Keltner 채널 계산 중 오류 발생 (방법: {method}): {e}")
            logger.debug(f"Exception type: {type(e)}")
            logger.debug(traceback.format_exc())
            logger.info(f"다른 방법으로 Keltner 채널 계산을 시도합니다.")
            continue
    
    # 모든 방법이 실패한 경우
    logger.error("모든 Keltner 채널 계산 방법이 실패했습니다. NaN DataFrame을 반환합니다.")
    return nan_df


@log_execution
def calculate_awesome_oscillator(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Awesome Oscillator
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.Series: Awesome Oscillator values
    """
    logger.debug(f"Awesome Oscillator 계산 시작. 입력 DF shape: {df.shape}")
    required_cols = ['high', 'low']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Awesome Oscillator 계산에 필요한 컬럼({required_cols}) 없음. NaN 반환.")
        return pd.Series(np.nan, index=df.index)
        
    if len(df) < 34: # Needs at least 34 periods for sma34
        logger.warning(f"Awesome Oscillator 계산 위한 데이터 부족 ({len(df)} < 34). NaN 반환.")
        return pd.Series(np.nan, index=df.index)

    try:
        # Ensure inputs are numeric, coercing errors to NaN
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')

        if high.isnull().all() or low.isnull().all():
             logger.warning("Awesome Oscillator 입력 high/low 전체가 NaN 또는 변환 불가. NaN 반환.")
             return pd.Series(np.nan, index=df.index)

        logger.debug(f"Awesome Oscillator 입력 데이터 타입: High={high.dtype}, Low={low.dtype}")
        logger.debug(f"Awesome Oscillator 입력 데이터 head:\\n{df[['high', 'low']].head()}")

        # Median price
        median_price = (high + low) / 2
        
        # Calculate 5-period simple moving average (use min_periods=1)
        sma5 = median_price.rolling(window=5, min_periods=1).mean()
        
        # Calculate 34-period simple moving average (use min_periods=1)
        sma34 = median_price.rolling(window=34, min_periods=1).mean()
        
        # Awesome Oscillator
        ao = sma5 - sma34

        # Replace inf/-inf with NaN
        ao.replace([np.inf, -np.inf], np.nan, inplace=True)

        logger.debug("Awesome Oscillator 계산 완료")
        return ao
        
    except Exception as e:
        logger.error(f"Awesome Oscillator 계산 중 오류 발생: {e}")
        # Log detailed info in except block
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Input data types causing error: High={df['high'].dtype}, Low={df['low'].dtype}")
        logger.error(f"Input data head causing error:\\n{df[['high', 'low']].head()}")
        logger.error(traceback.format_exc()) # Log the full traceback
        # Return NaN series on failure
        return pd.Series(np.nan, index=df.index)


@log_execution
def calculate_williams_r(df, period=14):
    """Williams %R 계산"""
    logger.debug(f"데이터프레임 길이: {len(df)}, 필요 기간: {period}")
    if len(df) < period:
        logger.warning(f"Williams %R 계산 위한 데이터 부족 ({len(df)} < {period}). NaN 반환.")
        return pd.Series(np.nan, index=df.index)

    try:
        # 입력 데이터 타입 확인 (추가)
        logger.debug(f"Williams %R 입력 데이터 타입:\nHigh: {df['high'].dtype}, Low: {df['low'].dtype}, Close: {df['close'].dtype}")
        logger.debug(f"Williams %R 입력 데이터 head:\n{df[['high', 'low', 'close']].head()}")

        # Ensure inputs are numeric, coercing errors to NaN
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')

        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()

        williams_r = (highest_high - close) / (highest_high - lowest_low) * -100

        # Replace inf/-inf with NaN resulting from division by zero
        williams_r.replace([np.inf, -np.inf], np.nan, inplace=True)

        logger.debug("Williams %R 계산 완료")
        return williams_r
    except Exception as e:
        logger.error(f"Williams %R 계산 중 오류 발생: {e}")
        # Log detailed info in except block
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Input data types causing error:\nHigh: {df['high'].dtype}, Low: {df['low'].dtype}, Close: {df['close'].dtype}")
        logger.error(f"Input data head causing error:\n{df[['high', 'low', 'close']].head()}")
        logger.error(traceback.format_exc()) # Log the full traceback
        # Return NaN series on failure
        return pd.Series(np.nan, index=df.index)


@log_execution
def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.Series: VWAP values or NaN Series on error
    """
    nan_series = pd.Series(np.nan, index=df.index)
    
    logger.debug(f"VWAP 계산 시작. 입력 DF shape: {df.shape}")
    required_cols = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"VWAP 계산에 필요한 컬럼({required_cols}) 없음. NaN Series 반환.")
        return nan_series

    try:
        # Ensure inputs are numeric, coercing errors to NaN
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        volume = pd.to_numeric(df['volume'], errors='coerce')

        # Check if coercion resulted in all NaNs for essential columns
        if high.isnull().all() or low.isnull().all() or close.isnull().all() or volume.isnull().all():
             logger.warning("VWAP 입력 high/low/close/volume 중 하나 이상이 전체 NaN 또는 변환 불가. NaN Series 반환.")
             return nan_series
             
        logger.debug(f"VWAP 입력 데이터 타입: High={high.dtype}, Low={low.dtype}, Close={close.dtype}, Volume={volume.dtype}")
        logger.debug(f"VWAP 입력 데이터 head:\\n{df[required_cols].head()}")

        # Typical price
        typical_price = (high + low + close) / 3
        
        # Volume * typical price (handle potential NaNs from conversion)
        vol_tp = typical_price * volume.fillna(0) # Fill volume NaN with 0 for this calculation
        
        # Cumulative values
        cumulative_vol_tp = vol_tp.cumsum()
        cumulative_volume = volume.fillna(0).cumsum()
        
        # VWAP calculation - avoid division by zero
        vwap = cumulative_vol_tp / cumulative_volume.replace(0, np.nan) # Replace 0 cumulative volume with NaN to avoid division by zero
        
        # Replace inf/-inf with NaN
        vwap.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        logger.debug("VWAP 계산 완료")
        return vwap
        
    except Exception as e:
        logger.error(f"VWAP 계산 중 오류 발생: {e}")
        # Log detailed info in except block
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Input data types causing error: High={df['high'].dtype}, Low={df['low'].dtype}, Close={df['close'].dtype}, Volume={df['volume'].dtype}")
        logger.error(f"Input data head causing error:\\n{df[required_cols].head()}")
        logger.error(traceback.format_exc()) # Log the full traceback
        # Return NaN series on failure
        return nan_series


@log_execution
def add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add custom technical indicators
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    if df is None or df.empty:
        logger.warning("add_custom_indicators: 입력 DataFrame이 비어 있습니다.")
        return df
        
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    try:
        # 필요한 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"add_custom_indicators: 필수 컬럼 '{col}'이(가) 없습니다.")
                return df
                
        # 데이터 길이 확인
        if len(result) < 200:  # EMA200이 필요할 수 있으므로 최소 길이 확인
            logger.warning(f"add_custom_indicators: 데이터 길이가 너무 짧습니다 ({len(result)} < 200)")
            # 계속 진행하지만 경고 로그 남김
            
        # 컬럼 데이터를 숫자형으로 변환
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.info(f"add_custom_indicators: '{col}' 컬럼을 숫자형으로 변환합니다.")
                result[col] = pd.to_numeric(result[col], errors='coerce')
        
        # Volume-based indicators
        result['volume_sma7'] = SMAIndicator(close=result['volume'], window=7).sma_indicator()
        result['volume_ema20'] = EMAIndicator(close=result['volume'], window=20).ema_indicator()
        
        # Volume / Moving Average ratio (Volume Surge indicator)
        result['volume_surge'] = result['volume'] / result['volume_sma7']
        
        # Price momentum indicators
        result['momentum_3d'] = result['close'] / result['close'].shift(3) - 1
        result['momentum_7d'] = result['close'] / result['close'].shift(7) - 1
        result['momentum_14d'] = result['close'] / result['close'].shift(14) - 1
        
        # Volatility ratio (comparing recent volatility to historical)
        result['volatility_ratio'] = result['close'].pct_change().rolling(5).std() / result['close'].pct_change().rolling(20).std()
        
        # Distance from moving averages (normalized)
        # ema200 컬럼이 존재하는지 확인하고 없으면 계산
        if 'ema200' not in result.columns:
            logger.info("add_custom_indicators: 'ema200' 컬럼이 없어 직접 계산합니다.")
            try:
                result['ema200'] = EMAIndicator(close=result['close'], window=200).ema_indicator()
            except Exception as ema_err:
                logger.error(f"add_custom_indicators: 'ema200' 계산 중 오류 발생: {str(ema_err)}")
                # ema200 컬럼을 사용하는 기능은 건너뛰기
                logger.warning("add_custom_indicators: 'distance_from_ema200' 계산을 건너뜁니다.")
                result['distance_from_ema200'] = np.nan
        
        # ema200이 계산되었을 때만 distance_from_ema200 계산
        if 'ema200' in result.columns and not result['ema200'].isna().all():
            # ema200에 NaN 값이 있는지 확인
            nan_ratio = result['ema200'].isna().mean()
            if nan_ratio > 0:
                logger.warning(f"add_custom_indicators: 'ema200'에 {nan_ratio:.2%}의 NaN 값이 있습니다.")
                # 보간 시도
                if nan_ratio < 0.5:  # NaN 비율이 50% 미만인 경우에만 보간
                    result['ema200'] = result['ema200'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                    logger.info("add_custom_indicators: 'ema200'의 NaN 값을 보간했습니다.")
                else:
                    logger.warning("add_custom_indicators: NaN 비율이 높아 'distance_from_ema200' 계산을 건너뜁니다.")
                    result['distance_from_ema200'] = np.nan # NaN 비율 높으면 NaN 처리

            # NaN 값이 없거나 처리된 경우에만 distance_from_ema200 계산 (들여쓰기 수정)
            if not result['ema200'].isna().all(): # 이 if 블록 전체 들여쓰기 수정
                result['distance_from_ema200'] = (result['close'] - result['ema200']) / result['ema200'] * 100
            else:
                result['distance_from_ema200'] = np.nan # 보간 후에도 NaN이면 NaN 처리
        else:
            logger.warning("add_custom_indicators: 유효한 'ema200' 값이 없어 'distance_from_ema200' 계산을 건너뜁니다.")
            result['distance_from_ema200'] = np.nan
        
        # RSI divergence (compare price direction with RSI direction)
        if 'rsi14' in result.columns:
            price_diff = result['close'].diff()
            rsi_diff = result['rsi14'].diff()
            result['rsi_divergence'] = np.sign(price_diff) * np.sign(rsi_diff)
        else:
            logger.warning("add_custom_indicators: 'rsi14' 컬럼이 없어 'rsi_divergence' 계산을 건너뜁니다.")
            result['rsi_divergence'] = np.nan
        
        # Efficiency ratio (how smoothly price moves)
        price_change = abs(result['close'] - result['close'].shift(14))
        price_path = result['high'].rolling(14).max() - result['low'].rolling(14).min()
        # 0으로 나누는 것을 방지
        zero_path = price_path == 0
        if zero_path.any():
            logger.warning(f"add_custom_indicators: 'price_path'에서 {zero_path.sum()}개의 0 값이 발견되었습니다.")
            price_path = price_path.replace(0, np.nan)
        result['efficiency_ratio'] = price_change / price_path
        
        # Volume profile (relative volume by price level)
        result['volume_price_ratio'] = result['volume'] / result['close']
        
        # 인피니티 값과 NaN 값 처리
        numeric_cols = result.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col in result.columns:
                inf_mask = np.isinf(result[col])
                if inf_mask.any():
                    logger.warning(f"add_custom_indicators: '{col}' 컬럼에서 무한대 값이 발견되어 NaN으로 대체합니다.")
                    result.loc[inf_mask, col] = np.nan
                    
                # NaN 갯수 확인
                nan_count = result[col].isna().sum()
                if nan_count > 0:
                    logger.info(f"add_custom_indicators: '{col}' 컬럼에 {nan_count}개의 NaN 값이 있습니다.")
        
        logger.info("커스텀 지표들을 성공적으로 추가했습니다.")
        
        return result
    
    except Exception as e:
        logger.error(f"커스텀 지표 추가 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # Return original DataFrame if error


@log_execution
def add_pattern_recognition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candlestick pattern recognition indicators
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added pattern indicators
    """
    if df is None or df.empty:
        logger.warning("add_pattern_recognition: 입력 DataFrame이 비어 있습니다.")
        return df
        
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    try:
        # 필요한 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"add_pattern_recognition: 필수 컬럼 '{col}'이(가) 없습니다.")
                return df
        
        # 컬럼 데이터를 숫자형으로 변환
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.info(f"add_pattern_recognition: '{col}' 컬럼을 숫자형으로 변환합니다.")
                result[col] = pd.to_numeric(result[col], errors='coerce')
        
        # 충분한 데이터가 있는지 확인
        if len(result) < 4:  # 캔들 패턴 중 최대 4개의 캔들이 필요함
            logger.warning(f"add_pattern_recognition: 데이터 길이가 너무 짧습니다 ({len(result)} < 4)")
            return df
        
        # Doji pattern
        result['doji'] = identify_doji(result)
        
        # Hammer and Hanging Man
        result['hammer'] = identify_hammer(result)
        
        # Engulfing pattern
        result['bullish_engulfing'] = identify_bullish_engulfing(result)
        result['bearish_engulfing'] = identify_bearish_engulfing(result)
        
        # Morning star and Evening star
        result['morning_star'] = identify_morning_star(result)
        result['evening_star'] = identify_evening_star(result)
        
        # Three Line Strike
        result['three_line_strike'] = identify_three_line_strike(result)
        
        logger.info("캔들스틱 패턴 인식 지표들을 성공적으로 추가했습니다.")
        
        return result
    
    except Exception as e:
        logger.error(f"캔들스틱 패턴 인식 지표 추가 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # Return original DataFrame if error


@log_execution
def identify_doji(df: pd.DataFrame, doji_size: float = 0.05) -> pd.Series:
    """
    Identify Doji candlestick pattern
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        doji_size (float, optional): Maximum size of the body relative to the candle range. Defaults to 0.05.
        
    Returns:
        pd.Series: Boolean series indicating Doji patterns
    """
    try:
        # 필요한 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"identify_doji: 필수 컬럼 '{col}'이(가) 없습니다.")
                return pd.Series(False, index=df.index)
        
        # 입력 데이터를 숫자형으로 변환
        open_prices = pd.to_numeric(df['open'], errors='coerce')
        high_prices = pd.to_numeric(df['high'], errors='coerce')
        low_prices = pd.to_numeric(df['low'], errors='coerce')
        close_prices = pd.to_numeric(df['close'], errors='coerce')
        
        # Calculate body size
        body_size = abs(close_prices - open_prices)
        
        # Calculate candle range
        candle_range = high_prices - low_prices
        
        # Identify Doji: small body compared to range
        doji = (body_size / candle_range) < doji_size
        
        # Ensure candle has some range to avoid division by zero
        doji = doji & (candle_range > 0)
        
        # Replace NaN values with False
        doji = doji.fillna(False)
        
        return doji
    
    except Exception as e:
        logger.error(f"Error in identify_doji: {str(e)}")
        return pd.Series(False, index=df.index)


@log_execution
def identify_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Identify Hammer and Hanging Man candlestick patterns
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.Series: Series with values indicating pattern (1 for hammer, -1 for hanging man, 0 for neither)
    """
    try:
        # 필요한 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"identify_hammer: 필수 컬럼 '{col}'이(가) 없습니다.")
                return pd.Series(0, index=df.index)
        
        if len(df) < 3:
            logger.warning("identify_hammer: 데이터가 3개 이하입니다. 최소 3개의 데이터가 필요합니다.")
            return pd.Series(0, index=df.index)
            
        # 입력 데이터를 숫자형으로 변환
        open_prices = pd.to_numeric(df['open'], errors='coerce')
        high_prices = pd.to_numeric(df['high'], errors='coerce')
        low_prices = pd.to_numeric(df['low'], errors='coerce')
        close_prices = pd.to_numeric(df['close'], errors='coerce')
        
        # Calculate body size, upper shadow, lower shadow
        body_size = abs(close_prices - open_prices)
        
        # Determine the trend (looking back at 3 candles)
        uptrend = close_prices.rolling(3).mean().shift(1) > close_prices.rolling(7).mean().shift(1)
        downtrend = close_prices.rolling(3).mean().shift(1) < close_prices.rolling(7).mean().shift(1)
        
        # For hammer/hanging man, we want small upper shadow and long lower shadow
        upper_shadow = high_prices - pd.DataFrame({'open': open_prices, 'close': close_prices}).max(axis=1)
        lower_shadow = pd.DataFrame({'open': open_prices, 'close': close_prices}).min(axis=1) - low_prices
        
        # Conditions for hammer/hanging man
        small_upper_shadow = upper_shadow < 0.2 * body_size
        long_lower_shadow = lower_shadow > 2 * body_size
        
        # Hammer (in downtrend) and Hanging Man (in uptrend)
        hammer = (small_upper_shadow & long_lower_shadow & downtrend).astype(int)
        hanging_man = (small_upper_shadow & long_lower_shadow & uptrend).astype(int) * -1
        
        # NaN 값을 0으로 대체
        result = hammer + hanging_man
        result = result.fillna(0)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in identify_hammer: {str(e)}")
        return pd.Series(0, index=df.index)


@log_execution
def identify_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Identify Bullish Engulfing candlestick pattern
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.Series: Boolean series indicating Bullish Engulfing patterns
    """
    try:
        # 필요한 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"identify_bullish_engulfing: 필수 컬럼 '{col}'이(가) 없습니다.")
                return pd.Series(False, index=df.index)
                
        if len(df) < 2:
            logger.warning("identify_bullish_engulfing: 데이터가 2개 이하입니다. 최소 2개의 데이터가 필요합니다.")
            return pd.Series(False, index=df.index)
        
        # 입력 데이터를 숫자형으로 변환
        open_prices = pd.to_numeric(df['open'], errors='coerce')
        close_prices = pd.to_numeric(df['close'], errors='coerce')
        
        # Prior candle is bearish (close < open)
        prior_bearish = close_prices.shift(1) < open_prices.shift(1)
        
        # Current candle is bullish (close > open)
        current_bullish = close_prices > open_prices
        
        # Current candle's body engulfs prior candle's body
        engulfing = (open_prices < close_prices.shift(1)) & (close_prices > open_prices.shift(1))
        
        # Combine conditions
        bullish_engulfing = prior_bearish & current_bullish & engulfing
        
        # NaN 값을 False로 대체
        bullish_engulfing = bullish_engulfing.fillna(False)
        
        return bullish_engulfing
    
    except Exception as e:
        logger.error(f"Error in identify_bullish_engulfing: {str(e)}")
        return pd.Series(False, index=df.index)


@log_execution
def identify_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Identify Bearish Engulfing candlestick pattern
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.Series: Boolean series indicating Bearish Engulfing patterns
    """
    try:
        # 필요한 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"identify_bearish_engulfing: 필수 컬럼 '{col}'이(가) 없습니다.")
                return pd.Series(False, index=df.index)
                
        if len(df) < 2:
            logger.warning("identify_bearish_engulfing: 데이터가 2개 이하입니다. 최소 2개의 데이터가 필요합니다.")
            return pd.Series(False, index=df.index)
        
        # 입력 데이터를 숫자형으로 변환
        open_prices = pd.to_numeric(df['open'], errors='coerce')
        close_prices = pd.to_numeric(df['close'], errors='coerce')
        
        # Prior candle is bullish (close > open)
        prior_bullish = close_prices.shift(1) > open_prices.shift(1)
        
        # Current candle is bearish (close < open)
        current_bearish = close_prices < open_prices
        
        # Current candle's body engulfs prior candle's body
        engulfing = (open_prices > close_prices.shift(1)) & (close_prices < open_prices.shift(1))
        
        # Combine conditions
        bearish_engulfing = prior_bullish & current_bearish & engulfing
        
        # NaN 값을 False로 대체
        bearish_engulfing = bearish_engulfing.fillna(False)
        
        return bearish_engulfing
    
    except Exception as e:
        logger.error(f"Error in identify_bearish_engulfing: {str(e)}")
        return pd.Series(False, index=df.index)


@log_execution
def identify_morning_star(df: pd.DataFrame) -> pd.Series:
    """
    Identify Morning Star candlestick pattern
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.Series: Boolean series indicating Morning Star patterns
    """
    try:
        # 필요한 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"identify_morning_star: 필수 컬럼 '{col}'이(가) 없습니다.")
                return pd.Series(False, index=df.index)
                
        if len(df) < 3:
            logger.warning("identify_morning_star: 데이터가 3개 이하입니다. 최소 3개의 데이터가 필요합니다.")
            return pd.Series(False, index=df.index)
        
        # 입력 데이터를 숫자형으로 변환
        open_prices = pd.to_numeric(df['open'], errors='coerce')
        high_prices = pd.to_numeric(df['high'], errors='coerce')
        low_prices = pd.to_numeric(df['low'], errors='coerce')
        close_prices = pd.to_numeric(df['close'], errors='coerce')
        
        # First candle is bearish
        first_bearish = close_prices.shift(2) < open_prices.shift(2)
        
        # Second candle is small
        second_small = abs(close_prices.shift(1) - open_prices.shift(1)) < 0.3 * abs(close_prices.shift(2) - open_prices.shift(2))
        
        # Third candle is bullish and closes above midpoint of first candle
        third_bullish = close_prices > open_prices
        midpoint_first = (open_prices.shift(2) + close_prices.shift(2)) / 2
        third_above_midpoint = close_prices > midpoint_first
        
        # Gap down between first and second candles
        gap_down = high_prices.shift(1) < low_prices.shift(2)
        
        # Combine conditions
        morning_star = first_bearish & second_small & third_bullish & third_above_midpoint & gap_down
        
        # NaN 값을 False로 대체
        morning_star = morning_star.fillna(False)
        
        return morning_star
    
    except Exception as e:
        logger.error(f"Error in identify_morning_star: {str(e)}")
        return pd.Series(False, index=df.index)


@log_execution
def identify_evening_star(df: pd.DataFrame) -> pd.Series:
    """
    Identify Evening Star candlestick pattern
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.Series: Boolean series indicating Evening Star patterns
    """
    try:
        # 필요한 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"identify_evening_star: 필수 컬럼 '{col}'이(가) 없습니다.")
                return pd.Series(False, index=df.index)
                
        if len(df) < 3:
            logger.warning("identify_evening_star: 데이터가 3개 이하입니다. 최소 3개의 데이터가 필요합니다.")
            return pd.Series(False, index=df.index)
        
        # 입력 데이터를 숫자형으로 변환
        open_prices = pd.to_numeric(df['open'], errors='coerce')
        high_prices = pd.to_numeric(df['high'], errors='coerce')
        low_prices = pd.to_numeric(df['low'], errors='coerce')
        close_prices = pd.to_numeric(df['close'], errors='coerce')
        
        # First candle is bullish
        first_bullish = close_prices.shift(2) > open_prices.shift(2)
        
        # Second candle is small
        second_small = abs(close_prices.shift(1) - open_prices.shift(1)) < 0.3 * abs(close_prices.shift(2) - open_prices.shift(2))
        
        # Third candle is bearish and closes below midpoint of first candle
        third_bearish = close_prices < open_prices
        midpoint_first = (open_prices.shift(2) + close_prices.shift(2)) / 2
        third_below_midpoint = close_prices < midpoint_first
        
        # Gap up between first and second candles
        gap_up = low_prices.shift(1) > high_prices.shift(2)
        
        # Combine conditions
        evening_star = first_bullish & second_small & third_bearish & third_below_midpoint & gap_up
        
        # NaN 값을 False로 대체
        evening_star = evening_star.fillna(False)
        
        return evening_star
    
    except Exception as e:
        logger.error(f"Error in identify_evening_star: {str(e)}")
        return pd.Series(False, index=df.index)


@log_execution
def identify_three_line_strike(df: pd.DataFrame) -> pd.Series:
    """
    Identify Three Line Strike candlestick pattern
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.Series: Series with values indicating pattern (1 for bullish, -1 for bearish, 0 for neither)
    """
    try:
        # 필요한 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"identify_three_line_strike: 필수 컬럼 '{col}'이(가) 없습니다.")
                return pd.Series(0, index=df.index)
                
        if len(df) < 4:
            logger.warning("identify_three_line_strike: 데이터가 4개 이하입니다. 최소 4개의 데이터가 필요합니다.")
            return pd.Series(0, index=df.index)
        
        # 입력 데이터를 숫자형으로 변환
        open_prices = pd.to_numeric(df['open'], errors='coerce')
        close_prices = pd.to_numeric(df['close'], errors='coerce')
        
        # Bullish Three Line Strike
        three_bullish_candles = (
            (close_prices.shift(3) > open_prices.shift(3)) &  # First candle bullish
            (close_prices.shift(2) > open_prices.shift(2)) &  # Second candle bullish
            (close_prices.shift(1) > open_prices.shift(1)) &  # Third candle bullish
            (close_prices.shift(2) > close_prices.shift(3)) &  # Each close higher than previous
            (close_prices.shift(1) > close_prices.shift(2))
        )
        
        bearish_fourth = (
            (open_prices > close_prices.shift(1)) &  # Fourth opens above third close
            (close_prices < open_prices.shift(3))    # Fourth closes below first open
        )
        
        bullish_pattern = three_bullish_candles & bearish_fourth
        
        # Bearish Three Line Strike
        three_bearish_candles = (
            (close_prices.shift(3) < open_prices.shift(3)) &  # First candle bearish
            (close_prices.shift(2) < open_prices.shift(2)) &  # Second candle bearish
            (close_prices.shift(1) < open_prices.shift(1)) &  # Third candle bearish
            (close_prices.shift(2) < close_prices.shift(3)) &  # Each close lower than previous
            (close_prices.shift(1) < close_prices.shift(2))
        )
        
        bullish_fourth = (
            (open_prices < close_prices.shift(1)) &  # Fourth opens below third close
            (close_prices > open_prices.shift(3))    # Fourth closes above first open
        )
        
        bearish_pattern = three_bearish_candles & bullish_fourth
        
        # Combine into a single series
        result = bullish_pattern.astype(int) - bearish_pattern.astype(int)
        
        # NaN 값을 0으로 대체
        result = result.fillna(0)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in identify_three_line_strike: {str(e)}")
        return pd.Series(0, index=df.index)


@log_execution
def handle_nan_values(df: pd.DataFrame) -> pd.DataFrame:
     """ Handle NaN values in the DataFrame after indicator calculation """
     result = df.copy()
     total_nan_sum = result.isna().sum().sum()
     total_nan_count = int(total_nan_sum.item()) if hasattr(total_nan_sum, 'item') else int(total_nan_sum)

     if total_nan_count > 0:
          logger.info(f"지표 계산 후 {total_nan_count}개의 NaN 값이 남아있어 처리합니다.")
          numeric_cols = result.select_dtypes(include=np.number).columns.tolist()

          for col in numeric_cols:
               col_nan_sum = result[col].isna().sum()
               col_nan_count = int(col_nan_sum.item()) if hasattr(col_nan_sum, 'item') else int(col_nan_sum)

               if col_nan_count > 0:
                    if col_nan_count == len(result):
                         fill_value = 0
                         result[col] = fill_value
                         logger.info(f"컬럼 '{col}' 전체가 NaN이므로 {fill_value}로 대체했습니다.")
                    else:
                         try:
                              # Try filling with median first
                              fill_value = result[col].median()
                              if pd.isna(fill_value):
                                   # If median is NaN, try mean
                                   fill_value = result[col].mean()
                                   if pd.isna(fill_value):
                                        # If mean is also NaN, use 0
                                        fill_value = 0
                              result[col] = result[col].fillna(fill_value)
                              logger.info(f"컬럼 '{col}'의 {col_nan_count}개 NaN 값을 {fill_value}(median/mean/0)로 대체했습니다.")
                         except Exception as fill_e:
                              logger.warning(f"컬럼 '{col}' NaN 처리 중 오류 발생 ({fill_e}). 0으로 대체합니다.")
                              result[col] = result[col].fillna(0)
     return result

@log_execution
def calculate_percent_b(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, window_dev: int = 2) -> pd.Series:
    """Calculate Bollinger Bands %B"""
    try:
        # Ensure inputs are numeric
        high = pd.to_numeric(high, errors='coerce')
        low = pd.to_numeric(low, errors='coerce')
        close = pd.to_numeric(close, errors='coerce')

        if high.isnull().all() or low.isnull().all() or close.isnull().all():
            logger.warning("%B 계산 입력 데이터에 문제가 있습니다.")
            return pd.Series(np.nan, index=close.index)

        bollinger = BollingerBands(close=close, window=window, window_dev=window_dev, fillna=True)
        bb_high = bollinger.bollinger_hband()
        bb_low = bollinger.bollinger_lband()
        
        # Avoid division by zero when bands are equal
        band_diff = bb_high - bb_low
        percent_b = (close - bb_low) / band_diff.replace(0, np.nan) # Replace 0 with NaN before division

        percent_b.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero/inf
        # Fill NaNs (from division by zero or initial calculation)
        return percent_b.fillna(method='ffill').fillna(method='bfill').fillna(0.5) # Fill remaining NaNs with 0.5 (mid-band)
    except Exception as e:
        logger.error(f"Error calculating %B: {e}")
        logger.debug(traceback.format_exc())
        return pd.Series(np.nan, index=close.index)

@log_execution
def calculate_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Chaikin Money Flow (CMF)"""
    try:
        # Ensure inputs are numeric
        high = pd.to_numeric(high, errors='coerce')
        low = pd.to_numeric(low, errors='coerce')
        close = pd.to_numeric(close, errors='coerce')
        volume = pd.to_numeric(volume, errors='coerce')

        if high.isnull().all() or low.isnull().all() or close.isnull().all() or volume.isnull().all():
            logger.warning("CMF 계산 입력 데이터에 문제가 있습니다.")
            return pd.Series(np.nan, index=close.index)

        # Avoid division by zero if high equals low
        hl_diff = (high - low).replace(0, np.nan) # Replace 0 difference with NaN

        # Calculate Money Flow Multiplier safely
        mfm = ((close - low) - (high - close)) / hl_diff
        mfm = mfm.fillna(0) # Fill NaNs where high == low or calculation resulted in NaN

        # Calculate Money Flow Volume
        mfv = mfm * volume.fillna(0) # Use filled volume

        # Calculate CMF denominator safely
        vol_sum = volume.rolling(window=window, min_periods=1).sum().replace(0, np.nan)

        cmf = mfv.rolling(window=window, min_periods=1).sum() / vol_sum
        cmf.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle potential division by zero
        return cmf.fillna(method='ffill').fillna(method='bfill').fillna(0) # Fill NaNs with 0

    except Exception as e:
        logger.error(f"Error calculating CMF: {e}")
        logger.debug(traceback.format_exc())
        return pd.Series(np.nan, index=close.index)


@log_execution
def calculate_ema_dist_norm(close: pd.Series, window_fast: int = 12, window_slow: int = 26) -> pd.Series:
    """Calculate normalized distance between fast and slow EMAs"""
    try:
        close_numeric = pd.to_numeric(close, errors='coerce')
        if close_numeric.isnull().all():
             logger.warning("EMA Distance Norm 계산 입력 데이터(close)에 문제가 있습니다.")
             return pd.Series(np.nan, index=close.index)

        ema_fast = EMAIndicator(close=close_numeric, window=window_fast, fillna=True).ema_indicator()
        ema_slow = EMAIndicator(close=close_numeric, window=window_slow, fillna=True).ema_indicator()
        ema_dist = ema_fast - ema_slow

        # Normalize the distance using division by close price (handle zero close price)
        close_safe = close_numeric.replace(0, np.nan)
        ema_dist_norm = ema_dist / close_safe

        ema_dist_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaNs (from division by zero or initial calculation)
        return ema_dist_norm.fillna(method='ffill').fillna(method='bfill').fillna(0) # Fill remaining NaNs with 0
    except Exception as e:
        logger.error(f"Error calculating EMA Distance Norm: {e}")
        logger.debug(traceback.format_exc())
        return pd.Series(np.nan, index=close.index)

@log_execution
def calculate_all_indicators(df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
    """
    Calculate all technical indicators ensuring consistent columns.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        timeframe (str, optional): Timeframe identifier for logging. Defaults to None.
        
    Returns:
        pd.DataFrame: DataFrame with all indicators
    """
    if df is None or df.empty:
        logger.warning(f"입력 DataFrame이 비어 있어 지표를 계산할 수 없습니다. timeframe: {timeframe or '알 수 없음'}")
        # 빈 DataFrame에 모든 가능한 컬럼 추가하여 반환
        all_cols = df.columns.tolist() if df is not None else []
        all_cols = all_cols + ALL_POSSIBLE_INDICATOR_COLUMNS
        return pd.DataFrame(columns=list(dict.fromkeys(all_cols)))  # 중복 제거된 컬럼 목록

    logger.info(f"'{timeframe or '알 수 없는 기간'}'에 대한 모든 기술적 지표 계산 시작")
    logger.debug(f"입력 데이터 크기: {df.shape}, 컬럼: {df.columns.tolist()}, 인덱스 타입: {type(df.index)}")
    
    # 안전한 데이터프레임 복사
    try:
        # 인덱스 타입 확인 및 변환
        result = df.copy(deep=True)
        
        # 데이터타임 인덱스 확인 및 변환
        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                logger.info("인덱스를 DatetimeIndex로 변환합니다.")
                # 인덱스 이름 저장
                index_name = result.index.name
                
                # 인덱스를 datetime으로 변환
                result.index = pd.to_datetime(result.index, errors='coerce')
                
                # 변환 실패한 경우 (NaT) 확인
                nat_count = result.index.isna().sum()
                if nat_count > 0:
                    logger.warning(f"인덱스 변환 중 {nat_count}개의 NaT가 발생했습니다.")
                    # NaT 인덱스 제거
                    result = result[~result.index.isna()]
                    if result.empty:
                        logger.error("모든 인덱스가 NaT로 변환되어 빈 DataFrame이 됐습니다.")
                        return pd.DataFrame(columns=df.columns.tolist() + ALL_POSSIBLE_INDICATOR_COLUMNS)
                
                # 인덱스 이름 복원
                if index_name:
                    result.index.name = index_name
                
                logger.info(f"인덱스 변환 완료. 변환 후 크기: {result.shape}")
            except Exception as e:
                logger.error(f"인덱스 변환 실패: {str(e)}")
                logger.warning("원본 인덱스를 유지합니다.")
        
        # 인덱스 정렬 - 중요: 시간 순서대로 정렬하여 계산 보장
        try:
            # 중복 인덱스 확인
            if result.index.duplicated().any():
                dup_count = result.index.duplicated().sum()
                logger.warning(f"중복된 인덱스가 {dup_count}개 발견되었습니다. 첫 번째 값을 유지합니다.")
                result = result[~result.index.duplicated(keep='first')]
            
            # 인덱스 정렬
            result = result.sort_index()
        except Exception as e:
            logger.error(f"인덱스 정렬 실패: {str(e)}")
            logger.warning("정렬 없이 계속 진행합니다.")
        
        # 필수 컬럼 확인
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in result.columns]
        if missing_cols:
            logger.error(f"필수 컬럼이 누락되었습니다: {missing_cols}")
            # 누락된 컬럼을 NaN으로 추가
            for col in missing_cols:
                logger.warning(f"누락된 컬럼 '{col}'을 NaN으로 추가합니다.")
                result[col] = np.nan
        
        # 데이터 타입 확인 및 변환
        for col in required_cols:
            if col in result.columns and not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(f"컬럼 '{col}'이 숫자형이 아닙니다. 숫자형으로 변환합니다.")
                original_type = result[col].dtype
                result[col] = pd.to_numeric(result[col], errors='coerce')
                nan_count = result[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"'{col}' 변환 중 {nan_count}개의 NaN이 발생했습니다.")
        
        # NaN 값 초기 확인
        nan_count_before = result[required_cols].isna().sum().sum()
        if nan_count_before > 0:
            logger.warning(f"지표 계산 전 필수 컬럼에 {nan_count_before}개의 NaN이 있습니다.")
            # 연속된 NaN이 너무 많으면 경고
            max_consecutive_nans = max(result[required_cols].isna().sum(axis=1))
            if max_consecutive_nans >= 3:  # 한 행에 3개 이상의 NaN이 있으면 경고
                logger.warning(f"일부 행에 너무 많은 NaN이 있습니다 (최대 {max_consecutive_nans}/5개).")
            
            # NaN을 앞/뒤 값으로 채움
            for col in required_cols:
                if col in result.columns:
                    na_count = result[col].isna().sum()
                    if na_count > 0:
                        before = na_count
                        result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
                        after = result[col].isna().sum()
                        logger.info(f"'{col}' 컬럼의 NaN을 {before}개 → {after}개로 처리했습니다.")
    except Exception as e:
        logger.error(f"DataFrame 준비 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        # 오류 발생 시 원본 데이터프레임 사용
        logger.warning("원본 DataFrame을 사용하여 진행합니다.")
    result = df.copy()

    # 원본 컬럼 목록 저장
    original_columns = result.columns.tolist()
    
    # 모든 가능한 지표 컬럼 초기화 (존재하지 않는 경우에만)
    for col in ALL_POSSIBLE_INDICATOR_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    # 각 지표 함수 정의
    indicator_functions = [
        add_basic_indicators,
        add_advanced_indicators,
        add_custom_indicators,
        add_pattern_recognition
    ]

    # 지표 계산 적용
    for indicator_func in indicator_functions:
        func_name = indicator_func.__name__
        try:
            logger.debug(f"{func_name} 적용 중...")

            # 함수 실행을 위한 안전한 데이터프레임 복사
            temp_df = result.copy(deep=True)

            # 타입 변환 및 NaN 처리 재확인
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col in temp_df.columns:
                    if not pd.api.types.is_numeric_dtype(temp_df[col]):
                        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                    if temp_df[col].isna().any():
                        temp_df[col] = temp_df[col].fillna(method='ffill').fillna(method='bfill')

            # 함수 실행
            indicator_df = indicator_func(temp_df)

            # 실행 결과 검증
            if indicator_df is None:
                logger.error(f"{func_name} 실행 결과가 None입니다.")
                continue # 다음 함수로 넘어감

            if indicator_df.empty:
                logger.error(f"{func_name} 실행 결과가 빈 DataFrame입니다.")
                continue # 다음 함수로 넘어감

            # 인덱스 호환성 확인 및 처리
            if not result.index.equals(indicator_df.index):
                logger.warning(f"{func_name} 실행 결과의 인덱스가 원본과 다릅니다. 재조정 시도.")
                try:
                    indicator_df_reindexed = indicator_df.reindex(result.index)
                    # Check if reindexing created too many NaNs (optional)
                    # if indicator_df_reindexed.isna().sum().sum() > indicator_df.isna().sum().sum() * 1.1:
                    #     logger.warning("Reindexing created many NaNs. Skipping merge for this function.")
                    #     continue
                    indicator_df = indicator_df_reindexed
                except Exception as idx_err:
                    logger.error(f"인덱스 재조정 중 오류: {str(idx_err)}. {func_name} 결과 적용 건너뜀.")
                    continue # 다음 함수로 넘어감

            # 계산된 지표를 원본 결과에 병합
            new_cols = [col for col in indicator_df.columns if col not in original_columns]

            if not new_cols:
                logger.debug(f"{func_name}에서 새로운 컬럼이 생성되지 않았습니다.")
                continue # 다음 함수로 넘어감

            # 컬럼별로 데이터 복사 (인덱스 기준으로)
            for col in new_cols:
                if col in indicator_df.columns:
                    # Use loc for safer index alignment during assignment
                    result.loc[:, col] = indicator_df.loc[:, col]

            logger.debug(f"{func_name} 적용 완료. {len(new_cols)}개 컬럼 추가됨")

        except Exception as e: # 여기가 try 블록(1683라인)에 대한 except 블록
            logger.error(f"{func_name} 적용 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
    
    # NaN 비율 확인
    indicator_cols = [col for col in result.columns if col not in original_columns]
    if indicator_cols:
        nan_ratio = result[indicator_cols].isna().mean().mean()
        logger.info(f"지표 컬럼의 평균 NaN 비율: {nan_ratio:.2%}")
        
        # 지표별 NaN 비율이 높은 컬럼 로깅
        high_nan_cols = []
        for col in indicator_cols:
            col_nan_ratio = result[col].isna().mean()
            if col_nan_ratio > 0.5:  # 50% 이상 NaN인 컬럼
                high_nan_cols.append((col, col_nan_ratio))
        
        if high_nan_cols:
            logger.warning(f"NaN 비율이 높은 지표들: {high_nan_cols}")
    
    # 데이터 길이 확인
    min_required_rows = 200  # 가장 긴 지표 기간 기준으로 조정
    if len(result) < min_required_rows:
        logger.warning(f"데이터 길이({len(result)})가 신뢰할 수 있는 지표 계산에 너무 짧을 수 있습니다.")
    
    # NaN 값 처리
    try:
        logger.info("NaN 값 처리 시작...")
        result = handle_nan_values(result)
        logger.info("NaN 값 처리 완료")
    except Exception as e:
        logger.error(f"NaN 처리 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 기본 NaN 처리 방법 적용
        logger.warning("기본 NaN 처리 방법을 적용합니다.")
        for col in result.columns:
            if col not in original_columns:
                na_count_before = result[col].isna().sum()
                if na_count_before > 0:
                    # 앞의 값으로 채우기 -> 뒤의 값으로 채우기 -> 0으로 채우기
                    result[col] = result[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    na_count_after = result[col].isna().sum()
                    
                    # 모든 값이 NaN인 경우 특별 처리
                    if na_count_after == len(result):
                        logger.warning(f"컬럼 '{col}' 전체가 NaN이므로 0로 대체했습니다.")
                        result[col] = 0

    # 누락된 컬럼 확인 및 추가
    final_columns = original_columns + [col for col in ALL_POSSIBLE_INDICATOR_COLUMNS 
                                       if col not in original_columns]
    missing_columns = [col for col in final_columns if col not in result.columns]
    
    for col in missing_columns:
        logger.warning(f"컬럼 '{col}'이 예기치 않게 누락되었습니다. 0으로 추가합니다.")
        result[col] = 0
    
    # 컬럼 순서 정리 (중복 제거 및 순서 유지)
    unique_final_columns = list(dict.fromkeys(final_columns))
    try:
        result = result[unique_final_columns]
    except KeyError as e:
        logger.error(f"컬럼 재정렬 중 오류: {str(e)}")
        # 실패하면 기존 컬럼 유지
    
    # 결과 데이터프레임 검증
    logger.info(f"모든 지표 계산 완료. 최종 크기: {result.shape}")
    
    # 데이터 무결성 검사
    if result.isna().any().any():
        nan_count = result.isna().sum().sum()
        logger.warning(f"결과에 {nan_count}개의 NaN이 있습니다. 남은 NaN을 0으로 채웁니다.")
        result = result.fillna(0)
    
    # 인덱스 중복 확인
    if result.index.duplicated().any():
        dup_count = result.index.duplicated().sum()
        logger.warning(f"결과에 {dup_count}개의 중복 인덱스가 있습니다. 이로 인해 문제가 발생할 수 있습니다.")
    
    # 데이터 유형 불일치 확인 (문자열 컬럼 등)
    non_numeric_cols = result.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"결과에 비숫자형 컬럼이 있습니다: {non_numeric_cols}")
        
        # 비숫자형 컬럼 변환 시도
        for col in non_numeric_cols:
            try:
                result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)
                logger.info(f"컬럼 '{col}'을 숫자형으로 변환했습니다.")
            except Exception as e:
                logger.error(f"컬럼 '{col}' 변환 중 오류: {str(e)}")
    
    # 최종 무결성 검사
    final_inf_check = np.isinf(result.select_dtypes(include=['number'])).any().any()
    if final_inf_check:
        logger.warning("최종 결과에 무한값(inf)이 있습니다. 0으로 대체합니다.")
        result = result.replace([np.inf, -np.inf], 0)

    return result

# 이전 버전과의 호환성을 위한 별칭
add_indicators = calculate_all_indicators


@log_execution
def filter_indicators(df: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
    """
    Filter DataFrame to include only specified indicators
    
    Args:
        df (pd.DataFrame): DataFrame with all indicators
        indicators (List[str], optional): List of indicator column names to keep. 
                                         If None, returns all columns.
                                         
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if indicators is None:
        return df
    
    # Always include OHLCV columns
    essential_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Add 'ticker' column if it exists
    if 'ticker' in df.columns:
        essential_columns.append('ticker')
    
    # Filter columns
    available_indicators = [col for col in indicators if col in df.columns]
    all_columns = essential_columns + available_indicators
    
    # Get unique columns (in case there are duplicates)
    unique_columns = list(dict.fromkeys(all_columns))
    
    # Return filtered DataFrame
    return df[unique_columns] 