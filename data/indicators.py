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
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, AccDistIndexIndicator, VolumeWeightedAveragePrice
from ta.others import DailyReturnIndicator
from sklearn.preprocessing import StandardScaler
from ta.momentum import WilliamsRIndicator, AwesomeOscillatorIndicator

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
    'cci', 'tsi', 'mfi', 'adi', 'roc10', 'ichimoku_tenkan_sen',
    'ichimoku_kijun_sen', 'ichimoku_senkou_span_a', 'ichimoku_senkou_span_b', 'ichimoku_chikou_span',
    'keltner_mband', 'keltner_hband', 'keltner_lband', 'hma9', 'ao', 'williams_r', 'vwap',
    # Custom Indicators
    'volume_change_1', 'price_change_1', 'price_volume_corr', 'market_breadth', 'volatility_ratio', 'trend_strength',
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
def calculate_price_volume_correlation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    안전하게 가격과 거래량 간의 상관관계를 계산하는 함수
    
    Args:
        df (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
        window (int): 상관관계 계산을 위한 윈도우 크기
        
    Returns:
        pd.Series: 가격과 거래량 간의 상관관계 시리즈
    """
    try:
        # 입력 데이터 확인 및 변환
        if 'close' not in df.columns or 'volume' not in df.columns:
            logger.warning("가격-거래량 상관관계 계산을 위한 필수 컬럼이 없습니다.")
            return pd.Series(np.nan, index=df.index)
            
        # 숫자형으로 변환
        close = pd.to_numeric(df['close'], errors='coerce')
        volume = pd.to_numeric(df['volume'], errors='coerce')
        
        # 데이터 유효성 검사
        if close.isna().all() or volume.isna().all():
            logger.warning("가격 또는 거래량 데이터가 모두 NaN입니다.")
            return pd.Series(np.nan, index=df.index)
            
        # 변화율 계산 (상관관계는 가격이 아닌 변화율로 계산하는 것이 더 의미 있음)
        close_changes = close.pct_change().fillna(0)
        volume_changes = volume.pct_change().fillna(0)
        
        # 무한값 처리
        close_changes = close_changes.replace([np.inf, -np.inf], np.nan)
        volume_changes = volume_changes.replace([np.inf, -np.inf], np.nan)
        
        # 상관관계 시리즈 초기화
        correlation = pd.Series(np.nan, index=df.index)
        
        # 롤링 윈도우로 상관관계 계산
        for i in range(window, len(df) + 1):
            window_close = close_changes.iloc[i-window:i]
            window_volume = volume_changes.iloc[i-window:i]
            
            # 데이터 유효성 확인
            if window_close.isna().all() or window_volume.isna().all():
                correlation.iloc[i-1] = np.nan
                continue
                
            # 결측치 제거
            valid_data = pd.concat([window_close, window_volume], axis=1).dropna()
            if len(valid_data) <= 1:  # 최소 2개 이상의 데이터 필요
                correlation.iloc[i-1] = np.nan
                continue
                
            # 표준편차 확인 (0인 경우 상관관계 계산 불가)
            close_std = valid_data.iloc[:, 0].std()
            volume_std = valid_data.iloc[:, 1].std()
            
            if close_std == 0 or volume_std == 0 or np.isnan(close_std) or np.isnan(volume_std):
                correlation.iloc[i-1] = 0  # 표준편차가 0이면 상관관계 0으로 설정
                # 표준편차가 0인 경우 로그 추가
                logger.debug(f"표준편차가 0 또는 NaN: close_std={close_std}, volume_std={volume_std}, 인덱스: {i-1}")
                continue
                
            # np.corrcoef 대신 pandas의 corr() 메서드를 사용하여 안전하게 계산
            try:
                corr_value = valid_data.iloc[:, 0].corr(valid_data.iloc[:, 1])
                correlation.iloc[i-1] = corr_value if not np.isnan(corr_value) else 0
            except Exception as corr_err:
                logger.debug(f"상관관계 계산 중 오류: {corr_err} (인덱스: {i-1})")
                correlation.iloc[i-1] = 0
        
        # NaN 값을 0으로 채우기
        correlation = correlation.fillna(0)
        
        return correlation
        
    except Exception as e:
        logger.error(f"가격-거래량 상관관계 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(0, index=df.index)  # 오류 발생 시 0으로 채워진 시리즈 반환

@log_execution
def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    기본 기술적 지표를 데이터프레임에 추가합니다.
    
    Args:
        df (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
        
    Returns:
        pd.DataFrame: 기본 지표가 추가된 데이터프레임
    """
    try:
        # 단순이동평균 (SMA)
        df['sma7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['sma25'] = df['close'].rolling(window=25, min_periods=1).mean()
        df['sma50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['sma100'] = df['close'].rolling(window=100, min_periods=1).mean()
        df['sma200'] = df['close'].rolling(window=200, min_periods=1).mean()
        
        # 지수이동평균 (EMA)
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # 볼린저 밴드 (Bollinger Bands)
        middle_band = df['close'].rolling(window=20, min_periods=1).mean()
        std_dev = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = middle_band + (std_dev * 2)
        df['bb_middle'] = middle_band
        df['bb_lower'] = middle_band - (std_dev * 2)
        
        # RSI
        df['rsi14'] = calculate_rsi(df['close'], period=14)
        
        # MACD
        macd_line = df['ema12'] - df['ema26']
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_diff'] = macd_line - signal_line
        
        # 스토캐스틱 (Stochastic Oscillator)
        df['stoch_k'] = calculate_stochastic(df, period=14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()
        
        # ATR (Average True Range)
        df['atr'] = calculate_atr(df, period=14)
        
        # OBV (On-Balance Volume)
        df['obv'] = calculate_obv(df)
        
        # 일일 수익률
        df['daily_return'] = df['close'].pct_change()
        
        logger.info("기본 지표 계산 완료")
        return df
    except Exception as e:
        logger.error(f"기본 지표 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # 오류 발생 시 원본 데이터프레임 반환

@log_execution
def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    고급 기술적 지표를 데이터프레임에 추가합니다.
    
    Args:
        df (pd.DataFrame): 기본 지표가 포함된 데이터프레임
        
    Returns:
        pd.DataFrame: 고급 지표가 추가된 데이터프레임
    """
    try:
        # 추가 RSI 기간
        df['rsi7'] = calculate_rsi(df['close'], period=7)
        df['rsi21'] = calculate_rsi(df['close'], period=21)
        
        # 추가 이동평균
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
        
        # 파라볼릭 SAR
        df['psar'] = calculate_parabolic_sar(df)
        
        # ADX
        df['adx'] = calculate_adx(df)
        
        # Ichimoku Cloud 컴포넌트
        ichimoku = calculate_ichimoku(df)
        if not ichimoku.empty and not ichimoku.isnull().all().all():
            df = pd.concat([df, ichimoku], axis=1)
        
        # Keltner Channel
        keltner = calculate_keltner_channel(df)
        if not keltner.empty and not keltner.isnull().all().all():
            df = pd.concat([df, keltner], axis=1)
        
        # Williams %R
        df['williams_r'] = calculate_williams_r(df)
        
        # CCI (Commodity Channel Index)
        df['cci'] = calculate_cci(df)
        
        # Chaikin Oscillator
        df['chaikin_oscillator'] = calculate_chaikin_oscillator(df)
        
        # CMF (Chaikin Money Flow)
        df['cmf'] = calculate_cmf(df['high'], df['low'], df['close'], df['volume'])
        
        # MFI (Money Flow Index)
        df['mfi'] = calculate_mfi(df)
        
        # 모멘텀 지표
        df['roc'] = calculate_rate_of_change(df['close'], 10)
        
        # Awesome Oscillator
        df['awesome_oscillator'] = calculate_awesome_oscillator(df)
        
        logger.info("고급 지표 계산 완료")
        return df
    except Exception as e:
        logger.error(f"고급 지표 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # 오류 발생 시 원본 데이터프레임 반환

@log_execution
def add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    사용자 정의 기술적 지표를 데이터프레임에 추가합니다.
    
    Args:
        df (pd.DataFrame): 기본 및 고급 지표가 포함된 데이터프레임
        
    Returns:
        pd.DataFrame: 사용자 정의 지표가 추가된 데이터프레임
    """
    try:
        # 볼리전 밴드 %B
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['percent_b'] = calculate_percent_b(df['high'], df['low'], df['close'])
        
        # 이동평균 변화율
        if 'ema9' in df.columns and 'ema21' in df.columns:
            df['ema_ratio'] = (df['ema9'] / df['ema21']) - 1
        
        # 상대 강도 (RS)
        if 'rsi14' in df.columns:
            # RSI 공식에서 RS = 100 / (100 - RSI) - 1
            df['rs'] = 100 / (100 - df['rsi14']) - 1
        
        # 변동성 비율
        if 'atr' in df.columns and 'close' in df.columns:
            df['volatility_ratio'] = df['atr'] / df['close'] * 100
        
        # 이동평균 교차 신호
        if 'ema12' in df.columns and 'ema26' in df.columns:
            # 0보다 크면 골든 크로스, 작으면 데드 크로스
            df['ema_cross'] = df['ema12'] - df['ema26']
            df['ema_cross_signal'] = np.where(df['ema_cross'] > 0, 1, np.where(df['ema_cross'] < 0, -1, 0))
        
        # 추세 강도 지표
        if 'adx' in df.columns:
            df['trend_strength'] = np.where(df['adx'] > 25, 1, np.where(df['adx'] < 20, -1, 0))
        
        # 가격 추세 방향
        if 'close' in df.columns and 'ema50' in df.columns:
            df['price_trend'] = np.where(df['close'] > df['ema50'], 1, np.where(df['close'] < df['ema50'], -1, 0))
        
        # 통합 모멘텀 지표
        momentum_cols = ['rsi14', 'macd', 'stoch_k']
        if all(col in df.columns for col in momentum_cols):
            # RSI 기준: 30 미만은 과매도, 70 초과는 과매수
            rsi_signal = np.where(df['rsi14'] < 30, 1, np.where(df['rsi14'] > 70, -1, 0))
            
            # MACD 기준: 0보다 크면 상승, 작으면 하락
            macd_signal = np.where(df['macd'] > 0, 1, np.where(df['macd'] < 0, -1, 0))
            
            # 스토캐스틱 기준: 20 미만은 과매도, 80 초과는 과매수
            stoch_signal = np.where(df['stoch_k'] < 20, 1, np.where(df['stoch_k'] > 80, -1, 0))
            
            # 통합 모멘텀 (-3 ~ 3 범위)
            df['combined_momentum'] = rsi_signal + macd_signal + stoch_signal
        
        logger.info("사용자 정의 지표 계산 완료")
        return df
    except Exception as e:
        logger.error(f"사용자 정의 지표 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # 오류 발생 시 원본 데이터프레임 반환

@log_execution
def add_pattern_recognition(df: pd.DataFrame) -> pd.DataFrame:
    """
    캔들스틱 패턴 인식 결과를 데이터프레임에 추가합니다.
    
    Args:
        df (pd.DataFrame): 기술적 지표가 포함된 데이터프레임
        
    Returns:
        pd.DataFrame: 패턴 인식 결과가 추가된 데이터프레임
    """
    try:
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"캔들스틱 패턴 인식에 필요한 컬럼({required_cols}) 없음. 패턴 인식 스킵.")
            return df
            
        # 간단한 패턴 인식 구현
        # 도지 패턴
        df['pattern_doji'] = np.where(
            abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) < 0.1,
            100,  # 존재하면 100 (또는 양수 값)
            0     # 없으면 0
        )
        
        # 해머 패턴 (간단 구현)
        df['pattern_hammer'] = np.where(
            (df['close'] > df['open']) &  # 양봉
            ((df['high'] - df['close']) < 0.2 * (df['high'] - df['low'] + 1e-8)) &  # 위쪽 꼬리가 짧음
            ((df['open'] - df['low']) > 0.6 * (df['high'] - df['low'] + 1e-8)),  # 아래쪽 꼬리가 긺
            100,
            0
        )
        
        # 행잉맨 패턴 (간단 구현)
        df['pattern_hanging_man'] = np.where(
            (df['close'] < df['open']) &  # 음봉
            ((df['high'] - df['open']) < 0.2 * (df['high'] - df['low'] + 1e-8)) &  # 위쪽 꼬리가 짧음
            ((df['close'] - df['low']) > 0.6 * (df['high'] - df['low'] + 1e-8)),  # 아래쪽 꼬리가 긺
            -100,
            0
        )
        
        # 슈팅스타 패턴 (간단 구현)
        df['pattern_shooting_star'] = np.where(
            (df['close'] < df['open']) &  # 음봉
            ((df['high'] - df['open']) > 0.6 * (df['high'] - df['low'] + 1e-8)) &  # 위쪽 꼬리가 긺
            ((df['close'] - df['low']) < 0.2 * (df['high'] - df['low'] + 1e-8)),  # 아래쪽 꼬리가 짧음
            -100,
            0
        )
        
        # 엥걸핑 패턴 (간단 구현)
        engulfing_values = np.zeros(len(df))
        for i in range(1, len(df)):
            # 양봉 엥걸핑 (매수 신호)
            if (df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i-1] < df['open'].iloc[i-1] and
                df['close'].iloc[i] > df['open'].iloc[i-1] and
                df['open'].iloc[i] < df['close'].iloc[i-1]):
                engulfing_values[i] = 100
            # 음봉 엥걸핑 (매도 신호)
            elif (df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i-1] > df['open'].iloc[i-1] and
                df['close'].iloc[i] < df['open'].iloc[i-1] and
                df['open'].iloc[i] > df['close'].iloc[i-1]):
                engulfing_values[i] = -100
        df['pattern_engulfing'] = engulfing_values
        
        # 모닝스타 패턴 (간단 구현)
        morning_star_values = np.zeros(len(df))
        for i in range(2, len(df)):
            # 첫번째 캔들: 큰 음봉
            # 두번째 캔들: 작은 몸통
            # 세번째 캔들: 큰 양봉
            if (df['close'].iloc[i-2] < df['open'].iloc[i-2] and  # 첫번째 캔들 음봉
                abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) < 0.3 * abs(df['close'].iloc[i-2] - df['open'].iloc[i-2]) and  # 두번째 캔들 작은 몸통
                df['close'].iloc[i] > df['open'].iloc[i] and  # 세번째 캔들 양봉
                df['close'].iloc[i] > (df['open'].iloc[i-2] + df['close'].iloc[i-2]) / 2):  # 세번째 캔들이 첫번째 캔들의 중간 이상 상승
                morning_star_values[i] = 100
        df['pattern_morning_star'] = morning_star_values
        
        # 이브닝스타 패턴 (간단 구현)
        evening_star_values = np.zeros(len(df))
        for i in range(2, len(df)):
            # 첫번째 캔들: 큰 양봉
            # 두번째 캔들: 작은 몸통
            # 세번째 캔들: 큰 음봉
            if (df['close'].iloc[i-2] > df['open'].iloc[i-2] and  # 첫번째 캔들 양봉
                abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) < 0.3 * abs(df['close'].iloc[i-2] - df['open'].iloc[i-2]) and  # 두번째 캔들 작은 몸통
                df['close'].iloc[i] < df['open'].iloc[i] and  # 세번째 캔들 음봉
                df['close'].iloc[i] < (df['open'].iloc[i-2] + df['close'].iloc[i-2]) / 2):  # 세번째 캔들이 첫번째 캔들의 중간 이하로 하락
                evening_star_values[i] = -100
        df['pattern_evening_star'] = evening_star_values
        
        # 쓰리 화이트 솔저스 (간단 구현)
        three_white_soldiers_values = np.zeros(len(df))
        for i in range(2, len(df)):
            # 세 개의 연속적인 양봉, 각 캔들의 시가가 이전 캔들의 몸통 내에 있음
            if (df['close'].iloc[i-2] > df['open'].iloc[i-2] and  # 첫번째 캔들 양봉
                df['close'].iloc[i-1] > df['open'].iloc[i-1] and  # 두번째 캔들 양봉
                df['close'].iloc[i] > df['open'].iloc[i] and  # 세번째 캔들 양봉
                df['open'].iloc[i-1] > df['open'].iloc[i-2] and  # 두번째 캔들 시가 > 첫번째 캔들 시가
                df['open'].iloc[i] > df['open'].iloc[i-1] and  # 세번째 캔들 시가 > 두번째 캔들 시가
                df['close'].iloc[i-1] > df['close'].iloc[i-2] and  # 두번째 캔들 종가 > 첫번째 캔들 종가
                df['close'].iloc[i] > df['close'].iloc[i-1]):  # 세번째 캔들 종가 > 두번째 캔들 종가
                three_white_soldiers_values[i] = 100
        df['pattern_three_white_soldiers'] = three_white_soldiers_values
        
        # 쓰리 블랙 크로우즈 (간단 구현)
        three_black_crows_values = np.zeros(len(df))
        for i in range(2, len(df)):
            # 세 개의 연속적인 음봉, 각 캔들의 시가가 이전 캔들의 몸통 내에 있음
            if (df['close'].iloc[i-2] < df['open'].iloc[i-2] and  # 첫번째 캔들 음봉
                df['close'].iloc[i-1] < df['open'].iloc[i-1] and  # 두번째 캔들 음봉
                df['close'].iloc[i] < df['open'].iloc[i] and  # 세번째 캔들 음봉
                df['open'].iloc[i-1] < df['open'].iloc[i-2] and  # 두번째 캔들 시가 < 첫번째 캔들 시가
                df['open'].iloc[i] < df['open'].iloc[i-1] and  # 세번째 캔들 시가 < 두번째 캔들 시가
                df['close'].iloc[i-1] < df['close'].iloc[i-2] and  # 두번째 캔들 종가 < 첫번째 캔들 종가
                df['close'].iloc[i] < df['close'].iloc[i-1]):  # 세번째 캔들 종가 < 두번째 캔들 종가
                three_black_crows_values[i] = -100
        df['pattern_three_black_crows'] = three_black_crows_values
        
        # 추세 확인 컬럼 추가
        if 'ema50' in df.columns:
            uptrend = df['close'] > df['ema50']
            
            # 패턴 신호 강도 조정 (추세와 일치하면 강화, 아니면 약화)
            for col in df.columns:
                if col.startswith('pattern_'):
                    # 매수 신호 (양수)와 상승 추세가 일치하면 신호 강화 (1.5배)
                    df[col] = np.where(
                        (df[col] > 0) & uptrend,
                        df[col] * 1.5,
                        df[col]
                    )
                    
                    # 매도 신호 (음수)와 하락 추세가 일치하면 신호 강화 (1.5배)
                    df[col] = np.where(
                        (df[col] < 0) & (~uptrend),
                        df[col] * 1.5,
                        df[col]
                    )
                    
                    # 매수 신호와 하락 추세가 충돌하면 신호 약화 (0.5배)
                    df[col] = np.where(
                        (df[col] > 0) & (~uptrend),
                        df[col] * 0.5,
                        df[col]
                    )
                    
                    # 매도 신호와 상승 추세가 충돌하면 신호 약화 (0.5배)
                    df[col] = np.where(
                        (df[col] < 0) & uptrend,
                        df[col] * 0.5,
                        df[col]
                    )
        
        # 통합 패턴 신호
        pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
        if pattern_cols:
            df['pattern_signal'] = df[pattern_cols].sum(axis=1)
            
            # 신호를 -100 ~ +100 범위로 정규화
            max_signal = np.abs(df['pattern_signal']).max()
            if max_signal > 0:
                df['pattern_signal'] = df['pattern_signal'] * (100 / max_signal)
        
        logger.info("패턴 인식 계산 완료")
        return df
    except Exception as e:
        logger.error(f"패턴 인식 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # 오류 발생 시 원본 데이터프레임 반환

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
def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    윌리엄스 %R을 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLC 데이터가 포함된 데이터프레임
        period (int): 계산 기간
        
    Returns:
        pd.Series: 윌리엄스 %R 값 시리즈
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.Series(np.nan, index=range(len(df) if hasattr(df, '__len__') else 0))
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"윌리엄스 %R 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.Series(np.nan, index=df.index)
        
        # 데이터 변환
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        
        # 기간 내 최고가, 최저가 계산
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()
        
        # 윌리엄스 %R 계산
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        # 0으로 나누는 경우 처리
        williams_r = williams_r.replace([np.inf, -np.inf], np.nan)
        
        # NaN 값 처리
        williams_r = williams_r.fillna(method='ffill').fillna(method='bfill').fillna(-50)
        
        return williams_r
    except Exception as e:
        logger.error(f"윌리엄스 %R 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(-50, index=df.index)  # 오류 발생 시 -50 반환


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
    켈트너 채널(Keltner Channel)을 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLC 데이터가 포함된 데이터프레임
        ema_period (int): EMA 계산 기간
        atr_period (int): ATR 계산 기간
        multiplier (float): ATR 곱셈 계수
        
    Returns:
        pd.DataFrame: 켈트너 채널 컴포넌트가 포함된 데이터프레임
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.DataFrame()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"켈트너 채널 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.DataFrame()
        
        # 데이터 변환
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        
        # 켈트너 채널 계산
        # 1. EMA 계산
        ema = close.ewm(span=ema_period, adjust=False).mean()
        
        # 2. ATR 계산
        atr = calculate_atr(df, period=atr_period)
        
        # 3. 채널 계산
        upper_band = ema + (multiplier * atr)
        lower_band = ema - (multiplier * atr)
        
        # 결과 데이터프레임 생성
        keltner_df = pd.DataFrame({
            'keltner_mband': ema,
            'keltner_hband': upper_band,
            'keltner_lband': lower_band
        }, index=df.index)
        
        # NaN 값 처리
        keltner_df = keltner_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return keltner_df
    except Exception as e:
        logger.error(f"켈트너 채널 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()  # 오류 발생 시 빈 DataFrame 반환

@log_execution
def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 기술적 지표를 계산하여 데이터프레임에 추가합니다.
    
    Args:
        df (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
        
    Returns:
        pd.DataFrame: 모든 지표가 추가된 데이터프레임
    """
    if df is None or len(df) < 14:  # 대부분의 지표에 14일 이상의 데이터 필요
        logger.warning("지표 계산을 위한 충분한 데이터가 없습니다.")
        return df

    try:
        # 기본 OHLCV 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"필수 OHLCV 컬럼이 누락되었습니다: {missing_columns}")
            return df
            
        # 계산 시작
        logger.info("기술적 지표 계산 시작...")
        
        # 1. 기본 지표 추가
        df = add_basic_indicators(df)
        
        # 2. 고급 지표 추가
        df = add_advanced_indicators(df)
        
        # 3. 사용자 정의 지표 추가
        try:
            # 새로 생성한 모듈에서 함수 임포트
            from data.indicators import add_custom_indicators
            df = add_custom_indicators(df)
        except ImportError:
            logger.warning("custom_indicators 모듈을 임포트할 수 없습니다. 사용자 정의 지표 계산을 건너뜁니다.")
            
        # 4. 패턴 인식 지표 추가
        try:
            # 새로 생성한 모듈에서 함수 임포트
            from data.indicators import add_pattern_recognition
            df = add_pattern_recognition(df)
        except ImportError:
            logger.warning("custom_indicators 모듈을 임포트할 수 없습니다. 패턴 인식 지표 계산을 건너뜁니다.")
        
        # 5. NaN 값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 앞쪽의 NaN 값 제거 (충분한 기간의 데이터가 있을 경우)
        if len(df) > 200:  # 200일 이상의 데이터가 있는 경우
            df = df.iloc[200:].copy()  # 앞쪽 200일 제거
        
        # 남은 NaN 값은 forward fill로 채우기
        df = df.fillna(method='ffill')
        
        # 그래도 남은 NaN 값은 backward fill로 채우기
        df = df.fillna(method='bfill')
        
        # 최종적으로 남은 NaN 값은 0으로 대체
        df = df.fillna(0)
        
        logger.info(f"모든 기술적 지표 계산 완료. 총 컬럼 수: {len(df.columns)}")
        return df
        
    except Exception as e:
        logger.error(f"지표 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # 오류 발생 시 원본 데이터프레임 반환

@log_execution
def filter_indicators(df: pd.DataFrame, indicator_list: List[str] = None) -> pd.DataFrame:
    """
    필터링된 지표들만 포함하는 데이터프레임을 반환합니다.
    
    Args:
        df (pd.DataFrame): 모든 지표가 포함된 데이터프레임
        indicator_list (List[str], optional): 유지할 지표 리스트. None이면 기본 지표 사용.
        
    Returns:
        pd.DataFrame: 필터링된 지표들만 포함하는 데이터프레임
    """
    try:
        if indicator_list is None:
            # 기본 주요 지표 목록
            indicator_list = [
                'open', 'high', 'low', 'close', 'volume',  # 기본 OHLCV
                'sma7', 'sma25', 'ema12', 'ema26', 'ema200',  # 이동평균
                'macd', 'macd_signal', 'macd_diff',  # MACD
                'rsi14',  # RSI
                'bb_upper', 'bb_middle', 'bb_lower',  # 볼린저 밴드
                'stoch_k', 'stoch_d',  # 스토캐스틱
                'atr', 'obv',  # 기타 인기 지표
                'daily_return'  # 수익률
            ]
        
        # 실제 데이터프레임에 존재하는 컬럼만 필터링
        available_columns = [col for col in indicator_list if col in df.columns]
        
        if len(available_columns) < len(indicator_list):
            missing_columns = set(indicator_list) - set(available_columns)
            logger.warning(f"일부 요청된 지표가 데이터프레임에 없습니다: {missing_columns}")
        
        # 필터링된 데이터프레임 반환
        return df[available_columns].copy()
    
    except Exception as e:
        logger.error(f"지표 필터링 중 오류 발생: {str(e)}")
        # 기본 OHLCV 컬럼만 반환 (최소한의 안전망)
        basic_cols = ['open', 'high', 'low', 'close', 'volume']
        available_basic = [col for col in basic_cols if col in df.columns]
        return df[available_basic].copy()

@log_execution
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    상대강도지수(RSI)를 계산합니다.
    
    Args:
        prices (pd.Series): 가격 시리즈
        period (int): RSI 계산 기간
        
    Returns:
        pd.Series: RSI 값 시리즈
    """
    try:
        # RSI가 이미 계산되어 있는지 확인
        if isinstance(prices, pd.DataFrame) and 'rsi' in prices.columns:
            logger.debug("이미 계산된 RSI 값 반환")
            return prices['rsi']
        
        # 입력이 DataFrame인 경우 'close' 컬럼 사용
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close']
            else:
                logger.error("DataFrame에 'close' 컬럼이 없습니다.")
                return pd.Series(np.nan, index=prices.index)
        
        # 입력이 Series인지 확인
        if not isinstance(prices, pd.Series):
            logger.error(f"올바르지 않은 입력 유형: {type(prices)}")
            return pd.Series(np.nan, index=range(len(prices) if hasattr(prices, '__len__') else 0))
        
        # 숫자형으로 변환
        prices = pd.to_numeric(prices, errors='coerce')
        
        # 충분한 데이터가 있는지 확인
        if len(prices) < period + 1:
            logger.warning(f"RSI 계산을 위한 데이터가 부족합니다. 필요: {period + 1}, 실제: {len(prices)}")
            return pd.Series(50.0, index=prices.index)  # 기본값 50.0 반환
        
        # 가격 변화 계산
        delta = prices.diff()
        
        # 상승/하락 구분
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 첫번째 평균 계산
        avg_gain = gain.rolling(window=period, min_periods=1).mean().iloc[period-1]
        avg_loss = loss.rolling(window=period, min_periods=1).mean().iloc[period-1]
        
        # RSI 시리즈 초기화
        rsi = pd.Series(np.nan, index=prices.index)
        
        # 첫 번째 RSI 값 계산
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi.iloc[period] = 100 - (100 / (1 + rs))
        else:
            rsi.iloc[period] = 100
        
        # 나머지 RSI 값 계산
        for i in range(period + 1, len(prices)):
            avg_gain = ((avg_gain * (period - 1)) + gain.iloc[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + loss.iloc[i]) / period
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi.iloc[i] = 100 - (100 / (1 + rs))
            else:
                rsi.iloc[i] = 100
        
        # 결측값 처리
        rsi = rsi.fillna(50)  # NaN 값을 50으로 대체 (중립적인 RSI 값)
        
        return rsi
    except Exception as e:
        logger.error(f"RSI 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(50.0, index=prices.index)  # 오류 발생 시 50.0 반환

@log_execution
def calculate_stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    스토캐스틱 오실레이터의 %K 라인을 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLC 데이터가 포함된 데이터프레임
        period (int): 계산 기간
        
    Returns:
        pd.Series: 스토캐스틱 %K 값 시리즈
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.Series(np.nan, index=range(len(df) if hasattr(df, '__len__') else 0))
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"스토캐스틱 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.Series(np.nan, index=df.index)
        
        # 데이터 변환
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        
        # 이동 최고값/최저값 계산
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()
        
        # %K 계산
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # 0으로 나누는 경우 처리
        k = k.replace([np.inf, -np.inf], np.nan)
        
        # NaN 값 처리
        k = k.fillna(50)  # NaN 값을 50으로 대체 (중립적인 값)
        
        return k
    except Exception as e:
        logger.error(f"스토캐스틱 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(50, index=df.index)  # 오류 발생 시 50 반환

@log_execution
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    평균 실제 범위(ATR)를 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLC 데이터가 포함된 데이터프레임
        period (int): 계산 기간
        
    Returns:
        pd.Series: ATR 값 시리즈
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.Series(np.nan, index=range(len(df) if hasattr(df, '__len__') else 0))
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"ATR 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.Series(np.nan, index=df.index)
        
        # 데이터 변환
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        close_prev = close.shift(1)
        
        # 실제 범위(TR) 계산
        tr1 = high - low  # 당일 고가 - 당일 저가
        tr2 = np.abs(high - close_prev)  # 당일 고가 - 전일 종가
        tr3 = np.abs(low - close_prev)  # 당일 저가 - 전일 종가
        
        tr = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        }).max(axis=1)
        
        # ATR 계산 (Wilder의 평활화 방법)
        atr = np.nan * np.ones(len(df))
        atr[period-1] = tr.iloc[:period].mean()  # 첫 번째 ATR 값
        
        for i in range(period, len(df)):
            atr[i] = ((atr[i-1] * (period - 1)) + tr.iloc[i]) / period
            
        atr_series = pd.Series(atr, index=df.index)
        
        # NaN 값 처리
        atr_series = atr_series.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return atr_series
    except Exception as e:
        logger.error(f"ATR 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(0, index=df.index)  # 오류 발생 시 0 반환

@log_execution
def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    온발런스 볼륨(OBV)을 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
        
    Returns:
        pd.Series: OBV 값 시리즈
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.Series(np.nan, index=range(len(df) if hasattr(df, '__len__') else 0))
        
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"OBV 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.Series(np.nan, index=df.index)
        
        # 데이터 변환
        close = pd.to_numeric(df['close'], errors='coerce')
        volume = pd.to_numeric(df['volume'], errors='coerce')
        
        # 가격 변화 방향 계산
        price_change = close.diff()
        
        # OBV 초기화
        obv = pd.Series(0, index=df.index)
        
        # 첫 번째 값은 0으로 설정
        obv.iloc[0] = 0
        
        # OBV 계산
        for i in range(1, len(df)):
            if np.isnan(price_change.iloc[i]):
                obv.iloc[i] = obv.iloc[i-1]
            elif price_change.iloc[i] > 0:  # 가격 상승
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:  # 가격 하락
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:  # 가격 동일
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    except Exception as e:
        logger.error(f"OBV 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(0, index=df.index)  # 오류 발생 시 0 반환

@log_execution
def calculate_parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """
    파라볼릭 SAR(Stop And Reverse)을 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLC 데이터가 포함된 데이터프레임
        af_start (float): 가속 팩터 시작값
        af_max (float): 가속 팩터 최대값
        
    Returns:
        pd.Series: 파라볼릭 SAR 값 시리즈
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.Series(np.nan, index=range(len(df) if hasattr(df, '__len__') else 0))
        
        required_cols = ['high', 'low']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"파라볼릭 SAR 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.Series(np.nan, index=df.index)
        
        # 데이터 변환
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        
        # 최소 2개의 데이터 포인트 필요
        if len(df) < 2:
            logger.warning("파라볼릭 SAR 계산을 위한 데이터가 부족합니다.")
            return pd.Series(np.nan, index=df.index)
        
        # SAR 초기화
        sar = np.zeros(len(df))
        trend = np.zeros(len(df))  # 1: 상승 추세, -1: 하락 추세
        extreme_point = np.zeros(len(df))
        af = np.zeros(len(df))
        
        # 초기 추세 설정 (처음 2개 바 기준)
        if high.iloc[1] > high.iloc[0]:
            trend[1] = 1  # 상승 추세
            sar[1] = low.iloc[0]  # SAR은 전일 저가
            extreme_point[1] = high.iloc[1]  # EP는 현재 고가
        else:
            trend[1] = -1  # 하락 추세
            sar[1] = high.iloc[0]  # SAR은 전일 고가
            extreme_point[1] = low.iloc[1]  # EP는 현재 저가
            
        af[1] = af_start  # 초기 가속 팩터
        
        # 나머지 SAR 계산
        for i in range(2, len(df)):
            # 이전 추세 계속
            if trend[i-1] == 1:  # 상승 추세
                # SAR 계산
                sar[i] = sar[i-1] + af[i-1] * (extreme_point[i-1] - sar[i-1])
                
                # SAR 제한 (전일, 전전일 저가보다 높으면 안 됨)
                sar[i] = min(sar[i], low.iloc[i-2], low.iloc[i-1])
                
                # 추세 반전 확인
                if low.iloc[i] < sar[i]:  # 현재 저가가 SAR 아래로 내려가면 추세 반전
                    trend[i] = -1  # 하락 추세로 전환
                    sar[i] = extreme_point[i-1]  # SAR은 이전 EP로 설정
                    extreme_point[i] = low.iloc[i]  # EP는 현재 저가
                    af[i] = af_start  # 가속 팩터 초기화
                else:
                    trend[i] = 1  # 상승 추세 유지
                    # 새로운 최고가 갱신 확인
                    if high.iloc[i] > extreme_point[i-1]:
                        extreme_point[i] = high.iloc[i]  # EP 갱신
                        af[i] = min(af[i-1] + af_start, af_max)  # AF 증가
                    else:
                        extreme_point[i] = extreme_point[i-1]  # EP 유지
                        af[i] = af[i-1]  # AF 유지
            else:  # 하락 추세
                # SAR 계산
                sar[i] = sar[i-1] + af[i-1] * (extreme_point[i-1] - sar[i-1])
                
                # SAR 제한 (전일, 전전일 고가보다 낮으면 안 됨)
                sar[i] = max(sar[i], high.iloc[i-2], high.iloc[i-1])
                
                # 추세 반전 확인
                if high.iloc[i] > sar[i]:  # 현재 고가가 SAR 위로 올라가면 추세 반전
                    trend[i] = 1  # 상승 추세로 전환
                    sar[i] = extreme_point[i-1]  # SAR은 이전 EP로 설정
                    extreme_point[i] = high.iloc[i]  # EP는 현재 고가
                    af[i] = af_start  # 가속 팩터 초기화
                else:
                    trend[i] = -1  # 하락 추세 유지
                    # 새로운 최저가 갱신 확인
                    if low.iloc[i] < extreme_point[i-1]:
                        extreme_point[i] = low.iloc[i]  # EP 갱신
                        af[i] = min(af[i-1] + af_start, af_max)  # AF 증가
                    else:
                        extreme_point[i] = extreme_point[i-1]  # EP 유지
                        af[i] = af[i-1]  # AF 유지
        
        # SAR 시리즈 반환
        sar_series = pd.Series(sar, index=df.index)
        
        # 초기 값 NaN으로 설정 (계산을 위한 충분한 데이터가 없어서)
        sar_series.iloc[0] = np.nan
        
        # NaN 값 처리
        sar_series = sar_series.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return sar_series
    except Exception as e:
        logger.error(f"파라볼릭 SAR 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(0, index=df.index)  # 오류 발생 시 0 반환

@log_execution
def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    평균 방향성 지수(ADX)를 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLC 데이터가 포함된 데이터프레임
        period (int): 계산 기간
        
    Returns:
        pd.Series: ADX 값 시리즈
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.Series(np.nan, index=range(len(df) if hasattr(df, '__len__') else 0))
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"ADX 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.Series(np.nan, index=df.index)
        
        # 데이터 변환
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        
        # 최소한 계산 기간 + 1일 데이터 필요
        if len(df) <= period + 1:
            logger.warning(f"ADX 계산을 위한 데이터가 부족합니다. 필요: {period + 1}, 실제: {len(df)}")
            return pd.Series(np.nan, index=df.index)
        
        # True Range 계산
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Directional Movement 계산
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        # Positive Directional Movement (+DM)
        pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        
        # Negative Directional Movement (-DM)
        ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM (Wilder's smoothing)
        tr_period = pd.Series(tr).rolling(window=period).sum()
        pdm_period = pd.Series(pdm).rolling(window=period).sum()
        ndm_period = pd.Series(ndm).rolling(window=period).sum()
        
        # Smooth TR, +DM, -DM for remaining periods
        for i in range(period + 1, len(df)):
            tr_period.iloc[i] = tr_period.iloc[i-1] - (tr_period.iloc[i-1] / period) + tr.iloc[i]
            pdm_period.iloc[i] = pdm_period.iloc[i-1] - (pdm_period.iloc[i-1] / period) + pdm[i]
            ndm_period.iloc[i] = ndm_period.iloc[i-1] - (ndm_period.iloc[i-1] / period) + ndm[i]
        
        # +DI, -DI 계산
        pdi = 100 * pdm_period / tr_period
        ndi = 100 * ndm_period / tr_period
        
        # Directional Index (DX)
        dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
        
        # Average Directional Index (ADX)
        adx = pd.Series(np.nan, index=df.index)
        
        # 첫 번째 ADX 값 (초기 기간의 DX 평균)
        adx.iloc[2 * period - 1] = dx.iloc[period:2 * period].mean()
        
        # 나머지 ADX 값 (Wilder의 평활화)
        for i in range(2 * period, len(df)):
            adx.iloc[i] = (adx.iloc[i-1] * (period - 1) + dx.iloc[i]) / period
        
        # NaN 값 처리
        adx = adx.fillna(method='bfill').fillna(0)
        
        return adx
    except Exception as e:
        logger.error(f"ADX 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(0, index=df.index)  # 오류 발생 시 0 반환

@log_execution
def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    일목균형표(Ichimoku Cloud) 지표를 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLC 데이터가 포함된 데이터프레임
        
    Returns:
        pd.DataFrame: 일목균형표 컴포넌트가 포함된 데이터프레임
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.DataFrame()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"일목균형표 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.DataFrame()
        
        # 데이터 변환
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        
        # 일목균형표 계산
        # 전환선(Tenkan-sen): 9일 고가와 저가의 중간값
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # 기준선(Kijun-sen): 26일 고가와 저가의 중간값
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # 선행스팬A(Senkou Span A): (전환선 + 기준선) / 2, 26일 선행
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # 선행스팬B(Senkou Span B): 52일 고가와 저가의 중간값, 26일 선행
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # 후행스팬(Chikou Span): 종가를 26일 후행
        chikou_span = close.shift(-26)
        
        # 결과 데이터프레임 생성
        ichimoku_df = pd.DataFrame({
            'ichimoku_tenkan_sen': tenkan_sen,
            'ichimoku_kijun_sen': kijun_sen,
            'ichimoku_senkou_span_a': senkou_span_a,
            'ichimoku_senkou_span_b': senkou_span_b,
            'ichimoku_chikou_span': chikou_span
        }, index=df.index)
        
        # NaN 값 처리
        ichimoku_df = ichimoku_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return ichimoku_df
    except Exception as e:
        logger.error(f"일목균형표 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()  # 오류 발생 시 빈 DataFrame 반환

@log_execution
def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    상품채널지수(CCI)를 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLC 데이터가 포함된 데이터프레임
        period (int): 계산 기간
        
    Returns:
        pd.Series: CCI 값 시리즈
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.Series(np.nan, index=range(len(df) if hasattr(df, '__len__') else 0))
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"CCI 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.Series(np.nan, index=df.index)
        
        # 데이터 변환
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        
        # 일반적인 가격(TP) 계산
        tp = (high + low + close) / 3
        
        # 이동 평균(SMA) 계산
        tp_sma = tp.rolling(window=period, min_periods=1).mean()
        
        # 평균 편차(MD) 계산
        md = tp.rolling(window=period, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        # CCI 계산
        cci = (tp - tp_sma) / (0.015 * md)
        
        # NaN 값 처리
        cci = cci.fillna(0)
        
        return cci
    except Exception as e:
        logger.error(f"CCI 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(0, index=df.index)  # 오류 발생 시 0 반환

@log_execution
def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    자금흐름지수(MFI)를 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
        period (int): 계산 기간
        
    Returns:
        pd.Series: MFI 값 시리즈
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.Series(np.nan, index=range(len(df) if hasattr(df, '__len__') else 0))
        
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"MFI 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.Series(np.nan, index=df.index)
        
        # 데이터 변환
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        volume = pd.to_numeric(df['volume'], errors='coerce')
        
        # 일반적인 가격(TP) 계산
        tp = (high + low + close) / 3
        
        # 자금 흐름(MF) 계산
        mf = tp * volume
        
        # 양수 자금 흐름(PMF)과 음수 자금 흐름(NMF) 계산
        pmf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=df.index)
        nmf = pd.Series(np.where(tp < tp.shift(1), mf, 0), index=df.index)
        
        # PMF와 NMF의 이동 합계 계산
        pmf_sum = pmf.rolling(window=period, min_periods=1).sum()
        nmf_sum = nmf.rolling(window=period, min_periods=1).sum()
        
        # 자금 비율(MR) 계산 (0으로 나누는 것 방지)
        mr = np.where(nmf_sum != 0, pmf_sum / nmf_sum, 1)
        
        # MFI 계산
        mfi = 100 - (100 / (1 + mr))
        
        # NaN 값 처리
        mfi_series = pd.Series(mfi, index=df.index)
        mfi_series = mfi_series.fillna(method='ffill').fillna(method='bfill').fillna(50)
        
        return mfi_series
    except Exception as e:
        logger.error(f"MFI 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(50, index=df.index)  # 오류 발생 시 50 반환

@log_execution
def calculate_chaikin_oscillator(df: pd.DataFrame, fast_period: int = 3, slow_period: int = 10) -> pd.Series:
    """
    차이킨 오실레이터(Chaikin Oscillator)를 계산합니다.
    
    Args:
        df (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
        fast_period (int): 빠른 EMA 기간
        slow_period (int): 느린 EMA 기간
        
    Returns:
        pd.Series: 차이킨 오실레이터 값 시리즈
    """
    try:
        # 입력 유효성 검사
        if not isinstance(df, pd.DataFrame):
            logger.error(f"입력이 DataFrame이 아닙니다: {type(df)}")
            return pd.Series(np.nan, index=range(len(df) if hasattr(df, '__len__') else 0))
        
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"차이킨 오실레이터 계산에 필요한 컬럼이 없습니다. 필요: {required_cols}")
            return pd.Series(np.nan, index=df.index)
        
        # 데이터 변환
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        volume = pd.to_numeric(df['volume'], errors='coerce')
        
        # A/D Line 계산
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.replace([np.inf, -np.inf], 0)  # 0으로 나누는 경우 처리
        mfm = mfm.fillna(0)  # NaN 값 처리
        
        mfv = mfm * volume
        ad_line = mfv.cumsum()
        
        # 빠른 EMA와 느린 EMA 계산
        ad_ema_fast = ad_line.ewm(span=fast_period, adjust=False).mean()
        ad_ema_slow = ad_line.ewm(span=slow_period, adjust=False).mean()
        
        # 차이킨 오실레이터 계산
        chaikin_osc = ad_ema_fast - ad_ema_slow
        
        # NaN 값 처리
        chaikin_osc = chaikin_osc.fillna(0)
        
        return chaikin_osc
    except Exception as e:
        logger.error(f"차이킨 오실레이터 계산 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(0, index=df.index)  # 오류 발생 시 0 반환
