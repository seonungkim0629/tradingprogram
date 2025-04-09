"""
데이터 수집 스크립트
업비트에서 비트코인 데이터를 수집하여 저장합니다.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import time
from data.collectors import upbit_collector
from data.indicators import calculate_all_indicators
from data.storage import save_ohlcv_data, save_processed_data
from utils.logging import get_logger
import traceback

# 로거 초기화
logger = get_logger(__name__)

def collect_and_save_data(ticker: str = "KRW-BTC", days: int = 1000):
    """
    데이터 수집 및 저장
    
    Args:
        ticker (str): 티커 심볼
        days (int): 수집할 일수
    """
    try:
        # 데이터 디렉토리 생성
        if not os.path.exists('data/raw'):
            os.makedirs('data/raw')
            
        # 일별 데이터 수집
        logger.info(f"{ticker} 일별 데이터 수집 시작")
        daily_data = upbit_collector.get_daily_ohlcv(ticker=ticker)
        
        if daily_data is not None and not daily_data.empty:
            # 데이터 저장
            save_path = f'data/raw/{ticker}_daily_{datetime.now().strftime("%Y%m%d")}.csv'
            daily_data.to_csv(save_path)
            logger.info(f"일별 데이터 저장 완료: {save_path}")
            
            # 시간별 데이터 수집
            logger.info(f"{ticker} 시간별 데이터 수집 시작")
            hourly_data = upbit_collector.get_hourly_ohlcv(ticker=ticker)
            
            if hourly_data is not None and not hourly_data.empty:
                # 데이터 저장
                save_path = f'data/raw/{ticker}_hourly_{datetime.now().strftime("%Y%m%d")}.csv'
                hourly_data.to_csv(save_path)
                logger.info(f"시간별 데이터 저장 완료: {save_path}")
            else:
                logger.warning(f"{ticker} 시간별 데이터 수집 실패")
        else:
            logger.warning(f"{ticker} 일별 데이터 수집 실패")
            
    except Exception as e:
        logger.error(f"데이터 수집 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())

def collect_short_term_data(ticker: str = "KRW-BTC"):
    """
    단기 트레이딩을 위한 1분봉, 5분봉, 시간봉 데이터를 수집하여 저장합니다.
    하루 6회 매매를 위한 적정량의 데이터를 수집합니다.
    
    Args:
        ticker (str): 티커 심볼
    
    Returns:
        dict: 각 시간대별 데이터프레임을 담은 사전
    """
    try:
        result = {}
        
        # 1분봉 데이터 수집 (최대 500개 = 약 8시간)
        logger.info(f"{ticker} 1분봉 데이터 수집 시작")
        minute1_data = upbit_collector.get_minute1_ohlcv(ticker=ticker, count=500)
        
        if minute1_data is not None and not minute1_data.empty:
            result['minute1'] = minute1_data
            # 1분봉 데이터 저장
            save_path = save_ohlcv_data(minute1_data, ticker, 'minute1')
            logger.info(f"1분봉 데이터가 {save_path}에 저장되었습니다 (총 {len(minute1_data)}개)")
            
            # 지표 추가
            if len(minute1_data) >= 100:
                minute1_data_with_indicators = calculate_all_indicators(minute1_data, timeframe='minute1')
                save_path = save_processed_data(minute1_data_with_indicators, ticker, 'indicators', 'minute1')
                logger.info(f"지표가 추가된 1분봉 데이터가 {save_path}에 저장되었습니다")
        else:
            logger.warning(f"{ticker} 1분봉 데이터 수집 실패")
        
        # API 호출 간 딜레이
        time.sleep(0.5)
        
        # 5분봉 데이터 수집 (최대 1000개 = 약 3.5일)
        logger.info(f"{ticker} 5분봉 데이터 수집 시작")
        minute5_data = upbit_collector.get_minute5_ohlcv(ticker=ticker, count=1000)
        
        if minute5_data is not None and not minute5_data.empty:
            result['minute5'] = minute5_data
            # 5분봉 데이터 저장
            save_path = save_ohlcv_data(minute5_data, ticker, 'minute5')
            logger.info(f"5분봉 데이터가 {save_path}에 저장되었습니다 (총 {len(minute5_data)}개)")
            
            # 지표 추가
            if len(minute5_data) >= 100:
                minute5_data_with_indicators = calculate_all_indicators(minute5_data, timeframe='minute5')
                save_path = save_processed_data(minute5_data_with_indicators, ticker, 'indicators', 'minute5')
                logger.info(f"지표가 추가된 5분봉 데이터가 {save_path}에 저장되었습니다")
        else:
            logger.warning(f"{ticker} 5분봉 데이터 수집 실패")
        
        # API 호출 간 딜레이
        time.sleep(0.5)
        
        # 시간봉 데이터 수집 (최대 500개 = 약 20일)
        logger.info(f"{ticker} 시간봉 데이터 수집 시작")
        hourly_data = upbit_collector.get_hourly_ohlcv(ticker=ticker, count=500)
        
        if hourly_data is not None and not hourly_data.empty:
            result['hourly'] = hourly_data
            # 시간봉 데이터 저장
            save_path = save_ohlcv_data(hourly_data, ticker, 'hour')
            logger.info(f"시간봉 데이터가 {save_path}에 저장되었습니다 (총 {len(hourly_data)}개)")
            
            # 지표 추가
            if len(hourly_data) >= 100:
                hourly_data_with_indicators = calculate_all_indicators(hourly_data, timeframe='hour')
                save_path = save_processed_data(hourly_data_with_indicators, ticker, 'indicators', 'hour')
                logger.info(f"지표가 추가된 시간봉 데이터가 {save_path}에 저장되었습니다")
        else:
            logger.warning(f"{ticker} 시간봉 데이터 수집 실패")
            
        # 통합 데이터세트 생성 및 저장
        if all(k in result for k in ['minute1', 'minute5', 'hourly']):
            logger.info("모든 시간대 데이터가 수집되었습니다. 통합 데이터세트를 생성합니다.")
            create_combined_dataset(result, ticker)
        
        return result
        
    except Exception as e:
        logger.error(f"단기 데이터 수집 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def create_combined_dataset(data_dict, ticker="KRW-BTC"):
    """
    수집된 여러 시간대의 데이터를 하나의 통합 데이터셋으로 생성하여 저장합니다.
    
    Args:
        data_dict (dict): 각 시간대별 데이터프레임을 담은 사전
        ticker (str): 티커 심볼
    """
    try:
        # 저장 디렉토리 생성
        combined_dir = 'data_storage/combined'
        os.makedirs(combined_dir, exist_ok=True)
        
        # 현재 타임스탬프
        timestamp = datetime.now().strftime('%Y%m%d')
        
        # 데이터셋 조합
        combined_data = {
            'minute1': data_dict.get('minute1'),
            'minute5': data_dict.get('minute5'),
            'hourly': data_dict.get('hourly')
        }
        
        # 저장 경로
        save_path = f"{combined_dir}/{ticker}_combined_{timestamp}.pkl"
        
        # pickle로 저장
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(combined_data, f)
        
        logger.info(f"통합 데이터세트가 {save_path}에 저장되었습니다.")
        
        # CSV 파일도 저장 (필요시 참조용)
        for timeframe, df in combined_data.items():
            if df is not None and not df.empty:
                csv_path = f"{combined_dir}/{ticker}_{timeframe}_{timestamp}.csv"
                df.to_csv(csv_path)
                logger.info(f"{timeframe} 데이터가 {csv_path}에 CSV 형식으로 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"통합 데이터세트 생성 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """메인 함수"""
    try:
        # 비트코인 데이터 수집
        collect_and_save_data("KRW-BTC")
        
    except Exception as e:
        logger.error(f"메인 프로세스 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 