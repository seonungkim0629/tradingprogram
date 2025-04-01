"""
데이터 수집 스크립트
업비트에서 비트코인 데이터를 수집하여 저장합니다.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from data.collectors import upbit_collector
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