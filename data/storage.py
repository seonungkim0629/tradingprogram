"""
Data storage module for Bitcoin Trading Bot

This module provides functionality to save and load market data to/from disk.
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np

from config import settings

from utils.logging import get_logger, log_execution


# Initialize logger
logger = get_logger(__name__)

# 경로 상수를 settings 모듈에서 가져옴 - 중복 코드 제거
OHLCV_DIR = settings.OHLCV_DIR
INDICATORS_DIR = settings.INDICATORS_DIR
RESULTS_DIR = settings.RESULTS_DIR
DATA_DIR = settings.DATA_DIR
MODELS_DIR = settings.MODELS_DIR
FEATURES_DIR = settings.FEATURES_DIR
COMBINED_DIR = settings.COMBINED_DIR
STACKED_DIR = settings.STACKED_DIR
RAW_DIR = settings.RAW_DIR


@log_execution
def save_dataframe_to_csv(df: pd.DataFrame, filename: str, directory: str = OHLCV_DIR) -> str:
    """
    Save a DataFrame to CSV file
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Filename without extension
        directory (str, optional): Directory to save to. Defaults to OHLCV_DIR.
        
    Returns:
        str: Path to saved file
    """
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Add .csv extension if not already present
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Create full path
    filepath = os.path.join(directory, filename)
    
    try:
        df.to_csv(filepath)
        logger.info(f"Saved DataFrame to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving DataFrame to {filepath}: {str(e)}")
        return ""


@log_execution
def load_dataframe_from_csv(filename: str, directory: str = OHLCV_DIR, parse_dates: bool = True) -> Optional[pd.DataFrame]:
    """
    Load a DataFrame from CSV file
    
    Args:
        filename (str): Filename without extension
        directory (str, optional): Directory to load from. Defaults to OHLCV_DIR.
        parse_dates (bool, optional): Whether to parse dates from the index. Defaults to True.
        
    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame or None if error
    """
    # Add .csv extension if not already present
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Create full path
    filepath = os.path.join(directory, filename)
    
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
        
        # Load the data
        if parse_dates:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            df = pd.read_csv(filepath, index_col=0)
        
        logger.info(f"Loaded DataFrame from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {filepath}: {str(e)}")
        return None


@log_execution
def save_dataframe_to_pickle(obj: Any, filename: str, directory: str = OHLCV_DIR) -> str:
    """
    Save a DataFrame or any object to pickle file
    
    Args:
        obj (Any): Object to save
        filename (str): Filename without extension
        directory (str, optional): Directory to save to. Defaults to OHLCV_DIR.
        
    Returns:
        str: Path to saved file
    """
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Add .pkl extension if not already present
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    
    # Create full path
    filepath = os.path.join(directory, filename)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Saved object to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving object to {filepath}: {str(e)}")
        return ""


@log_execution
def load_dataframe_from_pickle(filename: str, directory: str = OHLCV_DIR) -> Optional[Any]:
    """
    Load a DataFrame or any object from pickle file
    
    Args:
        filename (str): Filename without extension
        directory (str, optional): Directory to load from. Defaults to OHLCV_DIR.
        
    Returns:
        Optional[Any]: Loaded object or None if error
    """
    # Add .pkl extension if not already present
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    
    # Create full path
    filepath = os.path.join(directory, filename)
    
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
        
        # Load the data
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        
        logger.info(f"Loaded object from {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {filepath}: {str(e)}")
        return None

def stack_hourly_data(ticker: str, new_data: pd.DataFrame, max_records: int = 2000) -> str:
    """
    Stack hourly data to files, creating new files when exceeding max_records
    
    Args:
        ticker (str): Ticker symbol
        new_data (pd.DataFrame): New hourly data to stack
        max_records (int): Maximum records per file
        
    Returns:
        str: Path to the file where data was saved
    """
    # 설정 모듈의 경로 사용
    hourly_dir = os.path.join(STACKED_DIR, ticker)
    os.makedirs(hourly_dir, exist_ok=True)
    
    # List existing hourly files for this ticker
    existing_files = [f for f in os.listdir(hourly_dir) if f.endswith('.csv')]
    existing_files.sort()  # Sort to get the latest file last
    
    if not existing_files:
        # No existing file, create a new one
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_hourly_{timestamp}.csv"
        filepath = os.path.join(hourly_dir, filename)
        new_data.to_csv(filepath)
        logger.info(f"Created new hourly data file: {filepath}")
        return filepath
    
    # Try to append to the latest file
    latest_file = os.path.join(hourly_dir, existing_files[-1])
    existing_data = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    
    # Check if adding new data would exceed max_records
    if len(existing_data) + len(new_data) > max_records:
        # Create a new file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_hourly_{timestamp}.csv"
        filepath = os.path.join(hourly_dir, filename)
        new_data.to_csv(filepath)
        logger.info(f"Created new hourly data file (max records exceeded): {filepath}")
    else:
        # Combine data, remove duplicates by index, and sort
        combined_data = pd.concat([existing_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()
        combined_data.to_csv(latest_file)
        logger.info(f"Appended data to existing file: {latest_file}")
        filepath = latest_file
    
    return filepath

@log_execution
def save_json(data: Dict[str, Any], filename: str, directory: str = RESULTS_DIR) -> str:
    """
    Save data to JSON file
    
    Args:
        data (Dict[str, Any]): Data to save
        filename (str): Filename without extension
        directory (str, optional): Directory to save to. Defaults to RESULTS_DIR.
        
    Returns:
        str: Path to saved file
    """
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Add .json extension if not already present
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Create full path
    filepath = os.path.join(directory, filename)
    
    try:
        # Convert NaN values to null before serializing to JSON
        def convert_nan(obj):
            if isinstance(obj, dict):
                return {key: convert_nan(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_nan(item) for item in obj]
            elif isinstance(obj, (np.float, float)) and np.isnan(obj):
                return None
            elif isinstance(obj, np.ndarray):
                return convert_nan(obj.tolist())
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        # Apply conversion
        converted_data = convert_nan(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved JSON data to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving JSON data to {filepath}: {str(e)}")
        return ""


@log_execution
def load_json(filename: str, directory: str = RESULTS_DIR) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file
    
    Args:
        filename (str): Filename without extension
        directory (str, optional): Directory to load from. Defaults to RESULTS_DIR.
        
    Returns:
        Optional[Dict[str, Any]]: Loaded data or None if error
    """
    # Add .json extension if not already present
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Create full path
    filepath = os.path.join(directory, filename)
    
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
        
        # Load the data
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON data from {filepath}: {str(e)}")
        return None


@log_execution
def save_ohlcv_data(df: pd.DataFrame, ticker: str, timeframe: str) -> str:
    """
    Save OHLCV data to CSV file with standardized naming
    
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data
        ticker (str): Ticker symbol
        timeframe (str): Timeframe (e.g., 'day', 'hour', 'minute60')
        
    Returns:
        str: Path to saved file
    """
    # Create standardized filename
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{ticker}_{timeframe}_{timestamp}.csv"
    
    return save_dataframe_to_csv(df, filename, OHLCV_DIR)


@log_execution
def load_latest_ohlcv_data(ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Load the latest OHLCV data for a specific ticker and timeframe
    
    Args:
        ticker (str): Ticker symbol
        timeframe (str): Timeframe (e.g., 'day', 'hour', 'minute60')
        
    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame or None if no data found
    """
    # Get all files matching the pattern
    pattern = f"{ticker}_{timeframe}_"
    matching_files = [f for f in os.listdir(OHLCV_DIR) if f.startswith(pattern) and f.endswith('.csv')]
    
    if not matching_files:
        logger.warning(f"No OHLCV data found for {ticker} ({timeframe})")
        return None
    
    # Sort by timestamp to get the latest
    latest_file = sorted(matching_files)[-1]
    
    return load_dataframe_from_csv(latest_file, OHLCV_DIR)


@log_execution
def save_processed_data(df: pd.DataFrame, ticker: str, data_type: str, timeframe: str) -> str:
    """
    Save processed data (indicators, features, predictions) to CSV file
    
    Args:
        df (pd.DataFrame): DataFrame containing processed data
        ticker (str): Ticker symbol
        data_type (str): Type of data (e.g., 'indicators', 'features', 'predictions')
        timeframe (str): Timeframe (e.g., 'day', 'hour')
        
    Returns:
        str: Path to saved file
    """
    # 데이터 유형에 따른 디렉토리 결정
    if data_type == 'indicators':
        directory = INDICATORS_DIR
    elif data_type == 'features':
        directory = FEATURES_DIR
    else:
        directory = os.path.join(DATA_DIR, data_type)
    
    # Create directory if needed
    os.makedirs(directory, exist_ok=True)
    
    # Create standardized filename
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{ticker}_{timeframe}_{data_type}_{timestamp}.csv"
    
    return save_dataframe_to_csv(df, filename, directory)


@log_execution
def save_model(model: Any, model_name: str, model_type: str, ticker: str = None) -> str:
    """
    Save a trained model
    
    Args:
        model (Any): Model object to save
        model_name (str): Name of the model
        model_type (str): Type of model (e.g., 'lstm', 'random_forest', 'reinforcement')
        ticker (str, optional): Ticker symbol. Defaults to None.
        
    Returns:
        str: Path to saved file
    """
    # 모델 디렉토리 사용
    model_dir = settings.MODELS_DIR
    
    # Create filename with metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ticker_str = f"{ticker}_" if ticker else ""
    filename = f"{ticker_str}{model_type}_{model_name}_{timestamp}.pkl"
    
    return save_dataframe_to_pickle(model, filename, model_dir)


@log_execution
def load_model(model_path: str) -> Optional[Any]:
    """
    Load a trained model
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Optional[Any]: Loaded model or None if error
    """
    try:
        if not os.path.exists(model_path):
            # Check if it's a relative path within the models directory
            absolute_path = os.path.join(settings.MODELS_DIR, model_path)
            if not os.path.exists(absolute_path):
                logger.warning(f"Model file not found: {model_path}")
                return None
            model_path = absolute_path
            
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None


@log_execution
def save_combined_dataset(combined_data: Dict[str, pd.DataFrame], ticker: str) -> str:
    """
    여러 타임프레임의 데이터를 통합하여 저장
    
    Args:
        combined_data (Dict[str, pd.DataFrame]): 타임프레임별 데이터프레임 딕셔너리
        ticker (str): 티커 심볼
        
    Returns:
        str: 저장된 파일 경로
    """
    # 통합 데이터 디렉토리 사용
    combined_dir = COMBINED_DIR
    os.makedirs(combined_dir, exist_ok=True)
    
    # 파일명 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"{ticker}_combined_{timestamp}.pkl"
    filepath = os.path.join(combined_dir, filename)
    
    try:
        # 파일 저장
        with open(filepath, 'wb') as f:
            pickle.dump(combined_data, f)
            
        # 오래된 파일 정리 (최신 5개만 유지)
        existing_files = [f for f in os.listdir(combined_dir) 
                          if f.startswith(f"{ticker}_combined_") and f.endswith('.pkl')]
        if len(existing_files) > 5:
            existing_files.sort()
            for old_file in existing_files[:-5]:
                try:
                    os.remove(os.path.join(combined_dir, old_file))
                    logger.debug(f"Removed old combined dataset: {old_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove old file {old_file}: {str(e)}")
                    
        logger.info(f"Saved combined dataset to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving combined dataset: {str(e)}")
        return ""


@log_execution
def load_latest_combined_dataset(ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    최신 통합 데이터셋 로드
    
    Args:
        ticker (str): 티커 심볼
        
    Returns:
        Optional[Dict[str, pd.DataFrame]]: 타임프레임별 데이터프레임 딕셔너리 또는 None
    """
    combined_dir = COMBINED_DIR
    if not os.path.exists(combined_dir):
        logger.warning(f"Combined data directory does not exist: {combined_dir}")
        return None
        
    # 해당 티커의 통합 데이터 파일 찾기
    pattern = f"{ticker}_combined_"
    matching_files = [f for f in os.listdir(combined_dir) 
                      if f.startswith(pattern) and f.endswith('.pkl')]
    
    if not matching_files:
        logger.warning(f"No combined dataset found for {ticker}")
        return None
        
    # 날짜별 정렬하여 최신 파일 가져오기
    matching_files.sort(reverse=True)
    latest_file = os.path.join(combined_dir, matching_files[0])
    
    try:
        with open(latest_file, 'rb') as f:
            combined_data = pickle.load(f)
            
        logger.info(f"Loaded latest combined dataset from {latest_file}")
        return combined_data
    except Exception as e:
        logger.error(f"Error loading combined dataset from {latest_file}: {str(e)}")
        return None


@log_execution
def save_backtest_results(results: Dict[str, Any], strategy_name: str, ticker: str, timeframe: str) -> str:
    """
    Save backtest results to JSON file
    
    Args:
        results (Dict[str, Any]): Results data
        strategy_name (str): Name of the strategy
        ticker (str): Ticker symbol
        timeframe (str): Timeframe (e.g., 'day', 'hour')
        
    Returns:
        str: Path to saved file
    """
    # Create backtest results directory
    backtest_dir = os.path.join(RESULTS_DIR, 'backtest')
    os.makedirs(backtest_dir, exist_ok=True)
    
    # Create filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"backtest_{strategy_name}_{ticker}_{timeframe}_{timestamp}.json"
    
    return save_json(results, filename, backtest_dir)


@log_execution
def get_available_data_info() -> Dict[str, List[str]]:
    """
    Get information about available data files
    
    Returns:
        Dict[str, List[str]]: Dictionary with directory names as keys and lists of files as values
    """
    result = {}
    
    # OHLCV data
    result['ohlcv'] = os.listdir(OHLCV_DIR) if os.path.exists(OHLCV_DIR) else []
    
    # Indicators data
    result['indicators'] = os.listdir(INDICATORS_DIR) if os.path.exists(INDICATORS_DIR) else []
    
    # Models
    result['models'] = []
    if os.path.exists(MODELS_DIR):
        for model_type in os.listdir(MODELS_DIR):
            model_type_dir = os.path.join(MODELS_DIR, model_type)
            if os.path.isdir(model_type_dir):
                for model_file in os.listdir(model_type_dir):
                    result['models'].append(f"{model_type}/{model_file}")
    
    # Results
    result['results'] = []
    if os.path.exists(RESULTS_DIR):
        for result_type in os.listdir(RESULTS_DIR):
            result_type_dir = os.path.join(RESULTS_DIR, result_type)
            if os.path.isdir(result_type_dir):
                for result_file in os.listdir(result_type_dir):
                    result['results'].append(f"{result_type}/{result_file}")
            elif os.path.isfile(os.path.join(RESULTS_DIR, result_type)):
                result['results'].append(result_type)
    
    return result


@log_execution
def clean_old_data(directory: str, max_files: int = 10, pattern: str = None) -> int:
    """
    Clean old data files, keeping only the most recent ones
    
    Args:
        directory (str): Directory to clean
        max_files (int, optional): Maximum number of files to keep per pattern. Defaults to 10.
        pattern (str, optional): File pattern to match. Defaults to None.
        
    Returns:
        int: Number of files deleted
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return 0
    
    # Get all files matching the pattern
    if pattern:
        matching_files = [f for f in os.listdir(directory) if f.startswith(pattern)]
    else:
        matching_files = os.listdir(directory)
    
    # Group files by pattern if no specific pattern is provided
    if not pattern:
        file_groups = {}
        for file in matching_files:
            # Extract base pattern (usually ticker_timeframe)
            parts = file.split('_')
            if len(parts) >= 2:
                base_pattern = '_'.join(parts[:2])
                if base_pattern not in file_groups:
                    file_groups[base_pattern] = []
                file_groups[base_pattern].append(file)
    else:
        file_groups = {pattern: matching_files}
    
    deleted_count = 0
    
    # Process each group
    for pattern, files in file_groups.items():
        # Sort files by modification time (oldest first)
        sorted_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
        
        # Delete oldest files if we have too many
        files_to_delete = sorted_files[:-max_files] if len(sorted_files) > max_files else []
        
        for file in files_to_delete:
            try:
                os.remove(os.path.join(directory, file))
                logger.debug(f"Deleted old file: {file}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting file {file}: {str(e)}")
    
    if deleted_count > 0:
        logger.info(f"Cleaned {deleted_count} old files from {directory}")
    
    return deleted_count

@log_execution
def save_indicator_data(df: pd.DataFrame, ticker: str, timeframe: str) -> str:
    """
    Save indicator data to a CSV file
    
    Args:
        df (pd.DataFrame): DataFrame with indicators
        ticker (str): Ticker symbol
        timeframe (str): Timeframe (day, hour, etc.)
        
    Returns:
        str: Path to saved file
    """
    try:
        # Create directory if it doesn't exist
        save_dir = f"data_storage/indicators"
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename with date
        today = datetime.now().strftime('%Y%m%d')
        filename = f"{ticker}_{timeframe}_indicators_{today}.csv"
        save_path = os.path.join(save_dir, filename)
        
        # Save DataFrame to CSV
        df.to_csv(save_path)
        logger.info(f"Saved DataFrame to {save_path}")
        
        return save_path
    except Exception as e:
        logger.error(f"Error saving indicator data: {str(e)}")
        return "" 