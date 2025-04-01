"""
Optimizer Module for Bitcoin Trading Bot

This module provides functions to optimize trading strategy parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import optuna
from sklearn.model_selection import RandomizedSearchCV
import traceback
import os
import gc
import json
import tensorflow as tf
# SMOTE를 위한 라이브러리 추가
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imblearn 라이브러리가 설치되지 않았습니다. SMOTE를 사용할 수 없습니다.")

from config import settings
from utils.logging import get_logger, log_execution
from backtest.engine import BacktestEngine
from strategies.base import BaseStrategy
import strategies
from utils.evaluation import format_backtest_results
from data.storage import save_backtest_results
from models.gru import GRUPriceModel, GRUDirectionModel
from models.random_forest import RandomForestDirectionModel
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Initialize logger
logger = get_logger(__name__)


@log_execution
def run_optimization(
    strategy: str,
    market: str,
    start_date: str,
    end_date: str,
    parameter_grid: Optional[Dict[str, List[Any]]] = None,
    metric: str = 'sharpe_ratio',
    initial_balance: float = settings.TRADING_AMOUNT,
    commission_rate: float = settings.TRADING_FEE,
    data_frequency: str = "minute60",
    slippage: float = 0.0002
) -> Dict[str, Any]:
    """
    Run parameter optimization for the specified strategy and market
    
    Args:
        strategy (str): Strategy name to optimize
        market (str): Market to trade (e.g., KRW-BTC)
        start_date (str): Start date for backtesting (YYYY-MM-DD)
        end_date (str): End date for backtesting (YYYY-MM-DD)
        parameter_grid (Optional[Dict[str, List[Any]]]): Parameter grid to search
        metric (str): Metric to optimize for
        initial_balance (float): Initial account balance
        commission_rate (float): Trading commission rate
        data_frequency (str): Data timeframe to use
        slippage (float): Slippage to apply to trades
        
    Returns:
        Dict[str, Any]: Optimization results
    """
    logger.info(f"Running parameter optimization for {strategy} on {market} from {start_date} to {end_date}")
    
    # Convert string dates to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_balance=initial_balance,
        commission_rate=commission_rate,
        data_frequency=data_frequency,
        slippage=slippage
    )
    
    # Get strategy class
    try:
        # Get the strategy class dynamically
        strategy_class = getattr(strategies, strategy)
        strategy_instance = strategy_class(market=market)
    except (AttributeError, ImportError):
        logger.error(f"Strategy {strategy} not found")
        raise ValueError(f"Strategy {strategy} not found")
    
    # Set default parameter grid if not provided
    if parameter_grid is None:
        # Use strategy's default parameter grid
        parameter_grid = strategy_instance.get_parameter_grid()
        
        # If strategy doesn't provide a grid, use a basic one
        if not parameter_grid:
            logger.warning(f"No parameter grid provided for {strategy}, using defaults")
            parameter_grid = {
                'short_window': [5, 10, 15, 20],
                'long_window': [30, 50, 70, 90]
            }
    
    # Run optimization
    best_params, best_result = engine.optimize(
        strategy=strategy_instance,
        start_date=start_dt,
        end_date=end_dt,
        parameter_grid=parameter_grid,
        metric=metric
    )
    
    # Format results
    formatted_results = format_backtest_results(best_result)
    
    # Add optimization-specific info
    formatted_results['optimization_metric'] = metric
    formatted_results['best_parameters'] = best_params
    formatted_results['tested_parameters'] = parameter_grid
    
    # Save results with a special prefix
    save_path = save_backtest_results(
        results=formatted_results,
        strategy_name=f"optimized_{strategy}",
        ticker=market,
        timeframe=data_frequency
    )
    
    # Log the file path
    logger.info(f"Optimization results saved to {save_path}")
    
    return formatted_results

@log_execution
def optimize_gru_price_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    input_shape: Tuple[int, int],
    n_trials: int = 50,
    timeout: int = 3600 * 3  # 3 hours
) -> Dict[str, Any]:
    """
    GRU 가격 예측 모델의 하이퍼파라미터 최적화
    
    Args:
        X_train (np.ndarray): 훈련 특성, shape (samples, sequence_length, features)
        y_train (np.ndarray): 훈련 타겟, shape (samples,)
        X_val (np.ndarray): 검증 특성, shape (samples, sequence_length, features)
        y_val (np.ndarray): 검증 타겟, shape (samples,)
        input_shape (Tuple[int, int]): 입력 형태 (sequence_length, features)
        n_trials (int): 최적화 시도 횟수
        timeout (int): 최적화 제한 시간(초)
        
    Returns:
        Dict[str, Any]: 최적화 결과 (최적 파라미터, 성능 지표 등)
    """
    logger.info(f"GRU 가격 예측 모델 하이퍼파라미터 최적화 시작 - {n_trials}회 시도, {timeout}초 제한")
    
    # Objective 함수 정의
    def objective(trial):
        # 하이퍼파라미터 샘플링
        units_layer1 = trial.suggest_int('units_layer1', 32, 128, step=8)
        units_layer2 = trial.suggest_int('units_layer2', 16, 64, step=8)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # 모델 생성
        model = GRUPriceModel(
            name="GRUPrice_Optimize",
            version="1.0.0",
            sequence_length=input_shape[0],
            units=[units_layer1, units_layer2],
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=100  # 최대 에폭 수 (조기 종료 적용)
        )
        
        # 메모리 사용량 관리를 위한 가비지 컬렉션
        gc.collect()
        tf.keras.backend.clear_session()
        
        try:
            # 모델 빌드
            model.build_model(input_shape)
            
            # 모델 훈련
            history = model.train(
                X_train, y_train, 
                X_val=X_val, y_val=y_val,
                early_stopping_patience=10,
                reduce_lr_patience=5
            )
            
            # 검증 RMSE 추출
            val_rmse = float(model.metrics.get('val_root_mean_squared_error', float('inf')))
            if np.isnan(val_rmse) or np.isinf(val_rmse):
                val_rmse = float('inf')
                
            # 최적화를 위한 중간 보고
            trial.report(val_rmse, 0)
            
            # 성능이 좋지 않으면 조기 종료
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # 메모리 정리
            tf.keras.backend.clear_session()
            
            return val_rmse
        
        except Exception as e:
            logger.error(f"최적화 시도 중 오류 발생: {str(e)}")
            # 오류 발생 시 큰 값 반환
            return float('inf')
    
    # Optuna 스터디 생성 및 최적화 실행
    try:
        study = optuna.create_study(
            direction='minimize',  # RMSE 최소화
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # 최적 하이퍼파라미터
        best_params = study.best_params
        best_val_rmse = study.best_value
        
        logger.info(f"최적화 완료! 최적 검증 RMSE: {best_val_rmse:.6f}")
        logger.info(f"최적 하이퍼파라미터: {best_params}")
        
        # 최적 모델 재생성 및 훈련
        best_model = GRUPriceModel(
            name="GRUPrice_Best",
            version="1.0.0",
            sequence_length=input_shape[0],
            units=[best_params['units_layer1'], best_params['units_layer2']],
            dropout_rate=best_params['dropout_rate'],
            learning_rate=best_params['learning_rate'],
            batch_size=best_params['batch_size'],
            epochs=100
        )
        
        # 메모리 정리
        gc.collect()
        tf.keras.backend.clear_session()
        
        # 최적 모델 빌드 및 훈련
        best_model.build_model(input_shape)
        
        # 전체 데이터로 학습
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)
        
        best_model.train(
            X_full, y_full,
            early_stopping_patience=15,
            reduce_lr_patience=7
        )
        
        # 모델 저장
        save_path = os.path.join('saved_models', f"GRUPrice_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        best_model.save(save_path)
        
        # H5 파일 형식으로도 저장
        h5_path = os.path.join('saved_models', f"GRUPrice_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
        best_model.save_h5(h5_path)
        
        # 최적화 결과 저장
        optimization_result = {
            'best_params': best_params,
            'best_val_rmse': best_val_rmse,
            'model': best_model
        }
        
        # JSON으로 저장 가능한 형태로 변환
        json_result = {
            'best_params': best_params,
            'best_val_rmse': best_val_rmse,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 결과 파일 저장
        os.makedirs('optimization_results', exist_ok=True)
        with open(os.path.join('optimization_results', 'gru_price_optimization.json'), 'w') as f:
            json.dump(json_result, f, indent=4)
        
        return optimization_result
    
    except Exception as e:
        logger.error(f"최적화 프로세스 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'error': str(e),
            'best_params': {},
            'best_val_rmse': float('inf')
        }

@log_execution
def optimize_lstm_direction_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_shape: Tuple[int, int],
    n_trials: int = 50,
    timeout: Optional[int] = None,
    study_name: str = "lstm_direction_optimization",
    use_smote: bool = True,
) -> Dict[str, Any]:
    """
    Bayesian optimization for GRU direction prediction model using Optuna
    (함수 이름은 호환성을 위해 'lstm'을 유지하지만 실제로는 GRU+LayerNormalization 모델을 최적화합니다)
    
    Args:
        X_train: Training data features
        y_train: Training data targets
        X_val: Validation data features
        y_val: Validation data targets
        input_shape: Model input shape (sequence_length, features)
        n_trials: Number of optimization trials
        timeout: Timeout in seconds (optional)
        study_name: Name for the optimization study
        use_smote: Whether to use SMOTE for class balancing (default: True)
        
    Returns:
        Dict containing best parameters and optimization results
    """
    logger.info(f"Starting Bayesian optimization for GRU+LayerNormalization Direction model with {n_trials} trials")
    
    def objective(trial):
        # Define hyperparameter search space (GRU 모델에 최적화된 범위로 조정)
        params = {
            'units': [
                trial.suggest_int('units_l1', 32, 128),  # 범위 축소: 32-256 → 32-128
                trial.suggest_int('units_l2', 16, 64),   # 범위 축소: 16-128 → 16-64
            ],
            'dropout_rate': trial.suggest_float('dropout_rate', 0.35, 0.5),  # 범위 상향: 0.3-0.5 → 0.35-0.5
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.005, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),  # 128 제거, 작은 배치가 더 효과적
            'sequence_length': trial.suggest_int('sequence_length', 30, 60)  # 범위 축소: 30-90 → 30-60
        }
        
        # Create and train the model (LSTMDirectionModel 클래스는 내부적으로 GRU 구현)
        model = LSTMDirectionModel(
            units=params['units'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            sequence_length=params['sequence_length'],
            epochs=100  # Use a fixed number of epochs with early stopping
        )

        # 외부 스코프에서 데이터 복사 (로컬 변수로 사용하기 위함)
        # 이렇게 하면 외부에서 y_train과 y_val이 변경되어도 안전하게 참조 가능
        local_X_train = X_train.copy() 
        local_y_train = y_train.copy()
        local_X_val = X_val.copy() if X_val is not None else None
        local_y_val = y_val.copy() if y_val is not None else None
        
        # 모델 입력 형태 계산 및 모델 빌드
        if len(local_X_train.shape) == 3:
            input_shape = (local_X_train.shape[1], local_X_train.shape[2])  # (sequence_length, features)
            model.build_model(input_shape)
        else:
            logger.error(f"GRU 모델에 맞지 않는 입력 형태: {local_X_train.shape}")
            return 0  # 실패 시 0 반환 (최대화 목표인 경우 최악의 점수)
            
        # 방향성 예측을 위한 타겟 데이터 처리
        binary_targets = False
        y_train_binary = None
        y_val_binary = None
        
        # 타겟 데이터가 0/1 이진 클래스가 아닌 실제 가격일 경우 처리
        if len(local_y_train.shape) == 1 or local_y_train.shape[1] == 1:
            logger.info(f"y_train 값 범위: 최소={np.min(local_y_train)} 최대={np.max(local_y_train)}")
            
            # 고가격 데이터인 경우 (이진 0/1 값이 아님)
            if np.max(local_y_train) > 1 or np.min(local_y_train) < 0:
                logger.warning("y_train에 0과 1 이외의 값이 있습니다. 이진 클래스로 변환합니다.")
                # 모든 값을 일단 1(상승)으로 설정하는 잘못된 방식 대신
                # 실제 가격 변화에 따른 방향성(상승/하락)을 계산
                y_train_binary = (np.diff(local_y_train.flatten(), prepend=local_y_train.flatten()[0]) > 0).astype(int)
                if local_y_val is not None:
                    y_val_binary = (np.diff(local_y_val.flatten(), prepend=local_y_val.flatten()[0]) > 0).astype(int)
                binary_targets = True
                logger.info(f"가격 데이터를 방향성으로 변환: 상승={np.sum(y_train_binary)}, 하락={len(y_train_binary) - np.sum(y_train_binary)}")
            else:
                # 이미 0/1 이진 값이면 그대로 사용
                y_train_binary = local_y_train.astype(np.int32)
                if local_y_val is not None:
                    y_val_binary = local_y_val.astype(np.int32)
                binary_targets = True
        
        # 클래스 가중치 계산 (이진 분류의 경우)
        class_weights = None
        if binary_targets:
            # 1차원 배열로 변환
            if len(y_train_binary.shape) == 2:
                y_train_binary = y_train_binary.reshape(-1)
            
            # 클래스별 샘플 수 계산
            unique_classes, class_counts = np.unique(y_train_binary, return_counts=True)
            n_samples = len(y_train_binary)
            n_classes = 2  # 이진 분류이므로 2개의 클래스(0과 1)가 있어야 함
            
            # 클래스 불균형이 심한지 확인
            is_imbalanced = False
            if len(unique_classes) == 2:
                ratio = min(class_counts) / max(class_counts)
                is_imbalanced = ratio < 0.4  # 클래스 비율이 40% 미만이면 불균형으로 간주
                logger.info(f"클래스 비율: {ratio:.2f}, 불균형 여부: {is_imbalanced}")
            
            # SMOTE를 사용하여 클래스 균형 맞추기
            if is_imbalanced and use_smote and SMOTE_AVAILABLE and len(unique_classes) > 1:
                try:
                    logger.info("SMOTE를 사용하여 클래스 불균형 처리를 시작합니다.")
                    # 2D 형태로 변환 (SMOTE는 2D 배열 필요)
                    if len(local_X_train.shape) == 3:
                        # 3D 데이터를 2D로 변환
                        seq_len, n_features = local_X_train.shape[1], local_X_train.shape[2]
                        reshaped_X = local_X_train.reshape(local_X_train.shape[0], -1)
                        
                        # SMOTE 적용
                        sm = SMOTE(random_state=42)
                        reshaped_X_balanced, y_train_binary = sm.fit_resample(reshaped_X, y_train_binary)
                        
                        # 다시 3D로 변환
                        n_samples_new = reshaped_X_balanced.shape[0]
                        local_X_train = reshaped_X_balanced.reshape(n_samples_new, seq_len, n_features)
                        
                        # 새로운 클래스 분포 확인
                        new_unique, new_counts = np.unique(y_train_binary, return_counts=True)
                        logger.info(f"SMOTE 적용 후 클래스 분포: {dict(zip(new_unique, new_counts))}")
                    else:
                        # 이미 2D 형태인 경우
                        sm = SMOTE(random_state=42)
                        local_X_train, y_train_binary = sm.fit_resample(local_X_train, y_train_binary)
                        
                        # 새로운 클래스 분포 확인
                        new_unique, new_counts = np.unique(y_train_binary, return_counts=True)
                        logger.info(f"SMOTE 적용 후 클래스 분포: {dict(zip(new_unique, new_counts))}")
                
                except Exception as e:
                    logger.error(f"SMOTE 적용 중 오류 발생: {str(e)}")
                    logger.warning("클래스 가중치를 사용하여 불균형을 처리합니다.")
                    
                    # 가중치 계산으로 대체
                    if len(unique_classes) < 2:
                        logger.warning(f"한 종류의 클래스만 발견됨: {unique_classes}. 인공적으로 균형 있는 클래스 가중치 생성.")
                        class_weights = {0: 1.0, 1: 1.0}
                        
                        for cls, count in zip(unique_classes, class_counts):
                            class_weights[int(cls)] = float(n_samples / (n_classes * count))
                    else:
                        class_weights = {
                            int(cls): float(n_samples / (n_classes * count))
                            for cls, count in zip(unique_classes, class_counts)
                        }
            else:
                # SMOTE를 사용하지 않는 경우, 기존 클래스 가중치 계산 로직 사용
                if len(unique_classes) < 2:
                    logger.warning(f"한 종류의 클래스만 발견됨: {unique_classes}. 인공적으로 균형 있는 클래스 가중치 생성.")
                    class_weights = {0: 1.0, 1: 1.0}
                    
                    for cls, count in zip(unique_classes, class_counts):
                        class_weights[int(cls)] = float(n_samples / (n_classes * count))
                else:
                    class_weights = {
                        int(cls): float(n_samples / (n_classes * count))
                        for cls, count in zip(unique_classes, class_counts)
                    }
            
            logger.info(f"클래스 분포: {dict(zip(unique_classes, class_counts))}")
            if class_weights:
                logger.info(f"계산된 클래스 가중치: {class_weights}")
        
        # Train with early stopping
        training_history = model.train(
            local_X_train, 
            y_train_binary if binary_targets else local_y_train, 
            local_X_val,
            y_val_binary if binary_targets else local_y_val,
            class_weights=class_weights,
            early_stopping_patience=10,
            reduce_lr_patience=5
        )
        
        # Return validation accuracy (higher is better, negate for minimization)
        val_accuracy = training_history.get('val_accuracy', 0)
        
        # 실제 검증 정확도 사용
        return -val_accuracy
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # Minimize validation accuracy
        sampler=optuna.samplers.TPESampler(seed=42)  # Use TPE algorithm with fixed seed
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters and results
    best_params = study.best_params
    best_value = -study.best_value  # Convert back to accuracy
    
    # Create complete parameter dict from best params
    best_model_params = {
        'units': [
            best_params['units_l1'],
            best_params['units_l2'],
        ],
        'dropout_rate': best_params['dropout_rate'],
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size'],
        'sequence_length': best_params['sequence_length'],
    }
    
    # Log best parameters
    logger.info(f"GRU+LayerNormalization 모델 최적화 완료. 최적 검증 정확도: {best_value:.6f}")
    logger.info(f"최적 파라미터: {best_model_params}")
    
    # Return optimization results
    return {
        'best_params': best_model_params,
        'best_val_accuracy': best_value,
        'best_val_f1': best_value,  # F1 점수는 현재 없어서 정확도를 대신 사용
        'n_trials': n_trials,
        'study': study
    }

@log_execution
def optimize_rf_direction_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_iter: int = 30,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Optimize Random Forest direction model hyperparameters using RandomizedSearchCV.
    
    This function is a wrapper around the existing RandomForestDirectionModel.optimize_hyperparams
    method that provides a consistent interface with the Bayesian optimization functions.
    
    Args:
        X_train: Training data features
        y_train: Training data targets
        X_val: Validation data features
        y_val: Validation data targets
        n_iter: Number of parameter settings sampled
        random_state: Random state for reproducibility
        
    Returns:
        Dict containing best parameters and optimization results
    """
    logger.info(f"Starting Random Search optimization for RandomForest Direction model with {n_iter} trials")
    
    # Function to check and remove string columns
    def remove_string_columns(X):
        if isinstance(X, np.ndarray):
            # Check dtype of each column
            string_cols = []
            for i in range(X.shape[1]):
                # Detect string columns
                try:
                    col_sample = X[0, i]
                    if isinstance(col_sample, str):
                        string_cols.append(i)
                except:
                    pass
                    
            if string_cols:
                logger.warning(f"Detected and removing {len(string_cols)} string columns.")
                # Select only non-string columns
                mask = np.ones(X.shape[1], dtype=bool)
                mask[string_cols] = False
                return X[:, mask]
        return X
    
    # Remove string columns
    logger.info(f"Original training data shape: {X_train.shape}")
    X_train_clean = remove_string_columns(X_train)
    X_val_clean = remove_string_columns(X_val)
    logger.info(f"Training data shape after removing string columns: {X_train_clean.shape}")
    
    # Define parameter grid (optimized parameter space)
    param_grid = {
        'n_estimators': [100, 150, 200],       # Increased from 50, 100, 150 → 100, 150, 200
        'max_depth': [7, 10, 15, 20],          # Increased from 5, 7, 10, 15 → 7, 10, 15, 20
        'min_samples_split': [10, 15, 20],     # Adjusted from 10, 20, 30 → 10, 15, 20
        'min_samples_leaf': [2, 3, 4]          # Adjusted from 2, 4, 6 → 2, 3, 4 (allowing smaller leaves)
    }
    
    # Create model instance
    rf_model = RandomForestDirectionModel()
    
    # Run optimization using the model's existing method
    optimization_result = rf_model.optimize_hyperparams(
        X_train_clean, y_train, 
        X_val_clean, y_val,
        param_grid=param_grid,
        n_iter=n_iter,
        method='random',
        random_state=random_state
    )
    
    # Log results
    logger.info(f"RandomForest model optimization complete. Best validation score: {optimization_result['best_score']:.6f}")
    logger.info(f"Best parameters: {optimization_result['best_params']}")
    
    return optimization_result

@log_execution
def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.15, target_column: str = 'target') -> Dict[str, pd.DataFrame]:
    """
    데이터를 훈련(train), 검증(validation), 테스트(test) 세트로 분할
    
    Args:
        df (pd.DataFrame): 분할할 데이터프레임
        test_size (float, optional): 테스트 세트 비율. 기본값 0.2.
        val_size (float, optional): 검증 세트 비율. 기본값 0.15.
        target_column (str, optional): 타겟 컬럼명. 기본값 'target'.
        
    Returns:
        Dict[str, pd.DataFrame]: 분할된 데이터 딕셔너리 ('train', 'val', 'test' 키 포함)
    """
    try:
        # 입력 데이터 검증
        if df is None or df.empty:
            logger.error("빈 데이터프레임은 분할할 수 없습니다.")
            return {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
        
        # target_column이 있는지 확인
        if target_column not in df.columns:
            logger.error(f"타겟 컬럼 '{target_column}'이 데이터프레임에 없습니다.")
            return {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
            
        # NaN 값 확인 및 처리
        if df.isna().any().any():
            logger.warning("데이터에 NaN 값이 있어 처리합니다.")
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 데이터 크기 계산
        data_size = len(df)
        test_count = int(data_size * test_size)
        val_count = int(data_size * val_size)
        train_count = data_size - test_count - val_count
        
        # 데이터 크기 검증
        if train_count <= 0 or val_count <= 0 or test_count <= 0:
            logger.error(f"데이터 크기가 너무 작아 분할할 수 없습니다: 전체={data_size}, 훈련={train_count}, 검증={val_count}, 테스트={test_count}")
            # 기본 분할(0.7, 0.15, 0.15)로 재시도
            test_size = 0.15
            val_size = 0.15
            test_count = int(data_size * test_size)
            val_count = int(data_size * val_size)
            train_count = data_size - test_count - val_count
            
            if train_count <= 0 or val_count <= 0 or test_count <= 0:
                logger.error("기본 분할도 실패했습니다. 최소 분할로 재시도합니다.")
                # 최소 분할(데이터가 매우 작은 경우)
                if data_size < 5:
                    return {
                        'train': df.copy(),
                        'val': df.copy(),
                        'test': df.copy()
                    }
                else:
                    train_count = max(1, data_size - 2)
                    val_count = 1
                    test_count = data_size - train_count - val_count
        
        # 시계열 데이터이므로 순서대로 분할
        train_df = df.iloc[:train_count].copy()
        val_df = df.iloc[train_count:train_count + val_count].copy()
        test_df = df.iloc[train_count + val_count:].copy()
        
        logger.info(f"데이터 분할 완료: 훈련={len(train_df)}개, 검증={len(val_df)}개, 테스트={len(test_df)}개")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    except Exception as e:
        logger.error(f"데이터 분할 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 오류 발생 시 빈 데이터프레임 반환
        return {
            'train': pd.DataFrame(),
            'val': pd.DataFrame(),
            'test': pd.DataFrame()
        } 