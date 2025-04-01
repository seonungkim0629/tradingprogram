"""
Bitcoin Trading Bot - Main Application

This is the main entry point for the Bitcoin Trading Bot application.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import importlib
import traceback
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.logging import get_logger, log_execution
import data.collectors as collectors
import data.processors as processors
from utils.evaluation import perform_walk_forward_analysis
import strategies
try:
    from config import settings
except ImportError:
    # settings 모듈이 없을 경우 기본값 설정
    class DefaultSettings:
        def __init__(self):
            self.TRADING_AMOUNT = 10_000_000
            self.TRADING_FEE = 0.0005
            self.MONITOR_RESOURCE_USAGE = False
            self.DEFAULT_MARKET = 'KRW-BTC'
    settings = DefaultSettings()

# Initialize logger
logger = get_logger(__name__)


@log_execution
def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Bitcoin Trading Bot')
    
    parser.add_argument('--backtest', action='store_true',
                      help='Run in backtest mode')
    
    parser.add_argument('--strategy', type=str, default='HarmonizingStrategy',
                      help='Trading strategy to use')
    
    parser.add_argument('--market', type=str, default='KRW-BTC',
                      help='Market to trade (e.g., KRW-BTC)')
    
    parser.add_argument('--backtest-start', type=str,
                      help='Backtest start date (YYYY-MM-DD)')
    
    parser.add_argument('--backtest-end', type=str,
                      help='Backtest end date (YYYY-MM-DD)')
    
    parser.add_argument('--optimize', action='store_true',
                      help='Run parameter optimization')
    
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                      help='Path to configuration file')
    
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
                      
    parser.add_argument('--no-ui', action='store_true',
                      help='Disable UI')
    
    args = parser.parse_args()
    
    # Set mode based on arguments
    if args.backtest:
        args.mode = 'backtest'
    elif args.optimize:
        args.mode = 'optimize'
    else:
        args.mode = 'live'
    
    return args


@log_execution
def load_config(args):
    """
    Load configuration from command line arguments
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        dict: Configuration
    """
    config = {
        'mode': args.mode,
        'strategy': args.strategy,
        'market': args.market,
        'debug': args.debug,
        'ui_enabled': not args.no_ui,
        'backtest_start': args.backtest_start,
        'backtest_end': args.backtest_end
    }
    
    # If debug is enabled, set log level to DEBUG
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    return config


@log_execution
def get_system_state() -> Dict[str, Any]:
    """
    Get current system state for checkpointing
    
    Returns:
        Dict[str, Any]: System state
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'performance': get_performance_summary(),
        'system_info': get_system_info(),
        'settings': {k: v for k, v in settings.get_all_settings().items() 
                   if isinstance(v, (str, int, float, bool, list, dict))}
    }


class TradingSystem:
    """
    Trading system that manages strategy execution and trade lifecycle
    """
    
    def __init__(self, strategy: str, market: str, is_live: bool = False):
        """
        Initialize trading system
        
        Args:
            strategy (str): Strategy name
            market (str): Market symbol
            is_live (bool): Whether to use live trading
        """
        self.strategy_name = strategy
        self.market = market
        self.is_live = is_live
        self.logger = get_logger(__name__)
        self.running = False
        
        # Initialize data manager for handling both daily and hourly data
        from data.data_manager import TradingDataManager
        self.data_manager = TradingDataManager(ticker=market)
        
        # Initialize strategy
        try:
            strategy_class = getattr(strategies, strategy)
            self.strategy = strategy_class(market=market)
        except (AttributeError, ImportError):
            self.logger.error(f"Strategy {strategy} not found")
            raise ValueError(f"Strategy {strategy} not found")
        
        # Initialize GPT analyzer if enabled and in live mode
        self.gpt_analyzer = None
        if settings.USE_GPT and is_live:
            try:
                from models.gpt_analyzer import GPTMarketAnalyzer
                self.gpt_analyzer = GPTMarketAnalyzer(
                    model=settings.GPT_MODEL,
                    max_tokens=settings.GPT_MAX_TOKENS,
                    temperature=settings.GPT_TEMPERATURE,
                    market_context_days=settings.GPT_MARKET_CONTEXT_DAYS,
                    include_technical_indicators=settings.GPT_INCLUDE_TECHNICAL_INDICATORS,
                    include_market_sentiment=settings.GPT_INCLUDE_MARKET_SENTIMENT,
                    include_news=settings.GPT_INCLUDE_NEWS
                )
                self.logger.info("GPT analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize GPT analyzer: {str(e)}")
        
        # Set trading mode
        self.mode = "live" if is_live else "paper"
        
        self.logger.info(f"Initialized {self.mode} trading system with {strategy} for {market}")
    
    def generate_signals(self) -> Dict[str, Any]:
        """
        Generate trading signals using both daily and hourly data
        
        Returns:
            Dict[str, Any]: Combined signal
        """
        # Get latest data
        current_data = self.data_manager.get_current_data()
        daily_data = current_data['daily']
        hourly_data = current_data['hourly']
        
        if daily_data is None or hourly_data is None:
            self.logger.error("Cannot generate signals: missing data")
            return {'signal': 'HOLD', 'reason': 'Missing data', 'confidence': 0.0}
        
        # Generate signals from daily data
        daily_signal = self.strategy.generate_signal(daily_data)
        
        # Generate signals from hourly data (assuming strategy can handle hourly data)
        hourly_signal = self.strategy.generate_signal(hourly_data)
        
        # Combine signals (simple implementation - can be more sophisticated)
        # If both signals agree, use that signal with higher confidence
        # Otherwise, prioritize daily signal but with reduced confidence
        if daily_signal['signal'] == hourly_signal['signal']:
            combined_signal = daily_signal.copy()
            combined_signal['confidence'] = max(daily_signal['confidence'], hourly_signal['confidence'])
            combined_signal['reason'] = f"Daily and hourly signals agree: {daily_signal['reason']}"
        else:
            combined_signal = daily_signal.copy()
            combined_signal['confidence'] *= 0.7  # Reduce confidence due to disagreement
            combined_signal['reason'] = (f"Daily signal ({daily_signal['signal']}, {daily_signal['confidence']:.2f}) "
                                       f"differs from hourly ({hourly_signal['signal']}, {hourly_signal['confidence']:.2f})")
        
        # Add metadata about both signals
        combined_signal['metadata'] = combined_signal.get('metadata', {})
        combined_signal['metadata']['daily_signal'] = daily_signal
        combined_signal['metadata']['hourly_signal'] = hourly_signal
        
        return combined_signal
    
    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process trading signal with GPT analysis if enabled
        
        Args:
            signal (Dict[str, Any]): Original trading signal
            
        Returns:
            Dict[str, Any]: Processed signal
        """
        # If GPT is not enabled or not initialized, return original signal
        if not self.gpt_analyzer:
            return signal
        
        try:
            # Get current data for GPT analysis
            current_data = self.data_manager.get_current_data()
            
            # Get GPT analysis
            gpt_analysis = self.gpt_analyzer.analyze_market(current_data['daily'])
            
            # Apply GPT filter
            if gpt_analysis['risk_level'] == 'high':
                self.logger.info("GPT analysis indicates high risk, changing signal to HOLD")
                return {
                    'signal': 'HOLD',
                    'reason': f"GPT risk assessment: {gpt_analysis['analysis']}",
                    'confidence': 0.5,
                    'metadata': {
                        'original_signal': signal,
                        'gpt_analysis': gpt_analysis
                    }
                }
            
            # If risk is not high, adjust signal confidence based on GPT analysis
            if signal['signal'] != 'HOLD':
                # Adjust position size based on GPT confidence
                if 'position_size' in signal:
                    signal['position_size'] *= gpt_analysis['confidence']
                
                # Add GPT analysis to metadata
                signal['metadata'] = signal.get('metadata', {})
                signal['metadata']['gpt_analysis'] = gpt_analysis
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in GPT analysis: {str(e)}")
            return signal
    
    def start(self):
        """Start the trading system"""
        self.running = True
        self.logger.info(f"Starting trading system in {self.mode} mode")
        
        # Load initial data
        success = self.data_manager.load_initial_data()
        if not success:
            self.logger.error("Failed to load initial data, trading system cannot start")
            self.running = False
            return
        
        # Start trading loop in a separate thread
        import threading
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
    def _trading_loop(self):
        """Main trading loop - runs in a separate thread"""
        try:
            self.logger.info("Trading loop started")
            
            while self.running:
                # Update data
                daily_updated, hourly_updated = self.data_manager.update_data()
                
                # Generate signals if data was updated or first run
                if daily_updated or hourly_updated:
                    # Generate signals using both daily and hourly data
                    signal = self.generate_signals()
                    
                    # Process the signal (apply GPT analysis if enabled)
                    processed_signal = self.process_signal(signal)
                    
                    # Apply risk management
                    portfolio = {'cash': 1000000, 'position': 0}  # Placeholder - would be actual portfolio in real implementation
                    final_signal = self.strategy.apply_risk_management(processed_signal, portfolio)
                    
                    # Log the final signal
                    self.logger.info(f"Final signal: {final_signal['signal']} with confidence {final_signal['confidence']:.2f}")
                    
                    # Execute trade (placeholder - would implement actual trade execution in real implementation)
                    if final_signal['signal'] == 'BUY':
                        self.logger.info(f"Would execute BUY at {self.data_manager.get_latest_price()}")
                    elif final_signal['signal'] == 'SELL':
                        self.logger.info(f"Would execute SELL at {self.data_manager.get_latest_price()}")
                
                # Sleep for a minute before next check
                import time
                time.sleep(60)
                
        except Exception as e:
            self.logger.error(f"Error in trading loop: {str(e)}")
            self.running = False
    
    def stop(self):
        """Stop the trading system"""
        self.running = False
        self.logger.info("Stopping trading system")
        
        # Wait for trading thread to finish if it's running
        if hasattr(self, 'trading_thread') and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5.0)


@log_execution
def run_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """백테스트 실행
    
    Args:
        config: 설정 정보
        
    Returns:
        dict: Backtest results
    """
    try:
        # 사용자 입력 또는 설정 파일에서 백테스트 옵션 설정
        symbol = config.get('symbol', 'KRW-BTC')
        timeframe = config.get('timeframe', 'day')
        start_date_str = config.get('start_date', (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d'))
        end_date_str = config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        # 날짜 형식 변환
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        # 전략 설정
        strategy_name = config.get('strategy', 'HarmonizingStrategy')
        initial_balance = config.get('initial_balance', 10_000_000)
        
        # 전략 클래스 동적 임포트
        try:
            strategy_class = getattr(strategies, strategy_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Strategy {strategy_name} not found: {e}")
            available_strategies = find_available_strategies()
            logger.info(f"Available strategies: {', '.join(available_strategies)}")
            return
        
        # 데이터 획득 (기존 collectors 모듈 사용)
        logger.info(f"Fetching data for {symbol} ({timeframe}) from {start_date_str} to {end_date_str}")
        
        days_to_fetch = (end_date - start_date).days + 30  # 여유있게 데이터 가져오기
        data = collectors.get_historical_data(ticker=symbol, days=days_to_fetch, indicators=True)
        
        if data is None or data.empty:
            logger.error("No data available for the specified period")
            return
            
        # 날짜 범위로 필터링
        data = data[data.index >= start_date]
        data = data[data.index <= end_date]
        
        if len(data) == 0:
            logger.error(f"No data available for the specified period {start_date_str} to {end_date_str}")
            return
        
        # 전략 초기화
        strategy_params = config.get('strategy_params', {})
        if strategy_params is None:
            strategy_params = {}
            
        # HarmonizingStrategy를 위한 기본 파라미터 설정
        if strategy_name == 'HarmonizingStrategy':
            default_strategy_params = {
                'trend_weight': 0.3,
                'ma_weight': 0.3,
                'rsi_weight': 0.3,
                'hourly_weight': 0.05,
                'ml_weight': 0.3,
                'use_ml_models': True,
                'confidence_threshold': 0.0005
            }
            # 사용자 정의 파라미터가 있으면 기본값을 업데이트
            for k, v in strategy_params.items():
                default_strategy_params[k] = v
            strategy_params = default_strategy_params
            
        # 백테스트 모드 파라미터 추가
        if strategy_name == 'HarmonizingStrategy':
            strategy = strategy_class(market=symbol, timeframe=timeframe, strategy_params=strategy_params, is_backtest=True)
        else:
            strategy = strategy_class(market=symbol, **strategy_params)
        
        # 백테스트 엔진 초기화 (기존 backtest 모듈 사용)
        from backtest.engine import BacktestEngine
        
        backtest_params = {
            'initial_balance': initial_balance,
            'commission_rate': config.get('commission_rate', 0.0005),
            'data_frequency': timeframe
        }
        
        engine = BacktestEngine(**backtest_params)
        
        # 백테스트 실행
        logger.info(f"Starting backtest for {symbol} from {start_date_str} to {end_date_str}")
        result = engine.run(
            data=data,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date
        )
        
        # 결과 저장
        save_backtest_results(result, strategy_name, symbol, timeframe)
        
        # 결과 분석
        result_dict = result.to_dict()
        
        # 몬테카를로 시뮬레이션
        try:
            # 백테스트 엔진에 run_monte_carlo_simulation 메소드가 없음
            # 간단히 로그만 출력
            logger.info("몬테카를로 시뮬레이션은 별도 분석으로 확인 필요")
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
        
        # 백테스트 결과 직접 표시 (GUI 없이도 확인 가능하도록)
        display_backtest_results(result.to_dict())
        
        # 백테스트 모드 종료 설정
        config['backtest_mode'] = False
        
        return result
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        traceback.print_exc()
        return None


def display_backtest_results(result_dict):
    """백테스트 결과를 콘솔에 출력한다."""
    logger.info("\n===== 백테스트 상세 결과 =====")
    logger.info(f"초기 자본: {result_dict['initial_balance']:,.0f} KRW")
    logger.info(f"최종 자본: {result_dict['final_balance']:,.0f} KRW")
    
    # Calculate total_profit if it doesn't exist in result_dict
    total_profit = result_dict.get('total_profit', result_dict['final_balance'] - result_dict['initial_balance'])
    total_return = result_dict.get('total_return', total_profit / result_dict['initial_balance'])
    
    logger.info(f"총 수익: {total_profit:,.0f} KRW ({total_return:.2%})")
    
    # Use get() with default value 0 for all metrics to avoid KeyError
    logger.info(f"월 수익률: {result_dict.get('monthly_return', 0):.2%}")
    logger.info(f"연간 수익률: {result_dict.get('annualized_return', 0):.2%}")
    logger.info(f"최대 낙폭: {result_dict.get('max_drawdown', 0):.2%}")
    logger.info(f"샤프 비율: {result_dict.get('sharpe_ratio', 0):.2f}")
    logger.info(f"변동성: {result_dict.get('volatility', 0):.2%}")
    logger.info(f"승률: {result_dict.get('win_rate', 0):.2%}")
    logger.info(f"총 거래 수: {result_dict.get('total_trades', 0)}")
    logger.info(f"평균 수익: {result_dict.get('avg_profit', 0):.0f} KRW")
    logger.info(f"평균 보유 기간: {result_dict.get('avg_hold_time', 0):.1f} 일")
    logger.info(f"============================\n")


def save_backtest_results(result, strategy_name, symbol, timeframe):
    """백테스트 결과를 JSON 파일로 저장"""
    try:
        import os
        import json
        from datetime import datetime
        import numpy as np
        
        # 결과 저장 디렉토리 생성
        results_dir = os.path.join('data_storage', 'results', 'backtest')
        os.makedirs(results_dir, exist_ok=True)
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 파일명 생성
        filename = f"backtest_{strategy_name}_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(results_dir, filename)
        
        # NumPy 객체를 기본 Python 타입으로 변환하는 클래스
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return super(NumpyEncoder, self).default(obj)
        
        # 결과를 딕셔너리로 변환
        result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
        
        # 직렬화 가능한 데이터만 포함하도록 필터링
        serializable_result = {}
        for key, value in result_dict.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                serializable_result[key] = value
            elif isinstance(value, np.number):
                serializable_result[key] = float(value)
            elif value is None:
                serializable_result[key] = None
            else:
                try:
                    # 다른 객체는 문자열로 변환 시도
                    serializable_result[key] = str(value)
                except:
                    # 변환 불가능하면 스킵
                    pass
        
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, cls=NumpyEncoder, indent=4)
        
        logger.info(f"Backtest results saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving backtest results: {str(e)}")
        return None


def find_available_strategies():
    """사용 가능한 전략 클래스 목록을 반환"""
    try:
        import os
        import importlib
        import inspect
        
        strategies_dir = 'strategies'
        available_strategies = []
        
        # strategies 폴더의 모든 .py 파일 검색
        for file in os.listdir(strategies_dir):
            if file.endswith('.py') and not file.startswith('__'):
                module_name = file[:-3]  # .py 확장자 제거
                
                try:
                    # 모듈 동적 임포트
                    module_path = f"strategies.{module_name}"
                    module = importlib.import_module(module_path)
                    
                    # 모듈 내의 모든 클래스 검사
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # 같은 모듈에 정의된 클래스만 포함 (임포트된 클래스는 제외)
                        if obj.__module__ == module_path:
                            # 전략 클래스인지 확인 (명명 규칙 또는 상속 관계 등으로 판단 가능)
                            if name.endswith('Strategy'):
                                available_strategies.append(name)
                
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import module {module_name}: {e}")
        
        return sorted(available_strategies)
    except Exception as e:
        logger.error(f"Error finding available strategies: {e}")
        return ["HarmonizingStrategy"]  # 기본값으로 적어도 하나 반환


@log_execution
def run_optimization(config):
    """
    하모나이징 전략 최적화 실행
    
    Args:
        config (dict): 설정 정보
    
    Returns:
        dict: 최적화 결과
    """
    # 백테스트 모드 설정 - 최적화 시 불필요한 로그 감소
    from utils.logging import set_backtest_mode
    set_backtest_mode(True)
    
    # 설정된 날짜가 없으면 기본값 사용
    if config['backtest_start'] is None:
        # 기본 시작일 사용
        config['backtest_start'] = settings.BACKTEST_START_DATE
        logger.info(f"기본 최적화 시작일 사용: {config['backtest_start']}")
    
    if config['backtest_end'] is None:
        # 기본 종료일 사용
        config['backtest_end'] = settings.BACKTEST_END_DATE
        logger.info(f"기본 최적화 종료일 사용: {config['backtest_end']}")
    
    # 문자열 날짜를 datetime으로 변환
    start_dt = datetime.strptime(config['backtest_start'], "%Y-%m-%d")
    end_dt = datetime.strptime(config['backtest_end'], "%Y-%m-%d")
    
    # 최적화 모드 결정
    strategy_name = config.get('strategy', 'HarmonizingStrategy')
    
    if strategy_name != 'HarmonizingStrategy':
        logger.warning(f"주의: 현재 최적화는 HarmonizingStrategy에 최적화되어 있습니다. 다른 전략 '{strategy_name}'에 대한 최적화는 제한적일 수 있습니다.")
    
    # StrategyOptimizer 클래스 import
    from strategy_optimizer import StrategyOptimizer, run_optimization_and_test
    
    logger.info(f"StrategyOptimizer를 사용한 '{strategy_name}' 전략 최적화 시작 (GRU+LayerNormalization 적용)")
    
    # 최적화 실행 전 확인
    prompt = f"\n{strategy_name} 전략 최적화를 시작하시겠습니까? (GRU+LayerNormalization 적용됨) (y/n): "
    response = input(prompt)
    
    if response.lower() != 'y':
        logger.info("최적화가 사용자에 의해 취소되었습니다.")
        set_backtest_mode(False)  # 백테스트 모드 해제
        return None
    
    # 최적화 옵션 입력 받기
    reduced_search = True  # 기본값
    try:
        search_type = input("최적화 검색 방식 (1: 빠른 검색(기본값), 2: 전체 검색): ")
        if search_type == '2':
            reduced_search = False
            logger.info("전체 파라미터 공간 검색 모드 선택됨 (더 오래 걸립니다)")
        else:
            logger.info("빠른 검색 모드 선택됨 (제한된 파라미터 공간)")
    except:
        pass
    
    # 최적화 실행
    try:
        # 최적화 및 백테스트 실행
        logger.info("GRU+LayerNormalization 모델이 적용된 최적화 및 백테스트 시작...")
        optimal_result, comparison_results = run_optimization_and_test(reduced_search=reduced_search)
        
        if optimal_result is None:
            logger.error("최적화 실패!")
            set_backtest_mode(False)  # 백테스트 모드 해제
            return None
        
        # 결과 요약
        results = {
            'optimal_parameters': optimal_result.strategy.parameters,
            'returns': optimal_result.returns,
            'monthly_return': optimal_result.metrics.get('monthly_return', 0),
            'sharpe_ratio': optimal_result.metrics.get('sharpe_ratio', 0),
            'max_drawdown': optimal_result.metrics.get('max_drawdown', 0),
            'win_rate': optimal_result.metrics.get('win_rate', 0),
            'total_trades': len(optimal_result.trades) // 2
        }
        
        # 최적화 결과 출력
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Optimization Results:")
        logger.info(f"{'=' * 40}")
        logger.info(f"Optimal Parameters: {optimal_result.strategy.parameters}")
        logger.info(f"Total Return: {optimal_result.returns:.2%}")
        logger.info(f"Monthly Return: {results['monthly_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"{'=' * 40}\n")
        
        # 백테스트 모드 해제
        set_backtest_mode(False)
        
        return results
    except Exception as e:
        logger.error(f"최적화 중 오류 발생: {str(e)}", exc_info=True)
        set_backtest_mode(False)  # 백테스트 모드 해제
        return None


@log_execution
def run_trading_bot(config):
    """
    Run the trading bot with the specified configuration
    
    Args:
        config (dict): Configuration
    """
    mode = config['mode']
    market = config['market']
    
    logger.info(f"Starting Bitcoin Trading Bot in {mode} mode for {market}")
    
    try:
        # Start monitoring system
        monitoring_thread = start_monitoring(interval=60)
        
        # Set up recovery system
        setup_recovery_system(
            checkpoint_data_callback=get_system_state,
            state_data_callback=get_system_state
        )
        
        # Create initial checkpoint
        create_checkpoint(get_system_state())
        
        if mode == 'backtest':
            # Run backtest
            results = run_backtest(config)
            
            # Print backtest results
            print("\n=== 백테스트 결과 ===")
            
            # 거래 통계 계산
            if 'trades' in results:
                total_trades = len(results['trades'])
                buy_count = sum(1 for trade in results['trades'] if trade['action'] == 'BUY')
                sell_count = sum(1 for trade in results['trades'] if trade['action'] == 'SELL')
                
                # 각 기본 키에 대한 한글 매핑
                korean_keys = {
                    'strategy_name': '전략명',
                    'market': '거래시장',
                    'start_date': '시작일',
                    'end_date': '종료일',
                    'initial_balance': '초기 잔고',
                    'final_balance': '최종 잔고',
                    'returns': '수익률',
                    'period': '기간',
                    'total_return': '총 수익률',
                    'monthly_return': '월간 수익률',
                    'annualized_return': '연간 수익률',
                    'max_drawdown': '최대 낙폭',
                    'win_rate': '승률',
                    'total_trades': '총 거래 횟수',
                    'sharpe_ratio': '샤프 비율',
                    'sortino_ratio': '소르티노 비율',
                }
                
                # 기본 정보 출력 - 항목별로 가독성 높은 형태로 출력
                print("\n" + "="*50)
                for key, value in results.items():
                    if key in korean_keys:
                        # Format value based on key
                        if 'return' in key or 'rate' in key or 'drawdown' in key:
                            value = f"{value * 100:.2f}%" if isinstance(value, float) else value
                        elif key in ['initial_balance', 'final_balance']:
                            # 금액 형식으로 표시하고 소수점 제거
                            value = f"{int(value):,} 원" if isinstance(value, (int, float)) else value
                        elif 'ratio' in key:
                            value = f"{value:.2f}" if isinstance(value, float) else value
                        # 날짜 관련 항목은 그대로 출력 (이미 format_backtest_results에서 변환됨)
                        if key == 'period':
                            print(f"{korean_keys[key]}: {value}")
                        # 금액 표시 항목 (이미 형식이 적용되어 있으므로 확인)
                        elif key in ['initial_balance', 'final_balance']:
                            if isinstance(value, str) and ('원' in value or '₩' in value):
                                print(f"{korean_keys[key]}: {value}")
                            else:
                                # 숫자인 경우 형식화
                                try:
                                    num_value = float(value) if isinstance(value, str) else value
                                    print(f"{korean_keys[key]}: {num_value:,} 원")
                                except:
                                    print(f"{korean_keys[key]}: {value}")
                        # 비율 표시 항목
                        elif key in ['total_return', 'monthly_return', 'annualized_return', 'returns']:
                            if isinstance(value, str) and '%' in value:
                                print(f"{korean_keys[key]}: {value}")
                            else:
                                # 숫자인 경우 퍼센트로 형식화
                                try:
                                    num_value = float(value) if isinstance(value, str) else value
                                    print(f"{korean_keys[key]}: {num_value*100:.2f}%")
                                except:
                                    print(f"{korean_keys[key]}: {value}")
                        # 거래 횟수
                        elif key == 'total_trades':
                            if isinstance(value, str) and '회' in value:
                                print(f"{korean_keys[key]}: {value}")
                            else:
                                # 숫자인 경우 형식화
                                try:
                                    num_value = int(value) if isinstance(value, str) else value
                                    print(f"{korean_keys[key]}: {num_value:,}회")
                                except:
                                    print(f"{korean_keys[key]}: {value}")
                        # 비율 항목 (샤프, 소르티노)
                        elif 'ratio' in key:
                            if isinstance(value, str):
                                print(f"{korean_keys[key]}: {value}")
                            else:
                                try:
                                    num_value = float(value) if isinstance(value, str) else value
                                    print(f"{korean_keys[key]}: {num_value:.2f}")
                                except:
                                    print(f"{korean_keys[key]}: {value}")
                        else:
                            print(f"{korean_keys[key]}: {value}")
                    elif key != 'trades' and key != 'metrics' and key != 'parameters':
                        print(f"{key}: {value}")
                
                # 리스크/성과 지표 구분하여 출력
                print("\n" + "-"*20 + " 리스크/성과 지표 (GRU+LayerNormalization) " + "-"*20)
                
                # 주요 지표들을 직접 출력
                metrics = results.get('metrics', {})
                
                # MDD (최대 낙폭)
                mdd = metrics.get('max_drawdown', 0)
                print(f"최대 낙폭(MDD): {mdd*100:.2f}%")
                
                # 샤프 비율
                sharpe = metrics.get('sharpe_ratio', 0)
                print(f"샤프 비율: {sharpe:.2f}")
                
                # 소르티노 비율
                if 'sortino_ratio' in metrics:
                    sortino = metrics.get('sortino_ratio', 0)
                    print(f"소르티노 비율: {sortino:.2f}")
                
                # 승률
                win_rate = metrics.get('win_rate', 0)
                print(f"승률: {win_rate*100:.2f}%")
                
                # 수익 팩터
                if 'profit_factor' in metrics:
                    profit_factor = metrics.get('profit_factor', 0)
                    print(f"수익 팩터: {profit_factor:.2f}")
                
                # 리스크 리워드 비율
                if 'risk_reward_ratio' in metrics and metrics['risk_reward_ratio'] > 0:
                    risk_reward = metrics.get('risk_reward_ratio')
                    print(f"리스크 리워드 비율: {risk_reward:.2f}")
                
                # 거래 통계 출력
                print("\n" + "-"*30 + " 거래 통계 (GRU+LayerNormalization 적용) " + "-"*30)
                print(f"총 거래 횟수: {total_trades:,}회")
                print(f"총 매수(BUY): {buy_count:,}회")
                print(f"총 매도(SELL): {sell_count:,}회")
                print(f"총 홀딩(HOLD): {total_trades - buy_count - sell_count:,}회")
                
                # 평균 거래 정보
                if 'avg_profit' in metrics:
                    print(f"평균 수익: {metrics['avg_profit']:,.0f}원")
                if 'avg_win' in metrics and 'avg_loss' in metrics:
                    print(f"평균 이익: {metrics['avg_win']:,.0f}원")
                    print(f"평균 손실: {metrics['avg_loss']:,.0f}원")
                if 'avg_hold_time' in metrics:
                    print(f"평균 보유 시간: {metrics['avg_hold_time']:.1f}시간")
                
                print("="*50)
            else:
                # 기존 방식으로 출력
                for key, value in results.items():
                    print(f"{key}: {value}")
                print("="*50)
        
        elif mode == 'live' or mode == 'paper':
            # 실시간 트레이딩 전에 최적화된 모델 파인 확인 및 복사
            try:
                from utils.model_utils import copy_optimized_models_to_saved
                logger.info("백테스트 전 최적화된 모델 파인 중...")
                copy_results = copy_optimized_models_to_saved(force_copy=False)
                
                if any(copy_results.values()):
                    logger.info("최신 최적화 모델이 트레이딩에 사용됩니다.")
                else:
                    logger.info("최적화된 모델 파일이 없습니다. 기존 저장된 모델을 사용합니다.")
            except Exception as e:
                logger.warning(f"모델 파일 복사 중 오류: {str(e)}. 기존 모델을 사용합니다.")
            
            # Initialize trading system
            trading_system = TradingSystem(
                strategy=config['strategy'],
                market=config['market'],
                is_live=(mode == 'live')
            )
            
            # Start trading
            trading_system.start()
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                trading_system.stop()
        
        elif mode == 'optimize':
            # 최적화 실행
            results = run_optimization(config)
            
            # 최적화 결과가 있는 경우에만 출력
            if results:
                # 최적화 결과는 run_optimization 함수 내에서 이미 출력되므로 여기서는 생략
                logger.info("최적화가 성공적으로 완료되었습니다. 결과는 test_results 디렉토리에서 확인할 수 있습니다.")
            else:
                logger.warning("최적화 결과가 없습니다.")
        
        else:
            logger.error(f"Unknown mode: {mode}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error running trading bot: {str(e)}", exc_info=True)
        sys.exit(1)


@log_execution
def main():
    """
    Bitcoin Trading Bot의 메인 엔트리포인트
    """
    logger.info("Starting Bitcoin Trading Bot")
    
    # 프로그램 옵션 파싱
    args = parse_arguments()
    
    # 설정 로드
    config = load_config(args)
    
    try:
        # 백테스트 모드
        if args.backtest:
            run_backtest(config)
        
        # 최적화 모드
        elif args.optimize:
            run_optimization(config)
        
        # 라이브 트레이딩 모드
        else:
            run_trading_bot(config)
            
    except Exception as e:
        logger.error(f"Error running trading bot: {str(e)}")
        traceback.print_exc()
    
    logger.info("Bitcoin Trading Bot shutting down")


if __name__ == "__main__":
    main() 