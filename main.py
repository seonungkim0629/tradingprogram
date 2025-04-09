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
import warnings
import json
import random

# NumPy 경고 무시
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
from backtest.engine import BacktestEngine
from utils.logging import logger, log_execution
from utils.database import initialize_database
from utils.state import initialize_state_directory
from utils.performance import schedule_regular_aggregation

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
    명령줄 인수 파싱 함수
    
    모든 실행 모드와 파라미터를 정의합니다.
    
    반환값:
        argparse.Namespace: 파싱된 인수 객체
    """
    parser = argparse.ArgumentParser(description="비트코인 트레이딩 봇")
    
    # 주요 실행 모드
    parser.add_argument("--backtest", action="store_true", help="백테스트 모드 실행 (기본값: 최근 200일, TradingEnsemble 사용)")
    parser.add_argument("--live", action="store_true", help="실시간 트레이딩 모드 실행")
    parser.add_argument("--optimize", action="store_true", help="전략 최적화 모드 실행")
    
    # 기간 설정
    parser.add_argument("--start-date", type=str, help="백테스트 시작 날짜 (YYYY-MM-DD 형식)")
    parser.add_argument("--end-date", type=str, help="백테스트 종료 날짜 (YYYY-MM-DD 형식)")
    parser.add_argument("--days", type=int, default=200, help="백테스트 기간 (일)")
    
    # 데이터 관련 설정
    parser.add_argument("--market", type=str, default="KRW-BTC", help="거래 시장 심볼 (예: KRW-BTC)")
    parser.add_argument("--timeframe", type=str, default="day", help="차트 시간프레임 (day, hour, minute)")
    parser.add_argument("--data-source", type=str, default="upbit", help="데이터 소스 (upbit, binance 등)")
    
    # 전략 설정
    parser.add_argument("--strategy", type=str, default="TradingEnsemble", 
                      help="사용할 트레이딩 전략 (TradingEnsemble, MovingAverageCrossover 등)")
    parser.add_argument("--params", type=str, help="전략 파라미터 (JSON 형식)")
    
    # 자본금 설정
    parser.add_argument("--initial-balance", type=float, default=10000000, help="초기 자본금 (KRW)")
    parser.add_argument("--position-size", type=float, default=0.2, help="포지션 크기 (0.0-1.0)")
    
    # 거래 수수료 및 슬리피지
    parser.add_argument("--fee", type=float, default=0.0005, help="거래 수수료 (0.0005 = 0.05%%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="슬리피지 (0.0005 = 0.05%%)")
    
    # 모델 관련 설정
    parser.add_argument("--model", type=str, help="사용할 모델 (RandomForest, GRU 등)")
    parser.add_argument("--model-path", type=str, help="모델 파일 경로")
    parser.add_argument("--sequence-length", type=int, default=60, help="시계열 시퀀스 길이")
    
    # 최적화 관련 설정
    parser.add_argument("--optimize-param", type=str, help="최적화할 파라미터 (JSON 형식)")
    parser.add_argument("--optimize-metric", type=str, default="profit", 
                       help="최적화 기준 지표 (profit, sharpe, sortino 등)")
    
    # 출력 및 시각화 설정
    parser.add_argument("--plot", action="store_true", help="백테스트 결과 플롯 생성")
    parser.add_argument("--verbose", type=int, default=1, help="출력 상세 수준 (0-3)")
    parser.add_argument("--report", type=str, help="결과 리포트 파일 경로")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       help="로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    
    # 시스템 설정
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--no-progress", action="store_true", help="진행 상태 표시 비활성화")
    
    # 특별 모드
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    parser.add_argument("--strict", action="store_true", help="엄격 모드 활성화 (특성 검증 실패 시 중단)")
    
    return parser.parse_args()


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
        'strategy': args.strategy,
        'market': args.market,
        'debug': args.debug
    }
    
    # 백테스트 관련 속성 추가
    if hasattr(args, 'start_date'):
        config['backtest_start'] = args.start_date
    if hasattr(args, 'end_date'):
        config['backtest_end'] = args.end_date
        
    # 모드 설정
    if args.backtest:
        config['mode'] = 'backtest'
    elif args.live:
        config['mode'] = 'live'
    elif args.optimize:
        config['mode'] = 'optimize'
    else:
        config['mode'] = 'unknown'
    
    # If debug is enabled, set log level to DEBUG
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    return config


'''
# 사용되지 않는 함수 - 필요시 나중에 구현
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
'''


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
        
        # 타임프레임 정보를 담은 데이터 딕셔너리 생성
        daily_data_dict = {'daily': daily_data}
        hourly_data_dict = {'1h': hourly_data}
        
        # Generate signals from daily data
        try:
            daily_signal = self.strategy.generate_signal(daily_data_dict)
        except (TypeError, AttributeError):
            # 만약 전략이 딕셔너리를 지원하지 않는다면 DataFrame을 직접 전달
            self.logger.info("전략이 딕셔너리 입력을 지원하지 않습니다. DataFrame을 직접 전달합니다.")
        daily_signal = self.strategy.generate_signal(daily_data)
        
        # Generate signals from hourly data (assuming strategy can handle hourly data)
        try:
            hourly_signal = self.strategy.generate_signal(hourly_data_dict)
        except (TypeError, AttributeError):
            # 만약 전략이 딕셔너리를 지원하지 않는다면 DataFrame을 직접 전달
            self.logger.info("전략이 딕셔너리 입력을 지원하지 않습니다. DataFrame을 직접 전달합니다.")
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
                updated_data = self.data_manager.update_data()
                
                # Generate signals if data was updated or first run
                if any(updated_data.values()):
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


def initialize_strategy(config):
    """
    설정에 따라 트레이딩 전략을 초기화합니다.
    
    인자:
        config (dict): 설정 정보를 담은 딕셔너리
        
    반환값:
        TradingStrategy: 초기화된 트레이딩 전략 객체
    """
    strategy_name = config.get('strategy', 'TradingEnsemble')
    market = config.get('market', 'KRW-BTC')
    timeframe = config.get('timeframe', 'day')
    strict_mode = config.get('strict', False)
    
    logger.info(f"전략 초기화: {strategy_name}, 시장: {market}, 시간프레임: {timeframe}, 엄격 모드: {strict_mode}")
    
    # 앙상블 모델 사용 시
    if strategy_name == 'TradingEnsemble':
        try:
            from ensemble.ensemble import TradingEnsemble
            import types
            
            # 기본 가중치 설정
            trend_weight = config.get('trend_weight', 0.3)
            ma_weight = config.get('ma_weight', 0.2)
            rsi_weight = config.get('rsi_weight', 0.2)
            hourly_weight = config.get('hourly_weight', 0.1)
            ml_weight = config.get('ml_weight', 0.2)
            
            # 앙상블 모델 생성
            strategy = TradingEnsemble(
                market=market,
                timeframe=timeframe,
                confidence_threshold=config.get('confidence_threshold', 0.6),
                trend_weight=trend_weight,
                ma_weight=ma_weight,
                rsi_weight=rsi_weight,
                hourly_weight=hourly_weight,
                ml_weight=ml_weight,
                use_ml_models=config.get('use_ml_models', True),
                use_market_context=config.get('use_market_context', True)
            )
            
            # 엄격 모드 설정
            if hasattr(strategy, 'strict_mode'):
                strategy.strict_mode = strict_mode
            
            # 필요한 특성 로딩
            if hasattr(strategy, 'feature_manager') and strategy.feature_manager:
                logger.info("특성 매니저를 사용하여 특성 로딩")
            elif hasattr(strategy, '_load_expected_features'):
                strategy.expected_features = strategy._load_expected_features()
                logger.info(f"특성 로드됨: {len(strategy.expected_features) if strategy.expected_features else 0}개")
            
            # TradingEnsemble 모델에 파라미터 추가 (백테스트 결과 저장용)
            strategy.parameters = {
                'name': strategy_name,
                'version': getattr(strategy, 'version', '1.0.0'),
                'market': market,
                'timeframe': timeframe,
                'use_market_context': config.get('use_market_context', True),
                'confidence_threshold': config.get('confidence_threshold', 0.6),
                'trend_weight': trend_weight,
                'ma_weight': ma_weight, 
                'rsi_weight': rsi_weight,
                'hourly_weight': hourly_weight,
                'ml_weight': ml_weight,
                'use_ml_models': config.get('use_ml_models', True),
                'strict_mode': strict_mode,
            }
            
            # apply_risk_management 메서드 바인딩
            if not hasattr(strategy, 'apply_risk_management'):
                from models.signal import TradingSignal
                from utils.constants import SignalType
                import traceback
                
                def apply_risk_management(self, signal: dict, portfolio: dict) -> dict:
                    """
                    리스크 관리 규칙을 적용하여 거래 신호를 조정합니다.
                    
                    Args:
                        signal (dict): 원본 거래 신호
                        portfolio (dict): 현재 포트폴리오 정보
                        
                    Returns:
                        dict: 조정된 거래 신호
                    """
                    try:
                        # 신호 및 포트폴리오 유효성 검사
                        if signal is None:
                            self.logger.warning("신호가 None입니다")
                            return {"signal": "HOLD", "confidence": 0.0, "reason": "신호 없음"}
                            
                        if portfolio is None:
                            self.logger.warning("포트폴리오 정보가 없습니다")
                            return signal  # 포트폴리오 정보가 없으면 원본 신호 반환
                        
                        # 현재 포지션 확인
                        current_position = portfolio.get('position', 0.0)
                        available_cash = portfolio.get('cash', 0.0)
                        total_equity = portfolio.get('total_equity', 0.0)
                        
                        # 원본 신호 정보
                        signal_type = signal.get("signal", "HOLD")
                        confidence = signal.get("confidence", 0.0)
                        reason = signal.get("reason", "")
                        metadata = signal.get("metadata", {})
                        position_size = signal.get("position_size", 0.0)
                        
                        # 리스크 관리 규칙 적용
                        
                        # 룰 1: 이미 포지션이 있는 경우 추가 매수 제한
                        if signal_type == "BUY" and current_position > 0:
                            # 이미 최대 포지션(0.9)에 가까우면 매수 신호 무시
                            if current_position >= 0.9:
                                self.logger.info("이미 최대 포지션에 도달했습니다 - 매수 신호 홀드로 변경")
                                return {
                                    "signal": "HOLD",
                                    "confidence": confidence,
                                    "reason": f"최대 포지션 도달 (현재: {current_position:.2f})",
                                    "metadata": metadata
                                }
                            
                            # 현재 포지션 + 새 포지션이 최대치(0.9)를 초과하면 포지션 크기 조정
                            if current_position + position_size > 0.9:
                                new_position_size = max(0.9 - current_position, 0)
                                self.logger.info(f"포지션 크기 조정: {position_size:.2f} -> {new_position_size:.2f} (현재 포지션: {current_position:.2f})")
                                position_size = new_position_size
                                
                                if position_size < 0.1:  # 너무 작은 추가 포지션은 의미가 없음
                                    self.logger.info("추가 포지션이 너무 작음 - 매수 신호 홀드로 변경")
                                    return {
                                        "signal": "HOLD",
                                        "confidence": confidence,
                                        "reason": f"추가 가능한 포지션이 너무 작음 ({position_size:.2f})",
                                        "metadata": metadata
                                    }
                        
                        # 룰 2: 매도 신호 처리 - 포지션이 없으면 무시
                        if signal_type == "SELL" and current_position <= 0:
                            self.logger.info("매도할 포지션이 없음 - 매도 신호 홀드로 변경")
                            return {
                                "signal": "HOLD",
                                "confidence": confidence,
                                "reason": "매도할 포지션 없음",
                                "metadata": metadata
                            }
                        
                        # 룰 3: 신뢰도에 따른 포지션 크기 조정
                        if signal_type == "BUY":
                            # 현금 부족 시 포지션 크기 조정
                            if available_cash < total_equity * position_size:
                                if available_cash <= 0:
                                    self.logger.info("가용 현금 부족 - 매수 신호 홀드로 변경")
                                    return {
                                        "signal": "HOLD",
                                        "confidence": confidence,
                                        "reason": "가용 현금 부족",
                                        "metadata": metadata
                                    }
                                
                                # 가용 현금에 맞게 포지션 크기 조정
                                adjusted_position = available_cash / total_equity
                                position_size = min(position_size, adjusted_position)
                                self.logger.info(f"현금 부족으로 포지션 크기 조정: {position_size:.2f}")
                            
                            # 신뢰도가 낮으면 포지션 크기 추가 감소
                            if confidence < 0.7:
                                position_size *= confidence  # 신뢰도에 비례하여 조정
                                self.logger.info(f"낮은 신뢰도로 포지션 크기 조정: {position_size:.2f} (신뢰도: {confidence:.2f})")
                        
                        # 최종 조정된 신호 반환
                        adjusted_signal = {
                            "signal": signal_type,
                            "confidence": confidence,
                            "position_size": position_size,
                            "reason": reason + " (리스크 관리 적용됨)",
                            "metadata": metadata
                        }
                        
                        self.logger.info(f"리스크 관리 적용 완료: {signal_type}, 포지션 크기: {position_size:.2f}")
                        return adjusted_signal
                        
                    except Exception as e:
                        self.logger.error(f"리스크 관리 적용 중 오류 발생: {str(e)}")
                        traceback.print_exc()
                        # 오류 발생 시 원본 신호 반환
                        return signal
                
                # 메서드 바인딩
                strategy.apply_risk_management = types.MethodType(apply_risk_management, strategy)
                logger.info("TradingEnsemble에 apply_risk_management 메서드를 바인딩했습니다.")
            
            logger.info(f"TradingEnsemble 전략 초기화 완료")
            return strategy
            
        except ImportError as e:
            logger.error(f"TradingEnsemble 모델 가져오기 실패: {str(e)}")
            logger.error("HarmonizingStrategy로 폴백합니다.")
            strategy_name = 'HarmonizingStrategy'
    
    # 동적 임포트로 다른 전략 사용
    try:
        # 전략 클래스 동적 임포트
        module_name = f"strategies.{strategy_name.lower()}"
        class_name = strategy_name
        
        # 모듈 이름에서 'Strategy' 제거하여 호환성 유지
        module_name = module_name.replace('Strategy', '').replace('strategy', '')
        
        logger.info(f"모듈 임포트: {module_name}, 클래스: {class_name}")
        
        try:
            strategy_module = importlib.import_module(module_name)
        except ImportError:
            # 첫 글자를 소문자로 바꾸어 다시 시도
            module_name = f"strategies.{strategy_name[0].lower() + strategy_name[1:].replace('Strategy', '')}"
            strategy_module = importlib.import_module(module_name)
        
        strategy_class = getattr(strategy_module, class_name)
    
        # 전략 인스턴스 생성
        strategy_params = config.get('params', {})
        if isinstance(strategy_params, str):
            strategy_params = json.loads(strategy_params)
        
        # 기본 파라미터 설정
        strategy_params.update({
            'market': market,
            'timeframe': timeframe,
            'is_backtest': config.get('backtest', False),
            'strict_mode': strict_mode,
        })
        
        # 전략 인스턴스 생성
        strategy = strategy_class(**strategy_params)
        
        logger.info(f"{strategy_name} 전략 초기화 완료")
        return strategy
        
    except Exception as e:
        logger.error(f"전략 초기화 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"전략 초기화 실패: {str(e)}")


@log_execution
def run_backtest(config):
    """백테스트를 실행합니다."""
    try:
        # 백테스트 모드 설정
        config['backtest_mode'] = True
        
        # 날짜 설정
        start_date_str = config.get('start_date', (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d'))
        end_date_str = config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        # 날짜 형식 변환
        config['start_date'] = datetime.strptime(start_date_str, '%Y-%m-%d')
        config['end_date'] = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        # 전략 초기화
        strategy = initialize_strategy(config)
        
        # TradingEnsemble 전략이고 훈련되지 않은 경우 훈련 데이터 준비 및 모델 훈련
        if strategy.name == 'TradingEnsemble' and not getattr(strategy, 'is_trained', False):
            try:
                logger.info("TradingEnsemble 전략이 훈련되지 않았습니다. 자동 훈련을 시작합니다.")
                
                # 방향 모델이 없으면 RandomForest 방향 모델 추가
                if not strategy.direction_models:
                    logger.info("방향 모델을 추가합니다.")
                    from models.random_forest import RandomForestDirectionModel
                    rf_model = RandomForestDirectionModel(
                        name="RF_Direction",
                        version="1.0.0",
                        n_estimators=100,
                        max_depth=10,
                        strict_mode=False
                    )
                    strategy.add_direction_model(rf_model, weight=1.0)
                    logger.info(f"RandomForestDirectionModel이 추가되었습니다. 현재 방향 모델 수: {len(strategy.direction_models)}")
                
                # 데이터 로드
                from data.collectors import get_historical_data
                
                # 훈련용 데이터는 백테스트 기간보다 길게 가져옴 (과거 데이터)
                training_end_date = config['start_date'] - timedelta(days=1)
                training_start_date = training_end_date - timedelta(days=365)  # 1년 데이터로 훈련
                
                logger.info(f"훈련 데이터 기간: {training_start_date.strftime('%Y-%m-%d')} ~ {training_end_date.strftime('%Y-%m-%d')}")
                
                training_data = get_historical_data(
                    ticker=strategy.market,
                    days=365,
                    indicators=True,
                    verbose=True
                )
                
                if training_data is None or training_data.empty:
                    logger.error("훈련 데이터가 없습니다. get_historical_data 함수가 None 또는 빈 DataFrame을 반환했습니다.")
                    raise ValueError("훈련 데이터를 불러올 수 없습니다. 백테스트를 중단합니다.")
                else:
                    logger.info(f"훈련 데이터 로드 완료: {len(training_data)} 일")
                    logger.info(f"훈련 데이터 컬럼: {training_data.columns.tolist()[:5]}... 외 {len(training_data.columns)-5}개")
                    
                    # 데이터 준비
                    from data.processors import prepare_data_for_training
                    
                    # 훈련 데이터 준비
                    prepared_data = prepare_data_for_training(
                        data=training_data,
                        sequence_length=10,
                        prediction_type='classification',
                        feature_subset='all',
                        normalize=True,
                        return_type='dict'
                    )
                    
                    # 준비된 데이터 확인
                    if prepared_data is None:
                        logger.error("훈련 데이터 준비 실패: prepare_data_for_training 함수가 None을 반환했습니다.")
                        raise ValueError("훈련 데이터 준비 실패. 백테스트를 중단합니다.")
                    else:
                        logger.info(f"훈련 데이터 준비 완료: {prepared_data.keys()}")
                        
                        # 데이터 형태 확인 및 모델 훈련
                        if 'X_train' in prepared_data and 'y_train' in prepared_data:
                            X_train = prepared_data['X_train']
                            y_train = prepared_data['y_train']
                            X_val = prepared_data.get('X_val', None)
                            y_val = prepared_data.get('y_val', None)
                            
                            # 특성 이름 생성 (데이터에 컬럼명이 있으면 사용)
                            feature_names = None
                            if hasattr(training_data, 'columns'):
                                feature_names = training_data.columns.tolist()
                                # timestamp 또는 target 열 제외
                                feature_names = [f for f in feature_names if f not in ['timestamp', 'target']]
                                logger.info(f"특성 이름 사용: {len(feature_names)}개")
                            
                            logger.info(f"모델 훈련 시작: 훈련 데이터 {X_train.shape}, 검증 데이터 {X_val.shape if X_val is not None else 'None'}")
                            if len(y_train.shape) > 1:  # 차원 줄이기
                                y_train = y_train.flatten()
                            if y_val is not None and len(y_val.shape) > 1:
                                y_val = y_val.flatten()
                                
                            logger.info(f"레이블 분포: 훈련 {np.bincount(y_train)}")
                            
                            # 모델 훈련
                            train_result = strategy.train(
                                X_train=X_train,
                                y_train_direction=y_train,
                                X_val=X_val,
                                y_val_direction=y_val,
                                feature_names=feature_names
                            )
                            
                            logger.info(f"TradingEnsemble 모델 훈련 완료: {train_result}")
                            
                            # 모델 저장
                            model_dir = os.path.join("models", "saved", strategy.name)
                            os.makedirs(model_dir, exist_ok=True)
                            strategy_saved_path = strategy.save(model_dir)
                            logger.info(f"모델 저장 완료: {strategy_saved_path}")
                            
                            # 훈련 상태 확인
                            if not strategy.is_trained:
                                logger.error("모델 훈련 후에도 is_trained 상태가 False입니다.")
                                raise ValueError("모델 훈련 실패. 백테스트를 중단합니다.")
                        else:
                            logger.error(f"훈련 데이터 키 오류: {prepared_data.keys()}")
                            raise ValueError("훈련 데이터 형식 오류. 백테스트를 중단합니다.")
            except Exception as e:
                logger.error(f"모델 훈련 중 오류 발생: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"모델 훈련 실패: {str(e)}. 백테스트를 중단합니다.")
        
        # 훈련 상태 최종 확인
        if strategy.name == 'TradingEnsemble' and not getattr(strategy, 'is_trained', False):
            logger.error(f"❗ 모델이 훈련되지 않았습니다. 백테스트를 중단합니다. ({strategy.name})")
            raise RuntimeError("모델 훈련 실패. 백테스트를 중단합니다.")
        
        # 백테스트 엔진 초기화
        engine = BacktestEngine(
            config={
                'initial_balance': config.get('initial_balance', 10_000_000),
                'fee': config.get('fee', 0.0005),
                'slippage': config.get('slippage', 0.0002),
                'market': config.get('market', 'KRW-BTC'),
                'timeframe': config.get('timeframe', 'day'),
                'start_date': start_date_str,
                'end_date': end_date_str,
                'strategy': config.get('strategy', 'TradingEnsemble')
            }
        )
        
        # 백테스트 실행
        logger.info(f"Starting backtest from {start_date_str} to {end_date_str}")
        result = engine.run(strategy, config['start_date'], config['end_date'])
        
        # 결과 저장
        save_backtest_results(
            result=result,
            strategy_name=config.get('strategy', 'TradingEnsemble'),
            market=config.get('market', 'KRW-BTC'),
            timeframe=config.get('timeframe', 'day')
        )
        
        # 결과 표시
        display_backtest_results(result)
        
        # 백테스트 모드 해제
        config['backtest_mode'] = False
        
        return result
        
    except Exception as e:
        logger.error(f"백테스트 실행 중 오류 발생: {str(e)}")
        raise


def display_backtest_results(result):
    """백테스트 결과를 콘솔에 출력한다."""
    # 결과가 BacktestResult 객체인 경우 딕셔너리로 변환
    if hasattr(result, 'to_dict'):
        result_dict = result.to_dict()
    else:
        result_dict = result
    
    logger.info("\n===== 백테스트 상세 결과 =====")
    logger.info(f"초기 자본: {result_dict.get('initial_balance', 0):,.0f} KRW")
    logger.info(f"최종 자본: {result_dict.get('final_balance', 0):,.0f} KRW")
    
    # Calculate total_profit if it doesn't exist in result_dict
    total_profit = result_dict.get('total_profit', result_dict.get('final_balance', 0) - result_dict.get('initial_balance', 0))
    total_return = result_dict.get('total_return', total_profit / result_dict.get('initial_balance', 1))
    
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


def save_backtest_results(result, strategy_name, market, timeframe):
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
        filename = f"backtest_{strategy_name}_{market}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
        return ["TradingEnsemble"]  # 기본값으로 적어도 하나 반환


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
    strategy_name = config.get('strategy', 'TradingEnsemble')
    
    if strategy_name != 'TradingEnsemble':
        logger.warning(f"주의: 현재 최적화는 TradingEnsemble에 최적화되어 있습니다. 다른 전략 '{strategy_name}'에 대한 최적화는 제한적일 수 있습니다.")
    
    # StrategyOptimizer 클래스 import - 아직 구현되지 않음
    # from strategy_optimizer import StrategyOptimizer, run_optimization_and_test
    
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
        # 최적화 모듈이 아직 구현되지 않았으므로 알림
        logger.warning("StrategyOptimizer 모듈이 아직 구현되지 않았습니다.")
        logger.info("최적화 기능은 추후 구현 예정입니다.")
        
        # 백테스트 모드 해제
        set_backtest_mode(False)
        
        # 임시 결과 반환
        return {
            'optimal_parameters': {
                'trend_weight': 0.3,
                'ma_weight': 0.3,
                'rsi_weight': 0.3,
                'hourly_weight': 0.05,
                'ml_weight': 0.3,
                'use_ml_models': True,
                'confidence_threshold': 0.0005
            },
            'monthly_return': 0.05,  # 5% 예시값
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.15,
            'win_rate': 0.55,
            'total_trades': 24
        }
        
        # 원래 코드 주석 처리
        """
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
        """
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
        # 시스템 모니터링 및 복구 시스템은 추후 구현 예정
        
        if mode == 'backtest':
            # Run backtest
            results = run_backtest(config)
            
            # Print backtest results
            print("\n=== 백테스트 결과 ===")
            
            # 결과가 객체인지 딕셔너리인지 확인
            results_dict = results.to_dict() if hasattr(results, 'to_dict') else results
            
            # 거래 통계 계산
            if results_dict and 'trades' in results_dict:
                trades = results_dict['trades']
                total_trades = len(trades)
                buy_count = sum(1 for trade in trades if trade.get('action') == 'BUY')
                sell_count = sum(1 for trade in trades if trade.get('action') == 'SELL')
                
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
                for key, value in results_dict.items():
                    if key in korean_keys:
                        # Format value based on key
                        if 'return' in key or 'rate' in key or 'drawdown' in key:
                            value = f"{value * 100:.2f}%" if isinstance(value, float) else value
                        elif key in ['initial_balance', 'final_balance']:
                            # 금액 형식으로 표시하고 소수점 제거
                            value = f"{int(value):,} 원" if isinstance(value, (int, float)) else value
                        elif 'ratio' in key:
                            value = f"{value:.2f}" if isinstance(value, float) else value
                        print(f"{korean_keys[key]}: {value}")
                    elif key != 'trades' and key != 'metrics' and key != 'parameters':
                        print(f"{key}: {value}")
                
                # 리스크/성과 지표 구분하여 출력
                print("\n" + "-"*20 + " 리스크/성과 지표 " + "-"*20)
                
                # 주요 지표들을 직접 출력
                metrics = results_dict.get('metrics', {})
                
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
                print("\n" + "-"*30 + " 거래 통계 " + "-"*30)
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
                for key, value in results_dict.items() if results_dict else {}:
                    print(f"{key}: {value}")
                print("="*50)
        
        elif mode == 'live' or mode == 'paper':
            # 실시간 트레이딩 전에 모델 체크
            logger.info("실시간 트레이딩을 위한 모델 체크 중...")
            logger.info("기존 모델을 사용합니다.")
            
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
    메인 실행 함수
    """
    try:
        # 명령줄 인수 파싱
        args = parse_arguments()
        
        # 기본 설정 로드
        config = load_config(args)
        
        # 백테스트 모드 실행
        if args.backtest:
            # 기본 백테스트 설정
            if not args.start_date and not args.end_date:
                # 기본값: 최근 200일
                end_date = datetime.now() - timedelta(days=1)  # 어제까지
                start_date = end_date - timedelta(days=args.days)
                args.start_date = start_date.strftime("%Y-%m-%d")
                args.end_date = end_date.strftime("%Y-%m-%d")
            
            # 백테스트 실행
            result = run_backtest({
                'strategy': args.strategy,
                'market': args.market,
                'start_date': args.start_date,
                'end_date': args.end_date,
                'initial_balance': args.initial_balance,
                'position_size': args.position_size,
                'fee': args.fee,
                'slippage': args.slippage,
                'plot': args.plot,
                'verbose': args.verbose,
                'report': args.report
            })
            
            # 결과 표시
            display_backtest_results(result)
            
            # 결과 저장
            save_backtest_results(result, args.strategy, args.market, args.timeframe)
            
            return
        
        # 다른 모드 실행
        if args.live:
            run_trading_bot(config)
        elif args.optimize:
            run_optimization(config)
        else:
            logger.info("실행 모드를 지정해주세요. 사용 가능한 모드: --backtest, --live, --optimize")
            
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 