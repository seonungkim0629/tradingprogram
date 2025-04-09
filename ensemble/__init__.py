"""
앙상블 시스템 모듈 (Ensemble System Module)

이 모듈은 다양한 모델을 결합하여 더 강력한 예측 시스템을 구축하는 앙상블 기능을 제공합니다.
모델 가중치 조정, 성능 평가, 시장 문맥 분석 등 다양한 기능을 통해 예측 정확도를 향상시킵니다.
"""

# 코어 앙상블 클래스
from ensemble.ensemble_core import EnsembleBase, VotingEnsemble, StackingEnsemble, HybridEnsemble
from ensemble.ensemble import TradingEnsemble

# 지원 모듈
from ensemble.scoring import calculate_model_performance, evaluate_ensemble_performance
from ensemble.weights import adjust_weights_based_on_performance, calculate_optimal_weights
from ensemble.market_context import adjust_weights_by_market_condition, extract_market_features
from ensemble.combiners import weighted_average_combiner, adaptive_weights_combiner, majority_vote_combiner

# 외부로 노출할 클래스와 함수
__all__ = [
    'TradingEnsemble',
    'EnsembleBase',
    'VotingEnsemble',
    'StackingEnsemble',
    'HybridEnsemble',
    'calculate_model_performance',
    'evaluate_ensemble_performance',
    'adjust_weights_based_on_performance',
    'adjust_weights_by_market_condition',
    'weighted_average_combiner',
    'adaptive_weights_combiner'
] 