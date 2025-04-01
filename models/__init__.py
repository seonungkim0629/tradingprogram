"""
Models Package for Bitcoin Trading System

This package contains all the ML models used in the trading system,
including time-series models, classification models and reinforcement learning models.
"""

import os
from typing import Dict, Type, List

# 모델 임포트
from models.base import ModelBase, ClassificationModel
from models.gru import GRUPriceModel, GRUDirectionModel
from models.random_forest import RandomForestDirectionModel
from models.ensemble import VotingEnsemble

__all__ = [
    'ModelBase',
    'ClassificationModel',
    'GRUPriceModel',
    'GRUDirectionModel', 
    'RandomForestDirectionModel',
    'VotingEnsemble',
    'get_available_models'
]

def get_available_models() -> Dict[str, Type[ModelBase]]:
    """
    사용 가능한 모든 모델의 딕셔너리 반환
    
    Returns:
        Dict[str, Type[ModelBase]]: 모델 이름과 클래스 딕셔너리
    """
    return {
        'RandomForestDirectionModel': RandomForestDirectionModel,
        'GRUDirectionModel': GRUDirectionModel,
        'GRUPriceModel': GRUPriceModel,
        'VotingEnsemble': VotingEnsemble
    } 