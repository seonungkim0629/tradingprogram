"""
특성 관리와 검증을 위한 유틸리티 모듈

이 모듈은 머신러닝 모델의 특성 관리를 위한 도구를 제공합니다.
특성 목록 저장, 로드, 검증 및 정렬 기능을 포함합니다.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime

from utils.logging import get_logger

logger = get_logger(__name__)

class FeatureManager:
    """
    머신러닝 모델의 특성 관리를 위한 클래스
    
    특성 목록 저장, 로드, 검증 및 정렬 기능을 제공합니다.
    """
    
    def __init__(self, 
                model_name: str, 
                model_version: str = "1.0.0", 
                strict_mode: bool = False,
                features_dir: str = "models/features"):
        """
        FeatureManager 초기화
        
        Args:
            model_name (str): 모델 이름
            model_version (str): 모델 버전
            strict_mode (bool): 엄격 모드 활성화 여부 (True이면 특성 불일치 시 예외 발생)
            features_dir (str): 특성 파일 저장 디렉토리
        """
        self.model_name = model_name
        self.model_version = model_version
        self.strict_mode = strict_mode
        self.features_dir = features_dir
        self.expected_features = None
        self.expected_features_count = None
        self.logger = logger
        
        # 특성 파일 경로 설정
        os.makedirs(features_dir, exist_ok=True)
        version_dir = model_version.replace('.', '_')
        self.features_path = os.path.join(features_dir, f"{model_name}_{version_dir}_features.json")
        
        # 기존 특성 목록 로드 시도
        self._load_features()
    
    def _load_features(self) -> bool:
        """
        저장된 특성 목록 로드
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            if os.path.exists(self.features_path):
                with open(self.features_path, 'r') as f:
                    features_data = json.load(f)
                    
                if isinstance(features_data, dict):
                    self.expected_features = features_data.get('features', [])
                    self.metadata = features_data.get('metadata', {})
                elif isinstance(features_data, list):
                    # 레거시 포맷 지원 (단순 목록)
                    self.expected_features = features_data
                    self.metadata = {}
                
                if self.expected_features:
                    self.expected_features_count = len(self.expected_features)
                    self.logger.info(f"특성 목록 로드됨: {self.expected_features_count}개 ({self.features_path})")
                    return True
                else:
                    self.logger.warning(f"특성 목록이 비어 있습니다: {self.features_path}")
            else:
                self.logger.info(f"특성 목록 파일이 없습니다: {self.features_path}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"특성 목록 로드 중 오류: {str(e)}")
            return False
    
    def save_features(self, feature_names: List[str], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        특성 이름 목록을 파일로 저장
        
        Args:
            feature_names (List[str]): 저장할 특성 이름 목록
            metadata (Optional[Dict[str, Any]]): 함께 저장할 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            if not feature_names:
                self.logger.warning("저장할 특성 목록이 비어 있습니다.")
                return False
            
            # 메타데이터 준비
            if metadata is None:
                metadata = {}
            
            if 'saved_at' not in metadata:
                metadata['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if 'model_version' not in metadata:
                metadata['model_version'] = self.model_version
            
            # 특성 및 메타데이터 저장
            features_data = {
                'features': feature_names,
                'metadata': metadata
            }
            
            os.makedirs(os.path.dirname(self.features_path), exist_ok=True)
            with open(self.features_path, 'w') as f:
                json.dump(features_data, f, indent=2)
            
            self.expected_features = feature_names
            self.expected_features_count = len(feature_names)
            self.metadata = metadata
            
            self.logger.info(f"특성 목록 저장됨: {len(feature_names)}개 ({self.features_path})")
            return True
            
        except Exception as e:
            self.logger.error(f"특성 목록 저장 중 오류: {str(e)}")
            return False
    
    def validate_features(self, 
                        features: Union[List[str], pd.DataFrame, np.ndarray], 
                        threshold: float = 0.1) -> Dict[str, Any]:
        """
        입력 특성이 예상 특성과 일치하는지 검증
        
        Args:
            features (Union[List[str], pd.DataFrame, np.ndarray]): 검증할 특성 목록 또는 데이터
            threshold (float): 불일치 허용 비율 (0.1 = 10% 불일치까지 허용)
            
        Returns:
            Dict[str, Any]: 검증 결과
            {
                'is_valid': bool,
                'missing_features': List[str],
                'extra_features': List[str],
                'missing_ratio': float,
                'extra_ratio': float,
                'order_matched': bool
            }
            
        Raises:
            ValueError: strict_mode가 True이고 특성 불일치가 threshold를 초과할 경우
        """
        if self.expected_features is None:
            self.logger.warning("예상 특성 목록이 없습니다. 검증을 건너뜁니다.")
            return {'is_valid': False, 'reason': 'No expected features'}
        
        # 입력에서 특성 이름 추출
        current_features = []
        
        if isinstance(features, pd.DataFrame):
            current_features = features.columns.tolist()
        elif isinstance(features, np.ndarray):
            # NumPy 배열은 특성 이름이 없으므로 특성 수만 확인
            feature_count = features.shape[1] if len(features.shape) > 1 else 1
            if feature_count != self.expected_features_count:
                return {
                    'is_valid': False,
                    'reason': f'Feature count mismatch: expected {self.expected_features_count}, got {feature_count}',
                    'expected_count': self.expected_features_count,
                    'actual_count': feature_count
                }
            return {'is_valid': True}
        elif isinstance(features, list):
            current_features = features
        else:
            self.logger.error(f"지원되지 않는 입력 타입: {type(features)}")
            return {'is_valid': False, 'reason': f'Unsupported input type: {type(features)}'}
        
        # 특성 차이 계산
        missing_features = [f for f in self.expected_features if f not in current_features]
        extra_features = [f for f in current_features if f not in self.expected_features]
        
        # 불일치 비율 계산
        missing_ratio = len(missing_features) / len(self.expected_features) if self.expected_features else 0
        extra_ratio = len(extra_features) / len(self.expected_features) if self.expected_features else 0
        
        # 특성 순서 확인
        order_matched = True
        if len(current_features) == len(self.expected_features):
            for i, (expected, current) in enumerate(zip(self.expected_features, current_features)):
                if expected != current:
                    order_matched = False
                    self.logger.debug(f"특성 순서 불일치: 위치 {i}, 예상: {expected}, 실제: {current}")
                    break
        else:
            order_matched = False
        
        # 결과 준비
        result = {
            'is_valid': len(missing_features) == 0 and missing_ratio <= threshold,
            'missing_features': missing_features[:5] if missing_features else [],  # Top 5만 표시
            'missing_count': len(missing_features),
            'extra_features': extra_features[:5] if extra_features else [],  # Top 5만 표시
            'extra_count': len(extra_features),
            'missing_ratio': missing_ratio,
            'extra_ratio': extra_ratio,
            'order_matched': order_matched
        }
        
        # 불일치가 심각한 경우 로그
        if missing_ratio > threshold:
            msg = (f"심각한 특성 불일치: 예상 {len(self.expected_features)}개 중 {len(missing_features)}개 누락 "
                  f"({missing_ratio:.1%}, 허용 임계치: {threshold:.1%})")
            self.logger.warning(msg)
            
            # 누락 특성 로깅
            if missing_features:
                missing_str = ", ".join(missing_features[:5])
                if len(missing_features) > 5:
                    missing_str += f" 외 {len(missing_features) - 5}개"
                self.logger.warning(f"누락된 특성 (Top 5): {missing_str}")
            
            # 추가 특성 로깅
            if extra_features:
                extra_str = ", ".join(extra_features[:5])
                if len(extra_features) > 5:
                    extra_str += f" 외 {len(extra_features) - 5}개"
                self.logger.warning(f"추가된 특성 (Top 5): {extra_str}")
            
            # strict_mode에서는 예외 발생
            if self.strict_mode:
                raise ValueError(f"Strict Mode 활성화: {msg}")
        
        return result
    
    def align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        입력 데이터프레임의 특성을 예상 특성에 맞게 정렬
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            
        Returns:
            pd.DataFrame: 정렬된 데이터프레임
            
        Raises:
            ValueError: strict_mode가 True이고 특성 불일치가 심각한 경우
        """
        if self.expected_features is None:
            self.logger.warning("예상 특성 목록이 없습니다. 현재 특성 순서를 저장합니다.")
            self.save_features(df.columns.tolist())
            return df
        
        # 특성 검증
        validation = self.validate_features(df)
        
        # 특성 정렬
        self.logger.info(f"특성 정렬: {df.shape[1]}개 -> {len(self.expected_features)}개")
        aligned_df = df.reindex(columns=self.expected_features, fill_value=0)
        
        # 누락 특성 로그
        if validation.get('missing_count', 0) > 0:
            missing_str = ", ".join(validation.get('missing_features', []))
            if validation.get('missing_count', 0) > 5:
                missing_str += f" 외 {validation.get('missing_count') - 5}개"
            self.logger.warning(f"누락된 특성: {missing_str}")
        
        # 추가 특성 로그
        if validation.get('extra_count', 0) > 0:
            extra_str = ", ".join(validation.get('extra_features', []))
            if validation.get('extra_count', 0) > 5:
                extra_str += f" 외 {validation.get('extra_count') - 5}개"
            self.logger.warning(f"무시된 추가 특성: {extra_str}")
        
        return aligned_df
    
    def adjust_feature_count(self, X: np.ndarray, expected_count: Optional[int] = None) -> np.ndarray:
        """
        특성 배열의 열 수를 예상 특성 수에 맞게 조정
        
        Args:
            X (np.ndarray): 입력 특성 배열
            expected_count (Optional[int]): 기대하는 특성 수 (제공하지 않으면 self.expected_features_count 사용)
            
        Returns:
            np.ndarray: 조정된 특성 배열
        """
        if expected_count is None:
            if self.expected_features_count is None:
                self.logger.warning("예상 특성 수가 설정되지 않았습니다. 원본 배열을 반환합니다.")
                return X
            expected_count = self.expected_features_count
        
        current_count = X.shape[1] if len(X.shape) > 1 else 1
        
        if current_count == expected_count:
            return X
            
        self.logger.info(f"특성 수 조정: {current_count} -> {expected_count}")
        
        if current_count < expected_count:
            # 부족한 특성 추가 (0으로 채움)
            self.logger.warning(f"특성 수가 부족함: {current_count} < {expected_count}, 부족한 특성을 0으로 채웁니다.")
            padding = np.zeros((X.shape[0], expected_count - current_count))
            return np.hstack((X, padding))
        else:
            # 초과 특성 제거
            self.logger.warning(f"특성 수가 초과함: {current_count} > {expected_count}, 처음 {expected_count}개만 사용합니다.")
            return X[:, :expected_count]
    
    def log_feature_drift(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        특성 드리프트 분석 및 로깅
        
        Args:
            df (pd.DataFrame): 분석할 데이터프레임
            
        Returns:
            Dict[str, Any]: 드리프트 분석 결과
        """
        if self.expected_features is None:
            self.logger.warning("예상 특성 목록이 없어 드리프트 분석을 건너뜁니다.")
            return {'status': 'skip', 'reason': 'No expected features'}
        
        current_features = df.columns.tolist()
        
        # 누락 및 추가 특성 찾기
        missing_features = [f for f in self.expected_features if f not in current_features]
        extra_features = [f for f in current_features if f not in self.expected_features]
        
        # 특성 값 분포 분석 (수치형 특성만)
        distribution_changes = {}
        common_features = [f for f in self.expected_features if f in current_features]
        
        for feature in common_features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                try:
                    stats = {
                        'mean': float(df[feature].mean()),
                        'std': float(df[feature].std()),
                        'min': float(df[feature].min()),
                        'max': float(df[feature].max()),
                        'missing_pct': float(df[feature].isna().mean())
                    }
                    distribution_changes[feature] = stats
                except Exception as e:
                    self.logger.debug(f"특성 {feature} 통계 계산 중 오류: {str(e)}")
        
        # 결과 생성
        drift_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'expected_feature_count': len(self.expected_features),
            'current_feature_count': len(current_features),
            'missing_features': missing_features,
            'missing_count': len(missing_features),
            'extra_features': extra_features,
            'extra_count': len(extra_features),
            'distributions': distribution_changes
        }
        
        # 로그 출력
        if missing_features:
            missing_str = ", ".join(missing_features[:5])
            if len(missing_features) > 5:
                missing_str += f" 외 {len(missing_features) - 5}개"
            self.logger.warning(f"특성 드리프트 - 누락된 특성 (Top 5): {missing_str}")
        
        if extra_features:
            extra_str = ", ".join(extra_features[:5])
            if len(extra_features) > 5:
                extra_str += f" 외 {len(extra_features) - 5}개"
            self.logger.warning(f"특성 드리프트 - 추가된 특성 (Top 5): {extra_str}")
        
        return drift_report
    
    @staticmethod
    def create_manager_for_model(model_obj: Any) -> 'FeatureManager':
        """
        모델 객체로부터 FeatureManager 인스턴스 생성
        
        Args:
            model_obj (Any): 모델 객체 (name, version 속성 필요)
            
        Returns:
            FeatureManager: 생성된 FeatureManager 인스턴스
        """
        try:
            model_name = getattr(model_obj, 'name', 'unknown_model')
            model_version = getattr(model_obj, 'version', '1.0.0')
            strict_mode = getattr(model_obj, 'strict_mode', False)
            
            # 특성 목록이 이미 있는 경우 가져와서 새 매니저에 설정
            manager = FeatureManager(model_name, model_version, strict_mode)
            
            # 모델에 특성 목록이 있으면 설정
            if hasattr(model_obj, 'feature_names') and model_obj.feature_names:
                manager.save_features(model_obj.feature_names)
            elif hasattr(model_obj, 'selected_feature_names') and model_obj.selected_feature_names:
                manager.save_features(model_obj.selected_feature_names)
            
            return manager
            
        except Exception as e:
            logger.error(f"모델에서 FeatureManager 생성 중 오류: {str(e)}")
            # 기본 매니저 반환
            return FeatureManager('unknown_model') 