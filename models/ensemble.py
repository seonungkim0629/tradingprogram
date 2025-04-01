"""
Ensemble Models for Bitcoin Trading Bot

This module implements ensemble models that combine multiple models for better predictions.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from datetime import datetime
import json
import traceback
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.base import EnsembleModel, ModelBase, ClassificationModel, RegressionModel
from models.gru import GRUDirectionModel
from models.random_forest import RandomForestDirectionModel
from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class VotingEnsemble(EnsembleModel):
    """Voting ensemble that combines multiple classification models"""
    
    def __init__(self, 
                name: str = "VotingEnsemble", 
                version: str = "1.0.0",
                models: Optional[List[ClassificationModel]] = None,
                weights: Optional[List[float]] = None,
                voting: str = 'soft'):
        """
        Initialize Voting Ensemble model
        
        Args:
            name (str): Model name
            version (str): Model version
            models (Optional[List[ClassificationModel]]): List of classification models
            weights (Optional[List[float]]): Weights for each model
            voting (str): Voting strategy ('hard' or 'soft')
        """
        super().__init__(name, version)
        
        self.models = models or []
        self.weights = weights
        self.voting = voting
        
        # Normalize weights if provided
        if self.weights is not None:
            self.weights = self._normalize_weights(self.weights)
        elif self.models:
            # Default: equal weights
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        # Store parameters
        self.params = {
            'voting': voting
        }
        
        self.logger.info(f"Initialized {self.name} model with {len(self.models)} sub-models")
    
    def add_model(self, 
                 model: ClassificationModel, 
                 weight: float = 1.0) -> None:
        """
        Add a model to the ensemble
        
        Args:
            model (ClassificationModel): Model to add
            weight (float): Weight for the model
        """
        if not isinstance(model, ClassificationModel):
            raise TypeError("Model must be a ClassificationModel")
        
        self.models.append(model)
        
        # Update weights
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)
            self.weights = self._normalize_weights(self.weights)
        
        self.logger.info(f"Added model {model.name} to ensemble with weight {weight}")
    
    def remove_model(self, index: int) -> ClassificationModel:
        """
        Remove a model from the ensemble
        
        Args:
            index (int): Index of model to remove
            
        Returns:
            ClassificationModel: Removed model
        """
        if index < 0 or index >= len(self.models):
            raise IndexError(f"Index {index} out of range for ensemble with {len(self.models)} models")
        
        model = self.models.pop(index)
        
        # Update weights
        if self.weights is not None:
            self.weights.pop(index)
            if self.weights:
                self.weights = self._normalize_weights(self.weights)
        
        self.logger.info(f"Removed model {model.name} from ensemble")
        return model
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray, 
             **kwargs) -> Dict[str, Any]:
        """
        Train each model in the ensemble
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            **kwargs: Additional parameters
                
        Returns:
            Dict[str, Any]: Training metrics
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        start_time = pd.Timestamp.now()
        self.logger.info(f"Training ensemble with {len(self.models)} models")
        
        # Train each model
        model_metrics = []
        for i, model in enumerate(self.models):
            self.logger.info(f"Training model {i+1}/{len(self.models)}: {model.name}")
            metrics = model.train(X_train, y_train, **kwargs)
            model_metrics.append(metrics)
        
        # Calculate ensemble metrics on training data
        y_pred = self.predict(X_train)
        
        train_accuracy = accuracy_score(y_train, y_pred)
        train_precision = precision_score(y_train, y_pred, average='weighted')
        train_recall = recall_score(y_train, y_pred, average='weighted')
        train_f1 = f1_score(y_train, y_pred, average='weighted')
        
        # Store metrics
        metrics = {
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'model_metrics': model_metrics,
            'training_time': (pd.Timestamp.now() - start_time).total_seconds()
        }
        
        self.metrics.update(metrics)
        self.is_trained = True
        self.last_update = pd.Timestamp.now()
        
        self.logger.info(f"Ensemble training completed. Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        return metrics
    
    def predict(self, 
               X: np.ndarray,
               **kwargs) -> np.ndarray:
        """
        Make predictions with the ensemble
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional parameters
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        if not all(model.is_trained for model in self.models):
            self.logger.warning("Not all models in ensemble are trained")
        
        if self.voting == 'hard':
            # Hard voting: majority rule
            predictions = np.array([model.predict(X) for model in self.models])
            weighted_votes = np.zeros((len(self.models), len(X), len(self.models[0].classes_)))
            
            for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
                for j, p in enumerate(pred):
                    weighted_votes[i, j, p] += weight
            
            # Sum votes across models
            summed_votes = np.sum(weighted_votes, axis=0)
            
            # Get class with highest vote
            return np.argmax(summed_votes, axis=1)
        else:
            # Soft voting: weight probabilities
            probas = self.predict_proba(X)
            return np.argmax(probas, axis=1)
    
    def predict_proba(self, 
                    X: np.ndarray, 
                    **kwargs) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional parameters
            
        Returns:
            np.ndarray: Class probabilities
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get probabilities from each model
        all_probas = []
        for model, weight in zip(self.models, self.weights):
            probas = model.predict_proba(X)
            
            # 확률 배열 형태 표준화
            if probas is None or len(probas) == 0:
                continue
                
            # 확률 형태가 1D 배열인 경우 2D로 변환 
            # 예시: [0.7] -> [[0.3, 0.7]]
            if len(probas.shape) == 1:
                # 1차원 배열을 2차원으로 변환 (이진 분류로 가정)
                probas_2d = np.zeros((probas.shape[0], 2))
                probas_2d[:, 1] = probas  # 두 번째 클래스 (긍정적/상승) 확률
                probas_2d[:, 0] = 1 - probas  # 첫 번째 클래스 (부정적/하락) 확률
                probas = probas_2d
            
            # 모델이 단일 클래스에 대한 확률만 반환하는 경우 (n_samples, 1)
            elif len(probas.shape) == 2 and probas.shape[1] == 1:
                probas_2d = np.zeros((probas.shape[0], 2))
                probas_2d[:, 1] = probas[:, 0]
                probas_2d[:, 0] = 1 - probas[:, 0]
                probas = probas_2d
                
            all_probas.append(probas * weight)
        
        if not all_probas:
            # 모든 모델이 실패한 경우
            return np.zeros((X.shape[0], 2))
        
        # Combine probabilities using weights
        if self.voting == 'soft':
            # For soft voting, weight and combine probabilities
            ensemble_proba = np.zeros_like(all_probas[0])
            for proba in all_probas:
                ensemble_proba += proba
            
            # Normalize to ensure probabilities sum to 1
            row_sums = ensemble_proba.sum(axis=1, keepdims=True)
            normalized_proba = ensemble_proba / np.maximum(row_sums, 1e-10)
            
            return normalized_proba
        else:
            # For hard voting, count votes
            class_predictions = []
            for i, model in enumerate(self.models):
                preds = model.predict(X)
                preds = np.asarray(preds)
                if len(preds.shape) == 1:
                    preds = preds.reshape(-1, 1)
                class_predictions.append(preds * self.weights[i])
            
            # Combine votes and convert to probabilities
            votes = np.sum(class_predictions, axis=0)
            proba = np.zeros((X.shape[0], 2))  # Assuming binary classification
            proba[:, 1] = votes / np.sum(self.weights)
            proba[:, 0] = 1 - proba[:, 1]
            
            return proba
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        Evaluate the ensemble
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        if not all(model.is_trained for model in self.models):
            self.logger.warning("Not all models in ensemble are trained")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store metrics
        metrics = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        
        # Evaluate each model
        model_metrics = []
        for model in self.models:
            model_metrics.append(model.evaluate(X_test, y_test, **kwargs))
        
        metrics['model_metrics'] = model_metrics
        
        self.metrics.update(metrics)
        self.logger.info(f"Ensemble evaluation completed. Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        return metrics
    
    def save(self, 
            directory: Optional[str] = None, 
            save_models: bool = True) -> str:
        """
        Save the ensemble model
        
        Args:
            directory (Optional[str]): Directory to save the ensemble
            save_models (bool): Whether to save individual models
            
        Returns:
            str: Path to saved ensemble
        """
        if directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(self.model_dir, f"{self.name}_{timestamp}")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save ensemble metadata
        metadata = {
            'name': self.name,
            'version': self.version,
            'params': self.params,
            'metrics': self.metrics,
            'weights': self.weights,
            'model_names': [model.name for model in self.models],
            'last_update': datetime.now().isoformat()
        }
        
        with open(os.path.join(directory, 'ensemble_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save individual models
        if save_models:
            models_dir = os.path.join(directory, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            for i, model in enumerate(self.models):
                model_path = os.path.join(models_dir, f"model_{i}")
                model.save(model_path)
        
        self.logger.info(f"Ensemble model saved to {directory}")
        return directory
    
    @classmethod
    def load(cls, 
            directory: str, 
            load_models: bool = True) -> 'VotingEnsemble':
        """
        Load an ensemble model
        
        Args:
            directory (str): Directory containing the saved ensemble
            load_models (bool): Whether to load individual models
            
        Returns:
            VotingEnsemble: Loaded ensemble model
        """
        try:
            # Load ensemble metadata
            with open(os.path.join(directory, 'ensemble_metadata.json'), 'r') as f:
                metadata = json.load(f)
            
            # Create ensemble instance
            ensemble = cls(
                name=metadata['name'],
                version=metadata['version'],
                voting=metadata['params']['voting']
            )
            
            # Set ensemble attributes
            ensemble.weights = metadata['weights']
            ensemble.metrics = metadata['metrics']
            ensemble.is_trained = True
            ensemble.last_update = datetime.fromisoformat(metadata['last_update'])
            
            # Load individual models
            if load_models:
                models_dir = os.path.join(directory, 'models')
                for i in range(len(metadata['model_names'])):
                    model_path = os.path.join(models_dir, f"model_{i}")
                    model = ModelBase.load(model_path)
                    ensemble.models.append(model)
            
            logger.info(f"Ensemble model loaded from {directory}")
            return ensemble
        except Exception as e:
            logger.error(f"Error loading ensemble model: {str(e)}")
            raise

    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """
        Normalize weights to sum to 1.0
        
        Args:
            weights (List[float]): List of weights
            
        Returns:
            List[float]: Normalized weights
        """
        if not weights:
            return []
            
        total = sum(weights)
        if total > 0:
            return [w / total for w in weights]
        return weights


class StackingEnsemble(EnsembleModel):
    """Stacking ensemble that trains a meta-model on the predictions of base models"""
    
    def __init__(self, 
                name: str = "StackingEnsemble", 
                version: str = "1.0.0",
                base_models: Optional[List[ModelBase]] = None,
                meta_model: Optional[ModelBase] = None):
        """
        Initialize Stacking Ensemble model
        
        Args:
            name (str): Model name
            version (str): Model version
            base_models (Optional[List[ModelBase]]): List of base models
            meta_model (Optional[ModelBase]): Meta-model to combine base models
        """
        super().__init__(name, version)
        
        self.base_models = base_models or []
        self.meta_model = meta_model
        
        # Store parameters
        self.params = {}
        
        self.logger.info(f"Initialized {self.name} model with {len(self.base_models)} base models")
    
    def add_model(self, model: ModelBase) -> None:
        """
        Add a base model to the ensemble
        
        Args:
            model (ModelBase): Model to add
        """
        self.base_models.append(model)
        self.logger.info(f"Added base model {model.name} to ensemble")
    
    def set_meta_model(self, model: ModelBase) -> None:
        """
        Set the meta-model
        
        Args:
            model (ModelBase): Meta-model to set
        """
        self.meta_model = model
        self.logger.info(f"Set meta-model to {model.name}")
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray, 
             X_val: np.ndarray,
             y_val: np.ndarray,
             **kwargs) -> Dict[str, Any]:
        """
        Train the stacking ensemble
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features for training meta-model
            y_val (np.ndarray): Validation targets for training meta-model
            **kwargs: Additional parameters
                
        Returns:
            Dict[str, Any]: Training metrics
        """
        if not self.base_models:
            raise ValueError("No base models in ensemble")
        
        if self.meta_model is None:
            raise ValueError("Meta-model not set")
        
        start_time = pd.Timestamp.now()
        self.logger.info(f"Training stacking ensemble with {len(self.base_models)} base models")
        
        # Train base models
        base_model_metrics = []
        for i, model in enumerate(self.base_models):
            self.logger.info(f"Training base model {i+1}/{len(self.base_models)}: {model.name}")
            metrics = model.train(X_train, y_train, **kwargs)
            base_model_metrics.append(metrics)
        
        # Generate meta-features
        meta_features_train = self._generate_meta_features(X_train)
        meta_features_val = self._generate_meta_features(X_val)
        
        # Train meta-model
        self.logger.info(f"Training meta-model: {self.meta_model.name}")
        meta_metrics = self.meta_model.train(meta_features_train, y_train, **kwargs)
        
        # Evaluate on validation set
        val_predictions = self.meta_model.predict(meta_features_val)
        
        # Calculate metrics based on model type
        if isinstance(self.meta_model, ClassificationModel):
            val_accuracy = accuracy_score(y_val, val_predictions)
            val_precision = precision_score(y_val, val_predictions, average='weighted')
            val_recall = recall_score(y_val, val_predictions, average='weighted')
            val_f1 = f1_score(y_val, val_predictions, average='weighted')
            
            val_metrics = {
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            }
            
            self.logger.info(f"Validation metrics - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        else:
            val_mse = mean_squared_error(y_val, val_predictions)
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(y_val, val_predictions)
            val_r2 = r2_score(y_val, val_predictions)
            
            val_metrics = {
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2
            }
            
            self.logger.info(f"Validation metrics - RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        
        # Store metrics
        metrics = {
            'base_model_metrics': base_model_metrics,
            'meta_model_metrics': meta_metrics,
            'validation_metrics': val_metrics,
            'training_time': (pd.Timestamp.now() - start_time).total_seconds()
        }
        
        self.metrics.update(metrics)
        self.is_trained = True
        self.last_update = pd.Timestamp.now()
        
        self.logger.info(f"Stacking ensemble training completed")
        return metrics
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate meta-features by getting predictions from base models
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Meta-features for meta-model
        """
        # Initialize meta-features
        if isinstance(self.base_models[0], ClassificationModel):
            # For classification models, use probabilities
            # Get the number of classes from the first model
            num_classes = len(self.base_models[0].classes_)
            meta_features = np.zeros((X.shape[0], len(self.base_models) * num_classes))
            
            for i, model in enumerate(self.base_models):
                # Get probabilities for each class
                probas = model.predict_proba(X)
                # Add probabilities to meta-features
                meta_features[:, i*num_classes:(i+1)*num_classes] = probas
        else:
            # For regression models, use predictions
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                # Get predictions
                preds = model.predict(X)
                # Add predictions to meta-features
                meta_features[:, i] = preds
        
        return meta_features
    
    def predict(self, 
               X: np.ndarray,
               **kwargs) -> np.ndarray:
        """
        Make predictions with the stacking ensemble
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional parameters
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.base_models:
            raise ValueError("No base models in ensemble")
        
        if self.meta_model is None:
            raise ValueError("Meta-model not set")
        
        if not all(model.is_trained for model in self.base_models) or not self.meta_model.is_trained:
            self.logger.warning("Not all models in ensemble are trained")
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Make predictions with meta-model
        return self.meta_model.predict(meta_features, **kwargs)
    
    def predict_proba(self, 
                    X: np.ndarray, 
                    **kwargs) -> np.ndarray:
        """
        Predict class probabilities (for classification ensembles)
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional parameters
            
        Returns:
            np.ndarray: Class probabilities
        """
        if not isinstance(self.meta_model, ClassificationModel):
            raise TypeError("Meta-model must be a ClassificationModel for predict_proba")
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Make probability predictions with meta-model
        return self.meta_model.predict_proba(meta_features, **kwargs)
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        Evaluate the stacking ensemble
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        if not self.base_models:
            raise ValueError("No base models in ensemble")
        
        if self.meta_model is None:
            raise ValueError("Meta-model not set")
        
        if not all(model.is_trained for model in self.base_models) or not self.meta_model.is_trained:
            self.logger.warning("Not all models in ensemble are trained")
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X_test)
        
        # Evaluate meta-model
        meta_metrics = self.meta_model.evaluate(meta_features, y_test, **kwargs)
        
        # Evaluate base models
        base_metrics = []
        for model in self.base_models:
            base_metrics.append(model.evaluate(X_test, y_test, **kwargs))
        
        # Combine metrics
        metrics = {
            'meta_model_metrics': meta_metrics,
            'base_model_metrics': base_metrics
        }
        
        self.metrics.update(metrics)
        
        if isinstance(self.meta_model, ClassificationModel):
            self.logger.info(f"Stacking ensemble evaluation completed. Accuracy: {meta_metrics['test_accuracy']:.4f}")
        else:
            self.logger.info(f"Stacking ensemble evaluation completed. RMSE: {meta_metrics['test_rmse']:.4f}")
        
        return metrics
    
    def save(self, 
            directory: Optional[str] = None, 
            save_models: bool = True) -> str:
        """
        Save the stacking ensemble
        
        Args:
            directory (Optional[str]): Directory to save the ensemble
            save_models (bool): Whether to save individual models
            
        Returns:
            str: Path to saved ensemble
        """
        if directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(self.model_dir, f"{self.name}_{timestamp}")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save ensemble metadata
        metadata = {
            'name': self.name,
            'version': self.version,
            'params': self.params,
            'metrics': self.metrics,
            'base_model_names': [model.name for model in self.base_models],
            'meta_model_name': self.meta_model.name if self.meta_model else None,
            'last_update': datetime.now().isoformat()
        }
        
        with open(os.path.join(directory, 'ensemble_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save individual models
        if save_models:
            base_models_dir = os.path.join(directory, 'base_models')
            os.makedirs(base_models_dir, exist_ok=True)
            
            for i, model in enumerate(self.base_models):
                model_path = os.path.join(base_models_dir, f"model_{i}")
                model.save(model_path)
            
            if self.meta_model:
                meta_model_path = os.path.join(directory, 'meta_model')
                self.meta_model.save(meta_model_path)
        
        self.logger.info(f"Stacking ensemble saved to {directory}")
        return directory
    
    @classmethod
    def load(cls, 
            directory: str, 
            load_models: bool = True) -> 'StackingEnsemble':
        """
        Load a stacking ensemble
        
        Args:
            directory (str): Directory containing the saved ensemble
            load_models (bool): Whether to load individual models
            
        Returns:
            StackingEnsemble: Loaded stacking ensemble
        """
        try:
            # Load ensemble metadata
            with open(os.path.join(directory, 'ensemble_metadata.json'), 'r') as f:
                metadata = json.load(f)
            
            # Create ensemble instance
            ensemble = cls(
                name=metadata['name'],
                version=metadata['version']
            )
            
            # Set ensemble attributes
            ensemble.metrics = metadata['metrics']
            ensemble.is_trained = True
            ensemble.last_update = datetime.fromisoformat(metadata['last_update'])
            
            # Load individual models
            if load_models:
                # Load base models
                base_models_dir = os.path.join(directory, 'base_models')
                for i in range(len(metadata['base_model_names'])):
                    model_path = os.path.join(base_models_dir, f"model_{i}")
                    model = ModelBase.load(model_path)
                    ensemble.base_models.append(model)
                
                # Load meta-model
                if metadata['meta_model_name']:
                    meta_model_path = os.path.join(directory, 'meta_model')
                    ensemble.meta_model = ModelBase.load(meta_model_path)
            
            logger.info(f"Stacking ensemble loaded from {directory}")
            return ensemble
        except Exception as e:
            logger.error(f"Error loading stacking ensemble: {str(e)}")
            raise 
        
class HierarchicalConfidenceEnsemble(EnsembleModel):
    """
    계층적 신뢰도 가중 앙상블 모델
    
    GRU 모델(중장기 예측, LayerNormalization 적용)과 RandomForest 모델(단기 예측)을 계층적으로 통합하고
    성능 기반 가중치를 동적으로 적용하여 예측 성능을 향상시킵니다.
    """
    
    def __init__(self, 
                name: str = "HierarchicalConfidenceEnsemble", 
                version: str = "1.0.0",
                lstm_model = None,
                rf_model = None,
                tech_indicator_strategy = None,
                lstm_weight: float = 0.55,  # 0.5에서 0.55로 증가 (GRU+LayerNormalization 적용)
                rf_weight: float = 0.45,    # 0.5에서 0.45로 감소
                tech_weight: float = 0.3,
                confidence_threshold: float = 0.55,  # 0.6에서 0.55로 감소 (더 많은 시그널 생성)
                adapt_weights: bool = True):
        """
        계층적 신뢰도 가중 앙상블 모델 초기화
        
        Args:
            name (str): 모델 이름
            version (str): 모델 버전
            lstm_model: GRU 모델 (중장기 예측, LayerNormalization 적용)
            rf_model: RandomForest 모델 (단기 예측)
            tech_indicator_strategy: 기술적 지표 전략
            lstm_weight (float): GRU 모델 초기 가중치
            rf_weight (float): RandomForest 모델 초기 가중치
            tech_weight (float): 기술적 지표 가중치
            confidence_threshold (float): 신뢰도 임계값
            adapt_weights (bool): 가중치 자동 조정 여부
        """
        super().__init__(name, version)
        
        self.lstm_model = lstm_model
        self.rf_model = rf_model
        self.tech_strategy = tech_indicator_strategy
        
        # 모델 가중치 설정
        self.lstm_weight = lstm_weight
        self.rf_weight = rf_weight
        self.tech_weight = tech_weight
        
        # 하이퍼파라미터
        self.confidence_threshold = confidence_threshold
        self.adapt_weights = adapt_weights
        
        # 성능 추적 지표
        self.performance_history = {
            'lstm': {'correct': 0, 'total': 0},
            'rf': {'correct': 0, 'total': 0},
            'tech': {'correct': 0, 'total': 0}
        }
        
        # 모델 파라미터 저장
        self.params = {
            'lstm_weight': lstm_weight,
            'rf_weight': rf_weight,
            'tech_weight': tech_weight,
            'confidence_threshold': confidence_threshold,
            'adapt_weights': adapt_weights
        }
        
        self.logger.info(f"계층적 신뢰도 가중 앙상블 모델 초기화: GRU 가중치={lstm_weight}, RF 가중치={rf_weight}")
    
    def add_lstm_model(self, model):
        """
        GRU 모델 추가
        
        Args:
            model: 추가할 GRU 모델
        """
        self.lstm_model = model
        self.logger.info(f"GRU 모델 추가됨: {model.name}")
    
    def add_rf_model(self, model):
        """
        RandomForest 모델 추가
        
        Args:
            model: 추가할 RandomForest 모델
        """
        self.rf_model = model
        self.logger.info(f"RandomForest 모델 추가됨: {model.name}")
    
    def add_tech_strategy(self, strategy):
        """
        기술적 지표 전략 추가
        
        Args:
            strategy: 추가할 기술적 지표 전략
        """
        self.tech_strategy = strategy
        self.logger.info(f"기술적 지표 전략 추가됨")
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray, 
             X_val: np.ndarray,
             y_val: np.ndarray,
             **kwargs) -> Dict[str, Any]:
        """
        앙상블 모델의 각 구성 모델 학습
        
        Args:
            X_train (np.ndarray): 학습 데이터
            y_train (np.ndarray): 학습 타겟
            X_val (np.ndarray): 검증 데이터
            y_val (np.ndarray): 검증 타겟
            **kwargs: 추가 매개변수
                
        Returns:
            Dict[str, Any]: 학습 지표
        """
        start_time = pd.Timestamp.now()
        self.logger.info(f"계층적 앙상블 모델 학습 시작")
        
        # 모델 유효성 검사
        if self.lstm_model is None:
            raise ValueError("GRU 모델이 설정되지 않았습니다. add_lstm_model()을 먼저 호출하세요.")
        if self.rf_model is None:
            raise ValueError("RandomForest 모델이 설정되지 않았습니다. add_rf_model()을 먼저 호출하세요.")
        
        # GRU 모델 학습
        self.logger.info(f"GRU 모델 학습 중: {self.lstm_model.name}")
        lstm_metrics = self.lstm_model.train(X_train, y_train, X_val=X_val, y_val=y_val, **kwargs)
        
        # RandomForest 모델 학습
        self.logger.info(f"RandomForest 모델 학습 중: {self.rf_model.name}")
        rf_metrics = self.rf_model.train(X_train, y_train, **kwargs)
        
        # 검증 데이터로 각 모델의 성능 평가
        lstm_val_pred = self.lstm_model.predict(X_val)
        rf_val_pred = self.rf_model.predict(X_val)
        
        # 각 모델의 정확도 계산
        lstm_accuracy = accuracy_score(y_val, lstm_val_pred)
        rf_accuracy = accuracy_score(y_val, rf_val_pred)
        
        # 가중치 동적 조정 (성능 기반)
        if self.adapt_weights:
            total_accuracy = lstm_accuracy + rf_accuracy
            if total_accuracy > 0:  # 0으로 나누기 방지
                self.lstm_weight = lstm_accuracy / total_accuracy
                self.rf_weight = rf_accuracy / total_accuracy
                self.logger.info(f"가중치 업데이트됨: GRU={self.lstm_weight:.4f}, RF={self.rf_weight:.4f}")
        
        # 앙상블 예측 성능 평가
        ensemble_pred = self._ensemble_predict(X_val)
        ensemble_accuracy = accuracy_score(y_val, ensemble_pred)
        
        # 각 모델의 성능 지표 기록
        training_metrics = {
            'lstm_metrics': lstm_metrics,
            'rf_metrics': rf_metrics,
            'lstm_accuracy': lstm_accuracy,
            'rf_accuracy': rf_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'lstm_weight': self.lstm_weight,
            'rf_weight': self.rf_weight,
            'training_time': (pd.Timestamp.now() - start_time).total_seconds()
        }
        
        self.metrics.update(training_metrics)
        self.is_trained = True
        self.last_update = pd.Timestamp.now()
        
        self.logger.info(f"앙상블 학습 완료. GRU 정확도: {lstm_accuracy:.4f}, RF 정확도: {rf_accuracy:.4f}, 앙상블 정확도: {ensemble_accuracy:.4f}")
        return training_metrics
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """
        내부 앙상블 예측 함수 (가중치 기반)
        
        Args:
            X (np.ndarray): 입력 데이터
            
        Returns:
            np.ndarray: 앙상블 예측 결과
        """
        # GRU 예측 (클래스 확률)
        lstm_proba = self.lstm_model.predict_proba(X)[:, 1]  # 상승 확률
        
        # RandomForest 예측 (클래스 확률)
        rf_proba = self.rf_model.predict_proba(X)[:, 1]  # 상승 확률
        
        # 가중치 적용하여 결합
        ensemble_proba = lstm_proba * self.lstm_weight + rf_proba * self.rf_weight
        
        # 확률 -> 클래스 변환
        return (ensemble_proba >= 0.5).astype(int)
    
    def predict(self, 
           X: np.ndarray,
           market_data: pd.DataFrame = None,
           apply_tech_filter: bool = True,
           **kwargs) -> np.ndarray:
        """
        앙상블 예측 수행
        
        Args:
            X (np.ndarray): 입력 데이터
            market_data (pd.DataFrame): 시장 데이터 (시장 상황 기반 조정용)
            apply_tech_filter (bool): 기술적 지표 필터 적용 여부
            **kwargs: 추가 매개변수
            
        Returns:
            np.ndarray: 예측 결과
        """
        if not self.is_trained:
            self.logger.warning("모델이 학습되지 않았습니다. 예측 결과가 부정확할 수 있습니다.")
        
        # 시장 상황에 따른 가중치 조정 (선택적)
        if market_data is not None and self.adapt_weights:
            self.adjust_confidence_by_market_condition(X, market_data)
        
        # 기본 앙상블 예측
        ensemble_pred = self._ensemble_predict(X)
        
        # 기술적 지표 필터를 적용할 경우 & 기술적 지표 전략이 존재할 경우
        if apply_tech_filter and self.tech_strategy is not None and market_data is not None:
            # indicators 모듈을 사용하여 기술적 지표 계산
            from data.indicators import calculate_all_indicators
            
            try:
                # 모든 기술적 지표 계산
                market_data_with_indicators = calculate_all_indicators(market_data)
                
                # 기술적 지표를 기반으로 매매 신호 생성
                if hasattr(self.tech_strategy, 'generate_signal'):
                    tech_signal = self.tech_strategy.generate_signal(market_data_with_indicators)
                    
                    # 기술적 지표 신호가 매우 강하고 모델 예측과 다른 경우에만 예측 변경
                    if 'signal' in tech_signal and 'confidence' in tech_signal:
                        tech_direction = 1 if tech_signal['signal'] == 'BUY' else 0 if tech_signal['signal'] == 'SELL' else None
                        tech_confidence = tech_signal['confidence']
                        
                        # 기술적 지표 신호가 강하고 방향이 확실한 경우
                        if tech_direction is not None and tech_confidence > 0.7:
                            # 모델 예측과 기술적 지표 신호가 다른 경우
                            if ensemble_pred[0] != tech_direction:
                                self.logger.info(f"강한 기술적 지표 신호로 예측 변경: {ensemble_pred[0]} -> {tech_direction} (신뢰도: {tech_confidence:.2f})")
                                ensemble_pred[0] = tech_direction
                            else:
                                self.logger.info(f"모델 예측과 기술적 지표 신호 일치: {tech_direction} (신뢰도: {tech_confidence:.2f})")
                else:
                    self.logger.warning("기술적 지표 전략에 generate_signal 메서드가 구현되지 않았습니다.")
            except Exception as e:
                self.logger.error(f"기술적 지표 필터 적용 중 오류 발생: {str(e)}")
        
        return ensemble_pred
    
    def predict_proba(self, 
                    X: np.ndarray, 
                    **kwargs) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional parameters
            
        Returns:
            np.ndarray: Class probabilities
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get probabilities from each model
        all_probas = []
        for model, weight in zip(self.models, self.weights):
            probas = model.predict_proba(X)
            
            # 확률 배열 형태 표준화
            if probas is None or len(probas) == 0:
                continue
                
            # 확률 형태가 1D 배열인 경우 2D로 변환 
            # 예시: [0.7] -> [[0.3, 0.7]]
            if len(probas.shape) == 1:
                # 1차원 배열을 2차원으로 변환 (이진 분류로 가정)
                probas_2d = np.zeros((probas.shape[0], 2))
                probas_2d[:, 1] = probas  # 두 번째 클래스 (긍정적/상승) 확률
                probas_2d[:, 0] = 1 - probas  # 첫 번째 클래스 (부정적/하락) 확률
                probas = probas_2d
            
            # 모델이 단일 클래스에 대한 확률만 반환하는 경우 (n_samples, 1)
            elif len(probas.shape) == 2 and probas.shape[1] == 1:
                probas_2d = np.zeros((probas.shape[0], 2))
                probas_2d[:, 1] = probas[:, 0]
                probas_2d[:, 0] = 1 - probas[:, 0]
                probas = probas_2d
                
            all_probas.append(probas * weight)
        
        if not all_probas:
            # 모든 모델이 실패한 경우
            return np.zeros((X.shape[0], 2))
        
        # Combine probabilities using weights
        if self.voting == 'soft':
            # For soft voting, weight and combine probabilities
            ensemble_proba = np.zeros_like(all_probas[0])
            for proba in all_probas:
                ensemble_proba += proba
            
            # Normalize to ensure probabilities sum to 1
            row_sums = ensemble_proba.sum(axis=1, keepdims=True)
            normalized_proba = ensemble_proba / np.maximum(row_sums, 1e-10)
            
            return normalized_proba
        else:
            # For hard voting, count votes
            class_predictions = []
            for i, model in enumerate(self.models):
                preds = model.predict(X)
                preds = np.asarray(preds)
                if len(preds.shape) == 1:
                    preds = preds.reshape(-1, 1)
                class_predictions.append(preds * self.weights[i])
            
            # Combine votes and convert to probabilities
            votes = np.sum(class_predictions, axis=0)
            proba = np.zeros((X.shape[0], 2))  # Assuming binary classification
            proba[:, 1] = votes / np.sum(self.weights)
            proba[:, 0] = 1 - proba[:, 1]
            
            return proba
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        앙상블 모델 평가
        
        Args:
            X_test (np.ndarray): 테스트 데이터
            y_test (np.ndarray): 테스트 타겟
            **kwargs: 추가 매개변수
            
        Returns:
            Dict[str, Any]: 평가 지표
        """
        if not self.is_trained:
            self.logger.warning("모델이 학습되지 않았습니다. 평가 결과가 부정확할 수 있습니다.")
        
        # 각 모델 개별 평가
        lstm_pred = self.lstm_model.predict(X_test)
        rf_pred = self.rf_model.predict(X_test)
        
        # 앙상블 예측
        ensemble_pred = self.predict(X_test)
        
        # 각 모델의 정확도
        lstm_accuracy = accuracy_score(y_test, lstm_pred)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        # 앙상블 세부 성능 지표
        ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted')
        ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted')
        ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
        
        # 평가 결과 저장
        evaluation_metrics = {
            'lstm_accuracy': lstm_accuracy,
            'rf_accuracy': rf_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_precision': ensemble_precision,
            'ensemble_recall': ensemble_recall,
            'ensemble_f1': ensemble_f1
        }
        
        # 성능 기반 가중치 동적 조정 (선택적)
        if self.adapt_weights and kwargs.get('update_weights', True):
            total_accuracy = lstm_accuracy + rf_accuracy
            if total_accuracy > 0:  # 0으로 나누기 방지
                self.lstm_weight = lstm_accuracy / total_accuracy
                self.rf_weight = rf_accuracy / total_accuracy
                self.logger.info(f"평가 기반 가중치 업데이트: GRU={self.lstm_weight:.4f}, RF={self.rf_weight:.4f}")
                
            evaluation_metrics.update({
                'updated_lstm_weight': self.lstm_weight,
                'updated_rf_weight': self.rf_weight
            })
        
        self.metrics.update(evaluation_metrics)
        self.logger.info(f"앙상블 평가 완료. 정확도: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")
        
        return evaluation_metrics
    
    def explain_prediction(self, X: np.ndarray) -> Dict[str, Any]:
        """
        예측 결과 설명
        
        Args:
            X (np.ndarray): 입력 데이터
            
        Returns:
            Dict[str, Any]: 예측 설명 정보
        """
        # GRU 예측
        lstm_pred = self.lstm_model.predict(X)
        lstm_proba = self.lstm_model.predict_proba(X)[:, 1]
        
        # RandomForest 예측
        rf_pred = self.rf_model.predict(X)
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        
        # 앙상블 예측
        ensemble_pred = self.predict(X)
        ensemble_proba = self.predict_proba(X)
        
        # 예측 설명 생성
        explanation = {
            'lstm_prediction': lstm_pred.tolist(),
            'lstm_probability': lstm_proba.tolist(),
            'rf_prediction': rf_pred.tolist(),
            'rf_probability': rf_proba.tolist(),
            'ensemble_prediction': ensemble_pred.tolist(),
            'ensemble_probability': ensemble_proba.tolist(),
            'lstm_weight': self.lstm_weight,
            'rf_weight': self.rf_weight
        }
        
        return explanation
    
    def save(self, 
            directory: Optional[str] = None, 
            save_models: bool = True) -> str:
        """
        앙상블 모델 저장
        
        Args:
            directory (Optional[str]): 저장할 디렉토리
            save_models (bool): 개별 모델 저장 여부
            
        Returns:
            str: 저장된 디렉토리 경로
        """
        if directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(self.model_dir, f"{self.name}_{timestamp}")
        
        os.makedirs(directory, exist_ok=True)
        
        # 앙상블 메타데이터 저장
        metadata = {
            'name': self.name,
            'version': self.version,
            'params': self.params,
            'metrics': self.metrics,
            'lstm_weight': self.lstm_weight,
            'rf_weight': self.rf_weight,
            'tech_weight': self.tech_weight,
            'lstm_model_name': self.lstm_model.name if self.lstm_model else None,
            'rf_model_name': self.rf_model.name if self.rf_model else None,
            'last_update': datetime.now().isoformat()
        }
        
        with open(os.path.join(directory, 'ensemble_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # 개별 모델 저장
        if save_models:
            # GRU 모델 저장
            if self.lstm_model:
                lstm_dir = os.path.join(directory, 'lstm_model')
                self.lstm_model.save(lstm_dir)
            
            # RandomForest 모델 저장
            if self.rf_model:
                rf_dir = os.path.join(directory, 'rf_model')
                self.rf_model.save(rf_dir)
        
        self.logger.info(f"계층적 앙상블 모델 저장 완료: {directory}")
        return directory
    
    @classmethod
    def load(cls, 
            directory: str, 
            load_models: bool = True) -> 'HierarchicalConfidenceEnsemble':
        """
        앙상블 모델 로드
        
        Args:
            directory (str): 로드할 디렉토리
            load_models (bool): 개별 모델 로드 여부
            
        Returns:
            HierarchicalConfidenceEnsemble: 로드된 앙상블 모델
        """
        try:
            # 앙상블 메타데이터 로드
            with open(os.path.join(directory, 'ensemble_metadata.json'), 'r') as f:
                metadata = json.load(f)
            
            # 앙상블 모델 생성
            ensemble = cls(
                name=metadata['name'],
                version=metadata['version'],
                lstm_weight=metadata['lstm_weight'],
                rf_weight=metadata['rf_weight'],
                tech_weight=metadata.get('tech_weight', 0.3),
                confidence_threshold=metadata['params'].get('confidence_threshold', 0.55),
                adapt_weights=metadata['params'].get('adapt_weights', True)
            )
            
            # 모델 속성 설정
            ensemble.metrics = metadata['metrics']
            ensemble.is_trained = True
            ensemble.last_update = datetime.fromisoformat(metadata['last_update'])
            
            # 개별 모델 로드
            if load_models:
                # GRU 모델 로드
                lstm_dir = os.path.join(directory, 'lstm_model')
                if os.path.exists(lstm_dir):
                    try:
                        from models.gru import GRUDirectionModel
                        ensemble.lstm_model = GRUDirectionModel.load(lstm_dir)
                        logger.info(f"GRU 모델 로드 성공: {lstm_dir}")
                    except Exception as e:
                        logger.error(f"GRU 모델 로드 실패: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"GRU 모델 디렉토리를 찾을 수 없음: {lstm_dir}")
                
                # RandomForest 모델 로드
                rf_dir = os.path.join(directory, 'rf_model')
                if os.path.exists(rf_dir):
                    try:
                        from models.random_forest import RandomForestDirectionModel
                        ensemble.rf_model = RandomForestDirectionModel.load(rf_dir)
                        logger.info(f"RandomForest 모델 로드 성공: {rf_dir}")
                    except Exception as e:
                        logger.error(f"RandomForest 모델 로드 실패: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"RandomForest 모델 디렉토리를 찾을 수 없음: {rf_dir}")
            
            logger.info(f"계층적 앙상블 모델 로드 완료: {directory}")
            return ensemble
        except Exception as e:
            logger.error(f"앙상블 모델 로드 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def update_model_confidence(self, 
                          predictions: Dict[str, np.ndarray], 
                          true_values: np.ndarray,
                          window_size: int = 10) -> None:
        """
        최근 예측 성능을 기반으로 모델 신뢰도 및 가중치 업데이트
        
        Args:
            predictions (Dict[str, np.ndarray]): 각 모델의 예측 결과
            true_values (np.ndarray): 실제 값
            window_size (int): 성능 평가 윈도우 크기
        """
        # 예측 성능 평가
        lstm_correct = np.sum(predictions['lstm'] == true_values)
        rf_correct = np.sum(predictions['rf'] == true_values)
        
        # 성능 기록 업데이트
        self.performance_history['lstm']['correct'] += lstm_correct
        self.performance_history['lstm']['total'] += len(true_values)
        self.performance_history['rf']['correct'] += rf_correct
        self.performance_history['rf']['total'] += len(true_values)
        
        # 성능 기록이 윈도우 크기를 초과하면 오래된 기록 제거
        if self.performance_history['lstm']['total'] > window_size:
            # 이동 평균 형태로 유지
            excess = self.performance_history['lstm']['total'] - window_size
            ratio_lstm = excess / self.performance_history['lstm']['total']
            ratio_rf = excess / self.performance_history['rf']['total']
            
            self.performance_history['lstm']['correct'] -= int(self.performance_history['lstm']['correct'] * ratio_lstm)
            self.performance_history['lstm']['total'] -= excess
            self.performance_history['rf']['correct'] -= int(self.performance_history['rf']['correct'] * ratio_rf)
            self.performance_history['rf']['total'] -= excess
        
        # 최신 성능 기반 가중치 업데이트
        self._update_weights_based_on_performance()
        
        self.logger.info(f"모델 신뢰도 업데이트 완료. GRU 정확도: {lstm_correct/len(true_values):.4f}, RF 정확도: {rf_correct/len(true_values):.4f}")

    def _update_weights_based_on_performance(self) -> None:
        """최근 성능 기록을 기반으로 모델 가중치 업데이트"""
        if not self.adapt_weights:
            return
        
        # 각 모델의 정확도 계산
        lstm_accuracy = self.performance_history['lstm']['correct'] / max(1, self.performance_history['lstm']['total'])
        rf_accuracy = self.performance_history['rf']['correct'] / max(1, self.performance_history['rf']['total'])
        
        # 모델별 신뢰 점수 계산 (정확도에 시장 상황 고려)
        lstm_confidence = lstm_accuracy
        rf_confidence = rf_accuracy
        
        # 가중치 업데이트 (정규화)
        total_confidence = lstm_confidence + rf_confidence
        if total_confidence > 0:  # 0으로 나누기 방지
            new_lstm_weight = lstm_confidence / total_confidence
            new_rf_weight = rf_confidence / total_confidence
            
            # 급격한 변화 방지를 위한 스무딩 (지수 이동 평균)
            alpha = 0.25  # 스무딩 계수 (0에 가까울수록 이전 가중치에 더 의존)
            self.lstm_weight = alpha * new_lstm_weight + (1 - alpha) * self.lstm_weight
            self.rf_weight = alpha * new_rf_weight + (1 - alpha) * self.rf_weight
            
            self.logger.info(f"가중치 업데이트: GRU={self.lstm_weight:.4f} (정확도: {lstm_accuracy:.4f}), RF={self.rf_weight:.4f} (정확도: {rf_accuracy:.4f})")

    def predict_with_confidence(self, 
                            X: np.ndarray,
                            **kwargs) -> Dict[str, Any]:
        """
        신뢰도 정보를 포함한 예측 수행
        
        Args:
            X (np.ndarray): 입력 데이터
            **kwargs: 추가 매개변수
                
        Returns:
            Dict[str, Any]: 예측 결과와 신뢰도 정보
        """
        # 각 모델 개별 예측
        lstm_pred = self.lstm_model.predict(X)
        lstm_proba = self.lstm_model.predict_proba(X)
        
        rf_pred = self.rf_model.predict(X)
        rf_proba = self.rf_model.predict_proba(X)
        
        # 앙상블 예측
        ensemble_pred = self._ensemble_predict(X)
        ensemble_proba = self.predict_proba(X)
        
        # 각 모델의 예측 신뢰도 계산
        lstm_confidence = np.max(lstm_proba, axis=1)  # 각 예측의 최대 확률을 신뢰도로 사용
        rf_confidence = np.max(rf_proba, axis=1)
        
        # 예측 동의 수준 (모델 간 일치도) 계산
        agreement = (lstm_pred == rf_pred).astype(float)
        
        # 최종 예측 신뢰도 계산
        if X.shape[0] == 1:  # 단일 샘플인 경우
            final_confidence = (
                lstm_confidence[0] * self.lstm_weight +
                rf_confidence[0] * self.rf_weight +
                agreement[0] * 0.2  # 모델 간 합의가 있으면 신뢰도 증가
            )
            
            # 신뢰도 스케일 조정 (0~1 범위로)
            final_confidence = min(1.0, final_confidence)
            
            # 각 모델의 예측 방향과 최종 예측 방향이 일치하는지 확인
            is_lstm_agree = lstm_pred[0] == ensemble_pred[0]
            is_rf_agree = rf_pred[0] == ensemble_pred[0]
            
            # 합의 정보 생성
            agreement_info = {
                'models_agree': bool(agreement[0]),
                'lstm_agrees_with_ensemble': is_lstm_agree,
                'rf_agrees_with_ensemble': is_rf_agree
            }
            
            # 최종 결과 반환
            result = {
                'prediction': int(ensemble_pred[0]),
                'probability': float(ensemble_proba[0, ensemble_pred[0]]),
                'confidence': float(final_confidence),
                'lstm_prediction': int(lstm_pred[0]),
                'lstm_confidence': float(lstm_confidence[0]),
                'rf_prediction': int(rf_pred[0]),
                'rf_confidence': float(rf_confidence[0]),
                'agreement': agreement_info,
                'is_confident': final_confidence >= self.confidence_threshold
            }
        else:  # 복수 샘플인 경우
            # 각 샘플별 신뢰도 계산
            final_confidence = (
                lstm_confidence * self.lstm_weight +
                rf_confidence * self.rf_weight +
                agreement * 0.2
            )
            
            # 신뢰도 스케일 조정 (0~1 범위로)
            final_confidence = np.minimum(1.0, final_confidence)
            
            # 최종 결과 반환 (복수 샘플)
            result = {
                'predictions': ensemble_pred.tolist(),
                'probabilities': ensemble_proba.tolist(),
                'confidences': final_confidence.tolist(),
                'lstm_predictions': lstm_pred.tolist(),
                'rf_predictions': rf_pred.tolist(),
                'agreement': agreement.tolist(),
                'is_confident': (final_confidence >= self.confidence_threshold).tolist()
            }
        
        return result
    
    def adjust_confidence_by_market_condition(self, 
                                        X: np.ndarray, 
                                        market_data: pd.DataFrame) -> Dict[str, float]:
        """
        시장 상황에 따라 모델 신뢰도 조정
        
        Args:
            X (np.ndarray): 입력 데이터
            market_data (pd.DataFrame): 시장 데이터
                
        Returns:
            Dict[str, float]: 조정된 모델 신뢰도
        """
        # 시장 변동성 계산 (예: 최근 20일 종가 수익률의 표준편차)
        if len(market_data) >= 20 and 'close' in market_data.columns:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns[-20:].std() if len(returns) >= 20 else 0.02
        else:
            volatility = 0.02  # 기본값
        
        # 추세 강도 계산 (예: 20일 이동평균과 현재 가격의 차이)
        if len(market_data) >= 20 and 'close' in market_data.columns:
            current_price = market_data['close'].iloc[-1]
            ma20 = market_data['close'].rolling(20).mean().iloc[-1]
            trend_strength = abs(current_price / ma20 - 1) if ma20 > 0 else 0.01
        else:
            trend_strength = 0.01  # 기본값
        
        # 시장 상황에 따른 모델 가중치 조정
        lstm_adjustment = 1.0
        rf_adjustment = 1.0
        
        # 변동성이 높은 경우
        if volatility > 0.03:
            # 변동성이 높을 때는 장기 모델(GRU)의 가중치를 높임
            lstm_adjustment = 1.2
            rf_adjustment = 0.8
            self.logger.info(f"높은 변동성 감지 ({volatility:.4f}): GRU 가중치 증가, RF 가중치 감소")
        # 변동성이 낮은 경우
        elif volatility < 0.01:
            # 변동성이 낮을 때는 단기 모델(RF)의 가중치를 높임
            lstm_adjustment = 0.8
            rf_adjustment = 1.2
            self.logger.info(f"낮은 변동성 감지 ({volatility:.4f}): RF 가중치 증가, GRU 가중치 감소")
        
        # 강한 추세가 있는 경우
        if trend_strength > 0.05:
            # 추세가 강할 때는 장기 모델(GRU)의 가중치를 높임
            lstm_adjustment *= 1.1
            self.logger.info(f"강한 추세 감지 ({trend_strength:.4f}): GRU 가중치 추가 증가")
        
        # 기존 가중치에 조정값 적용 (급격한 변화 방지를 위한 스무딩)
        alpha = 0.2  # 스무딩 계수
        adjusted_lstm_weight = alpha * (self.lstm_weight * lstm_adjustment) + (1 - alpha) * self.lstm_weight
        adjusted_rf_weight = alpha * (self.rf_weight * rf_adjustment) + (1 - alpha) * self.rf_weight
        
        # 가중치 정규화
        total_weight = adjusted_lstm_weight + adjusted_rf_weight
        adjusted_lstm_weight /= total_weight
        adjusted_rf_weight /= total_weight
        
        # 조정된 가중치 저장
        self.lstm_weight = adjusted_lstm_weight
        self.rf_weight = adjusted_rf_weight
        
        return {
            'lstm_weight': self.lstm_weight,
            'rf_weight': self.rf_weight,
            'volatility': volatility,
            'trend_strength': trend_strength
        }
        
    def update_confidence_from_trade_result(self, 
                                      trade_result: Dict[str, Any]) -> None:
        """
        실제 거래 결과를 기반으로 모델 신뢰도 업데이트
        
        Args:
            trade_result (Dict[str, Any]): 거래 결과 정보
                'prediction': 예측 방향 (1: 상승, 0: 하락)
                'actual': 실제 방향
                'profit_loss': 손익 (%, 양수: 이익, 음수: 손실)
                'model_predictions': 각 모델별 예측
        """
        if not self.adapt_weights:
            return
        
        # 필요한 정보 추출
        prediction = trade_result.get('prediction')
        actual = trade_result.get('actual')
        profit_loss = trade_result.get('profit_loss', 0.0)
        model_predictions = trade_result.get('model_predictions', {})
        
        if prediction is None or actual is None:
            self.logger.warning("거래 결과에 예측 또는 실제 방향 정보가 없습니다.")
            return
        
        # 예측 정확성 평가
        prediction_correct = prediction == actual
        
        # 각 모델의 예측 정확성
        lstm_prediction = model_predictions.get('lstm')
        rf_prediction = model_predictions.get('rf')
        
        lstm_correct = lstm_prediction == actual if lstm_prediction is not None else None
        rf_correct = rf_prediction == actual if rf_prediction is not None else None
        
        # 성능 업데이트를 위한 임시 딕셔너리
        predictions = {}
        if lstm_prediction is not None:
            predictions['lstm'] = np.array([lstm_prediction])
        if rf_prediction is not None:
            predictions['rf'] = np.array([rf_prediction])
        
        # 모델 성능 업데이트
        if predictions and actual is not None:
            self.update_model_confidence(predictions, np.array([actual]))
        
        # 거래 결과에 따른 추가 가중치 조정
        if prediction_correct and abs(profit_loss) > 0:
            # 성공한 예측은 해당 모델의 가중치를 약간 증가
            if lstm_correct and lstm_prediction is not None:
                self.lstm_weight *= 1.02
            if rf_correct and rf_prediction is not None:
                self.rf_weight *= 1.02
        elif not prediction_correct and abs(profit_loss) > 0:
            # 실패한 예측은 해당 모델의 가중치를 약간 감소
            if lstm_prediction is not None and not lstm_correct:
                self.lstm_weight *= 0.98
            if rf_prediction is not None and not rf_correct:
                self.rf_weight *= 0.98
        
        # 가중치 정규화
        total_weight = self.lstm_weight + self.rf_weight
        self.lstm_weight /= total_weight
        self.rf_weight /= total_weight
        
        self.logger.info(f"거래 결과 기반 가중치 업데이트: GRU={self.lstm_weight:.4f}, RF={self.rf_weight:.4f}, 손익={profit_loss:.2f}%")