"""
GPT-based Market Analysis Model for Bitcoin Trading Bot

This module implements models leveraging OpenAI's GPT API for market analysis and predictions.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from datetime import datetime
import json
import time
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import matplotlib.pyplot as plt
import openai
import traceback

from models.base import GPTAnalysisModel
from utils.logging import get_logger
from utils.constants import SignalType
from models.signal import TradingSignal, ModelOutput

# Initialize logger
logger = get_logger(__name__)


class GPTMarketAnalyzer(GPTAnalysisModel):
    """GPT-based market analyzer for Bitcoin price analysis and prediction"""
    
    def __init__(self, 
                name: str = "GPTMarketAnalyzer", 
                version: str = "1.0.0",
                api_key: Optional[str] = None,
                model: str = "gpt-4o",
                max_tokens: int = 1000,
                temperature: float = 0.2,
                market_context_days: int = 7,
                retry_attempts: int = 3,
                include_technical_indicators: bool = True,
                include_market_sentiment: bool = True,
                include_news: bool = False):
        """
        Initialize GPT Market Analyzer model
        
        Args:
            name (str): Model name
            version (str): Model version
            api_key (Optional[str]): OpenAI API key, if None, will try to get from environment
            model (str): GPT model to use
            max_tokens (int): Maximum tokens for response
            temperature (float): Model temperature (0.0-1.0)
            market_context_days (int): Number of days of market data to include in prompt
            retry_attempts (int): Number of retry attempts for API calls
            include_technical_indicators (bool): Whether to include technical indicators in analysis
            include_market_sentiment (bool): Whether to include market sentiment in analysis
            include_news (bool): Whether to include news in analysis
        """
        super().__init__(name, version)
        
        # Set API key
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
        else:
            self.logger.warning("OpenAI API key not provided")
        
        # Model parameters
        self.gpt_model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.market_context_days = market_context_days
        self.retry_attempts = retry_attempts
        
        # Analysis options
        self.include_technical_indicators = include_technical_indicators
        self.include_market_sentiment = include_market_sentiment
        self.include_news = include_news
        
        # Store parameters
        self.params = {
            'gpt_model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'market_context_days': market_context_days,
            'retry_attempts': retry_attempts,
            'include_technical_indicators': include_technical_indicators,
            'include_market_sentiment': include_market_sentiment,
            'include_news': include_news
        }
        
        # Analysis templates
        self.templates = self._load_default_templates()
        
        self.logger.info(f"Initialized {self.name} model with {model}")
    
    def _load_default_templates(self) -> Dict[str, str]:
        """
        Load default prompt templates
        
        Returns:
            Dict[str, str]: Dictionary of prompt templates
        """
        return {
            'market_analysis': """
            As a professional Bitcoin trader and market analyst, analyze the market data provided below and give a detailed analysis.
            
            Here's the Bitcoin market data for the past {market_context_days} days:
            {market_data}
            
            {technical_indicators}
            
            {market_sentiment}
            
            {news}
            
            Please provide a detailed analysis of the current market conditions and a prediction for the next 24 hours.
            Your analysis should include:
            1. Key support and resistance levels
            2. Market trend (bullish, bearish, or sideways)
            3. Important patterns or signals
            4. Expected price movement and range for the next 24 hours
            5. Confidence level in your prediction (low, medium, high)
            
            Return your analysis in JSON format with the following structure:
            {
                "market_condition": "bullish/bearish/sideways",
                "current_trend": "description",
                "key_levels": {
                    "support": [level1, level2],
                    "resistance": [level1, level2]
                },
                "patterns": ["pattern1", "pattern2"],
                "prediction": {
                    "direction": "up/down/sideways",
                    "expected_range": {
                        "low": value,
                        "high": value
                    },
                    "confidence": "low/medium/high"
                },
                "analysis": "detailed analysis text",
                "trade_recommendation": "buy/sell/hold",
                "risk_assessment": "low/medium/high"
            }
            """,
            
            'trade_signals': """
            As a professional Bitcoin trader, analyze the market data provided below and generate trade signals.
            
            Here's the Bitcoin market data for the past {market_context_days} days:
            {market_data}
            
            {technical_indicators}
            
            Current position: {current_position}
            
            Based on this data, generate trade signals (buy, sell, or hold) with specific entry and exit points.
            
            Return your analysis in JSON format with the following structure:
            {
                "signal": "buy/sell/hold",
                "entry_price": price or null,
                "stop_loss": price or null,
                "take_profit": price or null,
                "timeframe": "immediate/day/week",
                "confidence": "low/medium/high",
                "reasoning": "brief explanation of the signal"
            }
            """,
            
            'risk_assessment': """
            As a professional risk manager for a Bitcoin trading operation, analyze the market data provided below and assess the current trading risk.
            
            Here's the Bitcoin market data for the past {market_context_days} days:
            {market_data}
            
            {technical_indicators}
            
            {market_sentiment}
            
            Current position: {current_position}
            Current portfolio: {portfolio}
            
            Provide a detailed risk assessment for the current market conditions and our position.
            
            Return your analysis in JSON format with the following structure:
            {
                "overall_risk_level": "low/medium/high/extreme",
                "market_volatility": "low/medium/high",
                "liquidation_risk": "low/medium/high",
                "position_recommendation": "reduce/maintain/increase",
                "max_position_size": percentage,
                "max_leverage": value,
                "analysis": "detailed risk analysis",
                "warning_signals": ["signal1", "signal2"]
            }
            """
        }
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set the OpenAI API key
        
        Args:
            api_key (str): OpenAI API key
        """
        self.api_key = api_key
        openai.api_key = api_key
        self.logger.info("API key updated")
    
    def set_prompt_template(self, template_name: str, template: str) -> None:
        """
        Set or update a prompt template
        
        Args:
            template_name (str): Name of the template
            template (str): Template text
        """
        self.templates[template_name] = template
        self.logger.info(f"Updated prompt template: {template_name}")
    
    def get_prompt_template(self, template_name: str) -> Optional[str]:
        """
        Get a prompt template
        
        Args:
            template_name (str): Name of the template
            
        Returns:
            Optional[str]: Template text if found, None otherwise
        """
        return self.templates.get(template_name)
    
    def _prepare_market_data(self, 
                           data: pd.DataFrame, 
                           days: Optional[int] = None) -> str:
        """
        Prepare market data for GPT prompt
        
        Args:
            data (pd.DataFrame): Market data
            days (Optional[int]): Number of days to include
            
        Returns:
            str: Formatted market data
        """
        if days:
            data = data.tail(days)
        
        # Format data for prompt
        formatted_data = "Date, Open, High, Low, Close, Volume\n"
        for _, row in data.iterrows():
            date_str = row.name.strftime('%Y-%m-%d') if isinstance(row.name, pd.Timestamp) else str(row.name)
            formatted_data += f"{date_str}, {row['open']:.2f}, {row['high']:.2f}, {row['low']:.2f}, {row['close']:.2f}, {row['volume']:.2f}\n"
        
        return formatted_data
    
    def _prepare_technical_indicators(self, indicators: pd.DataFrame) -> str:
        """
        Prepare technical indicators for GPT prompt
        
        Args:
            indicators (pd.DataFrame): Technical indicators
            
        Returns:
            str: Formatted technical indicators
        """
        if not self.include_technical_indicators or indicators is None or indicators.empty:
            return ""
        
        # Get the last row (most recent indicators)
        last_indicators = indicators.iloc[-1]
        
        # Format indicators for prompt
        formatted_indicators = "Technical Indicators:\n"
        for indicator, value in last_indicators.items():
            # Format based on indicator type
            if isinstance(value, (int, float)):
                formatted_indicators += f"{indicator}: {value:.4f}\n"
            else:
                formatted_indicators += f"{indicator}: {value}\n"
        
        return formatted_indicators
    
    def _prepare_market_sentiment(self, sentiment_data: Dict[str, Any]) -> str:
        """
        Prepare market sentiment data for GPT prompt
        
        Args:
            sentiment_data (Dict[str, Any]): Market sentiment data
            
        Returns:
            str: Formatted market sentiment
        """
        if not self.include_market_sentiment or not sentiment_data:
            return ""
        
        # Format sentiment for prompt
        formatted_sentiment = "Market Sentiment:\n"
        
        if 'fear_greed_index' in sentiment_data:
            formatted_sentiment += f"Fear & Greed Index: {sentiment_data['fear_greed_index']} ({sentiment_data.get('fear_greed_category', 'Unknown')})\n"
        
        if 'social_sentiment' in sentiment_data:
            formatted_sentiment += f"Social Media Sentiment: {sentiment_data['social_sentiment']:.2f} (1 = Very Positive, -1 = Very Negative)\n"
        
        if 'long_short_ratio' in sentiment_data:
            formatted_sentiment += f"Long/Short Ratio: {sentiment_data['long_short_ratio']:.2f}\n"
        
        return formatted_sentiment
    
    def _prepare_news(self, news_data: List[Dict[str, str]]) -> str:
        """
        Prepare news data for GPT prompt
        
        Args:
            news_data (List[Dict[str, str]]): List of news items
            
        Returns:
            str: Formatted news
        """
        if not self.include_news or not news_data:
            return ""
        
        # Format news for prompt
        formatted_news = "Recent News:\n"
        
        for i, news in enumerate(news_data[:5]):  # Limit to 5 news items
            formatted_news += f"{i+1}. {news['date']} - {news['title']}"
            if 'summary' in news:
                formatted_news += f": {news['summary']}\n"
            else:
                formatted_news += "\n"
        
        return formatted_news
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_gpt_api(self, prompt: str) -> str:
        """
        Call the GPT API with the prompt
        
        Args:
            prompt (str): Prompt to send to the API
            
        Returns:
            str: API response text
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not set")
        
        try:
            start_time = time.time()
            self.logger.debug(f"Calling GPT API with prompt length: {len(prompt)}")
            
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=[{"role": "system", "content": "You are a professional Bitcoin trader and market analyst."},
                          {"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"GPT API response received in {elapsed_time:.2f} seconds")
            
            # Extract the response content
            if response and 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                return content
            
            raise ValueError("No valid response from OpenAI API")
            
        except Exception as e:
            self.logger.error(f"Error calling GPT API: {str(e)}")
            raise
    
    def _parse_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from the API response
        
        Args:
            response (str): API response text
            
        Returns:
            Dict[str, Any]: Parsed JSON
        """
        try:
            # Try to find JSON in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
            # If no JSON found, try to parse the entire response
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON from response: {str(e)}")
            self.logger.debug(f"Response: {response}")
            
            # Return a basic structure if parsing failed
            return {
                "error": "Failed to parse JSON from response",
                "raw_response": response
            }
    
    def analyze_market(self, 
                      market_data: pd.DataFrame, 
                      indicators: Optional[pd.DataFrame] = None,
                      sentiment_data: Optional[Dict[str, Any]] = None,
                      news_data: Optional[List[Dict[str, str]]] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Analyze the market using GPT
        
        Args:
            market_data (pd.DataFrame): Market OHLCV data
            indicators (Optional[pd.DataFrame]): Technical indicators
            sentiment_data (Optional[Dict[str, Any]]): Market sentiment data
            news_data (Optional[List[Dict[str, str]]]): News data
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Market analysis
        """
        # Prepare prompt components
        formatted_market_data = self._prepare_market_data(market_data, self.market_context_days)
        formatted_indicators = self._prepare_technical_indicators(indicators)
        formatted_sentiment = self._prepare_market_sentiment(sentiment_data)
        formatted_news = self._prepare_news(news_data)
        
        # Get template
        template = self.templates.get('market_analysis', "")
        
        # Fill template
        prompt = template.format(
            market_context_days=self.market_context_days,
            market_data=formatted_market_data,
            technical_indicators=formatted_indicators,
            market_sentiment=formatted_sentiment,
            news=formatted_news
        )
        
        # Call GPT API
        response = self._call_gpt_api(prompt)
        
        # Parse response
        analysis = self._parse_json_from_response(response)
        
        # Log analysis
        self.logger.info(f"Market analysis: {analysis.get('market_condition', 'Unknown')} | Prediction: {analysis.get('prediction', {}).get('direction', 'Unknown')}")
        
        # Update metrics
        timestamp = datetime.now()
        if 'analyses' not in self.metrics:
            self.metrics['analyses'] = []
        
        self.metrics['analyses'].append({
            'timestamp': timestamp.isoformat(),
            'market_condition': analysis.get('market_condition'),
            'prediction': analysis.get('prediction', {}).get('direction'),
            'confidence': analysis.get('prediction', {}).get('confidence')
        })
        
        self.is_trained = True
        self.last_update = timestamp
        
        return analysis
    
    def generate_trade_signals(self, 
                              market_data: pd.DataFrame, 
                              indicators: Optional[pd.DataFrame] = None,
                              current_position: str = "none",
                              **kwargs) -> Dict[str, Any]:
        """
        Generate trade signals using GPT
        
        Args:
            market_data (pd.DataFrame): Market OHLCV data
            indicators (Optional[pd.DataFrame]): Technical indicators
            current_position (str): Current position (none, long, short)
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Trade signals
        """
        # Prepare prompt components
        formatted_market_data = self._prepare_market_data(market_data, self.market_context_days)
        formatted_indicators = self._prepare_technical_indicators(indicators)
        
        # Get template
        template = self.templates.get('trade_signals', "")
        
        # Fill template
        prompt = template.format(
            market_context_days=self.market_context_days,
            market_data=formatted_market_data,
            technical_indicators=formatted_indicators,
            current_position=current_position
        )
        
        # Call GPT API
        response = self._call_gpt_api(prompt)
        
        # Parse response
        signals = self._parse_json_from_response(response)
        
        # Log signals
        self.logger.info(f"Trade signal: {signals.get('signal', 'Unknown')} | Confidence: {signals.get('confidence', 'Unknown')}")
        
        # Update metrics
        timestamp = datetime.now()
        if 'signals' not in self.metrics:
            self.metrics['signals'] = []
        
        self.metrics['signals'].append({
            'timestamp': timestamp.isoformat(),
            'signal': signals.get('signal'),
            'entry_price': signals.get('entry_price'),
            'confidence': signals.get('confidence')
        })
        
        self.last_update = timestamp
        
        return signals
    
    def assess_risk(self, 
                   market_data: pd.DataFrame, 
                   indicators: Optional[pd.DataFrame] = None,
                   sentiment_data: Optional[Dict[str, Any]] = None,
                   current_position: str = "none",
                   portfolio: Optional[Dict[str, Any]] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Assess trading risk using GPT
        
        Args:
            market_data (pd.DataFrame): Market OHLCV data
            indicators (Optional[pd.DataFrame]): Technical indicators
            sentiment_data (Optional[Dict[str, Any]]): Market sentiment data
            current_position (str): Current position (none, long, short)
            portfolio (Optional[Dict[str, Any]]): Portfolio details
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Risk assessment
        """
        # Prepare prompt components
        formatted_market_data = self._prepare_market_data(market_data, self.market_context_days)
        formatted_indicators = self._prepare_technical_indicators(indicators)
        formatted_sentiment = self._prepare_market_sentiment(sentiment_data)
        
        # Format portfolio
        portfolio_str = "No portfolio data available"
        if portfolio:
            portfolio_str = "Portfolio:\n"
            for key, value in portfolio.items():
                portfolio_str += f"{key}: {value}\n"
        
        # Get template
        template = self.templates.get('risk_assessment', "")
        
        # Fill template
        prompt = template.format(
            market_context_days=self.market_context_days,
            market_data=formatted_market_data,
            technical_indicators=formatted_indicators,
            market_sentiment=formatted_sentiment,
            current_position=current_position,
            portfolio=portfolio_str
        )
        
        # Call GPT API
        response = self._call_gpt_api(prompt)
        
        # Parse response
        assessment = self._parse_json_from_response(response)
        
        # Log assessment
        self.logger.info(f"Risk assessment: {assessment.get('overall_risk_level', 'Unknown')} | Recommendation: {assessment.get('position_recommendation', 'Unknown')}")
        
        # Update metrics
        timestamp = datetime.now()
        if 'risk_assessments' not in self.metrics:
            self.metrics['risk_assessments'] = []
        
        self.metrics['risk_assessments'].append({
            'timestamp': timestamp.isoformat(),
            'risk_level': assessment.get('overall_risk_level'),
            'recommendation': assessment.get('position_recommendation')
        })
        
        self.last_update = timestamp
        
        return assessment
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to disk
        
        Args:
            filepath (Optional[str]): Path to save the model
            
        Returns:
            str: Path where the model was saved
        """
        if filepath is None:
            # Create default filepath
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.model_dir, f"{self.name}_{timestamp}.json")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data for saving
        data = {
            'name': self.name,
            'version': self.version,
            'params': self.params,
            'metrics': self.metrics,
            'templates': self.templates,
            'last_update': datetime.now().isoformat()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        self.logger.info(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'GPTMarketAnalyzer':
        """
        Load a model from disk
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            GPTMarketAnalyzer: Loaded model
        """
        try:
            # Load from file
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Create instance with loaded parameters
            instance = cls(
                name=data['name'],
                version=data['version'],
                model=data['params']['gpt_model'],
                max_tokens=data['params']['max_tokens'],
                temperature=data['params']['temperature'],
                market_context_days=data['params']['market_context_days'],
                retry_attempts=data['params']['retry_attempts'],
                include_technical_indicators=data['params']['include_technical_indicators'],
                include_market_sentiment=data['params']['include_market_sentiment'],
                include_news=data['params']['include_news']
            )
            
            # Set instance variables
            instance.metrics = data['metrics']
            instance.templates = data['templates']
            instance.is_trained = True
            instance.last_update = datetime.fromisoformat(data['last_update'])
            
            logger.info(f"Model loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def plot_analysis_history(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot history of market analyses
        
        Args:
            figsize (Tuple[int, int]): Figure size
        """
        if 'analyses' not in self.metrics or not self.metrics['analyses']:
            self.logger.warning("No analysis history available")
            return
        
        # Extract data
        timestamps = []
        market_conditions = []
        predictions = []
        
        for analysis in self.metrics['analyses']:
            timestamps.append(datetime.fromisoformat(analysis['timestamp']))
            market_conditions.append(analysis['market_condition'])
            predictions.append(analysis['prediction'])
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot market conditions
        plt.subplot(2, 1, 1)
        condition_map = {'bullish': 1, 'sideways': 0, 'bearish': -1}
        condition_values = [condition_map.get(cond, 0) for cond in market_conditions]
        
        plt.plot(timestamps, condition_values, 'o-', label='Market Condition')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.yticks([-1, 0, 1], ['Bearish', 'Sideways', 'Bullish'])
        plt.title('Market Condition Over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot predictions
        plt.subplot(2, 1, 2)
        prediction_map = {'up': 1, 'sideways': 0, 'down': -1}
        prediction_values = [prediction_map.get(pred, 0) for pred in predictions]
        
        plt.plot(timestamps, prediction_values, 'o-', label='Price Prediction')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.yticks([-1, 0, 1], ['Down', 'Sideways', 'Up'])
        plt.title('Price Predictions Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelOutput:
        """
        Predict using GPT Market Analysis
        
        Args:
            X (np.ndarray): Input data (market data in numpy format)
            **kwargs: Additional parameters
                market_data (pd.DataFrame): Market OHLCV data
                indicators (Optional[pd.DataFrame]): Technical indicators
                sentiment_data (Optional[Dict[str, Any]]): Market sentiment data
                news_data (Optional[List[Dict[str, str]]]): News data
            
        Returns:
            ModelOutput: Standardized model output containing prediction results
        """
        # Check if we have a dataframe in kwargs
        market_data = kwargs.get('market_data')
        
        # If no market_data in kwargs, try to convert X to DataFrame
        if market_data is None:
            if isinstance(X, np.ndarray):
                try:
                    # Assume X is an OHLCV array
                    columns = ['open', 'high', 'low', 'close', 'volume']
                    if X.shape[1] < len(columns):
                        columns = columns[:X.shape[1]]
                    
                    market_data = pd.DataFrame(X, columns=columns)
                    
                    # Add timestamps if not present
                    if 'timestamp' not in market_data.columns:
                        end_date = datetime.now()
                        start_date = end_date - pd.Timedelta(days=len(market_data))
                        market_data['timestamp'] = pd.date_range(start=start_date, end=end_date, periods=len(market_data))
                        
                except Exception as e:
                    self.logger.error(f"Failed to convert X to DataFrame: {str(e)}")
                    return ModelOutput(
                        signal=TradingSignal(
                            signal_type=SignalType.HOLD,
                            confidence=0.0,
                            reason=f"Failed to convert input data: {str(e)}"
                        ),
                        confidence=0.0,
                        metadata={"error": str(e)}
                    )
            else:
                return ModelOutput(
                    signal=TradingSignal(
                        signal_type=SignalType.HOLD,
                        confidence=0.0,
                        reason="Input data format not supported"
                    ),
                    confidence=0.0,
                    metadata={"error": "Input data format not supported"}
                )
        
        # Extract other parameters from kwargs
        indicators = kwargs.get('indicators')
        sentiment_data = kwargs.get('sentiment_data')
        news_data = kwargs.get('news_data')
        
        try:
            # Call analyze_market
            analysis = self.analyze_market(
                market_data=market_data,
                indicators=indicators,
                sentiment_data=sentiment_data,
                news_data=news_data,
                **kwargs
            )
            
            # Extract market prediction
            prediction = analysis.get('prediction', {})
            direction = prediction.get('direction', 'sideways')
            confidence_text = prediction.get('confidence', 'medium')
            
            # Convert text confidence to numeric value
            confidence_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
            confidence = confidence_map.get(confidence_text.lower(), 0.5)
            
            # Convert direction to signal type
            signal_type = SignalType.HOLD
            if direction.lower() == 'up':
                signal_type = SignalType.BUY
            elif direction.lower() == 'down':
                signal_type = SignalType.SELL
            
            # Create TradingSignal
            signal = TradingSignal(
                signal_type=signal_type,
                confidence=confidence,
                reason=analysis.get('analysis', 'GPT market analysis'),
                metadata={
                    "market_condition": analysis.get('market_condition'),
                    "current_trend": analysis.get('current_trend'),
                    "key_levels": analysis.get('key_levels'),
                    "patterns": analysis.get('patterns'),
                    "expected_range": prediction.get('expected_range'),
                    "trade_recommendation": analysis.get('trade_recommendation'),
                    "risk_assessment": analysis.get('risk_assessment')
                }
            )
            
            # Create ModelOutput
            return ModelOutput(
                signal=signal,
                confidence=confidence,
                metadata={
                    "model_name": self.name,
                    "model_type": self.model_type,
                    "analysis_time": datetime.now().isoformat(),
                    "gpt_model": self.gpt_model,
                    "full_analysis": analysis
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in GPT market prediction: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return ModelOutput(
                signal=TradingSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    reason=f"GPT analysis error: {str(e)}"
                ),
                confidence=0.0,
                metadata={"error": str(e), "traceback": traceback.format_exc()}
            ) 