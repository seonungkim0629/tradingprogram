"""
Visualization module for Bitcoin Trading Bot

This module provides visualization functions for trading data and results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from matplotlib.ticker import FuncFormatter

from utils.logging import get_logger
from config import settings
from backtest.engine import BacktestResult

# Initialize logger
logger = get_logger(__name__)


def plot_equity_curve(equity_curve: List[float], 
                    dates: List[datetime],
                    title: str = "Equity Curve",
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot equity curve
    
    Args:
        equity_curve (List[float]): Equity curve values
        dates (List[datetime]): Dates corresponding to equity curve values
        title (str): Plot title
        save_path (Optional[str]): Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot equity curve
    ax.plot(dates, equity_curve, label="Portfolio Value", color="blue")
    
    # Add initial capital line
    ax.axhline(y=equity_curve[0], color="gray", linestyle="--", label="Initial Capital")
    
    # Format x-axis to show dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add return information
    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
    days = (dates[-1] - dates[0]).days
    annualized_return = (((equity_curve[-1] / equity_curve[0]) ** (365/days)) - 1) * 100
    
    return_text = f"Total Return: {total_return:.2f}%\nAnnualized Return: {annualized_return:.2f}%"
    ax.annotate(return_text, xy=(0.02, 0.95), xycoords="axes fraction", 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved equity curve plot to {save_path}")
    
    return fig


def plot_drawdown(equity_curve: List[float],
                dates: List[datetime],
                title: str = "Drawdown Analysis",
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot drawdown chart
    
    Args:
        equity_curve (List[float]): Equity curve values
        dates (List[datetime]): Dates corresponding to equity curve values
        title (str): Plot title
        save_path (Optional[str]): Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdowns
    drawdowns = ((running_max - equity_curve) / running_max) * 100
    
    # Plot drawdown
    ax.fill_between(dates, 0, drawdowns, color="red", alpha=0.3, label="Drawdown")
    
    # Format x-axis to show dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add max drawdown information
    max_drawdown = max(drawdowns)
    max_dd_date = dates[np.argmax(drawdowns)]
    
    max_dd_text = f"Maximum Drawdown: {max_drawdown:.2f}%\nDate: {max_dd_date.strftime('%Y-%m-%d')}"
    ax.annotate(max_dd_text, xy=(0.02, 0.95), xycoords="axes fraction", 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved drawdown plot to {save_path}")
    
    return fig


def plot_monthly_returns(equity_curve: List[float],
                      dates: List[datetime],
                      title: str = "Monthly Returns",
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot monthly returns heatmap
    
    Args:
        equity_curve (List[float]): Equity curve values
        dates (List[datetime]): Dates corresponding to equity curve values
        title (str): Plot title
        save_path (Optional[str]): Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Convert to DataFrame
    df = pd.DataFrame({"value": equity_curve}, index=dates)
    
    # Calculate daily returns
    df["daily_return"] = df["value"].pct_change()
    
    # Group by month and calculate monthly returns
    monthly_returns = df.resample("M").apply(
        lambda x: (x["value"].iloc[-1] / x["value"].iloc[0] - 1) * 100
    )["value"]
    
    # Create month-year index for heatmap
    monthly_returns_pivot = monthly_returns.unstack().T
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    cmap = plt.cm.RdYlGn  # Red for negative, Yellow for neutral, Green for positive
    im = ax.imshow(monthly_returns_pivot, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Returns (%)", rotation=-90, va="bottom")
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(monthly_returns_pivot.shape[1]))
    ax.set_yticks(np.arange(monthly_returns_pivot.shape[0]))
    
    # Label with months and years
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = monthly_returns_pivot.index.astype(str).tolist()
    
    ax.set_xticklabels(monthly_returns_pivot.columns.astype(str).tolist())
    ax.set_yticklabels(years)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations in each cell
    for i in range(len(years)):
        for j in range(len(monthly_returns_pivot.columns)):
            value = monthly_returns_pivot.iloc[i, j]
            if not math.isnan(value):
                text_color = "black" if abs(value) < 10 else "white"
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color)
    
    # Add title and labels
    ax.set_title(title)
    fig.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved monthly returns plot to {save_path}")
    
    return fig


def plot_trade_analysis(trades: List[Dict[str, Any]],
                      prices: pd.Series,
                      title: str = "Trade Analysis",
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot trade analysis chart with buy/sell points
    
    Args:
        trades (List[Dict[str, Any]]): List of trades
        prices (pd.Series): Price series
        title (str): Plot title
        save_path (Optional[str]): Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot price
    ax.plot(prices.index, prices.values, label="Price", color="blue")
    
    # Extract buy and sell points
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    
    for trade in trades:
        date = datetime.fromisoformat(trade["timestamp"])
        price = trade["price"]
        
        if trade["type"] == "BUY":
            buy_dates.append(date)
            buy_prices.append(price)
        elif trade["type"] == "SELL":
            sell_dates.append(date)
            sell_prices.append(price)
    
    # Plot buy and sell points
    ax.scatter(buy_dates, buy_prices, color="green", marker="^", s=100, label="Buy")
    ax.scatter(sell_dates, sell_prices, color="red", marker="v", s=100, label="Sell")
    
    # Format x-axis to show dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add trade statistics
    win_count = 0
    loss_count = 0
    total_profit = 0
    
    for i in range(min(len(buy_prices), len(sell_prices))):
        profit_pct = (sell_prices[i] / buy_prices[i] - 1) * 100
        total_profit += profit_pct
        
        if profit_pct > 0:
            win_count += 1
        else:
            loss_count += 1
    
    total_trades = win_count + loss_count
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    avg_profit = (total_profit / total_trades) if total_trades > 0 else 0
    
    stats_text = f"Total Trades: {total_trades}\nWin Rate: {win_rate:.1f}%\nAvg Profit/Trade: {avg_profit:.2f}%"
    ax.annotate(stats_text, xy=(0.02, 0.95), xycoords="axes fraction", 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved trade analysis plot to {save_path}")
    
    return fig


def plot_performance_metrics(metrics: Dict[str, float],
                           title: str = "Performance Metrics",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot performance metrics as bar chart
    
    Args:
        metrics (Dict[str, float]): Performance metrics
        title (str): Plot title
        save_path (Optional[str]): Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Select metrics to display
    key_metrics = {
        "monthly_return": "Monthly Return (%)",
        "annualized_return": "Annual Return (%)",
        "max_drawdown": "Max Drawdown (%)",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "win_rate": "Win Rate (%)",
        "profit_factor": "Profit Factor"
    }
    
    # Extract metrics to display
    plot_data = {}
    for key, label in key_metrics.items():
        if key in metrics:
            value = metrics[key]
            # Convert percentages for display
            if key in ["monthly_return", "annualized_return", "win_rate", "max_drawdown"]:
                value *= 100
            plot_data[label] = value
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(plot_data.keys(), plot_data.values())
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.2f}", ha="center", va="bottom")
    
    # Add title and labels
    ax.set_title(title)
    ax.set_ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved performance metrics plot to {save_path}")
    
    return fig


def create_interactive_dashboard(backtest_result: BacktestResult,
                               price_data: pd.DataFrame,
                               save_path: Optional[str] = None) -> go.Figure:
    """
    Create interactive dashboard with Plotly
    
    Args:
        backtest_result (BacktestResult): Backtest result
        price_data (pd.DataFrame): Price data
        save_path (Optional[str]): Path to save HTML file
        
    Returns:
        go.Figure: Plotly figure
    """
    # Extract data from backtest result
    equity_curve = backtest_result.equity_curve
    trades = backtest_result.trades
    metrics = backtest_result.metrics
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Portfolio Value", "Drawdown", 
                       "Trades", "Daily Returns",
                       "Key Metrics", "Monthly Returns"),
        specs=[
            [{"colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "domain"}, {"type": "xy"}]
        ],
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Create date range for equity curve
    dates = pd.date_range(
        start=backtest_result.start_date,
        end=backtest_result.end_date,
        periods=len(equity_curve)
    )
    
    # 1. Portfolio Value
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=equity_curve,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="blue")
        ),
        row=1, col=1
    )
    
    # Add initial capital line
    fig.add_trace(
        go.Scatter(
            x=[dates[0], dates[-1]],
            y=[equity_curve[0], equity_curve[0]],
            mode="lines",
            name="Initial Capital",
            line=dict(color="gray", dash="dash")
        ),
        row=1, col=1
    )
    
    # 2. Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = ((running_max - equity_curve) / running_max) * 100
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdowns,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red")
        ),
        row=2, col=1
    )
    
    # 3. Trades
    # Extract buy and sell points
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    
    for trade in trades:
        date = datetime.fromisoformat(trade["timestamp"])
        price = trade["price"]
        
        if trade["type"] == "BUY":
            buy_dates.append(date)
            buy_prices.append(price)
        elif trade["type"] == "SELL":
            sell_dates.append(date)
            sell_prices.append(price)
    
    # Plot price
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data["close"],
            mode="lines",
            name="Price",
            line=dict(color="blue")
        ),
        row=2, col=2
    )
    
    # Add buy points
    fig.add_trace(
        go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode="markers",
            name="Buy",
            marker=dict(color="green", size=10, symbol="triangle-up")
        ),
        row=2, col=2
    )
    
    # Add sell points
    fig.add_trace(
        go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode="markers",
            name="Sell",
            marker=dict(color="red", size=10, symbol="triangle-down")
        ),
        row=2, col=2
    )
    
    # 4. Key Metrics
    key_metrics = {
        "Monthly Return": f"{metrics.get('monthly_return', 0) * 100:.2f}%",
        "Annual Return": f"{metrics.get('annualized_return', 0) * 100:.2f}%",
        "Max Drawdown": f"{metrics.get('max_drawdown', 0) * 100:.2f}%",
        "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
        "Sortino Ratio": f"{metrics.get('sortino_ratio', 0):.2f}",
        "Win Rate": f"{metrics.get('win_rate', 0) * 100:.2f}%",
        "Profit Factor": f"{metrics.get('profit_factor', 0):.2f}",
        "Total Trades": f"{metrics.get('total_trades', 0)}"
    }
    
    # Create table for key metrics
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                font=dict(size=12),
                align="left"
            ),
            cells=dict(
                values=[
                    list(key_metrics.keys()),
                    list(key_metrics.values())
                ],
                font=dict(size=11),
                align="left"
            )
        ),
        row=3, col=1
    )
    
    # 5. Daily Returns
    # Calculate daily returns
    daily_returns = []
    for i in range(1, len(equity_curve)):
        daily_return = (equity_curve[i] / equity_curve[i-1] - 1) * 100
        daily_returns.append(daily_return)
    
    # Create histogram of daily returns
    fig.add_trace(
        go.Histogram(
            x=daily_returns,
            nbinsx=30,
            name="Daily Returns",
            marker_color="blue"
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Backtest Results: {backtest_result.strategy_name} on {backtest_result.market}",
        showlegend=False,
        height=900,
        width=1200
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Price", row=2, col=2)
    
    fig.update_xaxes(title_text="Daily Return (%)", row=3, col=2)
    fig.update_yaxes(title_text="Frequency", row=3, col=2)
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Saved interactive dashboard to {save_path}")
    
    return fig


def generate_all_plots(backtest_result: BacktestResult,
                     price_data: pd.DataFrame,
                     output_dir: str = None) -> Dict[str, plt.Figure]:
    """
    Generate all plots for a backtest result
    
    Args:
        backtest_result (BacktestResult): Backtest result
        price_data (pd.DataFrame): Price data
        output_dir (str): Directory to save plots
        
    Returns:
        Dict[str, plt.Figure]: Dictionary of plot figures
    """
    if output_dir is None:
        output_dir = os.path.join(settings.BACKTEST_RESULTS_DIR, "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create date range for equity curve
    dates = pd.date_range(
        start=backtest_result.start_date,
        end=backtest_result.end_date,
        periods=len(backtest_result.equity_curve)
    )
    
    # Generate plots
    plots = {}
    
    # Equity curve
    save_path = os.path.join(output_dir, "equity_curve.png")
    plots["equity_curve"] = plot_equity_curve(
        backtest_result.equity_curve,
        dates,
        title=f"Equity Curve: {backtest_result.strategy_name}",
        save_path=save_path
    )
    
    # Drawdown
    save_path = os.path.join(output_dir, "drawdown.png")
    plots["drawdown"] = plot_drawdown(
        backtest_result.equity_curve,
        dates,
        title=f"Drawdown: {backtest_result.strategy_name}",
        save_path=save_path
    )
    
    # Trade analysis
    save_path = os.path.join(output_dir, "trades.png")
    plots["trades"] = plot_trade_analysis(
        backtest_result.trades,
        price_data["close"],
        title=f"Trade Analysis: {backtest_result.strategy_name}",
        save_path=save_path
    )
    
    # Performance metrics
    save_path = os.path.join(output_dir, "metrics.png")
    plots["metrics"] = plot_performance_metrics(
        backtest_result.metrics,
        title=f"Performance Metrics: {backtest_result.strategy_name}",
        save_path=save_path
    )
    
    # Monthly returns
    save_path = os.path.join(output_dir, "monthly_returns.png")
    plots["monthly_returns"] = plot_monthly_returns(
        backtest_result.equity_curve,
        dates,
        title=f"Monthly Returns: {backtest_result.strategy_name}",
        save_path=save_path
    )
    
    # Interactive dashboard
    save_path = os.path.join(output_dir, "dashboard.html")
    create_interactive_dashboard(
        backtest_result,
        price_data,
        save_path=save_path
    )
    
    logger.info(f"Generated all plots for {backtest_result.strategy_name}")
    
    return plots 

def plot_trades(prices: List[float], 
               dates: List[datetime], 
               trades: List[Dict[str, Any]],
               title: str = "Trade History",
               save_path: Optional[str] = None) -> None:
    """
    가격 차트와 거래 표시 그래프 생성
    
    Args:
        prices (List[float]): 가격 데이터
        dates (List[datetime]): 날짜 리스트
        trades (List[Dict[str, Any]]): 거래 리스트
        title (str): 그래프 제목
        save_path (Optional[str]): 저장 경로 (없으면 화면에 표시)
    """
    if len(prices) != len(dates):
        logger.error(f"Prices length ({len(prices)}) doesn't match dates length ({len(dates)})")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, linewidth=1, alpha=0.7)
    
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    
    for trade in trades:
        trade_date = datetime.fromisoformat(trade['timestamp'])
        
        if trade['type'] == 'BUY':
            buy_dates.append(trade_date)
            buy_prices.append(trade['price'])
        elif trade['type'] == 'SELL':
            sell_dates.append(trade_date)
            sell_prices.append(trade['price'])
    
    # 매수/매도 표시
    plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy', alpha=0.8)
    plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell', alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 날짜 포맷 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # y축 가격 포맷 설정
    formatter = FuncFormatter(lambda x, pos: f'{x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Trade chart saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_returns_distribution(returns: List[float],
                            title: str = "Returns Distribution",
                            save_path: Optional[str] = None) -> None:
    """
    수익률 분포 히스토그램 생성
    
    Args:
        returns (List[float]): 수익률 리스트
        title (str): 그래프 제목
        save_path (Optional[str]): 저장 경로 (없으면 화면에 표시)
    """
    plt.figure(figsize=(10, 6))
    
    # 히스토그램 생성
    n, bins, patches = plt.hist(returns, bins=50, alpha=0.7, color='skyblue', density=True)
    
    # 정규 분포 곡선 추가
    mu = np.mean(returns)
    sigma = np.std(returns)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    plt.plot(bins, y, 'r--', linewidth=2, label='Normal Distribution')
    
    # 평균선 추가
    plt.axvline(mu, color='r', linestyle='-', label=f'Mean: {mu:.2%}')
    
    # 0선 추가
    plt.axvline(0, color='k', linestyle='-', alpha=0.5, label='Zero Return')
    
    plt.title(title)
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # x축 백분율 포맷 설정
    formatter = FuncFormatter(lambda x, pos: f'{x:.2%}')
    plt.gca().xaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Returns distribution saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_monte_carlo_simulation(mc_results: Dict[str, Any],
                               num_paths: int = 100,
                               title: str = "Monte Carlo Simulation",
                               save_path: Optional[str] = None) -> None:
    """
    몬테카를로 시뮬레이션 결과를 시각화합니다.
    
    Args:
        mc_results (Dict[str, Any]): 몬테카를로 시뮬레이션 결과
        num_paths (int): 표시할 경로 수
        title (str): 그래프 제목
        save_path (Optional[str]): 저장 경로 (없으면 화면에 표시)
    """
    # 결과에서 필요한 데이터 추출
    original_return = mc_results.get('original_return', 0)
    lower_ci, upper_ci = mc_results.get('return_ci', (0, 0))
    below_zero_prob = mc_results.get('return_below_zero_probability', 0)
    
    # 그래프 설정
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 수익률 분포 히스토그램
    # 정규 분포 가정하여 신뢰구간에서 표준편차 추정
    std_dev = (upper_ci - lower_ci) / 3.92  # 95% 신뢰구간 가정
    
    # 몬테카를로 결과 생성
    returns = np.random.normal(original_return, std_dev, 1000)
    
    # 히스토그램 그리기
    ax1.hist(returns, bins=50, alpha=0.6, color='skyblue', density=True)
    
    # 신뢰구간 표시
    ax1.axvline(lower_ci, color='red', linestyle='--', 
               label=f'95% 신뢰구간 하한: {lower_ci:.2%}')
    ax1.axvline(upper_ci, color='red', linestyle='--', 
               label=f'95% 신뢰구간 상한: {upper_ci:.2%}')
    
    # 원래 수익률 표시
    ax1.axvline(original_return, color='green', linewidth=2,
               label=f'실제 수익률: {original_return:.2%}')
    
    # 손익분기점 표시
    ax1.axvline(0, color='black', linestyle='-', 
               label=f'손익분기점 (손실확률: {below_zero_prob:.2%})')
    
    # 손실 부분 강조
    negative_returns = [r for r in returns if r < 0]
    if negative_returns:
        ax1.hist(negative_returns, bins=50, alpha=0.8, color='salmon', density=True)
    
    ax1.set_title('수익률 분포 및 신뢰구간')
    ax1.set_xlabel('총 수익률')
    ax1.set_ylabel('확률 밀도')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # x축 백분율 포맷 설정
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0%}'))
    
    # 2. 자본 곡선 시뮬레이션
    # 일별 수익률의 표준편차 추정
    daily_std_dev = std_dev / np.sqrt(252)  # 연간 표준편차를 일별로 변환
    
    # 거래일 수 (예: 252일)
    trading_days = 252
    
    # 시뮬레이션 경로
    paths = np.zeros((num_paths, trading_days + 1))
    paths[:, 0] = 1  # 초기 자본 = 1
    
    # 경로 생성
    mean_daily_return = original_return / 252  # 일평균 수익률 추정
    
    for i in range(num_paths):
        for t in range(1, trading_days + 1):
            # 로그정규 분포로 수익률 생성
            daily_return = np.random.normal(mean_daily_return, daily_std_dev)
            paths[i, t] = paths[i, t-1] * (1 + daily_return)
    
    # 경로 그리기
    x = np.arange(trading_days + 1)
    
    # 시뮬레이션 경로
    for i in range(num_paths):
        ax2.plot(x, paths[i], 'skyblue', linewidth=0.5, alpha=0.3)
    
    # 중앙값 경로
    median_path = np.median(paths, axis=0)
    ax2.plot(x, median_path, 'blue', linewidth=2, label='중앙값 경로')
    
    # 신뢰구간 경로
    lower_path = np.percentile(paths, 2.5, axis=0)
    upper_path = np.percentile(paths, 97.5, axis=0)
    ax2.plot(x, lower_path, 'red', linewidth=1.5, linestyle='--', label='95% 신뢰구간')
    ax2.plot(x, upper_path, 'red', linewidth=1.5, linestyle='--')
    
    # 원래 예상 경로
    expected_path = np.array([1 * (1 + mean_daily_return)**t for t in range(trading_days + 1)])
    ax2.plot(x, expected_path, 'green', linewidth=2, label='예상 경로')
    
    # 손익분기점
    ax2.axhline(1, color='black', linestyle='-', label='초기 투자금')
    
    ax2.fill_between(x, lower_path, upper_path, color='red', alpha=0.1)
    
    ax2.set_title('자본 곡선 시뮬레이션 (1년)')
    ax2.set_xlabel('거래일')
    ax2.set_ylabel('자본 (초기 자본 = 1)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # y축 백분율 포맷 설정
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}x'))
    
    # 전체 그래프 제목
    fig.suptitle(f"{title}\n(몬테카를로 시뮬레이션: {mc_results.get('number_of_simulations', 1000)}회)",
                fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Monte Carlo simulation chart saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_drawdowns(equity_curve: List[float],
                 dates: List[datetime],
                 title: str = "Drawdown Analysis",
                 save_path: Optional[str] = None) -> None:
    """
    낙폭(Drawdown) 분석 차트 생성
    
    Args:
        equity_curve (List[float]): 자본 곡선 데이터
        dates (List[datetime]): 날짜 리스트
        title (str): 그래프 제목
        save_path (Optional[str]): 저장 경로 (없으면 화면에 표시)
    """
    if len(equity_curve) != len(dates):
        logger.error(f"Equity curve length ({len(equity_curve)}) doesn't match dates length ({len(dates)})")
        return
    
    # 낙폭 계산
    equity_array = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # 자본 곡선 그래프
    ax1.plot(dates, equity_curve, linewidth=2, label='Equity Curve')
    ax1.plot(dates, peak, linestyle='--', color='green', alpha=0.5, label='Running Peak')
    
    # 주요 낙폭 표시
    threshold = 0.1  # 10% 이상 낙폭만 표시
    prev_end = -1
    
    for i in range(1, len(drawdown)):
        if drawdown[i] >= threshold and drawdown[i-1] < threshold:
            # 낙폭 시작
            start = i
        elif drawdown[i] < threshold and drawdown[i-1] >= threshold:
            # 낙폭 종료
            end = i
            if end - start > 5 and start > prev_end:  # 의미 있는 기간이고 이전 표시와 겹치지 않는 경우만
                ax1.axvspan(dates[start], dates[end], color='red', alpha=0.2)
                prev_end = end
    
    ax1.set_title(title)
    ax1.set_ylabel('Equity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 낙폭 그래프
    ax2.fill_between(dates, 0, drawdown * 100, color='red', alpha=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 최대 낙폭 표시
    max_dd = np.max(drawdown) * 100
    max_dd_idx = np.argmax(drawdown)
    ax2.axhline(max_dd, color='black', linestyle='--', 
               label=f'Maximum Drawdown: {max_dd:.2f}%')
    ax2.scatter(dates[max_dd_idx], max_dd, color='black', s=50)
    ax2.legend()
    
    # 날짜 포맷 설정
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # y축 포맷 설정
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:,.0f}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Drawdown analysis saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_market_comparison(strategy_equity: List[float],
                         market_equity: List[float],
                         dates: List[datetime],
                         title: str = "Strategy vs Market",
                         save_path: Optional[str] = None) -> None:
    """
    전략과 시장 성과 비교 차트 생성
    
    Args:
        strategy_equity (List[float]): 전략 자본 곡선
        market_equity (List[float]): 시장 자본 곡선 (단순 보유 전략)
        dates (List[datetime]): 날짜 리스트
        title (str): 그래프 제목
        save_path (Optional[str]): 저장 경로 (없으면 화면에 표시)
    """
    if len(strategy_equity) != len(market_equity) or len(strategy_equity) != len(dates):
        logger.error("Data lengths must match")
        return
    
    # 상대 성과 계산 (초기값 = 1로 정규화)
    norm_strategy = np.array(strategy_equity) / strategy_equity[0]
    norm_market = np.array(market_equity) / market_equity[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # 정규화된 자본 곡선 비교
    ax1.plot(dates, norm_strategy, linewidth=2, label='Strategy', color='blue')
    ax1.plot(dates, norm_market, linewidth=2, label='Market (Buy & Hold)', color='green')
    
    # 시장 대비 초과 성과 영역
    outperformance = norm_strategy - norm_market
    for i in range(1, len(dates)):
        if outperformance[i] >= 0 and outperformance[i-1] < 0:
            # 초과 성과 시작
            start = i
        elif outperformance[i] < 0 and outperformance[i-1] >= 0:
            # 초과 성과 종료
            end = i
            ax1.axvspan(dates[start], dates[end], color='blue', alpha=0.1)
        elif outperformance[i] < 0 and outperformance[i-1] < 0:
            # 저조한 성과 구간
            if i == 1:  # 첫 구간이 저조한 경우
                start = 0
            if i == len(dates) - 1:  # 마지막 구간이 저조한 경우
                ax1.axvspan(dates[start], dates[i], color='red', alpha=0.1)
    
    # 최종 성과 표시
    final_strategy_return = (norm_strategy[-1] - 1) * 100
    final_market_return = (norm_market[-1] - 1) * 100
    outperformance_pct = final_strategy_return - final_market_return
    
    ax1.text(0.02, 0.05, 
            f'전략 수익률: {final_strategy_return:.2f}%\n'
            f'시장 수익률: {final_market_return:.2f}%\n'
            f'초과 수익률: {outperformance_pct:.2f}%', 
            transform=ax1.transAxes,
            bbox=dict(facecolor='white', alpha=0.7))
    
    ax1.set_title(title)
    ax1.set_ylabel('Normalized Equity (Initial = 1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 상대 성과 차트
    ax2.plot(dates, outperformance, color='purple', label='Relative Performance')
    ax2.fill_between(dates, 0, outperformance, where=(outperformance > 0), 
                    color='blue', alpha=0.3, label='Outperformance')
    ax2.fill_between(dates, 0, outperformance, where=(outperformance < 0), 
                    color='red', alpha=0.3, label='Underperformance')
    
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Relative Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 날짜 포맷 설정
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # y축 포맷 설정
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}x'))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}x'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Market comparison saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_market_condition_performance(market_condition_results: Dict[str, Any],
                                    title: str = "Performance by Market Condition",
                                    save_path: Optional[str] = None) -> None:
    """
    시장 조건별 성과 분석 차트
    
    Args:
        market_condition_results (Dict[str, Any]): 시장 조건별 분석 결과
        title (str): 그래프 제목
        save_path (Optional[str]): 저장 경로 (없으면 화면에 표시)
    """
    # 시장 조건 유형과 각 조건별 결과 추출
    market_types = []
    returns = []
    sharpes = []
    win_rates = []
    
    for market_type in ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'low_volatile_market']:
        if market_type in market_condition_results and 'avg_return' in market_condition_results[market_type]:
            market_name = {
                'bull_market': '상승장',
                'bear_market': '하락장',
                'sideways_market': '횡보장',
                'volatile_market': '고변동성',
                'low_volatile_market': '저변동성'
            }.get(market_type, market_type)
            
            market_types.append(market_name)
            returns.append(market_condition_results[market_type]['avg_return'] * 100)  # 백분율로 변환
            sharpes.append(market_condition_results[market_type]['avg_sharpe'])
            win_rates.append(market_condition_results[market_type]['avg_win_rate'] * 100)  # 백분율로 변환
    
    if not market_types:
        logger.error("No valid market condition data found")
        return
    
    # 그래프 설정
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. 평균 수익률
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax1.bar(market_types, returns, color=colors, alpha=0.7)
    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    for i, v in enumerate(returns):
        ax1.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center')
    
    ax1.set_title('시장 조건별 평균 수익률')
    ax1.set_ylabel('평균 수익률 (%)')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 평균 샤프 지수
    colors = ['green' if s > 0 else 'red' for s in sharpes]
    ax2.bar(market_types, sharpes, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    for i, v in enumerate(sharpes):
        ax2.text(i, v + (0.1 if v >= 0 else -0.3), f"{v:.2f}", ha='center')
    
    ax2.set_title('시장 조건별 평균 샤프 지수')
    ax2.set_ylabel('평균 샤프 지수')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 평균 승률
    ax3.bar(market_types, win_rates, color='blue', alpha=0.7)
    ax3.axhline(50, color='black', linestyle='--', alpha=0.5, label='50% 승률')
    
    for i, v in enumerate(win_rates):
        ax3.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    ax3.set_title('시장 조건별 평균 승률')
    ax3.set_ylabel('평균 승률 (%)')
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend()
    
    # 전체 그래프 제목
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Market condition performance saved to {save_path}")
        plt.close()
    else:
        plt.show() 