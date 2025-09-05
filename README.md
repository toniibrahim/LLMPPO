# LLM-PPO Risk-Aware Trading System

This MQL5 implementation is based on the research paper "A Two-Stage Framework for Stock Price Prediction: LLM-Based Forecasting with Risk-Aware PPO Adjustment" by Qizhao Chen.

## Overview

The system implements a novel two-stage framework that combines:
1. **Large Language Model (LLM) simulation** for initial price predictions
2. **Proximal Policy Optimization (PPO)** for risk-aware prediction refinement
3. **Advanced risk management** using VaR and CVaR metrics

## Key Features

### 🧠 LLM-PPO Learning Model (`LLM_PPO_Model.mqh`)
- **Technical Indicator Integration**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Sentiment Analysis**: Simulated news sentiment scoring
- **PPO Adjustment Mechanism**: Risk-aware prediction refinement
- **Real-time Risk Metrics**: VaR, CVaR, volatility calculations
- **Historical Learning**: Maintains prediction history for continuous improvement

### 🤖 Expert Advisor (`LLM_PPO_EA.mq5`)
- **Automated Trading**: Based on LLM-PPO predictions
- **Risk-Based Position Sizing**: Dynamic lot calculation using confidence levels
- **Multi-layer Risk Management**: Daily limits, drawdown protection, volatility-based stops
- **Adaptive Stop Levels**: Dynamic SL/TP based on market volatility
- **Real-time Monitoring**: Comprehensive trading dashboard

### ⚠️ Advanced Risk Manager (`RiskManager.mqh`)
- **Multiple VaR Models**: Historical, Parametric, Monte Carlo
- **Performance Ratios**: Sharpe, Sortino, Calmar, Omega ratios
- **Position Sizing Models**: Kelly Criterion, Risk Parity, VaR-based
- **Stress Testing**: Portfolio stress tests and backtesting
- **Real-time Monitoring**: Comprehensive risk limit checks

## Installation & Setup

1. **Copy Files**: Place all `.mqh` and `.mq5` files in your MetaTrader 5 directory:
   - `MQL5/Experts/` for the EA
   - `MQL5/Include/` for the header files

2. **Compile**: Compile `LLM_PPO_EA.mq5` in MetaEditor

3. **Attach to Chart**: Drag the EA to your desired trading symbol/timeframe

## Configuration Parameters

### Model Settings
- `InpLearningRate`: PPO learning rate (default: 0.0003)
- `InpRiskWeight`: Risk penalty weight λ (default: 0.5)
- `InpConfidenceLevel`: VaR confidence level (default: 0.95)
- `InpPredictionPeriod`: Lookback period (default: 5)

### Trading Settings
- `InpLotSize`: Base position size (default: 0.1)
- `InpUseRiskSizing`: Enable risk-based sizing (default: true)
- `InpMaxRiskPercent`: Maximum risk per trade (default: 2.0%)
- `InpMinConfidence`: Minimum model confidence to trade (default: 0.6)

### Risk Management
- `InpMaxDrawdown`: Maximum portfolio drawdown (default: 20%)
- `InpDailyLossLimit`: Daily loss limit (default: 5%)
- `InpUseDynamicSL`: Dynamic stop loss based on volatility (default: true)
- `InpVolatilityMultip`: Volatility multiplier for stops (default: 2.0)

## How It Works

### Stage 1: LLM Prediction Simulation
```cpp
// Combines technical analysis with sentiment
double prediction = SimulateLLMOutput(technical_features, sentiment_score);
```

### Stage 2: PPO Risk Adjustment
```cpp
// Applies risk-aware adjustment
double ppo_adjustment = GetPPOAdjustment(llm_prediction, state);
double final_prediction = llm_prediction + ppo_adjustment;
```

### Risk-Based Reward Function
```cpp
// From the paper: R_t = -|ŷ_t - y_t*| - λ·CVaR_α
double reward = -prediction_error - risk_weight * cvar_value;
```

## Research Results Implemented

Based on the study's findings, this implementation includes:

- **20-40% Drawdown Reduction**: Advanced risk management
- **Improved Sharpe Ratios**: Risk-adjusted performance optimization
- **Superior Risk Metrics**: VaR/CVaR integration throughout the system
- **Adaptive Learning**: PPO-based continuous improvement

## Key Classes and Methods

### CLLM_PPO_Model
- `GenerateRiskAwarePrediction()`: Main prediction method
- `CalculateVaR()` / `CalculateCVaR()`: Risk metric calculations
- `SimulateLLMOutput()`: Initial prediction generation
- `GetPPOAdjustment()`: Risk-aware refinement

### CRiskManager
- `CalculateHistoricalVaR()`: Historical Value at Risk
- `CalculateParametricVaR()`: Parametric VaR model
- `CalculateMonteCarloVaR()`: Monte Carlo simulation
- `CalculateKellySize()`: Optimal position sizing
- `CheckRiskLimits()`: Real-time risk monitoring

## Monitoring & Display

The EA provides real-time information:
- Current vs. predicted prices
- Model confidence levels
- Risk metrics (VaR, CVaR, volatility)
- Sentiment scores
- Daily P&L and open positions

## Performance Optimization

### Memory Management
- Efficient array handling for historical data
- Circular buffers for continuous operation
- Optimized indicator calculations

### Risk Controls
- Multiple safety layers
- Real-time limit monitoring
- Automatic position closure on breach

## Limitations & Considerations

1. **LLM Simulation**: Uses ensemble methods instead of actual LLM
2. **Sentiment Analysis**: Simplified implementation (can be enhanced with external APIs)
3. **Computational Requirements**: Intensive calculations may affect performance
4. **Market Conditions**: Performance may vary across different market regimes

## Future Enhancements

- Integration with actual LLM APIs (GPT, Claude, etc.)
- Real-time news sentiment analysis
- Multi-asset portfolio optimization
- Machine learning model persistence
- Advanced backtesting framework

## Disclaimer

This implementation is for educational and research purposes. Always conduct thorough testing before live trading. Past performance does not guarantee future results. Trading involves substantial risk of loss.

## References

Chen, Q.Z. (2025) "A Two-Stage Framework for Stock Price Prediction: LLM-Based Forecasting with Risk-Aware PPO Adjustment." Journal of Computer and Communications, 13, 120-139.