# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MQL5 implementation of an LLM-PPO (Large Language Model with Proximal Policy Optimization) risk-aware trading system based on research by Qizhao Chen. The system combines LLM-based price predictions with PPO-based risk-aware adjustments for automated trading in MetaTrader 5.

## Architecture

### Core Components

**LLM_PPO_Model.mqh** - Main AI model class (`CLLM_PPO_Model`)
- Technical indicator integration (SMA, EMA, RSI, MACD, Bollinger Bands)
- LLM prediction simulation using ensemble methods
- PPO adjustment mechanism for risk-aware refinement
- VaR/CVaR risk metric calculations
- Historical learning and prediction tracking

**RiskManager.mqh** - Advanced risk management class (`CRiskManager`)
- Multiple VaR models: Historical, Parametric, Monte Carlo
- Performance ratios: Sharpe, Sortino, Calmar, Omega
- Position sizing: Kelly Criterion, Risk Parity, VaR-based
- Portfolio stress testing and monitoring

**ParameterOptimizer.mqh** - Optimization framework class (`CParameterOptimizer`)
- Genetic Algorithm, Brute Force, Particle Swarm, Simulated Annealing
- Walk-forward analysis and cross-validation
- Multi-objective fitness functions with constraint handling

### Trading Applications

**LLM_PPO_EA.mq5** - Main Expert Advisor
- Automated trading based on LLM-PPO predictions
- Risk-based position sizing using confidence levels
- Multi-layer risk management with daily limits and drawdown protection
- Real-time performance monitoring dashboard

**LLM_PPO_Backtest.mq5** - Backtesting engine
- Comprehensive historical testing framework
- Performance metrics and trade analysis
- Risk assessment and drawdown calculations

**LLM_PPO_Optimizer.mq5** - Parameter optimization script
- Advanced optimization using genetic algorithms and other methods
- Walk-forward validation and cross-validation
- Parameter sensitivity analysis

## Development Commands

### Compilation (MetaEditor)
```
Compile LLM_PPO_EA.mq5        // Main Expert Advisor
Compile LLM_PPO_Backtest.mq5  // Backtesting script
Compile LLM_PPO_Optimizer.mq5 // Optimization script
```

### File Installation
Place `.mqh` files in: `MQL5/Include/`
Place `.mq5` files in: `MQL5/Experts/` (EA) or `MQL5/Scripts/` (scripts)

## Code Conventions

### MQL5 Language Rules
- **Array parameters**: Must use reference syntax `double &array[]` for all array parameters
- **Struct constructors**: All structs with copying require explicit copy constructors and default constructors
- **Include statements**: Use angle brackets `#include <file.mqh>` in .mq5 files

### Technical Implementation
- **Risk-first approach**: All trading decisions incorporate VaR/CVaR calculations
- **PPO reward function**: `R_t = -|ŷ_t - y_t*| - λ·CVaR_α` where λ is risk weight
- **Two-stage prediction**: LLM simulation followed by PPO risk adjustment
- **Memory management**: Circular buffers for continuous operation with historical data arrays

### Parameter Patterns
- Learning rates: typically 0.0001 - 0.01
- Risk weights (λ): typically 0.1 - 1.0 
- Confidence levels: typically 0.90 - 0.99
- Position sizing: VaR-based with maximum risk limits per trade

## Key Classes and Methods

**CLLM_PPO_Model**
- `GenerateRiskAwarePrediction()`: Main prediction with risk adjustment
- `SimulateLLMOutput()`: Technical + sentiment ensemble prediction
- `GetPPOAdjustment()`: Risk-aware PPO refinement
- `CalculateVaR()` / `CalculateCVaR()`: Risk metrics

**CRiskManager**
- `CheckRiskLimits()`: Real-time portfolio risk monitoring
- `CalculateOptimalSize()`: VaR-based position sizing
- `CalculateHistoricalVaR()` / `CalculateParametricVaR()` / `CalculateMonteCarloVaR()`: Multiple VaR models

**CParameterOptimizer**
- `RunGeneticAlgorithm()`: Main optimization method
- `RunWalkForwardAnalysis()`: Time-series validation
- `CalculateFitness()`: Multi-objective performance evaluation

## Testing and Validation

### Backtesting Process
1. Use `LLM_PPO_Backtest.mq5` for historical validation
2. Minimum recommended data: 1+ years for meaningful statistics
3. Validate with walk-forward analysis using different market conditions

### Parameter Optimization
1. Use `LLM_PPO_Optimizer.mq5` with genetic algorithm (recommended)
2. Default population: 50, generations: 100
3. Always validate with out-of-sample testing
4. Monitor performance degradation (should be <25%)

### Risk Validation
- Verify VaR calculations align with actual historical losses
- Test stress scenarios and maximum drawdown limits
- Validate position sizing doesn't exceed account risk tolerance

## Important Considerations

### Limitations
- LLM simulation uses ensemble methods (not actual LLM API integration)
- Sentiment analysis is simplified (can be enhanced with external APIs)
- Intensive calculations may impact performance on slow systems

### Security
- No actual LLM API keys or external connections in current implementation
- All calculations performed locally within MetaTrader environment
- Risk management prevents excessive position sizes or account exposure

### Performance
- Memory-efficient circular buffer implementation for historical data
- Optimized indicator calculations to minimize computational load
- Real-time risk monitoring with automatic position closure on limit breaches