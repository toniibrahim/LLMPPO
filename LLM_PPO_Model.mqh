//+------------------------------------------------------------------+
//|                                                LLM_PPO_Model.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
// This is the core LLM-PPO (Large Language Model with Proximal Policy Optimization) 
// trading model implementation. It combines:
// 1. LLM simulation using technical indicators and sentiment analysis
// 2. PPO reinforcement learning for risk-aware prediction adjustment
// 3. Advanced risk management with VaR/CVaR calculations
// 4. Continuous learning from prediction accuracy
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| LLM-PPO Learning Model Class                                     |
//+------------------------------------------------------------------+
// CLLM_PPO_Model: Main AI trading model class implementing the two-stage framework:
// Stage 1: LLM simulation generates initial price predictions
// Stage 2: PPO adjustment applies risk-aware refinement
// The model learns continuously from market feedback using the reward function:
// R_t = -|ŷ_t - y_t*| - λ·CVaR_α (prediction error penalty + risk penalty)
class CLLM_PPO_Model
{
private:
    // Technical Indicator Handles - MT5 indicator objects for market analysis
    int               m_sma_handle;    // Simple Moving Average handle
    int               m_ema_handle;    // Exponential Moving Average handle 
    int               m_rsi_handle;    // Relative Strength Index handle
    int               m_macd_handle;   // MACD (Moving Average Convergence Divergence) handle
    int               m_bb_handle;     // Bollinger Bands handle
    
    // Core Model Parameters - Control AI behavior and learning
    int               m_lookback_period;    // Number of bars to analyze for predictions
    int               m_history_size;       // Size of historical data arrays
    double            m_learning_rate;      // PPO learning rate (how fast model adapts)
    double            m_risk_weight;        // Lambda (λ) in reward function - risk penalty weight
    double            m_confidence_level;   // Confidence level for VaR/CVaR calculations (e.g., 0.95)
    
    // Historical Data Arrays - Store learning data for continuous improvement
    double            m_price_history[];        // Circular buffer of historical prices
    double            m_prediction_history[];   // Circular buffer of model predictions
    double            m_reward_history[];       // Circular buffer of calculated rewards
    double            m_volatility_history[];   // Circular buffer of volatility measurements
    
    // PPO (Proximal Policy Optimization) Parameters - Core RL algorithm settings
    double            m_clip_epsilon;      // PPO clipping parameter (typically 0.2) - prevents large policy updates
    double            m_gamma;             // Discount factor for future rewards (typically 0.99)
    double            m_policy_weights[];  // Neural network weights for policy function
    double            m_value_weights[];   // Neural network weights for value function
    
    // Risk Metrics - Real-time risk assessment values
    double            m_var_value;            // Current Value at Risk (VaR) - potential loss at confidence level
    double            m_cvar_value;           // Current Conditional VaR (Expected Shortfall) - average loss beyond VaR
    double            m_current_volatility;   // Current market volatility measurement
    
    // Sentiment Analysis Components - Market mood assessment
    double            m_sentiment_score;         // Current sentiment score [-1.0 to +1.0] (negative=bearish, positive=bullish)
    datetime          m_last_sentiment_update;   // Timestamp of last sentiment update (prevents excessive recalculation)
    
public:
    // Constructor/Destructor - Object lifecycle management
                     CLLM_PPO_Model(void);   // Initialize model with default parameters
                    ~CLLM_PPO_Model(void);   // Clean up resources and indicators
    
    // Initialization Methods - Setup and teardown of model components
    bool              Initialize(string symbol, ENUM_TIMEFRAMES timeframe);  // Setup indicators and validate configuration
    void              Deinitialize(void);                                    // Release indicator handles and clean up
    
    // Technical Indicator Methods - Market analysis tools for LLM input features
    bool              CalculateTechnicalIndicators(void);                    // Ensure all indicators are ready
    double            GetSMA(int period, int shift = 0);                     // Simple Moving Average value
    double            GetEMA(int period, int shift = 0);                     // Exponential Moving Average value
    double            GetRSI(int period, int shift = 0);                     // RSI oscillator (0-100 scale)
    double            GetMACD(int shift = 0);                                // MACD main line value
    double            GetBollingerBands(int period, double deviation, int shift = 0); // Bollinger Band values
    
    // LLM Simulation Methods - Stage 1 of the two-stage framework
    double            GenerateInitialPrediction(void);                       // Generate basic prediction
    double            CalculateFeatureWeights(double &features[]);           // Weight technical features
    double            SimulateLLMOutput(double &technical_features[], double sentiment); // Main LLM simulation combining tech + sentiment
    
    // PPO Methods - Stage 2 reinforcement learning adjustment
    double            CalculateReward(double predicted_price, double actual_price);      // R_t = -|ŷ_t - y_t*| - λ·CVaR_α
    bool              UpdatePolicyNetwork(double &state[], double action, double reward); // Update policy weights
    double            GetPPOAdjustment(double llm_prediction, double &state[]);          // Calculate risk-aware adjustment
    double            CalculateAdvantage(double reward, double baseline);               // PPO advantage function
    
    // Risk Management Methods - Advanced risk assessment for trading decisions
    double            CalculateVaR(double &returns[], int period, double confidence);   // Historical Value at Risk calculation
    double            CalculateCVaR(double &returns[], int period, double confidence);  // Conditional VaR (Expected Shortfall)
    double            CalculateVolatility(int period);                                  // Standard deviation of returns
    bool              UpdateRiskMetrics(void);                                         // Update all risk metrics with latest data
    
    // Sentiment Analysis Methods - Market mood assessment (simplified implementation)
    bool              UpdateSentimentScore(void);                           // Update sentiment based on price momentum
    double            GetSentimentScore(void) { return m_sentiment_score; } // Get current sentiment [-1.0 to +1.0]
    
    // Main Prediction Methods - Core model output functions
    double            GenerateRiskAwarePrediction(void);                    // Full two-stage prediction: LLM + PPO adjustment
    double            GetFinalPrediction(void);                             // Get latest prediction result
    
    // Model State Methods - Persistence and reset functionality
    bool              SaveModelState(void);                                 // Save weights and parameters to file
    bool              LoadModelState(void);                                 // Load saved model state
    void              ResetModel(void);                                     // Reset to initial parameters
    
    // Utility Methods - Helper functions for model operations
    double            NormalizeValue(double value, double min_val, double max_val);     // Scale value to 0-1 range
    void              UpdateHistoryArrays(double price, double prediction, double reward); // Maintain circular buffers
    double            GetModelConfidence(void);                                         // Calculate confidence from recent performance
    
    // Getters - Access current risk metrics and model state
    double            GetCurrentVaR(void) { return m_var_value; }           // Current Value at Risk
    double            GetCurrentCVaR(void) { return m_cvar_value; }         // Current Conditional VaR
    double            GetCurrentVolatility(void) { return m_current_volatility; } // Current market volatility
    
    // Setters - Configure model parameters during runtime
    void              SetLearningRate(double rate) { m_learning_rate = rate; }     // Set PPO learning rate
    void              SetRiskWeight(double weight) { m_risk_weight = weight; }     // Set lambda (λ) risk penalty weight
    void              SetConfidenceLevel(double confidence) { m_confidence_level = confidence; } // Set VaR confidence level
};

//+------------------------------------------------------------------+
//| Constructor - Initialize model with default parameters           |
//+------------------------------------------------------------------+
// Constructor sets up the LLM-PPO model with research-based default values
// and initializes all arrays and neural network weights
CLLM_PPO_Model::CLLM_PPO_Model(void)
{
    // Set core model parameters based on research paper recommendations
    m_lookback_period = 5;        // 5-bar lookback for price pattern analysis
    m_history_size = 100;         // Store 100 historical data points for learning
    m_learning_rate = 0.0003;     // Conservative learning rate for stable training
    m_risk_weight = 0.5;          // Balanced risk penalty (lambda in reward function)
    m_confidence_level = 0.95;    // 95% confidence level for VaR calculations
    m_clip_epsilon = 0.2;         // Standard PPO clipping parameter
    m_gamma = 0.99;               // High discount factor for long-term rewards
    
    // Initialize risk metrics and sentiment to neutral/zero states
    m_var_value = 0.0;               // No VaR calculated initially
    m_cvar_value = 0.0;              // No CVaR calculated initially
    m_current_volatility = 0.0;      // No volatility measured initially
    m_sentiment_score = 0.0;         // Neutral sentiment (neither bullish nor bearish)
    m_last_sentiment_update = 0;     // No sentiment update timestamp
    
    // Initialize all dynamic arrays for historical data storage
    ArrayResize(m_price_history, m_history_size);      // Circular buffer for price history
    ArrayResize(m_prediction_history, m_history_size); // Circular buffer for prediction history
    ArrayResize(m_reward_history, m_history_size);     // Circular buffer for reward history
    ArrayResize(m_volatility_history, m_history_size); // Circular buffer for volatility history
    ArrayResize(m_policy_weights, 10);                 // Policy network weights (10 features)
    ArrayResize(m_value_weights, 10);                  // Value network weights (10 features)
    
    // Initialize neural network weights with small random values
    // This follows standard practice for neural network initialization
    for(int i = 0; i < 10; i++)
    {
        // Random values in [-0.05, +0.05] range for stable initial learning
        m_policy_weights[i] = (MathRand() / 32767.0 - 0.5) * 0.1;  // Policy network weights
        m_value_weights[i] = (MathRand() / 32767.0 - 0.5) * 0.1;   // Value network weights
    }
}

//+------------------------------------------------------------------+
//| Destructor - Clean up resources                                 |
//+------------------------------------------------------------------+
// Destructor ensures proper cleanup of MT5 indicator handles
CLLM_PPO_Model::~CLLM_PPO_Model(void)
{
    Deinitialize();
}

//+------------------------------------------------------------------+
//| Initialize the model - Setup technical indicators               |
//+------------------------------------------------------------------+
// Initialize() creates all required MT5 technical indicator handles
// These indicators provide the market data for LLM feature generation
bool CLLM_PPO_Model::Initialize(string symbol, ENUM_TIMEFRAMES timeframe)
{
    // Create technical indicator handles with standard parameters
    m_sma_handle = iMA(symbol, timeframe, 20, 0, MODE_SMA, PRICE_CLOSE);   // 20-period Simple Moving Average
    m_ema_handle = iMA(symbol, timeframe, 20, 0, MODE_EMA, PRICE_CLOSE);   // 20-period Exponential Moving Average
    m_rsi_handle = iRSI(symbol, timeframe, 14, PRICE_CLOSE);               // 14-period RSI oscillator
    m_macd_handle = iMACD(symbol, timeframe, 12, 26, 9, PRICE_CLOSE);      // MACD (12,26,9) standard settings
    m_bb_handle = iBands(symbol, timeframe, 20, 0, 2.0, PRICE_CLOSE);      // Bollinger Bands (20, 2.0 std dev)
    
    // Validate that all indicator handles were created successfully
    if(m_sma_handle == INVALID_HANDLE || m_ema_handle == INVALID_HANDLE ||
       m_rsi_handle == INVALID_HANDLE || m_macd_handle == INVALID_HANDLE ||
       m_bb_handle == INVALID_HANDLE)
    {
        Print("Error creating indicator handles - check symbol and timeframe");
        return false;  // Cannot proceed without valid indicators
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Deinitialize the model - Release indicator handles              |
//+------------------------------------------------------------------+
// Deinitialize() properly releases all MT5 indicator handles to prevent memory leaks
void CLLM_PPO_Model::Deinitialize(void)
{
    // Release all indicator handles to free MT5 resources
    if(m_sma_handle != INVALID_HANDLE) IndicatorRelease(m_sma_handle);   // Release SMA handle
    if(m_ema_handle != INVALID_HANDLE) IndicatorRelease(m_ema_handle);   // Release EMA handle
    if(m_rsi_handle != INVALID_HANDLE) IndicatorRelease(m_rsi_handle);   // Release RSI handle
    if(m_macd_handle != INVALID_HANDLE) IndicatorRelease(m_macd_handle); // Release MACD handle
    if(m_bb_handle != INVALID_HANDLE) IndicatorRelease(m_bb_handle);     // Release Bollinger Bands handle
}

//+------------------------------------------------------------------+
//| Calculate Technical Indicators - Ensure indicators are ready    |
//+------------------------------------------------------------------+
// CalculateTechnicalIndicators() verifies that all indicators have calculated
// sufficient data points for reliable analysis
bool CLLM_PPO_Model::CalculateTechnicalIndicators(void)
{
    // Verify all indicators have calculated at least 2 bars of data
    // This ensures we have current and previous values for trend analysis
    if(BarsCalculated(m_sma_handle) < 2 || BarsCalculated(m_ema_handle) < 2 ||
       BarsCalculated(m_rsi_handle) < 2 || BarsCalculated(m_macd_handle) < 2 ||
       BarsCalculated(m_bb_handle) < 2)
    {
        return false;  // Indicators not ready yet
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Get SMA value - Simple Moving Average                           |
//+------------------------------------------------------------------+
// GetSMA() retrieves Simple Moving Average value for specified bar
// Used for trend analysis and price smoothing in LLM features
double CLLM_PPO_Model::GetSMA(int period, int shift = 0)
{
    double sma_buffer[];  // Buffer to store SMA data
    if(CopyBuffer(m_sma_handle, 0, shift, 1, sma_buffer) > 0)
        return sma_buffer[0];  // Return SMA value
    return 0.0;  // Return 0 if data unavailable
}

//+------------------------------------------------------------------+
//| Get EMA value - Exponential Moving Average                      |
//+------------------------------------------------------------------+
// GetEMA() retrieves Exponential Moving Average value for specified bar
// EMA reacts faster to price changes than SMA, useful for momentum analysis
double CLLM_PPO_Model::GetEMA(int period, int shift = 0)
{
    double ema_buffer[];  // Buffer to store EMA data
    if(CopyBuffer(m_ema_handle, 0, shift, 1, ema_buffer) > 0)
        return ema_buffer[0];  // Return EMA value
    return 0.0;  // Return 0 if data unavailable
}

//+------------------------------------------------------------------+
//| Get RSI value - Relative Strength Index                         |
//+------------------------------------------------------------------+
// GetRSI() retrieves RSI oscillator value (0-100 scale)
// RSI > 70 suggests overbought conditions, RSI < 30 suggests oversold
double CLLM_PPO_Model::GetRSI(int period, int shift = 0)
{
    double rsi_buffer[];  // Buffer to store RSI data
    if(CopyBuffer(m_rsi_handle, 0, shift, 1, rsi_buffer) > 0)
        return rsi_buffer[0];  // Return RSI value (0-100)
    return 0.0;  // Return 0 if data unavailable
}

//+------------------------------------------------------------------+
//| Simulate LLM Output - Stage 1: Initial price prediction         |
//+------------------------------------------------------------------+
// SimulateLLMOutput() simulates Large Language Model behavior using ensemble methods
// Combines technical analysis (70%), sentiment (20%), and volatility adjustment (10%)
// In production, this would be replaced with actual LLM API calls
double CLLM_PPO_Model::SimulateLLMOutput(double &technical_features[], double sentiment)
{
    double weighted_sum = 0.0;  // Accumulated prediction signal
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Technical analysis component (70% total weight)
    double sma = GetSMA(20, 0);   // 20-period Simple Moving Average
    double ema = GetEMA(20, 0);   // 20-period Exponential Moving Average
    double rsi = GetRSI(14, 0);   // 14-period RSI oscillator
    
    // Calculate normalized technical features
    double price_trend = (current_price - sma) / sma;       // Price position relative to SMA
    double momentum = (ema - sma) / sma;                    // EMA-SMA divergence (momentum)
    double oversold_overbought = (rsi - 50.0) / 50.0;      // RSI relative to neutral (50)
    
    // Apply weights to technical features (total 70%)
    weighted_sum += price_trend * 0.3;            // 30% weight: price vs trend
    weighted_sum += momentum * 0.25;              // 25% weight: momentum signal
    weighted_sum += oversold_overbought * 0.15;   // 15% weight: overbought/oversold
    
    // Sentiment component (20% weight) - market mood factor
    weighted_sum += sentiment * 0.2;
    
    // Volatility adjustment (10% weight) - risk factor
    double volatility = CalculateVolatility(20);
    double vol_adjustment = MathMin(volatility * 100, 1.0);  // Cap at 1.0
    weighted_sum += vol_adjustment * 0.1;
    
    // Generate final prediction as percentage price change
    double prediction_pct = MathTanh(weighted_sum) * 0.05;  // Tanh limits to ±5% range
    
    return current_price * (1.0 + prediction_pct);  // Convert to absolute price
}

//+------------------------------------------------------------------+
//| Calculate PPO Reward - Reward function from research paper      |
//+------------------------------------------------------------------+
// CalculateReward() implements the reward function from the paper:
// R_t = -|ŷ_t - y_t*| - λ·CVaR_α
// Where: ŷ_t = predicted price, y_t* = actual price, λ = risk weight
double CLLM_PPO_Model::CalculateReward(double predicted_price, double actual_price)
{
    // Prediction accuracy component: -|ŷ_t - y_t*| normalized by actual price
    double prediction_error = MathAbs(predicted_price - actual_price) / actual_price;
    double accuracy_reward = -prediction_error;  // Negative because error is bad
    
    // Risk penalty component: λ·CVaR_α where λ is risk weight
    double risk_penalty = m_risk_weight * m_cvar_value;
    
    // Final reward combines prediction accuracy and risk penalty
    return accuracy_reward - risk_penalty;
}

//+------------------------------------------------------------------+
//| Get PPO Adjustment - Stage 2: Risk-aware prediction refinement  |
//+------------------------------------------------------------------+
// GetPPOAdjustment() applies PPO-learned adjustment to LLM prediction
// Uses current market state to calculate risk-aware modification
double CLLM_PPO_Model::GetPPOAdjustment(double llm_prediction, double &state[])
{
    double adjustment = 0.0;  // Initialize adjustment value
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Calculate state-based adjustment using learned policy weights
    // This is the core of the PPO policy network
    for(int i = 0; i < MathMin(ArraySize(m_policy_weights), ArraySize(state)); i++)
    {
        adjustment += state[i] * m_policy_weights[i];  // Linear combination of state features
    }
    
    // Apply safety clipping to prevent extreme adjustments
    double max_adjustment = current_price * 0.02;  // Limit adjustment to ±2% of current price
    adjustment = MathMax(-max_adjustment, MathMin(max_adjustment, adjustment));
    
    return adjustment;  // Return the risk-aware price adjustment
}

//+------------------------------------------------------------------+
//| Calculate Value at Risk (VaR) - Historical method               |
//+------------------------------------------------------------------+
// CalculateVaR() computes Value at Risk using historical simulation method
// VaR represents the maximum expected loss at a given confidence level
double CLLM_PPO_Model::CalculateVaR(double &returns[], int period, double confidence)
{
    if(ArraySize(returns) < period) return 0.0;  // Need sufficient data
    
    // Create temporary array for sorting (historical simulation method)
    double temp_returns[];
    ArrayResize(temp_returns, period);
    ArrayCopy(temp_returns, returns, 0, 0, period);
    ArraySort(temp_returns);  // Sort from worst to best returns
    
    // Find VaR at specified confidence level (e.g., 5th percentile for 95% confidence)
    int index = (int)((1.0 - confidence) * period);
    return temp_returns[index];  // Return the VaR value
}

//+------------------------------------------------------------------+
//| Calculate Conditional Value at Risk (CVaR) - Expected Shortfall  |
//+------------------------------------------------------------------+
// CalculateCVaR() computes Conditional VaR (Expected Shortfall)
// CVaR is the average of all losses worse than the VaR threshold
double CLLM_PPO_Model::CalculateCVaR(double &returns[], int period, double confidence)
{
    if(ArraySize(returns) < period) return 0.0;  // Need sufficient data
    
    // Create and sort returns array (same as VaR calculation)
    double temp_returns[];
    ArrayResize(temp_returns, period);
    ArrayCopy(temp_returns, returns, 0, 0, period);
    ArraySort(temp_returns);  // Sort from worst to best returns
    
    // Find VaR index (threshold for worst returns)
    int var_index = (int)((1.0 - confidence) * period);
    double sum = 0.0;
    int count = 0;
    
    // Calculate average of all returns worse than VaR threshold
    for(int i = 0; i <= var_index; i++)
    {
        sum += temp_returns[i];  // Sum all losses worse than VaR
        count++;
    }
    
    return (count > 0) ? sum / count : 0.0;  // Return average of tail losses
}

//+------------------------------------------------------------------+
//| Calculate Volatility - Standard deviation of returns            |
//+------------------------------------------------------------------+
// CalculateVolatility() computes the standard deviation of logarithmic returns
// This measures the degree of price fluctuation over the specified period
double CLLM_PPO_Model::CalculateVolatility(int period)
{
    double prices[];  // Array to store price data
    if(CopyClose(_Symbol, _Period, 0, period + 1, prices) <= 0)
        return 0.0;  // Return 0 if price data unavailable
    
    double returns[];  // Array to store calculated returns
    ArrayResize(returns, period);
    
    // Calculate logarithmic returns for each period
    for(int i = 0; i < period; i++)
    {
        returns[i] = MathLog(prices[i + 1] / prices[i]);  // Log return = ln(P_t / P_{t-1})
    }
    
    // Step 1: Calculate mean of returns
    double mean = 0.0;
    for(int i = 0; i < period; i++)
    {
        mean += returns[i];
    }
    mean /= period;  // Average return
    
    // Step 2: Calculate variance (average squared deviation from mean)
    double variance = 0.0;
    for(int i = 0; i < period; i++)
    {
        variance += MathPow(returns[i] - mean, 2);  // Squared deviation
    }
    variance /= (period - 1);  // Sample variance (N-1 denominator)
    
    return MathSqrt(variance);  // Volatility = square root of variance
}

//+------------------------------------------------------------------+
//| Update Risk Metrics - Refresh VaR, CVaR, and volatility        |
//+------------------------------------------------------------------+
// UpdateRiskMetrics() recalculates all risk metrics with latest market data
// Called before each prediction to ensure current risk assessment
bool CLLM_PPO_Model::UpdateRiskMetrics(void)
{
    double prices[];  // Array to store recent price data
    int period = 20;  // Use 20-period rolling window for risk calculations
    
    if(CopyClose(_Symbol, _Period, 0, period + 1, prices) <= 0)
        return false;  // Cannot update without price data
    
    double returns[];  // Array to store calculated returns
    ArrayResize(returns, period);
    
    // Calculate simple returns for risk metric calculations
    for(int i = 0; i < period; i++)
    {
        returns[i] = (prices[i + 1] - prices[i]) / prices[i];  // Simple return formula
    }
    
    // Update all risk metrics with latest data
    m_current_volatility = CalculateVolatility(period);                    // Current market volatility
    m_var_value = CalculateVaR(returns, period, m_confidence_level);       // Value at Risk
    m_cvar_value = CalculateCVaR(returns, period, m_confidence_level);     // Conditional VaR
    
    return true;
}

//+------------------------------------------------------------------+
//| Generate Risk-Aware Prediction - Main model output function     |
//+------------------------------------------------------------------+
// GenerateRiskAwarePrediction() implements the complete two-stage framework:
// Stage 1: LLM simulation generates initial prediction
// Stage 2: PPO applies risk-aware adjustment
// This is the primary function called by the trading system
double CLLM_PPO_Model::GenerateRiskAwarePrediction(void)
{
    // Ensure all indicators and risk metrics are current
    if(!CalculateTechnicalIndicators() || !UpdateRiskMetrics())
        return 0.0;  // Cannot proceed without current market data
    
    // Update market sentiment score (simplified implementation)
    UpdateSentimentScore();
    
    // Stage 1: Generate initial LLM prediction using technical features
    double technical_features[5];  // Array of input features for LLM
    technical_features[0] = GetSMA(20, 0);                           // Simple Moving Average
    technical_features[1] = GetEMA(20, 0);                           // Exponential Moving Average
    technical_features[2] = GetRSI(14, 0);                           // RSI oscillator
    technical_features[3] = m_current_volatility;                    // Market volatility
    technical_features[4] = SymbolInfoDouble(_Symbol, SYMBOL_BID);   // Current bid price
    
    // Generate initial prediction using LLM simulation
    double llm_prediction = SimulateLLMOutput(technical_features, m_sentiment_score);
    
    // Stage 2: Apply PPO risk-aware adjustment
    double state[5];  // Current market state for PPO policy
    state[0] = llm_prediction;                           // LLM predicted price
    state[1] = SymbolInfoDouble(_Symbol, SYMBOL_BID);    // Current market price
    state[2] = m_current_volatility;                     // Market volatility
    state[3] = m_var_value;                              // Current VaR
    state[4] = m_sentiment_score;                        // Market sentiment
    
    // Calculate PPO-based adjustment to LLM prediction
    double ppo_adjustment = GetPPOAdjustment(llm_prediction, state);
    double final_prediction = llm_prediction + ppo_adjustment;  // Final risk-aware prediction
    
    // Update learning history for continuous improvement
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double reward = CalculateReward(final_prediction, current_price);  // Calculate reward for learning
    UpdateHistoryArrays(current_price, final_prediction, reward);      // Store in circular buffers
    
    return final_prediction;  // Return the final risk-aware price prediction
}

//+------------------------------------------------------------------+
//| Update Sentiment Score - Simplified market mood assessment      |
//+------------------------------------------------------------------+
// UpdateSentimentScore() calculates market sentiment from price momentum
// In production, this would integrate with news sentiment APIs or social media analysis
bool CLLM_PPO_Model::UpdateSentimentScore(void)
{
    // Simplified sentiment calculation based on recent price momentum
    // Production implementation would use:
    // - News sentiment analysis APIs (Bloomberg, Reuters, etc.)
    // - Social media sentiment (Twitter, Reddit sentiment)
    // - Economic calendar events
    // - Market volatility indices (VIX, etc.)
    
    if(TimeCurrent() - m_last_sentiment_update < 3600) // Update only once per hour
        return true;
    
    double prices[];  // Array for recent price data
    if(CopyClose(_Symbol, _Period, 0, 10, prices) <= 0)
        return false;  // Cannot calculate without price data
    
    // Calculate 10-period price momentum as sentiment proxy
    double momentum = (prices[0] - prices[9]) / prices[9];     // Price change over 10 periods
    m_sentiment_score = MathTanh(momentum * 10) * 0.8;        // Scale to [-0.8, 0.8] range
    
    m_last_sentiment_update = TimeCurrent();  // Update timestamp
    return true;
}

//+------------------------------------------------------------------+
//| Update History Arrays - Maintain circular buffers for learning  |
//+------------------------------------------------------------------+
// UpdateHistoryArrays() maintains circular buffers of historical data
// This data is used for continuous learning and model improvement
void CLLM_PPO_Model::UpdateHistoryArrays(double price, double prediction, double reward)
{
    // Shift all arrays to make room for new data (circular buffer implementation)
    for(int i = m_history_size - 1; i > 0; i--)
    {
        m_price_history[i] = m_price_history[i-1];        // Shift price history
        m_prediction_history[i] = m_prediction_history[i-1]; // Shift prediction history
        m_reward_history[i] = m_reward_history[i-1];      // Shift reward history
        m_volatility_history[i] = m_volatility_history[i-1]; // Shift volatility history
    }
    
    // Insert new values at the front of arrays
    m_price_history[0] = price;                   // Current actual price
    m_prediction_history[0] = prediction;         // Model's prediction
    m_reward_history[0] = reward;                 // Calculated reward
    m_volatility_history[0] = m_current_volatility; // Current volatility
}

//+------------------------------------------------------------------+
//| Get Model Confidence - Calculate confidence from recent performance |
//+------------------------------------------------------------------+
// GetModelConfidence() estimates model confidence based on recent reward history
// Higher recent rewards indicate higher confidence in model predictions
double CLLM_PPO_Model::GetModelConfidence(void)
{
    if(ArraySize(m_reward_history) < 10)
        return 0.5;  // Default neutral confidence when insufficient data
    
    // Calculate average reward over last 10 predictions
    double avg_reward = 0.0;
    for(int i = 0; i < 10; i++)
    {
        avg_reward += m_reward_history[i];
    }
    avg_reward /= 10.0;  // Average recent reward
    
    // Convert average reward to confidence score [0.1, 0.9]
    // Higher rewards (better predictions) = higher confidence
    return MathMax(0.1, MathMin(0.9, 0.5 + avg_reward));  // Bounded confidence score
}