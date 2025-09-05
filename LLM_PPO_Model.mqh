//+------------------------------------------------------------------+
//|                                                LLM_PPO_Model.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| LLM-PPO Learning Model Class                                     |
//+------------------------------------------------------------------+
class CLLM_PPO_Model
{
private:
    // Technical Indicators
    int               m_sma_handle;
    int               m_ema_handle;
    int               m_rsi_handle;
    int               m_macd_handle;
    int               m_bb_handle;
    
    // Model Parameters
    int               m_lookback_period;
    int               m_history_size;
    double            m_learning_rate;
    double            m_risk_weight;
    double            m_confidence_level;
    
    // Data Arrays
    double            m_price_history[];
    double            m_prediction_history[];
    double            m_reward_history[];
    double            m_volatility_history[];
    
    // PPO Parameters
    double            m_clip_epsilon;
    double            m_gamma;
    double            m_policy_weights[];
    double            m_value_weights[];
    
    // Risk Metrics
    double            m_var_value;
    double            m_cvar_value;
    double            m_current_volatility;
    
    // Sentiment Analysis
    double            m_sentiment_score;
    datetime          m_last_sentiment_update;
    
public:
    // Constructor/Destructor
                     CLLM_PPO_Model(void);
                    ~CLLM_PPO_Model(void);
    
    // Initialization
    bool              Initialize(string symbol, ENUM_TIMEFRAMES timeframe);
    void              Deinitialize(void);
    
    // Technical Indicator Methods
    bool              CalculateTechnicalIndicators(void);
    double            GetSMA(int period, int shift = 0);
    double            GetEMA(int period, int shift = 0);
    double            GetRSI(int period, int shift = 0);
    double            GetMACD(int shift = 0);
    double            GetBollingerBands(int period, double deviation, int shift = 0);
    
    // LLM Simulation Methods
    double            GenerateInitialPrediction(void);
    double            CalculateFeatureWeights(double &features[]);
    double            SimulateLLMOutput(double &technical_features[], double sentiment);
    
    // PPO Methods
    double            CalculateReward(double predicted_price, double actual_price);
    bool              UpdatePolicyNetwork(double &state[], double action, double reward);
    double            GetPPOAdjustment(double llm_prediction, double &state[]);
    double            CalculateAdvantage(double reward, double baseline);
    
    // Risk Management Methods
    double            CalculateVaR(double &returns[], int period, double confidence);
    double            CalculateCVaR(double &returns[], int period, double confidence);
    double            CalculateVolatility(int period);
    bool              UpdateRiskMetrics(void);
    
    // Sentiment Analysis Methods
    bool              UpdateSentimentScore(void);
    double            GetSentimentScore(void) { return m_sentiment_score; }
    
    // Main Prediction Methods
    double            GenerateRiskAwarePrediction(void);
    double            GetFinalPrediction(void);
    
    // Model State Methods
    bool              SaveModelState(void);
    bool              LoadModelState(void);
    void              ResetModel(void);
    
    // Utility Methods
    double            NormalizeValue(double value, double min_val, double max_val);
    void              UpdateHistoryArrays(double price, double prediction, double reward);
    double            GetModelConfidence(void);
    
    // Getters
    double            GetCurrentVaR(void) { return m_var_value; }
    double            GetCurrentCVaR(void) { return m_cvar_value; }
    double            GetCurrentVolatility(void) { return m_current_volatility; }
    
    // Setters
    void              SetLearningRate(double rate) { m_learning_rate = rate; }
    void              SetRiskWeight(double weight) { m_risk_weight = weight; }
    void              SetConfidenceLevel(double confidence) { m_confidence_level = confidence; }
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CLLM_PPO_Model::CLLM_PPO_Model(void)
{
    m_lookback_period = 5;
    m_history_size = 100;
    m_learning_rate = 0.0003;
    m_risk_weight = 0.5;
    m_confidence_level = 0.95;
    m_clip_epsilon = 0.2;
    m_gamma = 0.99;
    
    m_var_value = 0.0;
    m_cvar_value = 0.0;
    m_current_volatility = 0.0;
    m_sentiment_score = 0.0;
    m_last_sentiment_update = 0;
    
    ArrayResize(m_price_history, m_history_size);
    ArrayResize(m_prediction_history, m_history_size);
    ArrayResize(m_reward_history, m_history_size);
    ArrayResize(m_volatility_history, m_history_size);
    ArrayResize(m_policy_weights, 10);
    ArrayResize(m_value_weights, 10);
    
    // Initialize weights randomly
    for(int i = 0; i < 10; i++)
    {
        m_policy_weights[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
        m_value_weights[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
    }
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CLLM_PPO_Model::~CLLM_PPO_Model(void)
{
    Deinitialize();
}

//+------------------------------------------------------------------+
//| Initialize the model                                             |
//+------------------------------------------------------------------+
bool CLLM_PPO_Model::Initialize(string symbol, ENUM_TIMEFRAMES timeframe)
{
    // Create technical indicator handles
    m_sma_handle = iMA(symbol, timeframe, 20, 0, MODE_SMA, PRICE_CLOSE);
    m_ema_handle = iMA(symbol, timeframe, 20, 0, MODE_EMA, PRICE_CLOSE);
    m_rsi_handle = iRSI(symbol, timeframe, 14, PRICE_CLOSE);
    m_macd_handle = iMACD(symbol, timeframe, 12, 26, 9, PRICE_CLOSE);
    m_bb_handle = iBands(symbol, timeframe, 20, 0, 2.0, PRICE_CLOSE);
    
    if(m_sma_handle == INVALID_HANDLE || m_ema_handle == INVALID_HANDLE ||
       m_rsi_handle == INVALID_HANDLE || m_macd_handle == INVALID_HANDLE ||
       m_bb_handle == INVALID_HANDLE)
    {
        Print("Error creating indicator handles");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Deinitialize the model                                           |
//+------------------------------------------------------------------+
void CLLM_PPO_Model::Deinitialize(void)
{
    if(m_sma_handle != INVALID_HANDLE) IndicatorRelease(m_sma_handle);
    if(m_ema_handle != INVALID_HANDLE) IndicatorRelease(m_ema_handle);
    if(m_rsi_handle != INVALID_HANDLE) IndicatorRelease(m_rsi_handle);
    if(m_macd_handle != INVALID_HANDLE) IndicatorRelease(m_macd_handle);
    if(m_bb_handle != INVALID_HANDLE) IndicatorRelease(m_bb_handle);
}

//+------------------------------------------------------------------+
//| Calculate Technical Indicators                                   |
//+------------------------------------------------------------------+
bool CLLM_PPO_Model::CalculateTechnicalIndicators(void)
{
    // Wait for indicators to calculate
    if(BarsCalculated(m_sma_handle) < 2 || BarsCalculated(m_ema_handle) < 2 ||
       BarsCalculated(m_rsi_handle) < 2 || BarsCalculated(m_macd_handle) < 2 ||
       BarsCalculated(m_bb_handle) < 2)
    {
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Get SMA value                                                    |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::GetSMA(int period, int shift = 0)
{
    double sma_buffer[];
    if(CopyBuffer(m_sma_handle, 0, shift, 1, sma_buffer) > 0)
        return sma_buffer[0];
    return 0.0;
}

//+------------------------------------------------------------------+
//| Get EMA value                                                    |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::GetEMA(int period, int shift = 0)
{
    double ema_buffer[];
    if(CopyBuffer(m_ema_handle, 0, shift, 1, ema_buffer) > 0)
        return ema_buffer[0];
    return 0.0;
}

//+------------------------------------------------------------------+
//| Get RSI value                                                    |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::GetRSI(int period, int shift = 0)
{
    double rsi_buffer[];
    if(CopyBuffer(m_rsi_handle, 0, shift, 1, rsi_buffer) > 0)
        return rsi_buffer[0];
    return 0.0;
}

//+------------------------------------------------------------------+
//| Simulate LLM Output                                              |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::SimulateLLMOutput(double &technical_features[], double sentiment)
{
    double weighted_sum = 0.0;
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Technical analysis component (70% weight)
    double sma = GetSMA(20, 0);
    double ema = GetEMA(20, 0);
    double rsi = GetRSI(14, 0);
    
    // Price trend analysis
    double price_trend = (current_price - sma) / sma;
    double momentum = (ema - sma) / sma;
    double oversold_overbought = (rsi - 50.0) / 50.0;
    
    weighted_sum += price_trend * 0.3;
    weighted_sum += momentum * 0.25;
    weighted_sum += oversold_overbought * 0.15;
    
    // Sentiment component (20% weight)
    weighted_sum += sentiment * 0.2;
    
    // Volatility adjustment (10% weight)
    double volatility = CalculateVolatility(20);
    double vol_adjustment = MathMin(volatility * 100, 1.0);
    weighted_sum += vol_adjustment * 0.1;
    
    // Generate prediction as percentage change
    double prediction_pct = MathTanh(weighted_sum) * 0.05; // Limit to ±5%
    
    return current_price * (1.0 + prediction_pct);
}

//+------------------------------------------------------------------+
//| Calculate PPO Reward                                             |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::CalculateReward(double predicted_price, double actual_price)
{
    // Prediction accuracy component
    double prediction_error = MathAbs(predicted_price - actual_price) / actual_price;
    double accuracy_reward = -prediction_error;
    
    // Risk penalty component
    double risk_penalty = m_risk_weight * m_cvar_value;
    
    return accuracy_reward - risk_penalty;
}

//+------------------------------------------------------------------+
//| Get PPO Adjustment                                               |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::GetPPOAdjustment(double llm_prediction, double &state[])
{
    double adjustment = 0.0;
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Calculate state-based adjustment
    for(int i = 0; i < MathMin(ArraySize(m_policy_weights), ArraySize(state)); i++)
    {
        adjustment += state[i] * m_policy_weights[i];
    }
    
    // Apply clipping to prevent large adjustments
    double max_adjustment = current_price * 0.02; // Max 2% adjustment
    adjustment = MathMax(-max_adjustment, MathMin(max_adjustment, adjustment));
    
    return adjustment;
}

//+------------------------------------------------------------------+
//| Calculate Value at Risk (VaR)                                    |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::CalculateVaR(double &returns[], int period, double confidence)
{
    if(ArraySize(returns) < period) return 0.0;
    
    double temp_returns[];
    ArrayResize(temp_returns, period);
    ArrayCopy(temp_returns, returns, 0, 0, period);
    ArraySort(temp_returns);
    
    int index = (int)((1.0 - confidence) * period);
    return temp_returns[index];
}

//+------------------------------------------------------------------+
//| Calculate Conditional Value at Risk (CVaR)                       |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::CalculateCVaR(double &returns[], int period, double confidence)
{
    if(ArraySize(returns) < period) return 0.0;
    
    double temp_returns[];
    ArrayResize(temp_returns, period);
    ArrayCopy(temp_returns, returns, 0, 0, period);
    ArraySort(temp_returns);
    
    int var_index = (int)((1.0 - confidence) * period);
    double sum = 0.0;
    int count = 0;
    
    for(int i = 0; i <= var_index; i++)
    {
        sum += temp_returns[i];
        count++;
    }
    
    return (count > 0) ? sum / count : 0.0;
}

//+------------------------------------------------------------------+
//| Calculate Volatility                                             |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::CalculateVolatility(int period)
{
    double prices[];
    if(CopyClose(_Symbol, _Period, 0, period + 1, prices) <= 0)
        return 0.0;
    
    double returns[];
    ArrayResize(returns, period);
    
    // Calculate returns
    for(int i = 0; i < period; i++)
    {
        returns[i] = MathLog(prices[i + 1] / prices[i]);
    }
    
    // Calculate standard deviation
    double mean = 0.0;
    for(int i = 0; i < period; i++)
    {
        mean += returns[i];
    }
    mean /= period;
    
    double variance = 0.0;
    for(int i = 0; i < period; i++)
    {
        variance += MathPow(returns[i] - mean, 2);
    }
    variance /= (period - 1);
    
    return MathSqrt(variance);
}

//+------------------------------------------------------------------+
//| Update Risk Metrics                                              |
//+------------------------------------------------------------------+
bool CLLM_PPO_Model::UpdateRiskMetrics(void)
{
    double prices[];
    int period = 20;
    
    if(CopyClose(_Symbol, _Period, 0, period + 1, prices) <= 0)
        return false;
    
    double returns[];
    ArrayResize(returns, period);
    
    // Calculate returns
    for(int i = 0; i < period; i++)
    {
        returns[i] = (prices[i + 1] - prices[i]) / prices[i];
    }
    
    m_current_volatility = CalculateVolatility(period);
    m_var_value = CalculateVaR(returns, period, m_confidence_level);
    m_cvar_value = CalculateCVaR(returns, period, m_confidence_level);
    
    return true;
}

//+------------------------------------------------------------------+
//| Generate Risk-Aware Prediction                                   |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::GenerateRiskAwarePrediction(void)
{
    if(!CalculateTechnicalIndicators() || !UpdateRiskMetrics())
        return 0.0;
    
    // Update sentiment (simplified - in real implementation would call external API)
    UpdateSentimentScore();
    
    // Stage 1: Generate initial LLM prediction
    double technical_features[5];
    technical_features[0] = GetSMA(20, 0);
    technical_features[1] = GetEMA(20, 0);
    technical_features[2] = GetRSI(14, 0);
    technical_features[3] = m_current_volatility;
    technical_features[4] = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    double llm_prediction = SimulateLLMOutput(technical_features, m_sentiment_score);
    
    // Stage 2: Apply PPO adjustment
    double state[5];
    state[0] = llm_prediction;
    state[1] = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    state[2] = m_current_volatility;
    state[3] = m_var_value;
    state[4] = m_sentiment_score;
    
    double ppo_adjustment = GetPPOAdjustment(llm_prediction, state);
    double final_prediction = llm_prediction + ppo_adjustment;
    
    // Update history for learning
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double reward = CalculateReward(final_prediction, current_price);
    UpdateHistoryArrays(current_price, final_prediction, reward);
    
    return final_prediction;
}

//+------------------------------------------------------------------+
//| Update Sentiment Score (Simplified)                             |
//+------------------------------------------------------------------+
bool CLLM_PPO_Model::UpdateSentimentScore(void)
{
    // Simplified sentiment calculation based on price momentum
    // In real implementation, this would call external news API
    
    if(TimeCurrent() - m_last_sentiment_update < 3600) // Update hourly
        return true;
    
    double prices[];
    if(CopyClose(_Symbol, _Period, 0, 10, prices) <= 0)
        return false;
    
    double momentum = (prices[0] - prices[9]) / prices[9];
    m_sentiment_score = MathTanh(momentum * 10) * 0.8; // Scale to [-0.8, 0.8]
    
    m_last_sentiment_update = TimeCurrent();
    return true;
}

//+------------------------------------------------------------------+
//| Update History Arrays                                            |
//+------------------------------------------------------------------+
void CLLM_PPO_Model::UpdateHistoryArrays(double price, double prediction, double reward)
{
    // Shift arrays
    for(int i = m_history_size - 1; i > 0; i--)
    {
        m_price_history[i] = m_price_history[i-1];
        m_prediction_history[i] = m_prediction_history[i-1];
        m_reward_history[i] = m_reward_history[i-1];
        m_volatility_history[i] = m_volatility_history[i-1];
    }
    
    // Add new values
    m_price_history[0] = price;
    m_prediction_history[0] = prediction;
    m_reward_history[0] = reward;
    m_volatility_history[0] = m_current_volatility;
}

//+------------------------------------------------------------------+
//| Get Model Confidence                                             |
//+------------------------------------------------------------------+
double CLLM_PPO_Model::GetModelConfidence(void)
{
    if(ArraySize(m_reward_history) < 10)
        return 0.5; // Default confidence
    
    double avg_reward = 0.0;
    for(int i = 0; i < 10; i++)
    {
        avg_reward += m_reward_history[i];
    }
    avg_reward /= 10.0;
    
    // Convert reward to confidence (0 to 1)
    return MathMax(0.1, MathMin(0.9, 0.5 + avg_reward));
}