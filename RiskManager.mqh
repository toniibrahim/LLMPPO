//+------------------------------------------------------------------+
//|                                                  RiskManager.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Advanced Risk Management Class                                   |
//+------------------------------------------------------------------+
class CRiskManager
{
private:
    // Risk parameters
    double            m_confidence_level;
    double            m_max_portfolio_risk;
    double            m_correlation_threshold;
    int               m_var_period;
    
    // Historical data
    double            m_returns_history[];
    double            m_price_history[];
    datetime          m_time_history[];
    int               m_history_size;
    
    // Risk metrics
    double            m_current_var;
    double            m_current_cvar;
    double            m_current_volatility;
    double            m_skewness;
    double            m_kurtosis;
    double            m_sharpe_ratio;
    double            m_sortino_ratio;
    double            m_max_drawdown;
    
    // Portfolio metrics
    double            m_portfolio_beta;
    double            m_portfolio_correlation;
    double            m_concentration_risk;
    
    // Risk limits
    struct RiskLimits
    {
        double max_position_size;
        double max_daily_loss;
        double max_weekly_loss;
        double max_monthly_loss;
        double max_var_limit;
        double max_cvar_limit;
        double max_volatility;
        double min_sharpe_ratio;
    };
    RiskLimits        m_limits;
    
    // Performance tracking
    struct PerformanceMetrics
    {
        double daily_pnl;
        double weekly_pnl;
        double monthly_pnl;
        double ytd_pnl;
        double hit_ratio;
        double avg_win;
        double avg_loss;
        double profit_factor;
        double recovery_factor;
    };
    PerformanceMetrics m_performance;

public:
    // Constructor/Destructor
                     CRiskManager(void);
                    ~CRiskManager(void);
    
    // Initialization
    bool              Initialize(double confidence_level = 0.95, int history_size = 252);
    void              SetRiskLimits(double max_pos_size, double max_daily, double max_var);
    
    // Data management
    bool              UpdatePriceData(double price, datetime time);
    bool              CalculateReturns(void);
    
    // VaR/CVaR calculations
    double            CalculateHistoricalVaR(double confidence = 0.0);
    double            CalculateParametricVaR(double confidence = 0.0);
    double            CalculateMonteCarloVaR(double confidence = 0.0, int simulations = 10000);
    double            CalculateCVaR(double confidence = 0.0);
    double            CalculateExpectedShortfall(double confidence = 0.0);
    
    // Advanced risk metrics
    double            CalculateVolatility(int period = 0);
    double            CalculateDownsideDeviation(double target_return = 0.0);
    double            CalculateSkewness(void);
    double            CalculateKurtosis(void);
    double            CalculateMaxDrawdown(void);
    double            CalculateCalmarRatio(void);
    
    // Performance ratios
    double            CalculateSharpeRatio(double risk_free_rate = 0.02);
    double            CalculateSortinoRatio(double target_return = 0.0);
    double            CalculateOmegaRatio(double threshold = 0.0);
    double            CalculateInformationRatio(double benchmark_return = 0.0);
    
    // Position sizing
    double            CalculateKellySize(double win_prob, double avg_win, double avg_loss);
    double            CalculateOptimalF(void);
    double            CalculateRiskParitySize(double volatility, double target_vol = 0.15);
    double            CalculateVaRBasedSize(double var_limit);
    
    // Risk monitoring
    bool              CheckRiskLimits(void);
    bool              IsPositionSizeAcceptable(double position_size, double entry_price);
    bool              IsVaRWithinLimits(void);
    bool              IsDrawdownAcceptable(void);
    
    // Correlation analysis
    double            CalculateCorrelation(double &data1[], double &data2[], int period);
    double            CalculateBeta(double &market_returns[], int period);
    bool              CheckConcentrationRisk(void);
    
    // Stress testing
    bool              PerformStressTest(double &shock_scenarios[][]);
    double            CalculateStressVaR(double stress_factor);
    bool              BacktestVaRModel(int test_period);
    
    // Monte Carlo simulation
    bool              RunMonteCarloSimulation(int num_paths, int time_horizon, 
                                            double initial_price, double &results[]);
    double            GenerateRandomReturn(double mean, double vol);
    
    // Portfolio risk aggregation
    double            CalculatePortfolioVaR(double &positions[], double &correlations[][]);
    double            CalculateMarginalVaR(int position_index);
    double            CalculateComponentVaR(int position_index);
    
    // Dynamic hedging
    double            CalculateHedgeRatio(double asset_vol, double hedge_vol, double correlation);
    bool              SuggestHedgingStrategy(double current_exposure);
    
    // Risk reporting
    string            GenerateRiskReport(void);
    bool              ExportRiskMetrics(string filename);
    void              LogRiskEvent(string event_type, string description, double value);
    
    // Real-time monitoring
    bool              MonitorIntraday(void);
    bool              CheckLiquidityRisk(void);
    double            CalculateMarketImpact(double order_size, double avg_volume);
    
    // Getters
    double            GetCurrentVaR(void) { return m_current_var; }
    double            GetCurrentCVaR(void) { return m_current_cvar; }
    double            GetCurrentVolatility(void) { return m_current_volatility; }
    double            GetSharpeRatio(void) { return m_sharpe_ratio; }
    double            GetSortinoRatio(void) { return m_sortino_ratio; }
    double            GetMaxDrawdown(void) { return m_max_drawdown; }
    double            GetSkewness(void) { return m_skewness; }
    double            GetKurtosis(void) { return m_kurtosis; }
    
    // Setters
    void              SetConfidenceLevel(double confidence) { m_confidence_level = confidence; }
    void              SetVaRPeriod(int period) { m_var_period = period; }
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CRiskManager::CRiskManager(void)
{
    m_confidence_level = 0.95;
    m_max_portfolio_risk = 0.02;
    m_correlation_threshold = 0.7;
    m_var_period = 20;
    m_history_size = 252;
    
    // Initialize risk metrics
    m_current_var = 0.0;
    m_current_cvar = 0.0;
    m_current_volatility = 0.0;
    m_skewness = 0.0;
    m_kurtosis = 0.0;
    m_sharpe_ratio = 0.0;
    m_sortino_ratio = 0.0;
    m_max_drawdown = 0.0;
    
    // Initialize risk limits
    m_limits.max_position_size = 0.1;
    m_limits.max_daily_loss = 0.02;
    m_limits.max_weekly_loss = 0.05;
    m_limits.max_monthly_loss = 0.10;
    m_limits.max_var_limit = 0.03;
    m_limits.max_cvar_limit = 0.05;
    m_limits.max_volatility = 0.25;
    m_limits.min_sharpe_ratio = 0.5;
    
    // Initialize performance metrics
    ZeroMemory(m_performance);
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CRiskManager::~CRiskManager(void)
{
    ArrayFree(m_returns_history);
    ArrayFree(m_price_history);
    ArrayFree(m_time_history);
}

//+------------------------------------------------------------------+
//| Initialize risk manager                                          |
//+------------------------------------------------------------------+
bool CRiskManager::Initialize(double confidence_level = 0.95, int history_size = 252)
{
    m_confidence_level = confidence_level;
    m_history_size = history_size;
    
    ArrayResize(m_returns_history, m_history_size);
    ArrayResize(m_price_history, m_history_size);
    ArrayResize(m_time_history, m_history_size);
    
    ArrayInitialize(m_returns_history, 0.0);
    ArrayInitialize(m_price_history, 0.0);
    ArrayInitialize(m_time_history, 0);
    
    return true;
}

//+------------------------------------------------------------------+
//| Update price data                                                |
//+------------------------------------------------------------------+
bool CRiskManager::UpdatePriceData(double price, datetime time)
{
    if(price <= 0) return false;
    
    // Shift arrays
    for(int i = m_history_size - 1; i > 0; i--)
    {
        m_price_history[i] = m_price_history[i-1];
        m_time_history[i] = m_time_history[i-1];
        if(i > 1) m_returns_history[i] = m_returns_history[i-1];
    }
    
    // Add new data
    m_price_history[0] = price;
    m_time_history[0] = time;
    
    // Calculate return if we have previous price
    if(m_price_history[1] > 0)
    {
        m_returns_history[0] = MathLog(price / m_price_history[1]);
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Calculate Historical VaR                                         |
//+------------------------------------------------------------------+
double CRiskManager::CalculateHistoricalVaR(double confidence = 0.0)
{
    if(confidence == 0.0) confidence = m_confidence_level;
    
    // Copy returns for sorting
    double sorted_returns[];
    int valid_returns = 0;
    
    // Count valid returns
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0) valid_returns++;
    }
    
    if(valid_returns < 10) return 0.0;
    
    ArrayResize(sorted_returns, valid_returns);
    ArrayCopy(sorted_returns, m_returns_history, 0, 0, valid_returns);
    ArraySort(sorted_returns);
    
    // Calculate VaR
    int var_index = (int)((1.0 - confidence) * valid_returns);
    m_current_var = -sorted_returns[var_index];
    
    return m_current_var;
}

//+------------------------------------------------------------------+
//| Calculate Parametric VaR                                         |
//+------------------------------------------------------------------+
double CRiskManager::CalculateParametricVaR(double confidence = 0.0)
{
    if(confidence == 0.0) confidence = m_confidence_level;
    
    // Calculate mean and standard deviation
    double mean = 0.0, variance = 0.0;
    int count = 0;
    
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            mean += m_returns_history[i];
            count++;
        }
    }
    
    if(count < 10) return 0.0;
    
    mean /= count;
    
    for(int i = 0; i < count; i++)
    {
        if(m_returns_history[i] != 0.0)
            variance += MathPow(m_returns_history[i] - mean, 2);
    }
    
    variance /= (count - 1);
    double std_dev = MathSqrt(variance);
    
    // Calculate Z-score for confidence level
    double z_score;
    if(confidence == 0.95) z_score = 1.645;
    else if(confidence == 0.99) z_score = 2.326;
    else z_score = 1.645; // Default
    
    m_current_var = -(mean - z_score * std_dev);
    
    return m_current_var;
}

//+------------------------------------------------------------------+
//| Calculate CVaR (Expected Shortfall)                             |
//+------------------------------------------------------------------+
double CRiskManager::CalculateCVaR(double confidence = 0.0)
{
    if(confidence == 0.0) confidence = m_confidence_level;
    
    // First calculate VaR
    double var = CalculateHistoricalVaR(confidence);
    
    // Calculate average of returns worse than VaR
    double sum = 0.0;
    int count = 0;
    
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0 && m_returns_history[i] <= -var)
        {
            sum += m_returns_history[i];
            count++;
        }
    }
    
    if(count > 0)
    {
        m_current_cvar = -sum / count;
    }
    else
    {
        m_current_cvar = var;
    }
    
    return m_current_cvar;
}

//+------------------------------------------------------------------+
//| Calculate volatility                                             |
//+------------------------------------------------------------------+
double CRiskManager::CalculateVolatility(int period = 0)
{
    if(period == 0) period = m_var_period;
    
    double mean = 0.0, variance = 0.0;
    int count = 0;
    
    for(int i = 0; i < MathMin(period, m_history_size - 1); i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            mean += m_returns_history[i];
            count++;
        }
    }
    
    if(count < 2) return 0.0;
    
    mean /= count;
    
    for(int i = 0; i < count; i++)
    {
        if(m_returns_history[i] != 0.0)
            variance += MathPow(m_returns_history[i] - mean, 2);
    }
    
    variance /= (count - 1);
    m_current_volatility = MathSqrt(variance) * MathSqrt(252); // Annualized
    
    return m_current_volatility;
}

//+------------------------------------------------------------------+
//| Calculate Sharpe Ratio                                           |
//+------------------------------------------------------------------+
double CRiskManager::CalculateSharpeRatio(double risk_free_rate = 0.02)
{
    double mean_return = 0.0;
    int count = 0;
    
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            mean_return += m_returns_history[i];
            count++;
        }
    }
    
    if(count == 0) return 0.0;
    
    mean_return = (mean_return / count) * 252; // Annualized
    double volatility = CalculateVolatility();
    
    if(volatility == 0.0) return 0.0;
    
    m_sharpe_ratio = (mean_return - risk_free_rate) / volatility;
    
    return m_sharpe_ratio;
}

//+------------------------------------------------------------------+
//| Calculate Downside Deviation                                     |
//+------------------------------------------------------------------+
double CRiskManager::CalculateDownsideDeviation(double target_return = 0.0)
{
    double sum_squared_deviations = 0.0;
    int count = 0;
    
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            double annualized_return = m_returns_history[i] * 252;
            if(annualized_return < target_return)
            {
                double deviation = target_return - annualized_return;
                sum_squared_deviations += deviation * deviation;
            }
            count++;
        }
    }
    
    if(count == 0) return 0.0;
    
    double downside_variance = sum_squared_deviations / count;
    return MathSqrt(downside_variance);
}

//+------------------------------------------------------------------+
//| Calculate Sortino Ratio                                          |
//+------------------------------------------------------------------+
double CRiskManager::CalculateSortinoRatio(double target_return = 0.0)
{
    double mean_return = 0.0;
    int count = 0;
    
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            mean_return += m_returns_history[i];
            count++;
        }
    }
    
    if(count == 0) return 0.0;
    
    mean_return = (mean_return / count) * 252; // Annualized
    double downside_deviation = CalculateDownsideDeviation(target_return);
    
    if(downside_deviation == 0.0) return 0.0;
    
    m_sortino_ratio = (mean_return - target_return) / downside_deviation;
    
    return m_sortino_ratio;
}

//+------------------------------------------------------------------+
//| Calculate position size using Kelly Criterion                    |
//+------------------------------------------------------------------+
double CRiskManager::CalculateKellySize(double win_prob, double avg_win, double avg_loss)
{
    if(avg_loss == 0.0 || win_prob <= 0.0 || win_prob >= 1.0) return 0.0;
    
    double loss_prob = 1.0 - win_prob;
    double win_loss_ratio = avg_win / MathAbs(avg_loss);
    
    double kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio;
    
    // Apply conservative scaling (typically 0.25 to 0.5 of full Kelly)
    kelly_fraction *= 0.25;
    
    return MathMax(0.0, MathMin(0.2, kelly_fraction)); // Cap at 20%
}

//+------------------------------------------------------------------+
//| Check risk limits                                                |
//+------------------------------------------------------------------+
bool CRiskManager::CheckRiskLimits(void)
{
    // Update current metrics
    CalculateHistoricalVaR();
    CalculateCVaR();
    CalculateVolatility();
    
    // Check VaR limit
    if(m_current_var > m_limits.max_var_limit)
    {
        Print("VaR limit exceeded: ", m_current_var, " > ", m_limits.max_var_limit);
        return false;
    }
    
    // Check CVaR limit
    if(m_current_cvar > m_limits.max_cvar_limit)
    {
        Print("CVaR limit exceeded: ", m_current_cvar, " > ", m_limits.max_cvar_limit);
        return false;
    }
    
    // Check volatility limit
    if(m_current_volatility > m_limits.max_volatility)
    {
        Print("Volatility limit exceeded: ", m_current_volatility, " > ", m_limits.max_volatility);
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Generate comprehensive risk report                               |
//+------------------------------------------------------------------+
string CRiskManager::GenerateRiskReport(void)
{
    string report = "=== RISK MANAGEMENT REPORT ===\n";
    report += "Timestamp: " + TimeToString(TimeCurrent()) + "\n\n";
    
    report += "--- RISK METRICS ---\n";
    report += "VaR (" + DoubleToString(m_confidence_level*100, 1) + "%): " + 
              DoubleToString(m_current_var*100, 3) + "%\n";
    report += "CVaR: " + DoubleToString(m_current_cvar*100, 3) + "%\n";
    report += "Volatility (Ann.): " + DoubleToString(m_current_volatility*100, 2) + "%\n";
    report += "Skewness: " + DoubleToString(m_skewness, 3) + "\n";
    report += "Kurtosis: " + DoubleToString(m_kurtosis, 3) + "\n";
    
    report += "\n--- PERFORMANCE RATIOS ---\n";
    report += "Sharpe Ratio: " + DoubleToString(m_sharpe_ratio, 3) + "\n";
    report += "Sortino Ratio: " + DoubleToString(m_sortino_ratio, 3) + "\n";
    report += "Max Drawdown: " + DoubleToString(m_max_drawdown*100, 2) + "%\n";
    
    report += "\n--- RISK LIMITS STATUS ---\n";
    report += "VaR Limit: " + (m_current_var <= m_limits.max_var_limit ? "OK" : "BREACH") + "\n";
    report += "CVaR Limit: " + (m_current_cvar <= m_limits.max_cvar_limit ? "OK" : "BREACH") + "\n";
    report += "Vol Limit: " + (m_current_volatility <= m_limits.max_volatility ? "OK" : "BREACH") + "\n";
    
    return report;
}

//+------------------------------------------------------------------+
//| Set risk limits                                                  |
//+------------------------------------------------------------------+
void CRiskManager::SetRiskLimits(double max_pos_size, double max_daily, double max_var)
{
    m_limits.max_position_size = max_pos_size;
    m_limits.max_daily_loss = max_daily;
    m_limits.max_var_limit = max_var;
}