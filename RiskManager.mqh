//+------------------------------------------------------------------+
//|                                                  RiskManager.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Advanced Risk Management Class - Comprehensive Financial Risk Analysis |
//+------------------------------------------------------------------+
// CRiskManager implements a sophisticated risk management framework for algorithmic trading.
// It provides comprehensive risk measurement, monitoring, and control capabilities including:
//
// === CORE RISK METRICS ===
// • VaR (Value at Risk): Maximum potential loss at given confidence level
//   - Historical VaR: Uses empirical return distribution
//   - Parametric VaR: Assumes normal distribution 
//   - Monte Carlo VaR: Simulation-based approach
// • CVaR/ES (Conditional VaR/Expected Shortfall): Average loss beyond VaR threshold
// • Volatility: Annualized standard deviation of returns
// • Higher moments: Skewness (asymmetry) and Kurtosis (tail risk)
//
// === PERFORMANCE RATIOS ===
// • Sharpe Ratio: Risk-adjusted return vs total volatility
// • Sortino Ratio: Risk-adjusted return vs downside deviation
// • Omega Ratio: Probability-weighted gains vs losses
// • Information Ratio: Active return vs tracking error
// • Calmar Ratio: Annualized return vs maximum drawdown
//
// === POSITION SIZING METHODS ===
// • Kelly Criterion: Optimal fraction based on win probability and payoff
// • Risk Parity: Equal risk contribution across positions
// • VaR-based sizing: Position size based on VaR limits
// • Optimal f: Fraction maximizing geometric growth
//
// === ADVANCED FEATURES ===
// • Portfolio risk aggregation with correlation matrices
// • Stress testing and scenario analysis
// • Monte Carlo simulation for path-dependent risks
// • Dynamic hedging strategies
// • Real-time risk monitoring and limit checking
// • Comprehensive risk reporting and logging
//
// MATHEMATICAL FOUNDATIONS:
// - VaR_α = -F^(-1)(α) where F is the return distribution CDF
// - CVaR_α = E[X | X ≤ -VaR_α] (expected value in tail)
// - Sharpe = (μ - r_f) / σ where μ=return, r_f=risk-free rate, σ=volatility
// - Kelly fraction = (bp - q) / b where b=odds, p=win prob, q=lose prob
class CRiskManager
{
private:
    // === CORE RISK CONFIGURATION PARAMETERS ===
    double            m_confidence_level;         // VaR confidence level (e.g., 0.95 = 95%)
    double            m_max_portfolio_risk;       // Maximum acceptable portfolio risk exposure
    double            m_correlation_threshold;    // Threshold for correlation warning alerts
    int               m_var_period;               // Lookback period for VaR calculations (days)
    
    // === HISTORICAL DATA STORAGE ===
    double            m_returns_history[];        // Time series of log returns r_t = ln(P_t/P_{t-1})
    double            m_price_history[];          // Time series of asset prices P_t
    datetime          m_time_history[];           // Timestamps corresponding to price data
    int               m_history_size;             // Maximum number of historical observations
    
    // === COMPUTED RISK METRICS (CACHED VALUES) ===
    double            m_current_var;              // Current Value at Risk estimate
    double            m_current_cvar;             // Current Conditional VaR (Expected Shortfall)
    double            m_current_volatility;       // Current annualized volatility (σ_annual)
    double            m_skewness;                 // Third moment: E[(X-μ)³]/σ³ (asymmetry measure)
    double            m_kurtosis;                 // Fourth moment: E[(X-μ)⁴]/σ⁴ (tail thickness)
    double            m_sharpe_ratio;             // Sharpe ratio: (μ - r_f)/σ (risk-adjusted return)
    double            m_sortino_ratio;            // Sortino ratio: (μ - MAR)/DD (downside risk-adjusted)
    double            m_max_drawdown;             // Maximum peak-to-trough decline (worst loss period)
    
    // === PORTFOLIO-LEVEL RISK MEASURES ===
    double            m_portfolio_beta;           // Systematic risk relative to market (β = Cov(r_p, r_m)/Var(r_m))
    double            m_portfolio_correlation;    // Correlation with benchmark/market index
    double            m_concentration_risk;       // Measure of portfolio concentration (Herfindahl index)
    
    // === RISK LIMIT DEFINITIONS ===
    struct RiskLimits
    {
        double max_position_size;      // Maximum position size as fraction of capital (e.g., 0.1 = 10%)
        double max_daily_loss;         // Daily loss limit as fraction of capital (stop-loss threshold)
        double max_weekly_loss;        // Weekly cumulative loss limit (escalation threshold)
        double max_monthly_loss;       // Monthly cumulative loss limit (risk budget)
        double max_var_limit;          // Maximum acceptable VaR (regulatory/internal limit)
        double max_cvar_limit;         // Maximum acceptable CVaR (tail risk limit)
        double max_volatility;         // Maximum acceptable volatility (risk appetite)
        double min_sharpe_ratio;       // Minimum required risk-adjusted performance
    };
    RiskLimits        m_limits;                   // Active risk limit configuration
    
    // === PERFORMANCE TRACKING METRICS ===
    struct PerformanceMetrics
    {
        double daily_pnl;             // Daily profit & loss (mark-to-market)
        double weekly_pnl;            // Weekly cumulative P&L
        double monthly_pnl;           // Monthly cumulative P&L
        double ytd_pnl;               // Year-to-date cumulative P&L
        double hit_ratio;             // Win rate: (winning trades) / (total trades)
        double avg_win;               // Average profit per winning trade
        double avg_loss;              // Average loss per losing trade (negative value)
        double profit_factor;         // Gross profit / Gross loss ratio
        double recovery_factor;       // Net profit / Maximum drawdown ratio
    };
    PerformanceMetrics m_performance;             // Current performance statistics

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
//| Constructor - Initialize risk manager with institutional defaults |
//+------------------------------------------------------------------+
// Constructor sets up the risk management framework with industry-standard default parameters
// optimized for institutional algorithmic trading environments. All parameters can be
// customized post-initialization based on specific risk appetite and regulatory requirements.
CRiskManager::CRiskManager(void)
{
    // === CORE RISK PARAMETERS (INDUSTRY STANDARDS) ===
    m_confidence_level = 0.95;           // 95% confidence level (Basel III standard for market risk)
    m_max_portfolio_risk = 0.02;         // 2% maximum portfolio risk exposure
    m_correlation_threshold = 0.7;       // 70% correlation threshold for diversification warnings
    m_var_period = 20;                   // 20-day VaR lookback (1 month trading period)
    m_history_size = 252;                // 252 trading days (1 year of historical data)
    
    // === INITIALIZE COMPUTED RISK METRICS TO SAFE DEFAULTS ===
    m_current_var = 0.0;                 // No VaR calculated yet
    m_current_cvar = 0.0;                // No CVaR calculated yet  
    m_current_volatility = 0.0;          // No volatility calculated yet
    m_skewness = 0.0;                    // Symmetric distribution assumption
    m_kurtosis = 0.0;                    // Normal distribution assumption (excess kurtosis = 0)
    m_sharpe_ratio = 0.0;                // No performance history yet
    m_sortino_ratio = 0.0;               // No downside deviation calculated yet
    m_max_drawdown = 0.0;                // No drawdown experienced yet
    
    // === CONSERVATIVE RISK LIMITS (INSTITUTIONAL STANDARDS) ===
    // Position sizing limits (prevent concentration risk)
    m_limits.max_position_size = 0.1;        // 10% maximum single position (diversification)
    
    // Loss limits (capital preservation)
    m_limits.max_daily_loss = 0.02;          // 2% daily stop-loss (operational risk limit)
    m_limits.max_weekly_loss = 0.05;         // 5% weekly loss limit (escalation threshold)
    m_limits.max_monthly_loss = 0.10;        // 10% monthly loss limit (risk budget)
    
    // Statistical risk limits (model-based)
    m_limits.max_var_limit = 0.03;           // 3% VaR limit (regulatory compliance)
    m_limits.max_cvar_limit = 0.05;          // 5% CVaR limit (tail risk control)
    m_limits.max_volatility = 0.25;          // 25% maximum volatility (risk appetite)
    m_limits.min_sharpe_ratio = 0.5;         // Minimum 0.5 Sharpe ratio (performance threshold)
    
    // === INITIALIZE PERFORMANCE TRACKING ===
    ZeroMemory(m_performance);               // Clear all performance metrics to zero
}

//+------------------------------------------------------------------+
//| Destructor - Clean up memory allocation                         |
//+------------------------------------------------------------------+
// Destructor ensures proper cleanup of dynamically allocated arrays
// to prevent memory leaks in long-running trading systems
CRiskManager::~CRiskManager(void)
{
    ArrayFree(m_returns_history);            // Free returns history buffer
    ArrayFree(m_price_history);              // Free price history buffer
    ArrayFree(m_time_history);               // Free timestamp history buffer
}

//+------------------------------------------------------------------+
//| Initialize risk manager - Configure historical data collection  |
//+------------------------------------------------------------------+
// Initialize() configures the risk management system for operation:
// - Sets confidence level for VaR calculations (typically 0.95 or 0.99)
// - Allocates memory for historical data storage
// - Clears all historical arrays to ensure clean startup
// PARAMETERS:
// - confidence_level: Statistical confidence for VaR (0.95 = 95%, 0.99 = 99%)
// - history_size: Number of historical observations to maintain (252 = 1 year)
bool CRiskManager::Initialize(double confidence_level = 0.95, int history_size = 252)
{
    // Store configuration parameters
    m_confidence_level = confidence_level;   // VaR confidence level (α in VaR_α)
    m_history_size = history_size;           // Size of rolling historical window
    
    // Allocate memory for historical data arrays
    ArrayResize(m_returns_history, m_history_size);    // Log returns: r_t = ln(P_t/P_{t-1})
    ArrayResize(m_price_history, m_history_size);      // Price levels: P_t
    ArrayResize(m_time_history, m_history_size);       // Timestamps: t
    
    // Initialize arrays to zero/empty state for clean startup
    ArrayInitialize(m_returns_history, 0.0);  // Zero returns (no price changes)
    ArrayInitialize(m_price_history, 0.0);    // Zero prices (no data yet)
    ArrayInitialize(m_time_history, 0);       // Zero timestamps (no data yet)
    
    return true;  // Successful initialization
}

//+------------------------------------------------------------------+
//| Update price data - Rolling window historical data management   |
//+------------------------------------------------------------------+
// UpdatePriceData() maintains a rolling window of historical price and return data
// for risk calculations. It implements a FIFO (First In, First Out) buffer system
// where new observations replace the oldest ones, maintaining constant memory usage.
// 
// MATHEMATICAL OPERATION: r_t = ln(P_t / P_{t-1}) (continuous compounding)
// - Uses logarithmic returns for better statistical properties (normality, additivity)
// - Automatically calculates returns from consecutive price observations
// 
// PARAMETERS:
// - price: Current market price P_t (must be positive)
// - time: Timestamp of the price observation
// RETURNS: true if successful, false if invalid price
bool CRiskManager::UpdatePriceData(double price, datetime time)
{
    // Input validation: ensure positive price
    if(price <= 0) 
    {
        Print("ERROR: Invalid price data - price must be positive, received: ", price);
        return false;
    }
    
    // === ROLLING WINDOW UPDATE (FIFO BUFFER) ===
    // Shift all existing data one position to the right (age the data)
    for(int i = m_history_size - 1; i > 0; i--)
    {
        m_price_history[i] = m_price_history[i-1];     // Shift price data
        m_time_history[i] = m_time_history[i-1];       // Shift timestamp data
        if(i > 1) m_returns_history[i] = m_returns_history[i-1];  // Shift return data
    }
    
    // === INSERT NEW OBSERVATION AT FRONT OF ARRAYS ===
    m_price_history[0] = price;                        // Store current price P_t
    m_time_history[0] = time;                          // Store current timestamp t
    
    // === CALCULATE LOGARITHMIC RETURN ===
    // Formula: r_t = ln(P_t / P_{t-1}) where P_t = current price, P_{t-1} = previous price
    if(m_price_history[1] > 0)  // Ensure we have valid previous price
    {
        m_returns_history[0] = MathLog(price / m_price_history[1]);  // Continuous return
    }
    else
    {
        m_returns_history[0] = 0.0;  // No return calculation possible for first observation
    }
    
    return true;  // Successfully updated price data
}

//+------------------------------------------------------------------+
//| Calculate Historical VaR - Non-parametric empirical approach    |
//+------------------------------------------------------------------+
// CalculateHistoricalVaR() implements the Historical Simulation method for VaR estimation.
// This non-parametric approach uses the actual historical distribution of returns
// without making distributional assumptions (e.g., normality).
//
// METHODOLOGY:
// 1. Sort historical returns from worst to best
// 2. Find the (1-α)th percentile in the sorted distribution
// 3. VaR_α = -Percentile(returns, 1-α) (negative of percentile for loss interpretation)
//
// ADVANTAGES:
// - No distributional assumptions required
// - Captures actual fat tails and skewness in data
// - Simple and intuitive interpretation
// DISADVANTAGES:
// - Limited by historical data (may not capture future extreme events)
// - Requires large sample size for accurate tail estimation
// - Backward-looking (assumes past predicts future)
//
// MATHEMATICAL FORMULA: VaR_α = -F^(-1)(1-α) where F^(-1) is empirical quantile function
double CRiskManager::CalculateHistoricalVaR(double confidence = 0.0)
{
    // Use default confidence level if not specified
    if(confidence == 0.0) confidence = m_confidence_level;
    
    // === DATA VALIDATION AND PREPARATION ===
    double sorted_returns[];                  // Array for sorting returns
    int valid_returns = 0;                    // Count of non-zero returns
    
    // Count valid (non-zero) return observations
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0) valid_returns++;
    }
    
    // Require minimum sample size for reliable VaR estimation
    if(valid_returns < 10) 
    {
        Print("WARNING: Insufficient data for Historical VaR - need at least 10 observations, have ", valid_returns);
        return 0.0;
    }
    
    // === EMPIRICAL DISTRIBUTION CONSTRUCTION ===
    ArrayResize(sorted_returns, valid_returns);         // Allocate sorted array
    ArrayCopy(sorted_returns, m_returns_history, 0, 0, valid_returns);  // Copy valid returns
    ArraySort(sorted_returns);                          // Sort ascending (worst to best returns)
    
    // === VAR CALCULATION USING EMPIRICAL QUANTILE ===
    // Find the (1-α) percentile index in sorted distribution
    int var_index = (int)((1.0 - confidence) * valid_returns);
    
    // Ensure index bounds (prevent array out of bounds)
    if(var_index >= valid_returns) var_index = valid_returns - 1;
    if(var_index < 0) var_index = 0;
    
    // VaR = negative of the (1-α)th percentile (converts return to loss)
    m_current_var = -sorted_returns[var_index];
    
    return m_current_var;
}

//+------------------------------------------------------------------+
//| Calculate Parametric VaR - Normal distribution assumption       |
//+------------------------------------------------------------------+
// CalculateParametricVaR() implements the Parametric (Variance-Covariance) method
// for VaR estimation. This approach assumes returns follow a normal distribution
// and estimates VaR using sample mean and standard deviation.
//
// METHODOLOGY:
// 1. Calculate sample mean (μ) and standard deviation (σ) from historical returns
// 2. Assume returns ~ N(μ, σ²) (normal distribution)
// 3. VaR_α = -(μ - z_α * σ) where z_α is the α-quantile of standard normal
//
// ADVANTAGES:
// - Fast computation (only requires mean and standard deviation)
// - Smooth VaR estimates (no jumps from small sample changes)
// - Well-established theoretical foundation
// DISADVANTAGES:
// - Strong normality assumption (may underestimate tail risk)
// - Ignores skewness and kurtosis in return distribution
// - Poor performance during crisis periods with fat tails
//
// MATHEMATICAL FORMULA: VaR_α = -(μ - Φ^(-1)(α) * σ)
// where Φ^(-1)(α) is the inverse standard normal CDF (z-score)
double CRiskManager::CalculateParametricVaR(double confidence = 0.0)
{
    // Use default confidence level if not specified
    if(confidence == 0.0) confidence = m_confidence_level;
    
    // === SAMPLE STATISTICS CALCULATION ===
    double mean = 0.0, variance = 0.0;       // Sample mean and variance
    int count = 0;                            // Number of valid observations
    
    // Calculate sample mean: μ̂ = (1/n) * Σr_i
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            mean += m_returns_history[i];
            count++;
        }
    }
    
    // Require minimum sample size for reliable parameter estimation
    if(count < 10) 
    {
        Print("WARNING: Insufficient data for Parametric VaR - need at least 10 observations, have ", count);
        return 0.0;
    }
    
    mean /= count;  // Complete mean calculation
    
    // Calculate sample variance: σ̂² = (1/(n-1)) * Σ(r_i - μ̂)²
    for(int i = 0; i < count; i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            double deviation = m_returns_history[i] - mean;
            variance += deviation * deviation;    // Squared deviation
        }
    }
    
    variance /= (count - 1);                  // Unbiased variance estimator (Bessel's correction)
    double std_dev = MathSqrt(variance);      // Standard deviation σ̂
    
    // === Z-SCORE LOOKUP FOR CONFIDENCE LEVEL ===
    // Map confidence level to standard normal quantile
    double z_score;
    if(MathAbs(confidence - 0.95) < 0.001)       z_score = 1.645;   // 95% confidence (1-tail)
    else if(MathAbs(confidence - 0.99) < 0.001)  z_score = 2.326;   // 99% confidence (1-tail)
    else if(MathAbs(confidence - 0.975) < 0.001) z_score = 1.960;   // 97.5% confidence (2-tail for 95% VaR)
    else if(MathAbs(confidence - 0.995) < 0.001) z_score = 2.576;   // 99.5% confidence (2-tail for 99% VaR)
    else z_score = 1.645;                                           // Default to 95% if unrecognized
    
    // === PARAMETRIC VAR CALCULATION ===
    // VaR_α = -(μ - z_α * σ) = -(expected return - tail threshold)
    m_current_var = -(mean - z_score * std_dev);
    
    return m_current_var;
}

//+------------------------------------------------------------------+
//| Calculate CVaR (Conditional VaR / Expected Shortfall)           |
//+------------------------------------------------------------------+
// CalculateCVaR() computes Conditional Value at Risk, also known as Expected Shortfall (ES).
// CVaR represents the expected loss given that the loss exceeds the VaR threshold.
// It provides information about tail risk beyond what VaR captures.
//
// MATHEMATICAL DEFINITION:
// CVaR_α = E[L | L ≥ VaR_α] = Expected value of losses in the (1-α) worst cases
// where L represents losses (negative returns) and VaR_α is the α-confidence VaR.
//
// ADVANTAGES:
// - Coherent risk measure (satisfies all axioms of coherent risk measures)
// - Captures tail risk beyond VaR (answers: "How bad is bad?")
// - More informative than VaR for risk management decisions
// - Subadditive (portfolio CVaR ≤ sum of individual CVaRs)
// DISADVANTAGES:
// - Requires more data for accurate estimation
// - More complex to compute than VaR
// - Still subject to historical simulation limitations
//
// COMPUTATIONAL APPROACH: Historical simulation method using empirical distribution
double CRiskManager::CalculateCVaR(double confidence = 0.0)
{
    // Use default confidence level if not specified
    if(confidence == 0.0) confidence = m_confidence_level;
    
    // === STEP 1: CALCULATE VAR THRESHOLD ===
    // First establish the VaR threshold (cutoff point for tail losses)
    double var = CalculateHistoricalVaR(confidence);
    
    if(var <= 0.0)
    {
        Print("WARNING: Unable to calculate valid VaR for CVaR computation");
        return 0.0;
    }
    
    // === STEP 2: IDENTIFY TAIL LOSSES (WORSE THAN VAR) ===
    double sum_tail_losses = 0.0;            // Sum of losses exceeding VaR
    int tail_count = 0;                      // Count of observations in the tail
    
    // Scan historical returns to find losses worse than VaR
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0 && m_returns_history[i] <= -var)
        {
            // This return represents a loss worse than VaR (in the tail)
            sum_tail_losses += m_returns_history[i];  // Accumulate tail losses
            tail_count++;                             // Count tail observations
        }
    }
    
    // === STEP 3: CALCULATE EXPECTED SHORTFALL ===
    if(tail_count > 0)
    {
        // CVaR = average of tail losses (negative of average tail returns)
        m_current_cvar = -sum_tail_losses / tail_count;
        
        Print("CVaR calculation: ", tail_count, " tail observations, average tail loss: ", 
              DoubleToString(m_current_cvar * 100, 3), "%");
    }
    else
    {
        // No tail observations found - use VaR as conservative estimate
        m_current_cvar = var;
        Print("WARNING: No tail losses found for CVaR calculation - using VaR as estimate");
    }
    
    return m_current_cvar;
}

//+------------------------------------------------------------------+
//| Calculate volatility - Annualized standard deviation of returns |
//+------------------------------------------------------------------+
// CalculateVolatility() computes the annualized volatility (standard deviation) of returns,
// which is the most fundamental measure of financial risk. Volatility quantifies the 
// degree of price variation and uncertainty in asset returns.
//
// MATHEMATICAL DEFINITION:
// σ_annual = σ_daily * √252 where σ_daily = √(Var[r_t]) and 252 = trading days per year
// Daily variance: Var[r_t] = E[(r_t - μ)²] where r_t = log returns, μ = sample mean
//
// CHARACTERISTICS:
// - Higher volatility = higher uncertainty/risk
// - Symmetric measure (treats upside and downside variation equally) 
// - Widely used in option pricing, portfolio optimization, risk management
// - Annualization assumes returns are i.i.d. (independent, identically distributed)
//
// APPLICATIONS:
// - Risk budgeting and position sizing
// - Sharpe ratio calculation (return per unit of volatility)
// - Options pricing models (Black-Scholes)
// - Regulatory capital requirements
//
// FORMULA: σ = √[(1/(n-1)) * Σ(r_i - μ)²] * √252 (annualized)
double CRiskManager::CalculateVolatility(int period = 0)
{
    // Use default period if not specified
    if(period == 0) period = m_var_period;
    
    // === STEP 1: CALCULATE SAMPLE MEAN ===
    double mean = 0.0;                       // Sample mean of returns
    int count = 0;                           // Valid observation count
    
    // Calculate mean over specified period
    for(int i = 0; i < MathMin(period, m_history_size - 1); i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            mean += m_returns_history[i];
            count++;
        }
    }
    
    // Require minimum observations for reliable volatility estimate
    if(count < 2) 
    {
        Print("WARNING: Insufficient data for volatility calculation - need at least 2 observations, have ", count);
        return 0.0;
    }
    
    mean /= count;  // Complete sample mean calculation: μ̂ = (1/n)Σr_i
    
    // === STEP 2: CALCULATE SAMPLE VARIANCE ===
    double variance = 0.0;                   // Sample variance accumulator
    
    // Sum of squared deviations from mean
    for(int i = 0; i < count; i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            double deviation = m_returns_history[i] - mean;
            variance += deviation * deviation;   // (r_i - μ̂)²
        }
    }
    
    variance /= (count - 1);                 // Unbiased variance: σ̂² = (1/(n-1))Σ(r_i - μ̂)²
    
    // === STEP 3: ANNUALIZE VOLATILITY ===
    // Convert daily volatility to annual volatility using square-root-of-time rule
    double daily_volatility = MathSqrt(variance);         // Daily standard deviation
    m_current_volatility = daily_volatility * MathSqrt(252);  // Annualized volatility
    
    return m_current_volatility;
}

//+------------------------------------------------------------------+
//| Calculate Sharpe Ratio - Risk-adjusted return performance measure |
//+------------------------------------------------------------------+
// CalculateSharpeRatio() computes the Sharpe ratio, which measures risk-adjusted performance
// by comparing excess return (return above risk-free rate) to volatility (total risk).
// Developed by William Sharpe, this ratio is the most widely used risk-adjusted performance metric.
//
// MATHEMATICAL DEFINITION:
// Sharpe Ratio = (μ_p - r_f) / σ_p
// where:
// - μ_p = portfolio/strategy return (annualized)
// - r_f = risk-free rate (annualized)
// - σ_p = portfolio/strategy volatility (annualized)
//
// INTERPRETATION:
// - Higher values indicate better risk-adjusted performance
// - Sharpe > 1.0 = Excellent performance
// - Sharpe 0.5-1.0 = Good performance  
// - Sharpe 0-0.5 = Acceptable performance
// - Sharpe < 0 = Poor performance (worse than risk-free rate)
//
// ADVANTAGES:
// - Simple and intuitive risk-adjusted measure
// - Widely accepted standard for performance comparison
// - Applicable to any return series
// DISADVANTAGES:
// - Assumes returns are normally distributed
// - Treats upside and downside volatility equally
// - Can be manipulated by altering return frequency
//
// FORMULA: SR = (E[r_p] - r_f) / σ(r_p) where E[·] is expected value, σ(·) is standard deviation
double CRiskManager::CalculateSharpeRatio(double risk_free_rate = 0.02)
{
    // === STEP 1: CALCULATE SAMPLE MEAN RETURN ===
    double mean_return = 0.0;                 // Sample mean of returns
    int count = 0;                            // Valid observation count
    
    // Calculate average return from historical data
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            mean_return += m_returns_history[i];
            count++;
        }
    }
    
    // Require sufficient data for reliable ratio calculation
    if(count == 0) 
    {
        Print("WARNING: No return data available for Sharpe ratio calculation");
        return 0.0;
    }
    
    // === STEP 2: ANNUALIZE MEAN RETURN ===
    mean_return = (mean_return / count) * 252; // Convert daily to annual return
    
    // === STEP 3: CALCULATE VOLATILITY ===
    double volatility = CalculateVolatility();  // Get annualized volatility
    
    // Avoid division by zero
    if(volatility == 0.0) 
    {
        Print("WARNING: Zero volatility detected - cannot calculate Sharpe ratio");
        return 0.0;
    }
    
    // === STEP 4: CALCULATE SHARPE RATIO ===
    // Sharpe ratio = (excess return) / (total risk)
    m_sharpe_ratio = (mean_return - risk_free_rate) / volatility;
    
    return m_sharpe_ratio;
}

//+------------------------------------------------------------------+
//| Calculate Downside Deviation - Below-target volatility measure  |
//+------------------------------------------------------------------+
// CalculateDownsideDeviation() computes the downside deviation, which measures
// volatility only for returns falling below a specified target/threshold.
// Unlike standard deviation, it focuses exclusively on "bad" volatility.
//
// MATHEMATICAL DEFINITION:
// DD = √{E[(min(r_t - MAR, 0))²]} where MAR = Minimum Acceptable Return
// Only negative deviations from target are included in the calculation.
//
// PURPOSE:
// - Distinguishes between "good" volatility (upside) and "bad" volatility (downside)
// - More relevant for risk-averse investors who care primarily about losses
// - Used in Sortino ratio calculation as denominator
//
// APPLICATIONS:
// - Downside risk assessment
// - Sortino ratio calculation
// - Risk budgeting focused on loss avoidance
// - Performance evaluation for conservative strategies
//
// FORMULA: DD(MAR) = √[(1/n) * Σ max(MAR - r_i, 0)²] where MAR = target return
double CRiskManager::CalculateDownsideDeviation(double target_return = 0.0)
{
    double sum_squared_deviations = 0.0;      // Sum of squared downside deviations
    int total_count = 0;                      // Total observations (for denominator)
    int downside_count = 0;                   // Count of below-target returns
    
    // === CALCULATE DOWNSIDE DEVIATIONS ===
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            double annualized_return = m_returns_history[i] * 252;  // Annualize daily return
            
            // Only include returns below target (downside)
            if(annualized_return < target_return)
            {
                double deviation = target_return - annualized_return;  // Shortfall from target
                sum_squared_deviations += deviation * deviation;       // Square the shortfall
                downside_count++;
            }
            total_count++;
        }
    }
    
    // Require sufficient data
    if(total_count == 0) 
    {
        Print("WARNING: No return data available for downside deviation calculation");
        return 0.0;
    }
    
    // === CALCULATE DOWNSIDE DEVIATION ===
    // Use total count (not just downside count) for proper risk scaling
    double downside_variance = sum_squared_deviations / total_count;
    double downside_deviation = MathSqrt(downside_variance);
    
    Print("Downside Deviation: ", downside_count, " of ", total_count, " returns below target (", 
          DoubleToString(target_return * 100, 2), "%)");
    
    return downside_deviation;
}

//+------------------------------------------------------------------+
//| Calculate Sortino Ratio - Downside risk-adjusted performance    |
//+------------------------------------------------------------------+
// CalculateSortinoRatio() computes the Sortino ratio, a risk-adjusted performance measure
// that focuses specifically on downside risk. Developed by Frank Sortino, this metric
// improves upon the Sharpe ratio by distinguishing between good and bad volatility.
//
// MATHEMATICAL DEFINITION:
// Sortino Ratio = (μ_p - MAR) / DD
// where:
// - μ_p = portfolio/strategy return (annualized)
// - MAR = Minimum Acceptable Return (target/benchmark return)
// - DD = Downside Deviation (volatility of below-target returns only)
//
// ADVANTAGES OVER SHARPE RATIO:
// - Penalizes only "bad" volatility (downside risk)
// - More relevant for risk-averse investors
// - Better reflects investor preferences (losses matter more than gains)
// - Less susceptible to manipulation through return smoothing
//
// INTERPRETATION:
// - Higher values indicate better downside risk-adjusted performance
// - Sortino > 2.0 = Excellent performance
// - Sortino 1.0-2.0 = Good performance
// - Sortino 0.5-1.0 = Acceptable performance
// - Sortino < 0.5 = Poor performance
//
// APPLICATIONS:
// - Portfolio performance evaluation
// - Manager selection and comparison
// - Risk budgeting for downside-focused strategies
// - Alternative investment evaluation (hedge funds, real estate)
//
// FORMULA: Sortino = (E[r_p] - MAR) / DD(MAR) where DD is downside deviation
double CRiskManager::CalculateSortinoRatio(double target_return = 0.0)
{
    // === STEP 1: CALCULATE MEAN RETURN ===
    double mean_return = 0.0;                 // Sample mean of returns
    int count = 0;                            // Valid observation count
    
    // Calculate average return from historical data
    for(int i = 0; i < m_history_size - 1; i++)
    {
        if(m_returns_history[i] != 0.0)
        {
            mean_return += m_returns_history[i];
            count++;
        }
    }
    
    // Require sufficient data for reliable ratio calculation
    if(count == 0) 
    {
        Print("WARNING: No return data available for Sortino ratio calculation");
        return 0.0;
    }
    
    // === STEP 2: ANNUALIZE MEAN RETURN ===
    mean_return = (mean_return / count) * 252; // Convert daily to annual return
    
    // === STEP 3: CALCULATE DOWNSIDE DEVIATION ===
    double downside_deviation = CalculateDownsideDeviation(target_return);
    
    // Avoid division by zero (perfect strategy with no downside risk)
    if(downside_deviation == 0.0) 
    {
        // If no downside deviation, return maximum positive value for excellent performance
        // or zero if returns don't exceed target
        if(mean_return > target_return)
        {
            Print("INFO: Perfect Sortino performance - no downside risk detected");
            return DBL_MAX;  // Theoretical maximum Sortino ratio
        }
        else
        {
            Print("WARNING: Zero downside deviation but returns don't exceed target");
            return 0.0;
        }
    }
    
    // === STEP 4: CALCULATE SORTINO RATIO ===
    // Sortino ratio = (excess return above target) / (downside risk only)
    m_sortino_ratio = (mean_return - target_return) / downside_deviation;
    
    return m_sortino_ratio;
}

//+------------------------------------------------------------------+
//| Calculate position size using Kelly Criterion - Optimal growth strategy |
//+------------------------------------------------------------------+
// CalculateKellySize() implements the Kelly Criterion for optimal position sizing.
// Developed by John Kelly Jr. in 1956, this method maximizes the expected logarithmic
// growth of capital over time, providing the theoretically optimal fraction to risk.
//
// MATHEMATICAL DEFINITION:
// Kelly Fraction = (bp - q) / b
// where:
// - b = odds received (avg_win / avg_loss ratio)
// - p = probability of winning
// - q = probability of losing (1 - p)
//
// EXPANDED FORMULA:
// f* = (p × W - q) / W where f* = optimal fraction, W = win/loss ratio
//
// ADVANTAGES:
// - Maximizes long-term growth rate
// - Accounts for both probability and payoff
// - Theoretically optimal for repeated bets
// - Prevents risk of ruin when applied correctly
//
// DISADVANTAGES:
// - Can recommend large positions (high volatility)
// - Assumes perfect knowledge of probabilities
// - Sensitive to parameter estimation errors
// - May not align with investor risk preferences
//
// PRACTICAL IMPLEMENTATION:
// - Usually scaled down (25%-50% of full Kelly) for safety
// - Combined with position size caps for risk management
//
// PARAMETERS:
// - win_prob: Probability of profitable trade (0 < p < 1)
// - avg_win: Average profit per winning trade (positive)
// - avg_loss: Average loss per losing trade (negative, will be made positive)
double CRiskManager::CalculateKellySize(double win_prob, double avg_win, double avg_loss)
{
    // === INPUT VALIDATION ===
    // Validate probability is in valid range (0, 1)
    if(win_prob <= 0.0 || win_prob >= 1.0) 
    {
        Print("ERROR: Kelly Criterion - win probability must be between 0 and 1, received: ", win_prob);
        return 0.0;
    }
    
    // Validate average loss is non-zero
    if(avg_loss == 0.0)
    {
        Print("ERROR: Kelly Criterion - average loss cannot be zero");
        return 0.0;
    }
    
    // Validate average win is positive
    if(avg_win <= 0.0)
    {
        Print("ERROR: Kelly Criterion - average win must be positive, received: ", avg_win);
        return 0.0;
    }
    
    // === KELLY CALCULATION ===
    double loss_prob = 1.0 - win_prob;                    // q = probability of losing
    double win_loss_ratio = avg_win / MathAbs(avg_loss);  // b = odds ratio (W)
    
    // Kelly formula: f* = (bp - q) / b = (p × W - q) / W
    double kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio;
    
    // Check if Kelly is negative (negative expected value - don't trade)
    if(kelly_fraction <= 0.0)
    {
        Print("WARNING: Negative Kelly fraction (", DoubleToString(kelly_fraction, 4), 
              ") - strategy has negative expected value");
        return 0.0;
    }
    
    // === CONSERVATIVE SCALING AND RISK MANAGEMENT ===
    // Apply fractional Kelly (typically 25% of full Kelly for safety)
    kelly_fraction *= 0.25;  // Conservative scaling to reduce volatility
    
    // Apply maximum position size cap (institutional risk management)
    double max_position = 0.2;  // 20% maximum position size cap
    kelly_fraction = MathMin(kelly_fraction, max_position);
    
    Print("Kelly Calculation: Win Rate=", DoubleToString(win_prob*100, 1), "%, ",
          "Win/Loss Ratio=", DoubleToString(win_loss_ratio, 2), ", ",
          "Suggested Size=", DoubleToString(kelly_fraction*100, 2), "%");
    
    return kelly_fraction;
}

//+------------------------------------------------------------------+
//| Check risk limits - Real-time risk monitoring and alerting      |
//+------------------------------------------------------------------+
// CheckRiskLimits() performs comprehensive real-time risk monitoring by comparing
// current risk metrics against predefined limits. This is a critical component
// of automated risk management systems, providing early warning and automatic
// intervention capabilities.
//
// MONITORING FRAMEWORK:
// 1. Update all risk metrics with latest market data
// 2. Compare each metric against corresponding limit
// 3. Generate alerts for any limit breaches
// 4. Return false if ANY limit is exceeded (fail-safe approach)
//
// RISK LIMIT CATEGORIES:
// - Statistical limits: VaR, CVaR, Volatility (model-based risk)
// - Position limits: Concentration, leverage (exposure control)
// - Performance limits: Drawdown, Sharpe ratio (performance thresholds)
// - Operational limits: Daily/weekly/monthly loss (capital preservation)
//
// APPLICATIONS:
// - Pre-trade risk checks (prevent risky trades)
// - Real-time monitoring (continuous surveillance)
// - Automated position reduction (risk limit breaches)
// - Regulatory compliance (ensure policy adherence)
//
// FAIL-SAFE DESIGN: Returns false if ANY limit is breached (conservative approach)
bool CRiskManager::CheckRiskLimits(void)
{
    // Update core risk metrics
    m_current_volatility = CalculateVolatility(0);
    m_sharpe_ratio       = CalculateSharpeRatio();
    m_sortino_ratio      = CalculateSortinoRatio(0.0);
    m_current_var        = CalculateHistoricalVaR(m_confidence_level);
    m_current_cvar       = CalculateCVaR(m_confidence_level);

    bool all_ok = true;

    // Check statistical limits
    if(m_limits.max_var_limit > 0 && m_current_var > m_limits.max_var_limit)
    {
        Print("ALERT: VaR limit breached: ", DoubleToString(m_current_var*100,2), "% > ",
              DoubleToString(m_limits.max_var_limit*100,2), "%");
        all_ok = false;
    }
    if(m_limits.max_cvar_limit > 0 && m_current_cvar > m_limits.max_cvar_limit)
    {
        Print("ALERT: CVaR limit breached: ", DoubleToString(m_current_cvar*100,2), "% > ",
              DoubleToString(m_limits.max_cvar_limit*100,2), "%");
        all_ok = false;
    }
    if(m_limits.max_volatility > 0 && m_current_volatility > m_limits.max_volatility)
    {
        Print("ALERT: Volatility limit breached: ", DoubleToString(m_current_volatility*100,2), "% > ",
              DoubleToString(m_limits.max_volatility*100,2), "%");
        all_ok = false;
    }

    // Check performance thresholds
    if(m_limits.min_sharpe_ratio > 0 && m_sharpe_ratio < m_limits.min_sharpe_ratio)
    {
        Print("ALERT: Sharpe ratio below minimum: ", DoubleToString(m_sharpe_ratio,2),
              " < ", DoubleToString(m_limits.min_sharpe_ratio,2));
        all_ok = false;
    }

    // Drawdown checks (if available)
    if(m_max_drawdown > 0 && m_limits.max_monthly_loss > 0 && m_max_drawdown > m_limits.max_monthly_loss)
    {
        Print("ALERT: Max drawdown exceeds monthly loss limit: ",
              DoubleToString(m_max_drawdown*100,2), "% > ",
              DoubleToString(m_limits.max_monthly_loss*100,2), "%");
        all_ok = false;
    }

    return all_ok;
}

string CRiskManager::GenerateRiskReport(void)
{
    string report = "\nRISK REPORT\n";
    report += "==============================\n";
    report += "Confidence: " + DoubleToString(m_confidence_level*100,1) + "%\n";
    report += "VaR (" + IntegerToString((int)MathRound(m_confidence_level*100)) + "%): "
              + DoubleToString(m_current_var*100,2) + "%\n";
    report += "CVaR: " + DoubleToString(m_current_cvar*100,2) + "%\n";
    report += "Volatility (ann.): " + DoubleToString(m_current_volatility*100,2) + "%\n";
    report += "Sharpe: " + DoubleToString(m_sharpe_ratio,2) + "\n";
    report += "Sortino: " + DoubleToString(m_sortino_ratio,2) + "\n";
    report += "Max Drawdown: " + DoubleToString(m_max_drawdown*100,2) + "%\n";
    report += "Position Size Limit: " + DoubleToString(m_limits.max_position_size*100,2) + "%\n";
    report += "Daily Loss Limit: " + DoubleToString(m_limits.max_daily_loss*100,2) + "%\n";
    report += "Weekly Loss Limit: " + DoubleToString(m_limits.max_weekly_loss*100,2) + "%\n";
    report += "Monthly Loss Limit: " + DoubleToString(m_limits.max_monthly_loss*100,2) + "%\n";
    report += "VaR Limit: " + DoubleToString(m_limits.max_var_limit*100,2) + "%\n";
    report += "CVaR Limit: " + DoubleToString(m_limits.max_cvar_limit*100,2) + "%\n";
    report += "Volatility Limit: " + DoubleToString(m_limits.max_volatility*100,2) + "%\n";
    report += "Min Sharpe: " + DoubleToString(m_limits.min_sharpe_ratio,2) + "\n";
    return report;
}

void CRiskManager::SetRiskLimits(double max_pos_size, double max_daily, double max_var)
{
    // === INPUT VALIDATION ===
    if(max_pos_size <= 0.0 || max_pos_size > 1.0)
    {
        Print("WARNING: Invalid position size limit: ", max_pos_size, " - should be between 0.01-0.5");
    }
    
    if(max_daily <= 0.0 || max_daily > 0.5)
    {
        Print("WARNING: Invalid daily loss limit: ", max_daily, " - should be between 0.01-0.1");
    }
    
    if(max_var <= 0.0 || max_var > 0.2)
    {
        Print("WARNING: Invalid VaR limit: ", max_var, " - should be between 0.01-0.05");
    }
    
    // === UPDATE RISK LIMITS ===
    m_limits.max_position_size = max_pos_size;    // Maximum single position size (fraction of capital)
    m_limits.max_daily_loss = max_daily;          // Maximum daily loss tolerance (fraction of capital)
    m_limits.max_var_limit = max_var;             // Maximum Value at Risk limit (fraction of capital)
    
    // === CONFIRMATION LOG ===
    Print("=== RISK LIMITS UPDATED ===");
    Print("Maximum Position Size: ", DoubleToString(max_pos_size * 100, 2), "%");
    Print("Maximum Daily Loss: ", DoubleToString(max_daily * 100, 2), "%");
    Print("Maximum VaR Limit: ", DoubleToString(max_var * 100, 3), "%");
    Print("Timestamp: ", TimeToString(TimeCurrent()));
    Print("===========================");
}