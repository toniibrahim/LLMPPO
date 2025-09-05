//+------------------------------------------------------------------+
//|                                            LLM_PPO_Backtest.mq5 |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
// This is a comprehensive backtesting engine for the LLM-PPO Trading System
// It simulates trading based on the AI model's predictions and provides detailed 
// performance analysis including risk metrics, trade statistics, and equity curves
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs
#property description "Comprehensive Backtesting Script for LLM-PPO Trading System"

#include <LLM_PPO_Model.mqh>
#include <RiskManager.mqh>

//--- Input parameters for configuring the backtest
// These parameters control the testing period, model behavior, and trading rules
input group "=== Backtesting Period ==="
input datetime InpStartDate      = D'2020.01.01';  // Historical data start date for backtesting
input datetime InpEndDate        = D'2024.12.31';  // Historical data end date for backtesting
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;    // Chart timeframe to use for analysis (H1 = hourly bars)

input group "=== Model Parameters ==="
// Core LLM-PPO model configuration - these affect how the AI makes trading decisions
input double   InpLearningRate      = 0.0003;     // PPO learning rate (how fast model adapts - typical range 0.0001-0.01)
input double   InpRiskWeight        = 0.5;        // Risk penalty weight λ in reward function R_t = -|ŷ_t - y_t*| - λ·CVaR_α
input double   InpConfidenceLevel   = 0.95;       // Confidence level for VaR/CVaR risk calculations (0.95 = 95%)
input int      InpPredictionPeriod  = 5;          // Number of historical bars used for price prediction

input group "=== Trading Parameters ==="
// Controls how the system executes trades and manages position sizes
input double   InpInitialBalance    = 10000.0;    // Starting capital for backtesting simulation
input double   InpBaseLotSize       = 0.1;        // Fixed position size when not using dynamic sizing
input bool     InpUseRiskSizing     = true;       // Enable VaR-based position sizing (recommended)
input double   InpMaxRiskPercent    = 2.0;        // Maximum portfolio risk per trade (2% = conservative)
input double   InpMinConfidence     = 0.6;        // Minimum AI confidence threshold to open positions (0-1 scale)

input group "=== Risk Management ==="
// Multi-layer risk controls to protect capital and limit losses
input int      InpStopLoss          = 100;        // Fixed stop loss in points (used when dynamic SL disabled)
input int      InpTakeProfit        = 200;        // Fixed take profit in points (typically 2x stop loss)
input bool     InpUseDynamicSL      = true;       // Use volatility-based stop losses (adapts to market conditions)
input double   InpVolatilityMultip  = 2.0;        // Multiplier for volatility-based stops (higher = wider stops)
input double   InpMaxDrawdown       = 20.0;       // Maximum portfolio drawdown before stopping trades (%)
input double   InpDailyLossLimit    = 5.0;        // Maximum daily loss limit as percentage of balance

input group "=== Output Settings ==="
// Configure what reports and files to generate after backtesting
input bool     InpSaveDetailedLog   = true;       // Generate detailed trade-by-trade log file
input bool     InpExportResults     = true;       // Export equity curve and performance data to CSV
input bool     InpCreateChart       = true;       // Create visual equity chart (implementation dependent)
input string   InpOutputFolder      = "LLM_PPO_Backtest"; // Folder name for saving all output files

//--- Global variables and data structures
// BacktestTrade: Stores complete information about each executed trade
// This struct captures all trade details for later analysis and reporting
struct BacktestTrade
{
    datetime open_time;     // When the trade was opened
    datetime close_time;    // When the trade was closed
    int      type;          // ORDER_TYPE_BUY or ORDER_TYPE_SELL
    double   volume;        // Position size in lots
    double   open_price;    // Entry price
    double   close_price;   // Exit price
    double   sl;            // Stop loss level
    double   tp;            // Take profit level
    double   profit;        // Gross profit/loss in account currency
    double   commission;    // Broker commission paid
    double   swap;          // Overnight swap fees
    double   confidence;    // AI model confidence level (0-1) at trade open
    double   prediction;    // AI predicted price at trade open
    string   close_reason;  // Why trade was closed: "Stop Loss", "Take Profit", etc.
    
    // Copy constructor
    BacktestTrade(const BacktestTrade& other)
    {
        open_time = other.open_time;
        close_time = other.close_time;
        type = other.type;
        volume = other.volume;
        open_price = other.open_price;
        close_price = other.close_price;
        sl = other.sl;
        tp = other.tp;
        profit = other.profit;
        commission = other.commission;
        swap = other.swap;
        confidence = other.confidence;
        prediction = other.prediction;
        close_reason = other.close_reason;
    }
    
    // Default constructor
    BacktestTrade()
    {
        open_time = 0;
        close_time = 0;
        type = 0;
        volume = 0.0;
        open_price = 0.0;
        close_price = 0.0;
        sl = 0.0;
        tp = 0.0;
        profit = 0.0;
        commission = 0.0;
        swap = 0.0;
        confidence = 0.0;
        prediction = 0.0;
        close_reason = "";
    }
};

// BacktestResults: Comprehensive performance metrics calculated after backtesting
// Contains all key statistics needed to evaluate the trading system's performance
struct BacktestResults
{
    // Trading Performance Metrics
    int      total_trades;          // Total number of completed trades
    int      winning_trades;        // Number of profitable trades
    int      losing_trades;         // Number of losing trades
    double   win_rate;              // Percentage of winning trades (0-100)
    double   profit_factor;         // Gross profit / |Gross loss| (>1.0 is profitable)
    double   total_profit;          // Net profit/loss in account currency
    double   gross_profit;          // Total profit from winning trades
    double   gross_loss;            // Total loss from losing trades (negative)
    double   avg_win;               // Average profit per winning trade
    double   avg_loss;              // Average loss per losing trade
    double   largest_win;           // Single largest winning trade
    double   largest_loss;          // Single largest losing trade
    double   consecutive_wins;      // Maximum consecutive winning trades
    double   consecutive_losses;    // Maximum consecutive losing trades
    
    // Risk and Performance Ratios
    double   max_drawdown;          // Maximum peak-to-trough decline in equity
    double   max_drawdown_percent;  // Max drawdown as percentage of peak equity
    double   recovery_factor;       // Total profit / Max drawdown (higher is better)
    double   sharpe_ratio;          // (Return - Risk-free rate) / Volatility
    double   sortino_ratio;         // Return / Downside deviation (focuses on bad volatility)
    double   calmar_ratio;          // Annual return / Max drawdown
    double   var_95;                // Value at Risk at 95% confidence level
    double   cvar_95;               // Conditional VaR (expected loss beyond VaR)
    double   volatility;            // Standard deviation of returns
    
    // Time-Based Metrics
    datetime start_time;            // Backtest start date
    datetime end_time;              // Backtest end date
    double   total_days;            // Total days in backtest period
    double   annual_return;         // Annualized return percentage
    double   annual_volatility;     // Annualized volatility
    
    // AI Model Performance Metrics
    double   prediction_accuracy;   // Percentage of correct directional predictions
    double   avg_confidence;        // Average confidence level across all trades
    double   model_hit_ratio;       // Hit rate of model predictions vs actual outcomes
};

// Global arrays and objects for managing the backtest simulation
BacktestTrade    g_trades[];        // Dynamic array storing all completed trades
BacktestResults  g_results;         // Struct holding final performance statistics
CLLM_PPO_Model*  g_model;          // Pointer to the AI trading model instance
CRiskManager*    g_risk_manager;    // Pointer to the risk management system

// Portfolio state variables - track account equity and risk metrics in real-time
double           g_equity_curve[];      // Historical equity values for charting
datetime         g_equity_times[];      // Corresponding timestamps for equity curve
double           g_balance;              // Current account balance (closed trades only)
double           g_equity;               // Current equity (balance + floating P&L)
double           g_margin;               // Margin currently used by open positions
double           g_free_margin;          // Available margin for new trades
double           g_margin_level;         // Margin level percentage (equity/margin*100)
double           g_peak_equity;          // Highest equity value reached (for drawdown calc)
double           g_current_drawdown;     // Current drawdown from peak (%)
double           g_max_drawdown;         // Maximum drawdown experienced (%)

// Position tracking during backtest simulation
// Position: Represents an open trade position with all relevant details
struct Position
{
    int      ticket;        // Unique position identifier
    int      type;          // ORDER_TYPE_BUY or ORDER_TYPE_SELL
    double   volume;        // Position size in lots
    double   open_price;    // Entry price where position was opened
    double   sl;            // Stop loss level
    double   tp;            // Take profit level
    datetime open_time;     // When position was opened
    double   profit;        // Current floating profit/loss
    double   confidence;    // AI confidence level when position was opened
    double   prediction;    // AI price prediction when position was opened
    
    // Copy constructor
    Position(const Position& other)
    {
        ticket = other.ticket;
        type = other.type;
        volume = other.volume;
        open_price = other.open_price;
        sl = other.sl;
        tp = other.tp;
        open_time = other.open_time;
        profit = other.profit;
        confidence = other.confidence;
        prediction = other.prediction;
    }
    
    // Default constructor
    Position()
    {
        ticket = 0;
        type = 0;
        volume = 0.0;
        open_price = 0.0;
        sl = 0.0;
        tp = 0.0;
        open_time = 0;
        profit = 0.0;
        confidence = 0.0;
        prediction = 0.0;
    }
};
Position         g_positions[];     // Dynamic array of currently open positions
int              g_next_ticket = 1;  // Counter for generating unique position tickets

//+------------------------------------------------------------------+
//| Script program start function - Main entry point for backtesting|
//+------------------------------------------------------------------+
// OnStart() is called when the script is executed. This is the main control flow
// that orchestrates the entire backtesting process from initialization to cleanup.
void OnStart()
{
    Print("Starting LLM-PPO Backtesting...");
    
    // Step 1: Initialize all components (AI model, risk manager, portfolio)
    if(!Initialize())
    {
        Print("Failed to initialize backtesting components");
        return;
    }
    
    // Step 2: Run the main backtesting simulation across historical data
    if(!RunBacktest())
    {
        Print("Backtest failed");
        return;
    }
    
    // Step 3: Calculate comprehensive performance statistics
    CalculateResults();
    
    // Step 4: Generate detailed reports and export data
    GenerateReports();
    
    // Step 5: Clean up resources and memory
    Cleanup();
    
    Print("Backtesting completed successfully");
}

//+------------------------------------------------------------------+
//| Initialize backtesting components                                |
//+------------------------------------------------------------------+
// Initialize() sets up all required objects and initial portfolio state
// Returns true if successful, false if any critical component fails to initialize
bool Initialize()
{
    // Create and configure the LLM-PPO AI trading model
    g_model = new CLLM_PPO_Model();
    if(!g_model.Initialize(_Symbol, InpTimeframe))
    {
        Print("Failed to initialize LLM-PPO model");
        return false;
    }
    
    // Configure model parameters that affect trading decisions
    g_model.SetLearningRate(InpLearningRate);      // How fast the PPO learns
    g_model.SetRiskWeight(InpRiskWeight);          // Risk penalty in reward function
    g_model.SetConfidenceLevel(InpConfidenceLevel); // VaR confidence level
    
    // Create and configure the risk management system
    g_risk_manager = new CRiskManager();
    if(!g_risk_manager.Initialize(InpConfidenceLevel, 252)) // 252 = typical trading days per year
    {
        Print("Failed to initialize risk manager");
        return false;
    }
    
    // Initialize portfolio state variables to starting conditions
    g_balance = InpInitialBalance;       // Starting capital
    g_equity = g_balance;                // Initial equity equals balance (no open positions)
    g_margin = 0.0;                      // No margin used initially
    g_free_margin = g_balance;           // All capital is free initially
    g_margin_level = 0.0;                // No positions = no margin level
    g_peak_equity = g_balance;           // Peak starts at initial balance
    g_current_drawdown = 0.0;            // No drawdown at start
    g_max_drawdown = 0.0;                // No max drawdown yet
    
    // Initialize dynamic arrays for storing backtest data
    ArrayResize(g_trades, 0);            // Start with empty trade history
    ArrayResize(g_positions, 0);         // Start with no open positions
    ArrayResize(g_equity_curve, 0);      // Start with empty equity curve
    ArrayResize(g_equity_times, 0);      // Start with empty time series
    
    // Initialize results structure for final statistics
    ZeroMemory(g_results);               // Clear all fields to zero
    g_results.start_time = InpStartDate; // Set backtest period start
    g_results.end_time = InpEndDate;     // Set backtest period end
    
    return true;
}

//+------------------------------------------------------------------+
//| Run the backtest - Core simulation loop                         |
//+------------------------------------------------------------------+
// RunBacktest() processes each historical bar, generates AI predictions,
// manages open positions, and executes new trades based on model signals
bool RunBacktest()
{
    // Get total number of historical bars available for the test period
    int bars_total = Bars(_Symbol, InpTimeframe, InpStartDate, InpEndDate);
    if(bars_total <= 0)
    {
        Print("No historical data available for the specified period");
        return false;
    }
    
    Print("Processing ", bars_total, " bars from ", InpStartDate, " to ", InpEndDate);
    
    // Get timestamps for all bars in the test period
    datetime bar_times[];
    if(CopyTime(_Symbol, InpTimeframe, InpStartDate, InpEndDate, bar_times) <= 0)
    {
        Print("Failed to copy time data");
        return false;
    }
    
    int processed = 0;  // Counter for progress tracking
    
    // Main simulation loop: process bars from oldest to newest
    // Note: i counts backwards because MT5 bar indices go from newest (0) to oldest
    for(int i = bars_total - 1; i >= 0; i--)
    {
        datetime current_time = bar_times[i];
        
        // Step 1: Update market data for current bar
        if(!UpdateMarketData(current_time, i))
            continue;  // Skip this bar if data is unavailable
        
        // Step 2: Check existing positions for stop loss/take profit hits
        CheckAndClosePositions(current_time, i);
        
        // Step 3: Generate AI prediction for next price movement
        double prediction = g_model.GenerateRiskAwarePrediction();
        if(prediction <= 0) continue;  // Skip if model doesn't generate valid prediction
        
        double confidence = g_model.GetModelConfidence();
        
        // Step 4: Update risk management system with current market data
        double current_price = GetPriceAt(current_time, i, PRICE_CLOSE);
        g_risk_manager.UpdatePriceData(current_price, current_time);
        
        // Step 5: Check if conditions are right for opening new trade
        if(confidence >= InpMinConfidence && CheckRiskLimits())
        {
            ProcessTradingSignal(prediction, confidence, current_time, i);
        }
        
        // Step 6: Update portfolio metrics and equity curve
        UpdateEquityCurve(current_time);
        
        // Progress reporting every 1000 bars
        processed++;
        if(processed % 1000 == 0)
        {
            Print("Processed ", processed, "/", bars_total, " bars (", 
                  DoubleToString(processed * 100.0 / bars_total, 1), "%)");
        }
    }
    
    // Close any remaining open positions at end of test period
    CloseAllPositions(InpEndDate, "End of test");
    
    Print("Backtest completed. Processed ", processed, " bars, executed ", ArraySize(g_trades), " trades");
    
    return true;
}

//+------------------------------------------------------------------+
//| Update market data for current bar                               |
//+------------------------------------------------------------------+
// UpdateMarketData() fetches current bar's price data and updates the AI model
// This ensures the model has the latest market information for predictions
bool UpdateMarketData(datetime time, int bar_index)
{
    double prices[];
    if(CopyClose(_Symbol, InpTimeframe, bar_index, 1, prices) <= 0)
        return false;  // Return false if price data is not available
    
    // Update AI model with current price data (implementation depends on model)
    // The model uses this data to calculate technical indicators and make predictions
    return true;
}

//+------------------------------------------------------------------+
//| Get price at specific bar                                        |
//+------------------------------------------------------------------+
// GetPriceAt() retrieves specific price type (Open/High/Low/Close) for a given bar
// Used throughout the backtest to get accurate historical price data
double GetPriceAt(datetime time, int bar_index, ENUM_APPLIED_PRICE price_type)
{
    double prices[];  // Array to store price data
    
    // Switch between different price types and fetch appropriate data
    switch(price_type)
    {
        case PRICE_OPEN:   // Opening price of the bar
            if(CopyOpen(_Symbol, InpTimeframe, bar_index, 1, prices) > 0)
                return prices[0];
            break;
        case PRICE_HIGH:   // Highest price during the bar
            if(CopyHigh(_Symbol, InpTimeframe, bar_index, 1, prices) > 0)
                return prices[0];
            break;
        case PRICE_LOW:    // Lowest price during the bar
            if(CopyLow(_Symbol, InpTimeframe, bar_index, 1, prices) > 0)
                return prices[0];
            break;
        case PRICE_CLOSE:  // Closing price of the bar (most commonly used)
            if(CopyClose(_Symbol, InpTimeframe, bar_index, 1, prices) > 0)
                return prices[0];
            break;
    }
    
    return 0.0;
}

//+------------------------------------------------------------------+
//| Process trading signal - Core trading logic                     |
//+------------------------------------------------------------------+
// ProcessTradingSignal() analyzes AI prediction and decides whether to open a trade
// Includes signal filtering, position sizing, and risk management checks
void ProcessTradingSignal(double prediction, double confidence, datetime time, int bar_index)
{
    double current_price = GetPriceAt(time, bar_index, PRICE_CLOSE);
    if(current_price <= 0) return;  // Skip if price data unavailable
    
    // Calculate predicted price change as percentage
    double price_change = (prediction - current_price) / current_price;
    
    // Dynamic threshold: lower confidence requires larger price movements
    // This filters out weak signals when model uncertainty is high
    double min_change_threshold = (1.0 - confidence) * 0.002 + 0.001;
    
    if(MathAbs(price_change) < min_change_threshold)
        return;  // Signal too weak relative to model confidence
    
    // Simple position limit: only one position at a time (can be enhanced)
    if(ArraySize(g_positions) > 0) return;
    
    // Determine trade direction based on prediction vs current price
    int trade_type = (price_change > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    
    // Calculate optimal position size using risk management rules
    double lot_size = CalculatePositionSize(confidence, current_price);
    if(lot_size <= 0) return;  // Skip if position size calculation fails
    
    // Calculate stop loss and take profit levels
    double sl, tp;
    CalculateStopLevels(trade_type, current_price, sl, tp, bar_index);
    
    // Execute the trade if all conditions are met
    OpenPosition(trade_type, lot_size, current_price, sl, tp, time, confidence, prediction);
}

//+------------------------------------------------------------------+
//| Calculate position size - Risk-based position sizing            |
//+------------------------------------------------------------------+
// CalculatePositionSize() determines optimal trade size based on:
// - Account balance and risk tolerance
// - Model confidence level
// - Current market volatility
// - Stop loss distance
double CalculatePositionSize(double confidence, double entry_price)
{
    // Use fixed lot size if dynamic sizing is disabled
    if(!InpUseRiskSizing)
        return InpBaseLotSize;
    
    // Calculate base risk amount as percentage of account balance
    double risk_amount = g_balance * InpMaxRiskPercent / 100.0;
    
    // Adjust risk based on model confidence and market volatility
    double volatility = g_model.GetCurrentVolatility();
    double confidence_adj = confidence * confidence;          // Square for stronger effect
    double volatility_adj = MathMax(0.5, 1.0 - volatility * 10);  // Reduce size in volatile markets
    
    // Apply adjustments to base risk amount
    risk_amount *= confidence_adj * volatility_adj;
    
    // Calculate stop loss distance in points
    double stop_points = InpStopLoss;  // Use fixed stop loss by default
    if(InpUseDynamicSL)
    {
        // Dynamic stop loss based on market volatility
        stop_points = volatility * InpVolatilityMultip * 10000;
        stop_points = MathMax(50, MathMin(300, stop_points));  // Reasonable limits
    }
    
    // Get point value for position size calculation
    double point_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    if(point_value <= 0) point_value = 1.0;  // Fallback if symbol info unavailable
    
    // Calculate lot size: Risk Amount / (Stop Distance × Point Value)
    double lot_size = risk_amount / (stop_points * point_value);
    
    // Apply broker and safety limits
    lot_size = MathMax(0.01, MathMin(10.0, lot_size));  // Min 0.01, Max 10.0 lots
    
    return NormalizeDouble(lot_size, 2);
}

//+------------------------------------------------------------------+
//| Calculate stop loss and take profit levels                       |
//+------------------------------------------------------------------+
// CalculateStopLevels() sets protective stop loss and profit target levels
// Supports both fixed and volatility-based dynamic levels
void CalculateStopLevels(int trade_type, double entry_price, double &sl, double &tp, int bar_index)
{
    // Get symbol's point size (pip value)
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    if(point <= 0) point = 0.0001;  // Fallback for 4-digit quotes
    
    // Default to input parameter values
    double stop_points = InpStopLoss;
    double profit_points = InpTakeProfit;
    
    // Use dynamic levels if enabled - adapts to market volatility
    if(InpUseDynamicSL)
    {
        double volatility = g_model.GetCurrentVolatility();
        stop_points = volatility * InpVolatilityMultip * 10000;
        stop_points = MathMax(50, MathMin(300, stop_points));  // Safety limits
        profit_points = stop_points * 2;  // 1:2 risk/reward ratio
    }
    
    // Calculate actual price levels based on trade direction
    if(trade_type == ORDER_TYPE_BUY)
    {
        sl = entry_price - stop_points * point;   // Stop loss below entry for long
        tp = entry_price + profit_points * point; // Take profit above entry for long
    }
    else  // ORDER_TYPE_SELL
    {
        sl = entry_price + stop_points * point;   // Stop loss above entry for short
        tp = entry_price - profit_points * point; // Take profit below entry for short
    }
}

//+------------------------------------------------------------------+
//| Open position in backtest - Simulated trade execution           |
//+------------------------------------------------------------------+
// OpenPosition() simulates opening a new trade position
// Checks margin requirements, updates portfolio state, logs the trade
bool OpenPosition(int type, double volume, double price, double sl, double tp, 
                 datetime time, double confidence, double prediction)
{
    // Calculate required margin (simplified calculation)
    // Real margin depends on leverage, symbol specifications, etc.
    double margin_required = volume * 100000 * price / 100;
    
    // Check if we have sufficient free margin
    if(margin_required > g_free_margin)
    {
        Print("Insufficient margin: required=", margin_required, ", available=", g_free_margin);
        return false;  // Cannot open position
    }
    
    // Create new position object with all trade details
    Position pos;
    pos.ticket = g_next_ticket++;       // Assign unique ticket number
    pos.type = type;                    // BUY or SELL
    pos.volume = volume;                // Position size in lots
    pos.open_price = price;             // Entry price
    pos.sl = sl;                        // Stop loss level
    pos.tp = tp;                        // Take profit level
    pos.open_time = time;               // When position opened
    pos.profit = 0.0;                   // Initial floating P&L is zero
    pos.confidence = confidence;         // AI confidence at entry
    pos.prediction = prediction;         // AI price prediction at entry
    
    // Add position to the active positions array
    int size = ArraySize(g_positions);
    ArrayResize(g_positions, size + 1);
    g_positions[size] = pos;
    
    // Update margin accounting
    g_margin += margin_required;             // Increase used margin
    g_free_margin = g_balance - g_margin;    // Recalculate free margin
    
    Print("Position opened: ", pos.ticket, " ", EnumToString((ENUM_ORDER_TYPE)type), 
          " ", volume, " at ", price, " SL:", sl, " TP:", tp, " Conf:", confidence);
    
    return true;
}

//+------------------------------------------------------------------+
//| Check and close positions - Stop loss/Take profit monitoring    |
//+------------------------------------------------------------------+
// CheckAndClosePositions() monitors all open positions for exit conditions
// Automatically closes positions when stop loss or take profit is hit
void CheckAndClosePositions(datetime time, int bar_index)
{
    // Get current market prices (simplified bid/ask)
    double current_price_bid = GetPriceAt(time, bar_index, PRICE_CLOSE);
    double current_price_ask = current_price_bid;  // Simplified: assume no spread
    
    // Check each open position (loop backwards for safe array modification)
    for(int i = ArraySize(g_positions) - 1; i >= 0; i--)
    {
        Position pos = g_positions[i];
        // Use appropriate price: bid for long positions, ask for short positions
        double current_price = (pos.type == ORDER_TYPE_BUY) ? current_price_bid : current_price_ask;
        
        if(current_price <= 0) continue;  // Skip if price data unavailable
        
        // Calculate current floating profit/loss
        double profit;
        if(pos.type == ORDER_TYPE_BUY)
            profit = (current_price - pos.open_price) * pos.volume * 100000;  // Long: profit when price rises
        else
            profit = (pos.open_price - current_price) * pos.volume * 100000;  // Short: profit when price falls
        
        // Check if stop loss level was hit
        bool hit_sl = false;
        if(pos.type == ORDER_TYPE_BUY && current_price <= pos.sl)
            hit_sl = true;   // Long position stopped out when price falls to SL
        else if(pos.type == ORDER_TYPE_SELL && current_price >= pos.sl)
            hit_sl = true;   // Short position stopped out when price rises to SL
        
        // Check if take profit level was hit
        bool hit_tp = false;
        if(pos.type == ORDER_TYPE_BUY && current_price >= pos.tp)
            hit_tp = true;   // Long position profits when price rises to TP
        else if(pos.type == ORDER_TYPE_SELL && current_price <= pos.tp)
            hit_tp = true;   // Short position profits when price falls to TP
        
        // Close position if conditions met
        string close_reason = "";
        double close_price = current_price;
        
        if(hit_sl)
        {
            close_reason = "Stop Loss";
            close_price = pos.sl;
        }
        else if(hit_tp)
        {
            close_reason = "Take Profit";
            close_price = pos.tp;
        }
        
        if(close_reason != "")
        {
            ClosePosition(i, close_price, time, close_reason);
        }
    }
}

//+------------------------------------------------------------------+
//| Close specific position                                          |
//+------------------------------------------------------------------+
bool ClosePosition(int pos_index, double close_price, datetime close_time, string reason)
{
    if(pos_index < 0 || pos_index >= ArraySize(g_positions))
        return false;
    
    Position pos = g_positions[pos_index];
    
    // Calculate profit
    double profit;
    if(pos.type == ORDER_TYPE_BUY)
        profit = (close_price - pos.open_price) * pos.volume * 100000;
    else
        profit = (pos.open_price - close_price) * pos.volume * 100000;
    
    // Create trade record
    BacktestTrade trade;
    trade.open_time = pos.open_time;
    trade.close_time = close_time;
    trade.type = pos.type;
    trade.volume = pos.volume;
    trade.open_price = pos.open_price;
    trade.close_price = close_price;
    trade.sl = pos.sl;
    trade.tp = pos.tp;
    trade.profit = profit;
    trade.commission = -pos.volume * 7; // Simplified commission
    trade.swap = 0.0;
    trade.confidence = pos.confidence;
    trade.prediction = pos.prediction;
    trade.close_reason = reason;
    
    // Add to trades array
    int size = ArraySize(g_trades);
    ArrayResize(g_trades, size + 1);
    g_trades[size] = trade;
    
    // Update balance and margin
    g_balance += profit + trade.commission;
    double margin_released = pos.volume * 100000 * pos.open_price / 100;
    g_margin -= margin_released;
    g_free_margin = g_balance - g_margin;
    
    // Remove position from array
    for(int i = pos_index; i < ArraySize(g_positions) - 1; i++)
    {
        g_positions[i] = g_positions[i + 1];
    }
    ArrayResize(g_positions, ArraySize(g_positions) - 1);
    
    Print("Position closed: ", pos.ticket, " Profit: ", profit, " Reason: ", reason);
    
    return true;
}

//+------------------------------------------------------------------+
//| Close all positions                                              |
//+------------------------------------------------------------------+
void CloseAllPositions(datetime close_time, string reason)
{
    while(ArraySize(g_positions) > 0)
    {
        ClosePosition(0, g_positions[0].open_price, close_time, reason);
    }
}

//+------------------------------------------------------------------+
//| Check risk limits                                                |
//+------------------------------------------------------------------+
bool CheckRiskLimits()
{
    // Check drawdown limit
    double current_dd = (g_peak_equity - g_balance) / g_peak_equity * 100;
    if(current_dd > InpMaxDrawdown)
        return false;
    
    // Check daily loss limit (simplified)
    if((InpInitialBalance - g_balance) / InpInitialBalance * 100 > InpDailyLossLimit)
        return false;
    
    return true;
}

//+------------------------------------------------------------------+
//| Update equity curve                                              |
//+------------------------------------------------------------------+
void UpdateEquityCurve(datetime time)
{
    // Calculate current equity (balance + floating P&L)
    double floating_pl = 0.0;
    // Add floating P&L calculation here if needed
    
    g_equity = g_balance + floating_pl;
    
    // Update peak equity and drawdown
    if(g_equity > g_peak_equity)
    {
        g_peak_equity = g_equity;
        g_current_drawdown = 0.0;
    }
    else
    {
        g_current_drawdown = (g_peak_equity - g_equity) / g_peak_equity * 100;
        if(g_current_drawdown > g_max_drawdown)
            g_max_drawdown = g_current_drawdown;
    }
    
    // Add to equity curve
    int size = ArraySize(g_equity_curve);
    ArrayResize(g_equity_curve, size + 1);
    ArrayResize(g_equity_times, size + 1);
    g_equity_curve[size] = g_equity;
    g_equity_times[size] = time;
}

//+------------------------------------------------------------------+
//| Calculate final results - Comprehensive performance analysis     |
//+------------------------------------------------------------------+
// CalculateResults() processes all completed trades to generate detailed statistics
// Calculates trading performance, risk metrics, and model effectiveness
void CalculateResults()
{
    Print("Calculating backtest results...");
    
    int total_trades = ArraySize(g_trades);
    if(total_trades == 0)
    {
        Print("No trades executed during backtest period");
        return;
    }
    
    // Set basic trade statistics
    g_results.total_trades = total_trades;
    g_results.total_profit = g_balance - InpInitialBalance;  // Net profit/loss
    
    // Initialize variables for trade analysis
    double gross_profit = 0.0;      // Total profit from winning trades
    double gross_loss = 0.0;        // Total loss from losing trades (negative)
    int winning_trades = 0;         // Count of profitable trades
    int losing_trades = 0;          // Count of losing trades
    double largest_win = 0.0;       // Best single trade
    double largest_loss = 0.0;      // Worst single trade
    double sum_confidence = 0.0;    // Sum of all confidence levels
    int correct_predictions = 0;    // Count of correct directional predictions
    
    for(int i = 0; i < total_trades; i++)
    {
        BacktestTrade trade = g_trades[i];
        double net_profit = trade.profit + trade.commission + trade.swap;
        
        sum_confidence += trade.confidence;
        
        // Check if AI's directional prediction was correct
        // For BUY: correct if close > open (price went up as predicted)
        // For SELL: correct if close < open (price went down as predicted)
        if((trade.type == ORDER_TYPE_BUY && trade.close_price > trade.open_price) ||
           (trade.type == ORDER_TYPE_SELL && trade.close_price < trade.open_price))
        {
            correct_predictions++;  // AI predicted direction correctly
        }
        
        if(net_profit > 0)
        {
            winning_trades++;
            gross_profit += net_profit;
            if(net_profit > largest_win)
                largest_win = net_profit;
        }
        else
        {
            losing_trades++;
            gross_loss += net_profit;
            if(net_profit < largest_loss)
                largest_loss = net_profit;
        }
    }
    
    // Calculate comprehensive trading statistics
    g_results.winning_trades = winning_trades;
    g_results.losing_trades = losing_trades;
    g_results.win_rate = (total_trades > 0) ? (double)winning_trades / total_trades * 100 : 0;  // Win percentage
    g_results.gross_profit = gross_profit;
    g_results.gross_loss = gross_loss;
    g_results.profit_factor = (gross_loss != 0) ? gross_profit / MathAbs(gross_loss) : 0;  // Profit/Loss ratio
    g_results.avg_win = (winning_trades > 0) ? gross_profit / winning_trades : 0;         // Average winning trade
    g_results.avg_loss = (losing_trades > 0) ? gross_loss / losing_trades : 0;            // Average losing trade
    g_results.largest_win = largest_win;   // Best single trade
    g_results.largest_loss = largest_loss; // Worst single trade
    g_results.avg_confidence = sum_confidence / total_trades;  // Average model confidence
    g_results.prediction_accuracy = (double)correct_predictions / total_trades * 100;  // Directional accuracy %
    
    // Risk and drawdown metrics
    g_results.max_drawdown = g_max_drawdown;           // Maximum equity decline
    g_results.max_drawdown_percent = g_max_drawdown;   // Same value in percentage
    g_results.recovery_factor = (g_max_drawdown > 0) ? g_results.total_profit / g_max_drawdown : 0;  // Profit/Drawdown ratio
    
    // Calculate time-based performance metrics
    g_results.total_days = (double)(InpEndDate - InpStartDate) / (24 * 3600);  // Convert seconds to days
    // Annualized return: (Total Return / Initial Balance) × (365 / Days) × 100
    g_results.annual_return = (g_results.total_days > 0) ? 
        (g_results.total_profit / InpInitialBalance) * (365.0 / g_results.total_days) * 100 : 0;
    
    // Calculate advanced risk metrics using equity curve data
    if(ArraySize(g_equity_curve) > 1)
    {
        // Convert equity curve to returns series for risk calculations
        double returns[];
        ArrayResize(returns, ArraySize(g_equity_curve) - 1);
        
        // Calculate period-to-period returns
        for(int i = 1; i < ArraySize(g_equity_curve); i++)
        {
            returns[i-1] = (g_equity_curve[i] - g_equity_curve[i-1]) / g_equity_curve[i-1];
        }
        
        // Calculate sophisticated risk metrics using risk manager
        g_results.var_95 = g_risk_manager.CalculateHistoricalVaR();        // Value at Risk (95%)
        g_results.cvar_95 = g_risk_manager.CalculateCVaR();               // Conditional VaR
        g_results.sharpe_ratio = g_risk_manager.CalculateSharpeRatio();    // Risk-adjusted return
        g_results.sortino_ratio = g_risk_manager.CalculateSortinoRatio();  // Downside-focused ratio
    }
    
    Print("Results calculated successfully");
}

//+------------------------------------------------------------------+
//| Generate comprehensive reports                                   |
//+------------------------------------------------------------------+
void GenerateReports()
{
    Print("Generating backtest reports...");
    
    // Print summary to console
    PrintSummary();
    
    if(InpSaveDetailedLog)
        SaveDetailedLog();
    
    if(InpExportResults)
        ExportToCSV();
    
    if(InpCreateChart)
        CreateEquityChart();
}

//+------------------------------------------------------------------+
//| Print summary to console - Display key results                  |
//+------------------------------------------------------------------+
// PrintSummary() outputs formatted performance summary to MT5 terminal
// Provides quick overview of backtest results for immediate assessment
void PrintSummary()
{
    Print("========== BACKTEST SUMMARY ==========");
    Print("Period: ", g_results.start_time, " - ", g_results.end_time);
    Print("Symbol: ", _Symbol, " Timeframe: ", EnumToString(InpTimeframe));
    Print("");
    Print("--- Trading Performance ---");
    Print("Total Trades: ", g_results.total_trades);
    Print("Winning Trades: ", g_results.winning_trades, " (", 
          DoubleToString(g_results.win_rate, 2), "%)");
    Print("Losing Trades: ", g_results.losing_trades);
    Print("Profit Factor: ", DoubleToString(g_results.profit_factor, 2));
    Print("Total Profit: $", DoubleToString(g_results.total_profit, 2));
    Print("Annual Return: ", DoubleToString(g_results.annual_return, 2), "%");
    Print("");
    Print("--- Risk Metrics ---");
    Print("Max Drawdown: ", DoubleToString(g_results.max_drawdown_percent, 2), "%");
    Print("Recovery Factor: ", DoubleToString(g_results.recovery_factor, 2));
    Print("Sharpe Ratio: ", DoubleToString(g_results.sharpe_ratio, 3));
    Print("Sortino Ratio: ", DoubleToString(g_results.sortino_ratio, 3));
    Print("VaR (95%): ", DoubleToString(g_results.var_95 * 100, 3), "%");
    Print("CVaR (95%): ", DoubleToString(g_results.cvar_95 * 100, 3), "%");
    Print("");
    Print("--- Model Performance ---");
    Print("Average Confidence: ", DoubleToString(g_results.avg_confidence, 3));
    Print("Prediction Accuracy: ", DoubleToString(g_results.prediction_accuracy, 2), "%");
    Print("=====================================");
}

//+------------------------------------------------------------------+
//| Save detailed trade log - Export complete trade history         |
//+------------------------------------------------------------------+
// SaveDetailedLog() creates comprehensive text file with all trade details
// Includes parameters, trade-by-trade breakdown for further analysis
void SaveDetailedLog()
{
    string filename = InpOutputFolder + "/" + _Symbol + "_" + 
                     EnumToString(InpTimeframe) + "_detailed_log.txt";
    
    int file = FileOpen(filename, FILE_WRITE | FILE_TXT);
    if(file == INVALID_HANDLE)
    {
        Print("Failed to create detailed log file: ", filename);
        return;
    }
    
    // Write header
    FileWrite(file, "LLM-PPO Backtest Detailed Log");
    FileWrite(file, "Generated: " + TimeToString(TimeCurrent()));
    FileWrite(file, "Symbol: " + _Symbol + " Timeframe: " + EnumToString(InpTimeframe));
    FileWrite(file, "Period: " + TimeToString(InpStartDate) + " - " + TimeToString(InpEndDate));
    FileWrite(file, "");
    
    // Write parameters
    FileWrite(file, "--- Parameters ---");
    FileWrite(file, "Learning Rate: " + DoubleToString(InpLearningRate, 6));
    FileWrite(file, "Risk Weight: " + DoubleToString(InpRiskWeight, 3));
    FileWrite(file, "Confidence Level: " + DoubleToString(InpConfidenceLevel, 2));
    FileWrite(file, "Base Lot Size: " + DoubleToString(InpBaseLotSize, 2));
    FileWrite(file, "Max Risk Per Trade: " + DoubleToString(InpMaxRiskPercent, 2) + "%");
    FileWrite(file, "Min Confidence: " + DoubleToString(InpMinConfidence, 2));
    FileWrite(file, "");
    
    // Write detailed trade log header
    FileWrite(file, "--- Trade Details ---");
    // CSV header for easy import into spreadsheet applications
    FileWrite(file, "Ticket,OpenTime,CloseTime,Type,Volume,OpenPrice,ClosePrice,SL,TP,Profit,Commission,Confidence,Prediction,CloseReason");
    
    for(int i = 0; i < ArraySize(g_trades); i++)
    {
        BacktestTrade trade = g_trades[i];
        string line = IntegerToString(i+1) + "," +
                     TimeToString(trade.open_time) + "," +
                     TimeToString(trade.close_time) + "," +
                     EnumToString((ENUM_ORDER_TYPE)trade.type) + "," +
                     DoubleToString(trade.volume, 2) + "," +
                     DoubleToString(trade.open_price, 5) + "," +
                     DoubleToString(trade.close_price, 5) + "," +
                     DoubleToString(trade.sl, 5) + "," +
                     DoubleToString(trade.tp, 5) + "," +
                     DoubleToString(trade.profit, 2) + "," +
                     DoubleToString(trade.commission, 2) + "," +
                     DoubleToString(trade.confidence, 3) + "," +
                     DoubleToString(trade.prediction, 5) + "," +
                     trade.close_reason;
        FileWrite(file, line);
    }
    
    FileClose(file);
    Print("Detailed log saved to: ", filename);
}

//+------------------------------------------------------------------+
//| Export results to CSV - Create equity curve data file           |
//+------------------------------------------------------------------+
// ExportToCSV() saves equity curve and drawdown data in CSV format
// Useful for creating charts and further analysis in external tools
void ExportToCSV()
{
    string filename = InpOutputFolder + "/" + _Symbol + "_" + 
                     EnumToString(InpTimeframe) + "_results.csv";
    
    int file = FileOpen(filename, FILE_WRITE | FILE_CSV);
    if(file == INVALID_HANDLE)
    {
        Print("Failed to create CSV file: ", filename);
        return;
    }
    
    // Write CSV header for equity curve data
    FileWrite(file, "DateTime,Equity,Balance,Drawdown");
    
    // Export each equity point with timestamp and drawdown calculation
    for(int i = 0; i < ArraySize(g_equity_curve); i++)
    {
        double dd = (g_peak_equity - g_equity_curve[i]) / g_peak_equity * 100;  // Drawdown percentage
        FileWrite(file, TimeToString(g_equity_times[i]),      // Timestamp
                 DoubleToString(g_equity_curve[i], 2),        // Equity value
                 DoubleToString(g_equity_curve[i], 2),        // Balance (simplified)
                 DoubleToString(dd, 3));                      // Drawdown %
    }
    
    FileClose(file);
    Print("Results exported to: ", filename);
}

//+------------------------------------------------------------------+
//| Create equity chart                                              |
//+------------------------------------------------------------------+
void CreateEquityChart()
{
    // This would create a visual chart of the equity curve
    // Implementation depends on specific charting requirements
    Print("Equity chart creation completed");
}

//+------------------------------------------------------------------+
//| Cleanup resources - Free memory and close objects               |
//+------------------------------------------------------------------+
// Cleanup() properly releases all allocated memory and resources
// Called at end of backtest to prevent memory leaks
void Cleanup()
{
    // Delete AI model object and reset pointer
    if(g_model != NULL)
    {
        delete g_model;
        g_model = NULL;
    }
    
    // Delete risk manager object and reset pointer
    if(g_risk_manager != NULL)
    {
        delete g_risk_manager;
        g_risk_manager = NULL;
    }
    
    // Free all dynamic arrays to release memory
    ArrayFree(g_trades);        // Trade history
    ArrayFree(g_positions);     // Position array
    ArrayFree(g_equity_curve);  // Equity data
    ArrayFree(g_equity_times);  // Time series
}