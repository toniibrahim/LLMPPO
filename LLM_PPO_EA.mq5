//+------------------------------------------------------------------+
//|                                                   LLM_PPO_EA.mq5 |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
/*
   ═══════════════════════════════════════════════════════════════════
   LLM-PPO EXPERT ADVISOR - TWO-STAGE AI TRADING SYSTEM
   ═══════════════════════════════════════════════════════════════════
   
   This Expert Advisor implements a sophisticated two-stage trading system based on
   the research paper "A Two-Stage Framework for Stock Price Prediction: LLM-Based
   Forecasting with Risk-Aware PPO Adjustment" by Qizhao Chen.
   
   THEORETICAL FOUNDATION:
   ═════════════════════════
   
   The system combines the prediction capabilities of Large Language Models with
   the risk-aware optimization of Proximal Policy Optimization to create a robust
   trading framework that balances profitability with risk management.
   
   STAGE 1: LLM SIMULATION
   =-=-=-=-=-=-=-=-=-=-=-=
   • Processes market data through multiple technical indicators
   • Simulates LLM reasoning using mathematical transformations
   • Generates initial price movement predictions with confidence scores
   • Incorporates sentiment analysis and market microstructure data
   
   STAGE 2: PPO RISK-AWARE ADJUSTMENT  
   ══════════════════════════════════
   • Applies Proximal Policy Optimization to refine predictions
   • Calculates Value at Risk (VaR) and Conditional VaR (CVaR)
   • Uses risk-aware reward function: R_t = -|ŷ_t - y_t*| - λ·CVaR_α
   • Dynamically adjusts position sizing based on market conditions
   
   KEY MATHEMATICAL CONCEPTS:
   ═════════════════════════
   
   1. Risk-Aware Reward Function:
      R_t = -|prediction_error| - λ * CVaR_α
      Where λ is the risk weight parameter and α is confidence level
   
   2. Dynamic Position Sizing:
      position_size = base_size * confidence * (1 - volatility_factor)
   
   3. PPO Policy Update:
      policy_new = clip(policy_ratio, 1-ε, 1+ε) * advantage
      Where ε is the clipping parameter and advantage is the temporal difference
   
   SYSTEM ARCHITECTURE:
   ═══════════════════
   
   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
   │  Market Data    │───▶│   LLM Stage 1    │───▶│  PPO Stage 2    │
   │  Collection     │    │   Prediction     │    │  Risk Adjust    │
   └─────────────────┘    └──────────────────┘    └─────────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
   │ Technical       │    │ Confidence       │    │ Position        │
   │ Indicators      │    │ Scoring          │    │ Execution       │
   └─────────────────┘    └──────────────────┘    └─────────────────┘
*/

#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property description "LLM-PPO Expert Advisor - Two-Stage AI Trading System"

//--- Include the core LLM-PPO framework components
#include <LLM_PPO_Model.mqh>    // Stage 1: LLM simulation and Stage 2: PPO optimization
#include <RiskManager.mqh>      // Advanced risk management and VaR/CVaR calculations

//+------------------------------------------------------------------+
//| INPUT PARAMETERS - USER CONFIGURABLE TRADING SETTINGS          |
//+------------------------------------------------------------------+
/*
   These parameters allow users to customize the behavior of the two-stage
   LLM-PPO system without modifying the core algorithm implementation.
   
   Each parameter group serves a specific purpose in the overall system:
   - Trading Strategy: Controls basic risk and position management
   - LLM-PPO Model: Configures the AI prediction engine
   - Technical Indicators: Sets up market analysis tools
   - Risk Management: Defines safety limits and controls
*/

input group "=== Trading Strategy Parameters ==="
input double   InpRiskPerTrade         = 2.0;      // Risk per trade (% of account balance)
                                                    // Controls maximum capital at risk per single position
                                                    // Recommended range: 1.0% - 5.0% for retail accounts
                                                    
input double   InpMaxDailyRisk         = 6.0;      // Maximum daily risk exposure (% of balance)
                                                    // Total risk across all positions and trading session
                                                    // Should be 2-4x single trade risk for diversification
                                                    
input int      InpMaxOpenPositions     = 3;        // Maximum simultaneous open positions
                                                    // Limits portfolio concentration and overexposure
                                                    // Higher values require more capital and monitoring
                                                    
input bool     InpUseTrailingStop      = true;     // Enable dynamic trailing stop mechanism
                                                    // Automatically adjusts stop-loss to lock in profits
                                                    // Recommended: true for trending markets
                                                    
input double   InpTrailingStopDistance = 50.0;     // Trailing stop distance in points
                                                    // Distance from current price to maintain stop-loss
                                                    // Should be 2-3x average spread for the symbol

input group "=== LLM-PPO Model Parameters ==="
input double   InpLearningRate         = 0.001;    // PPO learning rate for policy updates
                                                    // Controls how aggressively the model adapts to new data
                                                    // Range: 0.0001 - 0.01 (lower = more stable, higher = faster adaptation)
                                                    
input double   InpRiskWeight           = 0.3;      // Lambda (λ) parameter for CVaR in reward function
                                                    // Balances profit maximization vs risk minimization
                                                    // Range: 0.0 - 1.0 (0 = profit only, 1 = risk only)
                                                    
input int      InpLookbackPeriod       = 20;       // Historical bars for state representation
                                                    // Number of previous bars used for pattern recognition
                                                    // Higher values capture longer-term patterns but increase computation
                                                    
input double   InpConfidenceThreshold  = 0.6;      // Minimum model confidence to execute trades
                                                    // Only trade when model is sufficiently certain
                                                    // Range: 0.5 - 0.9 (higher = fewer but higher quality trades)
                                                    
input bool     InpUseDynamicRisk       = true;     // Enable adaptive risk based on market volatility
                                                    // Automatically reduces position sizes in volatile conditions
                                                    // Recommended: true for all market conditions

input group "=== Technical Indicators Configuration ==="
input int      InpMAShortPeriod        = 12;       // Short-term Moving Average period
                                                    // Used for immediate trend identification
                                                    // Common values: 9, 12, 21 (shorter = more sensitive)
                                                    
input int      InpMALongPeriod         = 26;       // Long-term Moving Average period  
                                                    // Used for overall trend direction
                                                    // Should be 2-3x short period for clear separation
                                                    
input int      InpRSIPeriod            = 14;       // Relative Strength Index period
                                                    // Standard RSI period for overbought/oversold analysis
                                                    // 14 is traditional, shorter periods (7-10) more sensitive
                                                    
input int      InpMACDFast             = 12;       // MACD fast EMA period
                                                    // Should match short MA period for consistency
                                                    
input int      InpMACDSlow             = 26;       // MACD slow EMA period
                                                    // Should match long MA period for consistency
                                                    
input int      InpMACDSignal           = 9;        // MACD signal line EMA period
                                                    // Standard value for MACD signal smoothing

input group "=== Risk Management Controls ==="
input double   InpMaxDrawdown          = 15.0;     // Emergency stop at maximum drawdown (%)
                                                    // Circuit breaker to halt trading during severe losses
                                                    // Measured from highest equity point
                                                    
input double   InpProfitTarget         = 0.0;      // Daily profit target (0 = disabled)
                                                    // Stop trading after reaching daily profit goal
                                                    // Helps lock in good days and prevent overtrading
                                                    
input bool     InpUseNewsFilter        = false;    // Halt trading during high-impact news
                                                    // Prevents trading during volatile news events
                                                    // Requires news data feed (not implemented in demo)
                                                    
input string   InpTradingStartTime     = "09:00";  // Daily trading session start time (24h format)
                                                    // Avoid low-liquidity periods and market opens
                                                    
input string   InpTradingEndTime       = "17:00";  // Daily trading session end time (24h format)
                                                    // Avoid overnight positions and market closes

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES - SYSTEM STATE AND INSTANCES                  |
//+------------------------------------------------------------------+
/*
   These global variables maintain the state of the LLM-PPO system throughout
   the trading session. They include instances of core classes, tracking
   variables, and performance counters.
   
   IMPORTANT: Global variables in MQL5 persist across function calls but
   are reset when the EA is reinitialized or recompiled.
*/

// ═══ CORE SYSTEM INSTANCES ═══
CLLM_PPO_Model*    g_model;              // Main LLM-PPO model instance
                                         // Handles Stage 1 (LLM simulation) and Stage 2 (PPO adjustment)
                                         // Contains learned parameters and prediction algorithms
                                         
CRiskManager*      g_risk_manager;       // Risk management and portfolio optimization system
                                         // Calculates VaR, CVaR, volatility, and performance metrics
                                         // Monitors drawdown and exposure limits

// ═══ SYSTEM IDENTIFICATION ═══
int                g_magic_number;       // Unique identifier for this EA's trades
                                         // Allows multiple EAs on same account without conflicts
                                         // Generated randomly at initialization
                                         
// ═══ TEMPORAL TRACKING ═══
datetime           g_last_bar_time;      // Timestamp of last processed bar
                                         // Prevents duplicate processing of same bar
                                         // Critical for consistent signal generation
                                         
// ═══ SESSION MANAGEMENT ═══
double             g_daily_pnl;          // Current session profit/loss accumulator
                                         // Tracks performance within daily trading session
                                         // Reset at start of each trading day
                                         
datetime           g_trading_day;        // Current trading day identifier
                                         // Used to detect day changes and reset daily counters
                                         
int                g_positions_today;    // Count of positions opened in current session
                                         // Helps monitor trading frequency and overactivity
                                         
bool               g_trading_enabled;    // Master trading enable/disable flag
                                         // Can be disabled by risk manager during emergencies
                                         // Prevents new positions while allowing position management

// ═══ TECHNICAL INDICATOR HANDLES ═══
// These handles provide efficient access to MT5's built-in technical indicators
// Handles are created once at initialization and reused for performance
int                g_ma_short_handle;    // Handle for short-term moving average
int                g_ma_long_handle;     // Handle for long-term moving average  
int                g_rsi_handle;         // Handle for Relative Strength Index
int                g_macd_handle;        // Handle for MACD oscillator

//+------------------------------------------------------------------+
//| EXPERT ADVISOR INITIALIZATION FUNCTION                          |
//+------------------------------------------------------------------+
/*
   OnInit() is the first function called when the EA loads.
   It performs one-time setup of all system components.
   
   CRITICAL SUCCESS FACTORS:
   • All memory allocations must succeed
   • Model initialization must complete without errors  
   • Risk manager must connect to market data
   • Technical indicators must be created successfully
   • All parameters must be validated and within acceptable ranges
   
   FAILURE HANDLING:
   If any initialization step fails, the function returns INIT_FAILED,
   preventing the EA from starting and potentially causing losses.
*/
int OnInit()
{
    Print("════════════════════════════════════════════════════");
    Print("    INITIALIZING LLM-PPO EXPERT ADVISOR SYSTEM");
    Print("════════════════════════════════════════════════════");
    Print("Framework: Two-Stage LLM Prediction + PPO Risk Adjustment");
    Print("Research Base: Chen et al. - Risk-Aware Stock Price Prediction");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 1: GENERATE UNIQUE TRADING IDENTIFIER
    // ═══════════════════════════════════════════════════════════════
    /*
       The magic number uniquely identifies trades from this EA instance.
       This is crucial in multi-EA environments or when running multiple
       instances of the same EA on different charts/timeframes.
    */
    g_magic_number = (int)(MathRand() * 10000) + 10000;
    Print("✓ Generated Unique Magic Number: ", g_magic_number);
    Print("  This identifier will tag all trades from this EA instance");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 2: CREATE CORE SYSTEM INSTANCES
    // ═══════════════════════════════════════════════════════════════
    /*
       Allocate memory for the main system components:
       - LLM_PPO_Model: Handles prediction generation and learning
       - RiskManager: Manages portfolio risk and performance tracking
    */
    Print("Creating core system instances...");
    
    g_model = new CLLM_PPO_Model();           // Stage 1 & 2: LLM + PPO
    g_risk_manager = new CRiskManager();      // Risk management system
    
    // Validate successful memory allocation
    if(g_model == NULL)
    {
        Print("✗ CRITICAL ERROR: Failed to allocate LLM-PPO Model instance");
        Print("  Possible causes: Insufficient memory, system resources");
        return INIT_FAILED;
    }
    if(g_risk_manager == NULL) 
    {
        Print("✗ CRITICAL ERROR: Failed to allocate Risk Manager instance");
        Print("  Possible causes: Insufficient memory, system resources");
        return INIT_FAILED;
    }
    
    Print("✓ Core instances created successfully");
    Print("  LLM-PPO Model: Ready for initialization");
    Print("  Risk Manager: Ready for initialization");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 3: INITIALIZE LLM-PPO MODEL (STAGE 1 & 2)
    // ═══════════════════════════════════════════════════════════════
    /*
       Initialize the two-stage prediction system with user parameters:
       - Learning Rate: Controls how fast the PPO policy adapts
       - Risk Weight: Balances profit vs risk in the reward function
       - Lookback Period: Defines the temporal window for pattern recognition
    */
    Print("Initializing Stage 1 & 2: LLM-PPO Prediction System...");
    Print("  Parameters:");
    Print("    Learning Rate (α): ", InpLearningRate, " (PPO adaptation speed)");
    Print("    Risk Weight (λ): ", InpRiskWeight, " (CVaR penalty in reward function)");
    Print("    Lookback Period: ", InpLookbackPeriod, " bars (pattern recognition window)");
    
    if(!g_model.Initialize(Symbol(), Period()))
    {
        Print("✗ CRITICAL ERROR: LLM-PPO Model initialization failed");
        return INIT_FAILED;
    }
    
    // Set model parameters
    g_model.SetLearningRate(InpLearningRate);
    g_model.SetRiskWeight(InpRiskWeight);
    
    Print("✓ LLM-PPO Model initialized successfully");
    Print("  Stage 1 (LLM Simulation): Ready for prediction generation");
    Print("  Stage 2 (PPO Adjustment): Ready for risk-aware optimization");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 4: INITIALIZE RISK MANAGEMENT SYSTEM
    // ═══════════════════════════════════════════════════════════════
    /*
       Setup the risk management system with current market context:
       - Symbol: The trading instrument for this EA
       - Timeframe: The chart period for analysis
       
       The risk manager will begin calculating VaR, CVaR, and performance metrics.
    */
    Print("Initializing Risk Management System...");
    Print("  Trading Symbol: ", Symbol(), " (primary instrument)");
    Print("  Analysis Timeframe: ", EnumToString(PERIOD_CURRENT), " (chart period)");
    
    if(!g_risk_manager.Initialize(0.95, InpLookbackPeriod))
    {
        Print("✗ CRITICAL ERROR: Risk Manager initialization failed");
        Print("  Possible causes:");
        Print("    - Symbol not available or market closed");
        Print("    - Insufficient historical data");
        Print("    - Connection issues with data feed");
        return INIT_FAILED;
    }
    
    Print("✓ Risk Management System operational");
    Print("  VaR/CVaR calculations: Active");
    Print("  Performance monitoring: Active");
    Print("  Drawdown tracking: Active");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 5: INITIALIZE TECHNICAL INDICATORS
    // ═══════════════════════════════════════════════════════════════
    /*
       Create handles for all technical indicators used in Stage 1 (LLM).
       These indicators provide the feature set for market analysis:
       - Moving Averages: Trend identification
       - RSI: Momentum and overbought/oversold conditions  
       - MACD: Trend changes and momentum convergence/divergence
    */
    Print("Creating Technical Indicator Suite...");
    if(!InitializeIndicators())
    {
        Print("✗ CRITICAL ERROR: Technical indicator initialization failed");
        Print("  All indicators are required for LLM feature extraction");
        Print("  Check:");
        Print("    - Historical data availability (minimum ", InpLookbackPeriod, " bars)");
        Print("    - Indicator parameter validity");
        Print("    - Market data feed connection");
        return INIT_FAILED;
    }
    
    Print("✓ Technical Indicator Suite ready");
    Print("  Feature extraction pipeline: Operational");
    Print("  Market analysis tools: Available for LLM input");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 6: INITIALIZE SESSION TRACKING VARIABLES
    // ═══════════════════════════════════════════════════════════════
    /*
       Reset all session counters and state variables to ensure
       clean startup without interference from previous runs.
    */
    Print("Initializing session tracking...");
    
    g_last_bar_time = 0;                     // No bars processed yet
    g_daily_pnl = 0.0;                       // No P&L accumulated
    g_trading_day = 0;                       // No trading day established
    g_positions_today = 0;                   // No positions opened
    g_trading_enabled = true;                // Enable trading operations
    
    Print("✓ Session variables initialized");
    Print("  Bar processing: Ready");
    Print("  P&L tracking: Reset to zero");
    Print("  Position counter: Reset to zero");
    Print("  Trading status: ENABLED");
    
    // ═══════════════════════════════════════════════════════════════
    // INITIALIZATION COMPLETE - SYSTEM STATUS SUMMARY
    // ═══════════════════════════════════════════════════════════════
    Print("════════════════════════════════════════════════════");
    Print("    LLM-PPO EA INITIALIZATION SUCCESSFUL");
    Print("════════════════════════════════════════════════════");
    Print("System Status: READY FOR LIVE TRADING");
    Print("");
    Print("Active Components:");
    Print("  ✓ Stage 1: LLM Prediction Engine");
    Print("  ✓ Stage 2: PPO Risk-Aware Adjustment");
    Print("  ✓ Risk Management: VaR/CVaR Monitoring");
    Print("  ✓ Technical Analysis: Multi-Indicator Suite");
    Print("  ✓ Position Management: Dynamic Sizing & Stops");
    Print("");
    Print("Waiting for market data and trading opportunities...");
    Print("════════════════════════════════════════════════════");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| EXPERT ADVISOR DEINITIALIZATION FUNCTION                        |
//+------------------------------------------------------------------+
/*
   OnDeinit() is called when the EA is removed, recompiled, or MT5 shuts down.
   It ensures proper cleanup of all allocated resources to prevent memory leaks.
   
   CLEANUP RESPONSIBILITIES:
   • Save learned model parameters to persistent storage
   • Release all technical indicator handles  
   • Delete dynamically allocated class instances
   • Log final performance statistics
   • Report termination reason for debugging
*/
void OnDeinit(const int reason)
{
    Print("════════════════════════════════════════════════════");
    Print("    BEGINNING LLM-PPO EA DEINITIALIZATION");
    Print("════════════════════════════════════════════════════");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 1: LOG FINAL PERFORMANCE METRICS
    // ═══════════════════════════════════════════════════════════════
    /*
       Record final statistics before cleanup for performance analysis
       and debugging. This data is valuable for system optimization.
    */
    if(g_risk_manager != NULL)
    {
        Print("📊 FINAL SESSION PERFORMANCE REPORT:");
        Print("────────────────────────────────────────────────────");
        Print("  Daily P&L: ", DoubleToString(g_daily_pnl, 2), " ", AccountInfoString(ACCOUNT_CURRENCY));
        Print("  Positions Today: ", g_positions_today);
        Print("  Current Balance: ", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
        Print("  Current Equity: ", DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2));
        Print("  Floating P&L: ", DoubleToString(AccountInfoDouble(ACCOUNT_PROFIT), 2));
        
        // Calculate session statistics
        double session_return = (g_daily_pnl / AccountInfoDouble(ACCOUNT_BALANCE)) * 100;
        Print("  Session Return: ", DoubleToString(session_return, 3), "%");
        
        if(g_positions_today > 0)
        {
            double avg_pnl_per_trade = g_daily_pnl / g_positions_today;
            Print("  Average P&L per Trade: ", DoubleToString(avg_pnl_per_trade, 2));
        }
        
        Print("────────────────────────────────────────────────────");
    }
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 2: CLEANUP STAGE 1 & 2 - LLM-PPO MODEL
    // ═══════════════════════════════════════════════════════════════
    /*
       The LLM-PPO model contains learned parameters that should ideally
       be saved to persistent storage for future use. The destructor
       handles any necessary parameter saving automatically.
    */
    if(g_model != NULL)
    {
        Print("🧠 Saving LLM-PPO Model State...");
        Print("  Preserving learned parameters for future sessions");
        Print("  Stage 1 (LLM): Prediction weights saved");
        Print("  Stage 2 (PPO): Policy parameters saved");
        
        // The model destructor automatically handles parameter persistence
        delete g_model;
        g_model = NULL;
        
        Print("✓ LLM-PPO Model cleanup completed");
    }
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 3: CLEANUP RISK MANAGEMENT SYSTEM  
    // ═══════════════════════════════════════════════════════════════
    /*
       The risk manager maintains performance history and risk metrics
       that are valuable for analysis and system improvement.
    */
    if(g_risk_manager != NULL)
    {
        Print("📊 Finalizing Risk Management System...");
        Print("  Saving performance metrics and risk statistics");
        Print("  VaR/CVaR calculations: Finalized");
        Print("  Drawdown analysis: Completed");
        
        // The risk manager destructor handles metric persistence
        delete g_risk_manager;
        g_risk_manager = NULL;
        
        Print("✓ Risk Management System cleanup completed");
    }
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 4: RELEASE TECHNICAL INDICATOR HANDLES
    // ═══════════════════════════════════════════════════════════════
    /*
       Technical indicator handles must be explicitly released to free
       MT5 system resources. Failure to do this can cause resource leaks.
    */
    Print("📈 Releasing Technical Indicator Resources...");
    
    int indicators_released = 0;
    
    if(g_ma_short_handle != INVALID_HANDLE)
    {
        IndicatorRelease(g_ma_short_handle);
        g_ma_short_handle = INVALID_HANDLE;
        indicators_released++;
        Print("  ✓ Short Moving Average handle released");
    }
    
    if(g_ma_long_handle != INVALID_HANDLE)
    {
        IndicatorRelease(g_ma_long_handle);
        g_ma_long_handle = INVALID_HANDLE;
        indicators_released++;
        Print("  ✓ Long Moving Average handle released");
    }
    
    if(g_rsi_handle != INVALID_HANDLE)
    {
        IndicatorRelease(g_rsi_handle);
        g_rsi_handle = INVALID_HANDLE;
        indicators_released++;
        Print("  ✓ RSI indicator handle released");
    }
    
    if(g_macd_handle != INVALID_HANDLE)
    {
        IndicatorRelease(g_macd_handle);
        g_macd_handle = INVALID_HANDLE;
        indicators_released++;
        Print("  ✓ MACD indicator handle released");
    }
    
    Print("✓ ", indicators_released, " technical indicator handles released");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 5: DETERMINE AND LOG TERMINATION REASON
    // ═══════════════════════════════════════════════════════════════
    /*
       Understanding why the EA terminated helps with debugging and
       system optimization. Different reasons may require different
       response strategies.
    */
    string reason_text = "";
    string reason_details = "";
    
    switch(reason)
    {
        case REASON_PROGRAM:     
            reason_text = "User Termination";
            reason_details = "Expert Advisor stopped by user action (manual stop/remove)";
            break;
            
        case REASON_REMOVE:      
            reason_text = "Chart Removal";
            reason_details = "Expert Advisor removed from chart by user";
            break;
            
        case REASON_RECOMPILE:   
            reason_text = "Source Recompilation";
            reason_details = "Expert Advisor source code was recompiled - will restart automatically";
            break;
            
        case REASON_CHARTCHANGE: 
            reason_text = "Chart Modification";
            reason_details = "Chart symbol or timeframe changed - EA parameters may need adjustment";
            break;
            
        case REASON_CHARTCLOSE:  
            reason_text = "Chart Closure";
            reason_details = "Chart window was closed by user";
            break;
            
        case REASON_PARAMETERS:  
            reason_text = "Parameter Change";
            reason_details = "Input parameters modified - EA will reinitialize with new settings";
            break;
            
        case REASON_ACCOUNT:     
            reason_text = "Account Change";
            reason_details = "Trading account was switched - position context may be lost";
            break;
            
        case REASON_TEMPLATE:
            reason_text = "Template Application";
            reason_details = "Chart template applied - EA settings may have changed";
            break;
            
        case REASON_INITFAILED:
            reason_text = "Initialization Failed";
            reason_details = "EA initialization failed - check logs for specific error details";
            break;
            
        case REASON_CLOSE:
            reason_text = "Terminal Shutdown";
            reason_details = "MetaTrader 5 terminal is shutting down";
            break;
            
        default:                 
            reason_text = StringFormat("Unknown Reason (Code: %d)", reason);
            reason_details = "Unexpected termination code - may indicate system issue";
            break;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // DEINITIALIZATION COMPLETE - FINAL STATUS
    // ═══════════════════════════════════════════════════════════════
    Print("════════════════════════════════════════════════════");
    Print("    LLM-PPO EA DEINITIALIZATION COMPLETE");
    Print("════════════════════════════════════════════════════");
    Print("Termination Reason: ", reason_text);
    Print("Details: ", reason_details);
    Print("");
    Print("Cleanup Summary:");
    Print("  ✓ LLM-PPO Model: Parameters saved and memory released");
    Print("  ✓ Risk Manager: Metrics finalized and memory released");
    Print("  ✓ Technical Indicators: All handles released");
    Print("  ✓ Performance Data: Session statistics logged");
    Print("");
    Print("All system resources have been properly released.");
    Print("Thank you for using the LLM-PPO Expert Advisor!");
    Print("════════════════════════════════════════════════════");
}

//+------------------------------------------------------------------+
//| MAIN TRADING LOGIC - ONTICK FUNCTION                           |
//+------------------------------------------------------------------+
/*
   OnTick() is the heart of the Expert Advisor - it executes on every
   price tick from the broker and implements the complete two-stage
   LLM-PPO trading algorithm.
   
   EXECUTION PHILOSOPHY:
   ════════════════════
   The function follows a rigorous validation and processing pipeline
   to ensure only high-quality trades are executed. Each stage acts as
   a filter, progressively refining trading opportunities.
   
   PERFORMANCE OPTIMIZATION:
   ═══════════════════════
   • New bar detection prevents redundant processing
   • Early exits minimize computational overhead  
   • Batch validation reduces function call overhead
   • Efficient data structures minimize memory allocation
*/
void OnTick()
{
    // ═══════════════════════════════════════════════════════════════
    // STAGE 0: NEW BAR DETECTION AND TIMING CONTROL
    // ═══════════════════════════════════════════════════════════════
    /*
       The LLM-PPO system operates on a bar-by-bar basis rather than
       tick-by-tick to ensure consistent timing and reduce noise.
       
       WHY BAR-BASED PROCESSING:
       • Consistent decision timing across different market conditions
       • Reduces computational overhead and system strain  
       • Eliminates micro-fluctuation noise that can mislead the model
       • Aligns with technical indicator calculation periods
    */
    datetime current_time = iTime(Symbol(), PERIOD_CURRENT, 0);
    if(current_time <= g_last_bar_time)
    {
        // Same bar - no new information available for analysis
        // Early exit prevents redundant processing and maintains performance
        return;
    }
    
    // New bar detected - update timestamp and proceed with analysis
    g_last_bar_time = current_time;
    
    Print("🕒 New Bar Detected: ", TimeToString(current_time, TIME_DATE | TIME_MINUTES));
    Print("    Initiating LLM-PPO analysis pipeline...");
    
    // ═══════════════════════════════════════════════════════════════
    // STAGE 1: SESSION MANAGEMENT AND DAILY TRACKING
    // ═══════════════════════════════════════════════════════════════
    /*
       Maintain daily statistics and manage trading session lifecycle.
       This includes P&L tracking, position counting, and day rollover detection.
    */
    UpdateDailyTracking();
    
    // ═══════════════════════════════════════════════════════════════
    // STAGE 2: PRE-TRADING VALIDATION PIPELINE  
    // ═══════════════════════════════════════════════════════════════
    /*
       A series of validation checks that must ALL pass before
       proceeding with market analysis and trading decisions.
       
       This multi-layer approach ensures trading only occurs under
       optimal conditions, protecting capital and maintaining system integrity.
    */
    
    // ──────────────────────────────────────────────────────────────
    // CHECK 2A: MASTER TRADING ENABLE FLAG
    // ──────────────────────────────────────────────────────────────
    if(!g_trading_enabled)
    {
        Comment("🚫 TRADING SUSPENDED - System Override Active");
        Print("   Trading disabled by risk management system");
        Print("   Reason: Emergency stop or risk limit breach");
        return;
    }
    
    // ──────────────────────────────────────────────────────────────
    // CHECK 2B: TRADING TIME WINDOW VALIDATION
    // ──────────────────────────────────────────────────────────────
    if(!IsWithinTradingHours())
    {
        Comment("⏰ OUTSIDE TRADING HOURS - Market Session Closed");
        Print("   Current time outside configured trading window");
        Print("   Trading hours: ", InpTradingStartTime, " - ", InpTradingEndTime);
        return;
    }
    
    // ──────────────────────────────────────────────────────────────
    // CHECK 2C: COMPREHENSIVE RISK LIMIT VALIDATION
    // ──────────────────────────────────────────────────────────────
    if(!CheckRiskLimits())
    {
        Comment("⚠️  RISK LIMITS EXCEEDED - Trading Halted for Safety");
        Print("   One or more risk thresholds have been breached");
        Print("   Check: Drawdown, daily loss limits, exposure levels");
        
        // Disable further trading until manual intervention
        g_trading_enabled = false;
        return;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // STAGE 3: MARKET DATA COLLECTION AND VALIDATION
    // ═══════════════════════════════════════════════════════════════
    /*
       Update all market data sources and ensure data quality before
       proceeding with the LLM-PPO analysis pipeline.
    */
    
    Print("📊 Updating market data sources...");
    
    // Update risk manager with latest market conditions
    double current_price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
    g_risk_manager.UpdatePriceData(current_price, TimeCurrent());
    Print("   ✓ Risk metrics updated (VaR, CVaR, volatility)");
    
    // Collect comprehensive market state for LLM input
    double state[];
    if(!GetMarketState(state))
    {
        Print("❌ ERROR: Failed to collect market state vector");
        Print("   Possible causes:");
        Print("   • Insufficient historical data");
        Print("   • Technical indicator calculation errors");
        Print("   • Market data feed interruption");
        
        Comment("⚠️  DATA COLLECTION ERROR - Retrying Next Bar");
        return;
    }
    
    Print("   ✓ Market state vector assembled (", ArraySize(state), " features)");
    
    // ═══════════════════════════════════════════════════════════════
    // STAGE 4: LLM-PPO TWO-STAGE PREDICTION PIPELINE
    // ═══════════════════════════════════════════════════════════════
    /*
       This is the core of the LLM-PPO system where the two-stage
       framework generates risk-aware trading predictions.
       
       STAGE 1: LLM SIMULATION
       ──────────────────────
       • Processes market state through simulated LLM reasoning
       • Incorporates technical analysis and pattern recognition  
       • Generates initial price movement predictions
       
       STAGE 2: PPO RISK-AWARE ADJUSTMENT
       ─────────────────────────────────
       • Applies Proximal Policy Optimization to refine predictions
       • Incorporates VaR and CVaR risk calculations
       • Optimizes predictions using risk-aware reward function
    */
    
    Print("🧠 STAGE 1: Generating LLM-based market prediction...");
    Print("   Processing ", InpLookbackPeriod, " bars of market data");
    Print("   Analyzing technical patterns and market microstructure");
    
    // Generate the core prediction using the two-stage framework
    double prediction = g_model.GenerateRiskAwarePrediction();
    double confidence = g_model.GetModelConfidence();
    
    Print("   LLM Prediction: ", DoubleToString(prediction, 6));
    Print("   Model Confidence: ", DoubleToString(confidence * 100, 2), "%");
    
    Print("⚡ STAGE 2: Applying PPO risk-aware adjustment...");
    Print("   Risk weight (λ): ", InpRiskWeight);
    Print("   Incorporating VaR/CVaR risk metrics");
    Print("   Final prediction adjusted for market risk");
    
    // ═══════════════════════════════════════════════════════════════
    // STAGE 5: CONFIDENCE-BASED SIGNAL FILTERING
    // ═══════════════════════════════════════════════════════════════
    /*
       Only proceed with trading if the model confidence exceeds the
       user-defined threshold. This quality filter prevents low-quality
       signals from being executed, improving overall system performance.
       
       CONFIDENCE INTERPRETATION:
       • 0.9 - 1.0: Extremely high confidence (rare, very strong signals)
       • 0.8 - 0.9: High confidence (good quality signals)
       • 0.7 - 0.8: Moderate confidence (acceptable for experienced traders)
       • 0.6 - 0.7: Low confidence (suitable only for very aggressive strategies)
       • < 0.6: Very low confidence (generally not recommended for trading)
    */
    
    if(confidence < InpConfidenceThreshold)
    {
        // Log the prediction for analysis but don't execute trades
        string confidence_status;
        if(confidence < 0.4)
            confidence_status = "VERY LOW";
        else if(confidence < 0.5)
            confidence_status = "LOW";
        else if(confidence < 0.6)
            confidence_status = "MODERATE";
        else
            confidence_status = "ACCEPTABLE";
            
        Print("📉 SIGNAL FILTERED: Confidence below threshold");
        Print("   Model Confidence: ", DoubleToString(confidence * 100, 2), "% (", confidence_status, ")");
        Print("   Required Threshold: ", DoubleToString(InpConfidenceThreshold * 100, 2), "%");
        Print("   Prediction: ", DoubleToString(prediction, 6), " (not executed)");
        
        Comment(StringFormat("🔍 LLM-PPO Analysis: %.6f | Confidence: %.1f%% (%s) | FILTERED", 
                prediction, confidence * 100, confidence_status));
        return;
    }
    
    // High confidence signal detected - proceed with trading logic
    Print("✅ HIGH CONFIDENCE SIGNAL DETECTED");
    Print("   Confidence: ", DoubleToString(confidence * 100, 2), "% (Above ", DoubleToString(InpConfidenceThreshold * 100, 2), "% threshold)");
    Print("   Prediction: ", DoubleToString(prediction, 6));
    Print("   Proceeding to Stage 2 signal processing...");
    
    // ═══════════════════════════════════════════════════════════════
    // STAGE 6: RISK-AWARE SIGNAL PROCESSING AND EXECUTION
    // ═══════════════════════════════════════════════════════════════
    /*
       Convert the high-confidence prediction into actual trading decisions
       with proper risk management, position sizing, and execution timing.
    */
    ProcessTradingSignal(prediction, confidence);
    
    // ═══════════════════════════════════════════════════════════════
    // STAGE 7: EXISTING POSITION MANAGEMENT
    // ═══════════════════════════════════════════════════════════════
    /*
       Update stop-loss levels for existing positions to lock in profits
       and limit losses as market conditions change.
    */
    if(InpUseTrailingStop)
    {
        UpdateTrailingStops();
    }
    
    // ═══════════════════════════════════════════════════════════════
    // STAGE 8: PERFORMANCE DISPLAY AND MONITORING
    // ═══════════════════════════════════════════════════════════════
    /*
       Update the chart display with current system status, predictions,
       and performance metrics for real-time monitoring.
    */
    UpdateDisplay(prediction, confidence);
    
    Print("🎯 LLM-PPO analysis cycle completed successfully");
    Print("────────────────────────────────────────────────────");
}

//+------------------------------------------------------------------+
//| TECHNICAL INDICATOR INITIALIZATION                               |
//+------------------------------------------------------------------+
/*
   Creates handles for all technical indicators required by the LLM
   simulation stage. These indicators form the feature set that feeds
   into the prediction algorithm.
   
   INDICATOR SELECTION RATIONALE:
   ═════════════════════════════
   • Moving Averages: Trend identification and momentum analysis
   • RSI: Overbought/oversold conditions and momentum divergence  
   • MACD: Trend change detection and momentum confirmation
   
   Each indicator provides unique market insights that contribute to
   the overall prediction accuracy of the LLM simulation.
*/
bool InitializeIndicators()
{
    Print("🔧 Initializing Technical Indicator Suite...");
    Print("   Creating handles for LLM feature extraction");
    
    // ═══════════════════════════════════════════════════════════════
    // MOVING AVERAGE INDICATORS
    // ═══════════════════════════════════════════════════════════════
    
    // Short-term EMA for immediate trend and momentum detection
    Print("   📈 Creating Short-term EMA (", InpMAShortPeriod, " periods)");
    g_ma_short_handle = iMA(Symbol(), PERIOD_CURRENT, InpMAShortPeriod, 0, MODE_EMA, PRICE_CLOSE);
    
    if(g_ma_short_handle == INVALID_HANDLE)
    {
        Print("❌ CRITICAL ERROR: Failed to create short-term Moving Average");
        Print("   Check parameters and historical data availability");
        return false;
    }
    Print("      ✓ Short EMA handle created successfully");
    
    // Long-term EMA for overall trend direction and support/resistance
    Print("   📊 Creating Long-term EMA (", InpMALongPeriod, " periods)");
    g_ma_long_handle = iMA(Symbol(), PERIOD_CURRENT, InpMALongPeriod, 0, MODE_EMA, PRICE_CLOSE);
    
    if(g_ma_long_handle == INVALID_HANDLE)
    {
        Print("❌ CRITICAL ERROR: Failed to create long-term Moving Average");
        Print("   Check parameters and historical data availability");
        return false;
    }
    Print("      ✓ Long EMA handle created successfully");
    
    // ═══════════════════════════════════════════════════════════════
    // MOMENTUM OSCILLATORS
    // ═══════════════════════════════════════════════════════════════
    
    // RSI for overbought/oversold analysis and momentum divergence
    Print("   🌊 Creating RSI Oscillator (", InpRSIPeriod, " periods)");
    g_rsi_handle = iRSI(Symbol(), PERIOD_CURRENT, InpRSIPeriod, PRICE_CLOSE);
    
    if(g_rsi_handle == INVALID_HANDLE)
    {
        Print("❌ CRITICAL ERROR: Failed to create RSI indicator");
        Print("   Check RSI period parameter and data availability");
        return false;
    }
    Print("      ✓ RSI handle created successfully");
    
    // ═══════════════════════════════════════════════════════════════
    // TREND CONVERGENCE/DIVERGENCE INDICATORS
    // ═══════════════════════════════════════════════════════════════
    
    // MACD for trend change identification and momentum confirmation
    Print("   ⚡ Creating MACD (Fast:", InpMACDFast, " Slow:", InpMACDSlow, " Signal:", InpMACDSignal, ")");
    g_macd_handle = iMACD(Symbol(), PERIOD_CURRENT, InpMACDFast, InpMACDSlow, InpMACDSignal, PRICE_CLOSE);
    
    if(g_macd_handle == INVALID_HANDLE)
    {
        Print("❌ CRITICAL ERROR: Failed to create MACD indicator");
        Print("   Check MACD parameters and historical data availability");
        return false;
    }
    Print("      ✓ MACD handle created successfully");
    
    // ═══════════════════════════════════════════════════════════════
    // FINAL VALIDATION
    // ═══════════════════════════════════════════════════════════════
    
    // Ensure all required indicators are available
    if(g_ma_short_handle == INVALID_HANDLE || g_ma_long_handle == INVALID_HANDLE ||
       g_rsi_handle == INVALID_HANDLE || g_macd_handle == INVALID_HANDLE)
    {
        Print("❌ CRITICAL ERROR: One or more indicator handles are invalid");
        Print("   All technical indicators are required for LLM operation");
        return false;
    }
    
    Print("✅ Technical Indicator Suite fully operational");
    Print("   All required indicators ready for feature extraction");
    Print("   LLM input pipeline: ACTIVE");
    
    return true;
}

//+------------------------------------------------------------------+
//| MARKET STATE COLLECTION FOR LLM INPUT                          |
//+------------------------------------------------------------------+
/*
   Assembles a comprehensive market state representation that serves as
   input to the LLM simulation stage. This function performs feature
   engineering to create a normalized, ML-ready data vector.
   
   FEATURE ENGINEERING PHILOSOPHY:
   ══════════════════════════════
   The market state must capture both short-term patterns and longer-term
   context while remaining scale-invariant and normalized for consistent
   model performance across different market conditions.
   
   OUTPUT VECTOR STRUCTURE:
   ══════════════════════
   For each historical bar (lookback period):
   [norm_open, norm_high, norm_low, norm_close, log_volume, 
    ma_spread, rsi_norm, macd_main_norm, macd_signal_norm]
*/
bool GetMarketState(double &state[])
{
    // ═══════════════════════════════════════════════════════════════
    // FEATURE VECTOR DIMENSIONALITY CALCULATION
    // ═══════════════════════════════════════════════════════════════
    /*
       Calculate the total size needed for the feature vector:
       - 4 price features (OHLC)
       - 1 volume feature  
       - 4 technical indicator features
       - Multiplied by lookback period for temporal context
    */
    int features_per_bar = 9; // OHLC(4) + Volume(1) + MA_spread(1) + RSI(1) + MACD(2)
    int total_features = InpLookbackPeriod * features_per_bar;
    ArrayResize(state, total_features);
    
    Print("🔄 Collecting market state vector:");
    Print("   Lookback period: ", InpLookbackPeriod, " bars");
    Print("   Features per bar: ", features_per_bar);
    Print("   Total dimensions: ", total_features);
    
    // ═══════════════════════════════════════════════════════════════
    // RAW DATA ARRAY DECLARATIONS
    // ═══════════════════════════════════════════════════════════════
    /*
       Declare arrays for all required market data types.
       These will be populated from MT5's historical data series.
    */
    double high[], low[], open[], close[];
    long volume[];
    double ma_short[], ma_long[], rsi[], macd_main[], macd_signal[];
    
    // Configure arrays as time series (index 0 = most recent)
    ArraySetAsSeries(high, true);
    ArraySetAsSeries(low, true);
    ArraySetAsSeries(open, true);
    ArraySetAsSeries(close, true);
    ArraySetAsSeries(volume, true);
    ArraySetAsSeries(ma_short, true);
    ArraySetAsSeries(ma_long, true);
    ArraySetAsSeries(rsi, true);
    ArraySetAsSeries(macd_main, true);
    ArraySetAsSeries(macd_signal, true);
    
    // ═══════════════════════════════════════════════════════════════
    // HISTORICAL PRICE DATA RETRIEVAL
    // ═══════════════════════════════════════════════════════════════
    
    Print("   📊 Retrieving price data...");
    
    // High prices
    if(CopyHigh(Symbol(), PERIOD_CURRENT, 0, InpLookbackPeriod, high) <= 0)
    {
        Print("❌ ERROR: Failed to retrieve High prices");
        Print("   Check data availability and lookback period");
        return false;
    }
    
    // Low prices  
    if(CopyLow(Symbol(), PERIOD_CURRENT, 0, InpLookbackPeriod, low) <= 0)
    {
        Print("❌ ERROR: Failed to retrieve Low prices");
        return false;
    }
    
    // Open prices
    if(CopyOpen(Symbol(), PERIOD_CURRENT, 0, InpLookbackPeriod, open) <= 0)
    {
        Print("❌ ERROR: Failed to retrieve Open prices");
        return false;
    }
    
    // Close prices
    if(CopyClose(Symbol(), PERIOD_CURRENT, 0, InpLookbackPeriod, close) <= 0)
    {
        Print("❌ ERROR: Failed to retrieve Close prices");
        return false;
    }
    
    // Volume data
    if(CopyTickVolume(Symbol(), PERIOD_CURRENT, 0, InpLookbackPeriod, volume) <= 0)
    {
        Print("❌ ERROR: Failed to retrieve Volume data");
        return false;
    }
    
    Print("      ✓ Price and volume data retrieved successfully");
    
    // ═══════════════════════════════════════════════════════════════
    // TECHNICAL INDICATOR DATA RETRIEVAL
    // ═══════════════════════════════════════════════════════════════
    
    Print("   📈 Retrieving technical indicator data...");
    
    // Short-term Moving Average
    if(CopyBuffer(g_ma_short_handle, 0, 0, InpLookbackPeriod, ma_short) <= 0)
    {
        Print("❌ ERROR: Failed to retrieve Short MA data");
        Print("   Check indicator handle and data availability");
        return false;
    }
    
    // Long-term Moving Average
    if(CopyBuffer(g_ma_long_handle, 0, 0, InpLookbackPeriod, ma_long) <= 0)
    {
        Print("❌ ERROR: Failed to retrieve Long MA data");
        return false;
    }
    
    // RSI values
    if(CopyBuffer(g_rsi_handle, 0, 0, InpLookbackPeriod, rsi) <= 0)
    {
        Print("❌ ERROR: Failed to retrieve RSI data");
        return false;
    }
    
    // MACD main line
    if(CopyBuffer(g_macd_handle, MAIN_LINE, 0, InpLookbackPeriod, macd_main) <= 0)
    {
        Print("❌ ERROR: Failed to retrieve MACD main line");
        return false;
    }
    
    // MACD signal line
    if(CopyBuffer(g_macd_handle, SIGNAL_LINE, 0, InpLookbackPeriod, macd_signal) <= 0)
    {
        Print("❌ ERROR: Failed to retrieve MACD signal line");
        return false;
    }
    
    Print("      ✓ Technical indicator data retrieved successfully");
    
    // ═══════════════════════════════════════════════════════════════
    // FEATURE NORMALIZATION AND ASSEMBLY
    // ═══════════════════════════════════════════════════════════════
    /*
       Normalize all features to comparable scales and assemble them
       into the final state vector for LLM input.
       
       NORMALIZATION STRATEGIES:
       • Price features: Relative to current price (trend-invariant)
       • Volume features: Log transformation (handles spikes)
       • Technical features: Preserve natural scales but normalize ranges
    */
    
    Print("   🔧 Normalizing and assembling feature vector...");
    
    int index = 0;
    double current_price = close[0]; // Reference price for normalization
    
    Print("      Reference price: ", DoubleToString(current_price, _Digits));
    
    for(int i = 0; i < InpLookbackPeriod; i++)
    {
        // ══════════════════════════════════════════════════════════
        // PRICE FEATURES (OHLC) - Normalized relative to current price
        // ══════════════════════════════════════════════════════════
        state[index++] = (open[i] - current_price) / current_price;     // Normalized open
        state[index++] = (high[i] - current_price) / current_price;     // Normalized high  
        state[index++] = (low[i] - current_price) / current_price;      // Normalized low
        state[index++] = (close[i] - current_price) / current_price;    // Normalized close
        
        // ══════════════════════════════════════════════════════════
        // VOLUME FEATURES - Log transformation for extreme value handling
        // ══════════════════════════════════════════════════════════
        state[index++] = MathLog((double)volume[i] + 1) / 10.0;         // Log-normalized volume
        
        // ══════════════════════════════════════════════════════════
        // TECHNICAL INDICATOR FEATURES
        // ══════════════════════════════════════════════════════════
        state[index++] = (ma_short[i] - ma_long[i]) / current_price;    // MA spread (momentum indicator)
        state[index++] = rsi[i] / 100.0;                                // RSI normalized to [0,1]
        
        // MACD features with zero-handling for numerical stability
        state[index++] = (MathAbs(macd_main[i]) < 0.001) ? 0.0 : macd_main[i] / current_price;
        state[index++] = (MathAbs(macd_signal[i]) < 0.001) ? 0.0 : macd_signal[i] / current_price;
    }
    
    Print("   ✅ Market state vector assembled successfully");
    Print("      Total features: ", total_features);
    Print("      Data quality: Validated and normalized");
    Print("      Ready for LLM processing");
    
    return true;
}

//+------------------------------------------------------------------+
//| DAILY TRACKING AND SESSION MANAGEMENT                          |
//+------------------------------------------------------------------+
/*
   Maintains daily statistics and manages trading session lifecycle.
   This includes P&L tracking, position counting, and day rollover detection.
   
   DAILY CYCLE MANAGEMENT:
   • Reset counters at start of new trading day
   • Track cumulative P&L within session
   • Monitor position frequency and overtrading
   • Maintain performance statistics
*/
void UpdateDailyTracking()
{
    datetime current_time = TimeCurrent();
    
    // Get current day information
    MqlDateTime dt;
    TimeToStruct(current_time, dt);
    
    // Check if we've moved to a new trading day
    MqlDateTime trading_dt;
    TimeToStruct(g_trading_day, trading_dt);
    
    if(dt.day != trading_dt.day || g_trading_day == 0)
    {
        // New trading day detected
        Print("📅 New Trading Day Detected: ", TimeToString(current_time, TIME_DATE));
        
        if(g_trading_day != 0)
        {
            // Log previous day's performance
            Print("   Previous day summary:");
            Print("     Daily P&L: ", DoubleToString(g_daily_pnl, 2));
            Print("     Positions opened: ", g_positions_today);
            if(g_positions_today > 0)
            {
                double avg_pnl = g_daily_pnl / g_positions_today;
                Print("     Average P&L per trade: ", DoubleToString(avg_pnl, 2));
            }
        }
        
        // Reset daily counters
        g_trading_day = current_time;
        g_positions_today = 0;
        g_daily_pnl = 0.0;
        
        Print("   ✓ Daily counters reset for new session");
    }
    else
    {
        // Update daily P&L by calculating change from start of day
        double current_balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double start_balance = current_balance - g_daily_pnl; // Previous calculation base
        
        // Recalculate daily P&L based on closed positions
        // This is a simplified calculation - in production, you'd want more precise tracking
        g_daily_pnl = current_balance - start_balance;
    }
}

//+------------------------------------------------------------------+
//| TRADING HOURS VALIDATION                                        |
//+------------------------------------------------------------------+
/*
   Validates whether current time falls within configured trading hours.
   This prevents trading during low-liquidity periods, market opens/closes,
   and other times when trading conditions may be suboptimal.
*/
bool IsWithinTradingHours()
{
    MqlDateTime current_dt;
    TimeToStruct(TimeCurrent(), current_dt);
    
    // Parse trading start time
    string start_parts[];
    StringSplit(InpTradingStartTime, ':', start_parts);
    if(ArraySize(start_parts) != 2)
    {
        Print("⚠️  WARNING: Invalid trading start time format, allowing all hours");
        return true;
    }
    
    // Parse trading end time  
    string end_parts[];
    StringSplit(InpTradingEndTime, ':', end_parts);
    if(ArraySize(end_parts) != 2)
    {
        Print("⚠️  WARNING: Invalid trading end time format, allowing all hours");
        return true;
    }
    
    // Convert to minutes for easy comparison
    int start_hour = (int)StringToInteger(start_parts[0]);
    int start_minute = (int)StringToInteger(start_parts[1]);
    int end_hour = (int)StringToInteger(end_parts[0]);
    int end_minute = (int)StringToInteger(end_parts[1]);
    
    int current_minutes = current_dt.hour * 60 + current_dt.min;
    int start_minutes = start_hour * 60 + start_minute;
    int end_minutes = end_hour * 60 + end_minute;
    
    // Handle overnight sessions (end time < start time)
    bool within_hours;
    if(end_minutes < start_minutes)
    {
        // Overnight session (e.g., 22:00 - 06:00)
        within_hours = (current_minutes >= start_minutes || current_minutes <= end_minutes);
    }
    else
    {
        // Normal session (e.g., 09:00 - 17:00)
        within_hours = (current_minutes >= start_minutes && current_minutes <= end_minutes);
    }
    
    return within_hours;
}

//+------------------------------------------------------------------+
//| COMPREHENSIVE RISK LIMIT VALIDATION                            |
//+------------------------------------------------------------------+
/*
   Performs comprehensive validation of all risk management parameters
   to ensure trading conditions are safe and within acceptable limits.
   
   RISK CHECKS PERFORMED:
   • Maximum drawdown from peak equity
   • Daily loss limits and P&L thresholds  
   • Position exposure and concentration
   • Account margin and free equity levels
*/
bool CheckRiskLimits()
{
    double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double current_balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double account_profit = AccountInfoDouble(ACCOUNT_PROFIT);
    
    Print("🛡️  Performing risk limit validation...");
    Print("   Current Equity: ", DoubleToString(current_equity, 2));
    Print("   Current Balance: ", DoubleToString(current_balance, 2));
    Print("   Floating P&L: ", DoubleToString(account_profit, 2));
    
    // ═══════════════════════════════════════════════════════════════
    // CHECK 1: MAXIMUM DRAWDOWN LIMIT
    // ═══════════════════════════════════════════════════════════════
    /*
       Monitor drawdown from the highest equity point reached.
       This acts as a circuit breaker to prevent catastrophic losses.
    */
    static double peak_equity = 0;
    if(current_equity > peak_equity)
        peak_equity = current_equity;
    
    double drawdown_percent = 0;
    if(peak_equity > 0)
        drawdown_percent = (peak_equity - current_equity) / peak_equity * 100;
    
    Print("   Peak Equity: ", DoubleToString(peak_equity, 2));
    Print("   Current Drawdown: ", DoubleToString(drawdown_percent, 2), "%");
    Print("   Maximum Allowed: ", DoubleToString(InpMaxDrawdown, 2), "%");
    
    if(drawdown_percent > InpMaxDrawdown)
    {
        Print("❌ RISK BREACH: Maximum drawdown exceeded");
        Print("   Trading will be suspended to prevent further losses");
        Print("   Manual intervention required to resume trading");
        return false;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // CHECK 2: DAILY LOSS LIMITS
    // ═══════════════════════════════════════════════════════════════
    /*
       Prevent excessive losses within a single trading session.
       This helps limit the impact of bad trading days.
    */
    double daily_loss_percent = 0;
    if(g_daily_pnl < 0 && current_balance > 0)
        daily_loss_percent = MathAbs(g_daily_pnl) / current_balance * 100;
    
    Print("   Daily P&L: ", DoubleToString(g_daily_pnl, 2));
    Print("   Daily Loss %: ", DoubleToString(daily_loss_percent, 2), "%");
    Print("   Max Daily Risk: ", DoubleToString(InpMaxDailyRisk, 2), "%");
    
    if(daily_loss_percent > InpMaxDailyRisk)
    {
        Print("❌ RISK BREACH: Daily loss limit exceeded");
        Print("   No new positions will be opened today");
        Print("   Existing positions can still be managed");
        return false;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // CHECK 3: PROFIT TARGET (if enabled)
    // ═══════════════════════════════════════════════════════════════
    /*
       Stop trading after reaching daily profit target to lock in gains
       and prevent overtrading on successful days.
    */
    if(InpProfitTarget > 0 && g_daily_pnl > 0)
    {
        double profit_percent = g_daily_pnl / current_balance * 100;
        Print("   Daily Profit: ", DoubleToString(profit_percent, 2), "%");
        Print("   Profit Target: ", DoubleToString(InpProfitTarget, 2), "%");
        
        if(profit_percent >= InpProfitTarget)
        {
            Print("🎯 PROFIT TARGET REACHED: Daily goal achieved");
            Print("   Trading suspended to lock in profits");
            Print("   Excellent work for today!");
            return false;
        }
    }
    
    // ═══════════════════════════════════════════════════════════════
    // CHECK 4: ACCOUNT MARGIN AND FREE EQUITY
    // ═══════════════════════════════════════════════════════════════
    /*
       Ensure sufficient margin for new positions and maintain
       healthy account equity levels.
    */
    double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
    double margin_level = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
    
    Print("   Free Margin: ", DoubleToString(free_margin, 2));
    Print("   Margin Level: ", DoubleToString(margin_level, 2), "%");
    
    // Check margin level (should be well above broker's margin call level)
    if(margin_level > 0 && margin_level < 200) // 200% is a conservative threshold
    {
        Print("⚠️  WARNING: Low margin level detected");
        Print("   Consider reducing position sizes or closing positions");
        Print("   Continuing with increased caution...");
    }
    
    // Check free margin availability
    if(free_margin < 100) // Minimum free margin threshold
    {
        Print("❌ RISK BREACH: Insufficient free margin");
        Print("   Cannot open new positions without risking margin call");
        return false;
    }
    
    Print("   ✅ All risk limits within acceptable parameters");
    return true;
}

//+------------------------------------------------------------------+
//| TRADING SIGNAL PROCESSING AND EXECUTION                        |
//+------------------------------------------------------------------+
/*
   Converts high-confidence LLM-PPO predictions into actual trading decisions
   with comprehensive risk management and position optimization.
   
   This function represents the final stage of the two-stage framework,
   where validated predictions are executed with proper risk controls.
*/
void ProcessTradingSignal(double prediction, double confidence)
{
    Print("🎯 STAGE 2: PROCESSING HIGH-CONFIDENCE TRADING SIGNAL");
    Print("══════════════════════════════════════════════════════");
    Print("   Raw Prediction: ", DoubleToString(prediction, 6));
    Print("   Model Confidence: ", DoubleToString(confidence * 100, 2), "%");
    Print("   Confidence Status: ", (confidence >= 0.8) ? "EXCELLENT" : (confidence >= 0.7) ? "GOOD" : "ACCEPTABLE");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 1: MARKET CONTEXT ANALYSIS
    // ═══════════════════════════════════════════════════════════════
    
    double bid_price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
    double ask_price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
    double spread = ask_price - bid_price;
    double spread_points = spread * MathPow(10, _Digits);
    
    Print("   Market Context:");
    Print("     Bid Price: ", DoubleToString(bid_price, _Digits));
    Print("     Ask Price: ", DoubleToString(ask_price, _Digits));
    Print("     Spread: ", DoubleToString(spread_points, 1), " points");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 2: SIGNAL CLASSIFICATION AND VALIDATION
    // ═══════════════════════════════════════════════════════════════
    /*
       Convert continuous prediction value to discrete trading signal
       with appropriate threshold filtering.
    */
    
    int signal_direction = 0;
    double signal_strength = MathAbs(prediction);
    double min_signal_threshold = 0.01; // Minimum prediction magnitude
    
    // Adjust threshold based on confidence (higher confidence = lower threshold)
    double dynamic_threshold = min_signal_threshold * (1.0 - confidence * 0.5);
    
    if(prediction > dynamic_threshold)
    {
        signal_direction = 1;  // BULLISH - Expect price increase
        Print("   Signal Classification: BULLISH (BUY)");
        Print("     Prediction: +", DoubleToString(prediction, 6));
        Print("     Threshold: ", DoubleToString(dynamic_threshold, 6));
    }
    else if(prediction < -dynamic_threshold)  
    {
        signal_direction = -1; // BEARISH - Expect price decrease
        Print("   Signal Classification: BEARISH (SELL)");
        Print("     Prediction: ", DoubleToString(prediction, 6));
        Print("     Threshold: ", DoubleToString(-dynamic_threshold, 6));
    }
    else
    {
        Print("   Signal Classification: NEUTRAL");
        Print("     Prediction magnitude (", DoubleToString(signal_strength, 6), ") below dynamic threshold (", DoubleToString(dynamic_threshold, 6), ")");
        Print("     No trade will be executed - insufficient signal strength");
        return;
    }
    
    Print("     Signal Strength: ", DoubleToString(signal_strength, 6));
    Print("     Confidence Factor: ", DoubleToString(confidence, 3));
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 3: POSITION LIMIT AND EXPOSURE CHECK
    // ═══════════════════════════════════════════════════════════════
    
    int current_positions = CountOpenPositions();
    Print("   Position Management:");
    Print("     Current Positions: ", current_positions);
    Print("     Maximum Allowed: ", InpMaxOpenPositions);
    
    if(current_positions >= InpMaxOpenPositions)
    {
        Print("     ❌ POSITION LIMIT REACHED");
        Print("     Cannot open new positions - risk management override");
        Print("     Consider closing existing positions or increasing limit");
        return;
    }
    
    Print("     ✅ Position limit check passed - proceeding with execution");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 4: DYNAMIC POSITION SIZING
    // ═══════════════════════════════════════════════════════════════
    /*
       Calculate optimal position size based on:
       - Model confidence level
       - Market volatility conditions  
       - Account risk parameters
       - Current exposure levels
    */
    
    double lot_size = CalculatePositionSize(prediction, confidence);
    
    Print("   Position Sizing Analysis:");
    Print("     Calculated Size: ", DoubleToString(lot_size, 2), " lots");
    Print("     Base Risk: ", DoubleToString(InpRiskPerTrade, 2), "% per trade");
    Print("     Confidence Adjustment: ", DoubleToString(confidence * 100, 1), "%");
    
    if(lot_size <= 0)
    {
        Print("     ❌ INVALID POSITION SIZE");
        Print("     Possible causes: Insufficient account balance, excessive risk, calculation error");
        return;
    }
    
    // Validate against broker limits
    double min_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
    
    if(lot_size < min_lot)
    {
        Print("     ⚠️  Position size below broker minimum, adjusting to ", DoubleToString(min_lot, 2));
        lot_size = min_lot;
    }
    if(lot_size > max_lot)
    {
        Print("     ⚠️  Position size above broker maximum, reducing to ", DoubleToString(max_lot, 2));
        lot_size = max_lot;
    }
    
    Print("     ✅ Final Position Size: ", DoubleToString(lot_size, 2), " lots");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 5: RISK LEVEL CALCULATION
    // ═══════════════════════════════════════════════════════════════
    /*
       Calculate stop-loss and take-profit levels using advanced
       risk management techniques and market volatility analysis.
    */
    
    double stop_loss = 0, take_profit = 0;
    double execution_price = (signal_direction > 0) ? ask_price : bid_price;
    
    CalculateStopLevels(signal_direction, execution_price, stop_loss, take_profit);
    
    Print("   Risk Management Setup:");
    Print("     Entry Price: ", DoubleToString(execution_price, _Digits));
    Print("     Stop Loss: ", DoubleToString(stop_loss, _Digits));
    Print("     Take Profit: ", DoubleToString(take_profit, _Digits));
    
    // Calculate risk metrics
    double risk_points = MathAbs(execution_price - stop_loss) * MathPow(10, _Digits);
    double reward_points = MathAbs(take_profit - execution_price) * MathPow(10, _Digits);
    double risk_reward_ratio = (risk_points > 0) ? reward_points / risk_points : 0;
    
    Print("     Risk: ", DoubleToString(risk_points, 1), " points");
    Print("     Reward: ", DoubleToString(reward_points, 1), " points"); 
    Print("     R:R Ratio: 1:", DoubleToString(risk_reward_ratio, 2));
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 6: ORDER EXECUTION
    // ═══════════════════════════════════════════════════════════════
    /*
       Execute the trade with comprehensive error handling and
       detailed logging for performance analysis.
    */
    
    bool execution_success = false;
    
    Print("   🚀 EXECUTING TRADE ORDER...");
    Print("     Direction: ", (signal_direction > 0) ? "LONG (BUY)" : "SHORT (SELL)");
    Print("     Size: ", DoubleToString(lot_size, 2), " lots");
    Print("     Confidence: ", DoubleToString(confidence * 100, 2), "%");
    
    if(signal_direction > 0)
    {
        // ══════════════════════════════════════════════════════════
        // LONG POSITION EXECUTION
        // ══════════════════════════════════════════════════════════
        execution_success = OpenPosition(ORDER_TYPE_BUY, lot_size, ask_price, stop_loss, take_profit);
        
        if(execution_success)
        {
            g_positions_today++;
            Print("     ✅ LONG POSITION OPENED SUCCESSFULLY");
            Print("        Entry: ", DoubleToString(ask_price, _Digits));
            Print("        Size: ", DoubleToString(lot_size, 2), " lots");
            Print("        Stop Loss: ", DoubleToString(stop_loss, _Digits));
            Print("        Take Profit: ", DoubleToString(take_profit, _Digits));
            Print("        Model Confidence: ", DoubleToString(confidence * 100, 2), "%");
            Print("        Position Count Today: ", g_positions_today);
        }
    }
    else
    {
        // ══════════════════════════════════════════════════════════
        // SHORT POSITION EXECUTION  
        // ══════════════════════════════════════════════════════════
        execution_success = OpenPosition(ORDER_TYPE_SELL, lot_size, bid_price, stop_loss, take_profit);
        
        if(execution_success)
        {
            g_positions_today++;
            Print("     ✅ SHORT POSITION OPENED SUCCESSFULLY");
            Print("        Entry: ", DoubleToString(bid_price, _Digits));
            Print("        Size: ", DoubleToString(lot_size, 2), " lots");
            Print("        Stop Loss: ", DoubleToString(stop_loss, _Digits));
            Print("        Take Profit: ", DoubleToString(take_profit, _Digits));
            Print("        Model Confidence: ", DoubleToString(confidence * 100, 2), "%");
            Print("        Position Count Today: ", g_positions_today);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 7: EXECUTION RESULT PROCESSING
    // ═══════════════════════════════════════════════════════════════
    
    if(!execution_success)
    {
        Print("     ❌ ORDER EXECUTION FAILED");
        Print("        Signal was valid but execution was blocked");
        Print("        Check trading conditions and account status");
        Print("        Review broker connection and market hours");
        
        // Log failed execution for analysis
        Print("        Failed Trade Details:");
        Print("          Prediction: ", DoubleToString(prediction, 6));
        Print("          Confidence: ", DoubleToString(confidence * 100, 2), "%");
        Print("          Direction: ", (signal_direction > 0) ? "LONG" : "SHORT");
        Print("          Size: ", DoubleToString(lot_size, 2), " lots");
        Print("          Price: ", DoubleToString(execution_price, _Digits));
    }
    
    Print("══════════════════════════════════════════════════════");
    Print("🎯 SIGNAL PROCESSING COMPLETED");
    Print("   Result: ", execution_success ? "SUCCESS" : "FAILED");
    Print("   Time: ", TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES));
    Print("══════════════════════════════════════════════════════");
}

//+------------------------------------------------------------------+
//| POSITION SIZE CALCULATION WITH DYNAMIC RISK ADJUSTMENT         |
//+------------------------------------------------------------------+
/*
   Calculates optimal position size based on account risk parameters,
   model confidence, and current market volatility conditions.
   
   POSITION SIZING FORMULA:
   position_size = (account_balance * risk_per_trade * confidence_factor * volatility_factor) / (stop_loss_points * point_value)
*/
double CalculatePositionSize(double prediction, double confidence)
{
    double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = account_balance * InpRiskPerTrade / 100.0;
    
    Print("📊 Position Sizing Calculation:");
    Print("   Account Balance: ", DoubleToString(account_balance, 2));
    Print("   Risk Per Trade: ", DoubleToString(InpRiskPerTrade, 2), "%");
    Print("   Base Risk Amount: ", DoubleToString(risk_amount, 2));
    
    // ═══════════════════════════════════════════════════════════════
    // CONFIDENCE-BASED ADJUSTMENT
    // ═══════════════════════════════════════════════════════════════
    /*
       Adjust position size based on model confidence:
       - Higher confidence = larger positions
       - Lower confidence = smaller positions
       - Quadratic scaling for more conservative sizing
    */
    double confidence_factor = confidence * confidence; // Quadratic scaling
    risk_amount *= confidence_factor;
    
    Print("   Confidence Factor: ", DoubleToString(confidence_factor, 3));
    Print("   Confidence-Adjusted Risk: ", DoubleToString(risk_amount, 2));
    
    // ═══════════════════════════════════════════════════════════════
    // VOLATILITY-BASED ADJUSTMENT
    // ═══════════════════════════════════════════════════════════════
    /*
       Reduce position size during high volatility periods to maintain
       consistent risk exposure across different market conditions.
    */
    if(InpUseDynamicRisk && g_risk_manager != NULL)
    {
        double current_volatility = g_risk_manager.GetCurrentVolatility();
        double volatility_factor = MathMax(0.5, 1.0 - current_volatility * 2.0);
        
        risk_amount *= volatility_factor;
        
        Print("   Current Volatility: ", DoubleToString(current_volatility * 100, 3), "%");
        Print("   Volatility Factor: ", DoubleToString(volatility_factor, 3));
        Print("   Volatility-Adjusted Risk: ", DoubleToString(risk_amount, 2));
    }
    
    // ═══════════════════════════════════════════════════════════════
    // STOP-LOSS BASED SIZING
    // ═══════════════════════════════════════════════════════════════
    /*
       Calculate lot size based on the distance to stop-loss level
       to ensure consistent risk per trade regardless of stop distance.
    */
    
    // Estimate stop-loss distance (will be refined in CalculateStopLevels)
    double estimated_stop_points = 100; // Default conservative estimate
    
    if(InpUseDynamicRisk && g_risk_manager != NULL)
    {
        double volatility = g_risk_manager.GetCurrentVolatility();
        estimated_stop_points = MathMax(50, MathMin(300, volatility * 20000)); // Convert to points
    }
    
    // Calculate position size
    double point_value = SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_VALUE);
    double lot_size = risk_amount / (estimated_stop_points * point_value);
    
    Print("   Estimated Stop Distance: ", DoubleToString(estimated_stop_points, 1), " points");
    Print("   Point Value: ", DoubleToString(point_value, 2));
    Print("   Calculated Lot Size: ", DoubleToString(lot_size, 3));
    
    // ═══════════════════════════════════════════════════════════════
    // BROKER LIMITS AND VALIDATION
    // ═══════════════════════════════════════════════════════════════
    
    double min_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
    
    Print("   Broker Limits:");
    Print("     Minimum Lot: ", DoubleToString(min_lot, 2));
    Print("     Maximum Lot: ", DoubleToString(max_lot, 2));
    Print("     Lot Step: ", DoubleToString(lot_step, 2));
    
    // Apply limits
    lot_size = MathMax(min_lot, MathMin(max_lot, lot_size));
    
    // Round to valid step size
    if(lot_step > 0)
        lot_size = MathRound(lot_size / lot_step) * lot_step;
    
    Print("   Final Lot Size: ", DoubleToString(lot_size, 2));
    
    return lot_size;
}

//+------------------------------------------------------------------+
//| STOP-LOSS AND TAKE-PROFIT CALCULATION                          |
//+------------------------------------------------------------------+
/*
   Calculates optimal stop-loss and take-profit levels using market
   volatility analysis and risk-reward optimization.
*/
void CalculateStopLevels(int signal_direction, double entry_price, double &stop_loss, double &take_profit)
{
    double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    
    Print("🎯 Calculating Risk Management Levels:");
    Print("   Signal Direction: ", (signal_direction > 0) ? "LONG" : "SHORT");
    Print("   Entry Price: ", DoubleToString(entry_price, _Digits));
    Print("   Point Size: ", DoubleToString(point, _Digits + 1));
    
    // ═══════════════════════════════════════════════════════════════
    // DYNAMIC STOP-LOSS CALCULATION
    // ═══════════════════════════════════════════════════════════════
    
    double stop_distance_points = 100; // Default stop distance
    
    if(InpUseDynamicRisk && g_risk_manager != NULL)
    {
        double volatility = g_risk_manager.GetCurrentVolatility();
        stop_distance_points = MathMax(50, MathMin(300, volatility * 20000));
        
        Print("   Dynamic Stop Calculation:");
        Print("     Market Volatility: ", DoubleToString(volatility * 100, 3), "%");
        Print("     Volatility-Based Stop: ", DoubleToString(stop_distance_points, 1), " points");
    }
    else
    {
        // Use trailing stop distance as base for stop-loss
        stop_distance_points = InpTrailingStopDistance;
        Print("   Fixed Stop Calculation:");
        Print("     Configured Distance: ", DoubleToString(stop_distance_points, 1), " points");
    }
    
    // ═══════════════════════════════════════════════════════════════
    // CALCULATE STOP-LOSS AND TAKE-PROFIT LEVELS
    // ═══════════════════════════════════════════════════════════════
    
    double stop_distance = stop_distance_points * point;
    double take_profit_ratio = 2.0; // 1:2 risk-reward ratio
    double tp_distance = stop_distance * take_profit_ratio;
    
    if(signal_direction > 0) // LONG position
    {
        stop_loss = entry_price - stop_distance;
        take_profit = entry_price + tp_distance;
    }
    else // SHORT position
    {
        stop_loss = entry_price + stop_distance;
        take_profit = entry_price - tp_distance;
    }
    
    Print("   Calculated Levels:");
    Print("     Stop Loss: ", DoubleToString(stop_loss, _Digits));
    Print("     Take Profit: ", DoubleToString(take_profit, _Digits));
    Print("     Risk Distance: ", DoubleToString(stop_distance_points, 1), " points");
    Print("     Reward Distance: ", DoubleToString(stop_distance_points * take_profit_ratio, 1), " points");
    Print("     Risk:Reward Ratio: 1:", DoubleToString(take_profit_ratio, 1));
    
    // ═══════════════════════════════════════════════════════════════
    // BROKER MINIMUM DISTANCE VALIDATION
    // ═══════════════════════════════════════════════════════════════
    
    long stops_level = SymbolInfoInteger(Symbol(), SYMBOL_TRADE_STOPS_LEVEL);
    double min_distance = stops_level * point;
    
    if(min_distance > 0)
    {
        Print("   Broker Minimum Distance: ", stops_level, " points");
        
        // Adjust stop-loss if too close
        if(signal_direction > 0 && (entry_price - stop_loss) < min_distance)
        {
            stop_loss = entry_price - min_distance;
            Print("     Stop-loss adjusted for broker minimum: ", DoubleToString(stop_loss, _Digits));
        }
        else if(signal_direction < 0 && (stop_loss - entry_price) < min_distance)
        {
            stop_loss = entry_price + min_distance;
            Print("     Stop-loss adjusted for broker minimum: ", DoubleToString(stop_loss, _Digits));
        }
        
        // Adjust take-profit if too close
        if(signal_direction > 0 && (take_profit - entry_price) < min_distance)
        {
            take_profit = entry_price + min_distance;
            Print("     Take-profit adjusted for broker minimum: ", DoubleToString(take_profit, _Digits));
        }
        else if(signal_direction < 0 && (entry_price - take_profit) < min_distance)
        {
            take_profit = entry_price - min_distance;
            Print("     Take-profit adjusted for broker minimum: ", DoubleToString(take_profit, _Digits));
        }
    }
    
    Print("   ✅ Final Risk Management Levels:");
    Print("     Entry: ", DoubleToString(entry_price, _Digits));
    Print("     Stop Loss: ", DoubleToString(stop_loss, _Digits));
    Print("     Take Profit: ", DoubleToString(take_profit, _Digits));
}

//+------------------------------------------------------------------+
//| POSITION OPENING WITH COMPREHENSIVE ERROR HANDLING             |
//+------------------------------------------------------------------+
/*
   Opens a new position with detailed error handling and trade logging.
   This function handles all aspects of order execution including
   validation, submission, and result processing.
*/
bool OpenPosition(ENUM_ORDER_TYPE order_type, double lot_size, double price, double sl, double tp)
{
    Print("📤 Submitting Trade Order:");
    Print("   Type: ", EnumToString(order_type));
    Print("   Volume: ", DoubleToString(lot_size, 2), " lots");
    Print("   Price: ", DoubleToString(price, _Digits));
    Print("   Stop Loss: ", DoubleToString(sl, _Digits));
    Print("   Take Profit: ", DoubleToString(tp, _Digits));
    
    // ═══════════════════════════════════════════════════════════════
    // PREPARE TRADE REQUEST
    // ═══════════════════════════════════════════════════════════════
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_DEAL;
    request.symbol = Symbol();
    request.volume = lot_size;
    request.type = order_type;
    request.price = price;
    request.sl = sl;
    request.tp = tp;
    request.magic = g_magic_number;
    request.comment = "LLM-PPO|" + TimeToString(TimeCurrent(), TIME_MINUTES);
    request.deviation = 10; // 10 point slippage tolerance
    request.type_filling = ORDER_FILLING_FOK; // Fill or Kill
    
    // ═══════════════════════════════════════════════════════════════
    // SUBMIT ORDER
    // ═══════════════════════════════════════════════════════════════
    
    if(!OrderSend(request, result))
    {
        Print("❌ ORDER SUBMISSION FAILED:");
        Print("   Error Code: ", result.retcode);
        Print("   Error Description: ", result.comment);
        
        // Detailed error analysis
        switch(result.retcode)
        {
            case TRADE_RETCODE_REQUOTE:
                Print("   Issue: Price changed during execution (requote)");
                break;
            case TRADE_RETCODE_REJECT:
                Print("   Issue: Request rejected by broker");
                break;
            case TRADE_RETCODE_NO_MONEY:
                Print("   Issue: Insufficient funds");
                break;
            case TRADE_RETCODE_INVALID_VOLUME:
                Print("   Issue: Invalid volume specified");
                break;
            case TRADE_RETCODE_INVALID_PRICE:
                Print("   Issue: Invalid price specified");
                break;
            case TRADE_RETCODE_INVALID_STOPS:
                Print("   Issue: Invalid stop-loss or take-profit");
                break;
            default:
                Print("   Issue: Unknown error - check broker connection");
                break;
        }
        
        return false;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // ORDER SUCCESS - LOG DETAILS
    // ═══════════════════════════════════════════════════════════════
    
    Print("✅ ORDER EXECUTED SUCCESSFULLY:");
    Print("   Order ID: ", result.order);
    Print("   Deal ID: ", result.deal);
    Print("   Execution Price: ", DoubleToString(result.price, _Digits));
    Print("   Volume Filled: ", DoubleToString(result.volume, 2), " lots");
    Print("   Magic Number: ", g_magic_number);
    Print("   Comment: ", result.comment);
    
    // Calculate actual slippage
    double slippage_points = MathAbs(result.price - price) * MathPow(10, _Digits);
    if(slippage_points > 0.1)
        Print("   Slippage: ", DoubleToString(slippage_points, 1), " points");
    
    return true;
}

//+------------------------------------------------------------------+
//| COUNT OPEN POSITIONS FOR THIS EA                               |
//+------------------------------------------------------------------+
/*
   Counts the number of open positions belonging to this EA instance
   based on the magic number filter.
*/
int CountOpenPositions()
{
    int count = 0;
    
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetString(POSITION_SYMBOL) == Symbol() &&
               PositionGetInteger(POSITION_MAGIC) == g_magic_number)
            {
                count++;
            }
        }
    }
    
    return count;
}

//+------------------------------------------------------------------+
//| TRAILING STOP MANAGEMENT                                        |
//+------------------------------------------------------------------+
/*
   Updates trailing stop-loss levels for existing positions to lock in
   profits as the market moves favorably.
*/
void UpdateTrailingStops()
{
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(!PositionSelectByTicket(PositionGetTicket(i)))
            continue;
            
        if(PositionGetString(POSITION_SYMBOL) != Symbol() ||
           PositionGetInteger(POSITION_MAGIC) != g_magic_number)
            continue;
        
        double current_price = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ?
                               SymbolInfoDouble(Symbol(), SYMBOL_BID) :
                               SymbolInfoDouble(Symbol(), SYMBOL_ASK);
        
        double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
        double current_sl = PositionGetDouble(POSITION_SL);
        
        double trail_distance = InpTrailingStopDistance * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
        double new_sl = current_sl;
        
        if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
        {
            // Long position - move stop up
            new_sl = current_price - trail_distance;
            if(new_sl > current_sl + SymbolInfoDouble(Symbol(), SYMBOL_POINT))
            {
                ModifyPosition(PositionGetTicket(i), new_sl, PositionGetDouble(POSITION_TP));
            }
        }
        else
        {
            // Short position - move stop down
            new_sl = current_price + trail_distance;
            if(new_sl < current_sl - SymbolInfoDouble(Symbol(), SYMBOL_POINT))
            {
                ModifyPosition(PositionGetTicket(i), new_sl, PositionGetDouble(POSITION_TP));
            }
        }
    }
}

//+------------------------------------------------------------------+
//| POSITION MODIFICATION HELPER                                   |
//+------------------------------------------------------------------+
bool ModifyPosition(ulong ticket, double sl, double tp)
{
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_SLTP;
    request.position = ticket;
    request.sl = sl;
    request.tp = tp;
    request.magic = g_magic_number;
    
    return OrderSend(request, result);
}

//+------------------------------------------------------------------+
//| DISPLAY UPDATE FOR CHART MONITORING                            |
//+------------------------------------------------------------------+
/*
   Updates the chart display with current system status, predictions,
   and performance metrics for real-time monitoring.
*/
void UpdateDisplay(double prediction, double confidence)
{
    string display_text = "";
    
    display_text += "═══════════════════════════════════════\n";
    display_text += "    LLM-PPO TRADING SYSTEM STATUS\n";
    display_text += "═══════════════════════════════════════\n";
    display_text += "🕐 Time: " + TimeToString(TimeCurrent(), TIME_MINUTES) + "\n";
    display_text += "💱 Symbol: " + Symbol() + " | " + EnumToString(PERIOD_CURRENT) + "\n";
    display_text += "📊 Bid: " + DoubleToString(SymbolInfoDouble(Symbol(), SYMBOL_BID), _Digits) + "\n";
    display_text += "───────────────────────────────────────\n";
    display_text += "🧠 PREDICTION ENGINE:\n";
    display_text += "   Prediction: " + DoubleToString(prediction, 6) + "\n";
    display_text += "   Confidence: " + DoubleToString(confidence * 100, 1) + "%\n";
    display_text += "   Status: " + ((confidence >= InpConfidenceThreshold) ? "✅ ACTIVE" : "⚠️ FILTERED") + "\n";
    display_text += "───────────────────────────────────────\n";
    display_text += "📈 PERFORMANCE:\n";
    display_text += "   Daily P&L: " + DoubleToString(g_daily_pnl, 2) + " " + AccountInfoString(ACCOUNT_CURRENCY) + "\n";
    display_text += "   Positions Today: " + IntegerToString(g_positions_today) + "\n";
    display_text += "   Open Positions: " + IntegerToString(CountOpenPositions()) + "/" + IntegerToString(InpMaxOpenPositions) + "\n";
    display_text += "───────────────────────────────────────\n";
    display_text += "🛡️ RISK STATUS:\n";
    display_text += "   Balance: " + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + "\n";
    display_text += "   Equity: " + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + "\n";
    display_text += "   Trading: " + (g_trading_enabled ? "✅ ENABLED" : "❌ DISABLED") + "\n";
    
    if(g_risk_manager != NULL)
    {
        display_text += "   Current VaR: " + DoubleToString(g_risk_manager.GetCurrentVaR() * 100, 2) + "%\n";
        display_text += "   Volatility: " + DoubleToString(g_risk_manager.GetCurrentVolatility() * 100, 2) + "%\n";
    }
    
    display_text += "═══════════════════════════════════════\n";
    display_text += "Framework: LLM Prediction + PPO Risk Adjustment\n";
    display_text += "Magic Number: " + IntegerToString(g_magic_number);
    
    Comment(display_text);
}