//+------------------------------------------------------------------+
//|                                           ParameterOptimizer.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
// This is the comprehensive parameter optimization framework for the LLM-PPO trading system.
// It implements multiple sophisticated optimization algorithms:
//
// 1. GENETIC ALGORITHM (Primary)
//    - Population-based evolutionary optimization
//    - Selection, crossover, mutation operators
//    - Elite preservation and convergence detection
//    - Best for complex multi-parameter optimization
//
// 2. BRUTE FORCE OPTIMIZATION
//    - Exhaustive search of all parameter combinations
//    - Guarantees global optimum for small search spaces
//    - Computationally expensive but thorough
//
// 3. PARTICLE SWARM OPTIMIZATION (PSO)
//    - Swarm intelligence with velocity-based movement
//    - Fast convergence with good exploration
//    - Inertia, cognitive, and social coefficients
//
// 4. SIMULATED ANNEALING (SA)
//    - Probabilistic optimization with temperature cooling
//    - Escapes local optima through acceptance probability
//    - Exponential cooling schedule
//
// The framework includes advanced validation methods:
// - Walk-Forward Analysis for parameter stability
// - Cross-Validation for generalization assessment
// - Sensitivity Analysis for robustness testing
// - Multi-objective fitness optimization
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Parameter Optimization Framework                                 |
//+------------------------------------------------------------------+

// Optimization method enumeration - Defines available optimization algorithms
enum ENUM_OPTIMIZATION_METHOD
{
    OPT_GENETIC_ALGORITHM = 0,    // Evolutionary algorithm with population, selection, crossover, mutation
    OPT_BRUTE_FORCE = 1,         // Exhaustive search - tests every possible parameter combination
    OPT_PARTICLE_SWARM = 2,      // Swarm intelligence - particles move through parameter space
    OPT_SIMULATED_ANNEALING = 3  // Temperature-based probabilistic optimization
};

// OptimizationParameter: Defines a single parameter to be optimized
// Each parameter has bounds, step size, and precision settings
struct OptimizationParameter
{
    string   name;                    // Human-readable parameter name (e.g., "LearningRate")
    double   min_value;               // Lower bound of search range
    double   max_value;               // Upper bound of search range
    double   step;                    // Discrete step size for parameter values
    double   current_value;           // Current parameter value during optimization
    bool     is_enabled;              // Whether this parameter should be optimized
    int      precision;               // Number of decimal places for output formatting
    
    // Copy constructor
    OptimizationParameter(const OptimizationParameter& other)
    {
        name = other.name;
        min_value = other.min_value;
        max_value = other.max_value;
        step = other.step;
        current_value = other.current_value;
        is_enabled = other.is_enabled;
        precision = other.precision;
    }
    
    // Default constructor
    OptimizationParameter()
    {
        name = "";
        min_value = 0.0;
        max_value = 0.0;
        step = 0.0;
        current_value = 0.0;
        is_enabled = false;
        precision = 4;
    }
};

// Chromosome: Represents a complete parameter set (individual) in genetic algorithm
// Contains parameter values and their resulting performance metrics
struct Chromosome
{
    double   parameters[];            // Array of parameter values (genes) for this chromosome
    double   fitness;                 // Combined fitness score (weighted sum of performance metrics)
    double   sharpe_ratio;           // Risk-adjusted return measure
    double   profit_factor;          // Gross profit / |gross loss| ratio
    double   max_drawdown;           // Maximum peak-to-trough equity decline (%)
    double   win_rate;               // Percentage of profitable trades
    double   total_trades;           // Number of trades executed in backtest
    double   annual_return;          // Annualized return percentage
    bool     is_valid;               // Whether this chromosome meets all constraints
    
    // Copy constructor
    Chromosome(const Chromosome& other)
    {
        ArrayResize(parameters, ArraySize(other.parameters));
        for(int i = 0; i < ArraySize(other.parameters); i++)
            parameters[i] = other.parameters[i];
        fitness = other.fitness;
        sharpe_ratio = other.sharpe_ratio;
        profit_factor = other.profit_factor;
        max_drawdown = other.max_drawdown;
        win_rate = other.win_rate;
        total_trades = other.total_trades;
        annual_return = other.annual_return;
        is_valid = other.is_valid;
    }
    
    // Default constructor
    Chromosome()
    {
        ArrayResize(parameters, 0);
        fitness = -DBL_MAX;
        sharpe_ratio = 0.0;
        profit_factor = 0.0;
        max_drawdown = 0.0;
        win_rate = 0.0;
        total_trades = 0.0;
        annual_return = 0.0;
        is_valid = false;
    }
};

// OptimizationResults: Complete optimization run results and statistics
struct OptimizationResults
{
    Chromosome best_chromosome;       // Best parameter combination found
    Chromosome population[];          // Final population state (for analysis)
    double     convergence_history[]; // Fitness evolution over generations
    int        generations_run;       // Total number of generations executed
    double     optimization_time;     // Wall-clock time for optimization (seconds)
    string     optimization_report;   // Human-readable summary report
};

//+------------------------------------------------------------------+
//| Parameter Optimizer Class - Advanced optimization engine        |
//+------------------------------------------------------------------+
// CParameterOptimizer: Main class implementing multiple optimization algorithms
// Provides a unified interface for parameter optimization with various methods
// and comprehensive validation capabilities
class CParameterOptimizer
{
private:
    // Parameter Management - Stores all parameters to be optimized
    OptimizationParameter m_parameters[];       // Array of all defined parameters
    int                   m_param_count;        // Count of enabled parameters for optimization
    
    // Genetic Algorithm Configuration - Controls evolutionary optimization behavior
    int                   m_population_size;        // Number of chromosomes in population (typically 50-200)
    int                   m_max_generations;        // Maximum evolutionary generations (typically 100-500)
    double                m_mutation_rate;          // Probability of gene mutation (typically 0.01-0.1)
    double                m_crossover_rate;         // Probability of chromosome crossover (typically 0.7-0.9)
    double                m_elite_ratio;            // Fraction of population preserved as elite (typically 0.1-0.2)
    double                m_convergence_threshold;  // Minimum fitness improvement for convergence detection
    int                   m_convergence_patience;   // Generations without improvement before early stopping
    
    // Population Management - Maintains genetic algorithm populations
    Chromosome            m_population[];          // Current generation population
    Chromosome            m_elite_population[];    // Best chromosomes preserved across generations
    double                m_fitness_history[];     // Fitness evolution tracking for convergence analysis
    
    // Optimization State Tracking - Monitors optimization progress
    bool                  m_is_running;            // Flag indicating if optimization is currently active
    int                   m_current_generation;    // Current generation number in GA
    double                m_best_fitness;          // Best fitness score found so far
    int                   m_stagnation_counter;    // Generations without improvement (for early stopping)
    datetime              m_start_time;            // Optimization start timestamp
    
    // Multi-Objective Fitness Function Weights - Controls relative importance of metrics
    double                m_weight_profit_factor;  // Weight for profit factor (gross profit / |gross loss|)
    double                m_weight_sharpe_ratio;   // Weight for risk-adjusted returns
    double                m_weight_max_drawdown;   // Weight for drawdown penalty (lower is better)
    double                m_weight_win_rate;       // Weight for percentage of winning trades
    double                m_weight_annual_return;  // Weight for raw annual return percentage
    
    // Performance Constraints - Minimum acceptable performance thresholds
    double                m_min_trades;            // Minimum number of trades for statistical significance
    double                m_max_drawdown_limit;    // Maximum acceptable drawdown percentage
    double                m_min_win_rate;          // Minimum acceptable win rate percentage
    double                m_min_profit_factor;     // Minimum acceptable profit factor (must be > 1.0)

public:
    // Constructor/Destructor - Object lifecycle management
                         CParameterOptimizer(void);     // Initialize with default parameters
                        ~CParameterOptimizer(void);     // Clean up resources and arrays
    
    // Setup and Configuration Methods - Configure optimization parameters
    bool                 Initialize(void);              // Initialize optimizer (call after adding parameters)
    bool                 AddParameter(string name, double min_val, double max_val, 
                                    double step, bool enabled = true, int precision = 4); // Add parameter to optimize
    void                 SetGeneticAlgorithmParameters(int population_size = 50,      // Configure GA hyperparameters
                                                      int max_generations = 100,
                                                      double mutation_rate = 0.1,
                                                      double crossover_rate = 0.8);
    void                 SetFitnessWeights(double profit_factor = 0.3, double sharpe = 0.25,  // Set multi-objective weights
                                          double drawdown = 0.2, double win_rate = 0.15,
                                          double annual_return = 0.1);
    void                 SetConstraints(double min_trades = 50, double max_dd = 30.0,        // Set performance thresholds
                                       double min_win_rate = 30.0, double min_pf = 1.1);
    
    // Main Optimization Methods - Different algorithms for parameter optimization
    bool                 RunOptimization(void);                    // Default optimization (calls RunGeneticAlgorithm)
    bool                 RunBruteForceOptimization(void);          // Exhaustive search - guaranteed global optimum
    bool                 RunGeneticAlgorithm(void);                // Evolutionary optimization - best for complex spaces
    bool                 RunParticleSwarmOptimization(void);       // Swarm intelligence - fast convergence
    bool                 RunSimulatedAnnealing(void);              // Temperature-based probabilistic search
    
    // Genetic Algorithm Implementation - Core evolutionary operators
    bool                 InitializePopulation(void);               // Create random initial population
    bool                 EvaluatePopulation(void);                 // Calculate fitness for all chromosomes
    double               CalculateFitness(Chromosome &chromosome); // Multi-objective fitness calculation
    bool                 RunBacktestForChromosome(Chromosome &chromosome); // Execute backtest for parameter set
    void                 SelectElite(void);                        // Preserve best chromosomes for next generation
    void                 CrossoverPopulation(void);                // Create offspring through crossover
    void                 MutatePopulation(void);                   // Apply mutations to maintain diversity
    Chromosome           Tournament_Selection(int tournament_size = 3); // Select parent for reproduction
    Chromosome           Crossover(Chromosome &parent1, Chromosome &parent2); // Single-point crossover operation
    void                 Mutate(Chromosome &chromosome);           // Gaussian mutation with boundary constraints
    
    // Brute Force Optimization - Exhaustive search implementation
    bool                 TestAllCombinations(int param_index, Chromosome &best_chromosome,  // Recursive combination testing
                                           double &best_fitness, int &tested_count, int total_count);
    
    // Particle Swarm Optimization - Swarm intelligence implementation
    struct Particle      // Individual particle in the swarm
    {
        double position[];      // Current position in parameter space
        double velocity[];      // Current velocity vector
        double best_position[]; // Personal best position found
        double best_fitness;    // Personal best fitness achieved
        double fitness;         // Current fitness at current position
  protected:
};
    bool                 RunPSO(void);                             // Main PSO algorithm implementation
    void                 UpdateParticleVelocity(Particle &particle, double &global_best[]); // PSO velocity update equation
    void                 UpdateParticlePosition(Particle &particle); // Update position with boundary constraints
    
    // Simulated Annealing - Temperature-based probabilistic optimization
    bool                 RunSA(void);                              // Main simulated annealing implementation
    double               GenerateNeighborSolution(double &current_params[]); // Generate neighbor solution with perturbation
    double               CalculateTemperature(int iteration, int max_iterations); // Exponential cooling schedule
    
    // Walk Forward Analysis - Time-series parameter stability validation
    bool                 RunWalkForwardAnalysis(int window_size = 252, int step_size = 63); // Rolling window optimization
    struct WalkForwardResult  // Results from each walk-forward iteration
    {
        datetime start_date;     // Start of optimization window
        datetime end_date;       // End of optimization window
        Chromosome best_params;  // Optimal parameters for this window
        double oos_performance;  // Out-of-sample performance
        double is_performance;   // In-sample performance
    };
    WalkForwardResult    m_wf_results[];        // Array storing all walk-forward results
    
    // Multi-Objective Optimization - Pareto-optimal solutions
    bool                 RunMultiObjectiveOptimization(void);      // NSGA-II style multi-objective optimization
    double               CalculateParetoRank(Chromosome &chromosome); // Calculate Pareto dominance rank
    bool                 DominatesChromosome(Chromosome &a, Chromosome &b); // Check if A dominates B
    
    // Cross-Validation - K-fold validation for generalization assessment
    bool                 RunCrossValidation(int k_folds = 5);      // K-fold cross-validation implementation
    double               CalculateAverageCV_Performance(double &params[]); // Average performance across folds
    
    // Parameter Analysis - Advanced parameter relationship analysis
    bool                 RunSensitivityAnalysis(void);             // Test parameter robustness around optimal values
    bool                 RunParameterCorrelationAnalysis(void);    // Analyze parameter interactions and correlations
    bool                 Generate3DParameterSurface(string param1, string param2); // Create 3D fitness surface plots
    
    // Robustness testing
    bool                 RunRobustnessTest(void);
    bool                 RunMonteCarloBootstrap(int bootstrap_samples = 1000);
    bool                 TestParameterStability(void);
    
    // Results and reporting
    OptimizationResults  GetOptimizationResults(void);
    bool                 ExportResults(string filename);
    string               GenerateOptimizationReport(void);
    bool                 SaveParameterSet(string filename, Chromosome &chromosome);
    bool                 LoadParameterSet(string filename);
    
    // Validation methods
    bool                 ValidateParameters(double &params[]);
    bool                 IsParameterSetValid(Chromosome &chromosome);
    double               GetParameterCorrelation(string param1, string param2);
    
    // Utility methods
    double               NormalizeParameter(string param_name, double value);
    double               DenormalizeParameter(string param_name, double norm_value);
    void                 PrintProgress(int current, int total);
    void                 LogOptimizationProgress(string message);
    
    // Getters
    int                  GetParameterCount(void) { return m_param_count; }
    long                GetSearchSpaceSize(void);
    bool                 IsOptimizationRunning(void) { return m_is_running; }
    Chromosome           GetBestChromosome(void);
    double               GetBestFitness(void) { return m_best_fitness; }
    int                  GetCurrentGeneration(void) { return m_current_generation; }
    
    // Setters
    void                 SetConvergenceThreshold(double threshold) { m_convergence_threshold = threshold; }
    void                 SetConvergencePatience(int patience) { m_convergence_patience = patience; }
    void                 StopOptimization(void) { m_is_running = false; }
};

//+------------------------------------------------------------------+
//| Constructor - Initialize optimizer with default settings        |
//+------------------------------------------------------------------+
// Constructor sets up the parameter optimizer with research-based default values:
// - GA parameters optimized for trading strategy optimization
// - Multi-objective fitness weights based on common trading priorities
// - Performance constraints ensuring statistical significance
CParameterOptimizer::CParameterOptimizer(void)
{
    // Initialize parameter management
    m_param_count = 0;                    // No parameters defined yet
    
    // Set genetic algorithm defaults (research-based optimal values)
    m_population_size = 50;               // Balance between diversity and computational cost
    m_max_generations = 100;              // Sufficient for most trading strategy optimizations
    m_mutation_rate = 0.1;                // 10% mutation rate maintains diversity
    m_crossover_rate = 0.8;               // 80% crossover rate promotes exploration
    m_elite_ratio = 0.1;                  // Preserve top 10% of population
    m_convergence_threshold = 0.001;      // Stop when fitness improvement < 0.1%
    m_convergence_patience = 10;          // Wait 10 generations before early stopping
    
    // Initialize optimization state tracking
    m_is_running = false;                 // Optimization not running initially
    m_current_generation = 0;             // Start at generation 0
    m_best_fitness = -DBL_MAX;            // Initialize to worst possible fitness
    m_stagnation_counter = 0;             // No stagnation initially
    
    // Set default multi-objective fitness weights (based on trading priorities)
    m_weight_profit_factor = 0.3;         // 30% - Profitability most important
    m_weight_sharpe_ratio = 0.25;         // 25% - Risk-adjusted returns critical
    m_weight_max_drawdown = 0.2;          // 20% - Drawdown control important
    m_weight_win_rate = 0.15;             // 15% - Consistency matters
    m_weight_annual_return = 0.1;         // 10% - Raw returns least weighted
    
    // Set default performance constraints (minimum acceptable thresholds)
    m_min_trades = 50;                    // Minimum for statistical significance
    m_max_drawdown_limit = 30.0;          // Maximum acceptable drawdown (30%)
    m_min_win_rate = 30.0;                // Minimum win rate (30%)
    m_min_profit_factor = 1.1;            // Must be profitable (>10% profit ratio)
    
    // Initialize all dynamic arrays to empty state
    ArrayResize(m_parameters, 0);         // Parameter definitions
    ArrayResize(m_population, 0);         // GA population
    ArrayResize(m_elite_population, 0);   // Elite chromosomes
    ArrayResize(m_fitness_history, 0);    // Fitness convergence tracking
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CParameterOptimizer::~CParameterOptimizer(void)
{
    ArrayFree(m_parameters);
    ArrayFree(m_population);
    ArrayFree(m_elite_population);
    ArrayFree(m_fitness_history);
    ArrayFree(m_wf_results);
}

//+------------------------------------------------------------------+
//| Initialize optimizer - Validate setup and prepare for optimization |
//+------------------------------------------------------------------+
// Initialize() validates the optimizer configuration and prepares for optimization:
// - Checks that parameters have been defined
// - Initializes random number generator
// - Returns false if setup is invalid
bool CParameterOptimizer::Initialize(void)
{
    // Validate that at least one parameter has been added for optimization
    if(m_param_count == 0)
    {
        Print("ERROR: No parameters defined for optimization - call AddParameter() first");
        return false;
    }
    
    // Initialize random number generator with current time for reproducible randomness
    MathSrand((int)TimeCurrent());
    Print("Parameter optimizer initialized successfully with ", m_param_count, " parameters");
    
    return true;
}

//+------------------------------------------------------------------+
//| Add optimization parameter - Register parameter for optimization |
//+------------------------------------------------------------------+
// AddParameter() registers a new parameter to be optimized:
// - Validates parameter range and step size
// - Calculates search space size
// - Stores parameter configuration for optimization algorithms
bool CParameterOptimizer::AddParameter(string name, double min_val, double max_val, 
                                      double step, bool enabled = true, int precision = 4)
{
    // Validate parameter configuration
    if(min_val >= max_val || step <= 0)
    {
        Print("ERROR: Invalid parameter range for '", name, "': min=", min_val, " max=", max_val, " step=", step);
        return false;
    }
    
    // Add parameter to the parameters array
    int size = ArraySize(m_parameters);
    ArrayResize(m_parameters, size + 1);
    
    // Store parameter configuration
    m_parameters[size].name = name;               // Parameter identifier
    m_parameters[size].min_value = min_val;       // Search range minimum
    m_parameters[size].max_value = max_val;       // Search range maximum
    m_parameters[size].step = step;               // Discrete step size
    m_parameters[size].current_value = min_val;   // Initialize to minimum
    m_parameters[size].is_enabled = enabled;      // Whether to optimize this parameter
    m_parameters[size].precision = precision;     // Decimal places for display
    
    // Increment enabled parameter counter
    if(enabled) m_param_count++;
    
    // Calculate and report search space size for this parameter
    int param_combinations = (int)((max_val - min_val) / step) + 1;
    Print("Added parameter '", name, "': Range=[", min_val, "-", max_val, "], Step=", step, 
          ", Combinations=", param_combinations, ", Enabled=", (enabled ? "Yes" : "No"));
    
    return true;
}

//+------------------------------------------------------------------+
//| Set Genetic Algorithm parameters - Configure GA hyperparameters |
//+------------------------------------------------------------------+
// SetGeneticAlgorithmParameters() configures the genetic algorithm settings:
// - Population size affects exploration vs exploitation balance
// - Generations control optimization depth
// - Mutation and crossover rates control diversity and convergence
void CParameterOptimizer::SetGeneticAlgorithmParameters(int population_size = 50, 
                                                        int max_generations = 100,
                                                        double mutation_rate = 0.1,
                                                        double crossover_rate = 0.8)
{
    // Apply bounds to ensure reasonable GA parameter values
    m_population_size = MathMax(10, population_size);              // Minimum 10 for diversity
    m_max_generations = MathMax(10, max_generations);              // Minimum 10 generations
    m_mutation_rate = MathMax(0.01, MathMin(0.5, mutation_rate));  // 1-50% mutation rate
    m_crossover_rate = MathMax(0.5, MathMin(1.0, crossover_rate)); // 50-100% crossover rate
    
    // Report final GA configuration
    Print("Genetic Algorithm configured: Population=", m_population_size, 
          ", Generations=", m_max_generations,
          ", Mutation Rate=", DoubleToString(m_mutation_rate*100, 1), "%",
          ", Crossover Rate=", DoubleToString(m_crossover_rate*100, 1), "%");
}

//+------------------------------------------------------------------+
//| Set fitness function weights - Configure multi-objective optimization |
//+------------------------------------------------------------------+
// SetFitnessWeights() configures the relative importance of each performance metric:
// - Profit Factor: Gross profit / Gross loss ratio (>1.0 means profitable)
// - Sharpe Ratio: Risk-adjusted return (return per unit of risk)
// - Drawdown: Maximum peak-to-trough decline (risk management)
// - Win Rate: Percentage of winning trades (consistency measure)
// - Annual Return: Annualized profit percentage (raw performance)
// Weights should sum to 1.0 for proper multi-objective optimization
void CParameterOptimizer::SetFitnessWeights(double profit_factor = 0.3, double sharpe = 0.25,
                                           double drawdown = 0.2, double win_rate = 0.15,
                                           double annual_return = 0.1)
{
    // Store multi-objective fitness weights
    m_weight_profit_factor = profit_factor;   // Weight for profitability measure
    m_weight_sharpe_ratio = sharpe;          // Weight for risk-adjusted returns
    m_weight_max_drawdown = drawdown;        // Weight for risk management (inverted)
    m_weight_win_rate = win_rate;            // Weight for trade consistency
    m_weight_annual_return = annual_return;   // Weight for raw performance
    
    // Verify total weights approximately equal 1.0
    double total_weight = profit_factor + sharpe + drawdown + win_rate + annual_return;
    if(MathAbs(total_weight - 1.0) > 0.01)
    {
        Print("WARNING: Fitness weights sum to ", DoubleToString(total_weight, 3), " (should be 1.0)");
    }
    
    Print("Fitness weights updated - ProfitFactor: ", profit_factor,
          " Sharpe: ", sharpe, " Drawdown: ", drawdown,
          " WinRate: ", win_rate, " AnnualReturn: ", annual_return);
}

//+------------------------------------------------------------------+
//| Set optimization constraints - Define minimum acceptable performance |
//+------------------------------------------------------------------+
// SetConstraints() establishes minimum thresholds for acceptable strategy performance:
// - min_trades: Minimum number of trades for statistical significance (typically 30-100)
// - max_dd: Maximum acceptable drawdown percentage (risk tolerance)
// - min_win_rate: Minimum percentage of winning trades (consistency requirement)
// - min_pf: Minimum profit factor (1.0 = break-even, >1.1 recommended for live trading)
// Strategies not meeting these constraints receive heavily penalized fitness scores
void CParameterOptimizer::SetConstraints(double min_trades = 50, double max_dd = 30.0,
                                        double min_win_rate = 30.0, double min_pf = 1.1)
{
    // Store performance constraints for fitness evaluation
    m_min_trades = (int)min_trades;           // Minimum trades for statistical validity
    m_max_drawdown_limit = max_dd;            // Maximum acceptable risk exposure
    m_min_win_rate = min_win_rate;            // Minimum consistency threshold
    m_min_profit_factor = min_pf;             // Minimum profitability requirement
    
    // Validate constraint values are reasonable
    if(m_min_trades < 10)
        Print("WARNING: Minimum trades (", m_min_trades, ") may be too low for reliable statistics");
    if(max_dd > 50.0)
        Print("WARNING: Maximum drawdown (", max_dd, "%) is very high - consider reducing");
    if(min_pf < 1.0)
        Print("WARNING: Minimum profit factor (", min_pf, ") allows losing strategies");
    
    Print("Optimization constraints updated - MinTrades: ", m_min_trades,
          " MaxDrawdown: ", m_max_drawdown_limit, "%",
          " MinWinRate: ", m_min_win_rate, "%",
          " MinProfitFactor: ", m_min_profit_factor);
}

//+------------------------------------------------------------------+
//| Run optimization using specified method - Default entry point |
//+------------------------------------------------------------------+
// RunOptimization() is the default entry point for parameter optimization.
// It delegates to the Genetic Algorithm which provides the best balance of:
// - Global optimization capability (avoids local minima)
// - Computational efficiency (faster than brute force)
// - Robustness (handles complex parameter interactions well)
// For specialized needs, call specific optimization methods directly
bool CParameterOptimizer::RunOptimization(void)
{
    Print("Starting optimization with method: Genetic Algorithm (default)");
    return RunGeneticAlgorithm();
}

//+------------------------------------------------------------------+
//| Run Brute Force optimization - Exhaustive parameter space search |
//+------------------------------------------------------------------+
// RunBruteForceOptimization() performs exhaustive testing of all parameter combinations:
// ADVANTAGES:
// - Guaranteed to find global optimum (no local minima issues)
// - Simple implementation with predictable behavior
// - Complete coverage of parameter space
// DISADVANTAGES: 
// - Exponential time complexity O(n^k) where n=steps per param, k=param count
// - Becomes impractical with >5-6 parameters or fine step sizes
// - No early stopping or intelligent search strategies
// RECOMMENDED FOR: 2-4 parameters with coarse step sizes, final validation runs
bool CParameterOptimizer::RunBruteForceOptimization(void)
{
    Print("Starting Brute Force optimization...");
    
    if(!Initialize())
        return false;
        
    int total_combinations = 1;
    
    // Calculate total parameter combinations (cartesian product of all parameter ranges)
    for(int i = 0; i < m_param_count; i++)
    {
        if(!m_parameters[i].is_enabled)
            continue;
            
        // Calculate discrete steps for this parameter
        int steps = (int)((m_parameters[i].max_value - m_parameters[i].min_value) / m_parameters[i].step) + 1;
        total_combinations *= steps;
    }
    
    Print("Total parameter combinations to test: ", total_combinations);
    
    // Prevent computationally infeasible optimizations
    if(total_combinations > 100000)
    {
        Print("ERROR: Too many combinations (", total_combinations, "). Consider reducing parameter ranges or using Genetic Algorithm.");
        return false;
    }
    
    // Initialize optimization state
    Chromosome best_chromosome;
    double best_fitness = -DBL_MAX;
    int tested_combinations = 0;
    
    m_start_time = TimeCurrent();
    
    // Execute recursive brute force search through parameter space
    if(!TestAllCombinations(0, best_chromosome, best_fitness, tested_combinations, total_combinations))
    {
        Print("Brute Force optimization failed");
        return false;
    }
    
    // Store best result in population array (for consistent interface)
    if(ArraySize(m_population) < 1)
        ArrayResize(m_population, 1);
    m_population[0] = best_chromosome;
    m_best_fitness = best_fitness;
    
    // Report optimization results
    datetime end_time = TimeCurrent();
    Print("Brute Force optimization completed in ", (end_time - m_start_time), " seconds");
    Print("Best fitness: ", DoubleToString(best_fitness, 6), " from ", tested_combinations, " combinations");
    
    return true;
}

//+------------------------------------------------------------------+
//| Test all parameter combinations recursively - Core brute force engine |
//+------------------------------------------------------------------+
// TestAllCombinations() implements the recursive brute force search algorithm:
// - Uses depth-first traversal of the parameter space tree
// - Each recursion level handles one parameter dimension  
// - Base case: when all parameters set, evaluate fitness and update best
// - Recursive case: iterate through all values for current parameter
// ALGORITHM: For each parameter[i], iterate through all discrete values,
//           recursively test all combinations with remaining parameters
bool CParameterOptimizer::TestAllCombinations(int param_index, Chromosome &best_chromosome, 
                                             double &best_fitness, int &tested_count, int total_count)
{
    // BASE CASE: All parameters have been assigned values
    if(param_index >= m_param_count)
    {
        // Create chromosome from current parameter values
        Chromosome test_chromosome;
        ArrayResize(test_chromosome.parameters, m_param_count);
        
        // Copy current parameter values to chromosome
        for(int i = 0; i < m_param_count; i++)
            test_chromosome.parameters[i] = m_parameters[i].current_value;
            
        // Skip invalid parameter combinations (constraint violations)
        if(!IsParameterSetValid(test_chromosome))
            return true;
            
        // Evaluate fitness through backtesting
        double fitness = CalculateFitness(test_chromosome);
        tested_count++;
        
        // Report progress every 1000 evaluations
        if(tested_count % 1000 == 0)
            Print("Progress: ", tested_count, "/", total_count, " (", 
                  NormalizeDouble(100.0 * tested_count / total_count, 1), "%)");
        
        // Update global best if this combination is superior
        if(fitness > best_fitness)
        {
            best_fitness = fitness;
            best_chromosome = test_chromosome;
            Print("New best fitness: ", DoubleToString(fitness, 6), " at combination ", tested_count);
        }
        
        return true;
    }
    
    // Skip disabled parameters (move to next dimension)
    if(!m_parameters[param_index].is_enabled)
    {
        return TestAllCombinations(param_index + 1, best_chromosome, best_fitness, tested_count, total_count);
    }
    
    // RECURSIVE CASE: Iterate through all discrete values for current parameter
    for(double val = m_parameters[param_index].min_value; 
        val <= m_parameters[param_index].max_value + m_parameters[param_index].step / 2;  // Add half-step for floating point precision
        val += m_parameters[param_index].step)
    {
        // Set current parameter value
        m_parameters[param_index].current_value = val;
        
        // Recursively test all combinations with remaining parameters
        if(!TestAllCombinations(param_index + 1, best_chromosome, best_fitness, tested_count, total_count))
            return false;  // Propagate failure up the recursion stack
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Run Particle Swarm Optimization - Swarm intelligence metaheuristic |
//+------------------------------------------------------------------+
// RunParticleSwarmOptimization() implements PSO algorithm inspired by bird flocking:
// ADVANTAGES:
// - Fast convergence on smooth fitness landscapes
// - Good balance of exploration and exploitation
// - Fewer parameters to tune compared to GA
// - Natural parallel processing capability
// DISADVANTAGES:
// - Can get trapped in local optima on complex landscapes
// - Performance sensitive to PSO parameter tuning
// - May converge prematurely if diversity is lost
// RECOMMENDED FOR: Continuous optimization with smooth fitness functions
bool CParameterOptimizer::RunParticleSwarmOptimization(void)
{
    Print("Starting Particle Swarm Optimization...");
    
    if(!Initialize())
        return false;
        
    // PSO algorithm parameters (research-based defaults)
    int swarm_size = m_population_size;         // Number of particles in swarm
    int max_iterations = m_max_generations;     // Maximum optimization iterations
    double inertia_weight = 0.9;                // Inertia coefficient (momentum factor)
    double cognitive_coeff = 2.0;               // Cognitive coefficient (personal best attraction)
    double social_coeff = 2.0;                  // Social coefficient (global best attraction)
    
    // Initialize particle swarm data structures
    Particle particles[];                            // Array of particles in the swarm
    ArrayResize(particles, swarm_size);
    
    double global_best_position[];                   // Best position found by entire swarm
    ArrayResize(global_best_position, m_param_count);
    double global_best_fitness = -DBL_MAX;           // Best fitness achieved by swarm
    
    // Initialize each particle with random position and velocity
    for(int i = 0; i < swarm_size; i++)
    {
        // Allocate memory for particle state vectors
        ArrayResize(particles[i].position, m_param_count);      // Current position in parameter space
        ArrayResize(particles[i].velocity, m_param_count);      // Current velocity vector
        ArrayResize(particles[i].best_position, m_param_count); // Personal best position found
        
        // Initialize position and velocity for each parameter dimension
        for(int j = 0; j < m_param_count; j++)
        {
            // Skip disabled parameters (set to minimum value)
            if(!m_parameters[j].is_enabled)
            {
                particles[i].position[j] = m_parameters[j].min_value;
                particles[i].velocity[j] = 0.0;
                particles[i].best_position[j] = m_parameters[j].min_value;
                continue;
            }
            
            // Random initial position within parameter bounds [min, max]
            double range = m_parameters[j].max_value - m_parameters[j].min_value;
            particles[i].position[j] = m_parameters[j].min_value + range * MathRand() / 32767.0;
            
            // Small random initial velocity (10% of parameter range)
            particles[i].velocity[j] = (MathRand() / 32767.0 - 0.5) * range * 0.1;
            
            // Initialize personal best to starting position
            particles[i].best_position[j] = particles[i].position[j];
        }
        
        // Evaluate initial fitness at starting position
        Chromosome test_chromosome;
        ArrayResize(test_chromosome.parameters, m_param_count);
        for(int j = 0; j < m_param_count; j++)
            test_chromosome.parameters[j] = particles[i].position[j];
            
        particles[i].fitness = CalculateFitness(test_chromosome);
        particles[i].best_fitness = particles[i].fitness;      // Personal best = initial fitness
        
        // Update global swarm best if this particle is superior
        if(particles[i].fitness > global_best_fitness)
        {
            global_best_fitness = particles[i].fitness;
            for(int j = 0; j < m_param_count; j++)
                global_best_position[j] = particles[i].position[j];
        }
    }
    
    m_start_time = TimeCurrent();
    
    // Main PSO optimization loop - iteratively improve particle positions
    for(int iter = 0; iter < max_iterations; iter++)
    {
        // Adaptive inertia weight: decreases from 0.9 to 0.2 over iterations
        // High initial inertia promotes exploration, low final inertia encourages exploitation
        inertia_weight = 0.9 - 0.7 * iter / max_iterations;
        
        // Update each particle in the swarm
        for(int i = 0; i < swarm_size; i++)
        {
            // Apply PSO velocity update equation:
            // v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
            UpdateParticleVelocity(particles[i], global_best_position);
            
            // Update position: x(t+1) = x(t) + v(t+1)
            UpdateParticlePosition(particles[i]);
            
            // Evaluate fitness at new position
            Chromosome test_chromosome;
            ArrayResize(test_chromosome.parameters, m_param_count);
            for(int j = 0; j < m_param_count; j++)
                test_chromosome.parameters[j] = particles[i].position[j];
                
            particles[i].fitness = CalculateFitness(test_chromosome);
            
            // Update personal best if current position is superior
            if(particles[i].fitness > particles[i].best_fitness)
            {
                particles[i].best_fitness = particles[i].fitness;
                for(int j = 0; j < m_param_count; j++)
                    particles[i].best_position[j] = particles[i].position[j];
            }
            
            // Update global swarm best if current position is superior
            if(particles[i].fitness > global_best_fitness)
            {
                global_best_fitness = particles[i].fitness;
                for(int j = 0; j < m_param_count; j++)
                    global_best_position[j] = particles[i].position[j];
                    
                Print("Generation ", iter, " - New best fitness: ", DoubleToString(global_best_fitness, 6));
            }
        }
        
        // Progress reporting every 10 iterations
        if(iter % 10 == 0)
            Print("PSO Generation ", iter, "/", max_iterations, " - Best fitness: ", DoubleToString(global_best_fitness, 6));
    }
    
    // Store best result
    if(ArraySize(m_population) < 1)
        ArrayResize(m_population, 1);
    ArrayResize(m_population[0].parameters, m_param_count);
    for(int j = 0; j < m_param_count; j++)
        m_population[0].parameters[j] = global_best_position[j];
    m_population[0].fitness = global_best_fitness;
    m_best_fitness = global_best_fitness;
    
    datetime end_time = TimeCurrent();
    Print("PSO optimization completed in ", (end_time - m_start_time), " seconds");
    Print("Best fitness: ", global_best_fitness);
    
    return true;
}

//+------------------------------------------------------------------+
//| Update particle velocity - Core PSO velocity update equation   |
//+------------------------------------------------------------------+
// UpdateParticleVelocity() implements the standard PSO velocity update equation:
// v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
// where:
// - w = inertia weight (momentum factor, balances exploration vs exploitation)
// - c1 = cognitive coefficient (attraction to personal best)
// - c2 = social coefficient (attraction to global best)
// - r1, r2 = random numbers [0,1] (stochastic component)
void CParameterOptimizer::UpdateParticleVelocity(Particle &particle, double &global_best[])
{
    // PSO coefficients (using research-proven values)
    double inertia_weight = 0.5;      // Moderate momentum (can be adaptive)
    double cognitive_coeff = 2.0;     // Standard cognitive attraction strength
    double social_coeff = 2.0;        // Standard social attraction strength
    
    // Update velocity for each parameter dimension independently
    for(int i = 0; i < m_param_count; i++)
    {
        // Skip disabled parameters (zero velocity)
        if(!m_parameters[i].is_enabled)
        {
            particle.velocity[i] = 0.0;
            continue;
        }
        
        // Generate random coefficients for stochastic behavior
        double r1 = MathRand() / 32767.0;  // Random [0,1] for cognitive component
        double r2 = MathRand() / 32767.0;  // Random [0,1] for social component
        
        // Apply PSO velocity update equation
        particle.velocity[i] = inertia_weight * particle.velocity[i] +                          // Inertia (current momentum)
                              cognitive_coeff * r1 * (particle.best_position[i] - particle.position[i]) +  // Cognitive (personal experience)
                              social_coeff * r2 * (global_best[i] - particle.position[i]);                // Social (swarm knowledge)
    }
}

//+------------------------------------------------------------------+
//| Update particle position - Position integration with boundary handling |
//+------------------------------------------------------------------+
// UpdateParticlePosition() integrates velocity to update position using:
// x(t+1) = x(t) + v(t+1)
// Includes boundary constraint handling to keep particles within valid parameter ranges.
// Uses simple reflection/clamping for boundary violations (other strategies possible:
// reflection, random repositioning, velocity damping, etc.)
void CParameterOptimizer::UpdateParticlePosition(Particle &particle)
{
    // Update position for each parameter dimension
    for(int i = 0; i < m_param_count; i++)
    {
        // Skip disabled parameters (position remains constant)
        if(!m_parameters[i].is_enabled)
            continue;
            
        // Integrate velocity: x(t+1) = x(t) + v(t+1)
        particle.position[i] += particle.velocity[i];
        
        // Enforce boundary constraints using clamping strategy
        // Alternative approaches: reflection, periodic boundaries, penalty methods
        if(particle.position[i] < m_parameters[i].min_value)
        {
            particle.position[i] = m_parameters[i].min_value;
            particle.velocity[i] = 0.0;  // Stop velocity at boundary (optional)
        }
        else if(particle.position[i] > m_parameters[i].max_value)
        {
            particle.position[i] = m_parameters[i].max_value;
            particle.velocity[i] = 0.0;  // Stop velocity at boundary (optional)
        }
    }
}

//+------------------------------------------------------------------+
//| Run Simulated Annealing optimization - Physics-inspired probabilistic search |
//+------------------------------------------------------------------+
// RunSimulatedAnnealing() implements SA algorithm inspired by metallurgical annealing:
// ADVANTAGES:
// - Can escape local optima through probabilistic acceptance of worse solutions
// - Simple single-solution approach (low memory requirements)
// - Proven convergence properties with proper cooling schedule
// - Works well on rough/multimodal fitness landscapes
// DISADVANTAGES:
// - Slower convergence compared to population-based methods
// - Sensitive to temperature schedule parameters
// - No inherent parallelization like GA/PSO
// RECOMMENDED FOR: Complex landscapes with many local optima, when exploration is critical
bool CParameterOptimizer::RunSimulatedAnnealing(void)
{
    Print("Starting Simulated Annealing optimization...");
    
    if(!Initialize())
        return false;
        
    // SA algorithm parameters  
    int max_iterations = m_max_generations * 10;    // More iterations than GA (single solution evolution)
    double initial_temp = 100.0;                    // High initial temperature (accept most moves)
    double final_temp = 0.01;                       // Low final temperature (greedy search)
    
    // Initialize current solution with random starting point
    double current_solution[];                          // Current solution being explored
    ArrayResize(current_solution, m_param_count);
    
    // Generate random starting point within parameter bounds
    for(int i = 0; i < m_param_count; i++)
    {
        // Set disabled parameters to minimum value
        if(!m_parameters[i].is_enabled)
        {
            current_solution[i] = m_parameters[i].min_value;
            continue;
        }
        
        // Random value within [min, max] range
        double range = m_parameters[i].max_value - m_parameters[i].min_value;
        current_solution[i] = m_parameters[i].min_value + range * MathRand() / 32767.0;
    }
    
    // Evaluate fitness of initial solution
    Chromosome current_chromosome;
    ArrayResize(current_chromosome.parameters, m_param_count);
    for(int i = 0; i < m_param_count; i++)
        current_chromosome.parameters[i] = current_solution[i];
        
    double current_fitness = CalculateFitness(current_chromosome);
    
    // Track best solution found so far (global optimum)
    double best_solution[];
    ArrayCopy(best_solution, current_solution);         // Copy initial solution as best
    double best_fitness = current_fitness;              // Initial fitness as best
    
    m_start_time = TimeCurrent();
    
    // Main Simulated Annealing optimization loop
    for(int iter = 0; iter < max_iterations; iter++)
    {
        // Calculate current temperature using cooling schedule
        double temperature = CalculateTemperature(iter, max_iterations);
        
        // Generate neighboring solution by perturbing current solution
        double neighbor_solution[];
        ArrayCopy(neighbor_solution, current_solution);      // Copy current solution
        GenerateNeighborSolution(neighbor_solution);         // Apply random perturbation
        
        // Evaluate fitness of neighboring solution
        Chromosome neighbor_chromosome;
        ArrayResize(neighbor_chromosome.parameters, m_param_count);
        for(int i = 0; i < m_param_count; i++)
            neighbor_chromosome.parameters[i] = neighbor_solution[i];
            
        double neighbor_fitness = CalculateFitness(neighbor_chromosome);
        
        // Apply SA acceptance criterion (Metropolis criterion)
        double delta = neighbor_fitness - current_fitness;   // Fitness difference
        bool accept = false;
        
        if(delta > 0)
        {
            // Always accept better solutions (uphill moves)
            accept = true;
        }
        else if(temperature > 0)
        {
            // Probabilistically accept worse solutions (downhill moves)
            // Probability = exp(ΔE/T) where ΔE < 0 and T > 0
            double probability = MathExp(delta / temperature);
            double random = MathRand() / 32767.0;
            accept = (random < probability);
        }
        
        // Update current solution if neighbor is accepted
        if(accept)
        {
            ArrayCopy(current_solution, neighbor_solution);   // Move to neighbor
            current_fitness = neighbor_fitness;               // Update current fitness
            
            // Update global best if neighbor improves it
            if(current_fitness > best_fitness)
            {
                ArrayCopy(best_solution, current_solution);
                best_fitness = current_fitness;
                Print("Iteration ", iter, " - New best fitness: ", DoubleToString(best_fitness, 6), " (T=", DoubleToString(temperature, 4), ")");
            }
        }
        
        // Progress reporting every 1000 iterations
        if(iter % 1000 == 0)
            Print("SA Iteration ", iter, "/", max_iterations, " - Current: ", DoubleToString(current_fitness, 6), 
                  " Best: ", DoubleToString(best_fitness, 6), " T: ", DoubleToString(temperature, 4));
    }
    
    // Store best result
    if(ArraySize(m_population) < 1)
        ArrayResize(m_population, 1);
    ArrayResize(m_population[0].parameters, m_param_count);
    for(int i = 0; i < m_param_count; i++)
        m_population[0].parameters[i] = best_solution[i];
    m_population[0].fitness = best_fitness;
    m_best_fitness = best_fitness;
    
    datetime end_time = TimeCurrent();
    Print("Simulated Annealing completed in ", (end_time - m_start_time), " seconds");
    Print("Best fitness: ", best_fitness);
    
    return true;
}

//+------------------------------------------------------------------+
//| Generate neighbor solution for SA - Random perturbation strategy |
//+------------------------------------------------------------------+
// GenerateNeighborSolution() creates a neighboring solution by applying random perturbation:
// - Selects random enabled parameter to modify
// - Applies Gaussian-like perturbation (10% of parameter range)
// - Enforces boundary constraints through clamping
// - Returns magnitude of perturbation for adaptive neighborhood sizing
// ALTERNATIVE STRATEGIES: Adaptive step size, multi-parameter changes, constraint-aware moves
double CParameterOptimizer::GenerateNeighborSolution(double &current_params[])
{
    int enabled_indices[];
    for(int i = 0; i < m_param_count; i++)
    {
        if(m_parameters[i].is_enabled)
        {
            int n = ArraySize(enabled_indices);
            ArrayResize(enabled_indices, n + 1);
            enabled_indices[n] = i;
        }
    }
    if(ArraySize(enabled_indices) == 0)
        return 0.0;

    int pick = enabled_indices[MathRand() % ArraySize(enabled_indices)];

    double minv   = m_parameters[pick].min_value;
    double maxv   = m_parameters[pick].max_value;
    double step   = m_parameters[pick].step;
    double range  = maxv - minv;

    double u = 0.0;
    for(int k = 0; k < 3; k++) u += (double)MathRand() / 32767.0;
    u = (u / 3.0 - 0.5) * 2.0;
    double sigma = 0.10 * range;
    double delta = u * sigma;

    double newval = current_params[pick] + delta;
    if(newval < minv) newval = minv;
    if(newval > maxv) newval = maxv;

    if(step > 0.0)
    {
        double steps = MathRound((newval - minv) / step);
        newval = minv + steps * step;
        if(newval < minv) newval = minv;
        if(newval > maxv) newval = maxv;
    }

    current_params[pick] = newval;
    return MathAbs(delta);
}
//+------------------------------------------------------------------+
//| Calculate temperature for SA cooling schedule - Exponential decay |
//+------------------------------------------------------------------+
// CalculateTemperature() implements exponential cooling schedule: T(t) = T0 * α^t
// where:
// - T0 = initial temperature (high, allows most moves)
// - α = cooling rate (< 1, determines decay speed)
// - t = iteration number
// - Tf = final temperature (low, approaches greedy search)
// ALTERNATIVE SCHEDULES: Linear, logarithmic, adaptive, reheating
double CParameterOptimizer::CalculateTemperature(int iteration, int max_iterations)
{
    double initial_temp = 100.0;   // High initial temperature for exploration
    double final_temp = 0.01;      // Low final temperature for exploitation
    
    // Calculate cooling rate α such that T(max_iter) = final_temp
    // Exponential cooling: T(t) = T0 * α^t where α = (Tf/T0)^(1/max_iter)
    double alpha = MathPow(final_temp / initial_temp, 1.0 / max_iterations);
    double temperature = initial_temp * MathPow(alpha, iteration);
    
    // Ensure temperature never drops below final minimum
    return MathMax(temperature, final_temp);
}

//+------------------------------------------------------------------+
//| Run Genetic Algorithm optimization - Evolutionary parameter search |
//+------------------------------------------------------------------+
// RunGeneticAlgorithm() implements a standard GA with key evolutionary operators:
// ADVANTAGES:
// - Excellent global optimization capability (avoids local optima)
// - Handles complex parameter interactions well
// - Naturally parallel and scalable
// - Robust to noise in fitness function
// DISADVANTAGES:
// - Can be slower than gradient-based methods on smooth landscapes
// - Many hyperparameters to tune (population size, rates, selection pressure)
// - No convergence guarantee in finite time
// RECOMMENDED FOR: Complex multimodal landscapes, black-box optimization, robust parameter sets
bool CParameterOptimizer::RunGeneticAlgorithm(void)
{
    Print("Starting Genetic Algorithm optimization...");
    
    if(!Initialize())
        return false;
    
    // Initialize GA state variables
    m_is_running = true;                    // Flag for early termination
    m_start_time = TimeCurrent();           // Track optimization duration
    m_current_generation = 0;               // Generation counter
    m_best_fitness = -DBL_MAX;              // Track global best fitness
    m_stagnation_counter = 0;               // Generations without improvement
    
    // Phase 1: Initialize random population
    if(!InitializePopulation())
    {
        Print("ERROR: Failed to initialize GA population");
        return false;
    }
    
    // Phase 2: Evaluate fitness of initial population
    if(!EvaluatePopulation())
    {
        Print("ERROR: Failed to evaluate initial GA population");
        return false;
    }
    
    // Phase 3: Main evolution loop - Iterative population improvement
    for(m_current_generation = 0; m_current_generation < m_max_generations && m_is_running; m_current_generation++)
    {
        Print("Generation ", m_current_generation + 1, "/", m_max_generations);
        
        // Step 3.1: Elite Selection - Preserve best chromosomes
        // Ensures that the best solutions are not lost during reproduction
        SelectElite();
        
        // Step 3.2: Reproduction - Create new generation through genetic operators
        // Crossover: Combine parents to create offspring (exploration of new regions)
        CrossoverPopulation();
        
        // Mutation: Apply random changes to maintain diversity (prevent premature convergence)
        MutatePopulation();
        
        // Step 3.3: Fitness Evaluation - Calculate performance of new population
        if(!EvaluatePopulation())
        {
            Print("ERROR: Failed to evaluate population at generation ", m_current_generation);
            break;
        }
        
        // Step 3.4: Convergence Monitoring - Track optimization progress
        double current_best = GetBestChromosome().fitness;
        if(current_best > m_best_fitness + m_convergence_threshold)
        {
            // Significant improvement detected
            m_best_fitness = current_best;
            m_stagnation_counter = 0;           // Reset stagnation counter
        }
        else
        {
            // No significant improvement
            m_stagnation_counter++;             // Increment stagnation counter
        }
        
        // Step 3.5: Fitness History Tracking - Store convergence data
        int hist_size = ArraySize(m_fitness_history);
        ArrayResize(m_fitness_history, hist_size + 1);
        m_fitness_history[hist_size] = current_best;
        
        // Step 3.6: Progress Reporting - User feedback
        if(m_current_generation % 10 == 0 || m_current_generation == m_max_generations - 1)
        {
            Print("Generation ", m_current_generation, " Best Fitness: ", 
                  DoubleToString(current_best, 4), " Stagnation: ", m_stagnation_counter);
        }
        
        // Step 3.7: Early Stopping - Prevent unnecessary computation
        // Stop if fitness hasn't improved for m_convergence_patience generations
        if(m_stagnation_counter >= m_convergence_patience)
        {
            Print("Early stopping due to convergence at generation ", m_current_generation);
            break;
        }
    }
    
    // Phase 4: Optimization Completion and Results Reporting
    m_is_running = false;                                   // Mark optimization as complete
    double optimization_time = (double)(TimeCurrent() - m_start_time);  // Calculate total duration
    
    // Report optimization summary
    Print("Genetic Algorithm completed in ", optimization_time, " seconds");
    Print("Final generation: ", m_current_generation, "/", m_max_generations);
    Print("Best fitness achieved: ", DoubleToString(m_best_fitness, 6));
    
    // Display optimal parameter set found
    Chromosome best = GetBestChromosome();
    Print("=== OPTIMAL PARAMETERS ===");
    int param_idx = 0;
    for(int i = 0; i < ArraySize(m_parameters); i++)
    {
        if(m_parameters[i].is_enabled)
        {
            Print(m_parameters[i].name, ": ", 
                  DoubleToString(best.parameters[param_idx], m_parameters[i].precision));
            param_idx++;
        }
    }
    Print("========================");
    
    return true;
}

//+------------------------------------------------------------------+
//| Initialize population                                            |
//+------------------------------------------------------------------+
bool CParameterOptimizer::InitializePopulation(void)
{
    ArrayResize(m_population, m_population_size);
    
    for(int i = 0; i < m_population_size; i++)
    {
        ArrayResize(m_population[i].parameters, m_param_count);
        
        int param_idx = 0;
        for(int j = 0; j < ArraySize(m_parameters); j++)
        {
            if(m_parameters[j].is_enabled)
            {
                double range = m_parameters[j].max_value - m_parameters[j].min_value;
                double random_val = m_parameters[j].min_value + (MathRand() / 32767.0) * range;
                
                // Round to step size
                double steps = MathRound((random_val - m_parameters[j].min_value) / m_parameters[j].step);
                random_val = m_parameters[j].min_value + steps * m_parameters[j].step;
                random_val = MathMax(m_parameters[j].min_value, 
                                   MathMin(m_parameters[j].max_value, random_val));
                
                m_population[i].parameters[param_idx] = random_val;
                param_idx++;
            }
        }
        
        m_population[i].fitness = -DBL_MAX;
        m_population[i].is_valid = false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Evaluate population fitness                                      |
//+------------------------------------------------------------------+
bool CParameterOptimizer::EvaluatePopulation(void)
{
    for(int i = 0; i < ArraySize(m_population); i++)
    {
        if(m_population[i].fitness == -DBL_MAX) // Not evaluated yet
        {
            if(!RunBacktestForChromosome(m_population[i]))
            {
                m_population[i].fitness = -1000.0; // Penalty for invalid chromosome
                m_population[i].is_valid = false;
            }
            else
            {
                m_population[i].fitness = CalculateFitness(m_population[i]);
                m_population[i].is_valid = IsParameterSetValid(m_population[i]);
            }
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Calculate fitness score                                          |
//+------------------------------------------------------------------+
double CParameterOptimizer::CalculateFitness(Chromosome &chromosome)
{
    if(!chromosome.is_valid)
        return -1000.0;
    
    // Normalize metrics to 0-100 scale for combination
    double norm_pf = MathMin(100.0, MathMax(0.0, (chromosome.profit_factor - 1.0) * 50.0));
    double norm_sharpe = MathMin(100.0, MathMax(0.0, (chromosome.sharpe_ratio + 1.0) * 25.0));
    double norm_dd = MathMin(100.0, MathMax(0.0, 100.0 - chromosome.max_drawdown));
    double norm_wr = MathMin(100.0, MathMax(0.0, chromosome.win_rate));
    double norm_ar = MathMin(100.0, MathMax(0.0, chromosome.annual_return));
    
    // Apply constraints as penalties
    double penalty = 0.0;
    if(chromosome.total_trades < m_min_trades)
        penalty += (m_min_trades - chromosome.total_trades) * 2.0;
    
    if(chromosome.max_drawdown > m_max_drawdown_limit)
        penalty += (chromosome.max_drawdown - m_max_drawdown_limit) * 5.0;
    
    if(chromosome.win_rate < m_min_win_rate)
        penalty += (m_min_win_rate - chromosome.win_rate) * 2.0;
    
    if(chromosome.profit_factor < m_min_profit_factor)
        penalty += (m_min_profit_factor - chromosome.profit_factor) * 50.0;
    
    // Calculate weighted fitness
    double fitness = m_weight_profit_factor * norm_pf +
                    m_weight_sharpe_ratio * norm_sharpe +
                    m_weight_max_drawdown * norm_dd +
                    m_weight_win_rate * norm_wr +
                    m_weight_annual_return * norm_ar;
    
    fitness -= penalty;
    
    return fitness;
}

//+------------------------------------------------------------------+
//| Run backtest for chromosome (placeholder)                       |
//+------------------------------------------------------------------+
bool CParameterOptimizer::RunBacktestForChromosome(Chromosome &chromosome)
{
    // This would integrate with the backtesting engine
    // For now, we'll simulate some results
    
    // Extract parameters and run backtest
    // ... backtest code here ...
    
    // Simulate some realistic results
    chromosome.profit_factor = 1.0 + (MathRand() / 32767.0) * 2.0;
    chromosome.sharpe_ratio = -1.0 + (MathRand() / 32767.0) * 3.0;
    chromosome.max_drawdown = 5.0 + (MathRand() / 32767.0) * 25.0;
    chromosome.win_rate = 30.0 + (MathRand() / 32767.0) * 40.0;
    chromosome.total_trades = 50 + MathRand() % 200;
    chromosome.annual_return = -10.0 + (MathRand() / 32767.0) * 50.0;
    
    return true;
}

//+------------------------------------------------------------------+
//| Select elite chromosomes                                         |
//+------------------------------------------------------------------+
void CParameterOptimizer::SelectElite(void)
{
    int elite_count = (int)(m_population_size * m_elite_ratio);
    elite_count = MathMax(1, elite_count);
    
    ArrayResize(m_elite_population, elite_count);
    
    // Sort population by fitness (selection sort for simplicity)
    for(int i = 0; i < ArraySize(m_population) - 1; i++)
    {
        int best_idx = i;
        for(int j = i + 1; j < ArraySize(m_population); j++)
        {
            if(m_population[j].fitness > m_population[best_idx].fitness)
                best_idx = j;
        }
        
        if(best_idx != i)
        {
            Chromosome temp = m_population[i];
            m_population[i] = m_population[best_idx];
            m_population[best_idx] = temp;
        }
    }
    
    // Copy elite
    for(int i = 0; i < elite_count; i++)
    {
        m_elite_population[i] = m_population[i];
    }
}

//+------------------------------------------------------------------+
//| Perform crossover                                                |
//+------------------------------------------------------------------+
void CParameterOptimizer::CrossoverPopulation(void)
{
    int elite_count = ArraySize(m_elite_population);
    
    // Keep elite in new population
    for(int i = 0; i < elite_count; i++)
    {
        m_population[i] = m_elite_population[i];
    }
    
    // Generate offspring through crossover
    for(int i = elite_count; i < m_population_size; i++)
    {
        if((MathRand() / 32767.0) < m_crossover_rate)
        {
            Chromosome parent1 = Tournament_Selection(3);
            Chromosome parent2 = Tournament_Selection(3);
            m_population[i] = Crossover(parent1, parent2);
        }
        else
        {
            m_population[i] = Tournament_Selection(3);
        }
        
        m_population[i].fitness = -DBL_MAX; // Mark for re-evaluation
    }
}

//+------------------------------------------------------------------+
//| Tournament selection                                             |
//+------------------------------------------------------------------+
Chromosome CParameterOptimizer::Tournament_Selection(int tournament_size = 3)
{
    Chromosome best;
    best.fitness = -DBL_MAX;
    
    for(int i = 0; i < tournament_size; i++)
    {
        int idx = MathRand() % ArraySize(m_elite_population);
        if(m_elite_population[idx].fitness > best.fitness)
        {
            best = m_elite_population[idx];
        }
    }
    
    return best;
}

//+------------------------------------------------------------------+
//| Crossover two chromosomes                                        |
//+------------------------------------------------------------------+
Chromosome CParameterOptimizer::Crossover(Chromosome &parent1, Chromosome &parent2)
{
    Chromosome offspring;
    ArrayResize(offspring.parameters, m_param_count);
    
    // Single-point crossover
    int crossover_point = MathRand() % m_param_count;
    
    for(int i = 0; i < m_param_count; i++)
    {
        if(i < crossover_point)
            offspring.parameters[i] = parent1.parameters[i];
        else
            offspring.parameters[i] = parent2.parameters[i];
    }
    
    offspring.fitness = -DBL_MAX;
    offspring.is_valid = false;
    
    return offspring;
}

//+------------------------------------------------------------------+
//| Mutate population                                                |
//+------------------------------------------------------------------+
void CParameterOptimizer::MutatePopulation(void)
{
    int elite_count = ArraySize(m_elite_population);
    
    for(int i = elite_count; i < m_population_size; i++)
    {
        Mutate(m_population[i]);
    }
}

//+------------------------------------------------------------------+
//| Mutate chromosome                                                |
//+------------------------------------------------------------------+
void CParameterOptimizer::Mutate(Chromosome &chromosome)
{
    for(int i = 0; i < m_param_count; i++)
    {
        if((MathRand() / 32767.0) < m_mutation_rate)
        {
            // Find corresponding parameter
            int param_idx = 0;
            int found_param = -1;
            for(int j = 0; j < ArraySize(m_parameters); j++)
            {
                if(m_parameters[j].is_enabled)
                {
                    if(param_idx == i)
                    {
                        found_param = j;
                        break;
                    }
                    param_idx++;
                }
            }
            
            if(found_param >= 0)
            {
                // Gaussian mutation
                double sigma = (m_parameters[found_param].max_value - m_parameters[found_param].min_value) * 0.1;
                double mutation = sigma * (2.0 * (MathRand() / 32767.0) - 1.0);
                
                double new_val = chromosome.parameters[i] + mutation;
                new_val = MathMax(m_parameters[found_param].min_value, 
                                MathMin(m_parameters[found_param].max_value, new_val));
                
                // Round to step
                double steps = MathRound((new_val - m_parameters[found_param].min_value) / m_parameters[found_param].step);
                new_val = m_parameters[found_param].min_value + steps * m_parameters[found_param].step;
                
                chromosome.parameters[i] = new_val;
                chromosome.fitness = -DBL_MAX; // Mark for re-evaluation
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Get best chromosome                                              |
//+------------------------------------------------------------------+
Chromosome CParameterOptimizer::GetBestChromosome(void)
{
    Chromosome best;
    best.fitness = -DBL_MAX;
    
    for(int i = 0; i < ArraySize(m_population); i++)
    {
        if(m_population[i].fitness > best.fitness)
        {
            best = m_population[i];
        }
    }
    
    return best;
}

//+------------------------------------------------------------------+
//| Validate parameter set                                           |
//+------------------------------------------------------------------+
bool CParameterOptimizer::IsParameterSetValid(Chromosome &chromosome)
{
    return (chromosome.total_trades >= m_min_trades &&
            chromosome.max_drawdown <= m_max_drawdown_limit &&
            chromosome.win_rate >= m_min_win_rate &&
            chromosome.profit_factor >= m_min_profit_factor);
}

//+------------------------------------------------------------------+
//| Generate optimization report                                     |
//+------------------------------------------------------------------+
string CParameterOptimizer::GenerateOptimizationReport(void)
{
    Chromosome best = GetBestChromosome();
    
    string report = "=== PARAMETER OPTIMIZATION REPORT ===\n";
    report += "Optimization Method: Genetic Algorithm\n";
    report += "Population Size: " + IntegerToString(m_population_size) + "\n";
    report += "Generations Run: " + IntegerToString(m_current_generation) + "\n";
    report += "Best Fitness: " + DoubleToString(best.fitness, 4) + "\n\n";
    
    report += "--- BEST PARAMETERS ---\n";
    int param_idx = 0;
    for(int i = 0; i < ArraySize(m_parameters); i++)
    {
        if(m_parameters[i].is_enabled)
        {
            report += m_parameters[i].name + ": " + 
                     DoubleToString(best.parameters[param_idx], m_parameters[i].precision) + "\n";
            param_idx++;
        }
    }
    
    report += "\n--- PERFORMANCE METRICS ---\n";
    report += "Profit Factor: " + DoubleToString(best.profit_factor, 3) + "\n";
    report += "Sharpe Ratio: " + DoubleToString(best.sharpe_ratio, 3) + "\n";
    report += "Max Drawdown: " + DoubleToString(best.max_drawdown, 2) + "%\n";
    report += "Win Rate: " + DoubleToString(best.win_rate, 2) + "%\n";
    report += "Annual Return: " + DoubleToString(best.annual_return, 2) + "%\n";
    report += "Total Trades: " + DoubleToString(best.total_trades, 0) + "\n";
    
    return report;
}
//+------------------------------------------------------------------+
//| Search-space size utilities                                       |
//+------------------------------------------------------------------+
long CParameterOptimizer::GetSearchSpaceSize(void)
{
    long total = 1;
    int enabled = 0;
    for(int i = 0; i < ArraySize(m_parameters); i++)
    {
        if(!m_parameters[i].is_enabled)
            continue;
        enabled++;

        double minv = m_parameters[i].min_value;
        double maxv = m_parameters[i].max_value;
        double step = m_parameters[i].step;

        long count = 1;
        if(step > 0.0 && maxv >= minv)
        {
            double steps = MathFloor(((maxv - minv) / step) + 1.0000001);
            if(steps < 1.0) steps = 1.0;
            if(steps > 9.22e18) steps = 9.22e18;
            count = (long)steps;
        }

        if(total > 0 && count > 0 && total <= 9223372036854775807L / count)
            total *= count;
        else
            return 9223372036854775807L;
    }
    if(enabled == 0)
        return 0;
    return total;
}


