//+------------------------------------------------------------------+
//|                                           ParameterOptimizer.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Parameter Optimization Framework                                 |
//+------------------------------------------------------------------+

// Optimization method enumeration
enum ENUM_OPTIMIZATION_METHOD
{
    OPT_GENETIC_ALGORITHM = 0,    // Genetic Algorithm
    OPT_BRUTE_FORCE = 1,         // Brute Force
    OPT_PARTICLE_SWARM = 2,      // Particle Swarm Optimization
    OPT_SIMULATED_ANNEALING = 3  // Simulated Annealing
};

// Optimization parameter structure
struct OptimizationParameter
{
    string   name;                    // Parameter name
    double   min_value;               // Minimum value
    double   max_value;               // Maximum value
    double   step;                    // Step size
    double   current_value;           // Current value
    bool     is_enabled;              // Enable/disable optimization
    int      precision;               // Decimal precision
    
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

// Individual chromosome for genetic algorithm
struct Chromosome
{
    double   parameters[];            // Parameter values
    double   fitness;                 // Fitness score
    double   sharpe_ratio;           // Sharpe ratio
    double   profit_factor;          // Profit factor
    double   max_drawdown;           // Maximum drawdown
    double   win_rate;               // Win rate
    double   total_trades;           // Total number of trades
    double   annual_return;          // Annualized return
    bool     is_valid;               // Valid chromosome flag
    
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

// Optimization results
struct OptimizationResults
{
    Chromosome best_chromosome;       // Best parameter set
    Chromosome population[];          // Final population
    double     convergence_history[]; // Fitness convergence
    int        generations_run;       // Generations executed
    double     optimization_time;     // Time taken in seconds
    string     optimization_report;   // Detailed report
};

//+------------------------------------------------------------------+
//| Parameter Optimizer Class                                        |
//+------------------------------------------------------------------+
class CParameterOptimizer
{
private:
    // Optimization parameters
    OptimizationParameter m_parameters[];
    int                   m_param_count;
    
    // Genetic Algorithm parameters
    int                   m_population_size;
    int                   m_max_generations;
    double                m_mutation_rate;
    double                m_crossover_rate;
    double                m_elite_ratio;
    double                m_convergence_threshold;
    int                   m_convergence_patience;
    
    // Population management
    Chromosome            m_population[];
    Chromosome            m_elite_population[];
    double                m_fitness_history[];
    
    // Optimization state
    bool                  m_is_running;
    int                   m_current_generation;
    double                m_best_fitness;
    int                   m_stagnation_counter;
    datetime              m_start_time;
    
    // Fitness calculation weights
    double                m_weight_profit_factor;
    double                m_weight_sharpe_ratio;
    double                m_weight_max_drawdown;
    double                m_weight_win_rate;
    double                m_weight_annual_return;
    
    // Optimization constraints
    double                m_min_trades;
    double                m_max_drawdown_limit;
    double                m_min_win_rate;
    double                m_min_profit_factor;

public:
    // Constructor/Destructor
                         CParameterOptimizer(void);
                        ~CParameterOptimizer(void);
    
    // Setup methods
    bool                 Initialize(void);
    bool                 AddParameter(string name, double min_val, double max_val, 
                                    double step, bool enabled = true, int precision = 4);
    void                 SetGeneticAlgorithmParameters(int population_size = 50, 
                                                      int max_generations = 100,
                                                      double mutation_rate = 0.1,
                                                      double crossover_rate = 0.8);
    void                 SetFitnessWeights(double profit_factor = 0.3, double sharpe = 0.25,
                                          double drawdown = 0.2, double win_rate = 0.15,
                                          double annual_return = 0.1);
    void                 SetConstraints(double min_trades = 50, double max_dd = 30.0,
                                       double min_win_rate = 30.0, double min_pf = 1.1);
    
    // Optimization methods
    bool                 RunOptimization(void);
    bool                 RunBruteForceOptimization(void);
    bool                 RunGeneticAlgorithm(void);
    bool                 RunParticleSwarmOptimization(void);
    bool                 RunSimulatedAnnealing(void);
    
    // Genetic Algorithm implementation
    bool                 InitializePopulation(void);
    bool                 EvaluatePopulation(void);
    double               CalculateFitness(Chromosome &chromosome);
    bool                 RunBacktestForChromosome(Chromosome &chromosome);
    void                 SelectElite(void);
    void                 CrossoverPopulation(void);
    void                 MutatePopulation(void);
    Chromosome           Tournament_Selection(int tournament_size = 3);
    Chromosome           Crossover(Chromosome &parent1, Chromosome &parent2);
    void                 Mutate(Chromosome &chromosome);
    
    // Brute Force Optimization
    bool                 TestAllCombinations(int param_index, Chromosome &best_chromosome, 
                                           double &best_fitness, int &tested_count, int total_count);
    
    // Particle Swarm Optimization
    struct Particle
    {
        double position[];
        double velocity[];
        double best_position[];
        double best_fitness;
        double fitness;
    };
    bool                 RunPSO(void);
    void                 UpdateParticleVelocity(Particle &particle, double &global_best[]);
    void                 UpdateParticlePosition(Particle &particle);
    
    // Simulated Annealing
    bool                 RunSA(void);
    double               GenerateNeighborSolution(double &current_params[]);
    double               CalculateTemperature(int iteration, int max_iterations);
    
    // Walk Forward Analysis
    bool                 RunWalkForwardAnalysis(int window_size = 252, int step_size = 63);
    struct WalkForwardResult
    {
        datetime start_date;
        datetime end_date;
        Chromosome best_params;
        double oos_performance;
        double is_performance;
    };
    WalkForwardResult    m_wf_results[];
    
    // Multi-objective optimization
    bool                 RunMultiObjectiveOptimization(void);
    double               CalculateParetoRank(Chromosome &chromosome);
    bool                 DominatesChromosome(Chromosome &a, Chromosome &b);
    
    // Cross-validation
    bool                 RunCrossValidation(int k_folds = 5);
    double               CalculateAverageCV_Performance(double &params[]);
    
    // Parameter analysis
    bool                 RunSensitivityAnalysis(void);
    bool                 RunParameterCorrelationAnalysis(void);
    bool                 Generate3DParameterSurface(string param1, string param2);
    
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
//| Constructor                                                      |
//+------------------------------------------------------------------+
CParameterOptimizer::CParameterOptimizer(void)
{
    m_param_count = 0;
    m_population_size = 50;
    m_max_generations = 100;
    m_mutation_rate = 0.1;
    m_crossover_rate = 0.8;
    m_elite_ratio = 0.1;
    m_convergence_threshold = 0.001;
    m_convergence_patience = 10;
    
    m_is_running = false;
    m_current_generation = 0;
    m_best_fitness = -DBL_MAX;
    m_stagnation_counter = 0;
    
    // Default fitness weights
    m_weight_profit_factor = 0.3;
    m_weight_sharpe_ratio = 0.25;
    m_weight_max_drawdown = 0.2;
    m_weight_win_rate = 0.15;
    m_weight_annual_return = 0.1;
    
    // Default constraints
    m_min_trades = 50;
    m_max_drawdown_limit = 30.0;
    m_min_win_rate = 30.0;
    m_min_profit_factor = 1.1;
    
    ArrayResize(m_parameters, 0);
    ArrayResize(m_population, 0);
    ArrayResize(m_elite_population, 0);
    ArrayResize(m_fitness_history, 0);
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
//| Initialize optimizer                                             |
//+------------------------------------------------------------------+
bool CParameterOptimizer::Initialize(void)
{
    if(m_param_count == 0)
    {
        Print("No parameters defined for optimization");
        return false;
    }
    
    // Initialize random seed
    MathSrand((int)TimeCurrent());
    
    return true;
}

//+------------------------------------------------------------------+
//| Add optimization parameter                                       |
//+------------------------------------------------------------------+
bool CParameterOptimizer::AddParameter(string name, double min_val, double max_val, 
                                      double step, bool enabled = true, int precision = 4)
{
    if(min_val >= max_val || step <= 0)
    {
        Print("Invalid parameter range: ", name);
        return false;
    }
    
    int size = ArraySize(m_parameters);
    ArrayResize(m_parameters, size + 1);
    
    m_parameters[size].name = name;
    m_parameters[size].min_value = min_val;
    m_parameters[size].max_value = max_val;
    m_parameters[size].step = step;
    m_parameters[size].current_value = min_val;
    m_parameters[size].is_enabled = enabled;
    m_parameters[size].precision = precision;
    
    if(enabled) m_param_count++;
    
    Print("Added parameter: ", name, " Range: [", min_val, ", ", max_val, "] Step: ", step);
    
    return true;
}

//+------------------------------------------------------------------+
//| Set Genetic Algorithm parameters                                 |
//+------------------------------------------------------------------+
void CParameterOptimizer::SetGeneticAlgorithmParameters(int population_size = 50, 
                                                        int max_generations = 100,
                                                        double mutation_rate = 0.1,
                                                        double crossover_rate = 0.8)
{
    m_population_size = MathMax(10, population_size);
    m_max_generations = MathMax(10, max_generations);
    m_mutation_rate = MathMax(0.01, MathMin(0.5, mutation_rate));
    m_crossover_rate = MathMax(0.5, MathMin(1.0, crossover_rate));
    
    Print("GA Parameters - Population: ", m_population_size, 
          " Generations: ", m_max_generations,
          " Mutation: ", m_mutation_rate,
          " Crossover: ", m_crossover_rate);
}

//+------------------------------------------------------------------+
//| Set fitness function weights                                     |
//+------------------------------------------------------------------+
void CParameterOptimizer::SetFitnessWeights(double profit_factor = 0.3, double sharpe = 0.25,
                                           double drawdown = 0.2, double win_rate = 0.15,
                                           double annual_return = 0.1)
{
    m_weight_profit_factor = profit_factor;
    m_weight_sharpe_ratio = sharpe;
    m_weight_max_drawdown = drawdown;
    m_weight_win_rate = win_rate;
    m_weight_annual_return = annual_return;
    
    Print("Fitness weights updated - ProfitFactor: ", profit_factor,
          " Sharpe: ", sharpe, " Drawdown: ", drawdown,
          " WinRate: ", win_rate, " AnnualReturn: ", annual_return);
}

//+------------------------------------------------------------------+
//| Set optimization constraints                                     |
//+------------------------------------------------------------------+
void CParameterOptimizer::SetConstraints(double min_trades = 50, double max_dd = 30.0,
                                        double min_win_rate = 30.0, double min_pf = 1.1)
{
    m_min_trades = (int)min_trades;
    m_max_drawdown_limit = max_dd;
    m_min_win_rate = min_win_rate;
    m_min_profit_factor = min_pf;
    
    Print("Optimization constraints updated - MinTrades: ", m_min_trades,
          " MaxDrawdown: ", m_max_drawdown_limit, "%",
          " MinWinRate: ", m_min_win_rate, "%",
          " MinProfitFactor: ", m_min_profit_factor);
}

//+------------------------------------------------------------------+
//| Run optimization using specified method                         |
//+------------------------------------------------------------------+
bool CParameterOptimizer::RunOptimization(void)
{
    Print("Starting optimization with method: Genetic Algorithm (default)");
    return RunGeneticAlgorithm();
}

//+------------------------------------------------------------------+
//| Run Brute Force optimization                                     |
//+------------------------------------------------------------------+
bool CParameterOptimizer::RunBruteForceOptimization(void)
{
    Print("Starting Brute Force optimization...");
    
    if(!Initialize())
        return false;
        
    int total_combinations = 1;
    
    // Calculate total combinations
    for(int i = 0; i < m_param_count; i++)
    {
        if(!m_parameters[i].is_enabled)
            continue;
            
        int steps = (int)((m_parameters[i].max_value - m_parameters[i].min_value) / m_parameters[i].step) + 1;
        total_combinations *= steps;
    }
    
    Print("Total parameter combinations to test: ", total_combinations);
    
    if(total_combinations > 100000)
    {
        Print("ERROR: Too many combinations (", total_combinations, "). Consider reducing parameter ranges or using Genetic Algorithm.");
        return false;
    }
    
    Chromosome best_chromosome;
    double best_fitness = -DBL_MAX;
    int tested_combinations = 0;
    
    m_start_time = TimeCurrent();
    
    // Recursive brute force testing
    if(!TestAllCombinations(0, best_chromosome, best_fitness, tested_combinations, total_combinations))
    {
        Print("Brute Force optimization failed");
        return false;
    }
    
    // Store best result
    if(ArraySize(m_population) < 1)
        ArrayResize(m_population, 1);
    m_population[0] = best_chromosome;
    m_best_fitness = best_fitness;
    
    datetime end_time = TimeCurrent();
    Print("Brute Force optimization completed in ", (end_time - m_start_time), " seconds");
    Print("Best fitness: ", best_fitness, " from ", tested_combinations, " combinations");
    
    return true;
}

//+------------------------------------------------------------------+
//| Test all parameter combinations recursively                     |
//+------------------------------------------------------------------+
bool CParameterOptimizer::TestAllCombinations(int param_index, Chromosome &best_chromosome, 
                                             double &best_fitness, int &tested_count, int total_count)
{
    if(param_index >= m_param_count)
    {
        // All parameters set, test this combination
        Chromosome test_chromosome;
        ArrayResize(test_chromosome.parameters, m_param_count);
        
        for(int i = 0; i < m_param_count; i++)
            test_chromosome.parameters[i] = m_parameters[i].current_value;
            
        if(!IsParameterSetValid(test_chromosome))
            return true;
            
        double fitness = CalculateFitness(test_chromosome);
        tested_count++;
        
        if(tested_count % 1000 == 0)
            Print("Progress: ", tested_count, "/", total_count, " (", 
                  NormalizeDouble(100.0 * tested_count / total_count, 1), "%)");
        
        if(fitness > best_fitness)
        {
            best_fitness = fitness;
            best_chromosome = test_chromosome;
            Print("New best fitness: ", fitness);
        }
        
        return true;
    }
    
    if(!m_parameters[param_index].is_enabled)
    {
        // Skip disabled parameter
        return TestAllCombinations(param_index + 1, best_chromosome, best_fitness, tested_count, total_count);
    }
    
    // Test all values for this parameter
    for(double val = m_parameters[param_index].min_value; 
        val <= m_parameters[param_index].max_value + m_parameters[param_index].step / 2; 
        val += m_parameters[param_index].step)
    {
        m_parameters[param_index].current_value = val;
        
        if(!TestAllCombinations(param_index + 1, best_chromosome, best_fitness, tested_count, total_count))
            return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Run Particle Swarm Optimization                                 |
//+------------------------------------------------------------------+
bool CParameterOptimizer::RunParticleSwarmOptimization(void)
{
    Print("Starting Particle Swarm Optimization...");
    
    if(!Initialize())
        return false;
        
    int swarm_size = m_population_size;
    int max_iterations = m_max_generations;
    double inertia_weight = 0.9;
    double cognitive_coeff = 2.0;
    double social_coeff = 2.0;
    
    // Initialize particle swarm
    Particle particles[];
    ArrayResize(particles, swarm_size);
    
    double global_best_position[];
    ArrayResize(global_best_position, m_param_count);
    double global_best_fitness = -DBL_MAX;
    
    // Initialize particles
    for(int i = 0; i < swarm_size; i++)
    {
        ArrayResize(particles[i].position, m_param_count);
        ArrayResize(particles[i].velocity, m_param_count);
        ArrayResize(particles[i].best_position, m_param_count);
        
        // Random initial position
        for(int j = 0; j < m_param_count; j++)
        {
            if(!m_parameters[j].is_enabled)
            {
                particles[i].position[j] = m_parameters[j].min_value;
                particles[i].velocity[j] = 0.0;
                particles[i].best_position[j] = m_parameters[j].min_value;
                continue;
            }
            
            double range = m_parameters[j].max_value - m_parameters[j].min_value;
            particles[i].position[j] = m_parameters[j].min_value + range * MathRand() / 32767.0;
            particles[i].velocity[j] = (MathRand() / 32767.0 - 0.5) * range * 0.1;
            particles[i].best_position[j] = particles[i].position[j];
        }
        
        // Evaluate initial fitness
        Chromosome test_chromosome;
        ArrayResize(test_chromosome.parameters, m_param_count);
        for(int j = 0; j < m_param_count; j++)
            test_chromosome.parameters[j] = particles[i].position[j];
            
        particles[i].fitness = CalculateFitness(test_chromosome);
        particles[i].best_fitness = particles[i].fitness;
        
        // Update global best
        if(particles[i].fitness > global_best_fitness)
        {
            global_best_fitness = particles[i].fitness;
            for(int j = 0; j < m_param_count; j++)
                global_best_position[j] = particles[i].position[j];
        }
    }
    
    m_start_time = TimeCurrent();
    
    // PSO iterations
    for(int iter = 0; iter < max_iterations; iter++)
    {
        // Update inertia weight (decreasing)
        inertia_weight = 0.9 - 0.7 * iter / max_iterations;
        
        for(int i = 0; i < swarm_size; i++)
        {
            // Update velocity and position
            UpdateParticleVelocity(particles[i], global_best_position);
            UpdateParticlePosition(particles[i]);
            
            // Evaluate fitness
            Chromosome test_chromosome;
            ArrayResize(test_chromosome.parameters, m_param_count);
            for(int j = 0; j < m_param_count; j++)
                test_chromosome.parameters[j] = particles[i].position[j];
                
            particles[i].fitness = CalculateFitness(test_chromosome);
            
            // Update personal best
            if(particles[i].fitness > particles[i].best_fitness)
            {
                particles[i].best_fitness = particles[i].fitness;
                for(int j = 0; j < m_param_count; j++)
                    particles[i].best_position[j] = particles[i].position[j];
            }
            
            // Update global best
            if(particles[i].fitness > global_best_fitness)
            {
                global_best_fitness = particles[i].fitness;
                for(int j = 0; j < m_param_count; j++)
                    global_best_position[j] = particles[i].position[j];
                    
                Print("Generation ", iter, " - New best fitness: ", global_best_fitness);
            }
        }
        
        if(iter % 10 == 0)
            Print("PSO Generation ", iter, "/", max_iterations, " - Best fitness: ", global_best_fitness);
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
//| Update particle velocity                                         |
//+------------------------------------------------------------------+
void CParameterOptimizer::UpdateParticleVelocity(Particle &particle, double &global_best[])
{
    double inertia_weight = 0.5;
    double cognitive_coeff = 2.0;
    double social_coeff = 2.0;
    
    for(int i = 0; i < m_param_count; i++)
    {
        if(!m_parameters[i].is_enabled)
        {
            particle.velocity[i] = 0.0;
            continue;
        }
        
        double r1 = MathRand() / 32767.0;
        double r2 = MathRand() / 32767.0;
        
        particle.velocity[i] = inertia_weight * particle.velocity[i] +
                              cognitive_coeff * r1 * (particle.best_position[i] - particle.position[i]) +
                              social_coeff * r2 * (global_best[i] - particle.position[i]);
    }
}

//+------------------------------------------------------------------+
//| Update particle position                                         |
//+------------------------------------------------------------------+
void CParameterOptimizer::UpdateParticlePosition(Particle &particle)
{
    for(int i = 0; i < m_param_count; i++)
    {
        if(!m_parameters[i].is_enabled)
            continue;
            
        particle.position[i] += particle.velocity[i];
        
        // Boundary constraints
        if(particle.position[i] < m_parameters[i].min_value)
            particle.position[i] = m_parameters[i].min_value;
        else if(particle.position[i] > m_parameters[i].max_value)
            particle.position[i] = m_parameters[i].max_value;
    }
}

//+------------------------------------------------------------------+
//| Run Simulated Annealing optimization                            |
//+------------------------------------------------------------------+
bool CParameterOptimizer::RunSimulatedAnnealing(void)
{
    Print("Starting Simulated Annealing optimization...");
    
    if(!Initialize())
        return false;
        
    int max_iterations = m_max_generations * 10; // More iterations for SA
    double initial_temp = 100.0;
    double final_temp = 0.01;
    
    // Initialize with random solution
    double current_solution[];
    ArrayResize(current_solution, m_param_count);
    
    for(int i = 0; i < m_param_count; i++)
    {
        if(!m_parameters[i].is_enabled)
        {
            current_solution[i] = m_parameters[i].min_value;
            continue;
        }
        
        double range = m_parameters[i].max_value - m_parameters[i].min_value;
        current_solution[i] = m_parameters[i].min_value + range * MathRand() / 32767.0;
    }
    
    // Evaluate initial solution
    Chromosome current_chromosome;
    ArrayResize(current_chromosome.parameters, m_param_count);
    for(int i = 0; i < m_param_count; i++)
        current_chromosome.parameters[i] = current_solution[i];
        
    double current_fitness = CalculateFitness(current_chromosome);
    
    // Best solution tracking
    double best_solution[];
    ArrayCopy(best_solution, current_solution);
    double best_fitness = current_fitness;
    
    m_start_time = TimeCurrent();
    
    // Simulated Annealing iterations
    for(int iter = 0; iter < max_iterations; iter++)
    {
        double temperature = CalculateTemperature(iter, max_iterations);
        
        // Generate neighbor solution
        double neighbor_solution[];
        ArrayCopy(neighbor_solution, current_solution);
        GenerateNeighborSolution(neighbor_solution);
        
        // Evaluate neighbor
        Chromosome neighbor_chromosome;
        ArrayResize(neighbor_chromosome.parameters, m_param_count);
        for(int i = 0; i < m_param_count; i++)
            neighbor_chromosome.parameters[i] = neighbor_solution[i];
            
        double neighbor_fitness = CalculateFitness(neighbor_chromosome);
        
        // Accept or reject neighbor
        double delta = neighbor_fitness - current_fitness;
        bool accept = false;
        
        if(delta > 0)
        {
            accept = true; // Better solution
        }
        else if(temperature > 0)
        {
            double probability = MathExp(delta / temperature);
            double random = MathRand() / 32767.0;
            accept = (random < probability);
        }
        
        if(accept)
        {
            ArrayCopy(current_solution, neighbor_solution);
            current_fitness = neighbor_fitness;
            
            // Update best if improved
            if(current_fitness > best_fitness)
            {
                ArrayCopy(best_solution, current_solution);
                best_fitness = current_fitness;
                Print("Iteration ", iter, " - New best fitness: ", best_fitness, " (T=", temperature, ")");
            }
        }
        
        if(iter % 1000 == 0)
            Print("SA Iteration ", iter, "/", max_iterations, " - Current: ", current_fitness, 
                  " Best: ", best_fitness, " T: ", temperature);
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
//| Generate neighbor solution for SA                               |
//+------------------------------------------------------------------+
double CParameterOptimizer::GenerateNeighborSolution(double &current_params[])
{
    // Randomly select parameter to modify
    int param_to_modify = -1;
    int attempts = 0;
    
    while(param_to_modify == -1 && attempts < 100)
    {
        int idx = (int)(MathRand() * m_param_count / 32767.0);
        if(idx >= 0 && idx < m_param_count && m_parameters[idx].is_enabled)
            param_to_modify = idx;
        attempts++;
    }
    
    if(param_to_modify == -1)
        return 0.0; // No enabled parameters
    
    // Generate neighbor by adding random perturbation
    double range = m_parameters[param_to_modify].max_value - m_parameters[param_to_modify].min_value;
    double perturbation = (MathRand() / 32767.0 - 0.5) * range * 0.1; // 10% of range
    
    current_params[param_to_modify] += perturbation;
    
    // Ensure bounds
    if(current_params[param_to_modify] < m_parameters[param_to_modify].min_value)
        current_params[param_to_modify] = m_parameters[param_to_modify].min_value;
    else if(current_params[param_to_modify] > m_parameters[param_to_modify].max_value)
        current_params[param_to_modify] = m_parameters[param_to_modify].max_value;
    
    return MathAbs(perturbation);
}

//+------------------------------------------------------------------+
//| Calculate temperature for SA cooling schedule                   |
//+------------------------------------------------------------------+
double CParameterOptimizer::CalculateTemperature(int iteration, int max_iterations)
{
    double initial_temp = 100.0;
    double final_temp = 0.01;
    
    // Exponential cooling schedule
    double alpha = MathPow(final_temp / initial_temp, 1.0 / max_iterations);
    double temperature = initial_temp * MathPow(alpha, iteration);
    
    return MathMax(temperature, final_temp);
}

//+------------------------------------------------------------------+
//| Run Genetic Algorithm optimization                               |
//+------------------------------------------------------------------+
bool CParameterOptimizer::RunGeneticAlgorithm(void)
{
    Print("Starting Genetic Algorithm optimization...");
    
    if(!Initialize())
        return false;
    
    m_is_running = true;
    m_start_time = TimeCurrent();
    m_current_generation = 0;
    m_best_fitness = -DBL_MAX;
    m_stagnation_counter = 0;
    
    // Initialize population
    if(!InitializePopulation())
    {
        Print("Failed to initialize population");
        return false;
    }
    
    // Evaluate initial population
    if(!EvaluatePopulation())
    {
        Print("Failed to evaluate initial population");
        return false;
    }
    
    // Evolution loop
    for(m_current_generation = 0; m_current_generation < m_max_generations && m_is_running; m_current_generation++)
    {
        Print("Generation ", m_current_generation + 1, "/", m_max_generations);
        
        // Select elite
        SelectElite();
        
        // Create new generation
        CrossoverPopulation();
        MutatePopulation();
        
        // Evaluate new population
        if(!EvaluatePopulation())
        {
            Print("Failed to evaluate population at generation ", m_current_generation);
            break;
        }
        
        // Check convergence
        double current_best = GetBestChromosome().fitness;
        if(current_best > m_best_fitness + m_convergence_threshold)
        {
            m_best_fitness = current_best;
            m_stagnation_counter = 0;
        }
        else
        {
            m_stagnation_counter++;
        }
        
        // Store fitness history
        int hist_size = ArraySize(m_fitness_history);
        ArrayResize(m_fitness_history, hist_size + 1);
        m_fitness_history[hist_size] = current_best;
        
        // Print progress
        if(m_current_generation % 10 == 0 || m_current_generation == m_max_generations - 1)
        {
            Print("Generation ", m_current_generation, " Best Fitness: ", 
                  DoubleToString(current_best, 4), " Stagnation: ", m_stagnation_counter);
        }
        
        // Check early stopping
        if(m_stagnation_counter >= m_convergence_patience)
        {
            Print("Early stopping due to convergence at generation ", m_current_generation);
            break;
        }
    }
    
    m_is_running = false;
    double optimization_time = (double)(TimeCurrent() - m_start_time);
    
    Print("Genetic Algorithm completed in ", optimization_time, " seconds");
    Print("Best fitness: ", DoubleToString(m_best_fitness, 4));
    
    // Print best parameters
    Chromosome best = GetBestChromosome();
    Print("Best parameters:");
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