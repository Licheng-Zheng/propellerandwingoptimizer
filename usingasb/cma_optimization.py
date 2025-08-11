import cma 
from objective import compute_result
from convertion_auxiliary_functions import array_to_kulfan_dict
import logging 

def run_cma_optimization(evolutionary_strategy, model_name, model_size, alpha, Re, epoch, wanted_lists:dict, importance_list:dict): 

    # initial_sigma = 0.3 


    # options = {
    #     'bounds' : [-1, 1],                 # Ensures the parameters never become too extreme, which would make the airfoil really bad
    #     'popsize': 50,                      # Population size (default would be ~15 for 18D)
    #     'maxiter': 200,                     # Maximum iterations
    #     'maxfevals': 10000,                 # Maximum function evaluations
    #     'tolfun': 1e-8,                     # Tolerance for function value changes
    #     'tolx': 1e-10,                      # Tolerance for parameter changes
    #     'verb_disp': 10,                    # Display progress every N iterations
    #     'verb_log': 1,                      # Log detailed information     
    #     'CMA_stds': initial_sigma,          # Can set different sigma for each parameter
    # }

    # evolutionary_strategy = cma.CMAEvolutionStrategy(initial_guess, initial_sigma, options)

    candidates = evolutionary_strategy.ask()

    candidates_dictionaries = [array_to_kulfan_dict(candidate) for candidate in candidates]

    fitnesses = [compute_result(cst_parameters=candidate_dict, epoch=epoch, alpha=alpha, Re=Re, model=model_size, wanted_lists=wanted_lists, importance_list=importance_list) for candidate_dict in candidates_dictionaries]
    
    # print("FITNESS LIST", fitnesses)

    # OK! Idk what is going on, but SOMEHOW, my fitnesses becomes a numpy array somewhere in my spaghetti, SO! rather than figure out where that is, we fix it here
    fitnesses = [float(fitness) if hasattr(fitness, '__iter__') else fitness for fitness in fitnesses]

    evolutionary_strategy.tell(candidates, fitnesses)

    # print(f"{model_name} - Best Fitness: {min(fitnesses)}")
    
    return min(fitnesses)

