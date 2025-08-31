import cma 
from objective import compute_result
from convertion_auxiliary_functions import array_to_kulfan_dict
import logging 

def run_cma_optimization(evolutionary_strategy, model_name, model_size, alpha, Re, wanted_lists:dict, importance_list:dict, epoch=None): 

    candidates = evolutionary_strategy.ask()

    candidates_dictionaries = [array_to_kulfan_dict(candidate) for candidate in candidates]

    fitnesses = [compute_result(cst_parameters=candidate_dict, epoch=epoch, alpha=alpha, Re=Re, model=model_size, wanted_lists=wanted_lists, importance_list=importance_list) for candidate_dict in candidates_dictionaries]
    
    # print("FITNESS LIST", fitnesses)

    # OK! Idk what is going on, but SOMEHOW, my fitnesses becomes a numpy array somewhere in my spaghetti, SO! rather than figure out where that is, we fix it here
    fitnesses = [float(fitness) if hasattr(fitness, '__iter__') else fitness for fitness in fitnesses]

    evolutionary_strategy.tell(candidates, fitnesses)

    # print(f"{model_name} - Best Fitness: {min(fitnesses)}")
    
    return min(fitnesses)

