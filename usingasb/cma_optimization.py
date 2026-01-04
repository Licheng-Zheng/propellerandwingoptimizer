import cma 
from objective import compute_result
from convertion_auxiliary_functions import array_to_kulfan_dict
import logging 

def fitness_sort(fitnesses: list[int], candidates: list[list[float]]):
    combined_list = list(zip(fitnesses, candidates))

    combined_list_sorted = sorted(combined_list, key=lambda x: x[0])

    sorted_fitnesses, sorted_candidates = zip(*combined_list_sorted)

    return list(sorted_fitnesses), list(sorted_candidates)

def run_cma_optimization(evolutionary_strategy, model_name, model_size, alpha, Re, wanted_lists:list, importance_list:list, elitist_candidates: list[list[int]], elitist_fitnesses: list[float], elite_length: int, epoch=None): 
    

    candidates = evolutionary_strategy.ask()

    candidates_dictionaries = [array_to_kulfan_dict(candidate) for candidate in candidates]

    fitnesses = [compute_result(cst_parameters=candidate_dict, epoch=epoch, alpha=alpha, Re=Re, model=model_size, wanted_lists=wanted_lists, importance_list=importance_list) for candidate_dict in candidates_dictionaries]
    
    # print("FITNESS LIST", fitnesses)

    # OK! Idk what is going on, but SOMEHOW, my fitnesses becomes a numpy array somewhere in my spaghetti, SO! rather than figure out where that is, we fix it here
    fitnesses = [float(fitness) if hasattr(fitness, '__iter__') else fitness for fitness in fitnesses]

    if elitist_candidates is not None:
        assert isinstance(elitist_candidates, (list, tuple)), "Elitist candidates should be a sequence"
        assert isinstance(elitist_fitnesses, (list, tuple)), "Elitist fitnesses should be a sequence"
                
        fitnesses, candidates = fitness_sort(fitnesses, candidates)

        # When using evolutionary methods, you can assume that the worst ones are usually pretty bad (this is how it explores), so I'm not going to bother with ensuring the elites are better than the worst current ones. I might implement it in the future. 
        fitnesses = fitnesses[:-elite_length]
        candidates = candidates[:-elite_length]

        for elite_candidate, elite_fitness in zip(elitist_candidates, elitist_fitnesses):
            candidates.append(elite_candidate)
            fitnesses.append(elite_fitness)

        # Now, the fitnesses and candidates list are back to their original length, but with the elites added in.

        fitnesses, candidates = fitness_sort(fitnesses, candidates)
    evolutionary_strategy.tell(candidates, fitnesses)

    # print(f"{model_name} - Best Fitness: {min(fitnesses)}")

    new_elites_candidates = candidates[:elite_length]
    new_elite_fitnesses = fitnesses[:elite_length]
    
    return min(fitnesses), new_elites_candidates, new_elite_fitnesses

