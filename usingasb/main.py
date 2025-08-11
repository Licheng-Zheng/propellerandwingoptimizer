import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
from objective import compute_result
from cma_optimization import run_cma_optimization
import cma
import numpy as np
from convertion_auxiliary_functions import kulfan_dict_to_array, array_to_kulfan_dict
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import logging 
from logging_auxiliary_functions import save_optimization_log, save_intermediate_results, plot_optimization_results, stop_functioning
import display_auxiliary_functions

# First time using logging because using print is for stupid people (idk how to use this)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Starting guess (I will refine how this is chosen later) 
airfoil = asb.Airfoil("naca4412")

# Convert it into CST parameters 
starting_guess_kulfan = get_kulfan_parameters(airfoil.coordinates)
starting_guess_kulfan = kulfan_dict_to_array(starting_guess_kulfan)

logging.debug("Starting CST parameters, does it look okie dokie?", starting_guess_kulfan)

stop_functioning("These are the current starting CST parameters and the drawn airfoil", additional_info_context="CST parameters", additional_info=starting_guess_kulfan, cst_parameters=starting_guess_kulfan)

### CMA Parameters 
initial_sigma = 0.3 

max_epochs = 100

# Set up bounds for the CMA-ES
param_dimension = len(starting_guess_kulfan) # I put starting guess kulfan here instead of the length of the kulfan dict (the integer) in case I ever want to create my own Neural Foil that takes more than just 8 CST parameters per side
lower_bounds = np.full(param_dimension, -1.0)
upper_bounds = np.full(param_dimension, 1.0)

options = {
    'bounds': [lower_bounds, upper_bounds],  # Ensures the parameters never become too extreme, which would make the airfoil really bad
    'popsize': 60,                      # Population size (default would be ~15 for 18D)
    'maxiter': 200,                     # Maximum iterations
    'maxfevals': 10000,                 # Maximum function evaluations
    'tolfun': 1e-8,                     # Tolerance for function value changes
    'tolx': 1e-10,                      # Tolerance for parameter changes
    'verb_disp': 10,                    # Display progress every N iterations
    'verb_log': 1,                      # Log detailed information     
    'CMA_stds': [initial_sigma] * param_dimension         # Can set different sigma for each parameter
}

model_name = ["cmaes_1"]
optimizers = []

# APPARENTLY I need to do this for proper logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"optimization_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

optimization_log = {}

# I'll need to add functionality for other optimization strategies in the future
for model_name in model_name: 
    strategy = cma.CMAEvolutionStrategy(starting_guess_kulfan, initial_sigma, options)
    optimizers.append((model_name, strategy))

    optimization_log[model_name] = {
        'epochs': [],
        'best_fitness_per_epoch': [],
        'sigma_per_epoch': [],
        'best_parameters_per_epoch': [],
        'all_fitness_values': [],
        'convergence_data': []
    }

wanted_list = ["analysis_confidence", "CL", "CD", "CM"]
importance_list = [0.4, 0.3, -0.2, -0.1]

best_params = None

for epoch in range(max_epochs):
    for model_name, es in optimizers:

        if es.stop(): 
            # if the evolutionary strategy has already converged, nothing is done
            continue

        best_fitness = run_cma_optimization(evolutionary_strategy=es, 
                                            model_name=model_name, 
                                            model_size="large", 
                                            alpha=5, 
                                            Re=1e6, 
                                            epoch=epoch, 
                                            wanted_lists=wanted_list, 
                                            importance_list=importance_list)
        
                # Log data for this epoch
        optimization_log[model_name]['epochs'].append(epoch + 1)
        optimization_log[model_name]['best_fitness_per_epoch'].append(best_fitness)
        optimization_log[model_name]['sigma_per_epoch'].append(es.sigma)
        optimization_log[model_name]['best_parameters_per_epoch'].append(es.result.xbest.copy())
        
        # Log convergence info
        convergence_info = {
            'iteration': len(optimization_log[model_name]['epochs']),
            'evaluations': es.countevals,
            'best_fitness': float(es.result.fbest) if es.result.fbest is not None else best_fitness,
            'sigma': float(es.sigma),
            'condition_number': float(es.D.max() / es.D.min()) if hasattr(es, 'D') else None
        }
        optimization_log[model_name]['convergence_data'].append(convergence_info)
        
        print(f"{model_name}: Epoch {epoch + 1}, Best: {best_fitness:.6f}, Sigma: {es.sigma:.6f}, Evals: {es.countevals}")
    
    # # Save intermediate results every 10 epochs
    # if (epoch + 1) % 10 == 0:
    #     save_intermediate_results(optimization_log, results_dir, epoch + 1)

    best_result = es.result.xbest

first_item = array_to_kulfan_dict(starting_guess_kulfan).copy()

cst_variations = [first_item]

best_airfoil = array_to_kulfan_dict(best_result)

print(best_airfoil)

cst_variations.append(best_airfoil)

print(cst_variations)

logging.debug(type(cst_variations), type(cst_variations[0]), type(cst_variations[1]))

display_auxiliary_functions.plot_multiple_cst_airfoils(cst_variations, labels=["starting guess", "best guess"], block=True)

print("\nOptimization completed!")

# Final results and plotting
for model_name, es in optimizers:
    print(f"\nFinal Results for {model_name}:")
    print(f"Best fitness: {es.result.fbest:.8f}")
    print(f"Total evaluations: {es.countevals}")
    print(f"Final sigma: {es.sigma:.6f}")
    print(f"Best parameters shape: {es.result.xbest.shape}")
    
    # Save final best parameters
    np.savetxt(
        os.path.join(results_dir, f"{model_name}_best_parameters.txt"), 
        es.result.xbest,
        header=f"Best CST parameters from {model_name} optimization\nFinal fitness: {es.result.fbest}"
    )


# {'lower_weights': array([-0.68093984, -0.98782294, -0.6950461 , -0.51509534, -0.85793511,
#         0.67903817,  0.62635203,  0.9287848 ]), 'upper_weights': array([ 0.71138293, -0.50625873, -0.01808301, -0.22555874,  0.66091669,
#         0.93262115,  0.75631632, -0.40774508]), 'leading_edge_weight': np.float64(0.8245977383605523), 'TE_thickness': np.float64(-0.020606797899721307)}
# [{'lower_weights': array([-0.16965146, -0.09364138, -0.06345896, -0.0067966 , -0.0902447 ,
#         0.02081845, -0.03575216, -0.00223623]), 'upper_weights': array([0.18109497, 0.21268419, 0.28098503, 0.24864887, 0.2402814 ,
#        0.27262843, 0.25776474, 0.27817638]), 'leading_edge_weight': np.float64(0.10647339061374254), 'TE_thickness': np.float64(0.002572011317150121)}, {'lower_weights': array([-0.68093984, -0.98782294, -0.6950461 , -0.51509534, -0.85793511,
#         0.67903817,  0.62635203,  0.9287848 ]), 'upper_weights': array([ 0.71138293, -0.50625873, -0.01808301, -0.22555874,  0.66091669,
#         0.93262115,  0.75631632, -0.40774508]), 'leading_edge_weight': np.float64(0.8245977383605523), 'TE_thickness': np.float64(-0.020606797899721307)}]
