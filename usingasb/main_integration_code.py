# ===================================================================
# ADD THIS CODE TO YOUR main.py AFTER THE OPTIMIZATION LOOP COMPLETES
# Replace the "Final results and plotting" section with this code
# ===================================================================

# Import the new state capture functionality
from optimal_wing_state import capture_and_save_optimal_state

print("\nOptimization completed!")

# === CAPTURE COMPLETE OPTIMAL STATE ===
for model_name, es in optimizers:
    print(f"\nFinal Results for {model_name}:")
    print(f"Best fitness: {es.result.fbest:.8f}")
    print(f"Total evaluations: {es.countevals}")
    print(f"Final sigma: {es.sigma:.6f}")
    print(f"Best parameters shape: {es.result.xbest.shape}")
    
    # Prepare optimization configuration for saving
    optimization_config = {
        'initial_sigma': initial_sigma,
        'max_epochs': max_epochs,
        'param_dimension': param_dimension,
        'bounds': {
            'lower': lower_bounds.tolist(),
            'upper': upper_bounds.tolist()
        },
        'cma_options': options.copy(),
        'population_size': options['popsize']
    }
    
    # Prepare optimization conditions
    optimization_conditions = {
        'alpha': 5,  # Your alpha value
        'Re': 1e6,   # Your Re value
        'model_size': "large",
        'wanted_lists': wanted_list,
        'importance_list': importance_list
    }
    
    # === CAPTURE AND SAVE COMPLETE STATE ===
    json_path, pickle_path, cst_path = capture_and_save_optimal_state(
        es=es,
        optimization_conditions=optimization_conditions,
        optimization_config=optimization_config,
        starting_guess=starting_guess_kulfan,
        optimization_log=optimization_log,
        results_dir=results_dir,
        model_name=model_name
    )
    
    # Save the traditional parameters file too (for backward compatibility)
    np.savetxt(
        os.path.join(results_dir, f"{model_name}_best_parameters.txt"), 
        es.result.xbest,
        header=f"Best CST parameters from {model_name} optimization\nFinal fitness: {es.result.fbest}"
    )
    
    # Store paths for easy access
    print(f"\n📁 Files saved:")
    print(f"   Complete state (JSON): {json_path}")
    print(f"   Complete state (Pickle): {pickle_path}")
    print(f"   CST only: {cst_path}")

# Continue with your existing plotting code
best_result = es.result.xbest
first_item = array_to_kulfan_dict(starting_guess_kulfan).copy()
cst_variations = [first_item]
best_airfoil = array_to_kulfan_dict(best_result)
print(best_airfoil)
cst_variations.append(best_airfoil)
print(cst_variations)
logging.debug(type(cst_variations), type(cst_variations[0]), type(cst_variations[1]))
display_auxiliary_functions.plot_multiple_cst_airfoils(cst_variations, labels=["starting guess", "best guess"], block=True)