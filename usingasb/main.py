import os
import json
import logging
from datetime import datetime
import multiprocessing as mp
from multiprocessing.queues import Queue as MPQueue

import numpy as np
import matplotlib.pyplot as plt
import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
import cma
from typing import Optional 

from convertion_auxiliary_functions import kulfan_dict_to_array, array_to_kulfan_dict
from cma_optimization import run_cma_optimization
from logging_auxiliary_functions import save_optimization_log, save_intermediate_results, plot_optimization_results, stop_functioning
import display_auxiliary_functions
from optimal_wing_state import capture_and_save_optimal_state

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Global optimization settings (shared by all processes)
INITIAL_SIGMA = 0.3
MAX_EPOCHS = 100
WANTED_LIST = ["analysis_confidence", "CL", "CD", "CM"]
IMPORTANCE_LIST = [0.4, 0.3, -0.2, -0.1]
ALPHA = 5
RE = 1e6
MODEL_SIZE = "large"

# Base results directory
RESULTS_BASE_DIR = r"C:\Users\liche\OneDrive\Desktop\PycharmProjects\PropellerDesign\usingasb\Optimization Results"


# -----------------------------------------------------------------------------
# Helper builders
# -----------------------------------------------------------------------------

def build_starting_guess() -> np.ndarray:
    """Build the starting CST parameter array using a default airfoil."""
    airfoil = asb.Airfoil("naca4412")
    starting_guess_kulfan = get_kulfan_parameters(airfoil.coordinates)
    starting_guess_kulfan = kulfan_dict_to_array(starting_guess_kulfan)
    logging.debug("Starting CST parameters: %s", starting_guess_kulfan)
    return starting_guess_kulfan


def build_cma_options(param_dimension: int, initial_sigma: float, seed: int) -> dict:
    lower_bounds = np.full(param_dimension, -1.0)
    upper_bounds = np.full(param_dimension, 1.0)

    options = {
        'bounds': [lower_bounds, upper_bounds],
        'popsize': 60,
        'maxiter': 200,
        'maxfevals': 10000,
        'tolfun': 1e-8,
        'tolx': 1e-10,
        'verb_disp': 10,
        'verb_log': 1,
        'CMA_stds': [initial_sigma] * param_dimension,
        'seed': seed,
    }
    return options


# -----------------------------------------------------------------------------
# Core single-run optimization (safe for multiprocessing)
# -----------------------------------------------------------------------------

def run_single_cma(run_id: int,
                   parent_results_dir: str,
                   interactive: bool = False,
                   results_queue: Optional[mp.Queue] = None) -> None:
    """
    Run a single CMA-ES optimization instance. Designed to be executed inside a
    separate process (Windows-safe). Saves all artifacts to a unique directory.

    Args:
        run_id: Unique identifier for this run (used for naming).
        parent_results_dir: Parent directory where a per-run folder will be created.
        interactive: If True, enables blocking prompts/plots. Should be False in multiprocessing.
        results_queue: Optional multiprocessing queue to return a summary.
    """
    # Basic per-process logging
    logging.basicConfig(level=logging.INFO, format=f'[PID {os.getpid()}] %(asctime)s - %(levelname)s - %(message)s')

    # Prepare results directory for this run
    model_name = f"cmaes_{run_id}"
    results_dir = os.path.join(parent_results_dir, model_name)
    os.makedirs(results_dir, exist_ok=True)

    # Build starting guess
    starting_guess_kulfan = build_starting_guess()

    # Optional interactive pause and plot (disabled in multiprocessing)
    if interactive:
        try:
            stop_functioning(
                "These are the current starting CST parameters and the drawn airfoil",
                additional_info_context="CST parameters",
                additional_info=starting_guess_kulfan,
                cst_parameters=starting_guess_kulfan,
            )
        except Exception as e:
            logging.warning(f"Interactive step failed or skipped: {e}")

    # Set up CMA-ES options
    param_dimension = len(starting_guess_kulfan)
    options = build_cma_options(param_dimension, INITIAL_SIGMA, seed=12345 + run_id)

    # Instantiate CMA-ES
    strategy = cma.CMAEvolutionStrategy(starting_guess_kulfan, INITIAL_SIGMA, options)

    # Prepare logging structure
    optimization_log = {
        model_name: {
            'epochs': [],
            'best_fitness_per_epoch': [],
            'sigma_per_epoch': [],
            'best_parameters_per_epoch': [],
            'all_fitness_values': [],
            'convergence_data': []
        }
    }

    # Optimization loop
    best_result = None
    for epoch in range(MAX_EPOCHS):
        if strategy.stop():
            logging.info(f"{model_name}: Early stop at epoch {epoch + 1}")
            break

        best_fitness = run_cma_optimization(
            evolutionary_strategy=strategy,
            model_name=model_name,
            model_size=MODEL_SIZE,
            alpha=ALPHA,
            Re=RE,
            epoch=epoch,
            wanted_lists=WANTED_LIST,
            importance_list=IMPORTANCE_LIST,
        )

        # Log epoch data
        optimization_log[model_name]['epochs'].append(epoch + 1)
        optimization_log[model_name]['best_fitness_per_epoch'].append(best_fitness)
        optimization_log[model_name]['sigma_per_epoch'].append(strategy.sigma)
        optimization_log[model_name]['best_parameters_per_epoch'].append(strategy.result.xbest.copy())

        convergence_info = {
            'iteration': len(optimization_log[model_name]['epochs']),
            'evaluations': strategy.countevals,
            'best_fitness': float(strategy.result.fbest) if strategy.result.fbest is not None else float(best_fitness),
            'sigma': float(strategy.sigma),
            'condition_number': float(strategy.D.max() / strategy.D.min()) if hasattr(strategy, 'D') else None
        }
        optimization_log[model_name]['convergence_data'].append(convergence_info)

        print(f"{model_name}: Epoch {epoch + 1}, Best: {best_fitness:.6f}, Sigma: {strategy.sigma:.6f}, Evals: {strategy.countevals}")

        best_result = strategy.result.xbest

        # Optionally save intermediate results
        # if (epoch + 1) % 10 == 0:
        #     save_intermediate_results(optimization_log, results_dir, epoch + 1)

    # Prepare visualization (interactive only)
    if interactive and best_result is not None:
        first_item = array_to_kulfan_dict(starting_guess_kulfan).copy()
        cst_variations = [first_item]
        best_airfoil = array_to_kulfan_dict(best_result)
        cst_variations.append(best_airfoil)
        display_auxiliary_functions.plot_multiple_cst_airfoils(
            cst_variations, labels=["starting guess", "best guess"], block=True
        )

    print("\nOptimization completed!")

    # Final results and saving
    if best_result is not None:
        # Report
        print(f"\nFinal Results for {model_name}:")
        print(f"Best fitness: {strategy.result.fbest:.8f}")
        print(f"Total evaluations: {strategy.countevals}")
        print(f"Final sigma: {strategy.sigma:.6f}")
        print(f"Best parameters shape: {strategy.result.xbest.shape}")

        # Configs to persist
        lower_bounds = np.full(param_dimension, -1.0)
        upper_bounds = np.full(param_dimension, 1.0)
        optimization_config = {
            'initial_sigma': INITIAL_SIGMA,
            'max_epochs': MAX_EPOCHS,
            'param_dimension': param_dimension,
            'bounds': {
                'lower': lower_bounds.tolist(),
                'upper': upper_bounds.tolist()
            },
            'cma_options': options.copy(),
            'population_size': options['popsize']
        }

        optimization_conditions = {
            'alpha': ALPHA,
            'Re': RE,
            'model_size': MODEL_SIZE,
            'wanted_lists': WANTED_LIST,
            'importance_list': IMPORTANCE_LIST
        }

        # Save complete state
        json_path, pickle_path, cst_path = capture_and_save_optimal_state(
            es=strategy,
            optimization_conditions=optimization_conditions,
            optimization_config=optimization_config,
            starting_guess=starting_guess_kulfan,
            optimization_log=optimization_log,
            results_dir=results_dir,
            model_name=model_name
        )

        # Save best parameters
        np.savetxt(
            os.path.join(results_dir, f"{model_name}_best_parameters.txt"),
            strategy.result.xbest,
            header=f"Best CST parameters from {model_name} optimization\nFinal fitness: {strategy.result.fbest}"
        )

        # Optional global optimization log per run
        save_optimization_log(optimization_log, results_dir)

        # Return a short summary via queue
        if results_queue is not None:
            results_queue.put({
                'model_name': model_name,
                'best_fitness': float(strategy.result.fbest),
                'evaluations': int(strategy.countevals),
                'results_dir': results_dir,
            })


# -----------------------------------------------------------------------------
# Multi-process launcher
# -----------------------------------------------------------------------------

def run_multiprocess_optimizations(num_instances: int | None = None, interactive_first: bool = False) -> None:
    """
    Launch multiple CMA-ES optimizations in parallel processes. Each process runs a
    full independent optimization and writes to its own results directory.

    Args:
        num_instances: Number of parallel CMA-ES runs. If None, decide from CPU count.
        interactive_first: If True, run the first instance interactively in the main process
                           (useful for quick visual sanity check) and others in parallel.
    """
    cpu_count = os.cpu_count() or 2
    if num_instances is None:
        # Conservative default: use up to half the cores, capped at 4
        num_instances = max(1, min(cpu_count // 2, 4))

    # Prepare parent results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_results_dir = os.path.join(RESULTS_BASE_DIR, f"optimization_results_{timestamp}")
    os.makedirs(parent_results_dir, exist_ok=True)

    print(f"Launching {num_instances} parallel CMA-ES runs on {cpu_count} CPU cores.")
    print(f"Results will be saved under: {parent_results_dir}")

    # Optional: run one interactive instance in the foreground first
    started_ids = []
    if interactive_first and num_instances > 0:
        run_single_cma(run_id=1, parent_results_dir=parent_results_dir, interactive=True, results_queue=None)
        started_ids.append(1)

    # Multiprocessing for remaining runs
    manager = mp.Manager()
    results_queue = manager.Queue()

    processes: list[mp.Process] = []
    for i in range(1, num_instances + 1):
        if i in started_ids:
            continue
        p = mp.Process(target=run_single_cma, args=(i, parent_results_dir, False, results_queue))
        processes.append(p)
        p.start()

    # Wait for completion
    for p in processes:
        p.join()

    # Collect summaries
    summaries = []
    while not results_queue.empty():
        try:
            summaries.append(results_queue.get_nowait())
        except Exception:
            break

    # Print summary
    if summaries:
        print("\nSummary of parallel runs:")
        summaries.sort(key=lambda x: x['best_fitness'])
        for s in summaries:
            print(f" - {s['model_name']}: best_fitness={s['best_fitness']:.6f}, evals={s['evaluations']}, dir={s['results_dir']}")
        best = summaries[0]
        print(f"\nBest overall: {best['model_name']} with fitness {best['best_fitness']:.6f}")


# -----------------------------------------------------------------------------
# Entrypoint (Windows-safe)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Windows requires spawn and freeze_support for multiprocessing
    try:
        mp.freeze_support()
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass

    # Default: run multiple instances based on system resources
    run_multiprocess_optimizations(num_instances=None, interactive_first=False)
