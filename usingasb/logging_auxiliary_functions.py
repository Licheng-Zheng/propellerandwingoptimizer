import matplotlib.pyplot as plt 
import json 
import os 
import numpy as np
import aerosandbox as asb
import convertion_auxiliary_functions
import display_auxiliary_functions 

# Save complete optimization log
def save_optimization_log(log_data, directory):
    """Save optimization log to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    json_log = {}
    for model_name, data in log_data.items():
        json_log[model_name] = {}
        for key, value in data.items():
            if key == 'best_parameters_per_epoch':
                json_log[model_name][key] = [param.tolist() if hasattr(param, 'tolist') else param for param in value]
            else:
                json_log[model_name][key] = value
    
    with open(os.path.join(directory, 'optimization_log.json'), 'w') as f:
        json.dump(json_log, f, indent=2)
    
    print(f"Optimization log saved to {directory}/optimization_log.json")

def save_intermediate_results(log_data, directory, epoch):
    """Save intermediate results"""
    filename = os.path.join(directory, f'intermediate_results_epoch_{epoch}.json')
    save_optimization_log(log_data, directory)
    print(f"Intermediate results saved at epoch {epoch}")

def plot_optimization_results(log_data, save_dir):
    """Create comprehensive optimization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (model_name, data) in enumerate(log_data.items()):
        color = colors[i % len(colors)]
        epochs = data['epochs']
        
        # Best fitness evolution
        axes[0, 0].plot(epochs, data['best_fitness_per_epoch'], 
                       label=model_name, color=color, linewidth=2)
        axes[0, 0].set_title('Best Fitness Evolution')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Best Fitness')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Sigma evolution
        axes[0, 1].plot(epochs, data['sigma_per_epoch'], 
                       label=model_name, color=color, linewidth=2)
        axes[0, 1].set_title('Step Size (Sigma) Evolution')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Sigma')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Parameter evolution (show first few parameters)
        if data['best_parameters_per_epoch']:
            params = np.array(data['best_parameters_per_epoch'])
            for j in range(min(3, params.shape[1])):  # Show first 3 parameters
                axes[1, 0].plot(epochs, params[:, j], 
                               label=f'{model_name}_param_{j}', 
                               alpha=0.7, linewidth=1)
        
        axes[1, 0].set_title('Parameter Evolution (First 3)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Parameter Value')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Convergence rate
        if len(data['best_fitness_per_epoch']) > 1:
            fitness_diff = np.diff(data['best_fitness_per_epoch'])
            axes[1, 1].plot(epochs[1:], -fitness_diff, 
                           label=f'{model_name}_improvement', 
                           color=color, alpha=0.7)
    
    axes[1, 1].set_title('Fitness Improvement per Epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Improvement')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimization_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate individual CMA-ES plots for each optimizer
    for model_name, es in optimizers:
        try:
            plt.figure(figsize=(12, 8))
            es.logger.plot()
            plt.suptitle(f'CMA-ES Convergence Details - {model_name}')
            plt.savefig(os.path.join(save_dir, f'{model_name}_cma_details.png'), dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not generate detailed CMA plot for {model_name}: {e}")

def stop_functioning(message: str, **kwargs):
    """
    Stops the program execution at certain points if the current execution is not satisfactory

    Exits the program if the user does not confirm to continue, preventing wasting computational resources. More logic can be added later to restart the process rather than just ending it all.

    Args:
        message (str): the message you want to display to the user
        **kwargs: additional arguments that can be used to customize the behavior
    """
    additional_info_context = kwargs.get('additional_info_context', None)
    additional_info = kwargs.get('additional_info', None)

    if kwargs.get('cst_parameters', None) is not None:
        cst_parameters = convertion_auxiliary_functions.array_to_kulfan_dict(kwargs.get('cst_parameters'))

        print(cst_parameters)

        display_auxiliary_functions.plot_cst_airfoil(cst_params=cst_parameters, n_points_per_side=200, title=additional_info_context, show_params=True, show=True)

    to_begin = input(f"{message} \n {additional_info_context}: \n {additional_info} \n Continue? (yes/no): ").strip().lower()

    if to_begin != 'yes':
        print("Exiting program.")
        exit(0)

    plt.close('all')
