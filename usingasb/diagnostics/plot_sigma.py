import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_sigma(file_name, save_plot=False, save_path=None):
    try:
        data_path = Path(file_name)
        with open(data_path, 'r') as f:
            data = json.load(f)

        plot_data = {}
        for key, run_data in data.items():
            if 'epochs' in run_data and 'sigma_per_epoch' in run_data:
                plot_data[key] = {
                    'epochs': run_data['epochs'],
                    'sigma': run_data['sigma_per_epoch']
                }

        if not plot_data:
            print("Error: No valid run data (epochs and sigma_per_epoch) found in the JSON file.")
            return

        plt.figure(figsize=(12, 7))

        for run_name, d in plot_data.items():
            epochs = d['epochs']
            sigma = d['sigma']
            plt.plot(epochs, sigma, label=rf'{run_name} $\sigma$') 

        plt.title(r'CMA-ES Step-Size ($\sigma$) Across Generations', fontsize=16) 
        plt.xlabel('Generation (Epoch)', fontsize=14)
        plt.ylabel(r'Step-Size ($\sigma$)', fontsize=14)

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='CMA-ES Run', loc='best')
        plt.autoscale(enable=True, axis='y', tight=True)

        if save_plot:
            suffix = data_path.stem[-2:] if len(data_path.stem) >= 2 else data_path.stem
            default_filename = f'sigma_across_generations_{suffix}.png'
            output_path = Path(default_filename)

            if save_path:
                candidate = Path(save_path)
                if candidate.suffix.lower() == '.png':
                    output_path = candidate
                else:
                    output_path = candidate / default_filename

            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            # print(f"Graph successfully saved as {output_path}")

            plt.close()

        if not save_plot:
            plt.show()

    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_name}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")