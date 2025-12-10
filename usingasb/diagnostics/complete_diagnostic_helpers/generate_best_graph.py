import os # For saving and loading the filesimport json # Because the data that I'm using is a json 
from pathlib import Path
import sys
import shutil # Delete the files that are already in the folder 

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Helper functions that do the actual graphing
from plot_best import plot_best_fitness
from plot_sigma import plot_sigma

# Relative path of where to save all the data and where to find all the data
best_image_save_path = r"usingasb\diagnostics\Modal_Best"
information_location = r"usingasb\diagnostics\Modal_Optimization"
sigma_image_save_path = r"usingasb\diagnostics\Modal_Sigma"

def generate_best_graphs():
    # best graph pngs are saved here, the source directory is where the data is gotten
    output_dir = Path(best_image_save_path)
    source_dir = Path(information_location)
    shutil.rmtree(output_dir, ignore_errors=True)

    # Makes sure the data directory exists (Might need to guard against not having the right files)
    if not source_dir.exists():
        print(f"Info directory does not exist: {source_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Glob is used to match the files ending in .json in the source directory. It finds every json file in the directory
    for json_file in sorted(source_dir.glob("*.json")):
        target_file = output_dir / f"{json_file.stem}.png"
        plot_best_fitness(json_file, save_plot=True, save_path=target_file)
    print(f"Finished generating best fitness graphs. Saved to {output_dir}")

def generate_sigma_graphs():
    output_dir = Path(sigma_image_save_path)
    source_dir = Path(information_location)
    shutil.rmtree(output_dir, ignore_errors=True)

    if not source_dir.exists():
        print(f"Info directory does not exist: {source_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in sorted(source_dir.glob("*.json")):
        target_file = output_dir / f"{json_file.stem}.png"
        plot_sigma(json_file, save_plot=True, save_path=target_file)
    
    print(f"Finished generating sigma graphs. Saved to {output_dir}")

def generate_all_graphs():
    generate_best_graphs()
    generate_sigma_graphs()

if __name__ == "__main__":
    generate_all_graphs()