import json 
import os 
import csv

# Two file paths, one contains the optimization log data, the other points to where we want the csv stored
csv_file_path2 = r"C:\Users\liche\OneDrive\Desktop\PycharmProjects\PropellerDesign\usingasb\diagnostics\Modal_Optimization\cmaes_30_optimization_log.json"
csv_file_path = r"C:\Users\liche\OneDrive\Desktop\PycharmProjects\PropellerDesign\usingasb\diagnostics\Modal_Optimization\sigma_best_scores.csv"

# Open all the required files, and opens the other file to read the contents that we want written
with open(csv_file_path, mode='w', newline='') as csv_file:
    with open(csv_file_path2, 'r') as f:

        # loads the dictionary
        data = json.load(f)

        # The list where we want to store all the data
        results = []

        # searches for the best fitness and sigma in the dictionary and adds it to results
        for key, run_data in data.items():
            if 'convergence_data' in run_data:
                convergence_data = run_data['convergence_data']
                for entry in convergence_data:
                    sigma = entry.get('sigma')
                    best_fitness = entry.get('best_fitness')

                    # Adds all the parameters to the list
                    if sigma is not None and best_fitness is not None:
                        results.append((sigma, best_fitness))

    writer = csv.writer(csv_file)
    writer.writerow(['Sigma', 'Best Score'])  # Write header
    writer.writerows(results)  # Write data rows

