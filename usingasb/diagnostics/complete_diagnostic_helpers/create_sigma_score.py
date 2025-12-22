import os
import csv
import json
from complete_diagnostic_helpers import run_collector

def create_sigma_score_csv(output_directory, csv_filename='sigma_best_scores.csv'):
    csv_file_path = os.path.join(output_directory, csv_filename)

    # Collect results from the Modal_Optimization folder
    results = []
    for filename in os.listdir(output_directory):
        if filename.endswith('_optimization_log.json'):  # Assuming results are stored in these JSON files
            file_path = os.path.join(output_directory, filename)
            with open(file_path) as file:
                data = json.load(file)
                convergence = data.get("convergence_data", [])
                for entry in convergence:
                    sigma = entry.get("sigma") or entry.get("strategy_sigma")
                    best_score = (
                        entry.get("best_score")
                        or entry.get("best_fitness")
                        or entry.get("best")
                    )
                    if sigma is not None and best_score is not None:
                        results.append([sigma, best_score])    # Write to CSV
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Sigma', 'Best Score'])  # Write header
        writer.writerows(results)  # Write data rows

    print(f"Created CSV file: {csv_file_path}")