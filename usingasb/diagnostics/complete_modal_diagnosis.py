import os
import csv
import json
from complete_diagnostic_helpers import run_collector, generate_best_graph
from complete_diagnostic_helpers import *

if __name__ == "__main__":
    download_folder = "optimization_results_20251206_204801"
    run_collector.external_collect_call(download_folder=download_folder)
    generate_best_graph.generate_all_graphs()

    # Doesn't work yet, not sure why gotta fix it 
    # create_sigma_score_csv(output_directory="usingasb\diagnostics\Modal_Optimization")