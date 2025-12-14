import os
import modal
import shutil
import json

# where everything is saved
output_directory_sigma_and_best = os.path.join("usingasb", "diagnostics", "Modal_Optimization")
output_directory_display = os.path.join("usingasb", "diagnostics", "Modal_Display_Info")

def clear_folder(folder_path):
    """
    Clears all contents of the specified folder. If the folder does not exist, it creates it.

    Args:
        folder_path (string): Path to the folder to be cleared.
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)        # remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)    # remove directory
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        os.makedirs(folder_path, exist_ok=True)

app = modal.App("propellerdesign-cmaes")
vol = modal.Volume.from_name("propeller-results-vol")

def txt_to_json(txt_filepath, json_filepath):
    with open(txt_filepath, 'r') as f:
        lines = f.readlines()
    # Take last 18 lines (skip first 2 lines)
    cst_lines = lines[-18:]
    # Parse floats
    cst_parameters = [float(line.strip()) for line in cst_lines]
    # Create dict
    output = {
        "cst_parameters": cst_parameters
    }
    # Write to json file
    with open(json_filepath, 'w') as f:
        json.dump(output, f, indent=2)

def download_results_local(download_folder:str):
    """
    The function that retrieves all the data from Modal volume and puts it into the related folder

    Args:
        download_folder (str): The folder where everything should be downloaded into
    """
    os.makedirs(output_directory_sigma_and_best, exist_ok=True)
    os.makedirs(output_directory_display, exist_ok=True)

    root_folder = download_folder

    # List all entries in the root folder
    entries = vol.listdir(root_folder)

    # Get only CMA-ES folders by path name
    cmaes_folders = [entry for entry in entries if "cmaes_" in entry.path]

    if not cmaes_folders:
        print("No CMA-ES folders found.")
        return

    for folder in cmaes_folders:
        log_path = f"{folder.path}/optimization_log.json"
        last_chunk = folder.path[folder.path.rfind("/") + 1:]
        best_parameters_remote_path = f"{folder.path}/{last_chunk}_best_parameters.txt"

        local_log_filename = f"{os.path.basename(folder.path)}_optimization_log.json"
        local_display_json_filename = f"{os.path.basename(folder.path)}_best_parameters.json"
        
        local_log_path = os.path.join(output_directory_sigma_and_best, local_log_filename)
        local_display_json_path = os.path.join(output_directory_display, local_display_json_filename)

        try:
            # Download optimization_log.json
            with open(local_log_path, "wb") as dst:
                for chunk in vol.read_file(log_path):
                    dst.write(chunk)

            # Download best_parameters.txt into memory buffer
            content_bytes = b""
            for chunk in vol.read_file(best_parameters_remote_path):
                content_bytes += chunk

            content_str = content_bytes.decode("utf-8")
            lines = content_str.splitlines()
            # Process lines directly without saving txt file
            cst_lines = lines[-18:]
            cst_parameters = [float(line.strip()) for line in cst_lines]
            output = {
                "cst_parameters": cst_parameters
            }
            with open(local_display_json_path, 'w') as f:
                json.dump(output, f, indent=2)

        except Exception as e:
            print(f"Skipping {folder.path}: {e}")

    print(f"Finished downloading {len(cmaes_folders)} logs to {output_directory_sigma_and_best} and displays information to {output_directory_display}")

def external_collect_call(download_folder: str):
    clear_folder(output_directory_sigma_and_best)
    clear_folder(output_directory_display)

    download_results_local(download_folder=download_folder) 
    

if __name__ == "__main__":
    clear_folder(output_directory_sigma_and_best)
    clear_folder(output_directory_display)
    download_results_local(download_folder = "optimization_results_20251206_204801")