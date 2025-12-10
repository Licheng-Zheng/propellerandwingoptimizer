import os
import modal
import shutil

output_directory = os.path.join("usingasb", "diagnostics", "Modal_Optimization")
def clear_folder(folder_path):
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

def download_results_local(download_folder:str):
    os.makedirs(output_directory, exist_ok=True)

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
        local_filename = f"{os.path.basename(folder.path)}_optimization_log.json"
        local_path = os.path.join(output_directory, local_filename)

        try:
            with open(local_path, "wb") as dst:
                for chunk in vol.read_file(log_path):
                    dst.write(chunk)
            # print(f"Downloaded {log_path} → {local_path}")
        except Exception as e:
            print(f"Skipping {folder.path}: {e}")

    print(f"Finished downloading {len(cmaes_folders)} logs to {output_directory}")

def external_collect_call(download_folder: str):
    clear_folder(output_directory)
    download_results_local(download_folder=download_folder) 
    

if __name__ == "__main__":
    clear_folder(output_directory)
    download_results_local(download_folder = "optimization_results_20251206_204801")
    