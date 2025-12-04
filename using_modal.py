"""
Modal wrapper to run usingasb.main optimization in parallel.
"""
from __future__ import annotations

import sys
import os

# Used to tell python which files to send to Modal. The entire folder "usingasb" is sent (other than a few larger files)
# The usingasb folder is in the root of this file, so we combine the path together to get the path to the folder!
sys.path.append(os.path.join(os.path.dirname(__file__), "usingasb"))


from datetime import datetime
from typing import Optional

# run_single_cma is the only function we need (it calls other functions), because we can chain a bunch of them to parallize
from usingasb.main import run_single_cma
from usingasb.PARAMETERS import RESULTS_BASE_DIR

# You need to have a modal environment variable (or an API key also works, but its easy to do either)
# This file is only for running with modal, so if there's no modal, we end it all 
try:
    import modal
except Exception:
    print("modal is not available") 
    sys.exit(0) 

# Name the project name that it is stored under for Modal and the "volume" which is just where the data is stored
# I did a bit of rough searching and I think its called the volume to refer to a physical entity where data is stored
APP_NAME = "propellerdesign-cmaes"
VOLUME_NAME = "propeller-results-vol"

# Creating the image, essentially create the environments with everything required for the program to run
image = (
    modal.Image.debian_slim()
    .apt_install("git")

    # Apparently, splitting up the heavy imports and the light ones helps with start up time, because Modal saves a snapshot
    # Putting the heavy ones together that I rarely change means I don't need to reimport as frequently
    .pip_install(
        "aerosandbox>=4.2.8",
        "neuralfoil>=0.3.2",
        "cma>=4.2.0",
        "casadi>=3.7.0",
    )

    .pip_install(
        "matplotlib>=3.10.3",
        "numpy"
    )

    # All the local files that I am going to need, but avoids the big files (which are not required to run)
    .add_local_dir(
        local_path="usingasb",
        remote_path="/root/usingasb",
        ignore=[
            "*.csv", "*.stl", "*.png", "*.log", 
            "__pycache__", ".git",
            "archive/", "logs/", "testing/"
        ]
    )
)

# Sets everything up in Modal! Almost time to run.
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

def _default_parent_results_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"/data/optimization_results_{timestamp}"


# The Worker (The Slave)
@app.function(
    image = image, # Provide each worker what they need to run
    volumes={"/data": vol},   # Where data is saved
    timeout=60 * 60 * 6, # No clue what this is for
    cpu=1.0, # CPUs given to each worker (each container, can also be fractions)
    memory=200 # The amount of memory that you are allowing a single worker to use in MB
)
def worker(run_id: int, parent_results_dir: str) -> dict:
    os.makedirs(parent_results_dir, exist_ok=True)
    summary_holder: list[dict] = []

    class _ListQueue:
        def put(self, item): summary_holder.append(item)

    try:
        # Run the optimization algorithm
        run_single_cma(
            run_id=run_id,
            parent_results_dir=parent_results_dir,
            interactive=False,
            results_queue=_ListQueue(),
        )
    except Exception as e:
        # Not sure if this works yet because its never crashed on me, and I don't want to make it crash because then it'll never work again
        print(f"❌ Worker {run_id} crashed: {e}")
        with open(f"{parent_results_dir}/crash_log.txt", "w") as f:
            f.write(str(e))
        raise e  
    finally:
        # Save even if the process fails so we still get a little data to work with
        print(f"💾 Committing volume for Worker {run_id}...")
        vol.commit()

# The Launcher (The Boss) 
@app.local_entrypoint()
# Launches a bunch of workers/containers 
def launch_many(num_instances: Optional[int] = None) -> None:
    if num_instances is None: num_instances = 10 
    
    parent_results_dir = _default_parent_results_dir()
    print(f"{num_instances} workers")
    
    # Spawn workers, its called futures because the list will one day be filled with results
    futures = []
    for i in range(1, num_instances + 1):
        futures.append(worker.spawn(i, parent_results_dir))

    # Wait for results
    results = [f.get() for f in futures]

    # Print Summary
    if results:
        print("\nSummary of Modal runs:")
        valid = [r for r in results if r.get("best_fitness") is not None]
        valid.sort(key=lambda x: x["best_fitness"]) if valid else None
        
        if valid:
            best = valid[0]
            print(f"\n🏆 Best overall: {best['model_name']} with fitness {best['best_fitness']}")
            print(f"   Files located in Volume: {best['results_dir']}")
    else:
        print("No results returned.")

if __name__ == "__main__":
    with app.run():
        launch_many()