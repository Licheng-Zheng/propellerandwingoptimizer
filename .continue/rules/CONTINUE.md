# CONTINUE Project Guide

## 1. Project Overview
- **Purpose:** Optimizes airfoils for wings (propellers in the far future I hope) using CMA-ES optimization and NeuralFoil evaluation, with tooling for constraint enforcement, logging, visualization, and state capture.
- **Key Technologies:** Python 3, [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox), [NeuralFoil](https://github.com/daniel-de-vries/neuralfoil), [cma](https://github.com/CMA-ES/pycma), NumPy, Matplotlib, optional [Modal](https://modal.com/) for cloud execution.
- **Architecture:**
  - `usingasb/main.py` calls single or multi-process CMA-ES runs.
  - Objective evaluation (`objective.py`) uses NeuralFoil scoring with soft/hard constraint builders (`constraints/`, hard and soft constraints).
  - CMA-ES loop  (`cma_optimization.py`) shared helpers for conversions, logging, display, and state capture sit in dedicated files/modules.
  - Diagnostics (`diagnostics/`) and quick testing (`quick_test_framework.py`) provide targeted tooling.
  - (`using_modal.py`) packages the workflow for Modal-hosted execution.

## 2. Getting Started
### Prerequisites
- Python **3.10+** 
- Ability to compile/install CasADi (bundled with requirements; Windows wheels available).
- Git, virtual environment tooling (venv, Conda, etc.).
- Optional: Modal account & CLI auth if using `using_modal.py` (it's just creating an environment variable).

### Installation
1. Clone the repository.
2. Create & activate a virtual environment.
3. Install dependencies:
   ```powershell
   pip install -r usingasb/requirements.txt
   ```
4. NeuralFoil may download model weights on first use—ensure network access.
5. **Update `RESULTS_BASE_DIR`** in `usingasb/PARAMETERS.py` to a valid path on your machine before running (It should work now because I changed it to a relative path, but until someone tries using it, I guess I'll never know).

### Basic Usage
- Run a multi-process optimization (default half the cores, max 4):
  ```powershell
  python -m usingasb.main
  ```
- Run a specific number of parallel instances with an interactive dry run first:
  ```powershell
  python -m usingasb.main --num_instances 3 --interactive_first True
  ```
  *(Pass kwargs by editing the `run_multiprocess_optimizations` call in `__main__`; no CLI parser yet.)*
- Execute a single run for debugging:
  ```powershell
  python -c "from usingasb.main import run_single_cma; run_single_cma(run_id=1, parent_results_dir='outputs')"
  ```
- Inspect saved states quickly:
  ```powershell
  python -m usingasb.quick_test_framework --state path\to\state.json --visualize
  ```

### Running Diagnostics / Tests
There is no automated test suite, but you can validate constraints and objectives with:
```powershell
python -m usingasb.diagnostics.constraint_diagnostic
```
Set the `MPLBACKEND` env var (e.g. to `Agg`) if running headless.

## 3. Project Structure
```
using_modal.py                 # Modal integration entrypoint
usingasb/
  __init__.py
  PARAMETERS.py                # Central optimization & environment constants
  main.py                      # Multiprocessing-friendly orchestration
  cma_optimization.py          # Single CMA-ES ask/tell step
  objective.py                 # Fitness, scoring, constraint suites
  constraints/                 # Hard & soft constraint implementations
  convertion_auxiliary_functions.py
  logging_auxiliary_functions.py
  display*.py                  # Visualization helpers
  optimal_wing_state.py        # Comprehensive result capture & loaders
  quick_test_framework.py      # CLI utilities for saved states
  diagnostics/                 # Constraint/objective debugging scripts
  requirements.txt             # Python dependency pinning
media/, temp/, etc.            # Reference assets & experimental scripts (not core)
```
*Confirm optional directories (`media`, `temp`) in your local tree; contents may vary.*

## 4. Development Workflow
- **Coding Style:** No formal linting enforced; follow standard Python style (PEP 8) and keep modules cohesive (e.g., constraints separated by type).
- **Configuration:** Adjust optimization behavior via `PARAMETERS.py` (sigma, epoch cap, weighting lists, results directory).
- **Execution Paths:**
  - Local multiprocessing via `main.py` (uses `spawn` start method on Windows).
  - Modal cloud runs via `using_modal.py` (requires Modal vol/image config and environment variables/API keys).
  - To run modal, use this in terminal: `modal run using_modal.py::launch_many --num-instances 10`
- **Testing:** Use diagnostics scripts to vet constraint logic before long CMA-ES runs.
- **Result Capture:** `optimal_wing_state.capture_and_save_optimal_state` writes JSON/Pickle/CST snapshots per run
- **Contribution Tips:**
  - Keep new utilities in dedicated modules (mirroring existing auxiliary structure).
  - Document new parameters in `PARAMETERS.py` docstrings/comments.
  - When editing multiprocessing logic, maintain the `if __name__ == "__main__"` guard for Windows compatibility.

## 5. Key Concepts
- **CST / Kulfan Parameters:** Parametric airfoil representation; conversions handled in `convertion_auxiliary_functions.py`.
- **NeuralFoil:** Surrogate aerodynamic evaluator producing CL/CD/CM/etc. for given Kulfan parameters, alpha, Re, and model size.
- **CMA-ES:** Evolutionary strategy used for optimization (`cma` package). `run_single_cma` handles strategy lifecycle.
- **Hard vs. Soft Constraints:** Hard constraints immediately reject candidates; soft constraints apply fitness penalties. Constraint factories/builders in `objective.py` manage suites.
- **Wanted & Importance Lists:** Determine objective weighting; modify in `PARAMETERS.py` or pass custom lists into `run_cma_optimization` calls.
- **OptimalWingState:** Object that snapshots optimization results, enabling post-run analysis and scenario testing.
- **RESULTS_BASE_DIR:** Absolute base directory for storing artifacts—must exist and be writable.

## 6. Common Tasks
1. **Adjust Optimization Targets**
   - Edit `WANTED_LIST` / `IMPORTANCE_LIST` in `PARAMETERS.py`.
   - (Optional) Extend `scoring_model_1` for new metrics after confirming NeuralFoil outputs them.

2. **Tune Constraint Behavior**
   - Modify constraint builders in `objective.py` (e.g., adjust thickness thresholds).
   - Add new constraint modules under `usingasb/constraints/` and register them in `ConstraintFactory`.

3. **Run a Batch and Review Results**
   - Execute `python -m usingasb.main` (adjust `num_instances`).
   - Inspect each run folder under `RESULTS_BASE_DIR/optimization_results_*` for logs, plots, and state dumps.
   - Use `logging_auxiliary_functions.plot_optimization_results` to generate summary figures.

4. **Replay / Stress-Test a Saved Wing**
   - Launch quick tester:
     ```powershell
     python -m usingasb.quick_test_framework --state path\to\cmaes_1_complete_state.json --test-alpha 6 --test-re 1500000 --visualize
     ```
   - Review new CL/CD/CM outputs and constraint penalties.

5. **Run on Modal**
   - Install `modal-client` separately if not already present.
   - Configure Modal secrets & volume (`APP_NAME`, `VOLUME_NAME`).
   - Execute locally:
     ```powershell
     python using_modal.py
     ```
   - Results sync to the Modal volume (`/data`).

6. **Visualize CST Shapes**
   - Use `display_auxiliary_functions.plot_cst_airfoil` or `plot_multiple_cst_airfoils` with saved parameter dictionaries for comparative plots.

## 7. Troubleshooting
- **`ModuleNotFoundError: neuralfoil`** — Ensure dependencies installed in the active environment; NeuralFoil may require Rust toolchain on unsupported platforms.
- **`PermissionError` when saving results** — Confirm `RESULTS_BASE_DIR` exists and is writable; adjust to a relative path (e.g., `Path(__file__).parent / 'results'`).
- **`RuntimeError: context has already been set`** — Avoid manually changing multiprocessing start method elsewhere; `main.py` already sets `spawn` under `__main__`.
- **Matplotlib backend errors on headless runs** — Set `MPLBACKEND=Agg` or follow the pattern in `diagnostics/constraint_diagnostic.py`.
- **NeuralFoil returns extreme values / NaNs** — Often due to invalid CST geometry; check constraint logs, use diagnostics to visualize surfaces, or reduce `INITIAL_SIGMA`.
- **Modal run crashes** — Inspect `/data/.../crash_log.txt`; ensure Modal volume has enough space and dependencies match `usingasb/requirements.txt`.

## 8. References & Resources
- [AeroSandbox Documentation](https://peterdsharpe.github.io/AeroSandbox/)
- [NeuralFoil Project](https://github.com/daniel-de-vries/neuralfoil)
- [PyCMA Documentation](https://cma-es.github.io/apidocs-pycma)
- [Modal Docs](https://modal.com/docs) *(required for `using_modal.py` flows)*
- [Blade Element Momentum Theory overview](https://en.wikipedia.org/wiki/Blade_element_theory) *(context for future integration)*

> **Next Steps:** Review and tailor this guide for your team’s processes, commit it to version control, and consider adding additional `.continue/rules/*.md` files for subsystem-specific conventions (e.g., `constraints/rules.md`).
