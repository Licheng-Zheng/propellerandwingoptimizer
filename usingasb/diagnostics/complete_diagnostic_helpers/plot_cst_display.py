import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def display_and_save_cst_airfoil(cst_params, save_path, n_points_per_side=200, title="CST Airfoil Display"):
    """Plots CST airfoil and saves it to a file instead of showing."""
    params = np.asarray(cst_params, dtype=float).ravel()
    if params.size < 4:
        raise ValueError("cst_params must include lower, upper weights and two scalars (LE weight, TE thickness).")

    if (params.size - 2) % 2 != 0:
        raise ValueError("Length of cst_params minus two scalars must be even (same number of lower/upper weights).")

    try:
        from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
    except ImportError:
        raise ImportError("AeroSandbox package is required for this function.")

    k = (params.size - 2) // 2
    lower_weights = params[:k]
    upper_weights = params[k:2 * k]
    leading_edge_weight = float(params[-2])
    te_thickness = float(params[-1])

    coords_cst = get_kulfan_coordinates(
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        leading_edge_weight=leading_edge_weight,
        TE_thickness=te_thickness,
        N1=0.5,
        N2=1.0,
        n_points_per_side=n_points_per_side,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(coords_cst[:, 0], coords_cst[:, 1], 'C1-', lw=2.5, label="CST Airfoil")
    plt.fill(coords_cst[:, 0], coords_cst[:, 1], color='C1', alpha=0.15)

    plt.axis('equal')
    plt.xlim(-0.05, 1.05)
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()


def process_all_cst_displays(input_directory, output_directory):
    input_dir = Path(input_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(input_dir.glob("*_best_parameters.json")):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            cst_params = data.get("cst_parameters")
            if cst_params is None:
                print(f"No 'cst_parameters' key in {filename.name}, skipping...")
                continue

            out_file = output_dir / f"{filename.stem}_airfoil.png"
            display_and_save_cst_airfoil(
                cst_params=cst_params,
                save_path=out_file,
                title=f"CST Airfoil from {filename.name}"
            )
            print(f"Saved CST Airfoil display for {filename.name} to {out_file}")

        except Exception as e:
            print(f"Failed to generate plot for {filename.name}: {e}")