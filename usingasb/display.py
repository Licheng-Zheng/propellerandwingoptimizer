import numpy as np
import matplotlib.pyplot as plt
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
import aerosandbox as asb
from PARAMETERS import starting_airfoil


def display_cst_airfoil(cst_params, n_points_per_side=200, title="CST vs Baseline NACA", block=True):
    """
    Plot an airfoil from a flat list of CST parameters and overlay with the baseline NACA from PARAMETERS.starting_airfoil.

    Expected input layout for cst_params (flat list or 1D array):
    [lower_weights..., upper_weights..., leading_edge_weight, TE_thickness]

    Notes:
    - lower and upper weights must be of equal length (Kulfan order).
    - starting_airfoil should be a NACA string like 'naca4412'.
    """
    params = np.asarray(cst_params, dtype=float).ravel()
    if params.size < 4:
        raise ValueError("cst_params must include lower, upper weights and two scalars (LE weight, TE thickness).")

    # Heuristic: assume even split for lower/upper and last two are scalars
    # If length is 2k + 2 -> k lower, k upper, then scalars
    if (params.size - 2) % 2 != 0:
        raise ValueError("Length of cst_params minus two scalars must be even (same number of lower/upper weights).")

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

    # Build baseline NACA from PARAMETERS.starting_airfoil
    base_airfoil = asb.Airfoil(name=starting_airfoil)
    coords_naca = base_airfoil.coordinates

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(coords_naca[:, 0], coords_naca[:, 1], 'k--', lw=2, label=f"Baseline {starting_airfoil.upper()}")
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
    plt.show(block=block)


# Example usage
if __name__ == "__main__":
    # Example with 8 weights per side (adjust as needed): 8 lower, 8 upper, LE, TE

    example = """-1.541734910423899207e-01
-2.338131286022233080e-01
-1.827990385894588399e-01
1.131691608906119778e-01
7.820662523103980024e-02
7.825951340583436888e-02
8.124837965313266852e-02
5.464429192369925359e-03
2.188534189358497806e-01
2.197465972287977465e-01
2.367238418352298479e-01
2.879041603031025631e-01
1.568834964714334035e-01
3.219681449744276058e-01
1.488885254877904130e-01
4.297555674010429261e-01
-1.468505033551100077e-02
2.661707325589582340e-03"""

    print(example.split("\n"))

    display_cst_airfoil(list(map(float, example.split("\n"))))