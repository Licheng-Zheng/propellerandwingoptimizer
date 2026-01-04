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

    example = """-2.839353027665307838e-01
-2.623289335105455367e-01
8.412364170263958441e-02
-1.816190717047370118e-01
4.898972607978330784e-01
-7.938511915811252706e-02
8.400002890175495285e-01
7.980061187048548277e-01
5.037868630831017214e-01
4.707219502044545645e-01
9.619412190581239086e-01
7.639555422037509480e-01
7.285056409488372964e-01
5.816941746793263812e-01
9.491557231581957410e-01
8.444914855836077594e-01
5.460924689230444162e-01
2.282167006939284093e-03"""


    print(example.split("\n"))
    print(len(example.split("\n")))

    display_cst_airfoil(list(map(float, example.split("\n"))))