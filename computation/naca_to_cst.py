import numpy as np
from scipy.optimize import least_squares

### --- NACA Airfoil Generator --- ###
def naca4_coordinates(code, n_points=100):
    m = int(code[0]) / 100
    p = int(code[1]) / 10
    t = int(code[2:]) / 100

    x = (1 - np.cos(np.linspace(0, np.pi, n_points))) / 2  # Cosine spacing
    yt = 5 * t * (
        0.2969 * np.sqrt(x) -
        0.1260 * x -
        0.3516 * x**2 +
        0.2843 * x**3 -
        0.1015 * x**4
    )

    yc = np.where(x < p,
                  m / p**2 * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))

    dyc_dx = np.where(x < p,
                      2 * m / p**2 * (p - x),
                      2 * m / (1 - p)**2 * (p - x))
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack((x_coords, y_coords))


import numpy as np
from scipy.special import comb

def coordinates_to_kulfan(coordinates, n_weights_per_side=8, N1=0.5, N2=1.0):
    n_coordinates = len(coordinates)
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    LE_index = np.argmin(x)
    is_upper = np.arange(len(x)) <= LE_index

    C = x ** N1 * (1 - x) ** N2

    N = n_weights_per_side - 1
    K = comb(N, np.arange(N + 1))

    dims = (n_weights_per_side, n_coordinates)

    def wide(vector):
        return np.tile(np.reshape(vector, (1, dims[1])), (dims[0], 1))

    def tall(vector):
        return np.tile(np.reshape(vector, (dims[0], 1)), (1, dims[1]))

    S_matrix = (
        tall(K)
        * wide(x) ** tall(np.arange(N + 1))
        * wide(1 - x) ** tall(N - np.arange(N + 1))
    )

    leading_edge_weight_row = x * np.maximum(1 - x, 0) ** (n_weights_per_side + 0.5)
    trailing_edge_thickness_row = np.where(is_upper, x / 2, -x / 2)

    A = np.concatenate(
        [
            np.where(wide(is_upper), 0, wide(C) * S_matrix).T,
            np.where(wide(is_upper), wide(C) * S_matrix, 0).T,
            leading_edge_weight_row.reshape(-1, 1),
            trailing_edge_thickness_row.reshape(-1, 1),
        ],
        axis=1,
    )

    b = y

    x_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    lower_weights = x_sol[:n_weights_per_side]
    upper_weights = x_sol[n_weights_per_side : 2 * n_weights_per_side]
    leading_edge_weight = x_sol[-2]
    te_thickness = x_sol[-1]

    if te_thickness < 0:
        # Retry with TE_thickness = 0
        x_sol, _, _, _ = np.linalg.lstsq(A[:, :-1], b, rcond=None)
        lower_weights = x_sol[:n_weights_per_side]
        upper_weights = x_sol[n_weights_per_side : 2 * n_weights_per_side]
        leading_edge_weight = x_sol[-1]
        te_thickness = 0.0

    return {
        "lower_weights": lower_weights,
        "upper_weights": upper_weights,
        "leading_edge_weight": leading_edge_weight,
        "TE_thickness": te_thickness,
        "N1": N1,
        "N2": N2
    }

def kulfan_to_coordinates(kulfan_params, n_points=100):
    lower_weights = kulfan_params["lower_weights"]
    upper_weights = kulfan_params["upper_weights"]
    le_weight = kulfan_params["leading_edge_weight"]
    te_thickness = kulfan_params["TE_thickness"]
    N1 = kulfan_params["N1"]
    N2 = kulfan_params["N2"]

    n_weights = len(lower_weights)
    K = comb(n_weights - 1, np.arange(n_weights))

    x = np.linspace(0, 1, n_points)
    C = x ** N1 * (1 - x) ** N2

    def shape_function(weights):
        B = np.array([
            K[i] * x**i * (1 - x)**(n_weights - 1 - i)
            for i in range(n_weights)
        ])
        S = np.sum(weights[:, np.newaxis] * B, axis=0)
        return C * S

    y_upper = shape_function(upper_weights)
    y_lower = shape_function(lower_weights)

    y_upper += le_weight * x * (1 - x) ** (n_weights + 0.5) + (x * te_thickness / 2)
    y_lower += le_weight * x * (1 - x) ** (n_weights + 0.5) - (x * te_thickness / 2)

    # Combine upper and lower surfaces
    x_full = np.concatenate([x[::-1], x[1:]])
    y_full = np.concatenate([y_upper[::-1], y_lower[1:]])

    return np.column_stack((x_full, y_full))

import numpy as np

def normalize_coordinates(coordinates):
    coords = coordinates.copy()

    # Translate so LE is at (0,0)
    le_index = np.argmin(coords[:, 0]**2 + coords[:, 1]**2)
    le_point = coords[le_index]
    coords -= le_point

    # Rotate so chord line aligns with x-axis
    te_point = coords[0] if coords[0,0] > coords[-1,0] else coords[-1]
    dx, dy = te_point[0], te_point[1]
    theta = -np.arctan2(dy, dx)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    coords = coords @ rotation_matrix.T

    # Scale so chord length is 1
    chord_length = np.max(coords[:,0]) - np.min(coords[:,0])
    coords /= chord_length

    return coords
