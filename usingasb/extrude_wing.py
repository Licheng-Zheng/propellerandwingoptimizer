import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import comb

# --- CST helper functions ---
def bernstein(n, i, x):
    return comb(n, i) * (x**i) * ((1-x)**(n-i))

def cst_surface(coeffs, x, LEM=0.0, TE_thickness=0.0):
    """
    Compute CST surface coordinates.
    coeffs: list of CST coefficients
    x: chordwise positions
    LEM: leading edge modification parameter
    TE_thickness: trailing edge thickness parameter
    """
    N = len(coeffs) - 1
    y = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        y += c * bernstein(N, i, x)
    # Apply leading edge modification and trailing edge thickness
    y += LEM * np.sqrt(x) + TE_thickness * x
    return y

# --- Input parameters ---
params = [
    -4.857476654314523690e-01, -2.146719130658100716e-02, -3.009772320674585777e-01,
    5.593555016026108273e-01, -4.137218632859339662e-01, 6.801698814157647321e-01,
    5.226085427781536064e-01, 9.070228429386175684e-01,  # upper surface (8)
    4.418196214490077711e-01, 5.333546305243734853e-01, 8.709016948985255357e-01,
    5.894239369255283023e-01, 8.737817612621111563e-01, 4.901136621783676039e-01,
    5.802977484625131410e-01, 9.533432040915910122e-01,  # lower surface (8)
    3.359459353875338117e-01,  # LEM
    9.992596404079442940e-03   # TE thickness
]

coeffs_upper = params[0:8]
coeffs_lower = params[8:16]
LEM = params[16]
TE_thickness = params[17]

# --- Generate airfoil coordinates ---
x = np.linspace(0, 1, 200)
y_upper = cst_surface(coeffs_upper, x, LEM, TE_thickness)
y_lower = cst_surface(coeffs_lower, x, LEM, TE_thickness)

# --- Extrude into rectangular wing ---
span = 10.0   # meters
n_span = 30   # number of spanwise stations
z_span = np.linspace(0, span, n_span)

coords = []
for z in z_span:
    for xi, yu, yl in zip(x, y_upper, y_lower):
        coords.append([xi, yu, z])  # upper surface
        coords.append([xi, yl, z])  # lower surface

coords = np.array(coords)

# --- Visualize ---
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=1, c='b')
ax.set_xlabel("Chord (x)")
ax.set_ylabel("Thickness (y)")
ax.set_zlabel("Span (z)")
ax.set_title("Rectangular Wing from CST Parameters")
plt.show()