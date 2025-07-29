import neuralfoil as nf
import numpy as np
import aerosandbox as asb
import matplotlib.pyplot as plt

lower_weights = np.array([-0.1318813,  -0.09232809, -0.01172289, -0.10578506,  0.05046554, -0.11528967, 0.06358729, -0.07139855])
upper_weights = np.array([0.2191141, 0.22408801, 0.29190666, 0.25998735, 0.20912236, 0.34193221, 0.20025737, 0.34125507])

leading_edge_weight = 0.01
te_thickness = 0.02
N1 = 0.5
N2 = 1.0

kulfan_parameters = {
    "lower_weights": lower_weights,
    "upper_weights": upper_weights,
    "leading_edge_weight": leading_edge_weight,
    "TE_thickness": te_thickness,
    "N1": N1,
    "N2": N2
}

airfoil = asb.Airfoil(
    kulfan_parameters=kulfan_parameters
)

# Get coordinates
coords = airfoil.coordinates  # shape (N,2)

# Plot
plt.plot(coords[:,0], coords[:,1], '-k')
plt.title("Airfoil from Kulfan Parameters")
plt.axis('equal')
plt.grid(True)
plt.show()

alpha = 5  # degrees
Re = 1e6

aero = nf.get_aero_from_kulfan_parameters(
    kulfan_parameters=nf.normalize_kulfan_parameters(kulfan_parameters),
    alpha=alpha,
    Re=Re,
    model_size="medium"
)

print("CL:", aero["CL"])
print("CD:", aero["CD"])
print("CM:", aero["CM"])
print("Confidence:", aero["analysis_confidence"])
