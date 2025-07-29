import json 
from naca_to_cst import *
import neuralfoil as nf
from usingasb.objective import objective_function, scoring_model_1
import aerosandbox as asb

cst = coordinates_to_kulfan(naca4_coordinates(code=4412))

for key, val in cst.items():
    print(f"{key}: {val}")


leading_edge_weight = 0.02
te_thickness = 0.002
N1 = 0.5
N2 = 1.0

alpha = 5  # degrees
Re = 1e6

aero = nf.get_aero_from_kulfan_parameters(
    kulfan_parameters=cst,
    alpha=alpha,
    Re=Re,
    model_size="medium"
)

print(aero)

print("CL:", aero["CL"])
print("CD:", aero["CD"])
print("CM:", aero["CM"])
print("Confidence:", aero["analysis_confidence"])



