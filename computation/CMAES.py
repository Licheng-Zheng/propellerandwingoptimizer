import cma 
from aerosandbox.geometry.airfoil import Airfoil
import me_constants
import numpy as np
import aerosandbox as asb

# loading in the first guess airfoil (I will update how the first guess is chosen later)
first_airfoil = Airfoil(me_constants.NACA_Airfoil)
print(type(first_airfoil))
kulfan_airfoil_object = first_airfoil.to_kulfan_airfoil(
    n_weights_per_side=8,
    N1=0.5,
    N2=1.0,
    normalize_coordinates=True,
    use_leading_edge_modification=True
)

kulfan_airfoil_object.draw()
modified_airfoil = kulfan_airfoil_object.kulfan_parameters 
modified_airfoil["lower_weights"][0] = 1

kulfan_airfoil_object.kulfan_parameters = modified_airfoil



kulfan_airfoil_object.draw()

# This function is going to do a lot of heavy lifting in the future
# get_aero_from_neuralfoil(alpha, Re, mach=0.0, n_crit=9.0, xtr_upper=1.0, xtr_lower=1.0, model_size='large', control_surfaces=None, include_360_deg_effects=True)[source]


# cma_input_array = np.concatenate([
#     kulfan_airfoil_object.upper_weights,  # 8 upper weights
#     kulfan_airfoil_object.lower_weights,  # 8 lower weights
#     np.array([kulfan_airfoil_object.TE_thickness]), # 1 trailing edge thickness (needs to be an array for concatenation)
#     np.array([kulfan_airfoil_object.TE_camber])     # 1 trailing edge camber (needs to be an array for concatenation)
# ])

# # Now, use this correctly formatted array to create the Airfoil object
# airfoil_to_evaluate = asb.Airfoil(
#     name="CMA-ES Airfoil",
#     kulfan_parameters=cma_input_array
# )

# print(f"\nSuccessfully created Airfoil object: {type(airfoil_to_evaluate)}")

# coordinates = airfoil_to_evaluate.coordinates()
# print(f"Airfoil has {len(coordinates)} coordinates.")



# airfoil_to_evaluate.draw()
# 1. Initialize CMA-ES
#     - Input: x0 (initial guess vector)
#     - Sigma (initial step size)
#     - CMAES = cma.CMAEvolutionStrategy(x0, sigma, options)

# 2. While not CMAES.stop():
#     a. Ask for candidate solutions
#         X = CMAES.ask()

#     b. Evaluate each solution
#         Y = [fitness(x) for x in X]

#     c. Tell CMAES the results
#         CMAES.tell(X, Y)

#     d. (Optional) Log, visualize, compare with other optimizers
#         print(CMAES.result.fbest)  # best fitness so far

#     e. (Optional) Inject elite solutions or custom logic

# 3. After loop ends:
#     - CMAES.result.xbest = best solution vector
#     - CMAES.result.fbest = best fitness score
