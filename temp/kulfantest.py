import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as pp
import aerosandbox.geometry.airfoil.airfoil_families as asb_airfoil_families # Import the specific module

# --- Dummy me_constants for demonstration ---
class MeConstants:
    NACA_Airfoil = "NACA2412" # Example NACA airfoil
me_constants = MeConstants()

print(f"AeroSandbox Version: {asb.__version__}")

# --- 1. Define the Airfoil Shape (18 Kulfan Parameters) ---
# A) Converting an existing airfoil (like NACA) to CST:
naca_airfoil = asb.Airfoil(me_constants.NACA_Airfoil) # Create a base NACA airfoil object
kulfan_obj = naca_airfoil.to_kulfan_airfoil(
    n_weights_per_side=8,
    N1=0.5,
    N2=1.0,
    normalize_coordinates=True,
    use_leading_edge_modification=True
)
# Assemble the 18-element array from the dictionary returned by kulfan_obj.kulfan_parameters
# This array is what CMA-ES will optimize.
airfoil_cst_params = np.concatenate([
    kulfan_obj.kulfan_parameters['upper_weights'],
    kulfan_obj.kulfan_parameters['lower_weights'],
    np.array([kulfan_obj.kulfan_parameters['TE_thickness']]),
    np.array([0.0]) # TE_camber, explicitly set to 0.0 for symmetrical NACA
])

print(f"Airfoil parameters (first 5 of 18): {airfoil_cst_params[:5]}")

# --- 2. Explicitly Generate Coordinates from Kulfan Parameters ---
# This is the crucial step to avoid the 'NoneType' error.
# We use the dedicated function to generate coordinates from the CST parameters.
# You can choose the number of points for the airfoil discretization.
n_points_per_side = 200 # A common number of points for airfoil definition
generated_coordinates = asb_airfoil_families.generate_n_point_cst_airfoil_coordinates(
    cst_parameters=airfoil_cst_params,
    n_points_per_side=n_points_per_side
)

print(f"\nGenerated {len(generated_coordinates)} coordinates from CST parameters.")
# print(f"First 5 generated coordinates:\n{generated_coordinates[:5]}")


# --- 3. Create the Airfoil Object by Passing the Generated Coordinates ---
# Now, we create the Airfoil object by providing the 'coordinates' directly.
# This bypasses the internal Kulfan-to-coordinates conversion that was failing.
my_custom_airfoil = asb.Airfoil(
    name="MyCSTAirfoil",
    coordinates=generated_coordinates # Pass the explicitly generated coordinates
)

print(f"\nAirfoil created: {my_custom_airfoil.name}")

# --- Verify coordinates attribute (should no longer be None) ---
print(f"Checking coordinates attribute: {my_custom_airfoil.coordinates is not None}")
if my_custom_airfoil.coordinates is None:
    print("CRITICAL ERROR: Coordinates are STILL None. This is highly unexpected.")
    exit() # Exit if still failing

print(f"Airfoil has {len(my_custom_airfoil.coordinates)} coordinates.")


# --- Optional: Draw the airfoil to visualize it ---
fig, ax = plt.subplots(figsize=(6, 2))
my_custom_airfoil.draw(show=False) # Draw without immediately showing
pp.show_plot(
    "Airfoil Shape for NeuralFoil Input (Generated Coordinates)",
    "Chordwise Position (X/c)",
    "Normal Position (Y/c)"
)


# --- 4. Define Other Inputs for NeuralFoil ---
alpha_deg = 5.0
reynolds_number = 1e6
mach_number = 0.2
n_crit_val = 9.0
x_tr_top_forced = None
x_tr_bot_forced = None
control_surfaces_list = None

# --- 5. Run NeuralFoil Analysis ---
print("\n--- Running NeuralFoil Analysis ---")

# This should now work correctly as the airfoil object has valid coordinates.
aero_results = my_custom_airfoil.get_aero_from_neuralfoil(
    alpha=alpha_deg,
    re=reynolds_number,
    mach=mach_number,
    n_crit=n_crit_val,
    x_transition_top=x_tr_top_forced,
    x_transition_bottom=x_tr_bot_forced,
    control_surfaces=control_surfaces_list
)

print("\n--- NeuralFoil Outputs (Bulk Outputs) ---")
print(f"Lift Coefficient (CL): {aero_results['CL']:.4f}")
print(f"Drag Coefficient (CD): {aero_results['CD']:.4f}")
print(f"Moment Coefficient (CM): {aero_results['CM']:.4f}")
print(f"Critical Mach number (M_crit): {aero_results['M_crit']:.4f}")
