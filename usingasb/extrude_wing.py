# import aerosandbox as asb
# import aerosandbox.numpy as np
# import matplotlib.pyplot as plt

# # --- Parameters ---
# params = [
#     -4.857476654314523690e-01, -2.146719130658100716e-02, -3.009772320674585777e-01,
#     5.593555016026108273e-01, -4.137218632859339662e-01, 6.801698814157647321e-01,
#     5.226085427781536064e-01, 9.070228429386175684e-01,  # upper surface (8)
#     4.418196214490077711e-01, 5.333546305243734853e-01, 8.709016948985255357e-01,
#     5.894239369255283023e-01, 8.737817612621111563e-01, 4.901136621783676039e-01,
#     5.802977484625131410e-01, 9.533432040915910122e-01,  # lower surface (8)
#     3.359459353875338117e-01,  # LEM
#     9.992596404079442940e-03   # TE thickness
# ]

# coeffs_upper = params[0:8]
# coeffs_lower = params[8:16]
# LEM = params[16]
# TE_thickness = params[17]

# # ✅ FIX: Remove 'n_points_per_side' from __init__
# airfoil = asb.KulfanAirfoil(
#     name="CST_Airfoil",
#     upper_weights=coeffs_upper,
#     lower_weights=coeffs_lower,
#     leading_edge_weight=LEM,
#     TE_thickness=TE_thickness
# )

# # Optional: If you strictly need 200 points per side, use repanel()
# # airfoil = airfoil.repanel(n_points_per_side=200)

# # # --- Plotting to verify ---
# # coords = airfoil.coordinates
# # plt.figure()
# # plt.plot(coords[:, 0], coords[:, 1], 'k-')
# # plt.axis("equal")
# # plt.title("Airfoil from CST Parameters")
# # plt.xlabel("x/c")
# # plt.ylabel("y/c")
# # plt.show()

# import aerosandbox as asb
# import aerosandbox.numpy as np
# import matplotlib.pyplot as plt

# # ... (Include your previous Parameters and Airfoil code here) ...

# # 1. Define the Wing (same as before)
# wing = asb.Wing(
#     name="Rectangular Wing",
#     xsecs=[
#         asb.WingXSec(
#             xyz_le=[0, 0, 0],
#             chord=1.0,
#             airfoil=airfoil
#         ),
#         asb.WingXSec(
#             xyz_le=[0, 10, 0],
#             chord=1.0,
#             airfoil=airfoil
#         )
#     ]
# )

# # 2. ✅ WRAP the wing in an Airplane object
# # AeroSandbox export tools work on 'Airplane' objects, not individual 'Wing' objects.
# airplane = asb.Airplane(
#     name="My_CST_Wing",
#     xyz_ref=[0, 0, 0],
#     wings=[wing]
# )

# print("Airplane object created. Attempting export...")

# # 3. ✅ EXPORT to STL using the PyVista backend
# # This renders the airplane into a 3D object (Plotter) and saves it.
# try:
#     # 'backend="pyvista"' creates a high-quality mesh for visualization/export
#     plotter = airplane.draw(
#         backend="pyvista", 
#         show=False  # Do not pop up a window, just create the object
#     )
    
#     # Save the mesh to STL
#     plotter.export_stl("cst_wing.stl")
#     print("Success! Saved 'cst_wing.stl' to your working directory.")

# except ImportError:
#     print("Error: The 'pyvista' library is missing.")
#     print("Please install it by running: pip install pyvista qtpy PyQt5")
# except Exception as e:
#     print(f"An error occurred during export: {e}")

# # --- 3D Visualization ---
# fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='3d'))
# wing.draw(ax=ax)
# # Set axis labels for clarity
# ax.set_xlabel("x")
# ax.set_ylabel("y (Span)")
# ax.set_zlabel("z")
# ax.view_init(elev=20, azim=-120) 
# plt.show()

# import aerosandbox as asb
# import aerosandbox.numpy as np

# # ... (Insert your previous Airfoil and Wing definition code here) ...

# # 1. Mesh the wing geometry
# # This converts the parametric wing definition into a surface mesh (triangles)
# mesh = wing.generate_mesh(
#     mesh_resolution_chordwise=20,  # Adjust resolution as needed
#     mesh_resolution_spanwise=20
# )

# # 2. Save the mesh to an STL file
# filename = "cst_wing.stl"
# mesh.export_stl(filename)

# print(f"Successfully saved wing to {filename}")

# # Optional: If you need to visualize the mesh in Python before saving
# # mesh.draw()

import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import pyvista as pv  # Import pyvista explicitly to handle saving

# --- 1. Parameters & Airfoil Definition ---
params = [
    -4.857476654314523690e-01, -2.146719130658100716e-02, -3.009772320674585777e-01,
    5.593555016026108273e-01, -4.137218632859339662e-01, 6.801698814157647321e-01,
    5.226085427781536064e-01, 9.070228429386175684e-01, 
    4.418196214490077711e-01, 5.333546305243734853e-01, 8.709016948985255357e-01,
    5.894239369255283023e-01, 8.737817612621111563e-01, 4.901136621783676039e-01,
    5.802977484625131410e-01, 9.533432040915910122e-01, 
    3.359459353875338117e-01, 
    9.992596404079442940e-03 
]

coeffs_upper = params[0:8]
coeffs_lower = params[8:16]
LEM = params[16]
TE_thickness = params[17]

# Create Airfoil
airfoil = asb.KulfanAirfoil(
    name="CST_Airfoil",
    upper_weights=coeffs_upper,
    lower_weights=coeffs_lower,
    leading_edge_weight=LEM,
    TE_thickness=TE_thickness
)

# --- 2. Wing Definition ---
wing = asb.Wing(
    name="Rectangular Wing",
    xsecs=[
        asb.WingXSec(xyz_le=[0, 0, 0], chord=1.0, airfoil=airfoil),
        asb.WingXSec(xyz_le=[0, 10, 0], chord=1.0, airfoil=airfoil) # Span along Y
    ]
)

# --- 3. Airplane Wrap ---
# We must wrap the wing in an Airplane object to use export tools
airplane = asb.Airplane(
    name="My_Wing_Project",
    wings=[wing]
)

# --- 4. Export to STL ---
print("Attempting to generate mesh and export...")

try:
    # We use the 'pyvista' backend to generate the mesh geometry
    # show=False returns the object instead of opening a window
    mesh_object = airplane.draw(backend="pyvista", show=False)

    # CHECK: Is it a Plotter or a PolyData mesh?
    if isinstance(mesh_object, pv.Plotter):
        # It's a Plotter (scene) -> Use export_stl
        mesh_object.export_stl("cst_wing.stl")
    elif isinstance(mesh_object, pv.PolyData):
        # It's a raw Mesh (PolyData) -> Use save
        mesh_object.save("cst_wing.stl")
    else:
        # Fallback: Sometimes it returns a list of meshes
        if hasattr(mesh_object, "save"):
            mesh_object.save("cst_wing.stl")
        else:
            # Last resort: Try to extract the mesh from the plotter manually
            print("Warning: unexpected object type. Attempting generic save...")
            mesh_object.save("cst_wing.stl")
            
    print("✅ Success! 'cst_wing.stl' has been saved.")

except AttributeError as e:
    print(f"Export Error: {e}")
    print("Tip: If you see 'PolyData object has no attribute export_stl', the code above handles this now.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")