import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import pyvista as pv  # never used pyvista before, I need it for saving so hopefully it works

params = [
    -0.35916714194000005,
    -0.32613168751256283,
    0.0027242415135564704,
    0.09944631488021428,
    -0.13855972736289612,
    0.9139015252785018,
    0.16684835720204239,
    0.8541981466670672,
    0.47964203788544413,
    0.7549932421578472,
    0.5732763470712393,
    0.9453704414046568,
    0.3409627498944472,
    0.5267430561419271,
    0.7399623447759083,
    0.9641490006816278,
    0.08109050731914888,
    0.007483141932256895
]

# The kulfan parameters of the wing (from aerosandbox)
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

# matplotlib the thing first to make sure I have the right wing 
coords = airfoil.coordinates
plt.figure(figsize=(8, 4))
plt.plot(coords[:, 0], coords[:, 1], 'k-', label="Airfoil")
plt.axis("equal")
plt.xlabel("x/c")
plt.ylabel("y/c")
plt.title("Airfoil Shape")
plt.grid(True)
plt.legend()
plt.show()

wing = asb.Wing(
    name="Rectangular Wing",
    xsecs=[
        asb.WingXSec(xyz_le=[0, 0, 0], chord=1.0, airfoil=airfoil),
        asb.WingXSec(xyz_le=[0, 10, 0], chord=1.0, airfoil=airfoil)  # Span along Y
    ]
)

# 3d wing in matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
wing.draw(ax=ax)
ax.set_xlabel("X (Chord)")
ax.set_ylabel("Y (Span)")
ax.set_zlabel("Z")
ax.view_init(elev=20, azim=-120)
plt.title("3D Wing")
plt.show()

# putting it into an object so I can export it
airplane = asb.Airplane(
    name="My_Wing_Project",
    wings=[wing]
)

# stl exporting
print("Attempting to generate mesh and export...")

try:
    mesh_object = airplane.draw(backend="pyvista", show=False)

    if isinstance(mesh_object, pv.Plotter):
        mesh_object.export_stl("cst_wing.stl")
    elif isinstance(mesh_object, pv.PolyData):
        mesh_object.save("cst_wing.stl")
    else:
        if hasattr(mesh_object, "save"):
            mesh_object.save("cst_wing.stl")
        else:
            print("Warning: unexpected object type. Attempting generic save...")
            mesh_object.save("cst_wing.stl")

    print("✅ Success! 'cst_wing.stl' has been saved.")
    del mesh_object

except AttributeError as e:
    print(f"Export Error: {e}")
    print("Tip: If you see 'PolyData object has no attribute export_stl', the code above handles this now.")
except ImportError:
    print("Error: The 'pyvista' library is missing.")
    print("Please install it by running: pip install pyvista qtpy PyQt5")
except Exception as e:
    print(f"An unexpected error occurred: {e}")