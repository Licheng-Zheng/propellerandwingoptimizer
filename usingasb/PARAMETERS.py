INITIAL_SIGMA = 0.4 # Initial mutation step size for optimization (controls exploration vs. exploitation)
MAX_EPOCHS = 100 # Maximum number of training/optimization iterations

# These are the things that I defined myself. It is basically all the parameters that I want to optimize for. 
# The improtance list specifies how important each one is, the higher the number, the more important it is. Negative values mean we want to optimize against it. 
WANTED_LIST = ["analysis_confidence", "CL", "CD", "CM"]  # List of target metrics to optimize (user-defined)
IMPORTANCE_LIST = [0.4, 0.3, -0.2, -0.1]  # Relative weights for each metric in WANTED_LIST (user-defined)

ALPHA = 5 # Scaling factor for importance weighting in the objective function either this or the angle of attack

# The RE encompasses a bunch of different things like air viscosity, speed and other things, but I will create something to give me more control over this in the future.
RE = 1e6 # Reynolds number, defines flow regime for aerodynamic analysis
# RE determines if the flow will be laminar (smooth) or turbulent (not smooth). Low: smooth, high: chaotic
# 1e6 is a common for aircraft wings 
MODEL_SIZE = "small" # Size/complexity of the aerosandbox model used for optimization
starting_airfoil = "naca4412" # Initial baseline airfoil shape to start optimization from

# Base results directory
RESULTS_BASE_DIR = r"C:\Users\liche\OneDrive\Desktop\PycharmProjects\PropellerDesign\usingasb\Optimization Results"

