import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from display_auxiliary_functions import plot_cst_airfoil

# === Load optimization log ===
with open(r"C:\Users\liche\OneDrive\Desktop\PycharmProjects\PropellerDesign\optimization_results_20250727_071625\optimization_log.json", "r") as f:
    log_data = json.load(f)

epoch_params = log_data["cmaes_1"]["best_parameters_per_epoch"]

# === CST airfoil wrapper ===
def params_to_cst_dict(param_array):
    param_array = np.array(param_array)
    upper_weights = param_array[:9]
    lower_weights = param_array[9:17]
    le_weight = param_array[17]
    te_thickness = 0.001  # or another constant if not tracked per epoch
    return {
        "upper_weights": upper_weights,
        "lower_weights": lower_weights,
        "leading_edge_weight": le_weight,
        "TE_thickness": te_thickness
    }

# === Initial plot ===
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.25)

current_epoch = 0
cst_dict = params_to_cst_dict(epoch_params[current_epoch])

# We will hijack the plot_cst_airfoil to use our own axes
def draw_airfoil(epoch_index):
    ax.clear()
    cst_dict = params_to_cst_dict(epoch_params[epoch_index])
    plot_cst_airfoil(
        cst_params=cst_dict,
        title=f"Epoch {epoch_index + 1}",
        show_params=True,
        show=False,  # prevent blocking plt.show()
        block=True
    )
    fig.canvas.draw_idle()

draw_airfoil(current_epoch)

# === Slider setup ===
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
epoch_slider = Slider(
    ax=ax_slider,
    label='Epoch',
    valmin=0,
    valmax=len(epoch_params) - 1,
    valinit=current_epoch,
    valfmt='%0.0f',
    valstep=1
)

def update(val):
    epoch = int(epoch_slider.val)
    draw_airfoil(epoch)

epoch_slider.on_changed(update)

plt.show()
