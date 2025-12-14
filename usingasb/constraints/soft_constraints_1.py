import neuralfoil as nf
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
import numpy as np
import logging
from typing import List, Callable, Dict, Any, Tuple

def internal_minimum_thickness_constraint(cst_parameters, minimum_thickness=0.02, start_check_at_x=0.1, check_up_to_x=0.9, N=200):
    """
    Soft constraint that penalizes airfoils if they are thinner than a minimum threshold,
    but only checks up to a certain point on the wing to avoid penalizing trailing edge closure.
    
    Args:
        cst_parameters (dict): The CST parameters of the airfoil
        minimum_thickness (float, optional): Minimum required thickness as fraction of chord. Defaults to 0.02 (2%).
        check_up_to_x (float, optional): X-coordinate up to which thickness is checked (0.0 to 1.0). Defaults to 0.9.
        N (int, optional): Number of points to sample for thickness calculation. Defaults to 200.
        
    Returns:
        float: Penalty value (0 if no violation, positive value proportional to thickness deficit)
    """
    
    try:
        # Generate points only up to the specified x-coordinate
        x = np.linspace(start_check_at_x, check_up_to_x, N)
        
        # Split trailing edge thickness between upper and lower surfaces
        TE_thickness_upper = cst_parameters.get('TE_thickness', 0) / 2.0
        TE_thickness_lower = cst_parameters.get('TE_thickness', 0) / 2.0
        leading_edge_weight = cst_parameters.get("leading_edge_weight", 0)
        
        # Generate upper and lower surface coordinates
        y_u = get_kulfan_coordinates(
            cst_parameters["upper_weights"],
            x,
            leading_edge_weight,
            TE_thickness_upper
        )
        y_l = get_kulfan_coordinates(
            cst_parameters["lower_weights"],
            x,
            leading_edge_weight,
            TE_thickness_lower
        )
        
        # Calculate thickness at each point
        thickness = y_u - y_l
        
        # Find the minimum thickness in the checked region
        min_thickness = np.min(thickness)
        
        # Apply penalty if thickness is below minimum
        if min_thickness < minimum_thickness:
            # Return penalty proportional to thickness deficit
            penalty = minimum_thickness - min_thickness
            return penalty
            
        return 0.0
        
    except Exception as e:
        logging.error(f"Error in minimum thickness constraint: {e}")
        # Return a large penalty if we can't evaluate the constraint
        return 1000.0

def internal_lift_coefficient_minimum(airfoil_lift, minimum=0.1, use_reward=True):
    """
    Checks if the lift coefficient is above a certain threshold, and penalizes it if the airfoil is not high enough (I will implement hard rejection in the future)

    The current minimum is just a placeholder, in the future, the minimum lift will be provided in the initial input package. The actual lift of the wing (calculated) will be compared to the required lift coefficient provided in the input package, and depending on whether soft or hard constraint is used, a pentalty will be applied. For soft constraints, the greater the difference between lift and minimum, the greater the penalty.

    Args:
        airfoil_lift (float): The lift coefficient of the airfoil
        minimum (float, optional): The minimum amount of lift. Will be provided by the input package in the future. Defaults to 0.1.
    """

    # I will need to multiply it by a certain amount based on the importance of the pentalty (I need to create another importance list) 
    if airfoil_lift < minimum:
        return airfoil_lift - minimum  # Return the difference as a penalty
    # I need to add functionality that will say if the lift is greater than the minimum, then it will receive a reward
    # if airfoil_lift > minimum and use_reward: 
    #     return airfoil_lift - minimum
    return 0 

def internal_drag_coefficient_maximum(airfoil_drag, maximum=0.1, use_reward=True):
    """
    Checks if the drag coefficient is below a certain threshold, and penalizes it if the airfoil is not low enough (I will implement hard rejection in the future)

    The current maximum is just a placeholder, in the future, the maximum drag will be provided in the initial input package. The actual drag of the wing (calculated) will be compared to the required drag coefficient provided in the input package, and depending on whether soft or hard constraint is used, a pentalty will be applied. For soft constraints, the greater the difference between drag and maximum, the greater the penalty.

    Args:
        airfoil_drag (float): The drag coefficient of the airfoil
        maximum (float, optional): The maximum amount of drag. Will be provided by the input package in the future. Defaults to 0.1.
    """

    if airfoil_drag > maximum:
        return airfoil_drag - maximum  # Return the difference as a penalty
    return 0

# ... existing code ...

def internal_slope_smoothness_constraint(
    cst_parameters,
    x_start=0.05,
    x_end=0.95,
    num_points=200,
    max_delta_slope=0.25,
    p_norm=2.0,
):
    """
    Soft constraint that penalizes rapid changes in surface slope between x_start and x_end.

    Args:
        cst_parameters (dict): CST parameter set describing the airfoil.
        x_start (float, optional): Lower bound of the x-range to enforce smoothness (0–1 chord fraction). Defaults to 0.05.
        x_end (float, optional): Upper bound of the x-range to enforce smoothness (0–1 chord fraction). Defaults to 0.95.
        num_points (int, optional): Number of samples used to evaluate slopes. Defaults to 200.
        max_delta_slope (float, optional): Allowed absolute change in slope between neighboring segments. Defaults to 0.25.
        p_norm (float, optional): Exponent for aggregating violations (use 1 for L1, 2 for RMS, np.inf for max). Defaults to 2.0.

    Returns:
        float: Penalty value (0 if smoothness is within tolerance, positive otherwise).
    """
    if not (0.0 <= x_start < x_end <= 1.0):
        raise ValueError("x_start and x_end must satisfy 0.0 <= x_start < x_end <= 1.0")

    x = np.linspace(x_start, x_end, num_points)

    te_half = cst_parameters.get("TE_thickness", 0.0) / 2.0
    le_weight = cst_parameters.get("leading_edge_weight", 0.0)

    y_upper = get_kulfan_coordinates(
        cst_parameters["upper_weights"],
        x,
        le_weight,
        te_half,
    )
    y_lower = get_kulfan_coordinates(
        cst_parameters["lower_weights"],
        x,
        le_weight,
        te_half,
    )

    dx = np.diff(x)
    slope_upper = np.diff(y_upper) / dx
    slope_lower = np.diff(y_lower) / dx

    delta_slope_upper = np.abs(np.diff(slope_upper))
    delta_slope_lower = np.abs(np.diff(slope_lower))

    violations = np.concatenate([
        np.clip(delta_slope_upper - max_delta_slope, a_min=0.0, a_max=None),
        np.clip(delta_slope_lower - max_delta_slope, a_min=0.0, a_max=None),
    ])

    if violations.size == 0:
        return 0.0

    if np.isinf(p_norm):
        return np.max(violations)

    return float(np.power(np.mean(np.power(violations, p_norm)), 1.0 / p_norm))

