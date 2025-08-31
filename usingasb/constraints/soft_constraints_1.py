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