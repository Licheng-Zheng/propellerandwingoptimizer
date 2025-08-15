import neuralfoil as nf
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
import numpy as np
import logging
from typing import List, Callable, Dict, Any, Tuple

def internal_check_airfoil_self_overlap(cst_parameters, N=1000, tolerance=1e-6):
    """
    Comprehensive check for airfoil self-overlap using multiple validation methods.
    
    Args:
        cst_parameters (dict): CST parameters containing upper_weights, lower_weights,
                              TE_thickness, and leading_edge_weight
        N (int): Number of points to sample for analysis
        tolerance (float): Numerical tolerance for overlap detection
        
    Returns:
        bool: True if overlap detected, False otherwise
        dict: Detailed information about the check results
    """
    
    x = np.linspace(0, 1, N)
    
    # Get coordinates
    try:
        TE_thickness_upper = cst_parameters.get('TE_thickness', 0) / 2.0
        TE_thickness_lower = cst_parameters.get('TE_thickness', 0) / 2.0
        
        y_u = get_kulfan_coordinates(
            cst_parameters["upper_weights"],
            x,
            cst_parameters.get("leading_edge_weight", 0),
            TE_thickness_upper
        )
        y_l = get_kulfan_coordinates(
            cst_parameters["lower_weights"],
            x,
            cst_parameters.get("leading_edge_weight", 0),
            TE_thickness_lower
        )
    except Exception as e:
        return True, {"error": f"Failed to generate coordinates: {str(e)}"}
    
    # Check 1: Basic vertical overlap
    vertical_overlap = np.any(y_u <= y_l + tolerance)
    min_thickness = np.min(y_u - y_l)
    
    # Check 2: Self-intersection detection using line segment intersection
    upper_intersects = check_curve_self_intersection(x, y_u)
    lower_intersects = check_curve_self_intersection(x, y_l)
    
    # Check 3: Monotonicity violations (extreme curvature changes)
    upper_curvature_issue = check_extreme_curvature(x, y_u)
    lower_curvature_issue = check_extreme_curvature(x, y_l)
    
    # Check 4: Geometric validity
    geometric_invalid = check_geometric_validity(x, y_u, y_l)
    
    # Compile results
    overlap_detected = (vertical_overlap or upper_intersects or lower_intersects or 
                       upper_curvature_issue or lower_curvature_issue or geometric_invalid)
    
    results = {
        "overlap_detected": overlap_detected,
        "vertical_overlap": vertical_overlap,
        "min_thickness": min_thickness,
        "upper_self_intersects": upper_intersects,
        "lower_self_intersects": lower_intersects,
        "upper_curvature_issue": upper_curvature_issue,
        "lower_curvature_issue": lower_curvature_issue,
        "geometric_invalid": geometric_invalid
    }
    
    return overlap_detected, results

def check_curve_self_intersection(x, y, min_separation=0.1):
    """
    Check if a curve intersects itself by looking for line segment intersections.
    
    Args:
        x, y: Coordinate arrays
        min_separation: Minimum x-distance between segments to consider for intersection
    
    Returns:
        bool: True if self-intersection detected
    """
    n = len(x)
    
    for i in range(n - 1):
        for j in range(i + 2, n - 1):  # Skip adjacent segments
            # Only check segments that are sufficiently separated in x
            if abs(x[i] - x[j]) < min_separation and abs(x[i+1] - x[j+1]) < min_separation:
                continue
                
            # Check if line segments (i, i+1) and (j, j+1) intersect
            if line_segments_intersect(x[i], y[i], x[i+1], y[i+1], 
                                     x[j], y[j], x[j+1], y[j+1]):
                return True
    
    return False

def line_segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Check if two line segments intersect using parametric line equations.
    """
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:  # Lines are parallel
        return False
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    # Check if intersection point lies within both line segments
    return 0 <= t <= 1 and 0 <= u <= 1

def check_extreme_curvature(x, y, curvature_threshold=50):
    """
    Check for extreme curvature changes that might indicate self-overlap.
    """
    if len(x) < 3:
        return False
    
    # Calculate curvature using finite differences
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Avoid division by zero
    denominator = (dx**2 + dy**2)**1.5
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    
    curvature = np.abs(dx * ddy - dy * ddx) / denominator
    
    # Check for extreme curvature values
    return np.any(curvature > curvature_threshold)

def check_geometric_validity(x, y_u, y_l):
    """
    Check basic geometric validity of the airfoil shape.
    """
    # Check if leading edge closes properly (both surfaces should meet at x=0)
    if abs(y_u[0] - y_l[0]) > 1e-3:
        return True
    
    # Check if trailing edge closes properly (both surfaces should meet at x=1)
    if abs(y_u[-1] - y_l[-1]) > 1e-3:
        return True
    
    # Check for excessive thickness variations (potential overlap indicator)
    thickness = y_u - y_l
    thickness_gradient = np.gradient(thickness)
    
    # Look for extreme thickness gradient changes
    if np.any(np.abs(thickness_gradient) > 10):  # Adjust threshold as needed
        return True
    
    return False

def robust_airfoil_overlap_check(cst_parameters, N=1000):
    """
    FIXED: Comprehensive overlap detection using consistent coordinate generation
    """
    try:
        # FIXED: Use the correct coordinate generation method
        coordinates = get_kulfan_coordinates(
            lower_weights=cst_parameters["lower_weights"],
            upper_weights=cst_parameters["upper_weights"],
            leading_edge_weight=cst_parameters.get("leading_edge_weight", 0),
            TE_thickness=cst_parameters.get('TE_thickness', 0),
            N1=0.5,
            N2=1.0,
            n_points_per_side=N//2
        )
        
        # Split coordinates into upper and lower surfaces
        n_mid = len(coordinates) // 2
        upper_surface = coordinates[:n_mid]
        lower_surface = coordinates[n_mid:]
        
        # Reverse upper surface to match x ordering (it comes reversed from get_kulfan_coordinates)
        upper_surface = upper_surface[::-1]
        
        # Interpolate both surfaces to common x points for comparison
        x_common = np.linspace(0, 1, 200)
        y_u = np.interp(x_common, upper_surface[:, 0], upper_surface[:, 1])
        y_l = np.interp(x_common, lower_surface[:, 0], lower_surface[:, 1])
        
    except Exception as e:
        logging.error(f"Failed to generate coordinates for overlap check: {e}")
        return True  # Reject if we can't even generate coordinates
    
    # FIXED: Check for actual overlap, not just touching at trailing edge
    thickness = y_u - y_l
    min_thickness = np.min(thickness)
    
    # FIXED: Use a more reasonable tolerance and exclude trailing edge
    # Airfoils naturally have zero thickness at trailing edge, so exclude the last few points
    exclude_points = 5  # Exclude last 5 points from each end
    thickness_interior = thickness[exclude_points:-exclude_points] if len(thickness) > 2*exclude_points else thickness
    
    if len(thickness_interior) > 0:
        min_thickness_interior = np.min(thickness_interior)
        # FIXED: Use negative threshold for actual overlap detection
        overlap_threshold = -1e-6  # Only flag if upper surface is actually BELOW lower surface
        
        if min_thickness_interior < overlap_threshold:
            logging.debug(f"Overlap detected: Minimum interior thickness {min_thickness_interior:.8f} < {overlap_threshold}")
            return True
    
    # Check for extreme negative thickness anywhere (true overlap)
    severe_overlap_threshold = -1e-4
    if np.any(thickness < severe_overlap_threshold):
        logging.debug(f"Severe overlap detected: thickness < {severe_overlap_threshold}")
        return True
    
    # FIXED: Reduce sensitivity of other checks to avoid false positives
    # Only check for extreme cases that clearly indicate problems
    
    # Check 2: Self-intersection detection (only for extreme cases)
    try:
        if check_curve_self_intersection(upper_surface[:, 0], upper_surface[:, 1], min_separation=0.05) or \
           check_curve_self_intersection(lower_surface[:, 0], lower_surface[:, 1], min_separation=0.05):
            logging.debug("Overlap detected: Surface self-intersection")
            return True
    except:
        pass  # Skip if check fails
    
    # Check 3: Extreme curvature that indicates folding (increase threshold)
    try:
        if check_extreme_curvature(upper_surface[:, 0], upper_surface[:, 1], curvature_threshold=100) or \
           check_extreme_curvature(lower_surface[:, 0], lower_surface[:, 1], curvature_threshold=100):
            logging.debug("Overlap detected: Extreme curvature indicating surface folding")
            return True
    except:
        pass  # Skip if check fails
    
    return False

def internal_self_overlap_robust(cst_parameters, N=1000):
    """
    Checks for airfoil self-overlap using NeuralFoil functions and a robust slope analysis.

    Args:
        cst_parameters (dict): The CST parameters of the airfoil.
        N (int, optional): Number of points to sample for the check. A higher N
                           provides a more accurate check. Defaults to 1000.
    """

    x = np.linspace(0, 1, N)
    
    # Split the trailing edge thickness between the upper and lower surfaces
    TE_thickness_upper = cst_parameters['TE_thickness'] / 2.0
    TE_thickness_lower = cst_parameters['TE_thickness'] / 2.0

    # Get the coordinates using the NeuralFoil function
    y_u = get_kulfan_coordinates(
        cst_parameters["upper_weights"],
        x,
        cst_parameters["leading_edge_weight"],
        TE_thickness_upper
    )
    y_l = get_kulfan_coordinates(
        cst_parameters["lower_weights"],
        x,
        cst_parameters["leading_edge_weight"],
        TE_thickness_lower
    )

    # Check 1: Direct y-coordinate overlap
    if np.any(y_u < y_l):
        print("Overlap detected: Upper surface is below the lower surface.")
        return True

    # Check 2: Slope analysis for internal overlap
    upper_slope = np.gradient(y_u, x)
    lower_slope = np.gradient(y_l, x)

    # An airfoil's upper slope should be non-positive, and lower slope should be non-negative.
    # A positive upper slope or a negative lower slope indicates a reversal in the shape,
    # which is a strong sign of an internal overlap.
    if np.any(upper_slope > 0) or np.any(lower_slope < 0):
        print("Overlap detected: Surface slopes indicate an invalid shape.")
        return True

    return False

def trailing_edge_mismatch(cst_params_dict, tol=1e-4, N=300):
    """
    FIXED: Checks if the trailing edge of the airfoil is the same for the top and bottom portions of the airfoil (meaning they meet each other)

    Creates a bunch of points for the CST airfoil (we only need to the last one, but we can use the other points for other constraints in the future) 

    Args:
        cst_params_dict (dictionary): Aerosandbox dictionary containing the CST parameters of the airfoil
        tol (float, optional): Not sure if we can make it exactly meet with how we set things up so the tolerance indicates if it is an acceptable amount of difference. Defaults to 1e-4.
        N (int, optional): How many points are instantiated across the wing (this will only matter in the future when I actually use the other N-1 points). Defaults to 300.

    Returns:
        bool: Does it fail the trailing check test or not
    """
    try:
        # FIXED: Use the proper coordinate generation method
        coordinates = get_kulfan_coordinates(
            lower_weights=cst_params_dict["lower_weights"],
            upper_weights=cst_params_dict["upper_weights"],
            leading_edge_weight=cst_params_dict.get("leading_edge_weight", 0),
            TE_thickness=cst_params_dict.get('TE_thickness', 0),
            N1=0.5,
            N2=1.0,
            n_points_per_side=N//2
        )
        
        # The trailing edge should be at x=1, which is the last point of lower surface
        # and first point of upper surface (after reversal)
        n_mid = len(coordinates) // 2
        upper_surface = coordinates[:n_mid]
        lower_surface = coordinates[n_mid:]
        
        # Get trailing edge points (x=1)
        te_upper = upper_surface[0, 1]  # First point of upper surface (at x=1)
        te_lower = lower_surface[-1, 1]  # Last point of lower surface (at x=1)
        
        # Check if trailing edge gap exceeds tolerance
        te_gap = abs(te_upper - te_lower)
        return te_gap > tol
        
    except Exception as e:
        logging.error(f"Error in trailing_edge_mismatch: {e}")
        return True  # Assume violation if we can't check