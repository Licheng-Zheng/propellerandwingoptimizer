import neuralfoil as nf
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
import numpy as np
import logging
from typing import List, Callable, Dict, Any, Tuple
logging.basicConfig(level=logging.INFO) 

def compute_result(cst_parameters, epoch, alpha, Re, model, wanted_lists:dict, importance_list:dict):
    """
    Computes the fitness of a cst airfoil

    Passes in the CST parameters into Neural foil as well as the expected conditions in which it will operate in. The wanted list and importance list are used to determine which parameters are the most important, and modify the fitness score to reflect that

    Args:
        cst_parameters (Kulfan Airfoil): 1D array of CST parameters for the airfoil. It is converted to Neural Foil format internally.
        alpha (float): Angle of attack in degrees
        Re (float): Reynolds Number, absolutely no idea what this is
        model (string): The model size to use for Neural Foil. Options are "xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge".
        wanted_lists (list): List of strings containing the results that we care about and want to use to score the airfoil.
        importance_list (list): List of floats containing the importance of each result in the wanted list 
    """

    aero_results = objective_function(cst_parameters, alpha, Re, model)

    score = scoring_model_1(aero_results, wanted_lists, importance_list)

    penalty, hard_constraint = internal_run_constraint_suite(cst_parameters=cst_parameters, aero_results=aero_results, epoch=epoch)

    if penalty > 0:
        logging.debug(f"Soft constrain penalty applied: {penalty}")

    # NEGATIVE IS GOOD FOR CMA-ES, POSITIVE IS BAD FOR CMA-ES CAS IT MINIMIZES
    if hard_constraint:
        # Return a large positive value to strongly discourage this solution
        return 1e6  # Or another very bad penalty value BRUH negative is good that's why you've been debugging for the past week you idiot
    
    # applies the soft constraints (my current targets are random, so I have no idea what effect they're going to have)
    score -= penalty

    return -score

def objective_function(cst_parameters, alpha, Re, model):
    """
    The airfoil cst parameters are passed into this function to be computed by Neural Foil. The dictionary of all results is returned. 

    Collects all the of the parameters that are provided, puts it into a neat package for Neural Foil, which then returns a dictionary of all results. 
    The returned dictionary contains all results from Neural Foil, but only the specified parameters (on top of lift, drag, moment and confidence) are returned. This was done in order to hopefully reduce the amount of data that is being transferred across different files, hopefully reducing the amount of computation time spent so we can get more runs in. 

    Args:
        cst_parameters (dict): Contains the CST parameters for the airfoil.
        alpha (float): Angle of attack in degrees.
        Re (float): Reynolds number.
        model (string): Model size  "xxsmall" "xsmall" "small" "medium" "large" "xlarge" "xxlarge" "xxxlarge"
        additional_params (list, optional): A list of all the other parameters that you want to return (currently, it just returns the entire dictionary) Defaults to None.

    Returns:
        dict: the results from Neural Foil on the airfoil and conditions provided
    """
    aero_results = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=cst_parameters,
        alpha=alpha,
        Re=Re,
        model_size=model
    )

    list_of_results = {
        "CL": aero_results["CL"], # Lift coefficient
        "CD": aero_results["CD"], # Drag coefficient
        "CM": aero_results["CM"], # Moment coefficient
        "AC": aero_results["analysis_confidence"], # Analysis confidence
    }

    # I'll be returning list of results in the future after I allow for multiple objectives to be optimized. For now, returning the dictionary makes it easier to debug 
    return aero_results 

# ---------------- SCORING MODELS ----------------

def scoring_model_1(aero_results, wanted_lists:list, importance_list:list):
    """
    Assigns a score to an airfoil based on results from Neural Foil and importance of each results specified by the user (in the importance list)

    Takes results from Neural Foil, looks for the wanted results in the results and multiplies it by the importance of that result based on the users specification. 
    Large usage of dictionaries because its faster! (did not know that till I searched it up)

    Args:
        aero_results (dict): List of results from Neural Foil containing only the results that will be used to reduce unnecessary transfer of data
        wanted_lists (list): List of strings containing the names of the results that we want to use to have the airfoil scored upon (to be compared against the other airfoils) 
        importance_list (list): List of floats containing the importance of each result in the wanted list. The higher the number, the more important it is to the user.

    Returns:
        float: The computed score based on weights and aero results.

        
    """

    assert len(wanted_lists) == len(importance_list), "The wanted lists and importance list must be the same length."
    
    score = 0.0
    
    for item in range(len(wanted_lists)): 
        score += aero_results[wanted_lists[item]] * importance_list[item]

    return score


# ============================================================================
# CONSTRAINT FACTORY SYSTEM
# ============================================================================

class ConstraintFactory:
    """Factory class for creating various types of airfoil constraints"""

    # Makes the aero_results dictionary into objects attribute    
    def __init__(self, aero_results: Dict[str, Any]):
        self.aero_results = aero_results
    
    # ---------------- HARD CONSTRAINT FACTORIES ----------------
    
    def create_robust_airfoil_overlap_constraint(self, N:int =1000):
        """Factory for self-overlap constraint because the other ones don't work"""
        def constraint(cst_parameters):
            return robust_airfoil_overlap_check(cst_parameters, N=N)
        
        constraint.metadata = {
            'type': 'hard',
            'name': 'robust_airfoil_overlap',
            'N': N
        }
        return constraint

    def create_self_overlap_constraint(self, N: int = 200):
        """Factory for self-overlap constraint"""
        def constraint(cst_parameters):
            return internal_self_overlap_simple(cst_parameters, N=N)
        
        constraint.metadata = {
            'type': 'hard',
            'name': 'self_overlap',
            'N': N
        }
        return constraint
    
    def create_self_overlap_constraint_robust(self, N: int = 1000):
        """Factory for self-overlap constraint"""
        def constraint(cst_parameters):
            return internal_self_overlap_robust(cst_parameters, N=N)
        
        constraint.metadata = {
            'type': 'hard',
            'name': 'self_overlap_robust',
            'N': N
        }
        return constraint
    
    def create_trailing_edge_constraint(self, tol: float = 0.01, N: int = 300):
        """Factory for trailing edge mismatch constraint - FIXED: More reasonable tolerance"""
        def constraint(cst_parameters):
            return trailing_edge_mismatch(cst_parameters, tol=tol, N=N)
        
        constraint.metadata = {
            'type': 'hard',
            'name': 'trailing_edge_mismatch',
            'tolerance': tol,
            'N': N
        }
        return constraint
    
    # ---------------- SOFT CONSTRAINT FACTORIES ----------------
    
    def create_lift_minimum_constraint(self, minimum: float = None, use_reward: bool = True):
        """Factory for minimum lift coefficient constraint"""
        if minimum is None:
            minimum = 0.1  # Default value, will be provided by input package in the future
            
        # Capture the current lift coefficient from aero_results
        current_lift = self.aero_results["CL"]
        
        def constraint(cst_parameters):
            # Use the captured lift coefficient instead of recalculating
            return internal_lift_coefficient_minimum(current_lift, minimum=minimum, use_reward=use_reward)
        
        constraint.metadata = {
            'type': 'soft',
            'name': 'lift_minimum',
            'minimum': minimum,
            'use_reward': use_reward,
            'current_value': current_lift
        }
        return constraint
    
    def create_drag_maximum_constraint(self, maximum: float = None, use_reward: bool = True):
        """Factory for maximum drag coefficient constraint"""
        if maximum is None:
            maximum = 0.1  # Default value, will be provided by input package in the future
            
        # Capture the current drag coefficient from aero_results
        current_drag = self.aero_results["CD"]
        
        def constraint(cst_parameters):
            # Use the captured drag coefficient instead of recalculating
            return internal_drag_coefficient_maximum(current_drag, maximum=maximum, use_reward=use_reward)
        
        constraint.metadata = {
            'type': 'soft',
            'name': 'drag_maximum',
            'maximum': maximum,
            'use_reward': use_reward,
            'current_value': current_drag
        }
        return constraint
    
    # ---------------- GENERIC CONSTRAINT FACTORY ----------------
    
    def create_constraint(self, constraint_type: str, **kwargs):
        """Generic constraint factory that routes to specific factories"""
        constraint_map = {
            # Hard constraints
            'self_overlap': self.create_self_overlap_constraint,
            'trailing_edge': self.create_trailing_edge_constraint,
            'self_overlap_robust': self.create_self_overlap_constraint_robust,
            
            # Soft constraints
            'lift_minimum': self.create_lift_minimum_constraint,
            'drag_maximum': self.create_drag_maximum_constraint,
        }
        
        if constraint_type not in constraint_map:
            raise ValueError(f"Unknown constraint type: {constraint_type}. Available types: {list(constraint_map.keys())}")
        
        return constraint_map[constraint_type](**kwargs)


class ConstraintSuiteBuilder:
    """Builder pattern for creating complete constraint suites"""
    
    def __init__(self, aero_results: Dict[str, Any]):
        self.factory = ConstraintFactory(aero_results)
        self.hard_constraints = []
        self.soft_constraints = []
    
    def add_hard_constraint(self, constraint_type: str, **kwargs):
        """Add a hard constraint to the suite"""
        constraint = self.factory.create_constraint(constraint_type, **kwargs)
        constraint.metadata['type'] = 'hard'  # Ensure it's marked as hard
        self.hard_constraints.append(constraint)
        return self
    
    def add_soft_constraint(self, constraint_type: str, **kwargs):
        """Add a soft constraint to the suite"""
        constraint = self.factory.create_constraint(constraint_type, **kwargs)
        constraint.metadata['type'] = 'soft'  # Ensure it's marked as soft
        self.soft_constraints.append(constraint)
        return self
    
    def build(self):
        """Build and return the constraint suite"""
        return {
            'hard': self.hard_constraints.copy(),
            'soft': self.soft_constraints.copy(),
            'all': self.hard_constraints + self.soft_constraints
        }
    
    def get_constraint_info(self):
        """Get information about all constraints in the suite"""
        info = {
            'hard_constraints': [c.metadata for c in self.hard_constraints],
            'soft_constraints': [c.metadata for c in self.soft_constraints]
        }
        return info


class ConstraintEvaluator:
    """Evaluates constraint suites and combines violations"""
    
    def __init__(self, hard_constraints: List[Callable], soft_constraints: List[Callable]):
        self.hard_constraints = hard_constraints
        self.soft_constraints = soft_constraints
    
    def evaluate(self, cst_parameters) -> Tuple[float, bool]:
        """
        Evaluate all constraints and return penalty and hard violation status
        
        Returns:
            tuple (float, bool): Returns a tuple of (penalty, hard_constraint_violated)
        """
        hard_violation = False
        total_penalty = 0.0
        
        # Loop through all the hard constraints and check if they are violated
        # As soon as True is returned, the loop ends (because as soon as one hard constraint is violated, 
        # the wing should be completely discarded)
        for constraint in self.hard_constraints:
            try:
                result = constraint(cst_parameters)
                
                # If the result of the function is true, a hard constraint has been violated 
                # and more computation is not required
                if result == True:
                    hard_violation = True
                    break
            except Exception as e:
                constraint_name = getattr(constraint, 'metadata', {}).get('name', 'unknown')
                logging.error(f"Error in hard constraint {constraint_name}: {e}")
                hard_violation = True
                break
        
        # Evaluate soft constraints only if no hard constraints are violated
        if not hard_violation:
            for constraint in self.soft_constraints:
                try:
                    penalty = constraint(cst_parameters)
                    if penalty is not None:
                        total_penalty += abs(penalty)  # Take absolute value for penalty
                except Exception as e:
                    constraint_name = getattr(constraint, 'metadata', {}).get('name', 'unknown')
                    logging.error(f"Error in soft constraint {constraint_name}: {e}")
                    # Add a large penalty for failed constraint evaluation
                    total_penalty += 1000.0
                    logging.warning(f"Soft constraint {constraint_name} evaluation failed with penalty: 1000.0")        
        return total_penalty, hard_violation


def create_constraint_suite_simple(aero_results: Dict[str, Any], epoch: int = 0) -> Tuple[List[Callable], List[Callable]]:
    """
    Simple factory function approach for creating constraint suites
    
    Args:
        aero_results: Dictionary containing aerodynamic results
        epoch: Current epoch (for future use in progressive constraint application)
        
    Returns:
        Tuple of (hard_constraints, soft_constraints)
    """
    factory = ConstraintFactory(aero_results)
    
    # Create hard constraints
    hard_constraints = [
        factory.create_robust_airfoil_overlap_constraint(),
        factory.create_trailing_edge_constraint(), 
    ]
    
    # Create soft constraints with captured aero results
    soft_constraints = [
        factory.create_lift_minimum_constraint(),
        factory.create_drag_maximum_constraint()
    ]
    
    return hard_constraints, soft_constraints


def create_constraint_suite_advanced(aero_results: Dict[str, Any], epoch: int = 0) -> Tuple[List[Callable], List[Callable]]:
    """
    Advanced builder pattern approach for creating constraint suites
    
    Args:
        aero_results: Dictionary containing aerodynamic results
        epoch: Current epoch (for future progressive constraint application)
        
    Returns:
        Tuple of (hard_constraints, soft_constraints)
    """
    builder = ConstraintSuiteBuilder(aero_results)
    
    # Build constraint suite using fluent interface
    suite = (builder
             .add_hard_constraint('self_overlap')
             .add_hard_constraint('trailing_edge')
             .add_soft_constraint('lift_minimum', minimum=0.1)
             .add_soft_constraint('drag_maximum', maximum=0.1)
             .build())
    
    return suite['hard'], suite['soft']


def internal_run_constraint_suite(cst_parameters, aero_results, epoch, N=200):
    """
    Returns a true or false on the penalty and whether or not a hard contraint has been violated 

    Runs all the different contraints that I will have created in the future, computing the penalty for soft contraints (how much these constraints have been violated and modifying the score of the airfoil accordingly), and finding out if any hard constraints have been violated (meaning the airfoil is essentially unusable), then multiplying the score by -1 to highly discourage these airfoils

    Args:
        cst_parameters (dict): The CST parameters of the airfoil being checked
        aero_results (dict): The aerodynamic results from Neural Foil
        epoch (int): The current epoch of the optimization, used to figure out how advanced a suite of constraints to run (save on computation time)
        N (int, optional): Points that will be checked. Defaults to 200.
        
    Returns:
        tuple (float, bool): Returns a tuple of (penalty, hard_constraint_violated)
    """
    
    # Create constraint suite using the factory pattern
    hard_constraints, soft_constraints = create_constraint_suite_simple(aero_results, epoch)
    
    # Create evaluator and run constraints
    evaluator = ConstraintEvaluator(hard_constraints, soft_constraints)
    penalty, hard_violation = evaluator.evaluate(cst_parameters)
    
    return penalty, hard_violation


# ============================================================================
# CONSTRAINT FUNCTIONS 
# ============================================================================

# ---------------- HARD CONSTRAINTS ---------------- #
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

# Alternative implementation using area calculation
def check_overlap_via_area(cst_parameters, N=1000):
    """
    Check for overlap by ensuring the airfoil has positive area everywhere.
    This catches cases where the airfoil folds back on itself.
    """
    x = np.linspace(0, 1, N)
    
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
    except:
        return True
    
    # Calculate local "area" using trapezoidal integration segments
    for i in range(len(x) - 1):
        dx = x[i+1] - x[i]
        local_area = dx * (y_u[i] + y_u[i+1] - y_l[i] - y_l[i+1]) / 2
        
        # If local area becomes negative, we have an overlap
        if local_area < -1e-6:
            return True
    
    return False


def internal_self_overlap_simple(cst_parameters, N=200): 
    """
    Check if the airfoil overlaps itself, which is obviously a very big no go

    For some reason, when the program tries to optimize the wing it always makes it overlap, not sure why it does that but it is not okee dokee, so we're gonna have to check for that. I'm going to use this as a hard constraint because I'm pretty sure the moment it overlaps, the wing is cooked.

    Args:
        cst_parameters (dict): The CST parameters of the airfoil that we will be checking
        N (int, optional): Number of points that we will check (200 from upper, 200 from bottom, we verify each one is not overlap the corresponding point and go from there). Defaults to 200.
    """

    x = np.linspace(0, 1, N)

    try:
        TE_thickness_upper = cst_parameters.get('TE_thickness', 0) / 2.0
        TE_thickness_lower = cst_parameters.get('TE_thickness', 0) / 2.0
        leading_edge_weight = cst_parameters.get("leading_edge_weight", 0)
        
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
    except Exception as e:
        logging.error(f"Failed to generate coordinates: {e}")
        return True
    
    # Simple overlap check
    if np.any(y_u <= y_l):
        return True
    
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


# ---------------- SOFT CONSTRAINTS ---------------- #

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