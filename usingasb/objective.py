import neuralfoil as nf
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
import numpy as np
import logging
from typing import List, Callable, Dict, Any, Tuple

# Calling every function in explicitedly so I can ensure that only required functions are imported
from constraints.hard_constraints_1 import (robust_airfoil_overlap_check, 
                                            trailing_edge_mismatch)
from constraints.soft_constraints_1 import (internal_minimum_thickness_constraint, 
                                            internal_lift_coefficient_minimum, 
                                            internal_drag_coefficient_maximum)

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

    def create_self_overlap_constraint_robust(self, N: int = 1000):
        """Factory for self-overlap constraint"""
        def constraint(cst_parameters):
            return robust_airfoil_overlap_check(cst_parameters, N=N)
        
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
    def create_internal_minimum_thickness_constraint(self, minimum_thickness: float = None, max_x: float = 0.9):
        """Factory for internal minimum thickness constraint"""
        
        def constraint(cst_parameters):
            return internal_minimum_thickness_constraint(cst_parameters=cst_parameters, minimum_thickness=minimum_thickness, check_up_to_x=max_x, N=1000)
    
        constraint.metadata = {
            'type': 'soft',
            'name': 'internal_minimum_thickness',
            'minimum': minimum_thickness,
            'max_x': max_x,
            'N': 1000
        }

        return constraint
    
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
            'self_overlap': self.create_self_overlap_constraint_robust,
            'trailing_edge': self.create_trailing_edge_constraint,
            
            # Soft constraints
            'internal_minimum_thickness': self.create_internal_minimum_thickness_constraint,
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
                    total_penalty += 0.4
                    
                    # SWITCHING LOGIC FOR HARD CONSTRAINTS: ADD 0.5 TO PENALTY INSTEAD OF BREAK
                    #hard_violation = True
                    # # break
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
    Simple factory function approach for creating constraint suites using builder pattern
    
    Args:
        aero_results: Dictionary containing aerodynamic results
        epoch: Current epoch (for future use in progressive constraint application)
        
    Returns:
        Tuple of (hard_constraints, soft_constraints)
    """
    builder = ConstraintSuiteBuilder(aero_results)
    
    # Build constraint suite using fluent interface
    suite = (builder
             .add_hard_constraint('self_overlap')
             .add_hard_constraint('trailing_edge')
             .add_soft_constraint('internal_minimum_thickness', minimum_thickness=0.01, max_x=0.9)
             .add_soft_constraint('lift_minimum')
             .add_soft_constraint('drag_maximum')
             .build())
    
    return suite['hard'], suite['soft']


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