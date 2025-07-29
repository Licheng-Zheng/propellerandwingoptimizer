import neuralfoil
import neuralfoil as nf

def compute_result(cst_parameters, alpha, Re, model, wanted_lists:dict, importance_list:dict):
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
    print("SCORE TYPE", type(score))
    print(score)

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
