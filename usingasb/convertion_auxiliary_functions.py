import numpy as np

def kulfan_dict_to_array(kulfan_dict) -> np.ndarray:
    """
    Converts a Kulfan parameter dictionary to a 1D numpy array.
    """
    return np.concatenate([
        kulfan_dict['lower_weights'],
        kulfan_dict['upper_weights'],
        [kulfan_dict['leading_edge_weight']],
        [kulfan_dict['TE_thickness']]
    ])


def array_to_kulfan_dict(array):
    """
    Converts a 1D numpy array back to a Kulfan parameter dictionary.
    Assumes:
    - 8 lower weights
    - 8 upper weights
    - 1 leading_edge_weight
    - 1 TE_thickness
    """
    return {
        'lower_weights': array[0:8],
        'upper_weights': array[8:16],
        'leading_edge_weight': array[16],
        'TE_thickness': array[17]
    }
