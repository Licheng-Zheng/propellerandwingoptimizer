import aerosandbox as asb

def fit_kulfan_params(coordinates, n_weights_per_side=8, N1=0.5, N2=1.0):
    airfoil = asb.Airfoil(name="Target Airfoil", coordinates=coordinates)
    kulfan_params = asb.geometry.airfoil.get_kulfan_parameters(
        coordinates=airfoil.normalize().coordinates,
        n_weights_per_side=n_weights_per_side,
        N1=N1,
        N2=N2
    )
    # kulfan_params['N1'] = N1
    # kulfan_params['N2'] = N2
    return kulfan_params

def reconstruct_coordinates_from_kulfan(kulfan_params, n_points_per_side=100):
    coords = asb.geometry.airfoil.airfoil_families.get_kulfan_coordinates(
        upper_weights=kulfan_params['upper_weights'],
        lower_weights=kulfan_params['lower_weights'],
        N1=kulfan_params['N1'],
        N2=kulfan_params['N2'],
        leading_edge_weight=kulfan_params['leading_edge_weight'],
        TE_thickness=kulfan_params['TE_thickness'],
        n_points_per_side=n_points_per_side
    )
    return coords
