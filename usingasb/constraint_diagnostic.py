#!/usr/bin/env python3
"""
Diagnostic script to test constraint functions and identify optimization issues
"""

import numpy as np
import os
import matplotlib
import copy
if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters, get_kulfan_coordinates
import aerosandbox as asb
from convertion_auxiliary_functions import array_to_kulfan_dict, kulfan_dict_to_array
from objective import (
    robust_airfoil_overlap_check, 
    trailing_edge_mismatch,
    internal_run_constraint_suite,
    compute_result
)
import display_auxiliary_functions

def create_test_airfoils():
    """Create test airfoils with known issues"""
    
    # Good airfoil (NACA 4412)
    good_airfoil = asb.Airfoil("naca4412")
    good_params = get_kulfan_parameters(good_airfoil.coordinates)
    
    # Bad airfoil 1: Overlapping surfaces
    bad_params_overlap = good_params.copy()
    bad_params_overlap['upper_weights'] = np.array([-0.5, -0.8, -0.6, -0.4, -0.3, 0.2, 0.1, 0.05])
    bad_params_overlap['lower_weights'] = np.array([0.5, 0.8, 0.6, 0.4, 0.3, -0.2, -0.1, -0.05])
    
    # Bad airfoil 2: Large trailing edge gap
    bad_params_te = good_params.copy()
    bad_params_te['TE_thickness'] = 0.1  # Large trailing edge thickness
    
    # Bad airfoil 3: Extreme parameters
    bad_params_extreme = good_params.copy()
    bad_params_extreme['upper_weights'] = np.array([2.0, -2.0, 1.5, -1.5, 1.0, -1.0, 0.5, -0.5])
    bad_params_extreme['lower_weights'] = np.array([-2.0, 2.0, -1.5, 1.5, -1.0, 1.0, -0.5, 0.5])
    bad_params_extreme['leading_edge_weight'] = 2.0
    
    return {
        'good': good_params,
        'overlap': bad_params_overlap,
        'trailing_edge': bad_params_te,
        'extreme': bad_params_extreme
    }

def test_constraint_function(constraint_func, params_dict, constraint_name):
    """Test a constraint function on different airfoils"""
    
    print(f"\n{'='*60}")
    print(f"Testing {constraint_name}")
    print(f"{'='*60}")
    
    results = {}
    
    for name, params in params_dict.items():
        try:
            print(f"\nTesting {name} airfoil:")
            print(f"  Parameters: {params}")
            
            # Test the constraint
            if constraint_name == "trailing_edge_mismatch":
                result = constraint_func(params, tol=1e-4, N=300)
            else:
                result = constraint_func(params)
            
            results[name] = result
            status = "VIOLATION" if result else "OK"
            print(f"  Result: {result} ({status})")
            
            # Try to plot the airfoil
            try:
                display_auxiliary_functions.plot_cst_airfoil(
                    params, 
                    title=f"{constraint_name} - {name} airfoil",
                    show=False,
                    block=False
                )
                plt.close()  # Close to prevent too many plots
                print(f"  ✅ Airfoil plotted successfully")
            except Exception as plot_error:
                print(f"  ❌ Plot failed: {plot_error}")
                
        except Exception as e:
            print(f"  ❌ Constraint test failed: {e}")
            results[name] = f"ERROR: {e}"
    
    return results

def test_coordinate_generation(params_dict):
    """Test coordinate generation for different airfoils"""
    
    print(f"\n{'='*60}")
    print(f"Testing Coordinate Generation")
    print(f"{'='*60}")
    
    for name, params in params_dict.items():
        print(f"\nTesting {name} airfoil coordinate generation:")
        
        try:
            # Test coordinate generation
            coordinates = get_kulfan_coordinates(
                lower_weights=params['lower_weights'],
                upper_weights=params['upper_weights'],
                leading_edge_weight=params['leading_edge_weight'],
                TE_thickness=params['TE_thickness'],
                N1=0.5,
                N2=1.0,
                n_points_per_side=200
            )
            
            print(f"  ✅ Generated {len(coordinates)} points")
            
            # Check for basic issues
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]
            
            print(f"  X range: [{np.min(x_coords):.4f}, {np.max(x_coords):.4f}]")
            print(f"  Y range: [{np.min(y_coords):.4f}, {np.max(y_coords):.4f}]")
            
            # Check for overlaps manually
            n_mid = len(coordinates) // 2
            upper_surface = coordinates[:n_mid]
            lower_surface = coordinates[n_mid:]
            
            # Simple overlap check
            x_common = np.linspace(0, 1, 100)
            y_upper = np.interp(x_common, upper_surface[::-1, 0], upper_surface[::-1, 1])
            y_lower = np.interp(x_common, lower_surface[:, 0], lower_surface[:, 1])
            
            thickness = y_upper - y_lower
            min_thickness = np.min(thickness)
            
            print(f"  Min thickness: {min_thickness:.6f}")
            
            if min_thickness <= 0:
                print(f"  ❌ OVERLAP DETECTED: Minimum thickness is {min_thickness}")
            else:
                print(f"  ✅ No overlap detected")
                
            # Check trailing edge
            te_gap = abs(coordinates[0, 1] - coordinates[-1, 1])
            print(f"  Trailing edge gap: {te_gap:.6f}")
            
        except Exception as e:
            print(f"  ❌ Coordinate generation failed: {e}")

def test_optimization_objective(params_dict):
    """Test the optimization objective function"""
    
    print(f"\n{'='*60}")
    print(f"Testing Optimization Objective")
    print(f"{'='*60}")
    
    # Test parameters
    alpha = 5.0
    Re = 1e6
    model_size = "large"
    wanted_lists = ["analysis_confidence", "CL", "CD", "CM"]
    importance_list = [0.4, 0.3, -0.2, -0.1]
    epoch = 0
    
    for name, params in params_dict.items():
        print(f"\nTesting {name} airfoil objective:")
        
        try:
            fitness = compute_result(
                cst_parameters=params,
                epoch=epoch,
                alpha=alpha,
                Re=Re,
                model=model_size,
                wanted_lists=wanted_lists,
                importance_list=importance_list
            )
            
            print(f"  Fitness: {fitness}")
            
            if fitness == 1e6:
                print(f"  ❌ HARD CONSTRAINT VIOLATION (fitness = 1e6)")
            elif fitness > 1000:
                print(f"  ⚠️  High penalty (fitness > 1000)")
            else:
                print(f"  ✅ Normal fitness value")
                
        except Exception as e:
            print(f"  ❌ Objective evaluation failed: {e}")

def test_constraint_suite(params_dict):
    """Test the complete constraint suite"""
    
    print(f"\n{'='*60}")
    print(f"Testing Complete Constraint Suite")
    print(f"{'='*60}")
    
    # Mock aero results for constraint suite
    mock_aero_results = {
        "CL": 0.8,
        "CD": 0.02,
        "CM": -0.1,
        "analysis_confidence": 0.95
    }
    
    for name, params in params_dict.items():
        print(f"\nTesting {name} airfoil constraint suite:")
        
        try:
            penalty, hard_violation = internal_run_constraint_suite(
                cst_parameters=params,
                aero_results=mock_aero_results,
                epoch=0
            )
            
            print(f"  Penalty: {penalty}")
            print(f"  Hard violation: {hard_violation}")
            
            if hard_violation:
                print(f"  ❌ HARD CONSTRAINT VIOLATED")
            elif penalty > 0:
                print(f"  ⚠️  Soft constraint penalty: {penalty}")
            else:
                print(f"  ✅ All constraints satisfied")
                
        except Exception as e:
            print(f"  ❌ Constraint suite failed: {e}")

def main():
    """Run all diagnostic tests"""
    
    print("🔍 Starting Constraint Diagnostic Tests")
    print("="*80)
    
    # Create test airfoils
    test_airfoils = create_test_airfoils()
    
    # Test coordinate generation
    test_coordinate_generation(test_airfoils)
    
    # Test individual constraint functions
    test_constraint_function(
        robust_airfoil_overlap_check, 
        test_airfoils, 
        "robust_airfoil_overlap_check"
    )
    
    test_constraint_function(
        trailing_edge_mismatch, 
        test_airfoils, 
        "trailing_edge_mismatch"
    )
    
    # Test constraint suite
    test_constraint_suite(test_airfoils)
    
    # Test optimization objective
    test_optimization_objective(test_airfoils)
    
    print(f"\n{'='*80}")
    print("🎉 Diagnostic tests completed!")
    print("="*80)

if __name__ == "__main__":
    main()