#!/usr/bin/env python3
"""
Simple NeuralFoil Kulfan Parameters Example
Just the basics - using Kulfan parameters directly with NeuralFoil
"""

import numpy as np
import neuralfoil as nf

def main():
    print("=== Simple NeuralFoil Kulfan Example ===\n")
    
    # Define Kulfan parameters for a simple airfoil
    # NeuralFoil expects 18 parameters total:
    # - 8 for upper surface
    # - 8 for lower surface  
    # - 1 for leading edge modification (LEM)
    # - 1 for trailing edge thickness
    
    # Example 1: Symmetric airfoil (like NACA 0012)
    upper_weights = np.array([0.1, 0.15, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001])
    lower_weights = np.array([-0.1, -0.15, -0.1, -0.05, -0.02, -0.01, -0.005, -0.001])
    leading_edge_weight = 0.0
    trailing_edge_thickness = 0.002
    
    # Combine all parameters into single array
    kulfan_params = np.concatenate([
        upper_weights,
        lower_weights,
        [leading_edge_weight],
        [trailing_edge_thickness]
    ])
    
    print("Kulfan parameters:")
    print(f"Upper weights: {upper_weights}")
    print(f"Lower weights: {lower_weights}")
    print(f"Leading edge weight: {leading_edge_weight}")
    print(f"Trailing edge thickness: {trailing_edge_thickness}")
    print(f"Total parameters: {len(kulfan_params)}")
    
    # Analysis conditions
    alpha = 5.0      # angle of attack [deg]
    Re = 1e6         # Reynolds number
    mach = 0.2       # Mach number
    
    print(f"\nAnalysis conditions:")
    print(f"Alpha: {alpha}°")
    print(f"Reynolds number: {Re:.0e}")
    print(f"Mach number: {mach}")
    
    # Run NeuralFoil analysis
    try:
        result = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kulfan_params,
            alpha=alpha,
            Re=Re
        )
        
        print(f"\n✓ Analysis successful!")
        print(f"Result type: {type(result)}")
        print(f"Result shape/length: {result.shape if hasattr(result, 'shape') else len(result) if hasattr(result, '__len__') else 'N/A'}")
        print(f"Result contents: {result}")
        
        # Try different ways to access the results
        if hasattr(result, 'shape') and len(result.shape) == 1:
            # It's a 1D array, likely [Cl, Cd, Cm]
            if len(result) >= 3:
                Cl, Cd, Cm = result[0], result[1], result[2]
                print(f"Lift coefficient (Cl): {Cl:.4f}")
                print(f"Drag coefficient (Cd): {Cd:.4f}")
                print(f"Moment coefficient (Cm): {Cm:.4f}")
                print(f"L/D ratio: {Cl/Cd:.2f}")
                return {'Cl': Cl, 'Cd': Cd, 'Cm': Cm}
        elif isinstance(result, dict):
            # It's a dictionary
            print(f"Available keys: {list(result.keys())}")
            Cl = result.get('Cl', result.get('CL', result.get('cl', 0)))
            Cd = result.get('Cd', result.get('CD', result.get('cd', 0)))
            Cm = result.get('Cm', result.get('CM', result.get('cm', 0)))
            print(f"Lift coefficient (Cl): {Cl:.4f}")
            print(f"Drag coefficient (Cd): {Cd:.4f}")
            print(f"Moment coefficient (Cm): {Cm:.4f}")
            print(f"L/D ratio: {Cl/Cd:.2f}")
            return {'Cl': Cl, 'Cd': Cd, 'Cm': Cm}
        else:
            print("Unknown result format")
            return None
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def test_different_airfoils():
    """Test a few different airfoil shapes"""
    
    print("\n=== Testing Different Airfoils ===")
    
    # Test cases
    test_cases = {
        'Symmetric': {
            'upper': np.array([0.1, 0.15, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]),
            'lower': np.array([-0.1, -0.15, -0.1, -0.05, -0.02, -0.01, -0.005, -0.001]),
            'le': 0.0,
            'te': 0.002
        },
        'Cambered': {
            'upper': np.array([0.12, 0.18, 0.12, 0.06, 0.025, 0.012, 0.006, 0.002]),
            'lower': np.array([-0.08, -0.12, -0.08, -0.04, -0.015, -0.008, -0.004, -0.001]),
            'le': 0.05,
            'te': 0.002
        },
        'Thick': {
            'upper': np.array([0.15, 0.2, 0.15, 0.08, 0.035, 0.015, 0.008, 0.003]),
            'lower': np.array([-0.15, -0.2, -0.15, -0.08, -0.035, -0.015, -0.008, -0.003]),
            'le': 0.0,
            'te': 0.003
        }
    }
    
    # Analysis conditions
    alpha = 5.0
    Re = 1e6
    mach = 0.2
    
    print(f"{'Airfoil':<10} {'Cl':<8} {'Cd':<8} {'Cm':<8} {'L/D':<8}")
    print("-" * 50)
    
    for name, params in test_cases.items():
        # Create parameter array
        kulfan_params = np.concatenate([
            params['upper'],
            params['lower'],
            [params['le']],
            [params['te']]
        ])
        
        try:
            result = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=kulfan_params,
                alpha=alpha,
                Re=Re
            )
            
            # Handle array or dict result
            if hasattr(result, 'shape') and len(result.shape) == 1 and len(result) >= 3:
                Cl, Cd, Cm = result[0], result[1], result[2]
                print(f"{name:<10} {Cl:<8.4f} {Cd:<8.4f} {Cm:<8.4f} {Cl/Cd:<8.2f}")
            elif isinstance(result, dict):
                Cl = result.get('Cl', result.get('CL', result.get('cl', 0)))
                Cd = result.get('Cd', result.get('CD', result.get('cd', 0)))
                Cm = result.get('Cm', result.get('CM', result.get('cm', 0)))
                print(f"{name:<10} {Cl:<8.4f} {Cd:<8.4f} {Cm:<8.4f} {Cl/Cd:<8.2f}")
            else:
                print(f"{name:<10} Unknown result format: {type(result)}")
            
        except Exception as e:
            print(f"{name:<10} ERROR: {e}")

def parameter_sweep():
    """Simple parameter sweep example"""
    
    print("\n=== Parameter Sweep Example ===")
    
    # Base symmetric airfoil
    base_upper = np.array([0.1, 0.15, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001])
    base_lower = -base_upper  # Symmetric
    
    # Sweep the first parameter (controls leading edge shape)
    param_values = np.linspace(0.05, 0.2, 5)
    
    print("Sweeping first Kulfan parameter:")
    print(f"{'Param':<8} {'Cl':<8} {'Cd':<8} {'L/D':<8}")
    print("-" * 35)
    
    for param_val in param_values:
        # Modify first parameter
        upper_weights = base_upper.copy()
        upper_weights[0] = param_val
        lower_weights = -upper_weights  # Keep symmetric
        
        kulfan_params = np.concatenate([
            upper_weights,
            lower_weights,
            [0.0],    # LE weight
            [0.002]   # TE thickness
        ])
        
        try:
            result = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=kulfan_params,
                alpha=5.0,
                Re=1e6
            )
            
            # Handle array or dict result
            if hasattr(result, 'shape') and len(result.shape) == 1 and len(result) >= 3:
                Cl, Cd, Cm = result[0], result[1], result[2]
                print(f"{param_val:<8.3f} {Cl:<8.4f} {Cd:<8.4f} {Cl/Cd:<8.2f}")
            elif isinstance(result, dict):
                Cl = result.get('Cl', result.get('CL', result.get('cl', 0)))
                Cd = result.get('Cd', result.get('CD', result.get('cd', 0)))
                print(f"{param_val:<8.3f} {Cl:<8.4f} {Cd:<8.4f} {Cl/Cd:<8.2f}")
            else:
                print(f"{param_val:<8.3f} Unknown result format: {type(result)}")
            
        except Exception as e:
            print(f"{param_val:<8.3f} ERROR: {e}")

if __name__ == "__main__":
    # Run basic example
    main()
    
    # Run additional tests
    test_different_airfoils()
    parameter_sweep()
    
    print("\n=== Usage Summary ===")
    print("• Use np.concatenate() to combine all 18 parameters")
    print("• Order: [upper_8, lower_8, le_weight, te_thickness]")
    print("• Call nf.get_aero_from_kulfan_parameters()")
    print("• Returns dict with 'Cl', 'Cd', 'Cm' keys")