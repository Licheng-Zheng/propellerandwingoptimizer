#!/usr/bin/env python3
"""
Debug script to identify why constraints are failing
"""

import numpy as np
import matplotlib.pyplot as plt
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters, get_kulfan_coordinates
import aerosandbox as asb
from convertion_auxiliary_functions import array_to_kulfan_dict, kulfan_dict_to_array

def debug_coordinate_generation():
    """Debug coordinate generation step by step"""
    
    print("🔍 Debugging Coordinate Generation")
    print("="*50)
    
    # Start with a known good airfoil
    good_airfoil = asb.Airfoil("naca4412")
    good_params = get_kulfan_parameters(good_airfoil.coordinates)
    
    print(f"Original NACA 4412 parameters:")
    for key, value in good_params.items():
        print(f"  {key}: {value} (type: {type(value)})")
    
    print(f"\nTesting coordinate generation...")
    
    try:
        # Test the coordinate generation
        coordinates = get_kulfan_coordinates(
            lower_weights=good_params["lower_weights"],
            upper_weights=good_params["upper_weights"],
            leading_edge_weight=good_params.get("leading_edge_weight", 0),
            TE_thickness=good_params.get('TE_thickness', 0),
            N1=0.5,
            N2=1.0,
            n_points_per_side=200
        )
        
        print(f"✅ Generated {len(coordinates)} coordinate points")
        print(f"Coordinate shape: {coordinates.shape}")
        print(f"X range: [{np.min(coordinates[:, 0]):.6f}, {np.max(coordinates[:, 0]):.6f}]")
        print(f"Y range: [{np.min(coordinates[:, 1]):.6f}, {np.max(coordinates[:, 1]):.6f}]")
        
        # Split coordinates into upper and lower surfaces
        n_mid = len(coordinates) // 2
        upper_surface = coordinates[:n_mid]
        lower_surface = coordinates[n_mid:]
        
        print(f"\nSurface analysis:")
        print(f"Upper surface points: {len(upper_surface)}")
        print(f"Lower surface points: {len(lower_surface)}")
        print(f"Upper X range: [{np.min(upper_surface[:, 0]):.6f}, {np.max(upper_surface[:, 0]):.6f}]")
        print(f"Lower X range: [{np.min(lower_surface[:, 0]):.6f}, {np.max(lower_surface[:, 0]):.6f}]")
        
        # Check if upper surface needs reversal
        print(f"\nUpper surface X ordering check:")
        print(f"First 5 upper X: {upper_surface[:5, 0]}")
        print(f"Last 5 upper X: {upper_surface[-5:, 0]}")
        
        if upper_surface[0, 0] > upper_surface[-1, 0]:
            print("⚠️  Upper surface is in reverse order (x decreasing)")
            upper_surface = upper_surface[::-1]
            print("✅ Reversed upper surface")
        else:
            print("✅ Upper surface is in correct order (x increasing)")
        
        print(f"\nLower surface X ordering check:")
        print(f"First 5 lower X: {lower_surface[:5, 0]}")
        print(f"Last 5 lower X: {lower_surface[-5:, 0]}")
        
        # Interpolate both surfaces to common x points for comparison
        x_common = np.linspace(0, 1, 100)
        y_u = np.interp(x_common, upper_surface[:, 0], upper_surface[:, 1])
        y_l = np.interp(x_common, lower_surface[:, 0], lower_surface[:, 1])
        
        # Check thickness
        thickness = y_u - y_l
        min_thickness = np.min(thickness)
        max_thickness = np.max(thickness)
        
        print(f"\nThickness analysis:")
        print(f"Min thickness: {min_thickness:.8f}")
        print(f"Max thickness: {max_thickness:.8f}")
        print(f"Mean thickness: {np.mean(thickness):.8f}")
        
        # Check for negative thickness (overlap)
        negative_thickness_count = np.sum(thickness <= 0)
        print(f"Points with negative/zero thickness: {negative_thickness_count}")
        
        if negative_thickness_count > 0:
            print("❌ OVERLAP DETECTED!")
            negative_indices = np.where(thickness <= 0)[0]
            print(f"Negative thickness at x positions: {x_common[negative_indices]}")
            print(f"Thickness values: {thickness[negative_indices]}")
        else:
            print("✅ No overlap detected")
        
        # Plot for visual inspection
        plt.figure(figsize=(12, 8))
        
        # Plot airfoil
        plt.subplot(2, 2, 1)
        plt.plot(coordinates[:, 0], coordinates[:, 1], 'b-', linewidth=2, label='Full airfoil')
        plt.plot(upper_surface[:, 0], upper_surface[:, 1], 'r-', linewidth=1, alpha=0.7, label='Upper surface')
        plt.plot(lower_surface[:, 0], lower_surface[:, 1], 'g-', linewidth=1, alpha=0.7, label='Lower surface')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.title('Airfoil Shape')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Plot thickness distribution
        plt.subplot(2, 2, 2)
        plt.plot(x_common, thickness, 'b-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('x/c')
        plt.ylabel('Thickness')
        plt.title('Thickness Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot upper and lower surfaces separately
        plt.subplot(2, 2, 3)
        plt.plot(x_common, y_u, 'r-', linewidth=2, label='Upper surface')
        plt.plot(x_common, y_l, 'g-', linewidth=2, label='Lower surface')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.title('Surface Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot difference (should be positive everywhere)
        plt.subplot(2, 2, 4)
        plt.plot(x_common, y_u - y_l, 'b-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('x/c')
        plt.ylabel('y_upper - y_lower')
        plt.title('Surface Difference (Thickness)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return coordinates, upper_surface, lower_surface, thickness
        
    except Exception as e:
        print(f"❌ Error in coordinate generation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def debug_simple_overlap_check():
    """Test a very simple overlap check"""
    
    print("\n🔍 Debugging Simple Overlap Check")
    print("="*50)
    
    # Create a simple test case
    x = np.linspace(0, 1, 10)
    y_upper = np.array([0.0, 0.05, 0.08, 0.10, 0.11, 0.10, 0.08, 0.05, 0.02, 0.0])
    y_lower = np.array([0.0, -0.02, -0.03, -0.04, -0.04, -0.03, -0.02, -0.01, 0.0, 0.0])
    
    print(f"Test case:")
    print(f"X: {x}")
    print(f"Y_upper: {y_upper}")
    print(f"Y_lower: {y_lower}")
    
    thickness = y_upper - y_lower
    print(f"Thickness: {thickness}")
    
    min_thickness = np.min(thickness)
    print(f"Min thickness: {min_thickness}")
    
    overlap = np.any(y_upper <= y_lower)
    print(f"Overlap detected (y_upper <= y_lower): {overlap}")
    
    # Test with tolerance
    tolerance = 1e-6
    overlap_with_tolerance = np.any(y_upper <= y_lower + tolerance)
    print(f"Overlap with tolerance {tolerance}: {overlap_with_tolerance}")
    
    return overlap

if __name__ == "__main__":
    print("🚀 Starting Constraint Debug Session")
    print("="*80)
    
    # Test simple overlap check first
    debug_simple_overlap_check()
    
    # Test coordinate generation
    coords, upper, lower, thickness = debug_coordinate_generation()
    
    print("\n🎉 Debug session completed!")