import numpy as np 
import matplotlib.pyplot as plt
import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

def plot_cst_airfoil(cst_params, n_points_per_side=200, title="CST Airfoil", show_params=True, 
                     save_path=None, figsize=(12, 8), show=True, block=False):
    """
    Plots the airfoil using CST(Class Shape Transformation)/Kulfan Parameters. Aerosandbox hard carries this function.

    Takes your Aerosandbox CST parameters and plots the airfoil shape, provides additional information if required such as the maximum width, and a couple details that might be useful. 


    Args:
        cst_params (dict): Dictionary containing:
        - 'lower_weights': array of lower surface CST weights
        - 'upper_weights': array of upper surface CST weights  
        - 'leading_edge_weight': leading edge weight (scalar)
        - 'TE_thickness': trailing edge thickness (scalar)

        n_points_per_side (int, optional): Number of points to generate for each surface. Defaults to 200.

        title (str, optional): Plot Title. Defaults to "CST Airfoil".
        show_params (bool, optional): Whether to display parameter information. Defaults to True.

        save_path (str, optional): Path to save the plot. Defaults to None.
        
        figsize (tuple, optional): Figure size (width, height). Defaults to (12, 8).

        show (bool, optional): Whether to call plt.show() after plotting. Defaults to True.

    Returns: 
    airfoil : asb.Airfoil
        AeroSandbox airfoil object
    coordinates : np.ndarray
        Airfoil coordinates as [x, y] array
    """

    # Extract parameters
    lower_weights = np.array(cst_params['lower_weights'])
    upper_weights = np.array(cst_params['upper_weights'])
    leading_edge_weight = float(cst_params['leading_edge_weight'])
    te_thickness = float(cst_params['TE_thickness'])
    
    # Use AeroSandbox's get_kulfan_coordinates function (because you can't do it yourself cas you're a dumbass)
    coordinates = get_kulfan_coordinates(
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        leading_edge_weight=leading_edge_weight,
        TE_thickness=te_thickness,
        N1=0.5,
        N2=1.0,
        n_points_per_side=n_points_per_side
    )

    # Create AeroSandbox airfoil object
    airfoil = asb.Airfoil(coordinates=coordinates)
    
    # Create the fancy plot with all the cool features
    fig, axes = plt.subplots(1, 2 if show_params else 1, figsize=figsize)
    
    if show_params:
        ax_airfoil = axes[0]
        ax_params = axes[1]
    else:
        ax_airfoil = axes if hasattr(axes, 'plot') else axes
    
    # Main airfoil plot with enhanced styling
    ax_airfoil.plot(coordinates[:, 0], coordinates[:, 1], 'b-', linewidth=2.5, label='Airfoil', zorder=3)
    ax_airfoil.fill(coordinates[:, 0], coordinates[:, 1], alpha=0.3, color='lightblue', zorder=2)
    
    # Add reference lines and grid
    ax_airfoil.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=0.8, zorder=1)
    ax_airfoil.axvline(x=0, color='k', linestyle='--', alpha=0.4, linewidth=0.8, zorder=1)
    ax_airfoil.axvline(x=1, color='k', linestyle='--', alpha=0.4, linewidth=0.8, zorder=1)
    ax_airfoil.grid(True, alpha=0.3, linestyle=':', zorder=0)
    
    # Mark leading and trailing edges with enhanced markers
    le_idx = np.argmin(coordinates[:, 0])
    te_idx = np.argmax(coordinates[:, 0])
    
    ax_airfoil.plot(coordinates[le_idx, 0], coordinates[le_idx, 1], 'ro', markersize=10, 
                markeredgewidth=2, markeredgecolor='darkred', label='Leading Edge', zorder=4)
    ax_airfoil.plot(coordinates[te_idx, 0], coordinates[te_idx, 1], 'go', markersize=10,
                markeredgewidth=2, markeredgecolor='darkgreen', label='Trailing Edge', zorder=4)
    
    # Enhanced axis styling
    ax_airfoil.set_xlim(-0.05, 1.05)
    ax_airfoil.set_ylim(-0.1, 0.4)
    ax_airfoil.set_xlabel('x/c', fontsize=14, fontweight='bold')
    ax_airfoil.set_ylabel('y/c', fontsize=14, fontweight='bold')
    ax_airfoil.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax_airfoil.set_aspect('equal')
    ax_airfoil.legend(loc='upper right', framealpha=0.9, fontsize=11)
    
    # Calculate and display comprehensive geometric information
    try:
        # FIXED: Try new methods first, fallback to manual calculation
        try:
            max_thickness = airfoil.max_thickness()
            max_thick_loc = airfoil.max_thickness_location()
            max_camber = airfoil.max_camber()
            max_camber_loc = airfoil.max_camber_location()
            
            # Add thickness and camber visualization
            x_analysis = np.linspace(0, 1, 100)
            try:
                thickness_dist = [airfoil.local_thickness(x) for x in x_analysis]
                camber_dist = [airfoil.local_camber(x) for x in x_analysis]
                
                # Mark maximum thickness location
                ax_airfoil.axvline(x=max_thick_loc, color='orange', linestyle=':', alpha=0.7, 
                                label=f'Max Thickness @ x/c={max_thick_loc:.3f}')
                
                # Mark maximum camber location
                if abs(max_camber) > 1e-6:  # Only if there's significant camber
                    ax_airfoil.axvline(x=max_camber_loc, color='purple', linestyle=':', alpha=0.7,
                                    label=f'Max Camber @ x/c={max_camber_loc:.3f}')
            except:
                pass  # Skip if local methods don't work
            
            info_text = f'Max Thickness: {max_thickness:.4f} @ x/c = {max_thick_loc:.3f}\n'
            info_text += f'Max Camber: {max_camber:.4f} @ x/c = {max_camber_loc:.3f}\n'
            info_text += f'LE Weight: {leading_edge_weight:.4f}\n'
            info_text += f'TE Thickness: {te_thickness:.6f}'
            
        except AttributeError as attr_error:
            # FIXED: Handle missing methods gracefully
            raise attr_error  # Re-raise to trigger fallback
            
    except Exception as e:
        # Fallback geometric calculations for older AeroSandbox versions
        print(f"Using fallback geometry calculations: {e}")
        
        # Calculate thickness distribution manually
        n_mid = len(coordinates) // 2
        upper_surface = coordinates[:n_mid]
        lower_surface = coordinates[n_mid:]
        
        # Reverse upper surface to match x ordering
        upper_surface = upper_surface[::-1]
        
        # Find common x points and interpolate
        x_common = np.linspace(0, 1, 100)
        y_upper = np.interp(x_common, upper_surface[:, 0], upper_surface[:, 1])
        y_lower = np.interp(x_common, lower_surface[:, 0], lower_surface[:, 1])
        
        thickness = y_upper - y_lower
        camber = (y_upper + y_lower) / 2
        
        max_thickness = np.max(thickness)
        max_thick_loc = x_common[np.argmax(thickness)]
        max_camber = camber[np.argmax(np.abs(camber))]
        max_camber_loc = x_common[np.argmax(np.abs(camber))]
        
        # Add visualization for fallback calculations
        ax_airfoil.axvline(x=max_thick_loc, color='orange', linestyle=':', alpha=0.7, 
                        label=f'Max Thickness @ x/c={max_thick_loc:.3f}')
        
        if abs(max_camber) > 1e-6:
            ax_airfoil.axvline(x=max_camber_loc, color='purple', linestyle=':', alpha=0.7,
                            label=f'Max Camber @ x/c={max_camber_loc:.3f}')
        
        info_text = f'Max Thickness: {max_thickness:.4f} @ x/c = {max_thick_loc:.3f}\n'
        info_text += f'Max Camber: {max_camber:.4f} @ x/c = {max_camber_loc:.3f}\n'
        info_text += f'LE Weight: {leading_edge_weight:.4f}\n'
        info_text += f'TE Thickness: {te_thickness:.6f}'
    
    # Enhanced info box styling
    ax_airfoil.text(0.02, 0.98, info_text, transform=ax_airfoil.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, 
                        edgecolor='orange', linewidth=1.5))
    
    # Enhanced parameter visualization
    if show_params:
        # Create enhanced parameter comparison plot
        n_params = max(len(upper_weights), len(lower_weights))
        x_indices = np.arange(n_params)
        
        # Pad shorter array with zeros for plotting
        upper_padded = np.pad(upper_weights, (0, max(0, n_params - len(upper_weights))))
        lower_padded = np.pad(lower_weights, (0, max(0, n_params - len(lower_weights))))
        
        width = 0.35
        bars1 = ax_params.bar(x_indices - width/2, upper_padded, width, 
                            alpha=0.8, color='crimson', label='Upper Surface', 
                            edgecolor='darkred', linewidth=1)
        bars2 = ax_params.bar(x_indices + width/2, lower_padded, width,
                            alpha=0.8, color='steelblue', label='Lower Surface',
                            edgecolor='darkblue', linewidth=1)
        
        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if i < len(upper_weights) and abs(upper_weights[i]) > 0.01:
                height = bar1.get_height()
                ax_params.text(bar1.get_x() + bar1.get_width()/2., height + 0.005 if height > 0 else height - 0.015,
                            f'{upper_weights[i]:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                            fontsize=8, rotation=45)
            
            if i < len(lower_weights) and abs(lower_weights[i]) > 0.01:
                height = bar2.get_height()
                ax_params.text(bar2.get_x() + bar2.get_width()/2., height + 0.005 if height > 0 else height - 0.015,
                            f'{lower_weights[i]:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                            fontsize=8, rotation=45)
        
        ax_params.set_xlabel('CST Parameter Index', fontsize=14, fontweight='bold')
        ax_params.set_ylabel('Weight Value', fontsize=14, fontweight='bold')
        ax_params.set_title('CST Weight Parameters', fontsize=16, fontweight='bold', pad=20)
        ax_params.grid(True, alpha=0.3, linestyle=':')
        ax_params.legend(fontsize=12, framealpha=0.9)
        ax_params.axhline(y=0, color='black', linewidth=0.8)
        
        # Enhanced parameter info box
        param_text = f'Configuration Details:\n'
        param_text += f'├─ Leading Edge Weight: {leading_edge_weight:.4f}\n'
        param_text += f'├─ Trailing Edge Thickness: {te_thickness:.6f}\n'
        param_text += f'├─ Upper Surface Weights: {len(upper_weights)}\n'
        param_text += f'├─ Lower Surface Weights: {len(lower_weights)}\n'
        param_text += f'├─ N1 (Class): 0.5\n'
        param_text += f'└─ N2 (Class): 1.0'
        
        ax_params.text(0.02, 0.98, param_text, transform=ax_params.transAxes,
                    verticalalignment='top', fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9,
                            edgecolor='gray', linewidth=1.5))
    
    # Final plot adjustments
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Plot saved to: {save_path}")
    
    if show:
        plt.show(block=block)
    
    # Print comprehensive airfoil statistics
    print(f"\n{'='*60}")
    print(f"🛩️  AIRFOIL ANALYSIS REPORT")
    print(f"{'='*60}")
    print(f"📊 Geometric Properties:")
    print(f"   ├─ Max Thickness: {max_thickness:.4f} @ x/c = {max_thick_loc:.3f}")
    print(f"   ├─ Max Camber: {max_camber:.4f} @ x/c = {max_camber_loc:.3f}")
    print(f"   └─ Coordinate Points: {len(coordinates)}")
    print(f"📋 CST Configuration:")
    print(f"   ├─ Leading Edge Weight: {leading_edge_weight:.4f}")
    print(f"   ├─ Trailing Edge Thickness: {te_thickness:.6f}")
    print(f"   ├─ Upper Surface Parameters: {len(upper_weights)}")
    print(f"   └─ Lower Surface Parameters: {len(lower_weights)}")
    print(f"{'='*60}")
    
    return airfoil, coordinates

def plot_multiple_cst_airfoils(cst_params_list, labels=None, title="CST Airfoil Comparison", 
                              figsize=(14, 8), save_path=None, show_legend=True, block=False):
    """
    FIXED: Plot multiple airfoils for comparison with enhanced styling and proper error handling.
    
    Parameters:
    -----------
    cst_params_list : list
        List of CST parameter dictionaries
    labels : list, optional
        Labels for each airfoil
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    show_legend : bool
        Whether to show the legend
    """
    
    print(f"\n🎨 Starting to plot {len(cst_params_list)} airfoils...")
    
    plt.figure(figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    airfoils_data = []
    successful_plots = 0
    
    for i, cst_params in enumerate(cst_params_list):
        print(f"\n📊 Processing airfoil {i+1}/{len(cst_params_list)}...")
        
        try:
            # FIXED: Validate and convert parameters properly
            print(f"  🔍 Validating parameters...")
            print(f"    Available keys: {list(cst_params.keys())}")
            
            # Convert numpy scalars to regular Python types
            lower_weights = np.asarray(cst_params['lower_weights'], dtype=float)
            upper_weights = np.asarray(cst_params['upper_weights'], dtype=float)
            leading_edge_weight = float(cst_params['leading_edge_weight'])
            te_thickness = float(cst_params['TE_thickness'])
            
            print(f"    ✅ Parameters validated:")
            print(f"      - lower_weights: {lower_weights.shape} {type(lower_weights)}")
            print(f"      - upper_weights: {upper_weights.shape} {type(upper_weights)}")
            print(f"      - leading_edge_weight: {leading_edge_weight} {type(leading_edge_weight)}")
            print(f"      - te_thickness: {te_thickness} {type(te_thickness)}")
            
            # FIXED: Generate coordinates with proper error handling
            print(f"  🔧 Generating coordinates...")
            try:
                coordinates = get_kulfan_coordinates(
                    lower_weights=lower_weights,
                    upper_weights=upper_weights,
                    leading_edge_weight=leading_edge_weight,
                    TE_thickness=te_thickness,
                    N1=0.5,
                    N2=1.0,
                    n_points_per_side=200
                )
                print(f"    ✅ Generated {len(coordinates)} coordinate points")
                
            except Exception as coord_error:
                print(f"    ❌ Coordinate generation failed: {coord_error}")
                print(f"    🔄 Trying alternative approach...")
                
                # Try alternative parameter passing
                coordinates = get_kulfan_coordinates(
                    lower_weights, upper_weights, leading_edge_weight, te_thickness
                )
                print(f"    ✅ Alternative approach successful: {len(coordinates)} points")
            
            # Plot the airfoil
            label = labels[i] if labels and i < len(labels) else f'Airfoil {i+1}'
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            plt.plot(coordinates[:, 0], coordinates[:, 1], linewidth=2.5, 
                    label=label, color=color, linestyle=linestyle, alpha=0.9)
            plt.fill(coordinates[:, 0], coordinates[:, 1], alpha=0.15, color=color)
            
            airfoils_data.append((label, coordinates, color))
            successful_plots += 1
            print(f"  ✅ Successfully plotted: {label}")
            
        except Exception as e:
            print(f"  ❌ Error plotting airfoil {i}: {e}")
            print(f"    Parameter details:")
            for key, value in cst_params.items():
                print(f"      {key}: {type(value)} = {value}")
            continue
    
    print(f"\n📈 Successfully plotted {successful_plots}/{len(cst_params_list)} airfoils")
    
    if successful_plots == 0:
        print("❌ No airfoils were successfully plotted!")
        plt.close()
        return []
    
    # Enhanced plot styling
    plt.xlim(-0.05, 1.05)
    plt.xlabel('x/c', fontsize=14, fontweight='bold')
    plt.ylabel('y/c', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.axis('equal')
    
    # Reference lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=0.8)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.4, linewidth=0.8)
    plt.axvline(x=1, color='k', linestyle='--', alpha=0.4, linewidth=0.8)
    
    if show_legend and airfoils_data:
        plt.legend(loc='upper right', framealpha=0.9, fontsize=11, 
                  bbox_to_anchor=(1.0, 1.0))
    
    # Add comparison statistics
    if len(airfoils_data) > 1:
        stats_text = f"Comparison of {len(airfoils_data)} airfoils"
        plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes,
                verticalalignment='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Comparison plot saved to: {save_path}")
    
    plt.show(block=block)
    print(f"🎉 Plotting completed successfully!")
    
    return airfoils_data

def validate_cst_parameters(cst_params):
    """
    ADDED: Validate CST parameters before plotting
    
    Parameters:
    -----------
    cst_params : dict
        CST parameter dictionary to validate
        
    Returns:
    --------
    bool : True if valid, False otherwise
    str : Error message if invalid
    """
    
    required_keys = ['lower_weights', 'upper_weights', 'leading_edge_weight', 'TE_thickness']
    
    # Check required keys
    for key in required_keys:
        if key not in cst_params:
            return False, f"Missing required key: {key}"
    
    try:
        # Validate array parameters
        lower_weights = np.asarray(cst_params['lower_weights'], dtype=float)
        upper_weights = np.asarray(cst_params['upper_weights'], dtype=float)
        
        if lower_weights.size == 0 or upper_weights.size == 0:
            return False, "Weight arrays cannot be empty"
        
        # Validate scalar parameters
        leading_edge_weight = float(cst_params['leading_edge_weight'])
        te_thickness = float(cst_params['TE_thickness'])
        
        # Check for reasonable ranges
        if not np.isfinite(leading_edge_weight) or not np.isfinite(te_thickness):
            return False, "Non-finite values in scalar parameters"
        
        if not np.all(np.isfinite(lower_weights)) or not np.all(np.isfinite(upper_weights)):
            return False, "Non-finite values in weight arrays"
            
        return True, "Valid parameters"
        
    except Exception as e:
        return False, f"Parameter validation error: {e}"