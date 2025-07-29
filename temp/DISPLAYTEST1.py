import numpy as np
import matplotlib.pyplot as plt
import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

def plot_cst_airfoil(cst_params, n_points_per_side=200, title="CST Airfoil", show_params=True, 
                     save_path=None, figsize=(12, 8), show=True):
    """
    Plot an airfoil from CST (Class Shape Transformation) parameters using AeroSandbox's implementation.
    
    Parameters:
    -----------
    cst_params : dict
        Dictionary containing:
        - 'lower_weights': array of lower surface CST weights
        - 'upper_weights': array of upper surface CST weights  
        - 'leading_edge_weight': leading edge weight (scalar)
        - 'TE_thickness': trailing edge thickness (scalar)
    n_points_per_side : int
        Number of points to generate for each surface
    title : str
        Plot title
    show_params : bool
        Whether to display parameter information
    save_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size (width, height)
    show : bool
        Whether to call plt.show() after plotting
    
    Returns:
    --------
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
    
    # Use AeroSandbox's get_kulfan_coordinates function (the working method!)
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
        
    except Exception as e:
        # Fallback geometric calculations
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
        plt.show()
    
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
                              figsize=(14, 8), save_path=None, show_legend=True):
    """
    Plot multiple airfoils for comparison with enhanced styling.
    
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
    
    plt.figure(figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    airfoils_data = []
    
    for i, cst_params in enumerate(cst_params_list):
        try:
            # Generate coordinates using the working method
            coordinates = get_kulfan_coordinates(
                lower_weights=cst_params['lower_weights'],
                upper_weights=cst_params['upper_weights'],
                leading_edge_weight=float(cst_params['leading_edge_weight']),
                TE_thickness=float(cst_params['TE_thickness']),
                N1=0.5,
                N2=1.0,
                n_points_per_side=200
            )
            
            label = labels[i] if labels and i < len(labels) else f'Airfoil {i+1}'
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            plt.plot(coordinates[:, 0], coordinates[:, 1], linewidth=2.5, 
                    label=label, color=color, linestyle=linestyle, alpha=0.9)
            plt.fill(coordinates[:, 0], coordinates[:, 1], alpha=0.15, color=color)
            
            airfoils_data.append((label, coordinates, color))
            
        except Exception as e:
            print(f"❌ Error plotting airfoil {i}: {e}")
            continue
    
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
    
    plt.show()
    
    return airfoils_data

def plot_optimization_progress(param_history, fitness_history, title="Optimization Progress"):
    """
    Plot the evolution of airfoil shape during optimization.
    
    Parameters:
    -----------
    param_history : list
        List of CST parameter dictionaries from different optimization steps
    fitness_history : list
        List of fitness values corresponding to each parameter set
    title : str
        Plot title
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Airfoil evolution
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_history)))
    
    for i, (params, fitness) in enumerate(zip(param_history, fitness_history)):
        try:
            coordinates = get_kulfan_coordinates(
                lower_weights=params['lower_weights'],
                upper_weights=params['upper_weights'],
                leading_edge_weight=float(params['leading_edge_weight']),
                TE_thickness=float(params['TE_thickness']),
                N1=0.5, N2=1.0, n_points_per_side=100
            )
            
            alpha = 0.3 + 0.7 * (i / len(param_history))  # Fade from transparent to opaque
            ax1.plot(coordinates[:, 0], coordinates[:, 1], color=colors[i], 
                    alpha=alpha, linewidth=1.5, 
                    label=f'Step {i}' if i % max(1, len(param_history)//5) == 0 else '')
            
        except Exception as e:
            print(f"Error plotting step {i}: {e}")
    
    ax1.set_title('Airfoil Shape Evolution', fontweight='bold')
    ax1.set_xlabel('x/c')
    ax1.set_ylabel('y/c')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    
    # Plot 2: Fitness evolution
    ax2.plot(fitness_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax2.set_title('Fitness Evolution', fontweight='bold')
    ax2.set_xlabel('Optimization Step')
    ax2.set_ylabel('Fitness Value')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter evolution (first few parameters)
    param_matrix = np.array([np.concatenate([p['upper_weights'][:3], p['lower_weights'][:3]]) 
                            for p in param_history])
    
    for i in range(min(6, param_matrix.shape[1])):
        ax3.plot(param_matrix[:, i], label=f'Param {i}', linewidth=1.5)
    
    ax3.set_title('Parameter Evolution (First 6)', fontweight='bold')
    ax3.set_xlabel('Optimization Step')
    ax3.set_ylabel('Parameter Value')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Leading edge weight and TE thickness evolution
    le_weights = [p['leading_edge_weight'] for p in param_history]
    te_thicknesses = [p['TE_thickness'] for p in param_history]
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(le_weights, 'r-', linewidth=2, marker='s', label='LE Weight')
    line2 = ax4_twin.plot(te_thicknesses, 'g-', linewidth=2, marker='^', label='TE Thickness')
    
    ax4.set_title('Special Parameters Evolution', fontweight='bold')
    ax4.set_xlabel('Optimization Step')
    ax4.set_ylabel('Leading Edge Weight', color='r')
    ax4_twin.set_ylabel('TE Thickness', color='g')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Example usage and testing:
if __name__ == "__main__":
    # Test with your custom parameters
    cst_parameters = {
        'lower_weights': np.array([-0.16965146, -0.09364138, -0.06345896, -0.0067966, -0.0902447,
                                  0.02081845, -0.03575216, -0.00223623]),
        'upper_weights': np.array([0.18109497, 0.21268419, 0.28098503, 0.24864887, 0.2402814,
                                  0.27262843, 0.25776474, 0.27817638]),
        'leading_edge_weight': 0.10647339061374254,
        'TE_thickness': 0.002572011317150121
    }
    
    print("🚀 Testing enhanced CST airfoil plotting...")
    
    # Test single airfoil plot
    airfoil, coordinates = plot_cst_airfoil(
        cst_parameters, 
        title="Enhanced CST Airfoil Visualization",
        show_params=True,
        # save_path="enhanced_airfoil.png"  # Uncomment to save
    )
    
    # Test comparison plot with multiple variations
    variations = [cst_parameters]
    
    # Create a slight variation for comparison
    cst_variation = cst_parameters.copy()
    cst_variation['upper_weights'] = cst_variation['upper_weights'] * 0.8
    cst_variation['leading_edge_weight'] = cst_variation['leading_edge_weight'] * 1.2
    variations.append(cst_variation)

    print(variations)

    plot_multiple_cst_airfoils(
        variations, 
        labels=['Original', 'Modified'],
        title="CST Airfoil Comparison"
    )