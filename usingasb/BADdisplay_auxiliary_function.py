import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import aerosandbox as asb

def plot_cst_airfoil(cst_params, n_points=200, title="CST Airfoil", show_params=True, 
                     save_path=None, figsize=(12, 8)):
    """
    Plot an airfoil from CST (Class Shape Transformation) parameters.
    
    Parameters:
    -----------
    cst_params : dict
        Dictionary containing:
        - 'lower_weights': array of lower surface CST weights
        - 'upper_weights': array of upper surface CST weights  
        - 'leading_edge_weight': leading edge weight (scalar)
        - 'TE_thickness': trailing edge thickness (scalar)
    n_points : int
        Number of points to generate for each surface
    title : str
        Plot title
    show_params : bool
        Whether to display parameter information
    save_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size (width, height)
    
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
    
    # CST parameters (fixed as specified)
    N1, N2 = 0.5, 1.0
    
    # Generate x coordinates
    x = np.linspace(0, 1, n_points)
    
    def cst_shape_function(x, weights, N1=0.5, N2=1.0):
        """
        Compute CST shape function.
        
        The CST method uses Bernstein polynomials to define the airfoil shape:
        S(x) = sum(weights[i] * B[i,n](x)) where B[i,n] is the Bernstein polynomial
        """
        x = np.atleast_1d(x)
        n = len(weights)
        shape = np.zeros_like(x)
        
        for i, weight in enumerate(weights):
            # Bernstein polynomial of degree n-1
            bernstein = comb(n-1, i, exact=True) * (x**i) * ((1-x)**(n-1-i))
            shape += weight * bernstein
            
        return shape
    
    def cst_class_function(x, N1=0.5, N2=1.0):
        """
        CST class function that enforces boundary conditions.
        For airfoils: C(x) = x^N1 * (1-x)^N2
        """
        return (x**N1) * ((1-x)**N2)
    
    # Calculate class function
    class_func = cst_class_function(x, N1, N2)
    
    # Calculate shape functions
    upper_shape = cst_shape_function(x, upper_weights, N1, N2)
    lower_shape = cst_shape_function(x, lower_weights, N1, N2)
    
    # Calculate surface coordinates
    # Upper surface: y = class_function * shape_function + leading_edge_contribution
    y_upper = class_func * upper_shape
    
    # Lower surface: y = -class_function * shape_function (negative for lower)
    y_lower = -class_func * lower_shape
    
    # Add leading edge contribution (affects both surfaces)
    le_contribution = leading_edge_weight * np.sqrt(x)
    y_upper += le_contribution
    y_lower -= le_contribution  # Subtract for lower surface
    
    # Apply trailing edge thickness
    if te_thickness > 0:
        te_offset = te_thickness * 0.5
        y_upper[-1] = te_offset
        y_lower[-1] = -te_offset
    
    # Combine surfaces into single coordinate array
    # Convention: Start from trailing edge upper, go to leading edge, then to trailing edge lower
    x_coords = np.concatenate([x[::-1], x[1:]])  # Upper surface reversed + lower surface
    y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])
    
    coordinates = np.column_stack([x_coords, y_coords])
    
    # Create AeroSandbox airfoil object
    airfoil = asb.Airfoil(coordinates=coordinates)
    
    # Create the plot
    fig, axes = plt.subplots(1, 2 if show_params else 1, figsize=figsize)
    
    if show_params:
        ax_airfoil = axes[0]
        ax_params = axes[1]
    else:
        ax_airfoil = axes if hasattr(axes, 'plot') else axes
    
    # Plot airfoil
    ax_airfoil.plot(coordinates[:, 0], coordinates[:, 1], 'b-', linewidth=2.5, label='Airfoil')
    ax_airfoil.fill(coordinates[:, 0], coordinates[:, 1], alpha=0.3, color='lightblue')
    
    # Add some reference lines and points
    ax_airfoil.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax_airfoil.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax_airfoil.axvline(x=1, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Mark leading and trailing edges
    ax_airfoil.plot(0, 0, 'ro', markersize=8, label='Leading Edge')
    ax_airfoil.plot(1, (y_upper[-1] + y_lower[-1])/2, 'go', markersize=8, label='Trailing Edge')
    
    ax_airfoil.set_xlim(-0.05, 1.05)
    ax_airfoil.set_xlabel('x/c', fontsize=12)
    ax_airfoil.set_ylabel('y/c', fontsize=12)
    ax_airfoil.set_title(title, fontsize=14, fontweight='bold')
    ax_airfoil.grid(True, alpha=0.3)
    ax_airfoil.set_aspect('equal')
    ax_airfoil.legend()
    
    # Add geometric information
    max_thickness = np.max(y_upper - y_lower)
    max_thick_loc = x[np.argmax(y_upper - y_lower)]
    
    info_text = f'Max thickness: {max_thickness:.4f} at x/c = {max_thick_loc:.3f}'
    ax_airfoil.text(0.02, 0.98, info_text, transform=ax_airfoil.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot parameters if requested
    if show_params:
        # Upper weights
        ax_params.bar(range(len(upper_weights)), upper_weights, 
                     alpha=0.7, color='red', label='Upper Weights')
        
        # Lower weights  
        ax_params.bar(range(len(lower_weights)), lower_weights, 
                     alpha=0.7, color='blue', label='Lower Weights')
        
        ax_params.set_xlabel('CST Parameter Index', fontsize=12)
        ax_params.set_ylabel('Weight Value', fontsize=12)
        ax_params.set_title('CST Weight Parameters', fontsize=14, fontweight='bold')
        ax_params.grid(True, alpha=0.3)
        ax_params.legend()
        
        # Add parameter info text
        param_text = f'Leading Edge Weight: {leading_edge_weight:.4f}\n'
        param_text += f'TE Thickness: {te_thickness:.6f}\n'
        param_text += f'N1: {N1}, N2: {N2}\n'
        param_text += f'Upper weights: {len(upper_weights)}\n'
        param_text += f'Lower weights: {len(lower_weights)}'
        
        ax_params.text(0.02, 0.98, param_text, transform=ax_params.transAxes,
                      verticalalignment='top', fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print some airfoil statistics
    print(f"\nAirfoil Statistics:")
    print(f"Max thickness: {max_thickness:.4f} at x/c = {max_thick_loc:.3f}")
    print(f"Leading edge weight: {leading_edge_weight:.4f}")
    print(f"Trailing edge thickness: {te_thickness:.6f}")
    print(f"Number of coordinate points: {len(coordinates)}")
    
    return airfoil, coordinates

def plot_multiple_cst_airfoils(cst_params_list, labels=None, title="CST Airfoil Comparison", 
                              figsize=(12, 8), save_path=None):
    """
    Plot multiple airfoils for comparison.
    
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
    """
    
    plt.figure(figsize=figsize)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, cst_params in enumerate(cst_params_list):
        airfoil, coords = plot_cst_airfoil(cst_params, show_params=False, title="")
        
        label = labels[i] if labels and i < len(labels) else f'Airfoil {i+1}'
        color = colors[i % len(colors)]
        
        plt.plot(coords[:, 0], coords[:, 1], linewidth=2, label=label, color=color)
        plt.fill(coords[:, 0], coords[:, 1], alpha=0.2, color=color)
    
    plt.xlim(-0.05, 1.05)
    plt.xlabel('x/c', fontsize=12)
    plt.ylabel('y/c', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

# Example usage:
# Your example parameters
cst_parameters = {
    'lower_weights': np.array([-0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02]),
    'upper_weights': np.array([0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.0, -0.02]),
    'leading_edge_weight': 0.1,  # Smaller LE radius (symmetric)
    'TE_thickness': 0.001  # Sharp TE
}

# Plot the airfoil
airfoil, coordinates = plot_cst_airfoil(
    cst_parameters, 
    title="Example CST Airfoil",
    show_params=True,
    # save_path="example_airfoil.png"  # Uncomment to save
)

print(f"\nAirfoil object created with {len(coordinates)} points")
print(f"Coordinate range: x=[{coordinates[:, 0].min():.3f}, {coordinates[:, 0].max():.3f}], "
    f"y=[{coordinates[:, 1].min():.3f}, {coordinates[:, 1].max():.3f}]")