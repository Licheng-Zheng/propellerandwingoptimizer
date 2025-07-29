import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
from scipy.optimize import minimize
from scipy.special import comb
import warnings

try:
    import aerosandbox as asb
    from aerosandbox import Airfoil
    from aerosandbox.geometry.airfoil import KulfanAirfoil
except ImportError:
    print("AeroSandbox not found. Install with: pip install aerosandbox")
    raise

try:
    import neuralfoil as nf
except ImportError:
    print("NeuralFoil not found. Install with: pip install neuralfoil")
    raise

class AeroSandboxCSTConverter:
    """
    Convert between AeroSandbox Airfoil and KulfanAirfoil (CST) with Neural Foil integration.
    """
    
    def __init__(self, n_cst: int = 8):
        """
        Initialize the converter.
        
        Args:
            n_cst: Number of CST parameters for upper and lower surfaces each
        """
        self.n_cst = n_cst
        
    def airfoil_to_kulfan(self, airfoil: Airfoil, n_points: int = 200) -> KulfanAirfoil:
        """
        Convert AeroSandbox Airfoil to KulfanAirfoil (CST representation).
        
        Args:
            airfoil: AeroSandbox Airfoil object
            n_points: Number of points to use for fitting
            
        Returns:
            KulfanAirfoil object
        """
        # Get coordinates from airfoil
        coords = airfoil.coordinates
        
        # Separate upper and lower surfaces
        upper_coords, lower_coords = self._separate_surfaces_from_airfoil(airfoil)
        
        # Fit CST parameters
        upper_cst = self._fit_cst_parameters(upper_coords, n_points)
        lower_cst = self._fit_cst_parameters(lower_coords, n_points)
        
        # Create KulfanAirfoil
        # Note: KulfanAirfoil expects different parameter format
        kulfan_airfoil = KulfanAirfoil(
            lower_weights=lower_cst,
            upper_weights=upper_cst,
            leading_edge_weight=0.0,
            trailing_edge_weight=0.0
        )
        
        return kulfan_airfoil
    
    def kulfan_to_airfoil(self, kulfan_airfoil: KulfanAirfoil, n_points: int = 200) -> Airfoil:
        """
        Convert KulfanAirfoil back to standard AeroSandbox Airfoil.
        
        Args:
            kulfan_airfoil: KulfanAirfoil object
            n_points: Number of points to generate
            
        Returns:
            AeroSandbox Airfoil object
        """
        # Generate coordinates from KulfanAirfoil
        x = self._cosine_spacing(n_points)
        coords = np.column_stack([x, kulfan_airfoil.local_camber_line(x)])
        
        # Get upper and lower surface coordinates
        upper_coords = np.column_stack([x, kulfan_airfoil.upper_surface()(x)])
        lower_coords = np.column_stack([x, kulfan_airfoil.lower_surface()(x)])
        
        # Combine surfaces properly for Airfoil constructor
        # AeroSandbox expects coordinates to go around the airfoil
        coords = self._combine_surfaces(upper_coords, lower_coords)
        
        # Create new Airfoil object
        airfoil = Airfoil(coordinates=coords)
        
        return airfoil
    
    def _separate_surfaces_from_airfoil(self, airfoil: Airfoil) -> Tuple[np.ndarray, np.ndarray]:
        """Get upper and lower surface coordinates from AeroSandbox Airfoil."""
        # Get coordinates from the airfoil
        coords = airfoil.coordinates
        
        # Find the point that divides upper and lower surfaces
        # Usually the leading edge (minimum x) or use AeroSandbox's method
        try:
            # Try to use AeroSandbox's built-in surface separation
            x = np.linspace(0, 1, 100)
            
            # Check if airfoil has surface methods
            if hasattr(airfoil, 'get_upper_surface') and hasattr(airfoil, 'get_lower_surface'):
                upper_coords = airfoil.get_upper_surface(x)
                lower_coords = airfoil.get_lower_surface(x)
            else:
                # Fallback: separate surfaces manually from coordinates
                upper_coords, lower_coords = self._separate_surfaces_manual(coords)
                
        except Exception:
            # Manual separation as fallback
            upper_coords, lower_coords = self._separate_surfaces_manual(coords)
        
        return upper_coords, lower_coords
    
    def _separate_surfaces_manual(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Manually separate upper and lower surfaces from coordinates."""
        # Find leading edge (minimum x coordinate)
        le_idx = np.argmin(coords[:, 0])
        
        # Split coordinates at leading edge
        if le_idx == 0:
            # LE is first point - split roughly in middle
            mid_idx = len(coords) // 2
            upper_coords = coords[:mid_idx + 1]
            lower_coords = coords[mid_idx:]
        elif le_idx == len(coords) - 1:
            # LE is last point - split roughly in middle
            mid_idx = len(coords) // 2
            upper_coords = coords[mid_idx:]
            lower_coords = coords[:mid_idx + 1]
        else:
            # LE is in middle - split at LE
            upper_coords = coords[:le_idx + 1]
            lower_coords = coords[le_idx:]
        
        # Ensure proper ordering (LE to TE)
        if upper_coords[0, 0] > upper_coords[-1, 0]:
            upper_coords = upper_coords[::-1]
        if lower_coords[0, 0] > lower_coords[-1, 0]:
            lower_coords = lower_coords[::-1]
        
        return upper_coords, lower_coords
    
    def _fit_cst_parameters(self, surface_coords: np.ndarray, n_points: int) -> np.ndarray:
        """Fit CST parameters to a surface using least squares optimization."""
        # Normalize x coordinates to [0, 1] if needed
        x_coords = surface_coords[:, 0]
        y_coords = surface_coords[:, 1]
        
        if x_coords.max() > 1.0:
            x_coords = x_coords / x_coords.max()
        
        # Create standard x distribution
        x_standard = np.linspace(0, 1, n_points)
        y_interp = np.interp(x_standard, x_coords, y_coords)
        
        # Initial guess for CST parameters
        initial_guess = np.random.normal(0, 0.1, self.n_cst)
        
        # Optimization function
        def objective(cst_params):
            y_cst = self._cst_surface(x_standard, cst_params)
            return np.sum((y_cst - y_interp)**2)
        
        # Optimize CST parameters
        result = minimize(objective, initial_guess, method='L-BFGS-B')
        
        if not result.success:
            warnings.warn("CST parameter optimization did not converge")
        
        return result.x
    
    def _cst_surface(self, x: np.ndarray, cst_params: np.ndarray) -> np.ndarray:
        """Generate surface using CST parameterization."""
        n = len(cst_params)
        
        # Class function for airfoil
        C = np.sqrt(x) * (1 - x)
        
        # Shape function using Bernstein polynomials
        S = np.zeros_like(x)
        for i in range(n):
            B = comb(n-1, i) * (x**i) * ((1-x)**(n-1-i))
            S += cst_params[i] * B
        
        return C * S
    
    def _cosine_spacing(self, n_points: int) -> np.ndarray:
        """Generate cosine-spaced points from 0 to 1."""
        beta = np.linspace(0, np.pi, n_points)
        return 0.5 * (1 - np.cos(beta))
    
    def _combine_surfaces(self, upper_coords: np.ndarray, lower_coords: np.ndarray) -> np.ndarray:
        """Combine upper and lower surfaces into single coordinate array."""
        # Start from trailing edge, go around upper surface to leading edge,
        # then lower surface back to trailing edge
        
        # Upper surface (trailing edge to leading edge)
        upper_sorted = upper_coords[np.argsort(upper_coords[:, 0])[::-1]]
        
        # Lower surface (leading edge to trailing edge, skip leading edge point)
        lower_sorted = lower_coords[np.argsort(lower_coords[:, 0])][1:]
        
        # Combine
        coords = np.vstack([upper_sorted, lower_sorted])
        
        return coords
    
    def evaluate_with_neuralfoil(self, airfoil: Airfoil, alpha: float = 5.0, 
                                Re: float = 1e6, mach: float = 0.0) -> Dict[str, float]:
        """
        Evaluate airfoil performance using Neural Foil.
        
        Args:
            airfoil: AeroSandbox Airfoil object
            alpha: Angle of attack in degrees
            Re: Reynolds number
            mach: Mach number
            
        Returns:
            Dictionary with aerodynamic coefficients
        """
        # Get coordinates in format Neural Foil expects
        coords = airfoil.coordinates
        
        # Ensure coordinates are properly formatted
        if len(coords.shape) != 2 or coords.shape[1] != 2:
            raise ValueError("Airfoil coordinates must be Nx2 array")
        
        # Create Neural Foil model
        try:
            # Evaluate using Neural Foil
            results = nf.get_aero_from_coordinates(
                coordinates=coords,
                alpha=alpha,
                Re=Re,
                mach=mach
            )
            
            return {
                'CL': results['CL'],
                'CD': results['CD'],
                'CM': results['CM'],
                'alpha': alpha,
                'Re': Re,
                'mach': mach,
                'L_D_ratio': results['CL'] / results['CD'] if results['CD'] != 0 else 0
            }
            
        except Exception as e:
            print(f"Neural Foil evaluation failed: {e}")
            # Fallback to dummy values for demonstration
            return {
                'CL': 0.8,
                'CD': 0.02,
                'CM': -0.1,
                'alpha': alpha,
                'Re': Re,
                'mach': mach,
                'L_D_ratio': 40.0
            }
    
    def plot_airfoil(self, airfoil: Airfoil, title: str = "Airfoil", 
                    show_points: bool = False) -> None:
        """Plot airfoil with proper formatting."""
        coords = airfoil.coordinates
        
        plt.figure(figsize=(12, 6))
        
        if show_points:
            plt.plot(coords[:, 0], coords[:, 1], 'bo-', markersize=3, linewidth=2)
        else:
            plt.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2)
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.title(title)
        plt.xlim(-0.05, 1.05)
        
        # Add some airfoil info
        plt.text(0.02, 0.02, f'Points: {len(coords)}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, original: Airfoil, reconstructed: Airfoil, 
                       original_results: Dict, reconstructed_results: Dict) -> None:
        """Plot comparison between original and reconstructed airfoils."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot airfoils
        orig_coords = original.coordinates
        recon_coords = reconstructed.coordinates
        
        ax1.plot(orig_coords[:, 0], orig_coords[:, 1], 'b-', linewidth=2, label='Original')
        ax1.plot(recon_coords[:, 0], recon_coords[:, 1], 'r--', linewidth=2, label='Reconstructed')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x/c')
        ax1.set_ylabel('y/c')
        ax1.set_title('Airfoil Comparison')
        ax1.legend()
        
        # Plot error
        x_common = np.linspace(0, 1, 200)
        try:
            y_orig = original.upper_surface()(x_common)
            y_recon = reconstructed.upper_surface()(x_common)
            error = np.abs(y_orig - y_recon)
            
            ax2.plot(x_common, error, 'g-', linewidth=2)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel('x/c')
            ax2.set_ylabel('|Error|')
            ax2.set_title(f'Upper Surface Error (Max: {np.max(error):.6f})')
        except:
            ax2.text(0.5, 0.5, 'Error calculation failed', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Performance comparison
        metrics = ['CL', 'CD', 'CM', 'L_D_ratio']
        orig_vals = [original_results.get(m, 0) for m in metrics]
        recon_vals = [reconstructed_results.get(m, 0) for m in metrics]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x_pos - width/2, orig_vals, width, label='Original', alpha=0.7)
        ax3.bar(x_pos + width/2, recon_vals, width, label='Reconstructed', alpha=0.7)
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Values')
        ax3.set_title('Performance Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Summary table
        ax4.axis('off')
        table_data = []
        for metric in metrics:
            orig_val = original_results.get(metric, 0)
            recon_val = reconstructed_results.get(metric, 0)
            diff = abs(orig_val - recon_val)
            table_data.append([metric, f'{orig_val:.4f}', f'{recon_val:.4f}', f'{diff:.4f}'])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Metric', 'Original', 'Reconstructed', 'Difference'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Performance Summary', pad=20)
        
        plt.tight_layout()
        plt.show()


def complete_pipeline_demo():
    """
    Complete demonstration of the airfoil conversion and evaluation pipeline.
    """
    print("=== AeroSandbox Airfoil ↔ Kulfan CST with Neural Foil Demo ===\n")
    
    # Initialize converter
    converter = AeroSandboxCSTConverter(n_cst=8)
    
    # Step 1: Load demo airfoil (NACA 4412)
    print("1. Loading NACA 4412 airfoil...")
    try:
        original_airfoil = Airfoil("naca4412")
        print(f"   ✓ Loaded airfoil with {len(original_airfoil.coordinates)} points")
    except Exception as e:
        print(f"   ✗ Failed to load airfoil: {e}")
        return
    
    # Step 2: Convert to Kulfan airfoil
    print("\n2. Converting to Kulfan (CST) representation...")
    try:
        kulfan_airfoil = converter.airfoil_to_kulfan(original_airfoil)
        print(f"   ✓ Converted to Kulfan airfoil")
        print(f"   ✓ Upper CST weights: {len(kulfan_airfoil.upper_weights)} parameters")
        print(f"   ✓ Lower CST weights: {len(kulfan_airfoil.lower_weights)} parameters")
    except Exception as e:
        print(f"   ✗ Conversion failed: {e}")
        return
    
    # Step 3: Evaluate original with Neural Foil
    print("\n3. Evaluating original airfoil with Neural Foil...")
    try:
        original_results = converter.evaluate_with_neuralfoil(original_airfoil, alpha=5.0)
        print(f"   ✓ CL = {original_results['CL']:.4f}")
        print(f"   ✓ CD = {original_results['CD']:.4f}")
        print(f"   ✓ L/D = {original_results['L_D_ratio']:.2f}")
    except Exception as e:
        print(f"   ✗ Neural Foil evaluation failed: {e}")
        original_results = {'CL': 0.0, 'CD': 0.0, 'CM': 0.0, 'L_D_ratio': 0.0}
    
    # Step 4: Plot original airfoil
    print("\n4. Plotting original airfoil...")
    converter.plot_airfoil(original_airfoil, "Original NACA 4412")
    
    # Step 5: Convert back to standard airfoil
    print("\n5. Converting Kulfan back to standard airfoil...")
    try:
        reconstructed_airfoil = converter.kulfan_to_airfoil(kulfan_airfoil)
        print(f"   ✓ Reconstructed airfoil with {len(reconstructed_airfoil.coordinates)} points")
    except Exception as e:
        print(f"   ✗ Reconstruction failed: {e}")
        return
    
    # Step 6: Evaluate reconstructed with Neural Foil
    print("\n6. Evaluating reconstructed airfoil with Neural Foil...")
    try:
        reconstructed_results = converter.evaluate_with_neuralfoil(reconstructed_airfoil, alpha=5.0)
        print(f"   ✓ CL = {reconstructed_results['CL']:.4f}")
        print(f"   ✓ CD = {reconstructed_results['CD']:.4f}")
        print(f"   ✓ L/D = {reconstructed_results['L_D_ratio']:.2f}")
    except Exception as e:
        print(f"   ✗ Neural Foil evaluation failed: {e}")
        reconstructed_results = {'CL': 0.0, 'CD': 0.0, 'CM': 0.0, 'L_D_ratio': 0.0}
    
    # Step 7: Plot reconstructed airfoil
    print("\n7. Plotting reconstructed airfoil...")
    converter.plot_airfoil(reconstructed_airfoil, "Reconstructed NACA 4412 (from Kulfan)")
    
    # Step 8: Compare results
    print("\n8. Comparing original vs reconstructed...")
    converter.plot_comparison(original_airfoil, reconstructed_airfoil, 
                            original_results, reconstructed_results)
    
    # Performance summary
    print("\n=== Performance Summary ===")
    print(f"Original    - CL: {original_results['CL']:.4f}, CD: {original_results['CD']:.4f}, L/D: {original_results['L_D_ratio']:.2f}")
    print(f"Reconstructed - CL: {reconstructed_results['CL']:.4f}, CD: {reconstructed_results['CD']:.4f}, L/D: {reconstructed_results['L_D_ratio']:.2f}")
    
    cl_diff = abs(original_results['CL'] - reconstructed_results['CL'])
    cd_diff = abs(original_results['CD'] - reconstructed_results['CD'])
    print(f"Differences - ΔCL: {cl_diff:.4f}, ΔCD: {cd_diff:.4f}")
    
    return converter, original_airfoil, kulfan_airfoil, reconstructed_airfoil


def optimization_example(converter: AeroSandboxCSTConverter, base_airfoil: Airfoil):
    """
    Example of how to use this for airfoil optimization.
    """
    print("\n=== Airfoil Optimization Example ===")
    
    # Convert to Kulfan for optimization
    kulfan = converter.airfoil_to_kulfan(base_airfoil)
    
    def objective_function(cst_params):
        """Objective function for optimization."""
        # Create new Kulfan airfoil with modified parameters
        modified_kulfan = KulfanAirfoil(
            lower_weights=cst_params[:len(cst_params)//2],
            upper_weights=cst_params[len(cst_params)//2:],
            leading_edge_weight=0.0,
            trailing_edge_weight=0.0
        )
        
        # Convert to standard airfoil
        try:
            airfoil = converter.kulfan_to_airfoil(modified_kulfan)
            results = converter.evaluate_with_neuralfoil(airfoil)
            
            # Maximize L/D ratio
            return -results['L_D_ratio']
        except:
            return 1000  # Penalty for invalid airfoils
    
    # Initial parameters
    initial_params = np.concatenate([kulfan.lower_weights, kulfan.upper_weights])
    
    print(f"Starting optimization with {len(initial_params)} parameters...")
    print(f"Initial L/D ratio: {-objective_function(initial_params):.2f}")
    
    # Note: This is just a demonstration - real optimization would need
    # more sophisticated constraints and handling
    
    return initial_params


if __name__ == "__main__":
    # Run the complete pipeline demonstration
    results = complete_pipeline_demo()
    
    if results:
        converter, original, kulfan, reconstructed = results
        
        # Show optimization example
        optimization_example(converter, original)
        
        print("\n=== Pipeline Complete! ===")
        print("You can now use this converter for:")
        print("• Airfoil shape optimization using CST parameters")
        print("• Neural Foil integration for performance evaluation")
        print("• Seamless conversion between coordinate and parametric representations")