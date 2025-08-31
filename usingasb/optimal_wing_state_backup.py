import json
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, Any, Optional
import neuralfoil as nf
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
from convertion_auxiliary_functions import array_to_kulfan_dict, kulfan_dict_to_array
from objective import internal_run_constraint_suite

class OptimalWingState:
    """
    Comprehensive state capture for optimal wing solutions
    Saves EVERYTHING that might be needed for future testing
    """
    
    def __init__(self):
        self.state_data = {}
        self.metadata = {}
        
    def capture_complete_state(self, 
                             es,  # CMA-ES evolutionary strategy
                             optimization_conditions: Dict,
                             optimization_config: Dict,
                             starting_guess: np.ndarray,
                             optimization_log: Dict):
        """
        Capture absolutely everything from the optimization process
        """
        
        print("🔄 Capturing complete optimal wing state...")
        
        # === CORE OPTIMIZATION RESULTS ===
        best_cst_array = es.result.xbest
        best_cst_dict = array_to_kulfan_dict(best_cst_array)
        
        self.state_data['optimization_results'] = {
            'best_cst_parameters': {
                'array_format': best_cst_array.tolist(),
                'dict_format': {
                    'lower_weights': best_cst_dict['lower_weights'].tolist(),
                    'upper_weights': best_cst_dict['upper_weights'].tolist(),
                    'leading_edge_weight': float(best_cst_dict['leading_edge_weight']),
                    'TE_thickness': float(best_cst_dict['TE_thickness'])
                }
            },
            'best_fitness': float(es.result.fbest),
            'total_evaluations': int(es.countevals),
            'final_sigma': float(es.sigma),
            'convergence_achieved': bool(es.stop()),
            'cma_es_internal_state': {
                'mean': es.mean.tolist() if hasattr(es, 'mean') else None,
                'condition_number': float(es.D.max() / es.D.min()) if hasattr(es, 'D') else None
            }
        }
        
        # === COMPLETE AERODYNAMIC ANALYSIS ===
        print("  📊 Computing complete aerodynamic analysis...")
        try:
            full_aero_results = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=best_cst_dict,
                alpha=optimization_conditions['alpha'],
                Re=optimization_conditions['Re'],
                model_size=optimization_conditions['model_size']
            )
            
            # Save ALL aerodynamic results (not just the ones used in optimization)
            # Convert numpy arrays and other non-serializable types to JSON-compatible formats
            self.state_data['aerodynamic_results'] = {}
            for key, value in full_aero_results.items():
                if isinstance(value, np.ndarray):
                    self.state_data['aerodynamic_results'][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    self.state_data['aerodynamic_results'][key] = float(value)
                elif np.isscalar(value):
                    self.state_data['aerodynamic_results'][key] = float(value)
                else:
                    # For any other type, try to convert to list or keep as is
                    try:
                        self.state_data['aerodynamic_results'][key] = value.tolist()
                    except:
                        self.state_data['aerodynamic_results'][key] = str(value)
            
        except Exception as e:
            print(f"  ⚠️  Warning: Aerodynamic analysis failed: {e}")
            self.state_data['aerodynamic_results'] = {'error': str(e)}
        
        # === AIRFOIL GEOMETRY ===
        print("  📐 Generating airfoil coordinates...")
        try:
            coordinates = get_kulfan_coordinates(
                lower_weights=best_cst_dict['lower_weights'],
                upper_weights=best_cst_dict['upper_weights'],
                leading_edge_weight=best_cst_dict['leading_edge_weight'],
                TE_thickness=best_cst_dict['TE_thickness'],
                n_points_per_side=200  # High resolution for future analysis
            )
            
            self.state_data['geometry'] = {
                'coordinates': coordinates.tolist(),
                'n_points': len(coordinates),
                'coordinate_bounds': {
                    'x_min': float(np.min(coordinates[:, 0])),
                    'x_max': float(np.max(coordinates[:, 0])),
                    'y_min': float(np.min(coordinates[:, 1])),
                    'y_max': float(np.max(coordinates[:, 1]))
                }
            }
            
            # Split into upper/lower surfaces for detailed analysis
            n_mid = len(coordinates) // 2
            upper_surface = coordinates[:n_mid]
            lower_surface = coordinates[n_mid:]
            
            self.state_data['geometry']['surfaces'] = {
                'upper_surface': upper_surface.tolist(),
                'lower_surface': lower_surface.tolist(),
                'leading_edge_point': coordinates[np.argmin(coordinates[:, 0])].tolist(),
                'trailing_edge_points': {
                    'upper': coordinates[0].tolist(),
                    'lower': coordinates[-1].tolist()
                }
            }
            
        except Exception as e:
            print(f"  ⚠️  Warning: Could not generate coordinates: {e}")
            self.state_data['geometry'] = {'error': str(e)}
        
        # === COMPLETE CONSTRAINT ANALYSIS ===
        print("  🔒 Evaluating all constraints...")
        try:
            constraint_penalty, hard_violation = internal_run_constraint_suite(
                cst_parameters=best_cst_dict,
                aero_results=self.state_data['aerodynamic_results'],
                epoch=999  # Use high epoch to ensure all constraints are active
            )
            
            self.state_data['constraint_results'] = {
                'total_penalty': float(constraint_penalty),
                'hard_constraint_violated': bool(hard_violation),
                'constraint_evaluation_successful': True
            }
            
        except Exception as e:
            print(f"  ⚠️  Warning: Constraint evaluation failed: {e}")
            self.state_data['constraint_results'] = {
                'error': str(e),
                'constraint_evaluation_successful': False
            }
        
        # === OPTIMIZATION CONFIGURATION ===
        self.state_data['optimization_configuration'] = {
            'flight_conditions': optimization_conditions.copy(),
            'cma_es_settings': optimization_config.copy(),
            'starting_guess': {
                'array_format': starting_guess.tolist(),
                'dict_format': {
                    'lower_weights': array_to_kulfan_dict(starting_guess)['lower_weights'].tolist(),
                    'upper_weights': array_to_kulfan_dict(starting_guess)['upper_weights'].tolist(),
                    'leading_edge_weight': float(array_to_kulfan_dict(starting_guess)['leading_edge_weight']),
                    'TE_thickness': float(array_to_kulfan_dict(starting_guess)['TE_thickness'])
                }
            }
        }
        
        # === COMPLETE OPTIMIZATION HISTORY ===
        # Convert numpy arrays to lists for JSON serialization
        serializable_log = {}
        for model_name, data in optimization_log.items():
            serializable_log[model_name] = {}
            for key, value in data.items():
                if key == 'best_parameters_per_epoch':
                    serializable_log[model_name][key] = [param.tolist() if hasattr(param, 'tolist') else param for param in value]
                else:
                    serializable_log[model_name][key] = value
        
        self.state_data['optimization_history'] = serializable_log
        
        # === METADATA ===
        self.metadata = {
            'capture_timestamp': datetime.now().isoformat(),
            'state_version': '1.0',
            'total_data_points': sum(len(str(v)) for v in self.state_data.values()),
            'capture_successful': True
        }
        
        print("  ✅ Complete state capture finished!")
        return True
    
    def save_to_files(self, base_path: str, model_name: str):
        """
        Save state to multiple formats for maximum flexibility
        """
        
        # JSON format (human readable, easy to inspect)
        json_path = f"{base_path}/{model_name}_complete_state.json"
        with open(json_path, 'w') as f:
            json.dump({
                'state_data': self.state_data,
                'metadata': self.metadata
            }, f, indent=2)
        
        # Pickle format (preserves exact numpy types, faster loading)
        pickle_path = f"{base_path}/{model_name}_complete_state.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'state_data': self.state_data,
                'metadata': self.metadata
            }, f)
        
        # Separate CST parameters file (for quick access)
        cst_path = f"{base_path}/{model_name}_cst_only.json"
        with open(cst_path, 'w') as f:
            json.dump({
                'cst_parameters': self.state_data['optimization_results']['best_cst_parameters'],
                'fitness': self.state_data['optimization_results']['best_fitness'],
                'flight_conditions': self.state_data['optimization_configuration']['flight_conditions']
            }, f, indent=2)
        
        print(f"💾 Complete state saved to:")
        print(f"   📄 JSON: {json_path}")
        print(f"   🥒 Pickle: {pickle_path}")
        print(f"   🎯 CST Only: {cst_path}")
        
        return json_path, pickle_path, cst_path
    
    def load_from_file(self, filepath: str):
        """Load state from JSON or pickle file"""
        
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        self.state_data = data['state_data']
        self.metadata = data['metadata']
        
        return True
    
    def get_summary(self):
        """Get a human-readable summary of the saved state"""
        
        if not self.state_data:
            return "No state data loaded"
        
        # Handle potential missing data gracefully
        try:
            aero = self.state_data['aerodynamic_results']
            geom = self.state_data['geometry']
            config = self.state_data['optimization_configuration']['flight_conditions']
            constraints = self.state_data['constraint_results']
            
            # Extract values safely to avoid complex f-string expressions
            cl_val = aero.get('CL', 'N/A')
            cl_str = f"{cl_val:.6f}" if isinstance(cl_val, (int, float)) else 'N/A'
            
            cd_val = aero.get('CD', 'N/A')
            cd_str = f"{cd_val:.6f}" if isinstance(cd_val, (int, float)) else 'N/A'
            
            cm_val = aero.get('CM', 'N/A')
            cm_str = f"{cm_val:.6f}" if isinstance(cm_val, (int, float)) else 'N/A'
            
            conf_val = aero.get('analysis_confidence', 'N/A')
            conf_str = f"{conf_val:.4f}" if isinstance(conf_val, (int, float)) else 'N/A'
            
            re_val = config.get('Re', 'N/A')
            re_str = f"{re_val:,.0f}" if isinstance(re_val, (int, float)) else 'N/A'
            
            # Geometry bounds
            bounds = geom.get('coordinate_bounds', {})
            x_min = bounds.get('x_min', 'N/A')
            x_max = bounds.get('x_max', 'N/A')
            y_min = bounds.get('y_min', 'N/A')
            y_max = bounds.get('y_max', 'N/A')
            
            x_min_str = f"{x_min:.3f}" if isinstance(x_min, (int, float)) else 'N/A'
            x_max_str = f"{x_max:.3f}" if isinstance(x_max, (int, float)) else 'N/A'
            y_min_str = f"{y_min:.3f}" if isinstance(y_min, (int, float)) else 'N/A'
            y_max_str = f"{y_max:.3f}" if isinstance(y_max, (int, float)) else 'N/A'
            
            penalty_val = constraints.get('total_penalty', 'N/A')
            penalty_str = f"{penalty_val:.6f}" if isinstance(penalty_val, (int, float)) else 'N/A'
            
            summary = f"""
🚀 OPTIMAL WING STATE SUMMARY
{'='*50}
📅 Captured: {self.metadata.get('capture_timestamp', 'Unknown')}
🎯 Best Fitness: {self.state_data['optimization_results']['best_fitness']:.8f}
🔄 Total Evaluations: {self.state_data['optimization_results']['total_evaluations']:,}

✈️  AERODYNAMIC PERFORMANCE:
   CL: {cl_str}
   CD: {cd_str}
   CM: {cm_str}
   Confidence: {conf_str}

🔧 FLIGHT CONDITIONS:
   Alpha: {config.get('alpha', 'N/A')}°
   Reynolds: {re_str}
   Model: {config.get('model_size', 'N/A')}

📐 GEOMETRY:
   Coordinate Points: {geom.get('n_points', 'N/A')}
   X Range: [{x_min_str}, {x_max_str}]
   Y Range: [{y_min_str}, {y_max_str}]

🔒 CONSTRAINTS:
   Hard Violation: {'❌ YES' if constraints.get('hard_constraint_violated') else '✅ NO'}
   Penalty: {penalty_str}
"""
        except Exception as e:
            summary = f"Error generating summary: {e}\nRaw metadata: {self.metadata}"
        
        return summary


# === CONVENIENCE FUNCTIONS FOR EASY INTEGRATION ===

def capture_and_save_optimal_state(es, optimization_conditions, optimization_config, 
                                 starting_guess, optimization_log, results_dir, model_name):
    """
    Convenience function to capture and save optimal state in one call
    
    Args:
        es: CMA-ES evolutionary strategy object
        optimization_conditions: Dict with alpha, Re, model_size, wanted_lists, importance_list
        optimization_config: Dict with CMA-ES settings
        starting_guess: Initial CST parameters array
        optimization_log: Complete optimization history
        results_dir: Directory to save files
        model_name: Name for the saved files
    
    Returns:
        tuple: (json_path, pickle_path, cst_path) - paths to saved files
    """
    
    print("\n" + "="*60)
    print("�� CAPTURING COMPLETE OPTIMAL WING STATE")
    print("="*60)
    
    # Create state capture object
    optimal_state = OptimalWingState()
    
    # Capture complete state
    success = optimal_state.capture_complete_state(
        es=es,
        optimization_conditions=optimization_conditions,
        optimization_config=optimization_config,
        starting_guess=starting_guess,
        optimization_log=optimization_log
    )
    
    if success:
        # Save in multiple formats
        json_path, pickle_path, cst_path = optimal_state.save_to_files(
            base_path=results_dir,
            model_name=model_name
        )
        
        # Print summary
        print(optimal_state.get_summary())
        
        print(f"\n🎉 COMPLETE STATE CAPTURE SUCCESSFUL!")
        print(f"   You can now quickly test modifications using these files.")
        print(f"   Main state file: {json_path}")
        
        return json_path, pickle_path, cst_path
        
    else:
        print("❌ State capture failed!")
        return None, None, None


def quick_load_and_summarize(state_file_path):
    """
    Convenience function to quickly load and display a saved state
    
    Args:
        state_file_path: Path to the saved state file (.json or .pkl)
    
    Returns:
        OptimalWingState: Loaded state object
    """
    
    optimal_state = OptimalWingState()
    optimal_state.load_from_file(state_file_path)
    
    print("✅ Loaded optimal wing state from:", state_file_path)
    print(optimal_state.get_summary())
    
    return optimal_state