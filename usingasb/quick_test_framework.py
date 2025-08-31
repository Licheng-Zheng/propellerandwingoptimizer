from optimal_wing_state import OptimalWingState
import argparse
import json
import numpy as np

class QuickTester:
    """Framework for quickly testing modifications on optimal wings"""
    
    def __init__(self, state_file: str):
        self.optimal_state = OptimalWingState()
        self.optimal_state.load_from_file(state_file)
        print("✅ Loaded optimal wing state")
        print(self.optimal_state.get_summary())
    
    def test_new_flight_conditions(self, new_alpha=None, new_Re=None):
        """Test the optimal wing at different flight conditions"""
        print(f"\n🧪 Testing new flight conditions...")
        
        # Use original conditions as defaults
        original_conditions = self.optimal_state.state_data['optimization_configuration']['flight_conditions']
        test_alpha = new_alpha if new_alpha is not None else original_conditions['alpha']
        test_Re = new_Re if new_Re is not None else original_conditions['Re']
        
        print(f"   Original: α={original_conditions['alpha']}°, Re={original_conditions['Re']:,.0f}")
        print(f"   Testing:  α={test_alpha}°, Re={test_Re:,.0f}")
        
        # Get CST parameters
        cst_dict = self.optimal_state.state_data['optimization_results']['best_cst_parameters']['dict_format']
        
        # Convert lists back to numpy arrays for NeuralFoil
        cst_dict_for_nf = {
            'lower_weights': np.array(cst_dict['lower_weights']),
            'upper_weights': np.array(cst_dict['upper_weights']),
            'leading_edge_weight': cst_dict['leading_edge_weight'],
            'TE_thickness': cst_dict['TE_thickness']
        }
        
        # Recompute aerodynamics at new conditions
        try:
            import neuralfoil as nf
            new_aero = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=cst_dict_for_nf,
                alpha=test_alpha,
                Re=test_Re,
                model_size=original_conditions['model_size']
            )
            
            # Compare results
            original_aero = self.optimal_state.state_data['aerodynamic_results']
            
            print(f"\n📊 Performance Comparison:")
            print(f"   CL: {original_aero['CL']:.6f} → {new_aero['CL']:.6f} (Δ{new_aero['CL']-original_aero['CL']:+.6f})")
            print(f"   CD: {original_aero['CD']:.6f} → {new_aero['CD']:.6f} (Δ{new_aero['CD']-original_aero['CD']:+.6f})")
            print(f"   CM: {original_aero['CM']:.6f} → {new_aero['CM']:.6f} (Δ{new_aero['CM']-original_aero['CM']:+.6f})")
            print(f"   Confidence: {original_aero['analysis_confidence']:.4f} → {new_aero['analysis_confidence']:.4f}")
            
            return new_aero
            
        except Exception as e:
            print(f"   ❌ Failed to compute new aerodynamics: {e}")
            return None
    
    def test_new_scoring_weights(self, new_importance_list):
        """Test how the optimal wing scores with different importance weights"""
        print(f"\n🎯 Testing new scoring weights...")
        
        try:
            from objective import scoring_model_1
            
            original_conditions = self.optimal_state.state_data['optimization_configuration']['flight_conditions']
            aero_results = self.optimal_state.state_data['aerodynamic_results']
            
            # Original score
            original_score = scoring_model_1(
                aero_results,
                original_conditions['wanted_lists'],
                original_conditions['importance_list']
            )
            
            # New score
            new_score = scoring_model_1(
                aero_results,
                original_conditions['wanted_lists'],
                new_importance_list
            )
            
            print(f"   Original weights: {original_conditions['importance_list']}")
            print(f"   New weights:      {new_importance_list}")
            print(f"   Original score:   {original_score:.6f}")
            print(f"   New score:        {new_score:.6f}")
            print(f"   Score change:     {new_score - original_score:+.6f}")
            
            return new_score
            
        except Exception as e:
            print(f"   ❌ Failed to compute new score: {e}")
            return None
    
    def test_constraints(self):
        """Test constraint evaluation on the optimal wing"""
        print(f"\n🔒 Testing constraint evaluation...")
        
        try:
            from objective import internal_run_constraint_suite
            
            # Get CST parameters
            cst_dict = self.optimal_state.state_data['optimization_results']['best_cst_parameters']['dict_format']
            
            # Convert lists back to numpy arrays
            cst_dict_for_constraints = {
                'lower_weights': np.array(cst_dict['lower_weights']),
                'upper_weights': np.array(cst_dict['upper_weights']),
                'leading_edge_weight': cst_dict['leading_edge_weight'],
                'TE_thickness': cst_dict['TE_thickness']
            }
            
            aero_results = self.optimal_state.state_data['aerodynamic_results']
            
            # Evaluate constraints
            penalty, hard_violation = internal_run_constraint_suite(
                cst_parameters=cst_dict_for_constraints,
                aero_results=aero_results,
                epoch=999
            )
            
            print(f"   Current constraint status:")
            print(f"   Penalty: {penalty:.6f}")
            print(f"   Hard violation: {'❌ YES' if hard_violation else '✅ NO'}")
            
            # Compare with saved results
            saved_constraints = self.optimal_state.state_data['constraint_results']
            print(f"   Saved penalty: {saved_constraints.get('total_penalty', 'N/A')}")
            print(f"   Saved violation: {'❌ YES' if saved_constraints.get('hard_constraint_violated') else '✅ NO'}")
            
            return penalty, hard_violation
            
        except Exception as e:
            print(f"   ❌ Failed to evaluate constraints: {e}")
            return None, None
    
    def visualize_airfoil(self):
        """Visualize the optimal airfoil"""
        print(f"\n📐 Visualizing optimal airfoil...")
        
        try:
            from display_auxiliary_functions import plot_cst_airfoil
            
            cst_dict = self.optimal_state.state_data['optimization_results']['best_cst_parameters']['dict_format']
            
            # Convert lists back to numpy arrays
            cst_dict_for_plot = {
                'lower_weights': np.array(cst_dict['lower_weights']),
                'upper_weights': np.array(cst_dict['upper_weights']),
                'leading_edge_weight': cst_dict['leading_edge_weight'],
                'TE_thickness': cst_dict['TE_thickness']
            }
            
            plot_cst_airfoil(
                cst_params=cst_dict_for_plot,
                title="Optimal Wing from Saved State",
                show_params=True,
                show=True
            )
            
            print("   ✅ Airfoil visualization displayed")
            
        except Exception as e:
            print(f"   ❌ Visualization failed: {e}")
    
    def get_cst_parameters(self):
        """Get CST parameters in usable format"""
        cst_dict = self.optimal_state.state_data['optimization_results']['best_cst_parameters']['dict_format']
        
        return {
            'lower_weights': np.array(cst_dict['lower_weights']),
            'upper_weights': np.array(cst_dict['upper_weights']),
            'leading_edge_weight': cst_dict['leading_edge_weight'],
            'TE_thickness': cst_dict['TE_thickness']
        }
    
    def get_flight_conditions(self):
        """Get original flight conditions"""
        return self.optimal_state.state_data['optimization_configuration']['flight_conditions']


def main():
    parser = argparse.ArgumentParser(description='Quick test optimal wing modifications')
    parser.add_argument('--state', required=True, help='Path to optimal wing state file')
    parser.add_argument('--test-alpha', type=float, help='Test at different angle of attack')
    parser.add_argument('--test-re', type=float, help='Test at different Reynolds number')
    parser.add_argument('--visualize', action='store_true', help='Show airfoil visualization')
    parser.add_argument('--test-constraints', action='store_true', help='Test constraint evaluation')
    
    args = parser.parse_args()
    
    # Create tester
    tester = QuickTester(args.state)
    
    # Run requested tests
    if args.test_alpha is not None or args.test_re is not None:
        tester.test_new_flight_conditions(args.test_alpha, args.test_re)
    
    if args.test_constraints:
        tester.test_constraints()
    
    if args.visualize:
        tester.visualize_airfoil()
    
    # Example scoring test
    print(f"\n🧪 Example: Testing different scoring weights...")
    tester.test_new_scoring_weights([0.5, 0.3, -0.15, -0.05])


if __name__ == "__main__":
    main()