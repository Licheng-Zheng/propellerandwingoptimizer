# Constraint Development Guide
Honestly, this thing was just to see if my in ide ai would work, and it cooked this thing up. Not sure if its any good, just used it to make sure it would run, but here it is! Its mostly correct, but I have my own version that I refer to so I'm not gonna bother fixing this thing up. 

This document summarizes how constraints are implemented and integrated into the optimization loop. Refer to `usingasb/objective.py`, `usingasb/constraints/hard_constraints_1.py`, and `usingasb/constraints/soft_constraints_1.py` for working examples.

## 1. Constraint anatomy

### Hard constraints
- Lives in `usingasb/constraints/hard_constraints_*.py`.
- Must accept `cst_parameters` (and optionally other metadata such as `N`, tolerances, etc.).
- Return a `bool` (or truthy value) where `True` signals a violation.
- Avoid expensive recomputation when possible (the builder already calls NeuralFoil once per candidate).
- Log informative messages using `logging.debug`/`logging.error` if you reject a shape to ease debugging.

Example signature:
```python
def my_hard_check(cst_parameters, tol=1e-3) -> bool:
    ...
    return overlap_detected
```

### Soft constraints
- Lives in `usingasb/constraints/soft_constraints_*.py`.
- Should return a non-negative penalty (`float`). Return `0.0` when satisfied.
- Large penalties (e.g., `1000.0`) can be returned on evaluation failure to discourage invalid candidates.
- Penalties are aggregated with `abs()`, so negative returns are acceptable as long as they encode "violation magnitude".

Example signature:
```python
def my_soft_check(cst_parameters, threshold=0.02) -> float:
    ...
    return max(0.0, threshold - thickness)
```

## 2. Implementing a new constraint module
1. Create a new helper function in the appropriate hard/soft module (or create a new module and import it in `objective.py`).
2. Use existing utilities such as `get_kulfan_coordinates` or cached aerodynamic results when available.
3. Document the constraint using inline comments and populate `metadata` (see builder usage below).

## 3. Registering constraints in `objective.py`
`ConstraintFactory` maps constraint names to constructor helpers. To add your constraint:
- Import the function in `objective.py`.
- Add a helper in `ConstraintFactory` that wraps your implementation, setting `metadata` for debugging.
- Update `create_constraint_suite_simple` or `create_constraint_suite_advanced` to include calls like `.add_hard_constraint('my_constraint', arg=value)`.

Example entry:
```python
def create_my_constraint(self, **kwargs):
    def constraint(cst_parameters):
        return my_hard_check(cst_parameters, **kwargs)
    constraint.metadata = {'type': 'hard', 'name': 'my_constraint', 'kwargs': kwargs}
    return constraint
```

`metadata` is used in logs when constraints fail, so keep it descriptive.

## 4. Using the builder and evaluator
- The builder (`ConstraintSuiteBuilder`) composes hard/soft lists via `.add_hard_constraint(...)` and `.add_soft_constraint(...)`.
- Callers like `create_constraint_suite_simple` toggle which constraints run for a given epoch.
- `ConstraintEvaluator.evaluate` executes hard constraints first (violations still accumulate into `total_penalty`, but `hard_violation` short-circuits soft checks).
- Hard constraints should short-circuit quickly; soft constraints should return proportional penalties so CMA-ES can reason about gradients.

## 5. Testing and debugging
- Reuse `logging.debug` inside constraints to trace values (e.g., thickness, gap).
- Run diagnostic scripts (e.g., `python -m usingasb.diagnostics.constraint_diagnostic`) to visualize constraint behavior.
- When adding new constraints, ensure both simple and advanced suites behave as expected (e.g., by temporarily raising `logging.DEBUG`).

By following this pattern you keep constraint logic organized, testable, and easy to integrate into the existing factory/evaluator pipeline.

## Sample Constraint Integration Pipeline

This outlines the step-by-step process to implement, integrate, and test a new constraint within the optimization framework.

### Step 1: Implement Your Constraint Function

- Decide if your constraint is **hard** (its now more of a beefier soft violation for things that are binary (true or false), the penalty is applied if it is true) or **soft** (penalize violations).
- Implement it in the appropriate file under `constraints/`:
  - Hard constraints go in `hard_constraints_*.py`.
  - Soft constraints go in `soft_constraints_*.py`.
- Follow the return convention: `bool` for hard, numeric penalty for soft.
- Use utilities like `get_kulfan_coordinates()` or cached aero results if needed.
- Add detailed comments and logging for debugging.

### Step 2: Register the Constraint in `objective.py`

- Import your function at the top of `objective.py`.
- Add a factory method to `ConstraintFactory` wrapping your constraint and setting `metadata`.
- Add your constraint to the the generic constraint map in `create_constraint` if applicable.
- Include your constraint in suites like `create_constraint_suite_simple` with `.add_hard_constraint(...)` or `.add_soft_constraint(...)`.

Example factory method:
```python
def create_my_constraint(self, **kwargs):
    def constraint(cst_parameters):
        return my_hard_check(cst_parameters, **kwargs)
    constraint.metadata = {'type': 'hard', 'name': 'my_constraint', 'kwargs': kwargs}
    return constraint
```

### Step 3: Test the Constraint Function

- Write small unit tests or standalone scripts feeding known inputs triggering your constraint.
- Use debug logs to trace computation values.
- Run diagnostic scripts to visualize constraint effects.

### Step 4: Run Full Optimization

- Run CMA-ES via `run_single_cma` or `main.py`.
- Monitor logs to confirm constraint enforcement.
- Verify optimizer stability and reasonable search behavior.

### Step 5: Tune and Debug

- Adjust tolerances and penalties based on results.
- Add or improve log messages and metadata descriptions.
- Avoid false positives/negatives.

### Optional Step 6: Automated Tests

- Add unit/integration tests for your constraint.
- Cover edge cases and normal cases.
- Integrate into your test or diagnostic runs.
