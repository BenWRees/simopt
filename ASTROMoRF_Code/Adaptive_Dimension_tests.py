"""Within this script, we test ASTROMoRF's adaptive dimension mechanic and see if it.

accurately predicts the optimal dimension within a trust-region.
"""

import numpy as np

# * Global variables to track iteration counts and current dimension
unsucusseful_iteration_count = 0
successful_iteration_count = 0
initial_dimension = 5
recent_objective_values = []  # store recent objective function values for plateau detection
plateau_window_size = 10
in_plateau = False
problem_dimension = 10
model_gradients = []  # store model gradients for dimension logic


#! ==== Adaptive Dimension Logic ====
def adaptive_dimension() -> int:  # noqa: D103
    if unsucusseful_iteration_count >= 3:
        # increase dimension and reset counter
        return run_dimension_update()
    return initial_dimension


def run_dimension_update() -> int:  # noqa: D103
    # check for plateau in function values
    # if plateaued, run the plateau logic
    if detect_plateau() and not in_plateau:  # noqa: F823
        new_d = plateau_logic()
        in_plateau = True
        new_d = dimension_reset()

    if in_plateau:
        new_d = plateau_decrease_dimension()

    else:
        # Normal update logic
        # create a moving window of the most recent model_gradients
        if len(model_gradients) > 5:
            model_gradients_window = model_gradients[-5:]

        new_d = dimension_logic(model_gradients_window)

    return new_d


#! ==== Plateau Detection and Logic ====
def detect_plateau() -> bool:
    """Detect if the solver progress has plateaued.

        A plateau is detected only when:
        1. The objective function has shown little improvement over many iterations
        2. Multiple unsuccessful iterations have occurred consecutively.


    Returns:
        bool: _description_
    """
    if len(recent_objective_values) < plateau_window_size:
        return False

    if in_plateau:
        return False

    recent_values = recent_objective_values[-plateau_window_size:]

    is_maximization = False  # assuming minimization problem
    improvement_threshold = 1e-4

    if is_maximization:
        best_recent = max(recent_values)
        worst_recent = min(recent_values)

    else:
        best_recent = min(recent_values)
        worst_recent = max(recent_values)

    first_obj = abs(recent_values[0]) if abs(recent_values[0]) > 1e-8 else 1.0
    relative_improvement = abs(best_recent - worst_recent) / first_obj

    no_improvement = all(
        abs(recent_values[i] - recent_values[0]) < improvement_threshold
        for i in range(1, len(recent_values))
    )

    return (
        relative_improvement < improvement_threshold
        and unsucusseful_iteration_count >= 3
    ) or (no_improvement and unsucusseful_iteration_count >= 3)


def dimension_reset() -> int:
    """Reset dimension to problem dimension for plateau.

    Returns:
        int: _description_.
    """
    return problem_dimension - 1


def plateau_logic() -> int:  # noqa: D103
    # run a series of subspace dimension tests to determine the best dimension
    pass


def plateau_decrease_dimension() -> int:  # noqa: D103
    # controlled decrease of dimension when in plateau
    pass


#! ==== Metrics for Dimension Selection ====
def dimension_logic(model_gradients: list[np.array]) -> int:
    """Computes the new dimension based on the following factors:.

        1. Consecutive unsuccessful iterations suggests a slight increase in dimension
        2. Consecutive successful iterations suggests a slight decrease in dimension
        3. flat model gradients suggest more dimensions
        4. high model prediction errer suggests more dimensions
        5. low model prediction error suggests fewer dimensions
        6. Estimate the optimal subspace from eigenvalue spectrum of model gradient.

    Returns:
        int: optimal subspace dimension for next iteration
    """


def estimate_from_eigenvalues() -> int:  # noqa: D103
    pass


#! ==== Run iterations ====
def run_iterations(num_iterations: int) -> None:  # noqa: D103
    pass


def main() -> None:  # noqa: D103
    pass


if __name__ == "__main__":
    main()
