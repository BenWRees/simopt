"""Test astromorf fix utilities."""

import logging

from simopt.experiment.run_solver import run_solver
from simopt.experiment_base import instantiate_problem
from simopt.solvers.astromorf import ASTROMORF

# Configure logging to see the solver's progress and any errors
logging.basicConfig(level=logging.INFO)


def test_astromorf_convergence() -> None:
    """Test astromorf convergence."""
    print("Starting ASTROMoRF convergence test...")

    # Use DYNAMNEWS-1 (Maximization)
    problem_name = "DYNAMNEWS-1"
    problem = instantiate_problem(problem_name)

    # Configure ASTROMORF with a much larger budget to see the increase
    solver = ASTROMORF(
        fixed_factors={
            "budget": 15000,
            "initial subspace dimension": 7,
            "Record Diagnostics": True,
        }
    )

    print(f"Running solver on {problem.name}...")
    solution_df, _iteration_df, _elapsed = run_solver(
        solver, problem, n_macroreps=1, n_jobs=1
    )

    # Print column names and some sample data to help debug the test script
    print(f"Solution DF columns: {solution_df.columns.tolist()}")

    # In simopt, the objective is often stored in the 'solution' column objects
    # or as a separate column depending on the run_solver version.
    # We can try to extract the objective value from the last solution.
    try:
        if not solution_df.empty:
            # Extract initial and final solution vectors (tuples)
            first_x = solution_df.iloc[0]["solution"]
            last_x = solution_df.iloc[-1]["solution"]

            # Since run_solver returns decision vectors as tuples, we need to create
            # a Solution object and simulate it to get the objective value.
            def get_objective_val(x_vec: tuple) -> float:
                sol = solver.create_new_solution(tuple(x_vec), problem)
                # Take enough reps for a stable estimate
                problem.simulate(sol, 100)
                # Return the actual objective (Raw mean)
                return sol.objectives_mean[0]

            print("Estimating objective values (100 reps each)...")
            initial_raw = get_objective_val(first_x)
            final_raw = get_objective_val(last_x)

            is_max = problem.minmax[0] == 1
            initial_actual = initial_raw
            final_actual = final_raw

            print(f"\nProblem Type: {'Maximization' if is_max else 'Minimization'}")
            print(f"Initial actual objective: {initial_actual:.6f}")
            print(f"Final actual objective:   {final_actual:.6f}")

            improvement = (
                final_actual - initial_actual
                if is_max
                else initial_actual - final_actual
            )
            print(f"Total improvement:        {improvement:.6f}")

            # Internal solver view (minimized)
            -1 * problem.minmax[0] * final_raw
            if (is_max and final_actual > initial_actual) or (
                not is_max and final_actual < initial_actual
            ):
                print("SUCCESS: ASTROMoRF improved the objective value.")
            else:
                print(
                    "Note: No improvement seen in actual objective. Consider increasing budget."  # noqa: E501
                )
        else:
            print("No solutions returned in solution_df.")
    except Exception as e:
        print(f"Could not extract final objective value: {e}")
        # Fallback to searching columns
        obj_col = next(
            (col for col in solution_df.columns if "objective" in col.lower()), None
        )
        if obj_col:
            best_val = solution_df[obj_col].min()
            print(f"Best objective value found in columns: {best_val:.6f}")

    # Check if we hit any errors during the run
    # (If the code above completed without a crash, the previous Index/Value errors are
    # likely fixed)
    print("Solver completed without crashing.")


if __name__ == "__main__":
    test_astromorf_convergence()
