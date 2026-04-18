"""Base classes for simulation optimization solvers."""

import contextlib
from abc import ABC, abstractmethod
from typing import Annotated, Any, ClassVar, cast

import pandas as pd
from boltons.typeutils import classproperty
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.multistage_problem import MultistageProblem
from simopt.problem import ProblemLike, Solution
from simopt.problem_types import (
    ConstraintType,
    ObjectiveType,
    SolverProblemType,
    VariableType,
)
from simopt.utils import get_specifications


class BudgetExhaustedError(Exception):
    """Raised when a solver exceeds its allotted simulation budget.

    This exception is thrown by :class:`Budget` when a call to
    :meth:`Budget.request` asks for more replications than remain in the
    available budget. It is caught in :meth:`Solver.run` to stop the
    macroreplication cleanly once the budget is exhausted.
    """


class Budget:
    """Tracks and enforces a solver's replication budget.

    A ``Budget`` instance is attached to each solver run and measures the number of
    simulation replications consumed. Solvers should call :meth:`request` before
    taking replications. If the request would exceed ``total``, a
    :class:`BudgetExhaustedException` is raised. This provides a consistent way for
    solvers to terminate exactly at the specified budget.

    Args:
        total (int): Total number of replications available for the run.
    """

    def __init__(self, total: int) -> None:
        """Initialize object with the total number of replications available."""
        self.total = total
        self._used = 0

    def request(self, amount: int) -> None:
        """Consume ``amount`` replications from the budget.

        Typical usage is to call ``request(r)`` immediately before taking ``r``
        replications at the current solution.

        Args:
            amount (int): Number of replications to consume.

        Raises:
            BudgetExhaustedException: If ``amount`` would cause usage to exceed
                :attr:`total`.
        """
        if self._used + amount > self.total:
            raise BudgetExhaustedError()
        self._used += amount

    @property
    def used(self) -> int:
        """Number of replications consumed so far."""
        return self._used

    @property
    def remaining(self) -> int:
        """Number of replications still available (``total - used``)."""
        return self.total - self._used


class SolverConfig(BaseModel):
    """Base class for solver configuration."""

    crn_across_solns: Annotated[
        bool, Field(default=True, description="use CRN across solutions?")
    ]


class Solver(ABC):
    """Base class to implement simulation-optimization solvers.

    This class defines the core structure for simulation-optimization
    solvers in SimOpt. Subclasses must implement the required methods
    for running simulations and updating solutions.

    Args:
        name (str): Name of the solver.
        fixed_factors (dict): Dictionary of user-specified solver factors.
    """

    class_name_abbr: ClassVar[str]
    """Short name of the solver class."""

    class_name: ClassVar[str]
    """Long name of the solver class."""

    config_class: ClassVar[type[SolverConfig]]
    """Configuration class for the solver."""

    objective_type: ClassVar[ObjectiveType]
    """Description of objective types."""

    constraint_type: ClassVar[ConstraintType]
    """Description of constraint types."""

    variable_type: ClassVar[VariableType]
    """Description of variable types."""

    supported_problem_type: ClassVar[SolverProblemType] = SolverProblemType.BOTH
    """Problem class support (Problem, MultistageProblem, or both)."""

    gradient_needed: ClassVar[bool]
    """True if gradient of objective function is needed, otherwise False."""

    def __init__(self, name: str = "", fixed_factors: dict | None = None) -> None:
        """Initialize a solver object.

        Args:
            name (str, optional): Name of the solver. Defaults to an empty string.
            fixed_factors (dict | None, optional): Dictionary of user-specified solver
                factors. Defaults to None.
        """
        self.name = name or self.name

        fixed_factors = fixed_factors or {}
        self.config = self.config_class(**fixed_factors)

        self.rng_list: list[MRG32k3a] = []
        self.solution_progenitor_rngs: list[MRG32k3a] = []

        self.recommended_solns = []
        self.intermediate_budgets = []
        self.fn_estimates = []
        self.budget_history = []
        self.iterations = []
        self.policy_solutions: list = []

    def __eq__(self, other: object) -> bool:
        """Check if two solvers are equivalent.

        Args:
            other (object): Other object to compare to self.

        Returns:
            bool: True if the two objects are equivalent, otherwise False.
        """
        if not isinstance(other, Solver):
            return False
        return type(self) is type(other) and self.factors == other.factors

    def __hash__(self) -> int:
        """Return the hash value of the solver.

        Returns:
            int: Hash value of the solver.
        """
        return hash((self.name, tuple(self.factors.items())))

    @classproperty
    def compatibility(cls) -> str:  # noqa: N805
        """Compatibility of the solver."""
        return (
            f"{cls.objective_type.symbol()}"
            f"{cls.constraint_type.symbol()}"
            f"{cls.variable_type.symbol()}"
            f"{'G' if cls.gradient_needed else 'N'}"
        )

    @classproperty
    def specifications(cls) -> dict[str, dict]:  # noqa: N805
        """Details of each factor (for GUI, data validation, and defaults)."""
        return get_specifications(cls.config_class)

    @property
    def factors(self) -> dict:
        """Changeable factors (i.e., parameters) of the solver."""
        return self.config.model_dump(by_alias=True)

    def attach_rngs(self, rng_list: list[MRG32k3a]) -> None:
        """Attach a list of random-number generators to the solver.

        Args:
            rng_list (list[``mrg32k3a.mrg32k3a.MRG32k3a``]): List of random-number
                generators used for the solver's internal purposes.
        """
        self.rng_list = rng_list

    @abstractmethod
    def solve(self, problem: ProblemLike) -> None:
        """Run a single macroreplication of a solver on a problem.

        Args:
            problem: Simulation-optimization problem to solve.
        """
        raise NotImplementedError

    def run(self, problem: ProblemLike) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Run the solver on a problem.

        Args:
            problem (Problem): The problem to solve.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]: A tuple containing:
                - Solution DataFrame with columns: step, budget, solution
                - Iteration DataFrame with columns: iteration, budget_history,
                fn_estimate
                  (or None if iteration data is not available)
        """
        if not self.supports_problem(problem):
            expected = self.supported_problem_type.display_name()
            error_msg = (
                f"{self.name} supports {expected} instances; "
                f"got {type(problem).__name__}."
            )
            raise ValueError(error_msg)

        self.budget = Budget(problem.factors["budget"])
        with contextlib.suppress(BudgetExhaustedError):
            self.solve(problem)

        # Auto-build policy solutions for multistage runs when a solver
        # supports multistage problems but does not populate them directly.
        if (
            not self.policy_solutions
            and self.supported_problem_type
            in (
                SolverProblemType.BOTH,
                SolverProblemType.MULTISTAGE,
            )
            and isinstance(problem, MultistageProblem)
            and self.recommended_solns
        ):
            # Import locally to avoid import cycles during module initialization.

            def _stage0_decision(x: object) -> tuple[float, ...]:
                if isinstance(x, tuple | list):
                    if x and isinstance(x[0], tuple | list):
                        nested = cast(tuple | list, x[0])
                        return tuple(float(cast(Any, v)) for v in nested)
                    base = cast(tuple | list, x)
                    return tuple(float(cast(Any, v)) for v in base)
                return ()

            n_stages = problem.model.n_stages
            auto_policy_solutions = []
            for rec_sol in self.recommended_solns:
                stage0_x = _stage0_decision(rec_sol.x)
                if not stage0_x:
                    continue

                policy_fn = problem.build_policy(self, stage0_x)
                decisions = tuple(
                    stage0_x if t == 0 else tuple(0.0 for _ in stage0_x)
                    for t in range(n_stages)
                )
                policy_solution = problem.create_policy_solution(
                    decisions,
                    policy=policy_fn,
                )
                policy_solution.attach_rngs(
                    [
                        MRG32k3a(s_ss_sss_index=[0, i, 0])
                        for i in range(problem.model.n_rngs)
                    ],
                    copy=False,
                )
                auto_policy_solutions.append(policy_solution)

            if auto_policy_solutions:
                self.policy_solutions = auto_policy_solutions

        # Capture solution-level data
        recommended_solns = self.recommended_solns
        intermediate_budgets = self.intermediate_budgets
        policy_solutions = self.policy_solutions

        # Capture iteration-level data
        budget_history = self.budget_history
        fn_estimates = self.fn_estimates
        iterations = self.iterations

        # Reset for next run
        self.recommended_solns = []
        self.intermediate_budgets = []
        self.budget_history = []
        self.fn_estimates = []
        self.iterations = []
        self.policy_solutions = []

        # Build the solution DataFrame
        solution_df = pd.DataFrame(
            {
                "step": range(len(recommended_solns)),
                "budget": intermediate_budgets,
                "solution": recommended_solns,
            }
        )
        solution_df["solution"] = solution_df["solution"].apply(
            lambda solution: solution.x
        )

        if policy_solutions and len(policy_solutions) == len(recommended_solns):
            solution_df["policy_solution"] = policy_solutions

        # Build the iteration DataFrame (if data is available)
        iteration_df = None
        if iterations and budget_history and fn_estimates:
            iteration_df = pd.DataFrame(
                {
                    "iteration": iterations,
                    "budget_history": budget_history,
                    "fn_estimate": fn_estimates,
                }
            )

        return solution_df, iteration_df

    def create_new_solution(self, x: tuple, problem: ProblemLike) -> Solution:
        """Create a new solution object with attached RNGs.

        Args:
            x (tuple): A vector of decision variables.
            problem (Problem): The problem instance associated with the solution.

        Returns:
            Solution: New solution object for the given decision variables and problem.
        """
        # Create new solution with attached rngs.
        new_solution = Solution(x, problem)
        new_solution.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
        # Manipulate progenitor rngs to prepare for next new solution.
        if not self.config.crn_across_solns:  # If CRN are not used ...
            # ...advance each rng to start of the substream
            # substream = current substream + # of model RNGs.
            for rng in self.solution_progenitor_rngs:
                for _ in range(problem.model.n_rngs):
                    rng.advance_substream()
        return new_solution

    def supports_problem(self, problem: ProblemLike) -> bool:
        """Return whether this solver supports the concrete problem class."""
        is_multistage = isinstance(problem, MultistageProblem)
        if self.supported_problem_type == SolverProblemType.BOTH:
            return True
        if self.supported_problem_type == SolverProblemType.MULTISTAGE:
            return is_multistage
        return not is_multistage
