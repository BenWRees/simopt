"""Base class for multistage simulation-optimization problems.

A multistage problem wraps a :class:`MultistageModel` and provides the
optimisation interface that solvers interact with. It manages:

- **State tracking** across stages (current state, current stage).
- **Lookahead budget** - how many Monte Carlo replications are used to
  estimate the cost-to-go when evaluating a candidate decision.
- **Stage advancement** - deterministically applying a chosen decision and
  stepping to the next stage.
- **Reset** - returning to the initial state for a new episode.

Solver-based lookahead
~~~~~~~~~~~~~~~~~~~~~~
When :meth:`MultistageProblem.simulate` is called with a
``lookahead_solver`` argument, the standard policy-based Monte Carlo
lookahead is replaced by an *optimisation-based* estimate of the
cost-to-go.  At each future stage the given solver is run on a
sub-problem that represents the remaining stages, and the best decision
found is used to advance the state.  The sub-problems themselves still
use the default rollout policy for *their* cost-to-go estimates so the
solver is **not** applied recursively, keeping the computational cost
manageable.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    cast,
    runtime_checkable,
)

import numpy as np
from boltons.typeutils import classproperty
from pydantic import BaseModel

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.multistage_model import MultistageModel
from simopt.problem import Objective, RepResult, Solution
from simopt.problem_types import ConstraintType, VariableType
from simopt.utils import get_specifications

if TYPE_CHECKING:
    from simopt.solver import Solver

logger = logging.getLogger(__name__)

TraceValue = str | int | float | None | Exception


@runtime_checkable
class SupportsInitialSolution(Protocol):
    """Protocol for objects that expose an ``initial_solution`` tuple."""

    initial_solution: tuple[float]  # refine later if you know the exact type


class PolicyFunction:
    """Picklable policy callable for ADP solver.

    Replaces the closure returned by ``_build_policy`` so that policy
    solutions can be serialised by joblib/loky for parallel post-replication.
    """

    def __init__(
        self,
        stage0_dec: tuple,
        solver: Solver,
        problem: MultistageProblem,
    ) -> None:
        """Initialize a picklable policy wrapper.

        Args:
            stage0_dec: First-stage decision to replay at stage 0.
            solver: Solver used for downstream stage decisions.
            problem: Multistage problem context for policy evaluation.
        """
        self.stage0_dec = stage0_dec
        self.solver = solver
        self.problem = problem

    def __call__(self, state: object, stage: int) -> tuple:
        """Policy function in SDP literature.

        Args:
            state (object): Current simulation state at this lookahead step.
            stage (int): Current stage index

        Returns:
            tuple: Decision tuple for the given state and stage.
        """
        if stage == 0:
            return self.stage0_dec

        action = self.problem.select_action_via_simulation(state=state, stage=stage)
        return tuple(float(v) for v in action)


class MultistageProblem(ABC):
    """Base class for multistage simulation-optimization problems.

    At each stage *t* the solver calls :meth:`simulate` to evaluate a
    candidate decision *x_t* at the current state. Under the hood this:

    1. Computes the immediate stage reward via the model.
    2. Estimates the cost-to-go for remaining stages via Monte Carlo
       lookahead using a rollout policy.
    3. Aggregates both into :class:`Objective` values stored on the
       :class:`Solution`.

    After the solver selects a decision, :meth:`advance_stage` transitions
    the problem to the next stage. :meth:`reset` returns to stage 0.
    """

    class_name_abbr: ClassVar[str]
    """Short name of the problem class."""

    class_name: ClassVar[str]
    """Long name of the problem class."""

    config_class: ClassVar[type[BaseModel]]
    """Configuration class for the problem."""

    model_class: ClassVar[type[MultistageModel]]
    """Multistage simulation model class."""

    constraint_type: ClassVar[ConstraintType]
    """Description of constraint types."""

    variable_type: ClassVar[VariableType]
    """Description of variable types."""

    gradient_available: ClassVar[bool]
    """Whether gradient information is available."""

    n_objectives: ClassVar[int]
    """Number of objectives."""

    minmax: ClassVar[tuple[int, ...]]
    """Indicators of maximization (+1) or minimization (-1) per objective."""

    n_stochastic_constraints: ClassVar[int]
    """Number of stochastic constraints."""

    model_default_factors: ClassVar[dict]
    """Default values overriding model-level defaults."""

    model_decision_factors: ClassVar[set[str]]
    """Factor keys that are decision variables."""

    n_lookahead_reps: ClassVar[int]
    """Number of Monte Carlo replications for cost-to-go estimation."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        name: str = "",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize a multistage problem.

        Args:
                name: Name of the problem instance.
                fixed_factors: User-specified problem factors.
                model_fixed_factors: Non-decision factors forwarded to the model.
        """
        self.name = name or self.class_name_abbr

        # Problem-level configuration
        fixed_factors = fixed_factors or {}
        self.config = self.config_class(**fixed_factors)
        self._factors = self.config.model_dump(by_alias=True)

        # Model-level configuration
        model_factors: dict = {}
        model_config = self.model_class.config_class()
        model_factors.update(model_config.model_dump(by_alias=True))
        model_factors.update(self.model_default_factors)
        model_fixed_factors = model_fixed_factors or {}
        model_factors.update(model_fixed_factors)

        self.model = self.model_class(model_factors)

        # RNG list (attached later via :meth:`attach_rngs`)
        self.rng_list: list[MRG32k3a] = []

        # Stage / state tracking
        self.current_stage: int = 0
        self.current_state: Any = self.model.get_initial_state()

        self.before_replicate_override = None

        # When True, simulate() honours the ``lookahead_solver``
        # argument.  Sub-problems created by _create_stage_subproblem
        # set this to False so the solver is NOT applied recursively.
        self._solver_lookahead_enabled: bool = True
        self._lookahead_trace_depth: int = 0
        self.lookahead_solver: Solver | None = None

    # ------------------------------------------------------------------
    # Equality / hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, MultistageProblem):
            return False
        if type(self) is type(other) and self.factors == other.factors:
            non_decision = set(self.model.factors.keys()) - self.model_decision_factors
            return all(
                self.model.factors[k] == other.model.factors[k] for k in non_decision
            )
        return False

    @staticmethod
    def _make_hashable(value: Any) -> Any:  # noqa: ANN401
        """Convert a value to a hashable form (lists → tuples, recursively)."""
        if isinstance(value, list):
            return tuple(MultistageProblem._make_hashable(v) for v in value)
        return value

    def __hash__(self) -> int:
        """Return hash."""
        non_decision = set(self.model.factors.keys()) - self.model_decision_factors
        return hash(
            (
                self.name,
                tuple((k, self._make_hashable(v)) for k, v in self.factors.items()),
                tuple(
                    (k, self._make_hashable(self.model.factors[k]))
                    for k in sorted(non_decision)
                ),
            )
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def optimal_value(self) -> float | None:
        """Optimal objective function value (if known)."""
        return None

    @property
    def optimal_solution(self) -> tuple | None:
        """Optimal solution (if known)."""
        return None

    @property
    @abstractmethod
    def dim(self) -> int:
        """Number of decision variables at each stage."""
        raise NotImplementedError

    @property
    @abstractmethod
    def lower_bounds(self) -> tuple[float, ...]:
        """Lower bound for each decision variable."""
        raise NotImplementedError

    @property
    @abstractmethod
    def upper_bounds(self) -> tuple[float, ...]:
        """Upper bound for each decision variable."""
        raise NotImplementedError

    @classproperty
    def compatibility(cls) -> str:  # noqa: N805
        """Solver compatibility string."""
        return (
            "S"
            f"{cls.constraint_type.symbol()}"
            f"{cls.variable_type.symbol()}"
            f"{'G' if cls.gradient_available else 'N'}"
        )

    @classproperty
    def specifications(cls) -> dict[str, dict]:  # noqa: N805
        """Factor specifications for the problem."""
        return get_specifications(cls.config_class)

    @property
    def factors(self) -> dict:
        """Changeable factors of the problem."""
        return self._factors

    @property
    def is_terminal(self) -> bool:
        """Whether all stages have been completed."""
        return self.current_stage >= self.model.n_stages

    @property
    def lookahead_enabled(self) -> bool:
        """Whether solver-based lookahead is enabled."""
        return self._solver_lookahead_enabled

    @property
    def lookahead_solver(self) -> Solver | None:
        """Solver used for lookahead when simulating."""
        return self._lookahead_solver

    @lookahead_solver.setter
    def lookahead_solver(self, solver: Solver | None) -> None:
        """Set the solver used for lookahead when simulating."""
        self._lookahead_solver = solver

    def _lookahead_trace_enabled(self) -> bool:
        """Return whether verbose lookahead tracing is enabled."""
        return os.getenv("SIMOPT_TRACE_LOOKAHEAD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _trace_lookahead(self, message: str, **fields: TraceValue) -> None:
        """Emit a concise print trace line for lookahead diagnostics."""
        if not self._lookahead_trace_enabled():
            return

        context: dict[str, object] = {
            "problem": type(self).__name__,
            "problem_id": id(self),
            "depth": self._lookahead_trace_depth,
            "current_stage": self.current_stage,
            "lookahead_enabled": self._solver_lookahead_enabled,
        }
        if self.lookahead_solver is not None:
            context["lookahead_solver"] = type(self.lookahead_solver).__name__
            context["lookahead_solver_supported"] = getattr(
                self.lookahead_solver,
                "supported_problem_type",
                None,
            )
        context.update(fields)

        context_str = " ".join(f"{k}={v}" for k, v in context.items())
        print(f"[LOOKAHEAD_TRACE] {context_str} | {message}")

    # ------------------------------------------------------------------
    # RNG management
    # ------------------------------------------------------------------

    def attach_rngs(self, rng_list: list[MRG32k3a]) -> None:
        """Attach random-number generators to the problem.

        Args:
                rng_list: RNGs for random initial solutions or problem instances.
        """
        self.rng_list = rng_list

    def _clone_problem(self) -> MultistageProblem:
        """Create a fresh multistage problem with copied runtime state."""
        cloned = type(self)(
            fixed_factors=dict(self.factors),
            model_fixed_factors=dict(self.model.factors),
        )
        cloned.name = self.name
        cloned.rng_list = list(self.rng_list)
        cloned.before_replicate_override = self.before_replicate_override
        cloned.current_stage = self.current_stage
        cloned.current_state = deepcopy(self.current_state)
        cloned._solver_lookahead_enabled = getattr(
            self, "_solver_lookahead_enabled", True
        )
        cloned._lookahead_trace_depth = getattr(self, "_lookahead_trace_depth", 0)
        cloned.model.model_created()
        return cloned

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """Convert a decision vector to a dictionary of factor keys.

        Args:
                vector: Decision variable values.

        Returns:
                Factor dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """Convert a factor dictionary to a decision vector.

        Args:
                factor_dict: Factor keys and values.

        Returns:
                Decision variable tuple.
        """
        raise NotImplementedError

    @abstractmethod
    def replicate(self, x: tuple) -> RepResult:
        """Evaluate a decision at the current state and stage.

        Implementations should call :meth:`MultistageModel.replicate` with
        the current ``(state, decision, stage)`` and wrap the response into
        a :class:`RepResult`.

        Args:
                x: Decision vector.

        Returns:
                Replication result containing objectives (and optional constraints).
        """
        raise NotImplementedError

    @abstractmethod
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """Generate a random feasible solution for restarts.

        Args:
                rand_sol_rng: RNG for sampling.

        Returns:
                Random decision vector.
        """
        raise NotImplementedError

    def check_deterministic_constraints(self, x: tuple, /) -> bool:
        """Check deterministic feasibility of a decision.

        By default only checks box constraints. Override to add more.

        Args:
                x: Decision vector.

        Returns:
                True if feasible.
        """
        return all(
            lb <= xi <= ub
            for xi, lb, ub in zip(x, self.lower_bounds, self.upper_bounds, strict=False)
        )

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def advance_stage(
        self,
        decision: tuple,
        rng_list: list[MRG32k3a],
    ) -> dict[str, float]:
        """Apply *decision* at the current stage and advance to the next.

        Transitions the state, increments the stage counter, and returns the
        realised (stochastic) stage reward.

        Args:
                decision: Decision applied at the current stage.
                rng_list: RNGs for the stochastic transition.

        Returns:
                Dictionary of stage reward responses.

        Raises:
                RuntimeError: If all stages have already been completed.
        """
        if self.is_terminal:
            raise RuntimeError(
                f"Cannot advance beyond the final stage "
                f"(n_stages={self.model.n_stages})."
            )
        responses, next_state = self.model.replicate_stage(
            self.current_state, decision, self.current_stage, rng_list
        )
        self.current_state = next_state
        self.current_stage += 1
        return responses

    def reset(self) -> None:
        """Reset the problem to the initial state and stage 0."""
        self.current_stage = 0
        self.current_state = self.model.get_initial_state()

    def get_candidate_decisions(
        self,
        state: object,
        stage: int,  # noqa: ARG002
        n_candidates: int,
        rng: np.random.Generator | None = None,
    ) -> list[tuple]:
        """Sample *n_candidates* decisions uniformly within the feasible box.

        The default implementation draws uniformly at random from
        ``[lower_bounds, upper_bounds]``.  Subclasses should override this
        method when the feasible region is not a simple box or when a more
        informed candidate set is available (e.g. neighbourhood around the
        current incumbent, problem-specific structure).

        ``lower_bounds`` and ``upper_bounds`` are evaluated with
        ``self.current_state`` **temporarily set to** *state*, so that
        subclass implementations whose bounds depend on the remaining
        capacity (e.g. ``AirlineRevenueMultistageProblem``) automatically
        return correct bounds for each future simulation step rather than
        stale bounds from the outermost stage.

        Args:
                state: Current simulation state at this lookahead step.  Used
                        to derive state-dependent bounds; passed to
                        ``lower_bounds`` / ``upper_bounds`` via the temporary
                        ``current_state`` swap.
                stage: Current stage index (available for context; not used by
                        the default implementation).
                n_candidates: Number of candidates to return.
                rng: Optional NumPy random generator.  When *None* a fresh
                        default-seeded generator is created.

        Returns:
                List of decision tuples, one per candidate.
        """
        rng = rng or np.random.default_rng()
        # Temporarily point current_state at the lookahead state so that
        # state-dependent upper_bounds (e.g. capped by remaining_capacity)
        # are evaluated correctly for this future step.
        saved_state = self.current_state
        self.current_state = state
        try:
            lb = np.array(self.lower_bounds, dtype=float)
            ub = np.array(self.upper_bounds, dtype=float)
        finally:
            self.current_state = saved_state
        samples = rng.uniform(lb, ub, size=(n_candidates, self.dim))
        return [tuple(float(v) for v in row) for row in samples]

    def select_action_via_simulation(
        self,
        state: object,
        stage: int,
        n_candidates: int = 24,
        n_eval_reps: int = 2,
        rng: np.random.Generator | None = None,
    ) -> tuple:
        """Select a stage action by simulation-based lookahead scoring.

        Candidate actions are sampled via :meth:`get_candidate_decisions` and
        evaluated by repeated calls to :meth:`replicate` on a stage-positioned
        sub-problem. The action with the best mean objective (respecting
        ``minmax[0]``) is returned.

        Args:
                state: State at which to choose an action.
                stage: Stage index corresponding to *state*.
                n_candidates: Number of sampled candidate actions.
                n_eval_reps: Number of replication evaluations per candidate.
                rng: Optional NumPy RNG for candidate sampling.

        Returns:
                Decision tuple chosen by simulation lookahead.
        """
        if stage < 0 or stage >= self.model.n_stages:
            raise ValueError(
                f"Stage {stage} is out of range for n_stages={self.model.n_stages}."
            )

        rng = rng or np.random.default_rng()
        sub_problem = self._create_stage_subproblem(deepcopy(state), stage)
        candidates = sub_problem.get_candidate_decisions(
            state=state,
            stage=stage,
            n_candidates=max(1, int(n_candidates)),
            rng=rng,
        )
        if not candidates:
            saved_state = sub_problem.current_state
            sub_problem.current_state = state
            try:
                lb = np.array(sub_problem.lower_bounds, dtype=float)
                ub = np.array(sub_problem.upper_bounds, dtype=float)
            finally:
                sub_problem.current_state = saved_state
            return tuple(float(v) for v in ((lb + ub) / 2.0).tolist())

        best_candidate: tuple | None = None
        best_utility = -np.inf
        n_eval_reps = max(1, int(n_eval_reps))

        for candidate in candidates:
            cand = tuple(float(v) for v in candidate)
            if not sub_problem.check_deterministic_constraints(cand):
                continue

            values: list[float] = []
            for _ in range(n_eval_reps):
                # Reposition before each replicate to avoid accidental state carry-over.
                sub_problem.current_state = deepcopy(state)
                sub_problem.current_stage = stage
                try:
                    rep_result = sub_problem.replicate(cand)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.debug(
                        "Action scoring failed at stage %d for candidate %s: %s",
                        stage,
                        cand,
                        exc,
                    )
                    continue

                if not rep_result.objectives:
                    continue
                obj_val = rep_result.objectives[0].stochastic
                if obj_val is None:
                    continue
                obj_float = float(obj_val)
                if not np.isfinite(obj_float):
                    continue
                values.append(obj_float)

            if not values:
                continue

            mean_obj = float(np.mean(values))
            utility = float(self.minmax[0]) * mean_obj
            if utility > best_utility:
                best_utility = utility
                best_candidate = cand

        if best_candidate is not None:
            return best_candidate

        feasible = [
            tuple(float(v) for v in c)
            for c in candidates
            if sub_problem.check_deterministic_constraints(tuple(float(v) for v in c))
        ]
        if feasible:
            return feasible[0]

        return tuple(float(v) for v in candidates[0])

    # ------------------------------------------------------------------
    # Simulation entry-point
    # ------------------------------------------------------------------
    def simulate_immediate_reward(
        self,
        solution: Solution,
        num_macroreps: int = 1,
    ) -> None:
        """Simulate only the immediate reward for the current stage, without lookahead.

        Useful for approximate dynamic programming approaches where the value
        function is approximated separately and not via simulation-based
        lookahead.

        Args:
                solution: The solution object containing the decision to
                        evaluate and RNGs.
                num_macroreps: The number of macro-replications to run.
        """
        self.model.factors.update(solution.decision_factors)
        for _ in range(num_macroreps):
            self.model.before_replication(solution.rng_list)
            if self.before_replicate_override is not None:
                self.before_replicate_override(self.model, solution.rng_list)

            self._trace_lookahead(
                "simulate macrorep start",
                num_macroreps=num_macroreps,
                solution_n_reps=solution.n_reps,
            )

            responses, _ = self.model.replicate(
                state=self.current_state,
                decision=solution.x,
                stage=self.current_stage,
                n_lookahead_reps=0,  # No lookahead
            )

            objectives = [
                Objective(stochastic=responses[key])
                for key in sorted(k for k in responses if k.startswith("total_"))
            ]
            rep_result = RepResult(objectives=objectives)
            solution.add_replicate_result(rep_result)

            # Advance RNGs to the start of the next sub-substream.
            for rng in solution.rng_list:
                rng.advance_subsubstream()

    def simulate_immediate_reward_up_to(
        self, solutions: list[Solution], n_reps: int
    ) -> None:
        """Simulate immediate rewards up to *n_reps* replications each.

        Args:
                solutions: Solutions to simulate.
                n_reps: Target number of replications for each solution.
        """
        for solution in solutions:
            if solution.n_reps < n_reps:
                self.simulate_immediate_reward(
                    solution, num_macroreps=n_reps - solution.n_reps
                )

    def simulate(
        self,
        solution: Solution,
        num_macroreps: int = 1,
    ) -> None:
        """Simulate *num_macroreps* replications at the current stage.

        Each replication evaluates the decision encoded in *solution* at the
        current ``(state, stage)`` - including an estimate of the cost-to-go
        for future stages - and stores the result on the solution object.

        Three lookahead modes are available, selected in priority order:

        1. **Solver-based** (``lookahead_solver`` is not *None*): runs a full
           optimisation solver at every future stage.  Most accurate but most
           expensive.
        3. **Policy rollout** (default): rolls out future stages with a fixed
           default policy.  Cheapest but does *not* optimise at future stages.

        Args:
                solution: Solution to evaluate (carries decision and RNGs).
                num_macroreps: Number of i.i.d. replications to run.
        """
        self.model.factors.update(solution.decision_factors)
        for _ in range(num_macroreps):
            self.model.before_replication(solution.rng_list)
            if self.before_replicate_override is not None:
                self.before_replicate_override(self.model, solution.rng_list)

            self._trace_lookahead(
                "simulate macrorep start",
                num_macroreps=num_macroreps,
                solution_n_reps=solution.n_reps,
            )

            logger.debug(
                "simulate called: solver_lookahead_enabled=%s, lookahead_solver=%s",
                self._solver_lookahead_enabled,
                type(self.lookahead_solver).__name__
                if self.lookahead_solver is not None
                else None,
            )

            if self._solver_lookahead_enabled:
                # Use the currently assigned solver so callers can swap
                # lookahead solvers between runs on the same problem instance.
                active_lookahead_solver = self.lookahead_solver
                if active_lookahead_solver is None:
                    self._trace_lookahead(
                        "solver lookahead enabled but no lookahead solver attached"
                    )
                    raise RuntimeError(
                        "Solver lookahead is enabled, but lookahead_solver is "
                        "None. Assign problem.lookahead_solver before "
                        "simulating or disable solver lookahead for this "
                        "problem/sub-problem."
                    )
                self._trace_lookahead(
                    "dispatching to solver-based lookahead replication",
                    active_solver=type(active_lookahead_solver).__name__,
                    active_solver_supported=getattr(
                        active_lookahead_solver,
                        "supported_problem_type",
                        None,
                    ),
                )
                result = self._replicate_with_solver_lookahead(
                    solution.x,
                    active_lookahead_solver,
                )
            else:
                # In sub-problems, always fall back to policy-based rollout,
                # ignoring any solver-based flags.
                self._trace_lookahead("solver lookahead disabled, using replicate")
                result = self.replicate(solution.x)

            solution.add_replicate_result(result)

            # Advance RNGs to the start of the next sub-substream.
            for rng in solution.rng_list:
                rng.advance_subsubstream()

    def simulate_up_to(self, solutions: list[Solution], n_reps: int) -> None:
        """Simulate solutions up to *n_reps* replications each.

        Args:
                solutions: Solutions to simulate.
                n_reps: Target number of replications for each solution.
        """
        for solution in solutions:
            if solution.n_reps < n_reps:
                self.simulate(solution, num_macroreps=n_reps - solution.n_reps)

    # ------------------------------------------------------------------
    # Policy-based full-trajectory evaluation
    # ------------------------------------------------------------------

    def create_policy_solution(
        self,
        decisions: tuple[tuple, ...],
        policy: Callable[[Any, int], tuple] | None = None,
    ) -> Solution:
        """Create a :class:`Solution` for full-trajectory policy evaluation.

        The returned solution has its ``policy`` attribute set so that
        :meth:`simulate_policy` can evaluate the full trajectory.  When no
        explicit *policy* callable is provided, the *decisions* tuple is
        used as an open-loop policy (``decisions[stage]`` at each stage).

        Args:
                decisions: Tuple of per-stage decision tuples, one per stage:
                        ``((x_0_1, ...), (x_1_1, ...), ..., (x_{T-1}_1, ...))``.
                policy: Optional callable ``(state, stage) -> decision_tuple``.
                        When *None*, an open-loop policy derived from *decisions*
                        is used.

        Returns:
                A :class:`Solution` ready for :meth:`simulate_policy`.
        """
        # Use first-stage decision for decision_factors / vector_to_factor_dict
        sol = Solution(decisions[0], self)
        sol.x = decisions  # Override with full tuple-of-tuples
        sol.policy = policy
        return sol

    def simulate_policy(
        self,
        solution: Solution,
        num_macroreps: int = 1,
    ) -> None:
        """Evaluate a policy over the full trajectory via MC sample paths.

        Each macroreplication simulates one complete episode from stage 0
        to the terminal stage, accumulating the total reward across all
        stages.  The policy is obtained from ``solution.policy`` (if set)
        or by indexing ``solution.x[stage]`` as an open-loop sequence of
        per-stage decisions.

        Args:
                solution: Solution carrying a policy and/or tuple-of-tuples
                        ``x``.  Typically created via :meth:`create_policy_solution`.
                num_macroreps: Number of independent sample paths to simulate.
        """
        if solution.policy is not None:
            policy = solution.policy
        else:
            # Open-loop: x is a tuple of tuples, one per stage
            def policy(state: object, stage: int) -> tuple:  # noqa: ARG001
                return solution.x[stage]

        for _ in range(num_macroreps):
            self.model.before_replication(solution.rng_list)
            if self.before_replicate_override is not None:
                self.before_replicate_override(self.model, solution.rng_list)

            total_responses: dict[str, float] = {}
            state = self.model.get_initial_state()

            for stage in range(self.model.n_stages):
                decision = policy(state, stage)
                reward, next_state = self.model.replicate_stage(
                    state, decision, stage, self.model._rng_list
                )
                for key, val in reward.items():
                    total_responses[key] = total_responses.get(key, 0.0) + val
                state = next_state

            objectives = [
                Objective(stochastic=total_responses[key])
                for key in sorted(total_responses)
            ]
            solution.add_replicate_result(RepResult(objectives=objectives))

            for rng in solution.rng_list:
                rng.advance_subsubstream()

    def simulate_policy_up_to(self, solutions: list[Solution], n_reps: int) -> None:
        """Simulate policy solutions up to *n_reps* replications each.

        Args:
                solutions: Solutions to simulate.
                n_reps: Target number of replications for each solution.
        """
        for solution in solutions:
            if solution.n_reps < n_reps:
                self.simulate_policy(solution, num_macroreps=n_reps - solution.n_reps)

    # ------------------------------------------------------------------
    # Solver-based lookahead
    # ------------------------------------------------------------------

    def _create_stage_subproblem(
        self,
        state: object,
        stage: int,
    ) -> MultistageProblem:
        """Create a *shallow* sub-problem positioned at *(state, stage)*.

        The sub-problem shares the same model class and configuration but
        has its internal state set to the given values.  It does **not**
        carry a lookahead solver - its :meth:`replicate` will use the
        standard policy-based Monte Carlo lookahead.

        Args:
                state: State to start the sub-problem from.
                stage: Stage index for the sub-problem.

        Returns:
                A new :class:`MultistageProblem` positioned at *(state, stage)*.
        """
        sub = self._clone_problem()
        sub.current_state = state
        sub.current_stage = stage
        # Prevent recursive solver-based lookahead: sub-problems
        # use the default policy-based Monte Carlo rollout.
        sub._solver_lookahead_enabled = False
        sub._lookahead_trace_depth = self._lookahead_trace_depth + 1
        self._trace_lookahead(
            "created stage sub-problem",
            sub_problem_id=id(sub),
            sub_problem_stage=stage,
            sub_problem_depth=sub._lookahead_trace_depth,
            sub_problem_lookahead_enabled=sub._solver_lookahead_enabled,
        )
        return sub

    def _replicate_with_solver_lookahead(
        self,
        x: tuple,
        solver: Solver,
    ) -> RepResult:
        """Like :meth:`replicate` but uses solver-based future estimation.

        Simulates the current stage with decision *x*, then uses a solver
        at each subsequent stage to find (approximately) optimal decisions.
        The sub-problems use policy-based Monte Carlo lookahead (not
        recursive solver lookahead) to keep computational cost manageable.

        Args:
                x: Decision vector for the current stage.
                solver: Configured solver instance.

        Returns:
                :class:`RepResult` with objectives derived from the
                solver-based cost-to-go across all remaining stages.
        """
        # Save original state/stage so we can restore after the rollout.
        saved_state = deepcopy(self.current_state)
        saved_stage = self.current_stage

        self._trace_lookahead(
            "entered solver-based lookahead replication",
            entry_stage=self.current_stage,
            horizon=self.model.n_stages,
            solver=type(solver).__name__,
            solver_supported=getattr(solver, "supported_problem_type", None),
        )

        total_responses: dict[str, float] = {}
        state = self.current_state
        decision: tuple[float, ...] = x

        for stage in range(self.current_stage, self.model.n_stages):
            # 1. Simulate this stage with the current decision.
            self._trace_lookahead(
                "simulating stage",
                stage=stage,
                decision_dim=len(decision),
            )
            reward, next_state = self.model.replicate_stage(
                state, decision, stage, self.model._rng_list
            )

            # 2. Accumulate stage rewards.
            for key, val in reward.items():
                total_responses[key] = total_responses.get(key, 0.0) + val

            state = next_state

            # 3. Use the solver to find the optimal decision for the
            #    *next* stage (skip if we just simulated the last stage).
            if stage + 1 < self.model.n_stages:
                sub_problem = self._create_stage_subproblem(state, stage + 1)
                stage_solver = deepcopy(solver)
                stage_solver.recommended_solns = []
                stage_solver.intermediate_budgets = []
                # Configure the sub-solver via its Pydantic config
                # (solver.factors is a read-only property that returns
                # model_dump(), so direct assignment has no effect).

                if isinstance(stage_solver, SupportsInitialSolution):
                    solver_with_init = cast(SupportsInitialSolution, stage_solver)
                    solver_with_init.initial_solution = decision
                if hasattr(solver, "budget"):
                    stage_solver.budget.total = solver.budget.remaining

                # Solver-based lookahead may receive a solver instance that was
                # never run via the experiment harness, so its progenitor RNGs
                # can be empty. Ensure stage-level solves can always create
                # solutions with valid RNG lists.
                if not stage_solver.solution_progenitor_rngs:
                    stage_solver.solution_progenitor_rngs = [
                        MRG32k3a(s_ss_sss_index=[0, i, 0])
                        for i in range(sub_problem.model.n_rngs)
                    ]

                self._trace_lookahead(
                    "about to solve lookahead sub-problem",
                    sub_problem_id=id(sub_problem),
                    sub_problem_stage=sub_problem.current_stage,
                    sub_problem_depth=sub_problem._lookahead_trace_depth,
                    sub_problem_lookahead_enabled=sub_problem._solver_lookahead_enabled,
                    stage_solver=type(stage_solver).__name__,
                    stage_solver_supported=getattr(
                        stage_solver,
                        "supported_problem_type",
                        None,
                    ),
                )

                try:
                    stage_solver.solve(sub_problem)
                    self._trace_lookahead(
                        "completed lookahead sub-problem solve",
                        solved_stage=stage + 1,
                        recommended_count=len(stage_solver.recommended_solns),
                    )
                    if stage_solver.recommended_solns:
                        decision = stage_solver.recommended_solns[-1].x
                    else:
                        if sub_problem.model._rng_list:
                            fallback_rng = sub_problem.model._rng_list[0]
                        elif self.model._rng_list:
                            fallback_rng = self.model._rng_list[0]
                        else:
                            fallback_rng = stage_solver.solution_progenitor_rngs[0]
                        decision = sub_problem.get_random_solution(fallback_rng)
                except Exception as e:
                    self._trace_lookahead(
                        "sub-problem solve raised exception; using fallback",
                        failed_stage=stage + 1,
                        exception=e,
                    )
                    logger.debug(
                        "Solver lookahead stage %d: exception during "
                        "solve: %s. Falling back to random solution.",
                        stage + 1,
                        e,
                    )
                    if sub_problem.model._rng_list:
                        fallback_rng = sub_problem.model._rng_list[0]
                    elif self.model._rng_list:
                        fallback_rng = self.model._rng_list[0]
                    else:
                        fallback_rng = stage_solver.solution_progenitor_rngs[0]
                    decision = sub_problem.get_random_solution(fallback_rng)

        # Restore original state/stage (the rollout is purely evaluative).
        self.current_state = saved_state
        self.current_stage = saved_stage

        self._trace_lookahead("exiting solver-based lookahead replication")

        # Build RepResult from the accumulated responses.
        objectives = [
            Objective(stochastic=total_responses[key])
            for key in sorted(total_responses)
        ]
        return RepResult(objectives=objectives)

    # ------------------------------------------------------------------
    # Policy Construction
    # ------------------------------------------------------------------
    def build_policy(
        self,
        solver: Solver,
        best_stage0_decision: list | tuple,
    ) -> Callable[[dict, int], tuple]:
        """Build a callable policy ``(state_dict, stage) -> decision_tuple``.

        that uses the trained GP-V models for stages 1+ and the best found
        stage-0 decision for stage 0.
        """
        stage0_dec = tuple(float(v) for v in best_stage0_decision)
        return PolicyFunction(stage0_dec, solver, self)
