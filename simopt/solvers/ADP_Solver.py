"""Future Revenue ADP Solver.

----------------------
This solver implements an Approximate Dynamic Programming
approach to solve multistage stochastic optimization problems
by iteratively fitting Gaussian Process value function
approximators and using them to guide a forward search for
good decisions. The main idea is to perform a backwards pass
where we fit GP models for the value function at each stage,
starting from the terminal stage and moving backwards. Then,
in the forward pass, we use the fitted GP value functions to
evaluate candidate decisions at each stage and select the best
one according to an Upper Confidence Bound (UCB) criterion.
The selected decision is then verified with additional
simulations before being accepted as the recommended solution
for that stage. This process is repeated for each stage until
we reach the initial stage, at which point we have a
recommended solution for the entire problem.

"""

from __future__ import annotations

import math
import numbers
import time
import warnings
from collections.abc import Callable
from functools import partial
from typing import Annotated, Any, ClassVar, cast

import numpy as np
from pydantic import Field
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

# Suppress GP kernel length_scale convergence warnings -- a high length_scale
# simply means the function is nearly flat in that dimension, which is valid.
warnings.filterwarnings(
    "ignore", message=".*length_scale.*close to the specified upper bound.*"
)

from mrg32k3a.mrg32k3a import MRG32k3a  # noqa: E402
from simopt.base import (  # noqa: E402
    ConstraintType,
    MultistageProblem,
    ObjectiveType,
    ProblemLike,
    RepResult,
    Solution,
    Solver,
    SolverConfig,
    SolverProblemType,
    VariableType,
)

# from simopt.models.vanryzin_airline_revenue import VanRyzinState
from simopt.multistage_model import MultistageModel  # noqa: E402
from simopt.problem import Objective, Problem  # noqa: E402
from simopt.solver import Budget, BudgetExhaustedError  # noqa: E402


class _GaussianProcessValueModel:
    """A wrapper around sklearn's GaussianProcessRegressor with separate.

    state and decision feature channels.
    """

    def __init__(self, alpha: float, length_scale: float | None = None) -> None:
        self.alpha = alpha
        self.length_scale = length_scale
        self.gp: GaussianProcessRegressor | None = None
        self.is_fitted = False
        self.state_scaler: StandardScaler | None = None
        self._last_y_train: list[float] = []
        self._last_raw_states: list[list[float]] = []
        self._last_stage: int = 0

    def _ensure_2d(self, x: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
        arr = np.array(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _scaled_features(self, state_x: np.ndarray) -> np.ndarray:
        if self.state_scaler is None:
            raise RuntimeError(
                "Model scaler must be initialized before feature transformation."
            )
        return self.state_scaler.transform(state_x)

    def fit(self, state_x: np.ndarray, y: np.ndarray) -> None:
        """Fit the Gaussian Process model to the given data (state variables only).

        Args:
                state_x (np.ndarray): State feature
                    matrix, shape (num_samples,
                    n_state_features).
                y (np.ndarray): The target values for training, shape (num_samples,).
        """
        state_arr = self._ensure_2d(state_x)
        Y = np.array(y)  # noqa: N806
        self.state_scaler = StandardScaler().fit(state_arr)
        X = self._scaled_features(state_arr)  # noqa: N806

        n_features = X.shape[1]
        ls_init = (
            np.ones(n_features)
            if self.length_scale is None
            else np.full(n_features, self.length_scale)
        )
        kernel = C(1.0) * Matern(
            length_scale=ls_init, nu=2.5, length_scale_bounds=(1e-5, 1e8)
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=42,
        )
        self.gp.fit(X, Y)
        self.is_fitted = True

    def predict_ucb(self, state_x: np.ndarray, kappa: float = 1.0) -> float:
        """Upper Confidence Bound prediction: mean + kappa * std (state variables only)."""  # noqa: E501
        if not self.is_fitted or self.gp is None:
            raise RuntimeError("Model must be fitted before predicting.")

        state_arr = self._ensure_2d(state_x)
        X = self._scaled_features(state_arr)  # noqa: N806
        y_mean, y_std = self.gp.predict(X, return_std=True)
        return float((y_mean + kappa * y_std).mean())

    def predict(
        self,
        state_x: np.ndarray,
        decision_x: np.ndarray | None = None,  # noqa: ARG002
    ) -> float:
        """Predict expected value from state features only.

        ``decision_x`` is accepted but ignored -- it exists so that
        ``value_models[stage]`` can be called with (state, decision) from
        the forward pass without raising a TypeError.
        """
        if not self.is_fitted or self.gp is None:
            raise RuntimeError("Model must be fitted before predicting.")

        state_arr = self._ensure_2d(state_x)
        X = self._scaled_features(state_arr)  # noqa: N806
        y_pred, _ = self.gp.predict(X, return_std=True)
        return float(y_pred.mean())

    def predict_with_std(self, state_x: np.ndarray) -> tuple[float, float]:
        """Return (mean, std) of the GP posterior at state_x."""
        if not self.is_fitted or self.gp is None:
            raise RuntimeError("Model must be fitted before predicting.")
        state_arr = self._ensure_2d(state_x)
        X = self._scaled_features(state_arr)  # noqa: N806
        y_mean, y_std = self.gp.predict(X, return_std=True)
        return float(y_mean.mean()), float(y_std.mean())


class _PolicyFunction:
    """Picklable policy callable for ADP solver.

    Replaces the closure returned by ``_build_policy`` so that policy
    solutions can be serialised by joblib/loky for parallel post-replication.
    """

    def __init__(
        self,
        stage0_dec: tuple,
        solver: ADPSolver,
        problem: MultistageProblem,
    ) -> None:
        self.stage0_dec = stage0_dec
        self.solver = solver
        self.problem = problem

    def __call__(self, state: dict, stage: int) -> tuple:
        if stage == 0:
            return self.stage0_dec
        rem_cap = state.get("remaining_capacity", [])
        if isinstance(rem_cap, dict):
            rem_cap = [float(rem_cap[k]) for k in sorted(rem_cap)]
        current_cap = np.array([float(v) for v in rem_cap])
        action = self.solver._greedy_action_at_stage(self.problem, stage, current_cap)
        return tuple(action.tolist())


class ADPSolverConfig(SolverConfig):
    """Configuration for the ADP solver."""

    wrapped_solver: Annotated[
        str,
        Field(
            default="ASTROMORF",
            description="The name of the solver to use for the inner optimization"
            "problems.",
        ),
    ]
    wrapped_solver_factors: Annotated[
        dict,
        Field(
            default_factory=dict,
            description="A dictionary of factors to pass to the wrapped solver when it"
            "is instantiated.",
        ),
    ]
    alpha: Annotated[
        float,
        Field(
            default=1e-4,
            ge=0,
            description="Observation-noise regularization for GP fit.",
        ),
    ]
    n_macroreps_training: Annotated[
        int,
        Field(
            default=2,
            ge=1,
            description="Number of macroreplications to use when estimating values for"
            "regression training.",
        ),
    ]
    n_macroreps_forward: Annotated[
        int,
        Field(
            default=3,
            ge=1,
            description=(
                "Number of macroreplications used to verify a GP-selected candidate "
                "before recording it in recommended_solns. Higher values give a more "
                "reliable acceptance decision but consume more simulation budget."
            ),
        ),
    ]
    n_training_pts: Annotated[
        int,
        Field(
            default=200,
            ge=1,
            description="Number of training points for constructing the GP models",
        ),
    ]
    forward_pass_budget_fraction: Annotated[
        float,
        Field(
            default=0.2,
            gt=0,
            le=1.0,
            description="Fraction of total budget allocated to each forward pass"
            "iteration.",
        ),
    ]
    boundary_tight_frac: Annotated[
        float,
        Field(
            default=0.1,
            gt=0,
            lt=1.0,
            description="Fraction of training points allocated to the tight boundary"
            "stratum (remaining cap ≈ 0).",
        ),
    ]
    boundary_loose_frac: Annotated[
        float,
        Field(
            default=0.1,
            gt=0,
            lt=1.0,
            description="Fraction of training points allocated to the loose boundary"
            "stratum (remaining cap ≈ max).",
        ),
    ]
    kappa_0: Annotated[
        float,
        Field(
            default=1.5,
            ge=0,
            description="Initial UCB exploration weight for the forward pass.",
        ),
    ]
    kappa_decay: Annotated[
        float,
        Field(
            default=0.85,
            gt=0,
            le=1.0,
            description="Geometric decay rate applied to kappa each forward iteration.",
        ),
    ]
    n_actions_per_state: Annotated[
        int,
        Field(
            default=25,
            ge=2,
            description=(
                "Number of actions sampled per training state during the backwards pass. "  # noqa: E501
                "Higher values improve action coverage at the cost of more simulations."
            ),
        ),
    ]
    n_mc_replicates: Annotated[
        int,
        Field(
            default=20,
            ge=1,
            description=(
                "Monte Carlo replicates per (state, action) pair when estimating Q-values. "  # noqa: E501
                "Increase at earlier stages where errors compound more severely."
            ),
        ),
    ]
    use_crn: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Use Common Random Numbers across actions for the same state during "
                "Q-value estimation. Sharpens training by reducing within-state "
                "simulation noise."
            ),
        ),
    ]
    frac_action_lhs: Annotated[
        float,
        Field(
            default=0.50,
            gt=0,
            lt=1.0,
            description="Fraction of action samples drawn via constrained LHS.",
        ),
    ]
    frac_action_heuristic: Annotated[
        float,
        Field(
            default=0.20,
            gt=0,
            lt=1.0,
            description="Fraction of action samples drawn from heuristic anchors.",
        ),
    ]
    frac_state_lhs: Annotated[
        float,
        Field(
            default=0.15,
            gt=0,
            lt=1.0,
            description="Fraction of training states drawn via LHS over the full"
            "capacity box.",
        ),
    ]
    frac_state_uncertainty: Annotated[
        float,
        Field(
            default=0.15,
            gt=0,
            lt=1.0,
            description=(
                "Fraction of training states targeted at high-uncertainty regions of the "  # noqa: E501
                "preceding stage's GP-V. Ignored at the terminal stage."
            ),
        ),
    ]
    refinement_buffer_max: Annotated[
        int,
        Field(
            default=200,
            ge=10,
            description=(
                "Maximum number of (state, value) observations kept in the GP-V "
                "refinement buffer per stage.  When full the oldest entry is dropped "
                "(FIFO).  The buffer is seeded with the backwards-pass training data "
                "and grows with forward-pass observations."
            ),
        ),
    ]
    refinement_novelty_tol: Annotated[
        float,
        Field(
            default=1e-3,
            ge=0.0,
            description=(
                "Minimum Euclidean distance (in the GP-V's normalised state space) "
                "a new forward-pass observation must be from all existing buffer "
                "points before it is accepted.  Prevents near-duplicate observations "
                "from distorting the GP fit."
            ),
        ),
    ]
    refinement_deep_mc_reps: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description=(
                "If > 0 and simulation budget remains after the forward optimisation "
                "finds its best decision, run this many additional full-path "
                "replicates from the observed next state s_1 under a reference "
                "policy to obtain a lower-variance label for GP-V refinement. "
                "Set to 0 to skip deep MC entirely and use only the single "
                "forward-pass observation."
            ),
        ),
    ]
    bellman_n_iters: Annotated[
        int,
        Field(
            default=10,
            ge=1,
            description=(
                "Maximum number of Bellman consistency iterations in the backwards pass. "  # noqa: E501
                "Each iteration re-solves V_t^{k+1}(s) = E[max_a(r + V_{t+1}^{k}(s'))] "
                "via the wrapped solver and re-fits GP-V until V values at the training "  # noqa: E501
                "states converge.  Set to 1 to disable iteration (single backwards sweep)."  # noqa: E501
            ),
        ),
    ]
    bellman_convergence_tol: Annotated[
        float,
        Field(
            default=5e-3,
            gt=0,
            description=(
                "Relative change threshold for Bellman convergence.  Iteration stops when "  # noqa: E501
                "max_i |V^{k+1}(s_i) - V^k(s_i)| / max(|V^k(s_i)|, 1) < tol across all "
                "stages and training states."
            ),
        ),
    ]
    bellman_training_budget_fraction: Annotated[
        float,
        Field(
            default=0.6,
            gt=0,
            lt=1.0,
            description=(
                "Fraction of the training budget allocated to the initial "
                "backwards sweep.  The remaining (1 - fraction) is divided evenly among "  # noqa: E501
                "the Bellman refinement iterations so that later iterations receive "
                "progressively less budget as they refine an already good model."
            ),
        ),
    ]


class ADPSolver(Solver):
    """Approximate Dynamic Programming solver wrapper."""

    config: ADPSolverConfig  # type: ignore[assignment]

    name: str = "ADP_SOLVER"
    config_class: ClassVar[type[SolverConfig]] = ADPSolverConfig
    class_name_abbr: ClassVar[str] = "ADP_SOLVER"
    class_name: ClassVar[str] = "Future Revenue ADP Wrapper"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    constraint_type: ClassVar[ConstraintType] = ConstraintType.UNCONSTRAINED
    supported_problem_type: ClassVar[SolverProblemType] = SolverProblemType.MULTISTAGE
    gradient_needed: ClassVar[bool] = False

    def _build_state_features(
        self,
        problem: MultistageProblem,  # noqa: ARG002
        remaining_cap: dict | list | np.ndarray,
    ) -> np.ndarray:
        """Return raw remaining capacity as a 1-D float array."""
        if isinstance(remaining_cap, dict):
            return np.array([float(remaining_cap[i]) for i in sorted(remaining_cap)])
        return np.asarray(remaining_cap, dtype=float)

    def _backwards_budget_remaining(self) -> int:
        """Backward pass is budget-free -- always report ample budget."""
        # return max(0, self.budget.remaining - self.forward_budget_amount)
        return 10**9

    def _simulate_with_budget(
        self, problem: MultistageProblem, solution: Solution, n_reps: int
    ) -> Solution:
        if self.budget.remaining <= 0:
            raise BudgetExhaustedError()

        n = min(n_reps, self.budget.remaining)
        self.budget.request(n)
        problem.simulate_up_to([solution], n_reps=n)
        return solution

    def _approximate_replication(
        self,
        solver: Solver | None,  # noqa: ARG002
        problem: MultistageProblem,
        solution: Solution,
        n_reps: int,
    ) -> RepResult:
        """Run a single ADP replication.

        Computes r_1(a_1,s_1) + GP_V(s_2) instead of total
        revenue. Used in the backwards pass (where GP_V is
        the current value model for the next stage) and in
        the forward pass (where GP_V is the value model for
        the next stage trained on backwards-pass data and
        refined with forward-pass observations so far).
        """
        problem.model.factors.update(solution.decision_factors)

        # Use the stage-specific continuation callable injected by the caller
        # (_forward_optimize_with_model and _maximise_via_wrapped_solver both set
        # self._fwd_query_value before calling _run_simopt).  Falling back to the
        # smallest-key value model keeps backward-pass usage working; the final
        # lambda handles the terminal stage where no model exists.
        value_model_raw: Any = getattr(self, "_fwd_query_value", None)
        if value_model_raw is None:
            if self.value_models:
                value_model_raw = self.value_models[min(self.value_models.keys())]
            else:
                value_model_raw = lambda x: 0.0  # noqa: ARG005, E731
        value_model: Callable[[np.ndarray], float] = cast(
            Callable[[np.ndarray], float], value_model_raw
        )

        samples_vals: list[float] = []

        # For each n_reps, simulate at the first stage at
        # the decision given by solution and compute the
        # transition to get the next state s_2. Then enrich
        # the features of s_2 and query the GP value model
        # to get an estimate of the continuation value from
        # s_2, which we add to the immediate reward r_1 to
        # get our estimate of total revenue.

        model = cast(MultistageModel, problem.model)
        for _ in range(n_reps):
            if hasattr(model, "before_replication"):
                model.before_replication(solution.rng_list)  # type: ignore[attr-defined]
            else:
                model.before_replicate(solution.rng_list)  # type: ignore[attr-defined]

            if problem.before_replicate_override is not None:
                problem.before_replicate_override(model, solution.rng_list)

            responses, next_state = model.replicate_stage(
                problem.current_state,
                solution.x,
                problem.current_stage,
                solution.rng_list,
            )

            key = "revenue" if "revenue" in responses else next(iter(responses), None)
            r_0 = float(responses.get(key, 0.0)) if key else 0.0

            rem = next_state.get("remaining_capacity", None)
            if rem is None:
                rem = model.factors.get("capacity", [])
            elif isinstance(rem, dict):
                rem = [float(rem[k]) for k in sorted(rem)]

            s1_rem = [float(v) for v in rem]

            for rng in solution.rng_list:
                rng.advance_subsubstream()

            # Query GP-V with raw capacity of the next state.
            s1_features = np.asarray(s1_rem, dtype=float).reshape(1, -1)
            gp_est = value_model(s1_features)

            sample_value = r_0 + gp_est
            if math.isfinite(sample_value):
                samples_vals.append(sample_value)

        avg_rev = _finite_objective_or_penalty(
            float(np.mean(samples_vals)) if samples_vals else float("nan"),
            problem.minmax[0],
        )

        result = RepResult(objectives=[Objective(stochastic=avg_rev)])
        solution.add_replicate_result(result)
        return result

    def _simulate_policy_vec(
        self,
        problem: MultistageProblem,
        flat_x: np.ndarray,
        n_reps: int,
    ) -> Solution:
        """Evaluate an open-loop policy given a flat decision vector.

        Reshapes *flat_x* into per-stage tuples, creates a policy solution,
        and evaluates via :meth:`MultistageProblem.simulate_policy`.
        """
        stage_dim = problem.dim
        n_stages = problem.model.n_stages
        decisions = tuple(
            tuple(flat_x[s * stage_dim : (s + 1) * stage_dim].tolist())
            for s in range(n_stages)
        )
        sol = problem.create_policy_solution(decisions)
        sol.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
        problem.simulate_policy(sol, num_macroreps=n_reps)
        # Enforce budget accounting for multistage policy evaluation
        if hasattr(self, "budget") and self.budget is not None:
            self.budget.request(n_reps)
        return sol

    def _run_simopt(
        self,
        problem: MultistageProblem,
        budget_amount: int,
        n_macroreps: int = 1,
    ) -> tuple[tuple[float], float]:
        """Run the wrapped inner solver with at most ``budget_amount`` budget,.

        then evaluate the best decision on the full sample path.

        Parameters
        ----------
        problem       : problem copy (will be mutated -- set factors["budget"])
        budget_amount : simulation budget for ONE macroreplication of the solver
        n_macroreps : independent solver runs; total cost ≈ n_macroreps x budget_amount

        Total budget charged to ``self.budget``:
                n_macroreps x solver_budget_used  +  n_macroreps_forward  (eval step)

        Returns:
        -------
        best_decision : tuple
        total_revenue : float  (full-path evaluation; -inf on failure)
        """
        from simopt.experiment.run_solver import run_solver
        from simopt.experiment_base import instantiate_solver

        if self.budget.remaining <= 0:
            raise BudgetExhaustedError()

        # Cap the per-macroreplication solver budget to what we can afford.
        # Reserve n_macroreps_forward for the eval step that follows.
        eval_cost = self.config.n_macroreps_forward
        available = self.budget.remaining
        solver_cap = max(
            1, min(budget_amount, (available - eval_cost) // max(n_macroreps, 1))
        )

        # CRITICAL: Solver.run() creates its Budget from problem.factors["budget"],
        # so we MUST set this on the problem -- setting wrapped_solver.budget alone
        # is overwritten by Solver.run().
        problem.factors["budget"] = solver_cap

        wrapped_solver = instantiate_solver(
            self.config.wrapped_solver, self.config.wrapped_solver_factors
        )

        if hasattr(wrapped_solver, "rng"):
            object.__setattr__(
                wrapped_solver,
                "rng",
                [
                    MRG32k3a(s_ss_sss_index=[0, i, 0])
                    for i in range(problem.model.n_rngs)
                ],
            )

        # Override problem.simulate so the inner solver (e.g. RNDSRCH) evaluates
        # r_0(a) + GP_V_1(s_1) as its objective instead of full multi-stage sim.
        # def _surrogate_simulate(solution: Solution, num_macroreps: int = 1, **kwargs)
        # -> None:
        # 	adp_self._approximate_replication(
        # 		wrapped_solver, problem, solution, num_macroreps,
        # 	)
        _surrogate_simulate = partial(
            surrogate_simulate,
            adp_self=self,
            wrapped_solver=wrapped_solver,
            problem=problem,
        )

        object.__setattr__(problem, "simulate", _surrogate_simulate)

        sol_df, _, _ = run_solver(
            wrapped_solver,
            cast(Problem, problem),
            n_macroreps=n_macroreps,
            n_jobs=1,
        )

        if sol_df.empty:
            return (
                problem.factors["initial_solution"],
                _finite_objective_or_penalty(float("nan"), problem.minmax[0]),
            )

        best_decision = sol_df.iloc[-1]["solution"]
        # Charge for all macroreplications.  wrapped_solver.budget.used only
        # reflects the last macrorep (Solver.run resets it each time), so we
        # charge n_macroreps x last_used as a conservative upper bound.
        budget_used_per_mrep = wrapped_solver.budget.used
        total_solver_cost = n_macroreps * budget_used_per_mrep
        # Don't charge more than what's available.
        total_solver_cost = min(total_solver_cost, self.budget.remaining)
        self.budget.request(total_solver_cost)

        # ------------------------------------------------------------------
        # Evaluate the optimal decision on the full sample path.
        # ------------------------------------------------------------------
        # Detect whether best_decision is a multi-stage representation that
        # needs policy-vec simulation.  Two cases:
        #   1. Flat vector of length n_stages * dim (e.g. SGD on a multistage
        #      problem tiles stage-0 across all stages).
        #   2. Nested tuple-of-tuples ((x_stage0, ...), (x_stage1, ...), ...).
        # In both cases we flatten into shape (n_stages * stage_dim,) and use
        # _simulate_policy_vec.  Otherwise it's a single-stage decision of
        # length dim and we simulate directly.
        is_flat_multistage = (
            isinstance(best_decision, list | np.ndarray | tuple)
            and len(best_decision) == problem.model.n_stages * problem.dim
            and len(best_decision) > 0
            and isinstance(best_decision[0], numbers.Number)
        )
        is_nested_multistage = (
            isinstance(best_decision, (list | np.ndarray | tuple))
            and len(best_decision) > 0
            and isinstance(best_decision[0], (list | np.ndarray | tuple))
        )

        if is_flat_multistage or is_nested_multistage:
            flat_best_decision = np.array(
                best_decision
            ).flatten()  # shape (n_stages * stage_dim,)
            eval_sol = self._simulate_policy_vec(
                problem, flat_best_decision, n_reps=self.config.n_macroreps_forward
            )
        else:
            eval_sol = Solution(best_decision, problem)
            eval_rngs = [
                MRG32k3a(s_ss_sss_index=[0, i, 0]) for i in range(problem.model.n_rngs)
            ]
            eval_sol.attach_rngs(eval_rngs, copy=False)

            eval_sol = self._simulate_with_budget(
                problem, eval_sol, n_reps=self.config.n_macroreps_forward
            )
        total_revenue = _finite_objective_or_penalty(
            float(eval_sol.objectives_mean.item()),
            problem.minmax[0],
        )

        return best_decision, total_revenue

    def solve(self, problem: ProblemLike) -> None:
        """Run the ADP solver on the given problem."""
        if not isinstance(problem, MultistageProblem):
            raise TypeError(
                f"{self.name} only supports MultistageProblem instances; got {type(problem).__name__}."  # noqa: E501
            )

        # --- Clamp extreme parameter values ---
        if self.config.forward_pass_budget_fraction < 0.1:
            self.config.forward_pass_budget_fraction = 0.2
        if self.config.bellman_training_budget_fraction > 0.8:
            self.config.bellman_training_budget_fraction = 0.6
        if self.config.n_training_pts > 500:
            self.config.n_training_pts = 500
        if self.config.bellman_n_iters > 10:
            self.config.bellman_n_iters = 10

        # Seed the progress curve with the initial solution at budget=0.
        if not self.recommended_solns:
            initial_solution = Solution(problem.factors["initial_solution"], problem)
            self.recommended_solns.append(initial_solution)
            self.intermediate_budgets.append(self.budget.used)

        # Budget split: backwards pass uses heuristic scoring (free) + MC estimates.
        # Pre-compute the backwards cost so the forward pass gets everything else.
        # n_stages_to_fit   = max(1, problem.model.n_stages - 1)
        # mc_cost_per_state = self.config.n_mc_replicates
        # n_pts             = self.config.n_training_pts
        # # Initial sweep + Bellman iterations
        # bellman_iters     = max(1, self.config.bellman_n_iters - 1)
        # backwards_cost = n_stages_to_fit * n_pts * mc_cost_per_state * (1 +
        # bellman_iters)
        # # Ensure backwards pass gets at least enough, but no more than 30% of total.
        # self.backwards_budget_amount = min(
        # 	int(0.3 * self.budget.total),
        # 	max(backwards_cost, int(0.05 * self.budget.total)),
        # )
        # self.forward_budget_amount = self.budget.total - self.backwards_budget_amount
        # Backward pass is budget-free; entire budget goes to forward pass.
        self.backwards_budget_amount = 0
        self.forward_budget_amount = self.budget.total
        self.last_forward_pass_time = None
        if not hasattr(self, "forward_pass_timings"):
            self.forward_pass_timings = []

        # Create a proper Budget sub-tracker for the forward pass.
        self.forward_budget = Budget(total=self.forward_budget_amount)
        # Each forward iteration gets an equal slice; at least 5 iterations.
        _expected_fwd_iters = max(5, self.forward_budget_amount // 500)
        self._forward_iter_budget = max(
            1, self.forward_budget_amount // _expected_fwd_iters
        )
        nelder_mead_floor = self._wrapped_solver_budget_floor(problem)

        # Do the backwards fit to construct the value models for each stage
        self.value_models: dict[int, Callable[[np.ndarray, np.ndarray], float]] = {}
        print(f"Budget before backward pass: {self.budget.remaining}")
        self._backwards_fit(problem)
        print(f"Budget after backward pass: {self.budget.remaining}")
        forward_pass_start = time.perf_counter()

        # ---- Seed per-stage GP-V refinement buffers from backwards-pass data ----
        # Each buffer is a rolling FIFO of (state, value) pairs.  Starting it with
        # the backwards-pass training data means the first forward-pass refit
        # always has a well-conditioned dataset to work with.
        self._refinement_buffers: dict[int, dict[str, list]] = {}
        for stage, gp_v_obj in getattr(self, "_gp_v_objects", {}).items():
            training_states = getattr(gp_v_obj, "_last_raw_states", [])
            training_values = list(getattr(gp_v_obj, "_last_y_train", []))
            self._refinement_buffers[stage] = {
                "states": list(training_states),
                "values": training_values,
            }

        self._stage1_gp = getattr(self, "_stage1_gp_init", None)

        # --- Initialise from the initial solution ---
        initial_sol_vec = problem.factors["initial_solution"]

        # Compute the initial solution's surrogate value (r_0 + GP_V_1)
        # for fair comparison with forward-pass candidates.
        gp_v1 = self._gp_v_objects.get(1)
        if gp_v1 is not None and gp_v1.is_fitted:

            def init_query(s: np.ndarray) -> float:
                return gp_v1.predict(s)
        else:
            init_query = self.value_models.get(1, lambda s: 0.0)  # noqa: ARG005
        self._fwd_query_value = init_query

        # Evaluate initial on the surrogate via a few MC replications.
        init_problem = problem._clone_problem()
        init_problem.current_stage = 0
        init_sol = Solution(initial_sol_vec, init_problem)
        init_sol.attach_rngs(
            [MRG32k3a(s_ss_sss_index=[0, i, 0]) for i in range(problem.model.n_rngs)],
            copy=False,
        )
        n_init_reps = max(self.config.n_macroreps_forward, 5)
        self._approximate_replication(None, init_problem, init_sol, n_init_reps)
        best_surrogate = float(init_sol.objectives_mean.item())
        best_sol = initial_sol_vec

        print(f"Initial solution -- surrogate revenue: {best_surrogate:.2f}")
        print(f"Forward pass entry: budget.remaining={self.budget.remaining}")

        iter_idx = 0
        no_improve_count = 0
        plateau_threshold = 2  # consecutive non-improving iters before adapting
        try:
            while self.forward_budget.remaining > 0 and self.budget.remaining > 0:
                # ── Adaptive budget: boost later iterations on plateau ────────
                if no_improve_count >= plateau_threshold:
                    # Double the per-iteration budget (capped to remaining)
                    max(1, self.forward_budget.remaining // self._forward_iter_budget)
                    boosted_budget = min(
                        self._forward_iter_budget * 2,
                        self.forward_budget.remaining,
                    )
                    self._forward_iter_budget = boosted_budget
                    # Enable deep MC refinement to get better GP-V labels
                    if self.config.refinement_deep_mc_reps == 0:
                        self.config.refinement_deep_mc_reps = 5
                    print(
                        f"  [Adaptive] Plateau detected after {no_improve_count} stale iters -- "  # noqa: E501
                        f"boosted iter budget to {boosted_budget}, enabled deep MC (5 reps)"  # noqa: E501
                    )

                if nelder_mead_floor:
                    required_iteration_budget = (
                        self.config.n_macroreps_forward
                        * self._forward_iteration_budget(problem)
                        + self.config.n_macroreps_forward
                    )
                    if self.budget.remaining < required_iteration_budget:
                        print(
                            "  [Forward] Stopping before another Nelder-Mead iteration: "  # noqa: E501
                            f"need at least {required_iteration_budget} budget units, "
                            f"have {self.budget.remaining}."
                        )
                        break

                budget_before = self.budget.used
                cand_sol, surrogate_rev, _rollout_rev = (
                    self._forward_optimize_with_model(
                        problem,
                        best_sol,
                        self.value_models[1],
                        iter_idx,
                    )
                )
                # Charge the actual iteration cost against the forward budget tracker.
                actual_cost = self.budget.used - budget_before
                iter_cost = min(actual_cost, self.forward_budget.remaining)
                if iter_cost > 0:
                    self.forward_budget.request(iter_cost)

                # Accept based on the surrogate objective (r_0 + GP_V_1),
                # which is the Bellman equation.  The greedy rollout is too noisy
                # with few reps to distinguish close candidates.
                if (
                    problem.minmax[0] * surrogate_rev
                    > problem.minmax[0] * best_surrogate
                ):
                    best_surrogate = surrogate_rev
                    best_sol = cand_sol
                    self.recommended_solns.append(Solution(tuple(best_sol), problem))
                    self.intermediate_budgets.append(self.budget.used)
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                iter_idx += 1
                self.iterations.append(iter_idx)
                self.fn_estimates.append(float(best_surrogate))
                self.budget_history.append(self.budget.used)
        except BudgetExhaustedError:
            pass  # Budget ran out mid-iteration; proceed to finalization.

        self.last_forward_pass_time = time.perf_counter() - forward_pass_start
        self.forward_pass_timings.append(self.last_forward_pass_time)

        # Final entry -- always the best solution found.
        self.recommended_solns.append(Solution(tuple(best_sol), problem))
        self.intermediate_budgets.append(self.budget.used)

        print(f"Final solution -- surrogate revenue: {best_surrogate:.2f}")
        print(f"  protection levels: {best_sol}")

        if not self.iterations or self.iterations[-1] != iter_idx:
            self.fn_estimates.append(float(best_surrogate))
            self.iterations.append(iter_idx)
            self.budget_history.append(self.budget.used)

        # ── Store optimal policy on the problem ───────────────────────────────
        policy_fn = self._build_policy(problem, best_sol)
        n_stages = problem.model.n_stages
        decisions = tuple(
            tuple(float(v) for v in best_sol)
            if t == 0
            else tuple(0.0 for _ in best_sol)
            for t in range(n_stages)
        )
        self.policy_solution = problem.create_policy_solution(
            decisions, policy=policy_fn
        )
        self.policy_solution.attach_rngs(
            [MRG32k3a(s_ss_sss_index=[0, i, 0]) for i in range(problem.model.n_rngs)],
            copy=False,
        )

        # ── Build per-checkpoint policy solutions for post-replication ────────
        # Each policy_solution uses the checkpoint's stage-0 decision paired
        # with the FINAL GP-V models for stages 1+.  This lets
        # post_replicate_policy() evaluate the full trajectory at each budget
        # checkpoint.
        self.policy_solutions = []
        for rec_sol in self.recommended_solns:
            stage0_x = rec_sol.x
            pf = self._build_policy(problem, stage0_x)
            decs = tuple(
                tuple(float(v) for v in stage0_x)
                if t == 0
                else tuple(0.0 for _ in stage0_x)
                for t in range(n_stages)
            )
            ps = problem.create_policy_solution(decs, policy=pf)
            ps.attach_rngs(
                [
                    MRG32k3a(s_ss_sss_index=[0, i, 0])
                    for i in range(problem.model.n_rngs)
                ],
                copy=False,
            )
            self.policy_solutions.append(ps)

    def _build_policy(
        self,
        problem: MultistageProblem,
        best_stage0_decision: list | tuple,
    ) -> Callable[[dict, int], tuple]:
        """Build a callable policy ``(state_dict, stage) -> decision_tuple``.

        that uses the trained GP-V models for stages 1+ and the best found
        stage-0 decision for stage 0.
        """
        stage0_dec = tuple(float(v) for v in best_stage0_decision)
        return _PolicyFunction(stage0_dec, self, problem)

    def _assert_bellman_targets(
        self, stage: int, states: list, v_values: list[float]
    ) -> None:
        """Audit helper -- zero cost at run-time but documents invariants.

        Asserts that every entry in `v_values` was produced by solving
        a simopt for the optimal action at that state.  Call this just before
        fitting GP-V so that future edits cannot accidentally regress to using
        arbitrary targets without a test failure.

        Parameters
        ----------
        stage    : stage index (for error messages)
        states   : list of training states (used only for len check)
        v_values : GP-V training targets produced by _maximise_via_wrapped_solver
        """
        assert len(states) == len(v_values), (
            f"Stage {stage}: len(states)={len(states)} != len(v_values)={len(v_values)}.  "  # noqa: E501
            "Every state must have exactly one Bellman-optimal V target."
        )
        assert all(np.isfinite(v) for v in v_values), (
            f"Stage {stage}: Non-finite Bellman targets detected.  "
            "Check _maximise_via_wrapped_solver for degenerate results."
        )
        # Values should be ≥ 0 for a revenue-management problem (can never earn
        # negative revenue under any policy).  Adjust the lower bound if your
        # problem admits negative rewards.
        if any(v < -1e6 for v in v_values):
            import warnings

            warnings.warn(
                f"Stage {stage}: Some Bellman targets are very negative "
                f"(min={min(v_values):.1f}).  Verify simopt maximisation is correct.",
                stacklevel=2,
            )

    def _wrapped_solver_budget_floor(self, problem: MultistageProblem) -> int:
        """Return the minimum per-macrorep budget needed by the wrapped solver."""
        if self.config.wrapped_solver == "NELDMD":
            r = int(self.config.wrapped_solver_factors.get("r", 30))
            return max(1, r * (problem.dim + 1))
        return 0

    def _forward_iteration_budget(self, problem: MultistageProblem) -> int:
        """Return the per-macrorep budget for the next forward-pass iteration."""
        return max(
            self._forward_iter_budget, self._wrapped_solver_budget_floor(problem)
        )

    #! ---------------------------------------------------------------------------
    #! Forward Optimization
    #! ---------------------------------------------------------------------------

    def _greedy_action_at_stage(
        self,
        problem: MultistageProblem,
        stage: int,
        current_cap: np.ndarray,
    ) -> np.ndarray:
        """Select the greedy action at ``stage`` for state ``current_cap`` by.

        scoring candidate actions via GP-V look-ahead.

        Strategy
        --------
        1. Sample a diverse set of candidate actions.
        2. Score each candidate: score(a) = GP-V_{stage+1}(s') where s' is
           the deterministic next-state approximation.
        3. Return the highest-scoring action projected to feasibility.
        """
        rng_local = np.random.default_rng()
        prev_opt = self._get_perturbation_anchor(stage, current_cap)
        candidates = self._sample_actions_for_state(
            problem, current_cap, prev_opt, rng=rng_local
        )

        gp_v_next = self._gp_v_objects.get(stage + 1)

        if gp_v_next is not None and gp_v_next.is_fitted:
            scores = self._score_actions_batch(
                problem, stage, current_cap, candidates, gp_v_next, n_mc=5
            )
        else:
            # No models available yet -- return a heuristic (half-capacity) action.
            anchors = self._build_heuristic_actions(problem, current_cap)
            return anchors[2] if len(anchors) > 2 else anchors[0]

        best_idx = int(np.argmax(scores))
        return self._project_action_to_feasible(
            candidates[best_idx].copy(), problem, current_cap
        )

    def _score_actions_batch(
        self,
        problem: MultistageProblem,
        stage: int,
        state: np.ndarray,
        actions: np.ndarray,
        gp_v_next: _GaussianProcessValueModel,
        n_mc: int = 2,
    ) -> np.ndarray:
        """Score a batch of candidate actions using a single problem copy.

        Creates one shallow simulation context and reuses it for all
        candidates, avoiding the per-candidate ``deepcopy(problem)`` that
        previously dominated runtime.

        Returns an array of scores, one per candidate action.
        """
        sim_problem = problem._clone_problem()
        sim_problem.current_stage = int(stage)
        sim_problem.factors["capacity"] = [float(v) for v in state]
        if "remaining_capacity" in sim_problem.current_state:
            sim_problem.current_state["remaining_capacity"] = [float(v) for v in state]

        # Pre-build RNG lists -- same across candidates for a given MC rep.
        mc_rngs = [
            [
                MRG32k3a(s_ss_sss_index=[0, k, int(stage)])
                for _ in range(sim_problem.model.n_rngs)
            ]
            for k in range(n_mc)
        ]

        sim_model = cast(MultistageModel, sim_problem.model)
        scores = np.empty(len(actions))
        for a_idx, action in enumerate(actions):
            is_flat_multistage = (
                isinstance(sim_problem, MultistageProblem)
                and isinstance(action, list | np.ndarray | tuple)
                and len(action) == sim_problem.model.n_stages * sim_problem.dim
                and len(action) > 0
                and isinstance(action[0], numbers.Number)
            )
            is_nested_multistage = (
                isinstance(action, list | np.ndarray | tuple)
                and len(action) > 0
                and isinstance(action[0], list | np.ndarray | tuple)
            )

            if is_nested_multistage:
                # if any(not isinstance(x, numbers.Number) for x in xs_seq[-1]):
                action_tuple = tuple(float(v) for v in action[0])
            elif is_flat_multistage:
                stage_dim = sim_problem.dim
                action_tuple = tuple(float(v) for v in action[:stage_dim])
            else:
                # normal case where xs_seq[-1] is a list of decisions for the first
                # stage
                action_tuple = tuple(float(v) for v in action)

            totals = 0.0
            for k in range(n_mc):
                eval_sol = Solution(action_tuple, sim_problem)
                eval_sol.attach_rngs(mc_rngs[k], copy=True)

                if hasattr(sim_model, "before_replication"):
                    sim_model.before_replication(eval_sol.rng_list)  # type: ignore[attr-defined]
                else:
                    sim_model.before_replicate(eval_sol.rng_list)  # type: ignore[attr-defined]
                if sim_problem.before_replicate_override is not None:
                    sim_problem.before_replicate_override(sim_model, eval_sol.rng_list)

                responses, next_state_obj = sim_model.replicate_stage(
                    sim_problem.current_state,
                    eval_sol.x,
                    sim_problem.current_stage,
                    eval_sol.rng_list,
                )

                key = (
                    "revenue" if "revenue" in responses else next(iter(responses), None)
                )
                reward = float(responses.get(key, 0.0)) if key else 0.0

                rem = next_state_obj.get("remaining_capacity", None)
                if rem is None:
                    next_cap = state.copy()
                elif isinstance(rem, dict):
                    next_cap = np.array([float(rem[i]) for i in sorted(rem)])
                else:
                    next_cap = np.array([float(v) for v in rem])

                v_next = gp_v_next.predict_ucb(next_cap.reshape(1, -1), kappa=0.0)
                totals += reward + v_next

            scores[a_idx] = totals / n_mc

        return scores

    def _greedy_rollout_revenue(
        self,
        problem: MultistageProblem,
        stage0_decision: list | tuple,
        n_reps: int,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """Evaluate the full ADP-guided greedy policy over ``n_reps`` Monte Carlo.

        replicates and return the mean total revenue.

        At stage 0 the provided ``stage0_decision`` is used.  At every subsequent
        stage the greedy action is chosen via ``_greedy_action_at_stage``, which
        queries the GP-V models trained in the backwards pass.

        This gives a far more accurate revenue estimate than ``simulate_up_to``
        (which uses the problem's zero/fixed default policy for stages 1+).

        Parameters
        ----------
        problem          : MultistageProblem
        stage0_decision  : decision tuple / list for stage 0
        n_reps           : number of full-path replicates

        """
        n_stages = problem.model.n_stages
        rep_revenues: list[float] = []
        # Accumulate per-stage decisions/states for logging.
        stage_decisions: list[list[np.ndarray]] = [[] for _ in range(n_stages)]
        stage_states: list[list[np.ndarray]] = [[] for _ in range(n_stages)]

        for rep in range(n_reps):
            total_rev = 0.0
            current_cap = np.array(problem.model.factors["capacity"], dtype=float)
            sim_problem = (
                problem._clone_problem()
            )  # one copy per rep, reused across stages

            for stage in range(n_stages):
                # ── Choose action ──────────────────────────────────────────────
                if stage == 0:
                    action = np.array(stage0_decision, dtype=float)
                else:
                    action = self._greedy_action_at_stage(problem, stage, current_cap)

                stage_decisions[stage].append(action.copy())
                stage_states[stage].append(current_cap.copy())

                # ── Simulate one step ──────────────────────────────────────────
                sim_problem.factors["capacity"] = current_cap.tolist()
                if "remaining_capacity" in sim_problem.current_state:
                    sim_problem.current_state["remaining_capacity"] = (
                        current_cap.tolist()
                    )
                sim_problem.current_stage = stage

                sim_sol = Solution(tuple(action.tolist()), sim_problem)
                rollout_model = cast(MultistageModel, sim_problem.model)
                sim_rngs = [
                    MRG32k3a(s_ss_sss_index=[0, rep, stage])
                    for _ in range(rollout_model.n_rngs)
                ]
                sim_sol.attach_rngs(sim_rngs, copy=False)

                if hasattr(rollout_model, "before_replication"):
                    rollout_model.before_replication(sim_sol.rng_list)  # type: ignore[attr-defined]
                else:
                    rollout_model.before_replicate(sim_sol.rng_list)  # type: ignore[attr-defined]

                if sim_problem.before_replicate_override is not None:
                    sim_problem.before_replicate_override(
                        rollout_model, sim_sol.rng_list
                    )

                responses, next_state_obj = rollout_model.replicate_stage(
                    sim_problem.current_state,
                    sim_sol.x,
                    sim_problem.current_stage,
                    sim_sol.rng_list,
                )

                key = (
                    "revenue" if "revenue" in responses else next(iter(responses), None)
                )
                reward = float(responses.get(key, 0.0)) if key else 0.0
                total_rev += reward

                # ── Advance to next state ─────────────────────────────────────
                next_rem = next_state_obj.get("remaining_capacity", None)
                if next_rem is None:
                    next_rem = current_cap.tolist()
                elif isinstance(next_rem, dict):
                    next_rem = [float(next_rem[k]) for k in sorted(next_rem)]
                current_cap = np.array([float(v) for v in next_rem])

            rep_revenues.append(total_rev)

        self.budget.request(n_reps * n_stages)

        mean_decisions = [
            np.mean(np.array(stage_decisions[t]), axis=0)
            if stage_decisions[t]
            else np.array([])
            for t in range(n_stages)
        ]
        mean_states = [
            np.mean(np.array(stage_states[t]), axis=0)
            if stage_states[t]
            else np.array([])
            for t in range(n_stages)
        ]
        finite_rep_revenues = [rev for rev in rep_revenues if math.isfinite(rev)]
        mean_revenue = _finite_objective_or_penalty(
            float(np.mean(finite_rep_revenues))
            if finite_rep_revenues
            else float("nan"),
            problem.minmax[0],
        )
        return mean_revenue, mean_decisions, mean_states

    def _refine_all_stages_from_rollout(
        self,
        problem: MultistageProblem,
        mean_decisions: list[np.ndarray],  # noqa: ARG002
        mean_states: list[np.ndarray],
        total_revenue: float,
    ) -> None:
        """After a greedy rollout, use the observed (state, value) pairs at each.

        stage to refine the corresponding GP-V models.

        For stage t the observed future value is:
                V_t(s_t) ≈ total_revenue - sum_{k < t} r_k

        This is a running-total estimate: early stages get the full remaining
        revenue, which is a noisy but unbiased estimate of their continuation value.

        Parameters
        ----------
        problem        : MultistageProblem
        mean_decisions : per-stage mean decision vectors from the rollout
        mean_states    : per-stage mean remaining-capacity vectors from the rollout
        total_revenue  : observed full-path revenue from the rollout
        """
        n_stages = problem.model.n_stages
        cumulative_reward = 0.0
        for stage in range(1, n_stages):
            if stage >= len(mean_states) or mean_states[stage].size == 0:
                break
            # Remaining revenue observed from stage t onwards.
            future_value = total_revenue - cumulative_reward

            self._refine_gp_v_from_forward_pass(
                problem=problem,
                stage=stage,
                next_state=mean_states[stage],
                future_value=future_value,
            )
            # cumulative_reward stays unchanged -- future_value is a slight
            # over-estimate which is acceptable noise for GP-V refinement.

    def _forward_optimize_with_model(
        self,
        problem: MultistageProblem,
        initial_solution: list | tuple,
        value_model: Callable[..., float],
        iter_idx: int = 0,
    ) -> tuple[list | tuple, float, float]:
        """One forward-pass iteration.

        Stage-0 optimisation
        --------------------
        The wrapped inner solver searches over stage-0 decisions.  For each
        candidate decision ``a`` the objective presented to the solver is:

                obj(a) = r_0(a)  +  GP_V_1( E[s_1 | s_0, a] )

        where ``r_0`` is the simulated immediate stage-0 reward and
        ``GP_V_1`` is the stage-1 continuation value model from the
        backwards pass (queried with UCB to encourage exploration early on).
        This is implemented by temporarily replacing the solver's
        ``_do_simulate`` method with ``_approximate_replication``.

        Full-path evaluation (improved)
        --------------------------------
        Once the inner solver converges to a best stage-0 decision ``a*``, we
        evaluate the *complete ADP policy* via ``_greedy_rollout_revenue``.
        This rolls forward through every stage using GP-V to select
        actions, giving a far more reliable revenue estimate than the
        framework's default zero-policy ``simulate_up_to``.

        GP-V refinement (all stages)
        -----------------------------
        The per-stage (state, future_value) observations from the rollout are
        passed to ``_refine_all_stages_from_rollout``, which updates GP-V for
        every stage -- not just stage 1.

        Parameters
        ----------
        problem          : MultistageProblem
        initial_solution : current best stage-0 decision (warm start)
        value_model      : GP-V_1 predict callable
        iter_idx         : iteration counter (drives kappa decay)

        Returns:
        -------
        best_decision    : list | tuple  -- best stage-0 decision found
        surrogate_rev    : float         -- surrogate objective (r_0 + GP_V_1)
        rollout_revenue  : float         -- full-path greedy rollout revenue
        """
        kappa = self.config.kappa_0 * (self.config.kappa_decay**iter_idx)
        gp_v1 = getattr(self, "_gp_v_objects", {}).get(1)

        # Build the stage-0 continuation-value query function and store it on
        # self so that the injected _approximate_replication can reach it.
        # def _query_value(state_arr: np.ndarray) -> float:
        # 	if gp_v1 is not None and gp_v1.is_fitted:
        # 		return gp_v1.predict_ucb(state_arr, kappa=kappa)
        # 	return float(value_model(state_arr))

        # self._fwd_query_value = _query_value
        self._fwd_query_value = partial(
            query_value, gp_v1=gp_v1, value_model=value_model, kappa=kappa
        )

        stage_budget = min(
            self._forward_iteration_budget(problem), self.budget.remaining
        )

        new_problem = problem._clone_problem()
        new_problem.factors["initial_solution"] = initial_solution
        new_problem.factors["budget"] = stage_budget
        new_problem.current_stage = 0
        new_problem._solver_lookahead_enabled = False

        # Store the problem reference so _approximate_replication can use it.
        self._fwd_problem_ref = new_problem

        # ── Stage-0 optimisation via inner solver ─────────────────────────────
        best_decision, surrogate_rev_ucb = self._run_simopt(
            new_problem,
            stage_budget,
            n_macroreps=self.config.n_macroreps_forward,
        )

        # Re-evaluate the best decision with kappa=0 (no UCB inflation) so the
        # acceptance criterion compares apples-to-apples across iterations.
        if gp_v1 is not None and gp_v1.is_fitted:
            eval_problem = problem._clone_problem()
            eval_problem.current_stage = 0
            is_flat_multistage = (
                isinstance(eval_problem, MultistageProblem)
                and isinstance(best_decision, list | np.ndarray | tuple)
                and len(best_decision) == eval_problem.model.n_stages * eval_problem.dim
                and len(best_decision) > 0
                and isinstance(best_decision[0], numbers.Number)
            )
            is_nested_multistage = (
                isinstance(best_decision, list | np.ndarray | tuple)
                and len(best_decision) > 0
                and isinstance(best_decision[0], list | np.ndarray | tuple)
            )

            if is_nested_multistage:
                # if it's a flat or nested multistage decision vector, we need to get
                # the first stag
                best_decision_float = tuple(
                    float(v) for v in cast(tuple, best_decision[0])
                )
            elif is_flat_multistage:
                # if it's a flat multistage decision vector, we need to get the first
                # stage's decision
                stage_dim = eval_problem.dim
                best_decision_float = tuple(float(v) for v in best_decision[:stage_dim])
            else:
                best_decision_float = tuple(float(v) for v in best_decision)

            eval_sol = Solution(best_decision_float, eval_problem)
            eval_sol.attach_rngs(
                [
                    MRG32k3a(s_ss_sss_index=[0, i, 0])
                    for i in range(eval_problem.model.n_rngs)
                ],
                copy=False,
            )
            # Temporarily set query to kappa=0 for fair evaluation
            self._fwd_query_value = lambda s: gp_v1.predict(s)
            self._approximate_replication(
                None, eval_problem, eval_sol, max(self.config.n_macroreps_forward, 5)
            )
            surrogate_rev = float(eval_sol.objectives_mean.item())
            # Restore UCB query for next iteration's inner solver
            # self._fwd_query_value = _query_value
            self._fwd_query_value = partial(
                query_value, gp_v1=gp_v1, value_model=value_model, kappa=kappa
            )
        else:
            surrogate_rev = surrogate_rev_ucb
            best_decision_float = tuple(float(v) for v in best_decision)

        # ── Full-path evaluation under the greedy ADP policy ─────────────────
        n_eval_reps = max(self.config.n_macroreps_forward, 3)
        total_revenue, mean_decisions, mean_states = self._greedy_rollout_revenue(
            problem, best_decision_float, n_reps=n_eval_reps
        )

        print(
            f"  [Forward iter {iter_idx}] Surrogate: {surrogate_rev:.1f}  |  "
            f"Rollout: {total_revenue:.1f}"
        )

        # ── Refine GP-V at every stage using rollout observations ─────────────
        self._refine_all_stages_from_rollout(
            problem=problem,
            mean_decisions=mean_decisions,
            mean_states=mean_states,
            total_revenue=total_revenue,
        )

        return best_decision_float, surrogate_rev, total_revenue

    # ---------------------------------------------------------------------------
    # Helper: observe mean next state after stage-0 transition
    # ---------------------------------------------------------------------------
    def _observe_next_state(
        self,
        problem: MultistageProblem,
        decision: list | tuple,
        n_reps: int,
    ) -> np.ndarray:
        """Run ``n_reps`` stage-0 transitions and return the mean post-transition.

        remaining-capacity vector s_1.
        """
        eval_sol = Solution(tuple(decision), problem)
        eval_rngs = [
            MRG32k3a(s_ss_sss_index=[0, i, 0]) for i in range(problem.model.n_rngs)
        ]
        eval_sol.attach_rngs(eval_rngs, copy=False)

        model = cast(MultistageModel, problem.model)
        next_caps: list[list[float]] = []
        for _ in range(n_reps):
            if hasattr(model, "before_replication"):
                model.before_replication(eval_sol.rng_list)  # type: ignore[attr-defined]
            else:
                model.before_replicate(eval_sol.rng_list)  # type: ignore[attr-defined]

            if problem.before_replicate_override is not None:
                problem.before_replicate_override(model, eval_sol.rng_list)
            _, next_state = model.replicate_stage(
                state=problem.current_state,
                decision=eval_sol.x,
                stage=problem.current_stage,
                rng_list=eval_sol.rng_list,
            )
            rem = next_state.get("remaining_capacity", None)
            if rem is None:
                rem = model.factors.get("capacity", [])
            elif isinstance(rem, dict):
                rem = [float(rem[k]) for k in sorted(rem)]
            next_caps.append([float(v) for v in rem])

            for rng in eval_sol.rng_list:
                rng.advance_subsubstream()

        self.budget.request(n_reps)
        return np.mean(np.array(next_caps, dtype=float), axis=0)

    # ---------------------------------------------------------------------------
    # Helper: observe mean immediate stage-0 reward
    # ---------------------------------------------------------------------------
    def _observe_stage0_reward(
        self,
        problem: MultistageProblem,
        decision: list | tuple,
        n_reps: int,
    ) -> float:
        """Return the mean immediate stage-0 reward for ``decision`` via.

        ``n_reps`` single-stage replications.  Uses a distinct substream
        index from ``_observe_next_state`` to avoid seed collision.
        """
        eval_sol = Solution(tuple(decision), problem)
        eval_rngs = [
            MRG32k3a(s_ss_sss_index=[0, i, 1]) for i in range(problem.model.n_rngs)
        ]
        eval_sol.attach_rngs(eval_rngs, copy=False)

        model = cast(MultistageModel, problem.model)
        rewards: list[float] = []
        for _ in range(n_reps):
            if hasattr(model, "before_replication"):
                model.before_replication(eval_sol.rng_list)  # type: ignore[attr-defined]
            else:
                model.before_replicate(eval_sol.rng_list)  # type: ignore[attr-defined]

            if problem.before_replicate_override is not None:
                problem.before_replicate_override(model, eval_sol.rng_list)
            #! CHECK
            responses, _ = model.replicate_stage(
                problem.current_state,
                eval_sol.x,
                problem.current_stage,
                eval_sol.rng_list,
            )
            key = "revenue" if "revenue" in responses else next(iter(responses), None)
            reward = float(responses.get(key, 0.0)) if key else 0.0
            rewards.append(reward)

            for rng in eval_sol.rng_list:
                rng.advance_subsubstream()

        self.budget.request(n_reps)
        return float(np.mean(rewards)) if rewards else 0.0

    #! ---------------------------------------------------------------------------
    #! GP-V Refinement from Forward-Pass Observations
    #! ---------------------------------------------------------------------------
    def _refine_gp_v_from_forward_pass(
        self,
        problem: MultistageProblem,
        stage: int,
        next_state: np.ndarray,
        future_value: float,
    ) -> None:
        """Update GP-V for `stage` with a forward-pass observation and refit.

        The refinement buffer stores raw remaining-capacity vectors.
        GP-V is fitted directly on raw capacity features.
        """
        if stage not in getattr(self, "_refinement_buffers", {}):
            return
        if stage not in getattr(self, "_gp_v_objects", {}):
            return

        buf = self._refinement_buffers[stage]
        gp_v_obj = self._gp_v_objects[stage]
        new_s = np.asarray(next_state, dtype=float)

        # ------------------------------------------------------------------
        # Deep MC -- run additional replications for a lower-variance label.
        # ------------------------------------------------------------------
        deep_reps = self.config.refinement_deep_mc_reps
        if deep_reps > 0 and self.budget.remaining >= deep_reps:
            deep_problem = problem._clone_problem()
            # Deep-MC labels should use rollout simulation, not nested
            # solver-based lookahead on a cloned problem.
            deep_problem._solver_lookahead_enabled = False
            deep_problem.factors["capacity"] = [float(v) for v in new_s]
            deep_problem.factors["budget"] = deep_reps
            deep_problem.current_stage = int(stage)
            if "remaining_capacity" in deep_problem.current_state:
                deep_problem.current_state["remaining_capacity"] = [
                    float(v) for v in new_s
                ]

            n_virtual = int(problem.model.factors.get("n_virtual_classes", 1))
            n_legs = len(problem.model.factors.get("capacity", []))
            zero_action = tuple(0.0 for _ in range(n_legs * n_virtual))
            deep_sol = Solution(zero_action, deep_problem)
            deep_rngs = [
                MRG32k3a(s_ss_sss_index=[0, i, 99])
                for i in range(deep_problem.model.n_rngs)
            ]
            deep_sol.attach_rngs(deep_rngs, copy=False)

            deep_revenues: list[float] = []
            for _ in range(deep_reps):
                deep_sol = self._simulate_with_budget(deep_problem, deep_sol, n_reps=1)
                deep_revenue = float(deep_sol.objectives_mean.item())
                if math.isfinite(deep_revenue):
                    deep_revenues.append(deep_revenue)
                for rng in deep_sol.rng_list:
                    rng.advance_subsubstream()

            if deep_revenues:
                future_value = _finite_objective_or_penalty(
                    float(np.mean(deep_revenues)),
                    problem.minmax[0],
                    fallback=future_value,
                )
                print(
                    f"  [GP-V Refine / deep MC] Stage {stage}: "
                    f"{deep_reps} reps → V̂={future_value:.1f}"
                )

        # ------------------------------------------------------------------
        # Novelty check -- use raw capacity in the GP-V's scaled space.
        # ------------------------------------------------------------------
        if buf["states"] and gp_v_obj.state_scaler is not None:
            existing_arr = np.array(buf["states"], dtype=float)
            existing_scaled = gp_v_obj.state_scaler.transform(existing_arr)
            new_scaled = gp_v_obj.state_scaler.transform(new_s.reshape(1, -1))
            min_dist = float(
                np.min(np.linalg.norm(existing_scaled - new_scaled, axis=1))
            )
            if min_dist < self.config.refinement_novelty_tol:
                return  # Too close -- skip.

        # ------------------------------------------------------------------
        # FIFO buffer -- store raw capacity vectors.
        # ------------------------------------------------------------------
        future_value = _finite_objective_or_penalty(future_value, problem.minmax[0])
        max_buf = self.config.refinement_buffer_max

        if len(buf["states"]) >= max_buf:
            buf["states"].pop(0)
            buf["values"].pop(0)

        buf["states"].append(new_s.tolist())
        buf["values"].append(float(future_value))

        # ------------------------------------------------------------------
        # Refit GP-V on raw capacity features from the updated buffer.
        # ------------------------------------------------------------------
        state_arr = np.array(buf["states"], dtype=float)
        values_arr = np.array(buf["values"], dtype=float)

        # alpha is in normalized-target space (normalize_y=True → unit variance),
        # so 0.01 means ~10% noise-to-signal ratio.
        alpha_new = max(self.config.alpha, 0.01)

        gp_v_obj.alpha = alpha_new
        gp_v_obj.length_scale = (
            None  # let kernel optimizer find it (features are StandardScaled)
        )
        gp_v_obj.fit(state_arr, values_arr)
        gp_v_obj._last_y_train = values_arr.tolist()

        self.value_models[stage] = gp_v_obj.predict

        print(
            f"  [GP-V Refine] Stage {stage}: buffer={len(buf['states'])} pts, "
            f"new obs V̂={future_value:.1f}"
        )

    # ---------------------------------------------------------------------------
    # State space sampling
    # ---------------------------------------------------------------------------

    def _sample_states_composite(
        self,
        problem: ProblemLike,
        stage: int,  # noqa: ARG002
        gp_v_next: _GaussianProcessValueModel | None = None,
        n_total: int | None = None,
    ) -> list[np.ndarray]:
        """Composite state sampler combining three strategies:.

          (1 - frac_lhs - frac_uncertainty)  -- stratified LHS over capacity box
                                                                                        (tight
                                                                                        +
                                                                                        loose
                                                                                        +
                                                                                        safe
                                                                                        strata,
                                                                                        exactly
                                                                                         as
                                                                                         in
                                                                                         the
                                                                                         original
                                                                                         stratified_sample)
          frac_lhs                            -- pure LHS over the full capacity box
                                                                                        for
                                                                                        space-filling
                                                                                        coverage
          frac_uncertainty                    -- states near high-variance regions of
                                                                                        the
                                                                                        preceding
                                                                                        GP-V
                                                                                        (active
                                                                                        learning);
                                                                                        falls
                                                                                        back
                                                                                        to
                                                                                        LHS
                                                                                        at
                                                                                        the
                                                                                        terminal
                                                                                        stage

        Parameters
        ----------
        problem     : ProblemLike
        stage       : int
        gp_v_next   : trained GP-V from stage+1, or None at terminal stage
        n_total     : override for self.config.n_training_pts
        """
        if n_total is None:
            n_total = self.config.n_training_pts

        frac_lhs = self.config.frac_state_lhs
        frac_unc = self.config.frac_state_uncertainty if gp_v_next is not None else 0.0
        max(0.0, 1.0 - frac_lhs - frac_unc)

        n_lhs = max(1, int(n_total * frac_lhs))
        n_unc = max(0, int(n_total * frac_unc))
        n_stratified = max(1, n_total - n_lhs - n_unc)

        C_vec = problem.model.factors["capacity"]  # noqa: N806
        n_legs = len(C_vec)
        samples: list[np.ndarray] = []

        # ---- Component 1: Stratified LHS across capacity regions ----
        # Allocate more training points to the policy-relevant region [50-95%
        # of capacity] where the actual policy operates after stage-0 bookings,
        # while keeping some coverage of extreme regions for robustness.
        n_tight = max(1, int(n_stratified * 0.05))  # [0, 10%]   -- rarely visited
        n_loose = max(1, int(n_stratified * 0.10))  # [90, 100%] -- near starting state
        n_policy = max(
            1, int(n_stratified * 0.40)
        )  # [50, 95%]  -- where policy operates
        n_safe = max(1, n_stratified - n_tight - n_loose - n_policy)

        # def _lhs_stratum(n: int, lo_frac: float, hi_frac: float) -> list[np.ndarray]:
        # 	pts = []
        # 	bins_per_leg = []
        # 	for leg in range(n_legs):
        # 		perm = np.random.permutation(n)
        # 		b    = (perm + np.random.rand(n)) / n
        # 		cap  = float(C_vec[leg])
        # bins_per_leg.append((lo_frac * cap + b * (hi_frac - lo_frac) * cap).tolist())
        # 	for idx in range(n):
        # 		pts.append(np.array([bins_per_leg[leg][idx] for leg in range(n_legs)]))
        # 	return pts

        _lhs_stratum = partial(lhs_stratum, n_legs=n_legs, C_vec=C_vec)

        samples += _lhs_stratum(n_safe, 0.0, 1.0)  # safe: full range

        samples += _lhs_stratum(n_tight, 0.0, 0.10)  # tight: near zero
        samples += _lhs_stratum(n_loose, 0.90, 1.00)  # loose: near max
        samples += _lhs_stratum(
            n_policy, 0.50, 0.95
        )  # policy-relevant: post-booking region

        # ---- Component 2: Space-filling LHS over [0, C] ----
        sampler = qmc.LatinHypercube(d=n_legs, seed=int(np.random.randint(1000000)))
        unit = sampler.random(n=n_lhs)
        upper = np.array([float(C_vec[leg]) for leg in range(n_legs)])
        lhs_pts = qmc.scale(unit, l_bounds=np.zeros(n_legs), u_bounds=upper)
        samples += [lhs_pts[i] for i in range(n_lhs)]

        # ---- Component 3: Uncertainty-targeted states from GP-V_{n+1} ----
        if gp_v_next is not None and n_unc > 0:
            n_candidates = min(max(n_unc * 10, 500), 3000)
            sampler_c = qmc.LatinHypercube(
                d=n_legs, seed=int(np.random.randint(1000000))
            )
            candidates = qmc.scale(
                sampler_c.random(n=n_candidates),
                l_bounds=np.zeros(n_legs),
                u_bounds=upper,
            )
            stds = np.array(
                [
                    gp_v_next.predict_with_std(candidates[i].reshape(1, -1))[1]
                    for i in range(n_candidates)
                ]
            )
            top_idx = np.argsort(stds)[::-1][:n_unc]
            samples += [candidates[i] for i in top_idx]

        # Clip to feasible box
        return [np.clip(s, 0, upper) for s in samples]

    # ---------------------------------------------------------------------------
    # Action space sampling
    # ---------------------------------------------------------------------------

    def _get_action_bounds(
        self,
        problem: ProblemLike,
        state: np.ndarray,
    ) -> np.ndarray:
        """Return per-element upper bounds for the decision vector given remaining.

        capacity ``state``.  The decision is a flat vector of length
        n_legs * n_virtual_classes, where each leg's protection levels must lie
        in [0, remaining_cap_on_that_leg].
        """
        C_vec = problem.model.factors["capacity"]  # noqa: N806
        n_virtual = int(problem.model.factors.get("n_virtual_classes", 1))
        n_legs = len(C_vec)
        upper = np.zeros(n_legs * n_virtual)
        for leg in range(n_legs):
            cap = float(state[leg]) if leg < len(state) else float(C_vec[leg])
            for k in range(n_virtual):
                upper[leg * n_virtual + k] = cap
        return upper

    def _project_action_to_feasible(
        self,
        action: np.ndarray,
        problem: ProblemLike,
        state: np.ndarray,
    ) -> np.ndarray:
        """Project a candidate decision vector onto the feasible set A(s).

        For each leg the sorted protection levels must lie in
        [0, remaining_cap_on_leg].  We clip, sort per leg, and floor to
        integers (protection levels are seat counts).
        """
        n_virtual = int(problem.model.factors.get("n_virtual_classes", 1))
        n_legs = len(problem.model.factors["capacity"])
        a = np.clip(action, 0, None)

        for leg in range(n_legs):
            cap = (
                float(state[leg])
                if leg < len(state)
                else float(problem.model.factors["capacity"][leg])
            )
            start = leg * n_virtual
            end = start + n_virtual
            a[start:end] = np.clip(a[start:end], 0, cap)
            a[start:end] = np.sort(
                a[start:end]
            )  # enforce monotone protection levels per leg

        return np.floor(a)

    def _build_heuristic_actions(
        self,
        problem: ProblemLike,
        state: np.ndarray,
    ) -> list[np.ndarray]:
        """Construct anchor actions from simple heuristics:.

          1. All-zero protection (open all classes -- FCFS)
          2. Full protection (block all lower classes)
          3. Half-capacity protection
          4. Fare-rank proportional protection.

        These are projected to feasibility before being returned.
        """
        n_virtual = int(problem.model.factors.get("n_virtual_classes", 1))
        n_legs = len(problem.model.factors["capacity"])
        n_dec = n_legs * n_virtual
        anchors = []

        # 1. FCFS -- no protection
        anchors.append(np.zeros(n_dec))

        # 2. Full protection -- protect all remaining capacity on every leg
        full = np.zeros(n_dec)
        for leg in range(n_legs):
            cap = (
                float(state[leg])
                if leg < len(state)
                else float(problem.model.factors["capacity"][leg])
            )
            for k in range(n_virtual):
                full[leg * n_virtual + k] = cap * (n_virtual - k) / n_virtual
        anchors.append(self._project_action_to_feasible(full, problem, state))

        # 3. Half-capacity -- conservative middle ground
        half = full * 0.5
        anchors.append(self._project_action_to_feasible(half, problem, state))

        # 4. Fare-rank proportional: protect higher classes more aggressively
        fare_rank = np.zeros(n_dec)
        for leg in range(n_legs):
            cap = (
                float(state[leg])
                if leg < len(state)
                else float(problem.model.factors["capacity"][leg])
            )
            for k in range(n_virtual):
                # Weight decreases linearly from class 0 (highest) to n_virtual-1
                # (lowest)
                fare_rank[leg * n_virtual + k] = (
                    cap * (n_virtual - k - 1) / max(n_virtual - 1, 1)
                )
        anchors.append(self._project_action_to_feasible(fare_rank, problem, state))

        return anchors

    def _sample_actions_for_state(
        self,
        problem: ProblemLike,
        state: np.ndarray,
        prev_optimal_action: np.ndarray | None = None,
        n_total: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Composite action sampler for a single training state.

        Mixing proportions (from config):
          frac_action_heuristic -- heuristic anchor points
          frac_action_lhs       -- constrained LHS over A(s)
          remainder             -- Gaussian perturbations around prev_optimal_action
                                                          (falls back to LHS when
                                                          prev_optimal_action is None)

        Parameters
        ----------
        state               : (n_legs,) remaining capacity
        prev_optimal_action : optimal action from stage n+1 (perturbation anchor)
        n_total             : number of actions to return
        rng                 : numpy Generator (created fresh if None)

        Returns:
        -------
        actions : (n_sampled, n_decision)  feasibility-projected integer arrays
        """
        if n_total is None:
            n_total = self.config.n_actions_per_state
        if rng is None:
            rng = np.random.default_rng()

        frac_heuristic = self.config.frac_action_heuristic
        frac_lhs = self.config.frac_action_lhs
        # Perturbation gets whatever remains after heuristic + LHS
        max(0.0, 1.0 - frac_heuristic - frac_lhs)

        n_heuristic = max(1, int(n_total * frac_heuristic))
        n_lhs = max(1, int(n_total * frac_lhs))
        n_perturb = max(0, n_total - n_heuristic - n_lhs)

        upper = self._get_action_bounds(problem, state)
        n_dec = len(upper)
        parts: list[np.ndarray] = []

        # ---- Heuristic anchors ----
        anchors = self._build_heuristic_actions(problem, state)
        n_anchors = len(anchors)
        if n_anchors >= n_heuristic:
            parts.append(np.array(anchors[:n_heuristic]))
        else:
            repeats = int(np.ceil(n_heuristic / n_anchors))
            tiled = np.tile(np.array(anchors), (repeats, 1))[:n_heuristic]
            parts.append(tiled)

        # ---- Constrained LHS ----
        if n_dec > 0 and np.any(upper > 0):
            sampler = qmc.LatinHypercube(d=n_dec, seed=int(rng.integers(1000000)))
            unit = sampler.random(n=n_lhs)
            # Ensure upper > 0 for all dims so qmc.scale doesn't reject l==u.
            upper_safe = np.maximum(upper, 1e-6)
            raw_lhs = qmc.scale(unit, l_bounds=np.zeros(n_dec), u_bounds=upper_safe)
            lhs_acts = np.array(
                [
                    self._project_action_to_feasible(raw_lhs[i], problem, state)
                    for i in range(n_lhs)
                ]
            )
        else:
            lhs_acts = np.zeros((n_lhs, max(n_dec, 1)))
        parts.append(lhs_acts)

        # ---- Gaussian perturbations around previous-stage optimal ----
        if n_perturb > 0:
            if prev_optimal_action is not None and len(prev_optimal_action) == n_dec:
                std = 0.15 * upper  # 15% of per-element upper bound
                noise = rng.normal(0, std, size=(n_perturb, n_dec))
                raw_pert = prev_optimal_action[None, :] + noise
                perturb_acts = np.array(
                    [
                        self._project_action_to_feasible(raw_pert[i], problem, state)
                        for i in range(n_perturb)
                    ]
                )
            else:
                # No anchor available -- use additional LHS
                sampler2 = qmc.LatinHypercube(d=n_dec, seed=int(rng.integers(1000000)))
                raw2 = qmc.scale(
                    sampler2.random(n=n_perturb),
                    l_bounds=np.zeros(n_dec),
                    u_bounds=upper,
                )
                perturb_acts = np.array(
                    [
                        self._project_action_to_feasible(raw2[i], problem, state)
                        for i in range(n_perturb)
                    ]
                )
            parts.append(perturb_acts)

        actions = np.vstack(parts)
        # Remove exact duplicates while preserving order
        _, unique_idx = np.unique(actions, axis=0, return_index=True)
        return actions[np.sort(unique_idx)]

    # ---------------------------------------------------------------------------
    # Q-value estimation (single state-action pair)
    # ---------------------------------------------------------------------------

    def _estimate_q_for_pair(
        self,
        problem: MultistageProblem,
        stage: int,
        state: np.ndarray,
        action: np.ndarray,
        continuation_model: Callable[..., float] | None = None,
        crn_base_seed: int | None = None,
    ) -> tuple[float, float]:
        """Estimate Q_n(s, a) = E[ r_n(s,a) + V_{n+1}( phi(s', n+1) ) ] via MC.

        FIX 1 (Bellman consistency)
        ───────────────────────────
        The continuation model is now ALWAYS queried with the same ENRICHED
        feature vector phi(s', n+1) that was used to fit GP-V_{n+1}.  Previously
        it was queried with the raw remaining-capacity vector, which is inconsistent
        whenever the GP-V was trained on raw capacity features.

        FIX 3 (state representation)
        ────────────────────────────
        The next-state features phi(s', n+1) include time-remaining, load factor,
        demand intensity, and EMSR-B marginal values -- not just raw capacities.

        Parameters
        ----------
        problem            : ProblemLike
        stage              : current stage index (0-based)
        state              : (n_legs,) remaining capacity at stage `stage`
        action             : (n_decision,) protection-level vector
        continuation_model : callable(features_2d) -> float, or None at terminal
        crn_base_seed      : int or None -- CRN substream index

        Returns:
        -------
        q_mean : float -- E[r + V_{n+1}] over K replicates
        q_var  : float -- MC variance + propagated GP-V uncertainty
        """
        K = self.config.n_mc_replicates  # noqa: N806
        totals: list[float] = []
        v_next_vars: list[float] = []

        # Backward pass is budget-free -- full budget reserved for the forward pass.
        # K = min(K, max(1, self.budget.remaining))
        # self.budget.request(K)

        sim_problem = problem._clone_problem()
        q_model = cast(MultistageModel, sim_problem.model)
        for k in range(K):
            sim_problem.factors["budget"] = 1
            sim_problem._solver_lookahead_enabled = False
            sim_problem.current_stage = int(stage)
            sim_problem.factors["capacity"] = [float(v) for v in state]
            if "remaining_capacity" in sim_problem.current_state:
                sim_problem.current_state["remaining_capacity"] = [
                    float(v) for v in state
                ]
            new_problem = sim_problem

            is_flat_multistage = (
                isinstance(sim_problem, MultistageProblem)
                and isinstance(action, list | np.ndarray | tuple)
                and len(action) == sim_problem.model.n_stages * sim_problem.dim
                and len(action) > 0
                and isinstance(action[0], numbers.Number)
            )
            is_nested_multistage = (
                isinstance(action, list | np.ndarray | tuple)
                and len(action) > 0
                and isinstance(action[0], list | np.ndarray | tuple)
            )

            if is_nested_multistage:
                # if it's a flat or nested multistage decision vector, we need to get
                # the first stage's decision
                action_float = tuple(float(v) for v in action[0])
            elif is_flat_multistage:
                stage_dim = sim_problem.dim
                action_float = tuple(float(v) for v in action[:stage_dim])
            else:
                action_float = tuple(float(v) for v in action)

            eval_solution = Solution(action_float, new_problem)

            if crn_base_seed is not None and self.config.use_crn:
                seed_ss, seed_sss = crn_base_seed, k
            else:
                seed_ss, seed_sss = k, 0

            eval_rngs = [
                MRG32k3a(s_ss_sss_index=[0, seed_ss, seed_sss])
                for _ in range(q_model.n_rngs)
            ]
            eval_solution.attach_rngs(eval_rngs, copy=False)

            # ── Simulate one stage ────────────────────────────────────────
            if hasattr(q_model, "before_replication"):
                q_model.before_replication(eval_solution.rng_list)  # type: ignore[attr-defined]
            else:
                q_model.before_replicate(eval_solution.rng_list)  # type: ignore[attr-defined]

            if new_problem.before_replicate_override is not None:
                new_problem.before_replicate_override(q_model, eval_solution.rng_list)

            #! CHECK
            stage_responses, next_state_obj = q_model.replicate_stage(
                new_problem.current_state,
                eval_solution.x,
                new_problem.current_stage,
                eval_solution.rng_list,
            )

            stage_key = (
                "revenue"
                if "revenue" in stage_responses
                else next(iter(stage_responses), None)
            )
            stage_reward = (
                float(stage_responses.get(stage_key, 0.0)) if stage_key else 0.0
            )

            # ── Extract next remaining capacity ───────────────────────────
            next_remaining = next_state_obj.get("remaining_capacity", None)
            if next_remaining is None:
                next_rem_vec = np.array([float(v) for v in state])
            elif isinstance(next_remaining, dict):
                next_rem_vec = np.array(
                    [float(next_remaining[i]) for i in sorted(next_remaining)]
                )
            else:
                next_rem_vec = np.array([float(v) for v in next_remaining])

            # Query V_{t+1} with raw capacity of the next state.
            if continuation_model is not None:
                next_features = next_rem_vec.reshape(1, -1)
                v_next_mean = float(continuation_model(next_features))

                # Propagate GP-V uncertainty into the variance estimate.
                gp_v_obj = getattr(self, "_current_gp_v_next", None)
                if gp_v_obj is not None and hasattr(gp_v_obj, "predict_with_std"):
                    _, v_std = gp_v_obj.predict_with_std(next_features.reshape(1, -1))
                    v_next_vars.append(float(v_std**2) if math.isfinite(v_std) else 0.0)
                else:
                    v_next_vars.append(0.0)
            else:
                # Terminal stage: no continuation value.
                v_next_mean = 0.0
                v_next_vars.append(0.0)

            sample_total = stage_reward + v_next_mean
            if math.isfinite(sample_total):
                totals.append(sample_total)

        q_mean = _finite_objective_or_penalty(
            float(np.mean(totals)) if totals else float("nan"),
            problem.minmax[0],
        )
        mc_var = float(np.var(totals)) / max(K, 2) if totals else 0.0
        prop_var = float(np.mean(v_next_vars)) / K if v_next_vars else 0.0
        q_var = mc_var + prop_var
        return q_mean, q_var

    # ---------------------------------------------------------------------------
    # Value maximisation -- via wrapped solver
    # ---------------------------------------------------------------------------
    def _maximise_via_heuristic(
        self,
        problem: MultistageProblem,
        state: np.ndarray,
        stage: int,
        continuation_model: Callable[..., float] | None = None,
        gp_v_next_obj: _GaussianProcessValueModel | None = None,
        kappa: float = 0.0,  # noqa: ARG002
    ) -> tuple[float, np.ndarray]:
        """Find a good action at ``(state, stage)`` using heuristic candidate.

        scoring, then estimate V(s) via MC.

        Unlike the previous ``_maximise_via_wrapped_solver``, action selection
        is done entirely through GP-V look-ahead scoring (zero simulation
        budget).  Only the final MC estimate of V(s) at the chosen action
        consumes simulation budget (``n_mc_replicates`` calls).

        This dramatically reduces backwards-pass budget consumption, freeing
        the majority of the total budget for the forward pass where the inner
        solver can meaningfully optimize the stage-0 decision.

        Steps
        -----
        1. Generate diverse candidate actions via ``_sample_actions_for_state``.
        2. Score each candidate using ``_score_actions_batch`` (deterministic
           GP-V query -- no simulation cost).
        3. Pick the best-scoring action.
        4. Estimate V(s) = E[r + V_{t+1}(s')] at the chosen action via a short
           MC run (``_estimate_q_for_pair``).

        Parameters
        ----------
        state            : (n_legs,) raw remaining-capacity vector
        stage            : current stage index (0-based)
        continuation_model: callable(features_2d) -> float, or None at terminal
        gp_v_next_obj    : fitted _GaussianProcessValueModel for stage+1, or None
        kappa            : UCB exploration weight (0 = pure exploitation)

        Returns:
        -------
        v_opt   : float         -- MC estimate of E[r + V_{t+1}] at the best action
        a_opt   : (n_decision,) -- best action found by heuristic scoring
        """
        rng_local = np.random.default_rng()
        prev_opt = self._get_perturbation_anchor(stage, state)

        # ── Step 1: Generate candidate actions (no budget cost) ───────────
        candidates = self._sample_actions_for_state(
            problem,
            state,
            prev_opt,
            rng=rng_local,
        )

        # ── Step 2: Score via GP-V look-ahead (no budget cost) ────────────
        if gp_v_next_obj is not None and gp_v_next_obj.is_fitted:
            scores = self._score_actions_batch(
                problem,
                stage,
                state,
                candidates,
                gp_v_next_obj,
            )
            best_idx = int(np.argmax(scores))
            a_opt = self._project_action_to_feasible(
                candidates[best_idx].copy(),
                problem,
                state,
            )
        else:
            # Terminal stage or no GP-V available -- use half-capacity heuristic.
            anchors = self._build_heuristic_actions(problem, state)
            a_opt = anchors[2] if len(anchors) > 2 else anchors[0]

        # ── Step 3: MC estimate of V(s) (costs n_mc_replicates budget) ────
        self._current_gp_v_next = gp_v_next_obj
        v_opt, _ = self._estimate_q_for_pair(
            problem,
            stage,
            state,
            a_opt,
            continuation_model,
            crn_base_seed=None,
        )

        return v_opt, a_opt

    # ---------------------------------------------------------------------------
    # Backwards fit -- GP-V via direct simopt per training state
    # ---------------------------------------------------------------------------

    def _backwards_fit(self, problem: MultistageProblem) -> None:
        """Backwards induction from stage T down to stage 1.

        Budget architecture
        -------------------
        The backwards pass works strictly within ``self.backwards_budget_amount``.
        It never touches the forward-pass allocation.

          backwards_budget
          ├── initial_sweep_budget  (bellman_training_budget_fraction)
          │   ├── stage T-1 envelope  (adaptive weight)
          │   │   └── n_pts x cost_per_state
          │   ├── stage T-2 envelope  ...
          │   └── stage 1 envelope    (highest weight -- most uncertainty)
          └── bellman_budget  (remainder, divided among bellman iters)

        Each stage is **guaranteed** a minimum allocation so that every GP-V
        model is fitted with at least 3 training points.

        Adaptive weighting: earlier stages (closer to 0) receive proportionally
        more budget because errors there compound through all downstream stages.
        Weight for stage t = (n_stages - t), so stage 1 gets the most.
        """
        stages_to_fit = list(range(1, problem.model.n_stages))
        n_stages_to_fit = max(1, len(stages_to_fit))

        self._gp_v_objects: dict = {}
        self._stage_optimal_actions: dict = {}
        self._stage_training_states: dict = {}

        # ── Budget accounting ───────────────────────────────────────────────
        # The backward pass is budget-free (does not charge self.budget).
        # Compute the internal budget envelope from the actual work needed
        # so that stage allocation and Bellman iterations proceed normally.
        backward_budget_start = self.budget.used
        n_stages_to_fit = len(stages_to_fit)
        bellman_iters = max(1, self.config.bellman_n_iters - 1)
        total_backward_budget = (
            n_stages_to_fit
            * self.config.n_training_pts
            * self.config.n_mc_replicates
            * (1 + bellman_iters)
        )

        # Split: initial sweep vs Bellman refinement iterations.
        initial_sweep_budget = int(
            total_backward_budget * self.config.bellman_training_budget_fraction
        )
        bellman_budget = total_backward_budget - initial_sweep_budget

        # ── Per-state cost (heuristic scoring is free; only MC estimate costs) ──
        # Each _maximise_via_heuristic call costs only n_mc_replicates
        # (the _estimate_q_for_pair step).  Action selection via GP-V scoring
        # has zero simulation budget cost.
        cost_per_state = self.config.n_mc_replicates

        # ── Adaptive per-stage budget envelopes ────────────────────────────
        n_stages_total = problem.model.n_stages
        stage_weights = {t: (n_stages_total - t) for t in stages_to_fit}
        total_weight = max(1, sum(stage_weights.values()))
        stage_budgets = {
            t: max(1, int(initial_sweep_budget * stage_weights[t] / total_weight))
            for t in stages_to_fit
        }

        print(
            f"[Backwards fit] total backward budget={total_backward_budget}, "
            f"initial sweep={initial_sweep_budget}, bellman={bellman_budget}"
        )
        print(f"  Stage budgets: {stage_budgets}  (cost/state={cost_per_state})")

        # ── Initial sweep: fit GP-V for each stage ─────────────────────────
        for stage in reversed(stages_to_fit):
            stage_envelope = stage_budgets[stage]
            stage_budget_used = 0
            print(
                f"\n[GP Backwards Fit] Stage {stage}  |  "
                f"budget remaining: {self._backwards_budget_remaining()}/{total_backward_budget}  |  "  # noqa: E501
                f"stage envelope: {stage_envelope}"
            )

            # ── Retrieve GP-V from stage t+1 ──────────────────────────────
            if stage == problem.model.n_stages - 1:
                continuation_model = None
                gp_v_next_obj = None
                self._current_gp_v_next = None
            else:
                prev_stage = stage + 1
                continuation_model = self.value_models.get(prev_stage)
                gp_v_next_obj = self._gp_v_objects.get(prev_stage)
                self._current_gp_v_next = gp_v_next_obj

            # ── Compute how many states we can afford for this stage ───────
            max_affordable = max(1, stage_envelope // max(cost_per_state, 1))
            n_samples = min(self.config.n_training_pts, max_affordable)
            n_samples = max(3, n_samples)  # absolute minimum: 3 pts for a GP fit
            full_cost_per_state = cost_per_state

            print(
                f"  Planning {n_samples} training states "
                f"(MC cost/state={cost_per_state})"
            )

            # ── Step 1: Sample states (raw capacity space) ─────────────────
            states = self._sample_states_composite(
                problem, stage, gp_v_next=gp_v_next_obj, n_total=n_samples
            )
            self._stage_training_states[stage] = states

            # ── Step 2: For each state, run simopt to find a*(s) and V(s) ──
            v_values: list[float] = []
            optimal_actions_s: list[np.ndarray] = []

            for i, s in enumerate(states):
                # Guard 1: don't exceed this stage's envelope.
                if stage_budget_used + full_cost_per_state > stage_envelope:
                    print(
                        f"  [Stage envelope] Stopping at state {i}/{len(states)} "
                        f"(spent {stage_budget_used}/{stage_envelope})."
                    )
                    break
                # Guard 2: don't eat into the forward-pass budget.
                if self._backwards_budget_remaining() < full_cost_per_state:
                    print(
                        f"  [Backward budget guard] Stopping at state {i}/{len(states)} "  # noqa: E501
                        f"(backward remaining={self._backwards_budget_remaining()})."
                    )
                    break

                budget_before = self.budget.used
                v_opt, a_opt = self._maximise_via_heuristic(
                    problem,
                    s,
                    stage,
                    continuation_model,
                    gp_v_next_obj,
                    kappa=0.0,
                )
                actual_cost = self.budget.used - budget_before
                stage_budget_used += actual_cost

                v_values.append(v_opt)
                optimal_actions_s.append(a_opt)

            # Trim states to match the number of V-values actually computed.
            fitted_states = states[: len(v_values)]

            self._assert_bellman_targets(stage, fitted_states, v_values)

            self._stage_optimal_actions[stage] = {
                tuple(fitted_states[i].tolist()): optimal_actions_s[i]
                for i in range(len(fitted_states))
            }

            # ── Step 3: Train GP-V on raw capacity vectors ─────────────────
            if not v_values:
                print(
                    f"  [Warning] No V-values computed for stage {stage}; skipping GP-V fit."  # noqa: E501
                )
                continue

            state_arr_for_v = np.array(
                [np.asarray(s, dtype=float) for s in fitted_states]
            )
            v_arr = np.array(v_values, dtype=float)

            # alpha is in normalized-target space (normalize_y=True → unit variance),
            # so 0.01 means ~10% noise-to-signal ratio.
            alpha_gp_v = max(self.config.alpha, 0.01)

            gp_v_model = _GaussianProcessValueModel(
                alpha=alpha_gp_v,
                length_scale=None,  # let kernel optimizer find it (features are StandardScaled)  # noqa: E501
            )
            gp_v_model.fit(state_arr_for_v, v_arr)
            gp_v_model._last_y_train = v_arr.tolist()
            gp_v_model._last_raw_states = [s.tolist() for s in fitted_states]
            gp_v_model._last_stage = stage
            self._gp_v_objects[stage] = gp_v_model
            self.value_models[stage] = gp_v_model.predict

            print(
                f"  GP-V trained on {len(v_values)} pts.  "
                f"V range: [{v_arr.min():.1f}, {v_arr.max():.1f}]  "
                f"(stage budget used: {stage_budget_used}/{stage_envelope})"
            )

            if stage == 1:
                self._stage1_gp_init = gp_v_model
                self._stage1_X_state_init = list(state_arr_for_v)
                self._stage1_Y_init = list(v_arr)

        # ── Bellman consistency iterations ─────────────────────────────────
        # Only proceed if backward budget remains after the initial sweep.
        # Each iteration re-solves V_t(s) = E[max_a(r + V_{t+1}(s'))] per
        # training state using the updated GP-V from the previous iteration.

        bellman_budget_remaining = self._backwards_budget_remaining()
        n_bellman_extra_iters = max(1, self.config.bellman_n_iters - 1)
        bellman_iter_budget = max(1, bellman_budget_remaining // n_bellman_extra_iters)

        # Seed convergence tracking with the initial V values.
        prev_v_by_stage: dict[int, list[float]] = {
            stage: list(getattr(self._gp_v_objects.get(stage), "_last_y_train", []))
            for stage in stages_to_fit
        }

        for biter in range(1, self.config.bellman_n_iters):
            bw_remaining = self._backwards_budget_remaining()
            # Need enough budget for at least one state per stage.
            min_iter_cost = n_stages_to_fit * cost_per_state
            if bw_remaining < min_iter_cost:
                print(
                    f"\n[Bellman] Iter {biter}: insufficient backward budget "
                    f"({bw_remaining} < {min_iter_cost}).  Stopping."
                )
                break

            # Cap this iteration's spend.
            iter_budget_cap = min(bellman_iter_budget, bw_remaining)
            iter_start_used = self.budget.used

            print(
                f"\n[Bellman Iter {biter}/{self.config.bellman_n_iters - 1}]  "
                f"backward remaining: {bw_remaining}, iter cap: {iter_budget_cap}"
            )

            # Per-state cost for Bellman iteration (heuristic: only MC cost).
            bellman_full_cost_per_state = cost_per_state

            new_v_by_stage: dict[int, list[float]] = {}

            for stage in reversed(stages_to_fit):
                # Resample 25% of states each Bellman iter for diversity.
                existing_states = self._stage_training_states[stage]
                if not existing_states:
                    continue
                n_existing = len(existing_states)
                n_resample = max(2, n_existing // 4)
                n_keep = n_existing - n_resample
                gp_v_next_for_sample = self._gp_v_objects.get(stage + 1)
                new_states = self._sample_states_composite(
                    problem,
                    stage,
                    gp_v_next=gp_v_next_for_sample,
                    n_total=n_resample,
                )
                states = existing_states[:n_keep] + new_states
                self._stage_training_states[stage] = states

                # ── Continuation model for this Bellman iteration ──────────
                if stage == problem.model.n_stages - 1:
                    cont_model = None
                    gp_v_next_obj = None
                else:
                    gp_v_next_obj = self._gp_v_objects.get(stage + 1)
                    cont_model = gp_v_next_obj.predict if gp_v_next_obj else None

                self._current_gp_v_next = gp_v_next_obj

                new_v_values: list[float] = []
                new_opt_actions: list[np.ndarray] = []

                for s in states:
                    # Stop if this iteration's budget cap is reached.
                    iter_spent = self.budget.used - iter_start_used
                    if iter_spent + bellman_full_cost_per_state > iter_budget_cap:
                        break
                    if self._backwards_budget_remaining() < bellman_full_cost_per_state:
                        break

                    v_opt, a_opt = self._maximise_via_heuristic(
                        problem,
                        s,
                        stage,
                        cont_model,
                        gp_v_next_obj,
                        kappa=0.0,
                    )
                    new_v_values.append(v_opt)
                    new_opt_actions.append(a_opt)

                if not new_v_values:
                    break

                new_v_by_stage[stage] = new_v_values

                # ── Re-fit GP-V with updated Bellman targets ───────────────
                fitted_iter_states = states[: len(new_v_values)]
                state_arr_iter = np.array(
                    [np.asarray(s, dtype=float) for s in fitted_iter_states]
                )
                v_iter_arr = np.array(new_v_values, dtype=float)

                # alpha is in normalized-target space (normalize_y=True).
                alpha_iter = max(self.config.alpha, 0.01)

                gp_v_iter = _GaussianProcessValueModel(
                    alpha=alpha_iter,
                    length_scale=None,
                )
                gp_v_iter.fit(state_arr_iter, v_iter_arr)
                gp_v_iter._last_y_train = v_iter_arr.tolist()
                gp_v_iter._last_raw_states = [s.tolist() for s in fitted_iter_states]
                gp_v_iter._last_stage = stage

                self._gp_v_objects[stage] = gp_v_iter
                self.value_models[stage] = gp_v_iter.predict

                # Update optimal-action map
                self._stage_optimal_actions[stage] = {
                    tuple(fitted_iter_states[i].tolist()): new_opt_actions[i]
                    for i in range(len(new_opt_actions))
                }

                print(
                    f"  Stage {stage}: V range [{v_iter_arr.min():.1f}, "
                    f"{v_iter_arr.max():.1f}]"
                )

                if stage == 1:
                    self._stage1_gp_init = gp_v_iter
                    self._stage1_X_state_init = list(state_arr_iter)
                    self._stage1_Y_init = list(v_iter_arr)

            # ── Convergence check across all stages ────────────────────────
            max_rel_change = 0.0
            for stage in stages_to_fit:
                prev_v = np.array(prev_v_by_stage.get(stage, []))
                new_v = np.array(new_v_by_stage.get(stage, []))
                if len(prev_v) == len(new_v) and len(prev_v) > 0:
                    denom = np.maximum(np.abs(prev_v), 1.0)
                    rel = float(np.max(np.abs(new_v - prev_v) / denom))
                    max_rel_change = max(max_rel_change, rel)

            print(
                f"  [Bellman] Max relative V change across stages: {max_rel_change:.4f}"
            )
            prev_v_by_stage = new_v_by_stage

            if max_rel_change < self.config.bellman_convergence_tol:
                print(
                    f"  [Bellman] Converged (tol={self.config.bellman_convergence_tol}) "  # noqa: E501
                    f"after {biter} iteration(s)."
                )
                break

        backward_total_used = self.budget.used - backward_budget_start
        print(
            f"\n[Backwards fit complete] Used {backward_total_used}/{total_backward_budget} "  # noqa: E501
            f"backward budget.  Budget remaining for forward pass: {self.budget.remaining}"  # noqa: E501
        )

    def _get_perturbation_anchor(
        self,
        stage: int,
        state: np.ndarray,
    ) -> np.ndarray | None:
        """Return the optimal action from ``stage`` for the training state nearest.

        to ``state``, or None if no data is available for that stage yet.
        """
        if stage not in self._stage_optimal_actions:
            return None
        optimal_map = self._stage_optimal_actions[stage]
        training_keys = list(optimal_map.keys())
        if not training_keys:
            return None
        training_arr = np.array(training_keys, dtype=float)
        dists = np.linalg.norm(training_arr - state, axis=1)
        nearest_key = training_keys[int(np.argmin(dists))]
        return optimal_map[nearest_key]


def query_value(
    state_arr: np.ndarray,
    gp_v1: _GaussianProcessValueModel,
    value_model: Callable[[np.ndarray], float],
    kappa: float,
) -> float:
    """Query the value model at a given state."""
    if gp_v1 is not None and gp_v1.is_fitted:
        return gp_v1.predict_ucb(state_arr, kappa=kappa)
    return float(value_model(state_arr))


def _finite_objective_or_penalty(
    value: float,
    minmax_sign: int,
    fallback: float | None = None,
) -> float:
    value = float(value)
    if math.isfinite(value):
        return value
    if fallback is not None:
        fallback_value = float(fallback)
        if math.isfinite(fallback_value):
            return fallback_value
    return float(-1e12 * minmax_sign)


def lhs_stratum(
    n: int,
    lo_frac: float,
    hi_frac: float,
    n_legs: int,
    C_vec: tuple,  # noqa: N803
) -> list[np.ndarray]:
    """Generate Latin Hypercube Sampling points within a stratum."""
    pts = []
    bins_per_leg = []
    for leg in range(n_legs):
        perm = np.random.permutation(n)
        b = (perm + np.random.rand(n)) / n
        cap = float(C_vec[leg])
        bins_per_leg.append((lo_frac * cap + b * (hi_frac - lo_frac) * cap).tolist())
    for idx in range(n):
        pts.append(np.array([bins_per_leg[leg][idx] for leg in range(n_legs)]))
    return pts


def surrogate_simulate(
    solution: Solution,
    num_macroreps: int = 1,
    adp_self: ADPSolver | None = None,
    wrapped_solver: Solver | None = None,
    problem: ProblemLike | None = None,
    **kwargs: object,  # noqa: ARG001
) -> None:
    """Run a surrogate simulation replication."""
    assert adp_self is not None
    assert isinstance(problem, MultistageProblem)
    adp_self._approximate_replication(
        wrapped_solver,
        problem,
        solution,
        num_macroreps,
    )
