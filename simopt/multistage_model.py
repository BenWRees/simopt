"""Base class for multistage simulation-optimization models.

In multistage simulation optimization, we solve sequential decision-making
problems under uncertainty. At each stage t = 0, 1, ..., T-1:

  1. Observe current state s_t
  2. Choose decision x_t
  3. Realize stochastic outcome ξ_t
  4. Transition to s_{t+1} = f(s_t, x_t, ξ_t)
  5. Receive stage reward r_t(s_t, x_t, s_{t+1})

The goal is to find decisions that optimize the total expected reward:
    max/min c_1(x_1) + E[max/min c_2(x_2) + E[max/min c_3(x_3) + ... + E[max/min
    c_T(x_T)]]]

This differs from standard simulation optimization because the decision at
stage t must account for both the immediate stage reward *and* the expected
value of future stages (cost-to-go).

Two strategies are available for estimating the cost-to-go:

1. **Policy-based lookahead** (default): A rollout policy
   ``(state, stage) -> decision`` is used for Monte Carlo simulation of
   future stages (see :meth:`MultistageModel.simulate_lookahead`).

2. **Solver-based lookahead**: A simulation-optimization solver is used
   to find (approximately) optimal decisions at each future stage.  This
   is orchestrated by :meth:`MultistageProblem.simulate_lookahead_with_solver`
   and activated by passing a ``lookahead_solver`` to
   :meth:`MultistageProblem.simulate`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import Any, ClassVar

from boltons.typeutils import classproperty
from pydantic import BaseModel

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.utils import get_specifications

logger = logging.getLogger(__name__)


class MultistageModel(ABC):
    """Base class for multistage simulation-optimization models.

    Subclasses must implement:
        - :meth:`get_initial_state` - returns the starting state *s₀*.
        - :meth:`transition` - computes *s_{t+1} = f(s_t, x_t, ξ_t)*.
        - :meth:`stage_reward` - computes the immediate reward at stage *t*.
        - :meth:`get_default_policy` - returns a rollout policy for lookahead.
    """

    class_name_abbr: ClassVar[str]
    """Short name of the model class."""

    class_name: ClassVar[str]
    """Long name of the model class."""

    config_class: ClassVar[type[BaseModel]]
    """Configuration class for the model."""

    n_rngs: ClassVar[int]
    """Number of RNGs used to run a simulation replication."""

    n_responses: ClassVar[int]
    """Number of responses (performance measures)."""

    n_stages: ClassVar[int]
    """Total number of decision stages *T*. Runs from stage 0 to stage T-1 with transitions to stage T (terminal)."""  # noqa: E501

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize a multistage model object.

        Args:
            fixed_factors: Dictionary of user-specified model factors.
        """
        fixed_factors = fixed_factors or {}
        self.config = self.config_class(**fixed_factors)
        self._factors = self.config.model_dump(by_alias=True)
        self._rng_list: list[MRG32k3a] = []

    def __eq__(self, other: object) -> bool:
        """Check if two models are equivalent.

        Args:
            other: Other object to compare to self.

        Returns:
            True if the two models are equivalent, otherwise False.
        """
        if not isinstance(other, MultistageModel):
            return False
        return type(self) is type(other) and self.factors == other.factors

    @staticmethod
    def _make_hashable(value: Any) -> Any:  # noqa: ANN401
        """Convert a value to a hashable form (lists → tuples, recursively)."""
        if isinstance(value, list):
            return tuple(MultistageModel._make_hashable(v) for v in value)
        return value

    def __hash__(self) -> int:
        """Return the hash value of the model."""
        hashable_items = tuple(
            (k, self._make_hashable(v)) for k, v in sorted(self.factors.items())
        )
        return hash((type(self), hashable_items))

    @classproperty
    def name(cls: type[MultistageModel]) -> str:  # noqa: N805
        """Return the name of the model."""
        return cls.__name__.replace("_", " ")

    @classproperty
    def specifications(cls) -> dict:  # noqa: N805
        """Details of each factor (for GUI, data validation, and defaults)."""
        return get_specifications(cls.config_class)

    @property
    def factors(self) -> dict:
        """Changeable factors of the simulation model."""
        return self._factors

    def model_created(self) -> None:  # noqa: B027
        """Hook for any additional initialization after the model is created."""
        pass

    # ------------------------------------------------------------------
    # Abstract interface - must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def get_initial_state(self) -> Any:  # noqa: ANN401
        """Return the initial state *s₀* for the problem.

        Returns:
            The initial state object (type is model-specific).
        """
        raise NotImplementedError

    @abstractmethod
    def transition(
        self,
        state: object,
        decision: tuple,
        stage: int,
        rng_list: list[MRG32k3a],
    ) -> object:
        """Compute the next state: *s_{t+1} = f(s_t, x_t, ξ_t)*.

        Args:
            state: Current state *s_t*.
            decision: Decision *x_t* taken at stage *t*.
            stage: Current stage index *t*.
            rng_list: RNGs for generating stochastic outcomes *ξ_t*.

        Returns:
            The next state *s_{t+1}*.
        """
        raise NotImplementedError

    @abstractmethod
    def stage_reward(
        self,
        state: object,
        decision: tuple,
        next_state: object,
        stage: int,
    ) -> dict[str, float]:
        """Compute the immediate reward/cost at stage *t*.

        Args:
            state: Current state *s_t*.
            decision: Decision *x_t* taken.
            next_state: Resulting state *s_{t+1}* after transition.
            stage: Current stage index *t*.

        Returns:
            Dictionary mapping response names to their values for this stage.
        """
        raise NotImplementedError

    @abstractmethod
    def get_default_policy(self) -> Callable[[Any, int], tuple]:
        """Return a default rollout policy for Monte Carlo lookahead.

        The policy is a callable ``policy(state, stage) -> decision`` used to
        simulate future stages when estimating the cost-to-go.

        Returns:
            A function ``(state, stage) -> decision_tuple``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Replication machinery
    # ------------------------------------------------------------------

    def before_replication(self, rng_list: list[MRG32k3a]) -> None:
        """Prepare the model before a replication.

        Stores the RNG list for use during stage simulation and lookahead.
        Override in subclasses to perform additional per-replication setup
        (e.g. attaching RNGs to input models).

        Args:
            rng_list: RNGs for driving the simulation.
        """
        self._rng_list = rng_list

    def replicate_stage(
        self,
        state: object,
        decision: tuple,
        stage: int,
        rng_list: list[MRG32k3a],
    ) -> tuple[dict[str, float], Any]:
        """Simulate a single stage: transition and compute the reward.

        This is the building-block used by both the main replication and the
        Monte Carlo lookahead.

        Args:
            state: Current state *s_t*.
            decision: Decision *x_t*.
            stage: Stage index *t*.
            rng_list: RNGs for the stochastic transition.

        Returns:
            tuple:
                - dict: Stage reward/response values.
                - Any: Next state *s_{t+1}*.
        """
        next_state = self.transition(state, decision, stage, rng_list)
        reward = self.stage_reward(state, decision, next_state, stage)
        return reward, next_state

    def simulate_lookahead(
        self,
        state: object,
        start_stage: int,
        policy: Callable[[Any, int], tuple] | None = None,
        n_reps: int = 30,
    ) -> dict[str, float]:
        """Estimate the cost-to-go from *state* via Monte Carlo rollout.

        Simulates from ``start_stage`` to the terminal stage ``n_stages``
        using the given policy (falling back to the default), then averages
        the cumulative response over *n_reps* replications.

        Args:
            state: Starting state for the lookahead.
            start_stage: Stage index to begin the rollout from.
            policy: Rollout policy ``(state, stage) -> decision``.
                Falls back to :meth:`get_default_policy` when *None*.
            n_reps: Number of Monte Carlo replications.

        Returns:
            Averaged cumulative response values over the lookahead horizon.
        """
        if start_stage >= self.n_stages:
            return {}

        policy = policy or self.get_default_policy()

        total_responses: dict[str, float] = {}
        for _ in range(n_reps):
            rep_responses: dict[str, float] = {}
            sim_state = deepcopy(state)
            for t in range(start_stage, self.n_stages):
                decision = policy(sim_state, t)
                stage_resp, sim_state = self.replicate_stage(
                    sim_state, decision, t, self._rng_list
                )
                for key, val in stage_resp.items():
                    rep_responses[key] = rep_responses.get(key, 0.0) + val

            for key, val in rep_responses.items():
                total_responses[key] = total_responses.get(key, 0.0) + val

        # Average over replications
        if n_reps > 0:
            for key in total_responses:
                total_responses[key] /= n_reps

        return total_responses

    def replicate(
        self,
        state: object,
        decision: tuple,
        stage: int,
        policy: Callable[[Any, int], tuple] | None = None,
        n_lookahead_reps: int = 30,
        future_responses: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict]:
        """Simulate a single replication at the given state and decision.

        Evaluates the current stage's immediate reward, then estimates the
        cost-to-go for remaining stages via Monte Carlo lookahead.

        The returned response dictionary contains three entries per response
        key ``k`` defined by :meth:`stage_reward`:

        - ``stage_<k>`` - immediate reward at the current stage.
        - ``future_<k>`` - estimated average future reward (cost-to-go).
        - ``total_<k>`` - sum of stage and future reward.

        Args:
            state: Current state *s_t*.
            decision: Decision *x_t* to evaluate.
            stage: Current stage index *t*.
            policy: Rollout policy for lookahead (defaults to
                :meth:`get_default_policy`).
            n_lookahead_reps: Number of MC replications for cost-to-go.
            future_responses: Pre-computed future reward estimates
                (e.g. from solver-based lookahead).  When provided the
                policy-based Monte Carlo lookahead is skipped and these
                values are used directly as the cost-to-go.

        Returns:
            tuple:
                - dict: Response measures (stage, future, and total).
                - dict: Gradient estimates (empty by default).
        """
        # --- Current stage (deterministic given state, decision, and RNG) ---
        current_responses, next_state = self.replicate_stage(
            state, decision, stage, self._rng_list
        )

        # advance the stage of the problem for logging purposes (if applicable)

        # self.current_stage = stage + 1

        # --- Cost-to-go estimation ---
        if future_responses is None:
            # Standard policy-based Monte Carlo lookahead
            future_responses = self.simulate_lookahead(
                next_state, stage + 1, policy, n_lookahead_reps
            )

        # --- Combine current and future responses ---
        responses: dict[str, float] = {}
        for key in current_responses:
            responses[f"stage_{key}"] = current_responses[key]
            responses[f"future_{key}"] = future_responses.get(key, 0.0)
            responses[f"total_{key}"] = current_responses[key] + future_responses.get(
                key, 0.0
            )

        return responses, {}
