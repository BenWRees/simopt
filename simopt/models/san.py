"""Simulate duration of a stochastic activity network (SAN)."""

from __future__ import annotations

import math
from collections import deque
from typing import Annotated, ClassVar, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    StochasticConstraint,
    VariableType,
)
from simopt.input_models import Exp

NUM_ARCS: Final[int] = 13
CONST_NODES: Final[list[int]] = [6, 8]


def _build_layered_dag(new_dim: int) -> tuple[list[tuple[int, int]], int]:
    """Construct a layered DAG with exactly ``new_dim`` distinct forward arcs.

    Layout: ``source = 1`` -> ``L`` middle layers of width ``W`` -> ``sink``.
    Arcs only go between consecutive layers (or source->layer-1 / layer-L->sink),
    so the result is acyclic by construction. We emit arcs in two passes:

      1. **Skeleton** (size ``W*(L+1)``) -- source-to-layer-1, one "rail" per
         layer transition (i-th node -> i-th node), and last-layer-to-sink.
         These guarantee every node has at least one in- and one out-arc and
         that every node sits on a source-sink path.
      2. **Diagonals** -- for each transition and offset ``o = 1..W-1``,
         emit ``(layer_k[i] -> layer_{k+1}[(i+o) mod W])``. These give the
         multiple competing paths that distinguish a true SAN longest-path
         problem from a single chain.

    Skeleton then diagonals are concatenated and truncated to ``new_dim``.
    Width is chosen so the skeleton always fits: ``W=3`` for ``dim>=9``,
    ``W=2`` for ``6<=dim<=8``, otherwise a chain (``W=1``).

    Returns:
        ``(arcs, num_nodes)`` -- ``arcs`` is the ordered list of decision
        variables; ``arcs[i]`` is uniquely the i-th decision variable.
    """
    if new_dim < 2:
        raise ValueError(f"new_dim must be >= 2 for SAN-1, got {new_dim!r}.")

    # Pick the largest width whose skeleton fits inside `new_dim` and whose
    # candidate-arc count covers `new_dim`. Falls through to chain if neither
    # W=3 nor W=2 fits (this only happens for new_dim in {2, 3, 5}).
    width = 0
    chosen_layers = 0
    for w in (3, 2):
        if new_dim < 2 * w:
            continue
        layers = max(1, math.ceil((new_dim - 2 * w) / (w * w)) + 1)
        skeleton_size = w * (layers + 1)
        total_size = 2 * w + (layers - 1) * w * w
        if skeleton_size <= new_dim <= total_size:
            width = w
            chosen_layers = layers
            break

    if width == 0:
        # Chain: 1 -> 2 -> ... -> new_dim+1.  No multi-path structure, but
        # this branch only fires for very small dimensions where a layered
        # construction can't accommodate the skeleton.
        arcs = [(i, i + 1) for i in range(1, new_dim + 1)]
        return arcs, new_dim + 1

    source = 1
    next_id = 2
    layer_nodes: list[list[int]] = []
    for _ in range(chosen_layers):
        layer_nodes.append(list(range(next_id, next_id + width)))
        next_id += width
    sink = next_id
    num_nodes = sink

    skeleton: list[tuple[int, int]] = []
    skeleton.extend((source, v) for v in layer_nodes[0])
    for k in range(chosen_layers - 1):
        skeleton.extend(
            (layer_nodes[k][i], layer_nodes[k + 1][i]) for i in range(width)
        )
    skeleton.extend((v, sink) for v in layer_nodes[-1])

    extras: list[tuple[int, int]] = []
    for k in range(chosen_layers - 1):
        for offset in range(1, width):
            for i in range(width):
                extras.append(
                    (layer_nodes[k][i], layer_nodes[k + 1][(i + offset) % width])
                )

    arcs = (skeleton + extras)[:new_dim]
    return arcs, num_nodes


class SANConfig(BaseModel):
    """Configuration for the Stochastic Activity Network model."""

    num_nodes: Annotated[
        int,
        Field(
            default=9,
            description="number of nodes",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    arcs: Annotated[
        list[tuple[int, int]],
        Field(
            default=[
                (1, 2),
                (1, 3),
                (2, 3),
                (2, 4),
                (2, 6),
                (3, 6),
                (4, 5),
                (4, 7),
                (5, 6),
                (5, 8),
                (6, 9),
                (7, 8),
                (8, 9),
            ],
            description="list of arcs",
            min_length=1,
        ),
    ]
    arc_means: Annotated[
        tuple[float, ...],
        Field(
            default=(8.0,) * NUM_ARCS,
            description="mean task durations for each arc",
        ),
    ]

    def __dfs(
        self, graph: dict[int, set], start: int, visited: set | None = None
    ) -> set:
        if visited is None:
            visited = set()
        visited.add(start)

        for next_point in graph[start] - visited:
            self.__dfs(graph, next_point, visited)
        return visited

    def _check_arcs(self) -> None:
        if len(self.arcs) <= 0:
            raise ValueError("The length of arcs must be greater than 0.")
        # Check graph is connected.
        graph = {node: set() for node in range(1, self.num_nodes + 1)}
        for a in self.arcs:
            graph[a[0]].add(a[1])
        visited = self.__dfs(graph, 1)

        if self.num_nodes not in visited:
            raise ValueError("Graph must be connected from node 1 to the final node.")

    def _check_arc_means(self) -> None:
        positive = True
        for x in list(self.arc_means):
            positive = positive and (x > 0)
        if not positive:
            raise ValueError("All elements in arc_means must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_arcs()
        self._check_arc_means()
        if len(self.arc_means) != len(self.arcs):
            raise ValueError(
                "The length of arc_means must be equal to the length of arcs."
            )
        return self


class SANLongestPathConfig(BaseModel):
    """Configuration model for SAN Longest Path Problem.

    Min Mean Longest Path for Stochastic Activity Network
    simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (8.0,) * NUM_ARCS,
            description="initial solution",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=1000,
            description="max # of replications for a solver to take",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    arc_costs: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (1,) * NUM_ARCS,
            description="Cost associated to each arc.",
        ),
    ]

    def _check_arc_costs(self) -> None:
        # Length consistency with `arcs` is enforced at the Problem level
        # (see SANLongestPath.validate_scaled). Hardcoded NUM_ARCS check
        # removed so the problem can be constructed at non-default dimensions.
        if any(x <= 0 for x in self.arc_costs):
            raise ValueError("All elements in arc_costs must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_arc_costs()
        return self


class SAN(Model):
    """Stochastic Activity Network (SAN) Model.

    A model that simulates a stochastic activity network problem with
    tasks that have exponentially distributed durations, and the selected
    means come with a cost.
    """

    class_name_abbr: ClassVar[str] = "SAN"
    class_name: ClassVar[str] = "Stochastic Activity Network"
    config_class: ClassVar[type[BaseModel]] = SANConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the SAN model.

        Args:
            fixed_factors : dict
                fixed factors of the simulation model
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.time_model = Exp()

    def __dfs(
        self, graph: dict[int, set], start: int, visited: set | None = None
    ) -> set:
        if visited is None:
            visited = set()
        visited.add(start)

        for next_point in graph[start] - visited:
            self.__dfs(graph, next_point, visited)
        return visited

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.time_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "longest_path_length": Length or duration of the longest path.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        num_nodes: int = self.factors["num_nodes"]
        arcs: list[tuple[int, int]] = self.factors["arcs"]
        arc_means: tuple[int, ...] = self.factors["arc_means"]

        # Topological sort.
        node_range = range(1, num_nodes + 1)
        graph_in = {node: set() for node in node_range}
        graph_out = {node: set() for node in node_range}
        for start, end in arcs:
            graph_in[end].add(start)
            graph_out[start].add(end)

        indegrees = [len(graph_in[n]) for n in node_range]
        # outdegrees = [len(graph_out[n]) for n in node_range]
        queue = deque(n for n in node_range if indegrees[n - 1] == 0)
        topo_order = []
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in graph_out[u]:
                indegrees[v - 1] -= 1
                if indegrees[v - 1] == 0:
                    queue.append(v)

        # Arc lengths
        arc_length = {
            arc: self.time_model.random(1 / arc_means[i]) for i, arc in enumerate(arcs)
        }

        # Longest path
        path_length = np.zeros(num_nodes)
        prev = np.full(num_nodes, -1)
        for vi in topo_order:
            for j in graph_out[vi]:
                new_len = path_length[vi - 1] + arc_length[(vi, j)]
                if new_len > path_length[j - 1]:
                    path_length[j - 1] = new_len
                    prev[j - 1] = vi

        longest_path = path_length[-1]

        # Calculate the IPA gradient w.r.t. arc means.
        # If an arc is on the longest path, the component of the gradient
        # is the length of the length of that arc divided by its mean.
        # If an arc is not on the longest path, the component of the gradient is zero.
        arc_to_index = {arc: i for i, arc in enumerate(arcs)}

        grads = []
        for node in topo_order:
            gradient = np.zeros(len(arcs))
            current = node
            backtrack = int(prev[node - 1])

            while current != topo_order[0]:
                arc = (backtrack, current)
                idx = arc_to_index[arc]
                gradient[idx] = arc_length[arc] / arc_means[idx]
                current = backtrack
                backtrack = int(prev[backtrack - 1])

            grads.append(gradient)

        # Compose responses and gradients.
        responses = {
            "longest_path_length": longest_path,
            "longest_path_to_all_nodes": path_length,
            "topo_order": topo_order,
        }
        gradients = {
            response_key: {
                factor_key: np.zeros(len(self.specifications))
                for factor_key in self.specifications
            }
            for response_key in responses
        }
        gradients["longest_path_length"]["arc_means"] = grads[-1]
        gradients["longest_path_to_all_nodes"]["arc_means"] = np.array(grads)
        return responses, gradients


class SANLongestPath(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "SAN-1"
    class_name: ClassVar[str] = "Min Mean Longest Path for Stochastic Activity Network"
    config_class: ClassVar[type[BaseModel]] = SANLongestPathConfig
    model_class: ClassVar[type[Model]] = SAN
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"arc_means"}
    _dim: int | None = None

    @property
    def dim(self) -> int:  # noqa: D102
        if self._dim is not None:
            return self._dim
        return len(self.model.factors["arcs"])

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (1e-2,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"arc_means": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return factor_dict["arc_means"]

    def replicate(self, x: tuple) -> RepResult:  # noqa: D102
        responses, gradients = self.model.replicate()
        objectives = [
            Objective(
                stochastic=responses["longest_path_length"],
                stochastic_gradients=gradients["longest_path_length"]["arc_means"],
                deterministic=np.sum(np.array(self.factors["arc_costs"]) / np.array(x)),
                deterministic_gradients=-np.array(self.factors["arc_costs"])
                / (np.array(x) ** 2),
            )
        ]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return all(x_i >= 0 for x_i in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple(
            [rand_sol_rng.lognormalvariate(lq=0.1, uq=10) for _ in range(self.dim)]
        )

    @classmethod
    def scale_to(cls, new_dim: int, budget: int) -> Problem:
        """Build SAN-1 at a target arc count using a layered DAG topology.

        Why a layered DAG (source -> L layers of width W -> sink):
          * Produces ``new_dim`` *distinct* arcs (no dict collisions in
            :meth:`SAN.replicate`); each arc is one decision variable.
          * Multiple parallel source-sink paths -- the longest-path
            competition that defines the SAN problem -- so every arc has
            positive probability of being on the realized longest path and
            therefore positive expected IPA gradient.
          * Bounded depth (``L = O(new_dim / W^2)`` with ``W=3``), so the
            longest-path objective stays of reasonable magnitude as
            ``new_dim`` grows; the noise level scales like ``sqrt(L)``.

        See :func:`_build_layered_dag` for the construction.  Mean / cost /
        initial values are propagated uniformly from the native defaults
        (8.0 mean, 1.0 cost, 8.0 initial), preserving per-arc scale.
        """
        from simopt.experiment_base import instantiate_problem

        if new_dim <= 0:
            raise ValueError(f"new_dim must be positive, got {new_dim!r}")

        arcs, num_nodes = _build_layered_dag(new_dim)
        arc_means = (8.0,) * new_dim
        arc_costs = (1.0,) * new_dim
        initial_solution = (8.0,) * new_dim

        return instantiate_problem(
            "SAN-1",
            problem_fixed_factors={
                "budget": budget,
                "initial_solution": initial_solution,
                "arc_costs": arc_costs,
            },
            model_fixed_factors={
                "num_nodes": num_nodes,
                "arcs": arcs,
                "arc_means": arc_means,
            },
        )

    def validate_scaled(self, expected_dim: int) -> None:
        """Structural sanity checks consumed by ``scale_dimension``."""
        arcs = self.model.factors["arcs"]
        num_nodes = self.model.factors["num_nodes"]
        if len(arcs) != expected_dim:
            raise ValueError(
                f"SAN-1: arcs has length {len(arcs)}, expected {expected_dim}."
            )
        if len(set(arcs)) != len(arcs):
            raise ValueError(
                "SAN-1: scaled `arcs` contains duplicate edges -- the longest-path "
                "computation deduplicates via dict keys, which yields zero-gradient "
                "decision variables."
            )
        if len(self.model.factors["arc_means"]) != expected_dim:
            raise ValueError("SAN-1: arc_means length mismatch with arcs.")
        if len(self.factors["arc_costs"]) != expected_dim:
            raise ValueError("SAN-1: arc_costs length mismatch with arcs.")

        # Every arc must lie on at least one source(1) -> sink(num_nodes) path,
        # otherwise its decision variable cannot influence the longest path.
        forward: dict[int, set[int]] = {n: set() for n in range(1, num_nodes + 1)}
        backward: dict[int, set[int]] = {n: set() for n in range(1, num_nodes + 1)}
        for u, v in arcs:
            forward[u].add(v)
            backward[v].add(u)

        reachable_from_source: set[int] = set()
        stack = [1]
        while stack:
            u = stack.pop()
            if u in reachable_from_source:
                continue
            reachable_from_source.add(u)
            stack.extend(forward[u])

        reaches_sink: set[int] = set()
        stack = [num_nodes]
        while stack:
            u = stack.pop()
            if u in reaches_sink:
                continue
            reaches_sink.add(u)
            stack.extend(backward[u])

        dead_arcs = [
            (u, v)
            for (u, v) in arcs
            if u not in reachable_from_source or v not in reaches_sink
        ]
        if dead_arcs:
            raise ValueError(
                f"SAN-1: {len(dead_arcs)} arc(s) lie outside any source->sink "
                f"path, e.g. {dead_arcs[:3]}; these would have zero stochastic "
                f"gradient."
            )


class SANLongestPathStochasticConfig(BaseModel):
    """Configuration model for SAN Longest Path Stochastic Problem."""

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (8.0,) * NUM_ARCS,
            description="initial solution",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=1000,
            description="max # of replications for a solver to take",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    arc_costs: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (1.0,) * NUM_ARCS,
            description="Cost associated to each arc.",
        ),
    ]
    constraint_nodes: Annotated[
        list[int],
        Field(
            default_factory=lambda: CONST_NODES.copy(),
            description="Nodes with corresponding stochastic constraints.",
            min_length=1,
        ),
    ]
    length_to_node_constraint: Annotated[
        list[float],
        Field(
            default_factory=lambda: [5.0] * len(CONST_NODES),
            description="Max allowable length to each constraint node.",
            min_length=1,
        ),
    ]

    def _check_arc_costs(self) -> None:
        if len(self.arc_costs) != NUM_ARCS:
            raise ValueError(f"arc_costs must be of length {NUM_ARCS}.")
        if any(cost <= 0 for cost in self.arc_costs):
            raise ValueError("All elements in arc_costs must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_arc_costs()
        return self


class SANLongestPathStochastic(Problem):
    """Minimize total cost s.t. reaching certain nodes within an expected length."""

    class_name_abbr: ClassVar[str] = "SAN-2"
    class_name: ClassVar[str] = "Min Cost SAN with Stochastic Constraints"
    config_class: ClassVar[type[BaseModel]] = SANLongestPathStochasticConfig
    model_class: ClassVar[type[Model]] = SAN
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = len(CONST_NODES)
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.STOCHASTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"arc_means"}
    _dim: int | None = None

    @property
    def dim(self) -> int:  # noqa: D102
        if self._dim is not None:
            return self._dim
        return len(self.model.factors["arcs"])

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (1e-2,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (100.0,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"arc_means": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return factor_dict["arc_means"]

    def replicate(self, x: tuple) -> RepResult:  # noqa: D102
        responses, gradients = self.model.replicate()

        objectives = [
            Objective(
                stochastic=responses["longest_path_length"],
                stochastic_gradients=gradients["longest_path_length"]["arc_means"],
                deterministic=np.sum(np.array(self.factors["arc_costs"]) / np.array(x)),
                deterministic_gradients=-np.array(self.factors["arc_costs"])
                / (np.array(x) ** 2),
            )
        ]

        topo_order = responses["topo_order"]
        longest_path_nodes = responses["longest_path_to_all_nodes"]
        node_positions = {node: idx for idx, node in enumerate(topo_order)}
        arc_gradients = gradients["longest_path_to_all_nodes"]["arc_means"]
        constraint_limits = self.factors["length_to_node_constraint"]
        constraint_nodes = self.factors["constraint_nodes"]

        stochastic_constraints = []
        for i, const_node in enumerate(constraint_nodes):
            idx = node_positions[const_node]
            stochastic_value = longest_path_nodes[idx]
            stochastic_grad = arc_gradients[idx]
            deterministic_value = -constraint_limits[i]
            deterministic_grad = [0.0] * self.dim
            stochastic_constraints.append(
                StochasticConstraint(
                    stochastic_value,
                    stochastic_grad,
                    deterministic_value,
                    deterministic_grad,
                )
            )

        return RepResult(
            objectives=objectives,
            stochastic_constraints=stochastic_constraints,
        )

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return all(x_i >= 0 for x_i in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple(
            [rand_sol_rng.lognormalvariate(lq=0.1, uq=10) for _ in range(self.dim)]
        )
