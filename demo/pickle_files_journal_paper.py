"""This script will produce the pickle files for the numerical results present in the.

journal paper.

TODO: Rewrite this so that we load in the same JSON but run on every solver, to reduce
the number of.
"""

import hashlib
import random
import sys
from pathlib import Path
from sys import argv

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the ProblemSolver class and other useful functions
import contextlib

from simopt.experiment_base import (
    Problem,
    ProblemSolver,
    instantiate_problem,
    instantiate_solver,
)
from simopt.solvers.astromorf import PolyBasisType

#! This needs to be populated from the results found through hyperparameter search
problems_optimal_hyper: dict = {
    "DYNAMNEWS-1": {"subspace_dimension": 1, "polynomial_degree": 2},  # Optimal
    "FACSIZE-1": {"subspace_dimension": 2, "polynomial_degree": 2},  # Optimal
    "CONTAM-2": {"subspace_dimension": 1, "polynomial_degree": 4},  # Optimal
    "ROSENBROCK-1": {"subspace_dimension": 9, "polynomial_degree": 2},  # Optimal
    "NETWORK-1": {"subspace_dimension": 6, "polynomial_degree": 4},  # optimal
    "FIXEDSAN-1": {"subspace_dimension": 1, "polynomial_degree": 2},  # Optimal
}

solver_renames = {
    "ASTROMORF": "ASTROMoRF",
    "OMoRF": "OMoRF",
    "ADAM": "ADAM",
    "ASTRODF": "ASTRO-DF",
    "NELDMD": "NELDER MEAD",
    "RNDSRCH": "RANDOM SEARCH",
    "STRONG": "STRONG",
}

poly_basis = {
    "HERMITE": PolyBasisType.HERMITE,
    "LEGENDRE": PolyBasisType.LEGENDRE,
    "CHEBYSHEV": PolyBasisType.CHEBYSHEV,
    "MONOMIAL": PolyBasisType.MONOMIAL,
    "NATURAL": PolyBasisType.NATURAL,
    "MONOMIAL_POLY": PolyBasisType.MONOMIAL_POLY,
    "LAGUERRE": PolyBasisType.LAGUERRE,
    "NFPOLY": PolyBasisType.NFP,
    "LAGRANGE": PolyBasisType.LAGRANGE,
}


def main(
    solver_name: str,
    problem_name: str,
    solver_factors: dict,
    budget: int,
    macroreplication_no: int,
    dim_size: int | None = None,
) -> None:
    """Run an experiment of a solver on a problem and store in a .pickle file.

    Args:
        solver_name (str): _name of the solver to run the experiment with.
        problem_name (str): _name of the problem to run the experiment on.
        solver_factors (dict): _fixed factors for the solver to run the experiment with.
        budget (int): _budget for the problem to run the experiment with.
        macroreplication_no (int): the number of macroreplications to run.
        dim_size (int | None, optional): The subspace dimension for the solver.
          Defaults to None.
    """
    # create multiple processes each on a different solver name
    file_name_path = run_experiment(
        solver_name, problem_name, dim_size, macroreplication_no, solver_factors, budget
    )
    print(f"SAVED AT {file_name_path}")


def run_experiment(
    solver_name: str,
    problem_name: str,
    dim_size: int | None,
    macroreplication_no: int,
    solver_factors: dict,
    budget: int,
) -> str:
    """Run an experiment of a solver on a problem and store the results in a .

    pickle file.

    Args:
        solver_name (str): Name of the solver
        problem_name (str): Name of the problem
        dim_size (int): Subspace Dimension for the solver (if applicable)
        macroreplication_no (int): Number of macroreplications to run
        solver_factors (dict): Fixed factors for the solver
        budget (int): Budget for the problem
    """
    file_name_path = f"{solver_name}_on_{problem_name}_budget{budget}_crn{solver_factors['crn_across_solns']}"  # noqa: E501

    # if polynomial basis is specified then add to the file name path
    if solver_factors["polynomial basis"] is not None:
        file_name_path += f"_basis{solver_factors['polynomial basis']}"

    # if initial subspace dimension is specified then add to the file name path
    if dim_size is not None:
        file_name_path += f"_dim{dim_size}"

    file_name_path += ".pickle"

    problem_dim = 100

    # If both dim_size and solver_factors['initial subspace dimension'] are None, set
    # both to optimal values from hyperparameter search
    if dim_size is None and solver_factors.get("initial subspace dimension") is None:
        if solver_name == "ASTROMoRF" or solver_name == "OMoRF":
            solver_factors["initial subspace dimension"] = problems_optimal_hyper[
                problem_name
            ]["subspace_dimension"]
            solver_factors["polynomial degree"] = problems_optimal_hyper[problem_name][
                "polynomial_degree"
            ]

    # If dim_size is given but solver_factors['initial subspace dimension'] is None, set
    # initial subspace dimension to dim_size and polynomial degree to optimal from
    # hyperparameter search
    elif solver_factors["polynomial degree"] is None:
        if solver_name == "ASTROMoRF" or solver_name == "OMoRF":
            solver_factors["initial subspace dimension"] = dim_size
            solver_factors["polynomial degree"] = problems_optimal_hyper[problem_name][
                "polynomial_degree"
            ]

    # if solver_factors['polynomial basis'] is given but dim_size is None, set dim_size
    # to initial subspace dimension and polynomial degree to optimal from hyperparameter
    # search
    elif dim_size is None:
        if solver_name == "ASTROMoRF" or solver_name == "OMoRF":
            solver_factors["initial subspace dimension"] = problems_optimal_hyper[
                problem_name
            ]["subspace_dimension"]
            solver_factors["polynomial basis"] = poly_basis[
                solver_factors["polynomial basis"]
            ]

    # If both are given, set both to given values
    else:
        if solver_name == "ASTROMoRF" or solver_name == "OMoRF":
            solver_factors["initial subspace dimension"] = dim_size
            solver_factors["polynomial basis"] = poly_basis[
                solver_factors["polynomial basis"]
            ]

    solver = None
    problem = None
    while solver is None:
        with contextlib.suppress(ValueError):
            solver = instantiate_solver(
                solver_name=solver_name,
                fixed_factors=solver_factors,
                solver_rename=solver_renames[solver_name],
            )

    while problem is None:
        with contextlib.suppress(ValueError):
            problem = scale_dimension(problem_name, budget, dimension=problem_dim)

    myexperiment = ProblemSolver(
        problem=problem, solver=solver, file_name_path=file_name_path
    )

    na = "N/A"
    dim_size = solver.factors.get("initial subspace dimension", na)
    poly_basis_name = solver_factors.get("polynomial basis", na)
    print(
        f"Running {solver_name} on {problem_name} with "
        f"budget {budget}, dim_size {dim_size}, "
        f"polynomial basis {poly_basis_name} for "
        f"{macroreplication_no} macroreplications."
    )
    myexperiment.run(n_macroreps=macroreplication_no)

    return file_name_path


SCALABLE_PROBLEMS = [
    "VANRYZIN-1",
    "FIXEDSAN-1",
    "NETWORK-1",
    "ROSENBROCK-1",
    "SAN-1",
    "DYNAMNEWS-1",
    "FACSIZE-2",
    "FACSIZE-1",
    "CONTAM-2",
]


def scale_dimension(
    problem_name: str, budget: int, dimension: int | None = None
) -> Problem:
    """Instantiate a problem with a scaled dimension.

    All model and problem factors that depend on the dimension are updated
    before instantiation to ensure consistency.

    Args:
        problem_name: The abbreviated name of the problem (e.g., "FACSIZE-2")
        budget: The simulation budget to use for the problem
        dimension: The desired dimension for the problem

    Returns:
        A Problem instance configured for the specified dimension
    """
    if problem_name not in SCALABLE_PROBLEMS or dimension is None:
        # For non-scalable problems, just instantiate with defaults
        # check initial objective funciton value of initial solution
        return instantiate_problem(problem_name, {"budget": budget})

    # Build the factors for the new dimension
    model_factors = get_scaled_model_factors(problem_name, dimension)
    problem_factors = get_scaled_problem_factors(problem_name, dimension)
    problem_factors["budget"] = budget

    # Instantiate the problem with the scaled factors
    problem = instantiate_problem(
        problem_name,
        problem_fixed_factors=problem_factors,
        model_fixed_factors=model_factors,
    )

    # Set the problem dimension explicitly
    problem.dim = dimension

    # Post-initialization updates for
    # factors that can't be validated during construction
    post_init_updates(problem, problem_name, dimension)

    return problem


def get_scaled_model_factors(problem_name: str, dimension: int) -> dict:
    """Generate model factors scaled to the specified dimension.

    Args:
        problem_name: The abbreviated name of the problem
        dimension: The target dimension

    Returns:
        Dictionary of model factors appropriate for the dimension
    """
    # Deterministic RNGs based on problem name + dimension
    seed = int(
        hashlib.sha256(f"{problem_name}:{dimension}".encode()).hexdigest(), 16
    ) % (2**32)
    rng_py = random.Random(seed)

    if problem_name == "DYNAMNEWS-1":
        return {
            "num_prod": dimension,
            "c_utility": [float(6 + j) for j in range(dimension)],
            "init_level": [3] * dimension,
            "price": [9.0] * dimension,
            "cost": [5.0] * dimension,
        }

    # dimension is increasing the number of ODF classes -- currently set to 6
    if problem_name in ("VANRYZIN-1"):
        # Scale VANRYZIN by number of ODF classes (products). "dimension" here
        # refers to the number of ODF classes. Choose a number of flight legs
        # that divides the number of classes so that n_virtual_classes =
        # dimension // n_legs is an integer. Use deterministic RNG for repeatability.
        possible_flights = [i for i in range(2, dimension + 1) if dimension % i == 0]
        if len(possible_flights) == 0:
            no_of_flights = 1
        else:
            no_of_flights = rng_py.choice(possible_flights)
        n_virtual = max(1, dimension // no_of_flights)

        # Build ODF matrix (n_classes x n_legs)
        odf = []
        for _ in range(dimension):
            row = [rng_py.randint(0, 1) for _ in range(no_of_flights)]
            if sum(row) == 0:
                row[rng_py.randint(0, no_of_flights - 1)] = 1
            odf.append(row)

        capacity = [float(rng_py.randint(100, 600)) for _ in range(no_of_flights)]
        fares = [float(rng_py.uniform(50, 500)) for _ in range(dimension)]

        # Virtual class indexing. 0 if product doesn't use leg,
        # otherwise an integer in [1, n_virtual]
        vc_index = []
        for j in range(dimension):
            row = []
            for l in range(no_of_flights):  # noqa: E741
                if odf[j][l] == 1:
                    row.append(rng_py.randint(1, n_virtual))
                else:
                    row.append(0)
            vc_index.append(row)

        # Protection levels
        prot = []
        for l_idx in range(no_of_flights):
            cap = capacity[l_idx]
            vals = sorted(rng_py.uniform(0.0, cap) for _ in range(n_virtual))
            prot.append([float(v) for v in vals])

        return {
            "ODF_leg_matrix": odf,
            "n_virtual_classes": n_virtual,
            "capacity": capacity,
            "n_classes": dimension,
            "fares": fares,
            "virtual_class_indexing": vc_index,
            "protection_levels": prot,
            "gamma_shape": tuple([2.0] * dimension),
            "gamma_scale": tuple([50.0] * dimension),
            "beta_alpha": tuple([2.0] * dimension),
            "beta_beta": tuple([1.0] * dimension),
        }

    if problem_name in ("FACSIZE-1", "FACSIZE-2"):
        # Use diagonal covariance to avoid expensive Cholesky and reduce rejection rate
        # With mean=500 and std=50 (variance=2500), P(X<0) ≈ 0 for each dimension
        # This makes rejection sampling nearly instant
        variance = 2500.0  # std = 50, mean = 500, so P(X<0) is negligible
        cov_matrix = np.eye(dimension) * variance
        return {
            "mean_vec": [500.0] * dimension,
            "cov": cov_matrix.tolist(),
            "capacity": [float(rng_py.randint(100, 900)) for _ in range(dimension)],
            "n_fac": dimension,
        }

    if problem_name == "SAN-1":
        # Calculate appropriate num_nodes for the number of edges (dimension)
        # For a DAG: we need num_nodes such that we can have 'dimension' edges
        # with a path from node 1 to num_nodes
        num_nodes = compute_num_nodes_for_dag(dimension)
        arcs = build_san_dag(num_nodes, dimension, rng=rng_py)
        return {
            "num_arcs": dimension,
            "num_nodes": num_nodes,
            "arcs": arcs,
            "arc_means": tuple(
                round(rng_py.uniform(1, 10), 2) for _ in range(dimension)
            ),
        }

    if problem_name == "FIXEDSAN-1":
        num_nodes = max(2, rng_py.randint(2, max(2, dimension)))
        return {
            "num_arcs": dimension,
            "num_nodes": num_nodes,
            "arc_means": tuple(float(rng_py.randint(1, 10)) for _ in range(dimension)),
        }

    if problem_name == "ROSENBROCK-1":
        return {
            "x": (2.0,) * dimension,
            "variance": 0.4,
        }
    if problem_name == "ZAKHAROV-1":
        return {
            "x": (2.0,) * dimension,
            "variance": 0.1,
        }

    if problem_name == "NETWORK-1":
        process_prob_elem = 1.0 / dimension
        mode_transit_time = [
            round(rng_py.uniform(0.01, 5), 3) for _ in range(dimension)
        ]
        return {
            "process_prob": [process_prob_elem] * dimension,
            "cost_process": [0.1 / (x + 1) for x in range(dimension)],
            "cost_time": [round(rng_py.uniform(0.01, 1), 3) for _ in range(dimension)],
            "mode_transit_time": mode_transit_time,
            "lower_limits_transit_time": [x / 2 for x in mode_transit_time],
            "upper_limits_transit_time": [2 * x for x in mode_transit_time],
            "n_networks": dimension,
        }

    if problem_name == "CONTAM-2":
        return {
            "stages": dimension,
            "prev_decision": (0.0,) * dimension,
        }

    return {}


def get_scaled_problem_factors(problem_name: str, dimension: int) -> dict:
    """Generate problem factors scaled to the specified dimension.

    Only includes factors that will pass validation during construction.
    Factors that depend on model state are updated post-initialization.

    Args:
        problem_name: The abbreviated name of the problem
        dimension: The target dimension

    Returns:
        Dictionary of problem factors appropriate for the dimension
    """
    if problem_name == "DYNAMNEWS-1":
        return {
            "initial_solution": (3.0,) * dimension,
        }

    if problem_name in ("FACSIZE-1", "FACSIZE-2"):
        # NOTE: installation_costs is validated against NUM_FACILITIES constant (=3)
        # So we can't pass it here - it will be updated post-initialization
        return {
            "initial_solution": (100.0,) * dimension,
            "installation_budget": 500.0 * (dimension / 3),
        }

    if problem_name in ("SAN-1", "FIXEDSAN-1"):
        # NOTE: arc_costs is validated against NUM_ARCS constant (=13)
        # So we can't pass it here - it will be updated post-initialization
        return {
            "initial_solution": (1.0,) * dimension,
        }

    if problem_name in ("VANRYZIN-1"):
        # Construct a sensible initial_solution (flattened protection levels)
        # using the same deterministic RNG logic as the model factors so
        # that the initial solution is consistent with the generated model.
        seed = int(
            hashlib.sha256(f"{problem_name}:{dimension}".encode()).hexdigest(), 16
        ) % (2**32)
        rng_py = random.Random(seed)
        possible_flights = [i for i in range(1, dimension + 1) if dimension % i == 0]
        if len(possible_flights) == 0:
            no_of_flights = 1
        else:
            no_of_flights = rng_py.choice([i for i in possible_flights if i >= 1])
        n_virtual = max(1, dimension // no_of_flights)

        # Build protection levels per leg and flatten in leg-major order
        prot = []
        for _l in range(no_of_flights):
            cap = float(rng_py.randint(100, 600))
            vals = sorted(rng_py.uniform(0.0, cap) for _ in range(n_virtual))
            prot.append([float(v) for v in vals])

        # Flatten leg-major: (y_{0,1},...,y_{0,K}, y_{1,1},...,y_{L-1,K})
        flat = tuple(prot[l][k] for l in range(len(prot)) for k in range(len(prot[l])))  # noqa: E741
        return {
            "initial_solution": flat,
        }

    if problem_name == "ROSENBROCK-1" or problem_name == "ZAKHAROV-1":
        return {
            "initial_solution": (2.0,) * dimension,
        }

    if problem_name == "NETWORK-1":
        return {
            "initial_solution": (1.0 / dimension,) * dimension,
        }

    if problem_name == "CONTAM-2":
        return {
            "initial_solution": (0.0,) * dimension,
        }

    return {}


def post_init_updates(problem: Problem, problem_name: str, dimension: int) -> None:
    """Update problem factors after initialization.

    Some factors are validated against hardcoded constants during construction,
    so they need to be updated after the problem is instantiated.

    Args:
        problem: The problem instance to update
        problem_name: The abbreviated name of the problem
        dimension: The target dimension
    """
    if problem_name in ("FACSIZE-1", "FACSIZE-2"):
        # Update installation_costs after construction to match the new dimension
        problem.factors["installation_costs"] = (1.0,) * dimension

    elif problem_name in ("SAN-1", "FIXEDSAN-1"):
        # Update arc_costs after construction to match the new dimension
        # arc_costs is used in replicate(): np.sum(arc_costs / x)
        problem.factors["arc_costs"] = (1.0,) * dimension

    elif problem_name in ("VANRYZIN-1", "VANRYZIN-2"):
        # Ensure initial_solution length matches n_legs * n_virtual_classes
        capacity = problem.model.factors["capacity"]
        n_virtual = int(problem.model.factors["n_virtual_classes"])
        n_virtual = max(1, n_virtual)
        prot = []
        for c in capacity:
            row = [float((k + 1) * c / (n_virtual + 1)) for k in range(n_virtual)]
            prot.append(row)
        flat = tuple(v for row in prot for v in row)
        problem.factors["initial_solution"] = flat


def compute_num_nodes_for_dag(num_edges: int) -> int:
    """Compute an appropriate number of nodes for a DAG with the given number of edges.

    For a DAG with n nodes where we need a path from 1 to n:
    - Minimum edges needed: n-1 (a simple path)
    - Maximum edges possible: n*(n-1)/2 (complete DAG)

    We want to find the smallest n such that n*(n-1)/2 >= num_edges
    and n-1 <= num_edges (so we have enough edges for connectivity).

    Args:
        num_edges: Desired number of edges

    Returns:
        Number of nodes to use
    """
    # We need at least num_edges + 1 nodes in the worst case (simple path),
    # but we want fewer nodes with more edges between them.
    # Solve: n*(n-1)/2 >= num_edges => n^2 - n - 2*num_edges >= 0
    # n >= (1 + sqrt(1 + 8*num_edges)) / 2

    import math

    min_nodes = math.ceil((1 + math.sqrt(1 + 8 * num_edges)) / 2)

    # Ensure we have at least 2 nodes and the path is possible
    min_nodes = max(2, min_nodes)

    # Also ensure num_edges >= min_nodes - 1 (need at least a spanning path)
    # If not, we need more nodes
    while min_nodes - 1 > num_edges:
        min_nodes -= 1

    return min_nodes


def build_san_dag(
    num_nodes: int, num_edges: int, rng: random.Random | None = None
) -> list[tuple[int, int]]:
    """Build a directed acyclic graph (DAG) suitable for the SAN model.

    The SAN model requires:
    1. Directed edges (arcs) from lower-numbered to higher-numbered nodes
    2. A path must exist from node 1 to node num_nodes
    3. Every node must be reachable from node 1 (for backtracking to work)

    This function first creates a simple sequential path from 1 to num_nodes
    (1→2→3→...→n), then adds additional random forward edges until reaching num_edges.

    Args:
        num_nodes: Number of nodes (numbered 1 to num_nodes)
        num_edges: Desired number of directed edges
        rng: Optional random number generator for reproducibility

    Returns:
        List of (source, target) tuples representing directed edges

    Raises:
        ValueError: If the requested configuration is impossible
    """
    min_edges = num_nodes - 1  # Simple path from 1 to num_nodes
    max_edges = num_nodes * (num_nodes - 1) // 2  # Complete DAG

    if num_edges < min_edges:
        raise ValueError(
            f"Cannot create DAG with path 1→{num_nodes}: "
            f"need at least {min_edges} edges, but only {num_edges} requested"
        )

    if num_edges > max_edges:
        raise ValueError(
            f"Cannot create DAG with {num_edges} edges: maximum possible is "
            f"{max_edges} for {num_nodes} nodes"
        )

    edges = set()

    # Step 1: Create a guaranteed SEQUENTIAL path from node 1 to node num_nodes
    # This ensures every node has a predecessor reachable from node 1
    # Path: 1 → 2 → 3 → ... → num_nodes
    for i in range(1, num_nodes):
        edges.add((i, i + 1))

    # Step 2: Add additional random forward edges until we reach num_edges
    if len(edges) < num_edges:
        if rng is None:
            rng = random.Random()
        # Generate all possible forward edges not yet in the graph
        all_possible_edges = []
        for i in range(1, num_nodes):
            for j in range(i + 1, num_nodes + 1):
                edge = (i, j)
                if edge not in edges:
                    all_possible_edges.append(edge)

        # Shuffle and add as many as needed
        rng.shuffle(all_possible_edges)
        edges_needed = num_edges - len(edges)

        for edge in all_possible_edges[:edges_needed]:
            edges.add(edge)

    return list(edges)


def validate_solver_and_problem_names(solver_name: str, problem_name: str) -> None:
    """Pre-flight validation for clearer errors before heavy work."""
    try:
        _ = instantiate_solver(
            solver_name=solver_name, fixed_factors={}, solver_rename=solver_name
        )
    except Exception as e:
        raise ValueError(
            f"Unknown or invalid solver code '{solver_name}'. Original error: {e}"
        ) from e

    try:
        _ = instantiate_problem(
            problem_name, problem_fixed_factors=None, model_fixed_factors=None
        )
    except Exception as e:
        raise ValueError(
            f"Unknown problem code '{problem_name}'. Original error: {e}"
        ) from e


def build_connected_graph(
    num_nodes: int, num_edges: int, rng: random.Random | None = None
) -> list[tuple[int, int]]:
    """Build a connected graph with the specified number of nodes and edges.

    Starts with a spanning tree to ensure connectivity, then adds random edges
    until the desired number is reached.
    """
    if num_edges < num_nodes - 1:
        raise ValueError(
            f"Cannot create connected graph: need at least {num_nodes - 1} "
            f"edges for {num_nodes} nodes, but only {num_edges} requested"
        )

    edges = set()

    if rng is None:
        rng = random.Random()

    # Create a spanning tree first to ensure connectivity
    nodes = list(range(num_nodes))
    rng.shuffle(nodes)
    for i in range(1, num_nodes):
        a = nodes[i]
        b = nodes[rng.randint(0, i - 1)]
        edges.add((min(a, b), max(a, b)))

    # Add random edges until we reach the desired number
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    target_edges = min(num_edges, max_possible_edges)

    attempts = 0
    max_attempts = target_edges * 10  # Prevent infinite loop
    while len(edges) < target_edges and attempts < max_attempts:
        a = rng.randint(0, num_nodes - 1)
        b = rng.randint(0, num_nodes - 1)
        if a != b:
            edges.add((min(a, b), max(a, b)))
        attempts += 1

    return list(edges)


def load_json(json_path: str) -> list[str]:
    """Load configuration from a JSON file to get solver names.

    Args:
        json_path (str): Path to the JSON configuration file.

    Returns:
        list[str]: A list of solver names.
    """
    import json

    with open(json_path) as f:  # noqa: PTH123
        config = json.load(f)

    # Extract individual variables if needed
    return config.get("solver_names", [])


if __name__ == "__main__":
    # Pass arguments
    solver_name = argv[1]
    problem_name = argv[2]
    dim_size = None if argv[3].lower() == "none" else int(argv[3])
    solver_factors = eval(argv[4])
    budget = int(argv[5])
    macroreplication_no = int(argv[6])

    diag = {
        "solver name": solver_name,
        "problem name": problem_name,
        "problem dimension": dim_size,
        "solver fixed factors": solver_factors,
        "simulation budget": budget,
        "number of macroreplications": macroreplication_no,
    }

    if dim_size is not None:
        main(
            solver_name,
            problem_name,
            solver_factors,
            budget,
            macroreplication_no,
            dim_size=dim_size,
        )
    else:
        main(solver_name, problem_name, solver_factors, budget, macroreplication_no)
