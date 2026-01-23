"""Base classes for problem-solver pairs and I/O/plotting helper functions."""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING
import signal  


import numpy as np
from joblib import Parallel, delayed

import simopt.directory as directory
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
	ConstraintType,
	Model,
	ObjectiveType,
	Problem,
	Solution,
	Solver,
	VariableType,
)
from simopt.curve import Curve
from simopt.experiment import run_solver
from simopt.feasibility import feasibility_score_history
from simopt.utils import resolve_file_path

# Workaround for AutoAPI
model_directory = directory.model_directory
problem_directory = directory.problem_directory
solver_directory = directory.solver_directory


# Imports exclusively used when type checking
# Prevents imports from being executed at runtime
if TYPE_CHECKING:
	from typing import Literal

	from matplotlib.lines import Line2D as Line2D
	from pandas import DataFrame as DataFrame

# Setup the experiment directory
EXPERIMENT_DIR = Path.cwd() / "experiments" / time.strftime("%Y-%m-%d_%H-%M-%S")
# Make sure the experiment directory gets created
# TODO: move this into __init__


class ProblemSolver:
	"""Base class for running one solver on one problem."""

	def __init__(
		self,
		solver_name: str | None = None,
		problem_name: str | None = None,
		solver_rename: str | None = None,
		problem_rename: str | None = None,
		solver: Solver | None = None,
		problem: Problem | None = None,
		solver_fixed_factors: dict | None = None,
		problem_fixed_factors: dict | None = None,
		model_fixed_factors: dict | None = None,
		file_name_path: Path | str | None = None,
		create_pickle: bool = True,
	) -> None:
		"""Initializes a ProblemSolver object.

		You can either:
		1. Provide solver and problem names to look them up via `directory.py`, or
		2. Provide solver and problem objects directly.

		Args:
			solver_name (str, optional): Name of the solver.
			problem_name (str, optional): Name of the problem.
			solver_rename (str, optional): User-defined name for the solver.
			problem_rename (str, optional): User-defined name for the problem.
			solver (Solver, optional): Simulation-optimization solver object.
			problem (Problem, optional): Simulation-optimization problem object.
			solver_fixed_factors (dict, optional): Fixed solver parameters.
			problem_fixed_factors (dict, optional): Fixed problem parameters.
			model_fixed_factors (dict, optional): Fixed model parameters.
			file_name_path (Path | str, optional): Path to save a pickled ProblemSolver
				object.
			create_pickle (bool, optional): Whether to save the object as a pickle file.
		"""
		# Default arguments
		if solver_fixed_factors is None:
			solver_fixed_factors = {}
		if problem_fixed_factors is None:
			problem_fixed_factors = {}
		if model_fixed_factors is None:
			model_fixed_factors = {}

		# Resolve file name path
		if file_name_path is not None:
			# If the provided path contains a directory component (e.g. 'output_dir/name.pickle')
			# prefer resolving relative to the current working directory so user-specified
			# output folders are respected. Otherwise, fall back to EXPERIMENT_DIR.
			p = Path(file_name_path) if isinstance(file_name_path, (str, Path)) else None
			if p is not None and (p.is_absolute() or p.parent != Path('.')):
				file_name_path = resolve_file_path(file_name_path, directory=Path.cwd())
			else:
				file_name_path = resolve_file_path(file_name_path, directory=EXPERIMENT_DIR)

		# Initialize values
		self.create_pickle = create_pickle
		self.has_run = False
		self.has_postreplicated = False
		self.has_postnormalized = False
		self.xstar = ()
		self.x0 = ()
		self.objective_curves = []
		self.progress_curves = []
		self.all_stoch_constraints = []
		self.all_est_lhs = []
		self.feasibility_curves = []

		self.n_macroreps: int = 0
		self.file_name_path: Path
		self.all_recommended_xs: list[list[tuple]] = []
		self.all_intermediate_budgets: list[list] = []
		self.timings: list[float] = []
		self.n_postreps: int = 0
		self.crn_across_budget: bool = False
		self.crn_across_macroreps: bool = False
		self.all_post_replicates: list[list[list]] = []
		self.all_est_objectives: list[list[float]] = []
		self.n_postreps_init_opt: int = 0
		self.crn_across_init_opt: bool = False
		self.x0_postreps: list[float] = []
		self.xstar_postreps: list[float] = []

		# Initialize solver.
		if isinstance(solver, Solver):  # Method 2
			self.solver = solver
		else:  # Method 1
			if solver_name is None:
				error_msg = (
					"Solver name must be provided if solver object is not provided."
				)
				raise ValueError(error_msg)
			if solver_name not in solver_directory:
				error_msg = "Solver name not found in solver directory."
				raise ValueError(error_msg)
			self.solver = solver_directory[solver_name](
				fixed_factors=solver_fixed_factors
			)
		# Rename solver if necessary.
		if solver_rename is not None:
			if solver_rename == "":
				error_msg = "Solver rename cannot be an empty string."
				raise ValueError(error_msg)
			self.solver.name = solver_rename

		# Initialize problem.
		if isinstance(problem, Problem):  # Method #2
			self.problem = problem
		else:  # Method #1
			if problem_name is None:
				error_msg = (
					"Problem name must be provided if problem object is not provided."
				)
				raise ValueError(error_msg)
			if problem_name not in problem_directory:
				error_msg = "Problem name not found in problem directory."
				raise ValueError(error_msg)
			self.problem = problem_directory[problem_name](
				fixed_factors=problem_fixed_factors,
				model_fixed_factors=model_fixed_factors,
			)
		self.problem.model.model_created()
		self.model_created(self.problem.model)
		# Rename problem if necessary.
		if problem_rename is not None:
			if problem_rename == "":
				error_msg = "Problem rename cannot be an empty string."
				raise ValueError(error_msg)
			self.problem.name = problem_rename
		self.problem.before_replicate_override = self.before_replicate

		# Initialize file path.
		if not isinstance(file_name_path, Path):
			if file_name_path is None:
				file_name_path_str = f"{self.solver.name}_on_{self.problem.name}.pickle"
			self.file_name_path = EXPERIMENT_DIR / file_name_path_str
		else:
			self.file_name_path = file_name_path

		# Make sure the experiment directory exists
		EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

	def model_created(self, model: Model) -> None:
		"""Hook called after the problem's model is instantiated.

		Args:
			model: The initialized model associated with the experiment's problem.

		This is a helper function to customize the experiment's input model.
		"""
		pass

	def before_replicate(self, model: Model, rng_list: list[MRG32k3a]) -> None:
		"""Hook executed immediately before each replication during an experiment.

		Args:
			model: The model about to be simulated.
			rng_list: The list of RNGs used for the replication.

		This is a helper function to customize behavior before each replication.
		"""
		pass

	# TODO: Convert this to throwing exceptions?
	# TODO: Convert this functionality to run automatically
	def check_compatibility(self) -> str:
		"""Check whether the experiment's solver and problem are compatible.

		Returns:
			str: Error message in the event problem and solver are incompatible.
		"""
		# make a string builder
		error_messages = []
		# Check number of objectives.
		if (
			self.solver.objective_type == ObjectiveType.SINGLE
			and self.problem.n_objectives > 1
		):
			error_message = "Solver cannot solve a multi-objective problem"
			error_messages.append(error_message)
		elif (
			self.solver.objective_type == ObjectiveType.MULTI
			and self.problem.n_objectives == 1
		):
			error_message = "Solver cannot solve a single-objective problem"
			error_messages.append(error_message)
		# Check constraint types.
		if self.solver.constraint_type.value < self.problem.constraint_type.value:
			solver_str = self.solver.constraint_type.name.lower()
			problem_str = self.problem.constraint_type.name.lower()
			error_message = (
				f"Solver can handle {solver_str} constraints, "
				f"but problem has {problem_str} constraints."
			)
			error_messages.append(error_message)

		if (
			# TODO: this is a hack to get around the fact that compatibility checks are
			# not implemented correctly.
			self.solver.class_name_abbr == "FCSA"
			and self.solver.constraint_type == ConstraintType.STOCHASTIC
			and self.problem.constraint_type != ConstraintType.STOCHASTIC
		):
			error_message = (
				"Solver is designed for problems with stochastic constraints, but "
				"the problem does not have any."
			)
			error_messages.append(error_message)

		if (
			self.solver.constraint_type != ConstraintType.STOCHASTIC
			and self.problem.constraint_type == ConstraintType.STOCHASTIC
		):
			error_message = (
				"Solver is not designed for problems with stochastic constraints, but "
				"the problem have some."
			)
			error_messages.append(error_message)

		# Check variable types.
		if (
			self.solver.variable_type == VariableType.DISCRETE
			and self.problem.variable_type != VariableType.DISCRETE
		):
			problem_type = self.problem.variable_type.name.lower()
			error_message = (
				"Solver is for discrete variables, "
				f"but problem variables are {problem_type}."
			)
			error_messages.append(error_message)
		elif (
			self.solver.variable_type == VariableType.CONTINUOUS
			and self.problem.variable_type != VariableType.CONTINUOUS
		):
			problem_type = self.problem.variable_type.name.lower()
			error_message = (
				"Solver is for continuous variables, "
				f"but problem variables are {problem_type}."
			)
			error_messages.append(error_message)
		# Check for existence of gradient estimates.
		if self.solver.gradient_needed and not self.problem.gradient_available:
			error_message = (
				"Solver requires gradient estimates but problem does not have them."
			)
			error_messages.append(error_message)
		# Strip trailing newline character.
		return "\n".join(error_messages)

	#! added in budget history, fn estimates, and iterations
	def run(self, n_macroreps: int, n_jobs: int = -1) -> None:
		"""Runs the solver on the problem for a given number of macroreplications.

		Note:
			RNGs for random problem instances are reserved but currently unused.
			This method is under development.

		Args:
			n_macroreps (int): Number of macroreplications to run.
			n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
				-1: use all available cores
				1: run sequentially

		Raises:
			ValueError: If `n_macroreps` is not positive.
		"""
		start = time.time()
		solution_df, iteration_df, elapsed_times = run_solver.run_solver(
			self.solver, self.problem, n_macroreps, n_jobs
		)
		elapsed = time.time() - start
		logging.info(f"Finished running {n_macroreps} mreps in {elapsed:.3f} seconds.")

		self.n_macroreps = n_macroreps
		self.all_recommended_xs = run_solver._to_list(solution_df, "solution")
		self.all_intermediate_budgets = run_solver._to_list(solution_df, "budget")

		# Store iteration-level data if available
		if iteration_df is not None:
			self.all_iterations = run_solver._iteration_to_list(iteration_df, "iteration")
			self.all_budget_history = run_solver._iteration_to_list(iteration_df, "budget_history")
			self.all_fn_estimates = run_solver._iteration_to_list(iteration_df, "fn_estimate")
		else:
			self.all_iterations = None
			self.all_budget_history = None
			self.all_fn_estimates = None
		
		self.timings = elapsed_times

		self.has_run = True
		self.has_postreplicated = False
		self.has_postnormalized = False

		# Save ProblemSolver object to .pickle file if specified.
		if self.create_pickle:
			file_name = self.file_name_path.name
			self.record_experiment_results(file_name=file_name)

	def run_resumable(
	self,
	n_macroreps_total: int,
	n_jobs: int = -1,
	macroreps_per_chunk: int = 10,
	checkpoint_path: Path | str | None = None,
	) -> None:
		"""
		Run the solver in a resumable way by splitting macroreplications into chunks
		and checkpointing after each chunk and on termination signals.

		Args:
			n_macroreps_total: Total number of macroreplications desired.
			n_jobs: Number of jobs to run in parallel (same meaning as in run()).
			macroreps_per_chunk: Number of macroreps per batch before checkpointing.
			checkpoint_path: Optional explicit checkpoint file path. If None, use
				self.file_name_path with suffix '.resume.pickle'.

		Usage (fresh run):
			ps = ProblemSolver(...)
			ps.run_resumable(500, n_jobs=-1, macroreps_per_chunk=20)

		Usage (resume):
			with open("...resume.pickle", "rb") as f:
				ps = pickle.load(f)
			ps.run_resumable(500, n_jobs=-1, macroreps_per_chunk=20)
		"""
		if n_macroreps_total <= 0:
			raise ValueError("Total number of macroreplications must be positive.")

		# Determine checkpoint file
		if checkpoint_path is None:
			base = getattr(self, "file_name_path", None)
			if base is None:
				checkpoint_path = EXPERIMENT_DIR / "problem_solver_resume.pickle"
			else:
				checkpoint_path = base.with_suffix(".resume.pickle")
		else:
			checkpoint_path = resolve_file_path(checkpoint_path, directory=EXPERIMENT_DIR)

		checkpoint_path = Path(checkpoint_path)

		# Initialise / extend storage if needed (for resumed objects)
		if not hasattr(self, "n_macroreps") or self.n_macroreps is None:
			self.n_macroreps = 0

		if not hasattr(self, "all_recommended_xs") or self.all_recommended_xs is None:
			self.all_recommended_xs = []
		if not hasattr(self, "all_intermediate_budgets") or self.all_intermediate_budgets is None:
			self.all_intermediate_budgets = []
		if not hasattr(self, "all_iterations"):
			self.all_iterations = None
		if not hasattr(self, "all_budget_history"):
			self.all_budget_history = None
		if not hasattr(self, "all_fn_estimates"):
			self.all_fn_estimates = None
		if not hasattr(self, "timings") or self.timings is None:
			self.timings = []

		# Flag for termination requested by signal
		self._termination_requested = False

		# Signal handler that checkpoints immediately
		def _signal_handler(signum, frame):
			logging.warning(
				f"Received signal {signum}; checkpointing ProblemSolver to {checkpoint_path}."
			)
			try:
				checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
				with checkpoint_path.open("wb") as f:
					pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				logging.error(f"Failed to checkpoint on signal {signum}: {e}")
			# Mark termination so the main loop can stop cleanly if it continues
			self._termination_requested = True

		# Install handlers (common for SLURM: SIGTERM; also catch SIGINT)
		old_sigterm = signal.getsignal(signal.SIGTERM)
		old_sigint = signal.getsignal(signal.SIGINT)
		signal.signal(signal.SIGTERM, _signal_handler)
		signal.signal(signal.SIGINT, _signal_handler)

		try:
			completed = self.n_macroreps
			if completed > n_macroreps_total:
				raise ValueError(
					f"Object already contains {completed} macroreps, "
					f"which exceeds requested total {n_macroreps_total}."
				)

			while completed < n_macroreps_total and not self._termination_requested:
				remaining = n_macroreps_total - completed
				n_chunk = min(macroreps_per_chunk, remaining)

				logging.info(
					f"Running macroreps {completed + 1}–{completed + n_chunk} "
					f"of {n_macroreps_total}."
				)

				start_chunk = time.time()

				# Run this chunk (same core call as in run)
				solution_df, iteration_df, elapsed_times = run_solver.run_solver(
					self.solver,
					self.problem,
					n_chunk,
					n_jobs,
				)

				elapsed_chunk = time.time() - start_chunk
				logging.info(
					f"Finished chunk of {n_chunk} macroreps "
					f"in {elapsed_chunk:.3f} seconds."
				)

				# Convert solution_df into lists for this chunk
				chunk_recommended_xs = run_solver._to_list(solution_df, "solution")
				chunk_intermediate_budgets = run_solver._to_list(solution_df, "budget")

				# Convert iteration-level data if available
				if iteration_df is not None:
					chunk_iterations = run_solver._iteration_to_list(
						iteration_df, "iteration"
					)
					chunk_budget_history = run_solver._iteration_to_list(
						iteration_df, "budget_history"
					)
					chunk_fn_estimates = run_solver._iteration_to_list(
						iteration_df, "fn_estimate"
					)
				else:
					chunk_iterations = None
					chunk_budget_history = None
					chunk_fn_estimates = None

				# Append chunk data to existing arrays
				self.all_recommended_xs.extend(chunk_recommended_xs)
				self.all_intermediate_budgets.extend(chunk_intermediate_budgets)

				if chunk_iterations is not None:
					if self.all_iterations is None:
						self.all_iterations = []
					self.all_iterations.extend(chunk_iterations)

				if chunk_budget_history is not None:
					if self.all_budget_history is None:
						self.all_budget_history = []
					self.all_budget_history.extend(chunk_budget_history)

				if chunk_fn_estimates is not None:
					if self.all_fn_estimates is None:
						self.all_fn_estimates = []
					self.all_fn_estimates.extend(chunk_fn_estimates)

				self.timings.extend(elapsed_times)

				# Update counters and flags
				completed += n_chunk
				self.n_macroreps = completed
				self.has_run = True
				self.has_postreplicated = False
				self.has_postnormalized = False

				# Checkpoint at end of chunk
				logging.info(
					f"Checkpointing after {completed} macroreps to {checkpoint_path}."
				)
				checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
				with checkpoint_path.open("wb") as f:
					pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

			# Optionally write the standard final pickle if requested
			if self.create_pickle:
				file_name = self.file_name_path.name
				self.record_experiment_results(file_name=file_name)

		finally:
			# Restore old signal handlers to avoid side effects outside this method
			signal.signal(signal.SIGTERM, old_sigterm)
			signal.signal(signal.SIGINT, old_sigint)


	def _has_stochastic_constraints(self) -> bool:
		return self.problem.n_stochastic_constraints > 0

	def post_replicate(
		self,
		n_postreps: int,
		crn_across_budget: bool = True,
		crn_across_macroreps: bool = False,
	) -> None:
		"""Runs postreplications at the solver's recommended solutions.

		Args:
			n_postreps (int): Number of postreplications at each recommended solution.
			crn_across_budget (bool, optional): If True, use CRN across solutions from
				different time budgets. Defaults to True.
			crn_across_macroreps (bool, optional): If True, use CRN across solutions
				from different macroreplications. Defaults to False.

		Raises:
			ValueError: If `n_postreps` is not positive.
		"""
		# Value checking
		if n_postreps <= 0:
			error_msg = "Number of postreplications must be positive."
			raise ValueError(error_msg)

		logging.debug(
			f"Setting up {n_postreps} postreplications for {self.n_macroreps} mreps of "
			f"{self.solver.name} on {self.problem.name}."
		)

		self.n_postreps = n_postreps
		self.crn_across_budget = crn_across_budget
		self.crn_across_macroreps = crn_across_macroreps
		# Initialize variables
		self.all_post_replicates = [[] for _ in range(self.n_macroreps)]
		for mrep in range(self.n_macroreps):
			self.all_post_replicates[mrep] = [] * len(
				self.all_intermediate_budgets[mrep]
			)
		self.timings = [0.0 for _ in range(self.n_macroreps)]

		function_start = time.time()

		logging.info("Starting postreplications")
		results: list[tuple] = Parallel(n_jobs=-1)(
			delayed(self.post_replicate_multithread)(mrep)
			for mrep in range(self.n_macroreps)
		)
		for mrep, post_rep, timing, stoch_constraints in results:
			self.all_post_replicates[mrep] = post_rep
			self.timings[mrep] = timing
			self.all_stoch_constraints.append(stoch_constraints)

		# Store estimated objective for each macrorep for each budget.
		self.all_est_objectives = []
		for mrep in range(self.n_macroreps):
			self.all_est_objectives.append(
				np.array(self.all_post_replicates[mrep]).mean(axis=1)
			)

		if self._has_stochastic_constraints():
			self.all_est_lhs = []
			for mrep in range(self.n_macroreps):
				self.all_est_lhs.append(
					np.array(self.all_stoch_constraints[mrep]).mean(axis=1)
				)

		runtime = round(time.time() - function_start, 3)
		logging.info(
			f"Finished running {self.n_macroreps} postreplications "
			f"in {runtime} seconds."
		)

		self.has_postreplicated = True
		self.has_postnormalized = False

		# Save ProblemSolver object to .pickle file if specified.
		if self.create_pickle:
			file_name = self.file_name_path.name
			self.record_experiment_results(file_name=file_name)

	def post_replicate_multithread(self, mrep: int) -> tuple:
		"""Runs postreplications for a given macroreplication's recommended solutions.

		Args:
			mrep (int): Index of the macroreplication.

		Returns:
			tuple: A tuple containing:
				- int: Macroreplication index.
				- list: Postreplicates for each recommended solution.
				- float: Runtime for the macroreplication.

		Raises:
			ValueError: If `mrep` is negative.
		"""
		# Value checking
		if mrep < 0:
			error_msg = "Macroreplication index must be non-negative."
			raise ValueError(error_msg)

		logging.debug(
			f"Macroreplication {mrep + 1}: Starting postreplications for "
			f"{self.solver.name} on {self.problem.name}."
		)
		# Create RNG list for the macroreplication.
		if self.crn_across_macroreps:
			# Use the same RNGs for all macroreps.
			baseline_rngs = [
				MRG32k3a(s_ss_sss_index=[0, self.problem.model.n_rngs + rng_index, 0])
				for rng_index in range(self.problem.model.n_rngs)
			]
		else:
			baseline_rngs = [
				MRG32k3a(
					s_ss_sss_index=[
						0,
						self.problem.model.n_rngs * (mrep + 1) + rng_index,
						0,
					]
				)
				for rng_index in range(self.problem.model.n_rngs)
			]

		tic = time.perf_counter()

		# Create an empty list for each budget
		post_replicates = []
		stoch_constraints = []
		# Loop over all recommended solutions.
		for budget_index in range(len(self.all_intermediate_budgets[mrep])):
			x = self.all_recommended_xs[mrep][budget_index]
			fresh_soln = Solution(x, self.problem)
			# Attach RNGs for postreplications.
			# If CRN is used across budgets, then we should use a copy rather
			# than passing in the original RNGs.
			if self.crn_across_budget:
				fresh_soln.attach_rngs(rng_list=baseline_rngs, copy=True)
			else:
				fresh_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
			self.problem.simulate(solution=fresh_soln, num_macroreps=self.n_postreps)
			# Store results
			post_replicates.append(
				list(fresh_soln.objectives[: fresh_soln.n_reps][:, 0])
			)  # 0 <- assuming only one objective

			if self._has_stochastic_constraints():
				stoch_constraints.append(fresh_soln.stoch_constraints)
		toc = time.perf_counter()
		runtime = toc - tic
		logging.debug(f"\t{mrep + 1}: Finished in {round(runtime, 3)} seconds")

		return mrep, post_replicates, runtime, stoch_constraints

	def bootstrap_sample(
		self,
		bootstrap_rng: MRG32k3a,
		normalize: bool = True,
		feasibility_score_method: Literal["inf_norm", "norm"] = "inf_norm",
		feasibility_norm_degree: int = 1,
		feasibility_two_sided: bool = False,
		disable_macrorep_bootstrap: bool = False,
	) -> tuple[list[Curve], list[Curve]]:
		"""Generates bootstrap samples of objective/progress and feasibility curves.

		Args:
			bootstrap_rng (MRG32k3a): Random number generator used for bootstrapping.
			normalize (bool, optional): If True, normalize progress curves with respect
				to optimality gaps. Defaults to True.
			feasibility_score_method (Literal["inf_norm", "norm"], optional):
				Feasibility scoring method. Defaults to "inf_norm".
			feasibility_norm_degree (int, optional): Degree of the norm when
			``feasibility_score_method == "norm"``. Defaults to 1.
			feasibility_two_sided (bool, optional): Whether to award a non-zero score
				to feasible solutions based on the best violation.
			disable_macrorep_bootstrap (bool, optional): Whether to disable bootstrap
				across macroreplications. Defaults to False.

		Returns:
			tuple[list[Curve], list[Curve]]: Bootstrapped progress curves and
			feasibility curves for all macroreplications.
		"""
		bootstrap_curves: list[Curve] = []
		bootstrap_feasibility_curves: list[Curve] = []
		has_stochastic_constraints = self.problem.n_stochastic_constraints >= 1

		# Uniformly resample M macroreplications (with replacement) from 0, 1, ..., M-1.
		# Subsubstream 0: reserved for this outer-level bootstrapping.
		if disable_macrorep_bootstrap:
			bs_mrep_idxs = list(range(self.n_macroreps))
		else:
			bs_mrep_idxs = bootstrap_rng.choices(
				range(self.n_macroreps), k=self.n_macroreps
			)
		# Advance RNG subsubstream to prepare for inner-level bootstrapping.
		bootstrap_rng.advance_subsubstream()
		# Subsubstream 1: reserved for bootstrapping at x0 and x*.
		# Bootstrap sample post-replicates at common x0.
		# Uniformly resample L postreps (with replacement) from 0, 1, ..., L-1.
		bs_postrep_idxs = bootstrap_rng.choices(
			range(self.n_postreps_init_opt), k=self.n_postreps_init_opt
		)
		# Compute the mean of the resampled postreplications.
		bs_initial_obj_val = np.mean(
			[self.x0_postreps[postrep] for postrep in bs_postrep_idxs]
		)
		# Reset subsubstream if using CRN across budgets.
		# This means the same postreplication indices will be used for resampling at
		# x0 and x*.
		if self.crn_across_init_opt:
			bootstrap_rng.reset_subsubstream()
		# Bootstrap sample postreplicates at reference optimal solution x*.
		# Uniformly resample L postreps (with replacement) from 0, 1, ..., L.
		bs_postrep_idxs = bootstrap_rng.choices(
			range(self.n_postreps_init_opt), k=self.n_postreps_init_opt
		)
		# Compute the mean of the resampled postreplications.
		bs_optimal_obj_val = np.mean(
			[self.xstar_postreps[postrep] for postrep in bs_postrep_idxs]
		)
		# Compute initial optimality gap.
		bs_initial_opt_gap = bs_initial_obj_val - bs_optimal_obj_val
		# Advance RNG subsubstream to prepare for inner-level bootstrapping.
		# Will now be at start of subsubstream 2.
		bootstrap_rng.advance_subsubstream()
		# Bootstrap within each bootstrapped macroreplication.
		# Option 1: Simpler (default) CRN scheme, which makes for faster code.
		if self.crn_across_budget and not self.crn_across_macroreps:
			for idx in range(self.n_macroreps):
				mrep = bs_mrep_idxs[idx]
				# Inner-level bootstrapping over intermediate recommended solutions.
				est_objectives = []
				est_lhs = []
				# Same postreplication indices for all intermediate budgets on
				# a given macroreplciation.
				bs_postrep_idxs = bootstrap_rng.choices(
					range(self.n_postreps), k=self.n_postreps
				)
				for budget in range(len(self.all_intermediate_budgets[mrep])):
					if has_stochastic_constraints:
						est_lhs.append(
							self.all_stoch_constraints[mrep][budget][
								bs_postrep_idxs
							].mean(axis=0)
						)
					else:
						est_lhs.append(np.array([]))
					# If solution is x0...
					if self.all_recommended_xs[mrep][budget] == self.x0:
						est_objectives.append(bs_initial_obj_val)
					# ...else if solution is x*...
					elif self.all_recommended_xs[mrep][budget] == self.xstar:
						est_objectives.append(bs_optimal_obj_val)
					# ... else solution other than x0 or x*.
					else:
						# Compute the mean of the resampled postreplications.
						est_objectives.append(
							np.mean(
								[
									self.all_post_replicates[mrep][budget][postrep]
									for postrep in bs_postrep_idxs
								]
							)
						)
				# Record objective or progress curve.
				if normalize:
					frac_intermediate_budgets = [
						budget / self.problem.factors["budget"]
						for budget in self.all_intermediate_budgets[mrep]
					]
					norm_est_objectives = [
						(est_objective - bs_optimal_obj_val) / bs_initial_opt_gap
						for est_objective in est_objectives
					]
					new_progress_curve = Curve(
						x_vals=frac_intermediate_budgets,
						y_vals=norm_est_objectives,
					)
					bootstrap_curves.append(new_progress_curve)
				else:
					new_objective_curve = Curve(
						x_vals=self.all_intermediate_budgets[mrep],
						y_vals=est_objectives,
					)
					bootstrap_curves.append(new_objective_curve)

				bootstrap_feasibility_curves.append(
					Curve(
						x_vals=self.all_intermediate_budgets[mrep],
						y_vals=feasibility_score_history(
							est_lhs,
							feasibility_score_method,
							feasibility_norm_degree,
							feasibility_two_sided,
						),
					)
				)
		# Option 2: Non-default CRN behavior.
		else:
			for idx in range(self.n_macroreps):
				mrep = bs_mrep_idxs[idx]
				# Inner-level bootstrapping over intermediate recommended solutions.
				est_objectives = []
				est_lhs = []
				for budget in range(len(self.all_intermediate_budgets[mrep])):
					if has_stochastic_constraints:
						indices = bootstrap_rng.choices(
							range(self.n_postreps), k=self.n_postreps
						)
						est_lhs.append(
							self.all_stoch_constraints[mrep][budget][indices].mean(
								axis=0
							)
						)
					else:
						est_lhs.append(np.array([]))

					# If solution is x0...
					if self.all_recommended_xs[mrep][budget] == self.x0:
						est_objectives.append(bs_initial_obj_val)
					# ...else if solution is x*...
					elif self.all_recommended_xs[mrep][budget] == self.xstar:
						est_objectives.append(bs_optimal_obj_val)
					# ... else solution other than x0 or x*.
					else:
						# Uniformly resample N postreps (with replacement)
						# from 0, 1, ..., N-1.
						bs_postrep_idxs = bootstrap_rng.choices(
							range(self.n_postreps), k=self.n_postreps
						)
						# Compute the mean of the resampled postreplications.
						est_objectives.append(
							np.mean(
								[
									self.all_post_replicates[mrep][budget][postrep]
									for postrep in bs_postrep_idxs
								]
							)
						)
						# Reset subsubstream if using CRN across budgets.
						if self.crn_across_budget:
							bootstrap_rng.reset_subsubstream()
				# If using CRN across macroreplications...
				if self.crn_across_macroreps:
					# ...reset subsubstreams...
					bootstrap_rng.reset_subsubstream()
				# ...else if not using CRN across macrorep...
				else:
					# ...advance subsubstream.
					bootstrap_rng.advance_subsubstream()
				# Record objective or progress curve.
				if normalize:
					frac_intermediate_budgets = [
						budget / self.problem.factors["budget"]
						for budget in self.all_intermediate_budgets[mrep]
					]
					norm_est_objectives = [
						(est_objective - bs_optimal_obj_val) / bs_initial_opt_gap
						for est_objective in est_objectives
					]
					new_progress_curve = Curve(
						x_vals=frac_intermediate_budgets,
						y_vals=norm_est_objectives,
					)
					bootstrap_curves.append(new_progress_curve)
				else:
					new_objective_curve = Curve(
						x_vals=self.all_intermediate_budgets[mrep],
						y_vals=est_objectives,
					)
					bootstrap_curves.append(new_objective_curve)
				bootstrap_feasibility_curves.append(
					Curve(
						x_vals=self.all_intermediate_budgets[mrep],
						y_vals=feasibility_score_history(
							est_lhs,
							feasibility_score_method,
							feasibility_norm_degree,
							feasibility_two_sided,
						),
					)
				)
		return bootstrap_curves, bootstrap_feasibility_curves

	def bootstrap_terminal_objective_and_feasibility(
		self,
		bootstrap_rng: MRG32k3a,
		feasibility_score_method: Literal["inf_norm", "norm"] = "inf_norm",
		feasibility_norm_degree: int = 1,
		feasibility_two_sided: bool = True,
	) -> tuple[list[float], list[float]]:
		"""Bootstraps terminal objective and feasibility scores."""
		bootstrap_objective_curves, bootstrap_feasibility_curves = (
			self.bootstrap_sample(
				bootstrap_rng,
				False,
				feasibility_score_method,
				feasibility_norm_degree,
				feasibility_two_sided,
				True,
			)
		)

		return (
			[obj.y_vals[-1] for obj in bootstrap_objective_curves],
			[feas.y_vals[-1] for feas in bootstrap_feasibility_curves],
		)

	def feasibility_score_history(
		self,
		feasibility_score_method: Literal["inf_norm", "norm"] = "inf_norm",
		feasibility_norm_degree: int = 1,
		feasibility_two_sided: bool = True,
	) -> None:
		"""Compute feasibility score history."""
		has_stochastic_constraints = self.problem.n_stochastic_constraints >= 1
		if not has_stochastic_constraints:
			return

		self.feasibility_curves = []

		for mrep in range(self.n_macroreps):
			lhs = self.all_est_lhs[mrep]
			self.feasibility_curves.append(
				Curve(
					x_vals=self.all_intermediate_budgets[mrep],
					y_vals=feasibility_score_history(
						lhs,
						feasibility_score_method,
						feasibility_norm_degree,
						feasibility_two_sided,
					),
				)
			)

	def record_experiment_results(self, file_name: str) -> None:
		"""Saves the ProblemSolver object to a .pickle file.

		Args:
			file_name (str): Name of the pickle file. It is saved under the
				EXPERIMENT_DIR path.
		"""
		# Accept either a bare filename, a relative path (treated as CWD-relative),
		# or an absolute path. If a directory component is present, prefer the
		# provided directory (resolved against CWD) so caller-specified output
		# folders are respected.
		p = Path(file_name)
		if p.is_absolute():
			file_path = p
		elif p.parent != Path('.'):
			file_path = (Path.cwd() / p).resolve()
		else:
			file_path = EXPERIMENT_DIR / file_name
		folder_name = file_path.parent

		logging.debug(f"Saving ProblemSolver object to {file_path}")

		# Create the directory if it does not exist.
		folder_name.mkdir(parents=True, exist_ok=True)
		# Delete the file if it already exists.
		if file_path.exists():
			file_path.unlink()
		# Create and dump the object to the file
		with file_path.open("xb") as file:
			pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

		logging.info(f"Saved experiment results to {file_path}")

	def log_experiment_results(self, print_solutions: bool = True, file_path: str | None = None) -> None:
		"""Creates a readable .txt log file from a problem-solver pair's .pickle file.

		Args:
			print_solutions (bool, optional): If True, include recommended solutions in
				the .txt file. Defaults to True.
		"""
		if file_path is None:
			results_file_path = EXPERIMENT_DIR / "experiment_results.txt"
		else:
			results_file_path = Path(file_path)

		with results_file_path.open("w") as file:
			# Title txt file with experiment information.
			file.write(str(self.file_name_path))
			file.write("\n")
			file.write(f"Problem: {self.problem.name}\n")
			file.write(f"Solver: {self.solver.name}\n\n")

			# Display model factors.
			file.write("Model Factors:\n")
			for key, value in self.problem.model.factors.items():
				# Excluding model factors corresponding to decision variables.
				if key not in self.problem.model_decision_factors:
					file.write(f"\t{key}: {value}\n")
			file.write("\n")
			# Display problem factors.
			file.write("Problem Factors:\n")
			for key, value in self.problem.factors.items():
				file.write(f"\t{key}: {value}\n")
			file.write("\n")
			# Display solver factors.
			file.write("Solver Factors:\n")
			for key, value in self.solver.factors.items():
				file.write(f"\t{key}: {value}\n")
			file.write("\n")

			# Display macroreplication information.
			file.write(f"{self.n_macroreps} macroreplications were run.\n")
			# If results have been postreplicated, list the number of post-replications.
			if self.has_postreplicated:
				file.write(
					f"{self.n_postreps} postreplications were run "
					"at each recommended solution.\n\n"
				)
			# If post-normalized, state initial solution (x0) and
			# proxy optimal solution (x_star) and how many replications
			# were taken of them (n_postreps_init_opt).
			if self.has_postnormalized:
				init_sol = tuple([round(x, 4) for x in self.x0])
				est_obj = round(np.mean(self.x0_postreps), 4)
				file.write(
					f"The initial solution is {init_sol}. "
					f"Its estimated objective is {est_obj}.\n"
				)
				if self.xstar is None:
					file.write(
						"No proxy optimal solution was used. "
						"A proxy optimal objective function value of "
						f"{self.problem.optimal_value} was provided.\n"
					)
				else:
					proxy_opt = tuple([round(x, 4) for x in self.xstar])
					est_obj = round(np.mean(self.xstar_postreps), 4)
					file.write(
						f"The proxy optimal solution is {proxy_opt}. "
						f"Its estimated objective is {est_obj}.\n"
					)
				file.write(
					f"{self.n_postreps_init_opt} postreplications were taken "
					"at x0 and x_star.\n\n"
				)
			# Display recommended solution at each budget value for
			# each macroreplication.
			file.write("Macroreplication Results:\n")
			for mrep in range(self.n_macroreps):
				file.write(f"\nMacroreplication {mrep + 1}:\n")
				mrep_int_budgets = self.all_intermediate_budgets[mrep]
				for budget in range(len(mrep_int_budgets)):
					file.write(f"\tBudget: {round(mrep_int_budgets[budget], 4)}")
					# Optionally print solutions.
					if print_solutions:
						all_rec_xs = self.all_recommended_xs[mrep][budget]
						rec_xs_tup = tuple([round(x, 4) for x in all_rec_xs])
						file.write(f"\tRecommended Solution: {rec_xs_tup}")
					# If postreplicated, add estimated objective function values.
					if self.has_postreplicated:
						est_obj = self.all_est_objectives[mrep][budget]
						file.write(f"\tEstimated Objective: {round(est_obj, 4)}\n")
				# file.write(
				#     "\tThe time taken to complete this macroreplication was "
				#     f"{round(self.timings[mrep], 2)} s.\n"
				# )
