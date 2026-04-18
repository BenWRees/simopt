"""Comprehensive test comparing ASTROMoRF performance under different adaptive subspace.

dimension policies.

This module creates actual solver subclasses that override the dimension scoring logic,
ensuring each policy truly uses different selection strategies.

Policies compared:
1. Current (Validation + Projection + Success weighted sum)
2. Cost-Penalized Multi-Objective
3. Inactive Subspace Minimization
4. Pareto-Optimal Selection
5. Fixed dimension baseline

Key difference from v2: This version creates ACTUAL SOLVER SUBCLASSES that override
the evaluate_and_score_candidate_dimensions method, guaranteeing different behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest
from scipy.special import comb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import dimension scaling function
from demo.pickle_files_journal_paper import SCALABLE_PROBLEMS, scale_dimension
from simopt.solvers.astromorf import ASTROMORF

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# SOLVER SUBCLASSES WITH DIFFERENT DIMENSION POLICIES
# =============================================================================


class ASTROMORF_CostPenalized(ASTROMORF):  # noqa: N801
    """ASTROMORF with cost-penalized dimension selection.

    Explicitly penalizes larger dimensions by their per-iteration sample cost.
    score(d) = performance(d) - cost_weight * normalized_cost(d)
    """

    name: str = "ASTROMORF_CostPenalized"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.cost_weight = 0.3  # Cost penalty weight

    def compute_optimal_subspace_dimension(self) -> None:
        """Override to use cost-penalized scoring without plateau logic."""
        # Skip plateau detection - rely purely on scoring
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return  # Keep current d if no history

        scores = self.evaluate_and_score_candidate_dimensions()
        if not scores:
            return

        # Choose best d by score
        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
        optimal_d = max(1, min(int(best_d), self.problem.dim - 1))

        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: [CostPenalized] Adaptive subspace dimension change: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """Cost-penalized scoring that explicitly penalizes sample cost."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        # Get eigenvalue spectrum
        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        # Compute sample costs for each dimension
        degree = self.factors.get("polynomial degree", 2)
        costs = {}
        for d in range(1, max_test_d + 1):
            # M = C(d + degree, degree) + 1
            costs[d] = int(comb(d + degree, degree, exact=True)) + 1

        max_cost = max(costs.values())
        min_cost = min(costs.values())
        cost_range = max(1, max_cost - min_cost)

        # Get validation errors
        validation_by_d = self._get_validation_by_d()
        val_errs = [v for v in validation_by_d.values() if v is not None]
        if val_errs:
            min_val, max_val = min(val_errs), max(val_errs)
        else:
            min_val, max_val = 0, 1
        val_range = max(1e-12, max_val - min_val)

        for d in range(1, max_test_d + 1):
            # Model quality from validation error
            val_err = validation_by_d.get(d, 0.5)
            model_quality = (
                1.0 - (val_err - min_val) / val_range if val_range > 1e-12 else 0.8
            )

            # Variance captured
            captured = np.sum(eig_source[:d])
            var_captured = captured / total_var

            # Success rate
            s_succ = self._get_success_rate_for_d(d)

            # Performance score
            performance = 0.5 * model_quality + 0.3 * var_captured + 0.2 * s_succ

            # Cost penalty (normalized)
            normalized_cost = (costs[d] - min_cost) / cost_range

            # Final score: performance minus cost penalty
            score = performance - self.cost_weight * normalized_cost

            results[d] = {
                "score": float(max(0, score)),
                "s_val": float(model_quality),
                "s_proj": float(var_captured),
                "s_succ": float(s_succ),
                "cost": float(normalized_cost),
                "performance": float(performance),
            }

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        """Get eigenvalue spectrum from history or fallback."""
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)

        # Fallback from previous model information
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)

        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        """Get validation errors by dimension."""
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d

        # Average from history
        vals = {}
        counts = {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1

        if counts:
            return {k: vals[k] / counts[k] for k in vals}
        return {}

    def _get_success_rate_for_d(self, d: int) -> float:
        """Get historical success rate for dimension d."""
        successes = 0
        total = 0
        for info in self.previous_model_information:
            if info.get("recommended_dimension") == d:
                total += 1
                if info.get("model_success"):
                    successes += 1
        return successes / max(1, total) if total > 0 else 0.5


class ASTROMORF_InactiveMinimization(ASTROMORF):  # noqa: N801
    """ASTROMORF with inactive subspace minimization.

    Explicitly minimizes the sum of eigenvalues in the inactive subspace
    (the orthogonal complement to the active subspace).
    """

    name: str = "ASTROMORF_InactiveMin"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.variance_threshold = 0.95
        self.cost_weight = 0.1

    def compute_optimal_subspace_dimension(self) -> None:
        """Override to use inactive subspace minimization without plateau logic."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        scores = self.evaluate_and_score_candidate_dimensions()
        if not scores:
            return

        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
        optimal_d = max(1, min(int(best_d), self.problem.dim - 1))

        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: [InactiveMin] Adaptive subspace dimension change: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """Scoring that minimizes inactive subspace eigenvalue sum."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        # Compute costs
        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost = max(costs.values())

        # Get validation errors
        validation_by_d = self._get_validation_by_d()

        for d in range(1, max_test_d + 1):
            # Sum of inactive eigenvalues (information lost)
            inactive_sum = np.sum(eig_source[d:])
            inactive_fraction = inactive_sum / total_var

            # Variance captured (information retained)
            active_sum = np.sum(eig_source[:d])
            active_fraction = active_sum / total_var

            # Model quality
            val_err = validation_by_d.get(d, 0.5)
            model_quality = 1.0 - val_err

            # Cost penalty
            cost_fraction = costs[d] / max_cost

            # Score: maximize active variance, minimize inactive variance, penalize cost
            score = (
                0.4 * active_fraction
                + 0.3 * (1.0 - inactive_fraction)  # Penalize inactive variance
                + 0.2 * model_quality
                - self.cost_weight * cost_fraction
            )

            results[d] = {
                "score": float(max(0, score)),
                "s_proj": float(active_fraction),
                "inactive_fraction": float(inactive_fraction),
                "s_val": float(model_quality),
                "cost": float(cost_fraction),
            }

        # Find smallest d that captures enough variance (threshold approach)
        variance_ratios = np.cumsum(eig_source) / total_var
        threshold_d = int(np.searchsorted(variance_ratios, self.variance_threshold) + 1)
        threshold_d = min(max(1, threshold_d), max_test_d)

        # Boost score of threshold_d if it's competitive
        if threshold_d in results:
            score_based_d = max(results, key=lambda k: results[k]["score"])
            if results[threshold_d]["score"] >= 0.9 * results[score_based_d]["score"]:
                # Prefer the smaller dimension
                results[threshold_d]["score"] = results[score_based_d]["score"] + 0.01

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d
        vals, counts = {}, {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: vals[k] / counts[k] for k in vals} if counts else {}


class ASTROMORF_ParetoOptimal(ASTROMORF):  # noqa: N801
    """ASTROMORF with Pareto-optimal dimension selection.

    Considers multiple objectives simultaneously:
    1. Maximize expected improvement (model quality * variance captured)
    2. Minimize inactive subspace eigenvalue sum
    3. Minimize sample cost

    Selects from the Pareto front using weighted Chebyshev scalarization.
    """

    name: str = "ASTROMORF_Pareto"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.improvement_weight = 0.5
        self.inactive_weight = 0.3
        self.cost_weight = 0.2

    def compute_optimal_subspace_dimension(self) -> None:
        """Override to use Pareto-optimal selection without plateau logic."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        scores = self.evaluate_and_score_candidate_dimensions()
        if not scores:
            return

        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
        optimal_d = max(1, min(int(best_d), self.problem.dim - 1))

        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: [Pareto] Adaptive subspace dimension change: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """Pareto-optimal multi-objective scoring."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost = max(costs.values())

        validation_by_d = self._get_validation_by_d()

        # Compute objectives for each dimension
        objectives = {}
        for d in range(1, max_test_d + 1):
            # Objective 1: Expected improvement (maximize)
            val_err = validation_by_d.get(d, 0.5)
            model_quality = 1.0 - val_err
            captured = np.sum(eig_source[:d])
            var_captured = captured / total_var
            s_succ = self._get_success_rate_for_d(d)
            expected_improvement = model_quality * var_captured * (0.5 + 0.5 * s_succ)

            # Objective 2: Negative inactive eigenvalue sum (maximize = minimize
            # inactive)
            inactive_sum = np.sum(eig_source[d:])
            neg_inactive = -inactive_sum / total_var

            # Objective 3: Negative cost (maximize = minimize cost)
            neg_cost = -costs[d] / max_cost

            objectives[d] = (expected_improvement, neg_inactive, neg_cost)

        # Find Pareto front
        pareto_front = self._compute_pareto_front(objectives)

        # Normalize objectives for weighted sum
        obj_array = np.array([objectives[d] for d in range(1, max_test_d + 1)])
        obj_min = obj_array.min(axis=0)
        obj_max = obj_array.max(axis=0)
        obj_range = np.maximum(obj_max - obj_min, 1e-12)

        weights = np.array(
            [self.improvement_weight, self.inactive_weight, self.cost_weight]
        )
        weights = weights / weights.sum()

        for d in range(1, max_test_d + 1):
            obj = np.array(objectives[d])
            normalized = (obj - obj_min) / obj_range
            score = float(np.dot(weights, normalized))

            # Bonus for being on Pareto front
            if d in pareto_front:
                score += 0.05

            results[d] = {
                "score": float(max(0, score)),
                "expected_improvement": float(objectives[d][0]),
                "neg_inactive": float(objectives[d][1]),
                "neg_cost": float(objectives[d][2]),
                "on_pareto_front": d in pareto_front,
            }

        return results

    def _compute_pareto_front(self, objectives: dict) -> list[int]:
        """Compute dimensions on the Pareto front."""
        pareto_front = []
        for d1, obj1 in objectives.items():
            is_dominated = False
            for d2, obj2 in objectives.items():
                if d1 != d2:  # noqa: SIM102
                    # Check if d2 dominates d1 (all objectives >= and at least one >)
                    if all(
                        o2 >= o1 for o1, o2 in zip(obj1, obj2, strict=False)
                    ) and any(o2 > o1 for o1, o2 in zip(obj1, obj2, strict=False)):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_front.append(d1)
        return sorted(pareto_front)

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d
        vals, counts = {}, {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: vals[k] / counts[k] for k in vals} if counts else {}

    def _get_success_rate_for_d(self, d: int) -> float:
        successes = 0
        total = 0
        for info in self.previous_model_information:
            if info.get("recommended_dimension") == d:
                total += 1
                if info.get("model_success"):
                    successes += 1
        return successes / max(1, total) if total > 0 else 0.5


class ASTROMORF_VarianceThreshold(ASTROMORF):  # noqa: N801
    """ASTROMORF with simple variance threshold selection.

    Selects the smallest dimension that captures target_variance of the
    total eigenvalue spectrum. Simple and fast baseline.
    """

    name: str = "ASTROMORF_VarThreshold"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.target_variance = 0.95

    def compute_optimal_subspace_dimension(self) -> None:
        """Override to use simple variance threshold without plateau logic."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        scores = self.evaluate_and_score_candidate_dimensions()
        if not scores:
            return

        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
        optimal_d = max(1, min(int(best_d), self.problem.dim - 1))

        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: [VarThreshold] Adaptive subspace dimension change: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """Simple variance-threshold scoring."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        # Find threshold dimension
        variance_ratios = np.cumsum(eig_source) / total_var
        threshold_d = int(np.searchsorted(variance_ratios, self.target_variance) + 1)
        threshold_d = min(max(1, threshold_d), max_test_d)

        for d in range(1, max_test_d + 1):
            # Score is highest at threshold_d, decreasing away from it
            # This encourages selecting exactly the threshold dimension
            distance = abs(d - threshold_d)
            score = 1.0 / (1.0 + 0.5 * distance)

            captured = np.sum(eig_source[:d])
            var_captured = captured / total_var

            results[d] = {
                "score": float(score),
                "s_proj": float(var_captured),
                "is_threshold": d == threshold_d,
            }

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6


# =============================================================================
# IMPROVED/HYBRID POLICIES
# =============================================================================


class ASTROMORF_AdaptiveCost(ASTROMORF):  # noqa: N801
    """CostPenalized with adaptive cost weight based on:.

    1. Remaining budget fraction (higher penalty early, lower late)
    2. Recent success rate (if doing well, allow more exploration).
    """

    name: str = "ASTROMORF_AdaptiveCost"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.base_cost_weight = 0.3
        self.min_cost_weight = 0.1
        self.max_cost_weight = 0.5

    def compute_optimal_subspace_dimension(self) -> None:
        """Compute optimal subspace dimension."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        scores = self.evaluate_and_score_candidate_dimensions()
        if not scores:
            return

        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
        optimal_d = max(1, min(int(best_d), self.problem.dim - 1))

        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: [AdaptiveCost] Adaptive subspace dimension change: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def _compute_adaptive_cost_weight(self) -> float:
        """Compute cost weight based on budget and success."""
        # Budget factor: penalize more when budget is plentiful
        if hasattr(self, "budget") and self.budget:
            budget_fraction = self.budget.remaining / self.budget.total
        else:
            budget_fraction = 0.5

        # Success factor: if doing well, can explore more (less penalty)
        recent_successes = (
            len(self.successful_iterations[-10:])
            if hasattr(self, "successful_iterations")
            else 0
        )
        recent_total = min(10, self.iteration_count) if self.iteration_count > 0 else 1
        success_rate = recent_successes / recent_total

        # Higher budget = higher cost weight (be more frugal early)
        # Higher success = lower cost weight (allow exploration when doing well)
        cost_weight = (
            self.base_cost_weight * (0.5 + 0.5 * budget_fraction) * (1.5 - success_rate)
        )

        return np.clip(cost_weight, self.min_cost_weight, self.max_cost_weight)

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """Cost-penalized scoring with adaptive weight."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost, min_cost = max(costs.values()), min(costs.values())
        cost_range = max(1, max_cost - min_cost)

        validation_by_d = self._get_validation_by_d()
        val_errs = [v for v in validation_by_d.values() if v is not None]
        if val_errs:
            min_val, max_val = min(val_errs), max(val_errs)
        else:
            min_val, max_val = 0, 1
        val_range = max(1e-12, max_val - min_val)

        # Get adaptive cost weight
        cost_weight = self._compute_adaptive_cost_weight()

        for d in range(1, max_test_d + 1):
            val_err = validation_by_d.get(d, 0.5)
            model_quality = (
                1.0 - (val_err - min_val) / val_range if val_range > 1e-12 else 0.8
            )
            var_captured = np.sum(eig_source[:d]) / total_var
            s_succ = self._get_success_rate_for_d(d)

            performance = 0.5 * model_quality + 0.3 * var_captured + 0.2 * s_succ
            normalized_cost = (costs[d] - min_cost) / cost_range
            score = performance - cost_weight * normalized_cost

            results[d] = {
                "score": float(max(0, score)),
                "s_val": float(model_quality),
                "s_proj": float(var_captured),
                "s_succ": float(s_succ),
                "cost": float(normalized_cost),
                "cost_weight": float(cost_weight),
            }

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d
        vals, counts = {}, {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: vals[k] / counts[k] for k in vals} if counts else {}

    def _get_success_rate_for_d(self, d: int) -> float:
        successes = 0
        total = 0
        for info in self.previous_model_information:
            if info.get("recommended_dimension") == d:
                total += 1
                if info.get("model_success"):
                    successes += 1
        return successes / max(1, total) if total > 0 else 0.5


class ASTROMORF_StructureAware(ASTROMORF):  # noqa: N801
    """Detects problem structure from eigenvalue spectrum and adapts policy:.

    - Sharp eigenvalue decay = separable/linear -> use aggressive reduction (like
    InactiveMin)
    - Gradual decay = coupled/complex -> use conservative approach (like Pareto).
    """

    name: str = "ASTROMORF_StructureAware"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.decay_threshold = (
            0.8  # If top d captures >80% variance, problem is separable
        )

    def compute_optimal_subspace_dimension(self) -> None:
        """Compute optimal subspace dimension."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        # Detect problem structure
        is_separable = self._detect_separable_structure()

        scores = self.evaluate_and_score_candidate_dimensions()
        if not scores:
            return

        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
        optimal_d = max(1, min(int(best_d), self.problem.dim - 1))

        if optimal_d != self.d:
            structure = "separable" if is_separable else "coupled"
            print(
                f"Iteration {self.iteration_count}: [StructureAware/{structure}] Adaptive subspace dimension change: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def _detect_separable_structure(self) -> bool:
        """Detect if problem has separable structure from eigenvalue spectrum."""
        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        # Check if a small number of dimensions capture most variance
        cumvar = np.cumsum(eig_source) / total_var

        # Check at d=3 (or 10% of problem dim, whichever is smaller)
        check_d = min(3, max(1, self.problem.dim // 10))

        if check_d <= len(cumvar):
            variance_at_check = cumvar[check_d - 1]
            return variance_at_check >= self.decay_threshold

        return False

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """Score dimensions based on detected structure."""
        is_separable = self._detect_separable_structure()
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost, min_cost = max(costs.values()), min(costs.values())
        cost_range = max(1, max_cost - min_cost)

        validation_by_d = self._get_validation_by_d()

        # Adjust weights based on structure
        if is_separable:
            # Aggressive: prioritize variance capture and cost reduction
            w_var = 0.5
            w_model = 0.2
            w_cost = 0.3
        else:
            # Conservative: prioritize model quality
            w_var = 0.3
            w_model = 0.5
            w_cost = 0.2

        for d in range(1, max_test_d + 1):
            val_err = validation_by_d.get(d, 0.5)
            model_quality = 1.0 - val_err
            var_captured = np.sum(eig_source[:d]) / total_var
            normalized_cost = (costs[d] - min_cost) / cost_range

            score = (
                w_var * var_captured
                + w_model * model_quality
                - w_cost * normalized_cost
            )

            results[d] = {
                "score": float(max(0, score)),
                "s_proj": float(var_captured),
                "s_val": float(model_quality),
                "cost": float(normalized_cost),
                "structure": "separable" if is_separable else "coupled",
            }

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d
        vals, counts = {}, {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: vals[k] / counts[k] for k in vals} if counts else {}


class ASTROMORF_ConservativeInactiveMin(ASTROMORF):  # noqa: N801
    """InactiveMin with a minimum dimension floor to avoid catastrophic failure.

    Uses a floor of max(2, 10% of problem dim) to ensure enough exploration.
    """

    name: str = "ASTROMORF_ConservativeInactiveMin"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.variance_threshold = 0.95
        self.cost_weight = 0.1

    def compute_optimal_subspace_dimension(self) -> None:
        """Compute optimal subspace dimension."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        scores = self.evaluate_and_score_candidate_dimensions()
        if not scores:
            return

        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]

        # Apply minimum dimension floor
        min_d = max(2, self.problem.dim // 10)  # At least 2, or 10% of problem dim
        optimal_d = max(min_d, min(int(best_d), self.problem.dim - 1))

        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: [ConservativeInactiveMin] Adaptive subspace dimension change: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """Same scoring as InactiveMin."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost = max(costs.values())

        validation_by_d = self._get_validation_by_d()

        for d in range(1, max_test_d + 1):
            inactive_sum = np.sum(eig_source[d:])
            inactive_fraction = inactive_sum / total_var
            active_sum = np.sum(eig_source[:d])
            active_fraction = active_sum / total_var

            val_err = validation_by_d.get(d, 0.5)
            model_quality = 1.0 - val_err
            cost_fraction = costs[d] / max_cost

            score = (
                0.4 * active_fraction
                + 0.3 * (1.0 - inactive_fraction)
                + 0.2 * model_quality
                - self.cost_weight * cost_fraction
            )

            results[d] = {
                "score": float(max(0, score)),
                "s_proj": float(active_fraction),
                "inactive_fraction": float(inactive_fraction),
                "s_val": float(model_quality),
                "cost": float(cost_fraction),
            }

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d
        vals, counts = {}, {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: vals[k] / counts[k] for k in vals} if counts else {}


class ASTROMORF_VarFloor(ASTROMORF):  # noqa: N801
    """Variance-threshold with a safety floor.

    Guarantees capturing target_variance (e.g., 95%) of eigenvalue spectrum,
    BUT also enforces a minimum dimension floor for safety.

    optimal_d = max(variance_threshold_d, floor_d)

    This ensures:
    1. Significant coordinate directions are captured (variance threshold)
    2. Catastrophic under-exploration is prevented (floor)
    """

    name: str = "ASTROMORF_VarFloor"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.target_variance = 0.95  # Capture 95% of variance
        self.floor_fraction = 0.10  # Minimum 10% of problem dim
        self.min_absolute_d = 2  # Absolute minimum

    def compute_optimal_subspace_dimension(self) -> None:
        """Compute optimal subspace dimension."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))
        max_test_d = min(self.max_d, self.problem.dim - 1)

        # 1. Find dimension needed for target variance capture
        cumvar = np.cumsum(eig_source) / total_var
        variance_d = int(np.searchsorted(cumvar, self.target_variance) + 1)
        variance_d = min(max(1, variance_d), max_test_d)

        # 2. Compute floor dimension
        floor_d = max(self.min_absolute_d, int(self.problem.dim * self.floor_fraction))
        floor_d = min(floor_d, max_test_d)

        # 3. Take the larger of the two
        optimal_d = max(variance_d, floor_d)

        # Log what's happening
        var_captured = cumvar[optimal_d - 1] if optimal_d <= len(cumvar) else 1.0

        if optimal_d != self.d:
            reason = "variance" if variance_d >= floor_d else "floor"
            print(
                f"Iteration {self.iteration_count}: [VarFloor/{reason}] Adaptive subspace dimension change: {self.d} -> {optimal_d} (captures {var_captured * 100:.1f}% variance)"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """Score based on variance capture and cost."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))
        cumvar = np.cumsum(eig_source) / total_var

        # Find target dimension
        variance_d = int(np.searchsorted(cumvar, self.target_variance) + 1)
        floor_d = max(self.min_absolute_d, int(self.problem.dim * self.floor_fraction))
        target_d = max(variance_d, floor_d)
        target_d = min(max(1, target_d), max_test_d)

        for d in range(1, max_test_d + 1):
            # Score highest at target_d
            distance = abs(d - target_d)
            score = 1.0 / (1.0 + 0.5 * distance)

            var_captured = cumvar[d - 1] if d <= len(cumvar) else 1.0

            results[d] = {
                "score": float(score),
                "s_proj": float(var_captured),
                "is_target": d == target_d,
                "variance_d": variance_d,
                "floor_d": floor_d,
            }

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6


class ASTROMORF_AdaptiveFloor(ASTROMORF):  # noqa: N801
    """ConservativeInactiveMin with adaptive expansion on plateau.

    Key idea:
    1. Start with floor dimension (10% of problem dim)
    2. Use InactiveMin scoring within floor constraints
    3. On plateau (no progress), expand to higher dimension (30% of dim)
    4. But never go to full d=dim-1 like Current does

    This combines:
    - ConservativeInactiveMin's efficiency (d=10 baseline)
    - Current's ability to escape local optima (expand on plateau)
    - But with bounded expansion (30% not 99%)
    """

    name: str = "ASTROMORF_AdaptiveFloor"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.base_floor_fraction = 0.10  # 10% floor normally
        self.expanded_floor_fraction = 0.30  # 30% when stuck
        self.min_absolute_d = 2
        self.plateau_threshold = 0.01  # Relative improvement threshold
        self.plateau_window = 5  # Iterations to detect plateau
        self.is_expanded = False
        self.recent_objectives = []
        self.expansion_count = 0
        self.max_expansions = 3  # Don't expand more than 3 times

    def compute_optimal_subspace_dimension(self) -> None:
        """Compute optimal subspace dimension."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        max_test_d = min(self.max_d, self.problem.dim - 1)

        # Track recent objectives for plateau detection
        if hasattr(self, "fn_estimates") and self.fn_estimates:
            self.recent_objectives = list(self.fn_estimates[-self.plateau_window :])

        # Check for plateau
        is_plateau = self._detect_plateau()

        # Decide floor based on plateau status
        if (
            is_plateau
            and not self.is_expanded
            and self.expansion_count < self.max_expansions
        ):
            self.is_expanded = True
            self.expansion_count += 1
            floor_fraction = self.expanded_floor_fraction
            reason = f"plateau_expand_{self.expansion_count}"
        elif self.is_expanded:
            floor_fraction = self.expanded_floor_fraction
            reason = "expanded"
        else:
            floor_fraction = self.base_floor_fraction
            reason = "floor"

        # Compute floor dimension
        floor_d = max(self.min_absolute_d, int(self.problem.dim * floor_fraction))
        floor_d = min(floor_d, max_test_d)

        # Use InactiveMin scoring but enforce floor
        scores = self.evaluate_and_score_candidate_dimensions()
        if scores:
            best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
        else:
            best_d = self.d

        # Apply floor
        optimal_d = max(floor_d, min(int(best_d), max_test_d))

        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: [AdaptiveFloor/{reason}] Adaptive subspace dimension change: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def _detect_plateau(self) -> bool:
        """Detect if optimization is on a plateau."""
        if len(self.recent_objectives) < self.plateau_window:
            return False

        # Compute relative improvement
        objs = np.array(self.recent_objectives)
        best_early = np.max(objs[:2]) if len(objs) >= 2 else objs[0]
        best_late = np.max(objs[-2:]) if len(objs) >= 2 else objs[-1]

        # For maximization, check if we're improving
        if hasattr(self, "problem") and self.problem.minmax[0] == 1:
            relative_improvement = (best_late - best_early) / (abs(best_early) + 1e-10)
        else:
            # For minimization, improvement is decrease
            relative_improvement = (best_early - best_late) / (abs(best_early) + 1e-10)

        return relative_improvement < self.plateau_threshold

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """Same scoring as InactiveMin."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost = max(costs.values())

        validation_by_d = self._get_validation_by_d()

        for d in range(1, max_test_d + 1):
            inactive_sum = np.sum(eig_source[d:])
            inactive_fraction = inactive_sum / total_var
            active_sum = np.sum(eig_source[:d])
            active_fraction = active_sum / total_var

            val_err = validation_by_d.get(d, 0.5)
            model_quality = 1.0 - val_err
            cost_fraction = costs[d] / max_cost

            score = (
                0.4 * active_fraction
                + 0.3 * (1.0 - inactive_fraction)
                + 0.2 * model_quality
                - 0.1 * cost_fraction
            )

            results[d] = {
                "score": float(max(0, score)),
                "s_proj": float(active_fraction),
                "inactive_fraction": float(inactive_fraction),
                "s_val": float(model_quality),
                "cost": float(cost_fraction),
            }

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d
        vals, counts = {}, {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: vals[k] / counts[k] for k in vals} if counts else {}


class ASTROMORF_HybridCostPareto(ASTROMORF):  # noqa: N801
    """Hybrid of CostPenalized and Pareto:.

    - Uses CostPenalized scoring as base
    - Switches to Pareto-style when dimension is unstable (oscillating).
    """

    name: str = "ASTROMORF_HybridCostPareto"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.cost_weight = 0.3
        self.dimension_history = []
        self.oscillation_threshold = (
            3  # Switch to Pareto if d changes 3+ times in last 5 iters
        )

    def compute_optimal_subspace_dimension(self) -> None:
        """Compute optimal subspace dimension."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        # Detect dimension oscillation
        is_oscillating = self._detect_oscillation()

        if is_oscillating:
            scores = self.evaluate_pareto_style()
        else:
            scores = self.evaluate_cost_penalized()

        if not scores:
            return

        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
        optimal_d = max(1, min(int(best_d), self.problem.dim - 1))

        if optimal_d != self.d:
            mode = "Pareto" if is_oscillating else "Cost"
            print(
                f"Iteration {self.iteration_count}: [Hybrid/{mode}] Adaptive subspace dimension change: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.dimension_history.append(self.d)
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def _detect_oscillation(self) -> bool:
        """Detect if dimension has been oscillating."""
        if len(self.dimension_history) < 5:
            return False

        recent = self.dimension_history[-5:]
        changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
        return changes >= self.oscillation_threshold

    def evaluate_cost_penalized(self) -> dict:
        """Standard cost-penalized scoring."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost, min_cost = max(costs.values()), min(costs.values())
        cost_range = max(1, max_cost - min_cost)

        validation_by_d = self._get_validation_by_d()

        for d in range(1, max_test_d + 1):
            val_err = validation_by_d.get(d, 0.5)
            model_quality = 1.0 - val_err
            var_captured = np.sum(eig_source[:d]) / total_var
            normalized_cost = (costs[d] - min_cost) / cost_range

            score = (
                0.5 * model_quality
                + 0.3 * var_captured
                - self.cost_weight * normalized_cost
            )

            results[d] = {"score": float(max(0, score))}

        return results

    def evaluate_pareto_style(self) -> dict:
        """More conservative Pareto-style scoring when oscillating."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost = max(costs.values())

        validation_by_d = self._get_validation_by_d()

        # Favor stability - bias toward current dimension
        current_d = self.d

        for d in range(1, max_test_d + 1):
            val_err = validation_by_d.get(d, 0.5)
            model_quality = 1.0 - val_err
            var_captured = np.sum(eig_source[:d]) / total_var
            cost_fraction = costs[d] / max_cost

            # Base score
            score = 0.5 * model_quality + 0.35 * var_captured - 0.15 * cost_fraction

            # Stability bonus - prefer staying at current dimension
            if d == current_d:
                score += 0.1

            results[d] = {"score": float(max(0, score))}

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d
        vals, counts = {}, {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: vals[k] / counts[k] for k in vals} if counts else {}


# =============================================================================
# NEW IMPROVED POLICIES BASED ON EXPERIMENT LEARNINGS
# =============================================================================


class ASTROMORF_ValidationGatedFloor(ASTROMORF):  # noqa: N801
    """Uses 10% floor as safety, but allows reduction below floor ONLY if.

    validation error at lower d is competitive with validation at floor.

    Key insight: InactiveMin fails on ROSENBROCK because it goes to d=1-2
    without checking if the model is actually good at those dimensions.
    This policy gates dimension reduction on validation performance.

    Rules:
    1. Default floor = 10% of problem dim (like ConservativeInactiveMin)
    2. Absolute minimum = 5 (never go to d=1-2)
    3. Can reduce below floor if validation_err(low_d) <= 1.3 * validation_err(floor_d)
    """

    name: str = "ASTROMORF_ValidationGatedFloor"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.floor_fraction = 0.10  # 10% floor
        self.absolute_min = 5  # Never below 5
        self.validation_tolerance = 1.3  # Allow 30% worse validation for lower d
        self.cost_weight = 0.1

    def compute_optimal_subspace_dimension(self) -> None:
        """Compute optimal subspace dimension."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        max_test_d = min(self.max_d, self.problem.dim - 1)
        floor_d = max(self.absolute_min, int(self.problem.dim * self.floor_fraction))
        floor_d = min(floor_d, max_test_d)

        # Get validation errors by dimension
        validation_by_d = self._get_validation_by_d()

        # Get scores for all dimensions
        scores = self.evaluate_and_score_candidate_dimensions()
        if not scores:
            return

        # Find best dimension according to scoring
        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]

        # Check if we can go below floor
        if best_d < floor_d and best_d >= self.absolute_min:
            # Get validation at floor and at best_d
            val_at_floor = validation_by_d.get(floor_d, 0.5)
            val_at_best = validation_by_d.get(best_d, 0.5)

            # Only allow reduction if validation at lower d is competitive
            if val_at_best <= self.validation_tolerance * val_at_floor:
                optimal_d = best_d
                reason = f"validation_gated({val_at_best:.3f}<={self.validation_tolerance}*{val_at_floor:.3f})"  # noqa: E501
            else:
                optimal_d = floor_d
                reason = f"floor_enforced(val {val_at_best:.3f}>{self.validation_tolerance}*{val_at_floor:.3f})"  # noqa: E501
        else:
            # Either best_d >= floor, or best_d < absolute_min
            optimal_d = max(floor_d, min(int(best_d), max_test_d))
            reason = "standard"

        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: [ValidationGatedFloor/{reason}] d: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """InactiveMin-style scoring."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost = max(costs.values())

        validation_by_d = self._get_validation_by_d()

        for d in range(1, max_test_d + 1):
            active_fraction = np.sum(eig_source[:d]) / total_var
            inactive_fraction = np.sum(eig_source[d:]) / total_var
            val_err = validation_by_d.get(d, 0.5)
            model_quality = 1.0 - val_err
            cost_fraction = costs[d] / max_cost

            score = (
                0.4 * active_fraction
                + 0.3 * (1.0 - inactive_fraction)
                + 0.2 * model_quality
                - self.cost_weight * cost_fraction
            )

            results[d] = {"score": float(max(0, score)), "val_err": float(val_err)}

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d
        vals, counts = {}, {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: vals[k] / counts[k] for k in vals} if counts else {}


class ASTROMORF_SuccessGatedFloor(ASTROMORF):  # noqa: N801
    """Uses 10% floor but allows reduction if recent success rate at lower d is high.

    Key insight: On SAN-1, InactiveMin succeeds at d=2 because the problem
    truly is low-dimensional. On ROSENBROCK, d=2 fails with low success rate.

    Rules:
    1. Default floor = 10% of problem dim
    2. Absolute minimum = 5
    3. Track success rate by dimension
    4. Can reduce below floor only if success_rate(low_d) >= 0.5
    """

    name: str = "ASTROMORF_SuccessGatedFloor"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.floor_fraction = 0.10
        self.absolute_min = 5
        self.min_success_rate = 0.4  # Need 40% success rate to go below floor
        self.cost_weight = 0.1
        self.success_by_d = {}  # Track {d: (successes, total)}

    def compute_optimal_subspace_dimension(self) -> None:
        """Compute optimal subspace dimension."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        max_test_d = min(self.max_d, self.problem.dim - 1)
        floor_d = max(self.absolute_min, int(self.problem.dim * self.floor_fraction))
        floor_d = min(floor_d, max_test_d)

        # Update success tracking
        self._update_success_tracking()

        scores = self.evaluate_and_score_candidate_dimensions()
        if not scores:
            return

        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]

        # Check if we can go below floor
        if best_d < floor_d and best_d >= self.absolute_min:
            success_rate = self._get_success_rate_for_d(best_d)

            if success_rate >= self.min_success_rate:
                optimal_d = best_d
                reason = (
                    f"success_gated(rate={success_rate:.2f}>={self.min_success_rate})"
                )
            else:
                optimal_d = floor_d
                reason = (
                    f"floor_enforced(rate={success_rate:.2f}<{self.min_success_rate})"
                )
        else:
            optimal_d = max(floor_d, min(int(best_d), max_test_d))
            reason = "standard"

        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: [SuccessGatedFloor/{reason}] d: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def _update_success_tracking(self) -> None:
        """Update success rate tracking by dimension."""
        for info in self.previous_model_information:
            d = info.get("recommended_dimension")
            if d is not None:
                d = int(d)
                if d not in self.success_by_d:
                    self.success_by_d[d] = [0, 0]
                # Count this iteration
                self.success_by_d[d][1] += 1
                if info.get("model_success"):
                    self.success_by_d[d][0] += 1

    def _get_success_rate_for_d(self, d: int) -> float:
        """Get success rate for dimension d."""
        if d in self.success_by_d and self.success_by_d[d][1] > 0:
            return self.success_by_d[d][0] / self.success_by_d[d][1]
        # Default: assume moderate success if no data
        return 0.5

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """InactiveMin-style scoring."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost = max(costs.values())

        validation_by_d = self._get_validation_by_d()

        for d in range(1, max_test_d + 1):
            active_fraction = np.sum(eig_source[:d]) / total_var
            inactive_fraction = np.sum(eig_source[d:]) / total_var
            val_err = validation_by_d.get(d, 0.5)
            model_quality = 1.0 - val_err
            cost_fraction = costs[d] / max_cost

            score = (
                0.4 * active_fraction
                + 0.3 * (1.0 - inactive_fraction)
                + 0.2 * model_quality
                - self.cost_weight * cost_fraction
            )

            results[d] = {"score": float(max(0, score))}

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d
        vals, counts = {}, {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: vals[k] / counts[k] for k in vals} if counts else {}


class ASTROMORF_LowerFloor(ASTROMORF):  # noqa: N801
    """Simple policy: Use 5% floor instead of 10%.

    Hypothesis: 10% floor (d=10) might be too conservative for some problems.
    5% floor (d=5) prevents catastrophic d=1-2 collapse while allowing more efficiency.
    """

    name: str = "ASTROMORF_LowerFloor"

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors=fixed_factors)
        self.floor_fraction = 0.05  # 5% floor instead of 10%
        self.absolute_min = 3  # Never below 3
        self.cost_weight = 0.1

    def compute_optimal_subspace_dimension(self) -> None:
        """Compute optimal subspace dimension."""
        if not self.previous_model_information and not hasattr(
            self, "last_validation_by_d"
        ):
            return

        max_test_d = min(self.max_d, self.problem.dim - 1)
        floor_d = max(self.absolute_min, int(self.problem.dim * self.floor_fraction))
        floor_d = min(floor_d, max_test_d)

        scores = self.evaluate_and_score_candidate_dimensions()
        if not scores:
            return

        best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
        optimal_d = max(floor_d, min(int(best_d), max_test_d))

        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: [LowerFloor] d: {self.d} -> {optimal_d}"  # noqa: E501
            )
            self.d = optimal_d
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            self.last_d_change_iteration = self.iteration_count
            if hasattr(self, "d_history"):
                self.d_history.append(self.d)

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """InactiveMin-style scoring."""
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        eig_source = self._get_eigenvalue_spectrum()
        total_var = max(1e-12, float(np.sum(eig_source)))

        degree = self.factors.get("polynomial degree", 2)
        costs = {
            d: int(comb(d + degree, degree, exact=True)) + 1
            for d in range(1, max_test_d + 1)
        }
        max_cost = max(costs.values())

        validation_by_d = self._get_validation_by_d()

        for d in range(1, max_test_d + 1):
            active_fraction = np.sum(eig_source[:d]) / total_var
            inactive_fraction = np.sum(eig_source[d:]) / total_var
            val_err = validation_by_d.get(d, 0.5)
            model_quality = 1.0 - val_err
            cost_fraction = costs[d] / max_cost

            score = (
                0.4 * active_fraction
                + 0.3 * (1.0 - inactive_fraction)
                + 0.2 * model_quality
                - self.cost_weight * cost_fraction
            )

            results[d] = {"score": float(max(0, score))}

        return results

    def _get_eigenvalue_spectrum(self) -> np.ndarray:
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            return np.array(self.gradient_eigenvalues[-1], dtype=float)
        for info in reversed(self.previous_model_information):
            spec = info.get("eigenvalue_spectrum")
            if spec is not None:
                return np.array(spec, dtype=float)
        return np.ones(self.problem.dim) * 1e-6

    def _get_validation_by_d(self) -> dict:
        if hasattr(self, "last_validation_by_d") and self.last_validation_by_d:
            return self.last_validation_by_d
        vals, counts = {}, {}
        for info in self.previous_model_information:
            vbd = info.get("validation_by_d")
            if isinstance(vbd, dict):
                for k, v in vbd.items():
                    k = int(k)
                    vals[k] = vals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: vals[k] / counts[k] for k in vals} if counts else {}


# =============================================================================
# SOLVER REGISTRY
# =============================================================================

POLICY_SOLVERS = {
    "Current": ASTROMORF,
    "CostPenalized": ASTROMORF_CostPenalized,
    "InactiveMin": ASTROMORF_InactiveMinimization,
    "Pareto": ASTROMORF_ParetoOptimal,
    "VarThreshold": ASTROMORF_VarianceThreshold,
    # New improved policies
    "AdaptiveCost": ASTROMORF_AdaptiveCost,
    "StructureAware": ASTROMORF_StructureAware,
    "ConservativeInactiveMin": ASTROMORF_ConservativeInactiveMin,
    "HybridCostPareto": ASTROMORF_HybridCostPareto,
    "VarFloor": ASTROMORF_VarFloor,
    "AdaptiveFloor": ASTROMORF_AdaptiveFloor,
    # Newest policies based on experiment learnings
    "ValidationGatedFloor": ASTROMORF_ValidationGatedFloor,
    "SuccessGatedFloor": ASTROMORF_SuccessGatedFloor,
    "LowerFloor": ASTROMORF_LowerFloor,
}


# =============================================================================
# EXPERIMENT INFRASTRUCTURE
# =============================================================================


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    policy_name: str
    problem_name: str
    n_macroreps: int
    budget: int

    final_objectives: list[float] = field(default_factory=list)
    best_objectives: list[float] = field(default_factory=list)
    iterations_completed: list[int] = field(default_factory=list)
    samples_used: list[int] = field(default_factory=list)
    dimensions_used: list[list[int]] = field(default_factory=list)
    mean_dimension: float = 0.0
    success_rates: list[float] = field(default_factory=list)
    wall_clock_times: list[float] = field(default_factory=list)

    def compute_summary(self) -> dict:
        """Compute summary."""
        return {
            "policy_name": self.policy_name,
            "problem_name": self.problem_name,
            "n_macroreps": self.n_macroreps,
            "budget": self.budget,
            "final_objective_mean": float(np.mean(self.final_objectives))
            if self.final_objectives
            else None,
            "final_objective_std": float(np.std(self.final_objectives))
            if self.final_objectives
            else None,
            "best_objective_mean": float(np.mean(self.best_objectives))
            if self.best_objectives
            else None,
            "iterations_mean": float(np.mean(self.iterations_completed))
            if self.iterations_completed
            else None,
            "samples_mean": float(np.mean(self.samples_used))
            if self.samples_used
            else None,
            "mean_dimension": self.mean_dimension,
            "success_rate_mean": float(np.mean(self.success_rates))
            if self.success_rates
            else None,
            "wall_time_mean": float(np.mean(self.wall_clock_times))
            if self.wall_clock_times
            else None,
        }


class MacroreplicationResult(TypedDict):
    """Typed container for one macroreplication output."""

    final_objective: float | None
    best_objective: float | None
    iterations: int | None
    samples_used: int | None
    dimensions_used: list[int]
    success_rate: float | None
    wall_time: float | None
    error: str | None


class PolicyScore(TypedDict):
    """Aggregate score tracking for each policy."""

    wins: int
    mean_obj: list[float]
    mean_dim: list[float]


def run_single_macroreplication(
    problem_name: str,
    solver_class: type,
    budget: int,
    solver_factors: dict,
    seed: int | None = None,
    dimension: int | None = None,
    n_postreps: int = 100,
) -> MacroreplicationResult:
    """Run a single macroreplication with a specific solver class.

    Args:
        problem_name: Name of the problem
        solver_class: Solver class to use
        budget: Simulation budget
        solver_factors: Solver configuration factors
        seed: Random seed for RNG
        dimension: If provided, scale problem to this dimension
        n_postreps: Number of post-replications for evaluating recommended solution
    """
    from mrg32k3a.mrg32k3a import MRG32k3a
    from simopt.base import Solution
    from simopt.directory import problem_directory
    from simopt.experiment.run_solver import _set_up_rngs

    results: MacroreplicationResult = {
        "final_objective": None,
        "best_objective": None,
        "iterations": None,
        "samples_used": None,
        "dimensions_used": [],
        "success_rate": None,
        "wall_time": None,
        "error": None,
    }

    try:
        # Create problem (with optional dimension scaling)
        if dimension is not None and problem_name in SCALABLE_PROBLEMS:
            problem = scale_dimension(problem_name, dimension, budget)
        else:
            problem = problem_directory[problem_name](fixed_factors={"budget": budget})

        # Create solver (using our specific subclass)
        solver = solver_class(fixed_factors=solver_factors)

        # Set up RNGs properly using the same function as run_solver
        mrep = seed if seed is not None else 0
        _set_up_rngs(solver, problem, mrep)

        start_time = time.perf_counter()

        # Run solver - use run() not solve() to set up budget etc.
        # run() returns (solution_df, iteration_df) and resets internal state
        solution_df, iteration_df = solver.run(problem)

        results["wall_time"] = time.perf_counter() - start_time

        # Extract the final recommended solution and post-evaluate it properly
        if solution_df is not None and len(solution_df) > 0:
            # Get the final recommended solution (last row)
            final_solution_x = solution_df["solution"].iloc[-1]

            # Post-evaluate: simulate the solution n_postreps times with fresh RNGs
            fresh_soln = Solution(final_solution_x, problem)

            # Create fresh RNGs for post-replication (different from optimization RNGs)
            postrep_rngs = [
                MRG32k3a(s_ss_sss_index=[1, problem.model.n_rngs + rng_index, mrep])
                for rng_index in range(problem.model.n_rngs)
            ]
            fresh_soln.attach_rngs(rng_list=postrep_rngs, copy=True)

            # Simulate the solution
            problem.simulate(solution=fresh_soln, num_macroreps=n_postreps)

            # Get the mean objective from post-replications
            objectives = fresh_soln.objectives[
                : fresh_soln.n_reps, 0
            ]  # Assuming single objective
            results["final_objective"] = float(np.mean(objectives))

        # For best objective, we'd need to post-evaluate all intermediate solutions
        # For now, use the final objective as an approximation
        results["best_objective"] = results["final_objective"]

        if iteration_df is not None:
            results["iterations"] = len(iteration_df)

        if hasattr(solver, "budget") and hasattr(solver.budget, "used"):
            results["samples_used"] = int(solver.budget.used)

        if hasattr(solver, "d_history") and solver.d_history:
            results["dimensions_used"] = list(solver.d_history)

        if hasattr(solver, "successful_iterations") and hasattr(
            solver, "unsuccessful_iterations"
        ):
            total = len(solver.successful_iterations) + len(
                solver.unsuccessful_iterations
            )
            if total > 0:
                results["success_rate"] = len(solver.successful_iterations) / total

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Error in macroreplication: {e}")
        traceback.print_exc()

    return results


def run_policy_experiment(
    problem_name: str,
    policy_name: str,
    solver_class: type,
    budget: int,
    n_macroreps: int,
    solver_factors: dict,
    fixed_dim: int | None = None,
    problem_dimension: int | None = None,
) -> ExperimentResult:
    """Run experiment for a single policy.

    Args:
        problem_name: Name of the problem
        policy_name: Name of the policy being tested
        solver_class: Solver class to use
        budget: Simulation budget
        n_macroreps: Number of macroreplications
        solver_factors: Solver configuration factors
        fixed_dim: If set, use fixed subspace dimension
        problem_dimension: If set, scale problem to this dimension
    """
    if fixed_dim is not None:
        display_name = f"Fixed_d{fixed_dim}"
        solver_factors = solver_factors.copy()
        solver_factors["initial subspace dimension"] = fixed_dim
        solver_factors["adaptive subspace dimension"] = False
    else:
        display_name = policy_name
        solver_factors = solver_factors.copy()
        solver_factors["adaptive subspace dimension"] = True

    logger.info(f"Running policy: {display_name}")

    result = ExperimentResult(
        policy_name=display_name,
        problem_name=problem_name,
        n_macroreps=n_macroreps,
        budget=budget,
    )

    all_dims = []

    for mrep in range(n_macroreps):
        logger.info(f"  Macroreplication {mrep + 1}/{n_macroreps}")

        mrep_result = run_single_macroreplication(
            problem_name=problem_name,
            solver_class=solver_class,
            budget=budget,
            solver_factors=solver_factors,
            seed=mrep * 100 + 42,
            dimension=problem_dimension,
        )

        if mrep_result["error"]:
            logger.warning(f"  Error: {mrep_result['error']}")
            continue

        if mrep_result["wall_time"] is not None:
            result.wall_clock_times.append(mrep_result["wall_time"])
        if mrep_result["final_objective"] is not None:
            result.final_objectives.append(mrep_result["final_objective"])
        if mrep_result["best_objective"] is not None:
            result.best_objectives.append(mrep_result["best_objective"])
        if mrep_result["iterations"] is not None:
            result.iterations_completed.append(mrep_result["iterations"])
        if mrep_result["samples_used"] is not None:
            result.samples_used.append(mrep_result["samples_used"])
        if mrep_result["dimensions_used"]:
            result.dimensions_used.append(mrep_result["dimensions_used"])
            all_dims.extend(mrep_result["dimensions_used"])
        if mrep_result["success_rate"] is not None:
            result.success_rates.append(mrep_result["success_rate"])

    if all_dims:
        result.mean_dimension = float(np.mean(all_dims))

    return result


def print_results_table(
    results: dict[str, ExperimentResult], problem_name: str
) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print(f"RESULTS: {problem_name}")
    print("=" * 100)

    # Header
    headers = [
        "Policy",
        "Final Obj",
        "Best Obj",
        "Mean Dim",
        "Iters",
        "Samples",
        "Success%",
        "Time(s)",
    ]
    widths = [20, 12, 12, 10, 8, 10, 10, 10]

    header_str = " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False))
    print(header_str)
    print("-" * len(header_str))

    for policy_name, result in results.items():
        summary = result.compute_summary()

        def fmt(val: float | None = None, precision: int = 4) -> str:
            if val is None:
                return "N/A"
            if isinstance(val, float):
                if abs(val) > 1000 or abs(val) < 0.01:
                    return f"{val:.2e}"
                return f"{val:.{precision}f}"
            return str(val)

        row = [
            policy_name[:20],
            fmt(summary["final_objective_mean"]),
            fmt(summary["best_objective_mean"]),
            fmt(summary["mean_dimension"], 2),
            fmt(summary["iterations_mean"], 1),
            fmt(summary["samples_mean"], 0),
            fmt(
                summary["success_rate_mean"] * 100
                if summary["success_rate_mean"]
                else None,
                1,
            ),
            fmt(summary["wall_time_mean"], 2),
        ]

        row_str = " | ".join(str(v).ljust(w) for v, w in zip(row, widths, strict=False))
        print(row_str)

    print("=" * 100)


def main() -> None:
    """Run main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare ASTROMoRF dimension selection policies"
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        type=str,
        default=["DYNAMNEWS-1"],
        help="Problem names (e.g., DYNAMNEWS-1 SAN-1 ROSENBROCK-1)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        help="Scale problems to this dimension (only for scalable problems)",
    )
    parser.add_argument("--budget", type=int, default=500, help="Simulation budget")
    parser.add_argument(
        "--macroreps", type=int, default=5, help="Number of macroreplications"
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help="Policies to test (Current, CostPenalized, InactiveMin, Pareto, VarThreshold)",  # noqa: E501
    )
    parser.add_argument(
        "--fixed-dims",
        nargs="+",
        type=int,
        default=[],
        help="Fixed dimensions to test as baselines",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with minimal settings"
    )
    parser.add_argument(
        "--output", type=str, default="policy_results.json", help="Output file"
    )

    args = parser.parse_args()

    if args.quick:
        args.budget = 200
        args.macroreps = 2
        args.policies = ["Current", "CostPenalized", "Pareto"]

    policies = args.policies or list(POLICY_SOLVERS.keys())

    # Validate policies
    for p in policies:
        if p not in POLICY_SOLVERS:
            logger.error(
                f"Unknown policy: {p}. Available: {list(POLICY_SOLVERS.keys())}"
            )
            return

    print("=" * 80)
    print("ASTROMORF DIMENSION POLICY COMPARISON (v3 - True Subclasses)")
    print("=" * 80)
    print(f"Problems: {args.problems}")
    print(f"Problem dimension: {args.dimension if args.dimension else 'default'}")
    print(f"Budget: {args.budget}")
    print(f"Macroreplications: {args.macroreps}")
    print(f"Policies: {policies}")
    print(f"Fixed dimensions: {args.fixed_dims if args.fixed_dims else 'None'}")
    print("=" * 80)

    base_solver_factors = {
        "Record Diagnostics": False,
        "polynomial degree": 2,
    }

    all_results = {}  # {problem_name: {policy_name: ExperimentResult}}

    for problem_name in args.problems:
        print(f"\n{'=' * 80}")
        print(f"RUNNING PROBLEM: {problem_name}")
        print(f"{'=' * 80}")

        results = {}

        # Run each policy
        for policy_name in policies:
            solver_class = POLICY_SOLVERS[policy_name]
            result = run_policy_experiment(
                problem_name=problem_name,
                policy_name=policy_name,
                solver_class=solver_class,
                budget=args.budget,
                n_macroreps=args.macroreps,
                solver_factors=base_solver_factors,
                problem_dimension=args.dimension,
            )
            results[policy_name] = result

        # Run fixed dimension baselines
        for fixed_dim in args.fixed_dims:
            result = run_policy_experiment(
                problem_name=problem_name,
                policy_name=f"Fixed_d{fixed_dim}",
                solver_class=ASTROMORF,  # Use base class for fixed
                budget=args.budget,
                n_macroreps=args.macroreps,
                solver_factors=base_solver_factors,
                fixed_dim=fixed_dim,
                problem_dimension=args.dimension,
            )
            results[f"Fixed_d{fixed_dim}"] = result

        # Print results for this problem
        print_results_table(results, problem_name)
        all_results[problem_name] = results

    # Save to JSON
    output_data = {
        "config": {
            "problems": args.problems,
            "dimension": args.dimension,
            "budget": args.budget,
            "macroreps": args.macroreps,
            "policies": policies,
            "fixed_dims": args.fixed_dims,
        },
        "results": {
            prob: {k: v.compute_summary() for k, v in res.items()}
            for prob, res in all_results.items()
        },
    }

    with open(args.output, "w") as f:  # noqa: PTH123
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Overall Analysis across all problems
    print("\n" + "=" * 80)
    print("OVERALL ANALYSIS ACROSS ALL PROBLEMS")
    print("=" * 80)

    # Aggregate results by policy
    policy_scores: dict[str, PolicyScore] = {
        p: {"wins": 0, "mean_obj": [], "mean_dim": []} for p in policies
    }

    for _problem_name, results in all_results.items():
        valid_results = {
            k: v for k, v in results.items() if v.final_objectives and k in policies
        }
        if not valid_results:
            continue

        # Find best for this problem (assuming maximization for most problems)
        best_policy = max(
            valid_results.items(), key=lambda x: np.mean(x[1].final_objectives)
        )
        policy_scores[best_policy[0]]["wins"] += 1

        for policy_name, result in valid_results.items():
            policy_scores[policy_name]["mean_obj"].append(
                np.mean(result.final_objectives)
            )
            if result.mean_dimension > 0:
                policy_scores[policy_name]["mean_dim"].append(result.mean_dimension)

    print("\nPolicy Performance Summary:")
    print("-" * 60)
    print(f"{'Policy':<20} | {'Wins':<6} | {'Avg Obj':<12} | {'Avg Dim':<8}")
    print("-" * 60)
    for policy_name in policies:
        scores = policy_scores[policy_name]
        avg_obj = np.mean(scores["mean_obj"]) if scores["mean_obj"] else float("nan")
        avg_dim = np.mean(scores["mean_dim"]) if scores["mean_dim"] else float("nan")
        print(
            f"{policy_name:<20} | {scores['wins']:<6} | {avg_obj:<12.4f} | {avg_dim:<8.2f}"  # noqa: E501
        )
    print("-" * 60)

    # Best overall
    best_policy = max(policy_scores.items(), key=lambda x: x[1]["wins"])
    print(
        f"\nBest overall policy (by wins): {best_policy[0]} ({best_policy[1]['wins']} wins)"  # noqa: E501
    )

    # Most dimension-efficient
    dim_efficient = min(
        [
            (p, np.mean(s["mean_dim"]))
            for p, s in policy_scores.items()
            if s["mean_dim"]
        ],
        key=lambda x: x[1],
    )
    print(
        f"Most dimension-efficient: {dim_efficient[0]} (avg d={dim_efficient[1]:.2f})"
    )


# =============================================================================
# PYTEST COVERAGE
# =============================================================================


def test_policy_registry_uses_astromorf_subclasses() -> None:
    """All registered policy solvers should derive from ASTROMORF."""
    for policy_name, solver_cls in POLICY_SOLVERS.items():
        assert issubclass(solver_cls, ASTROMORF), (
            f"Policy '{policy_name}' is not an ASTROMORF subclass."
        )


@pytest.mark.slow
def test_run_single_macroreplication_smoke() -> None:
    """Run one lightweight real macroreplication to validate harness wiring."""
    result = run_single_macroreplication(
        problem_name="ROSENBROCK-1",
        solver_class=ASTROMORF,
        budget=120,
        solver_factors={"Record Diagnostics": False, "polynomial degree": 2},
        seed=123,
        n_postreps=5,
    )

    assert result["error"] is None, result["error"]
    assert result["final_objective"] is not None
    assert result["best_objective"] is not None
    assert result["iterations"] is not None


def test_run_policy_experiment_aggregates_macroreps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy experiment should aggregate macrorep outputs and compute mean dim."""
    mock_results: list[MacroreplicationResult] = [
        {
            "final_objective": 1.0,
            "best_objective": 0.8,
            "iterations": 5,
            "samples_used": 90,
            "dimensions_used": [4, 3, 2],
            "success_rate": 0.5,
            "wall_time": 0.1,
            "error": None,
        },
        {
            "final_objective": 1.2,
            "best_objective": 0.7,
            "iterations": 6,
            "samples_used": 95,
            "dimensions_used": [3, 2, 2],
            "success_rate": 0.6,
            "wall_time": 0.2,
            "error": None,
        },
        {
            "final_objective": 0.9,
            "best_objective": 0.6,
            "iterations": 4,
            "samples_used": 88,
            "dimensions_used": [2, 2, 1],
            "success_rate": 0.4,
            "wall_time": 0.15,
            "error": None,
        },
    ]

    state = {"idx": 0}

    def _mock_run_single_macroreplication(**_: object) -> MacroreplicationResult:
        out = mock_results[state["idx"]]
        state["idx"] += 1
        return out

    monkeypatch.setattr(
        sys.modules[__name__],
        "run_single_macroreplication",
        _mock_run_single_macroreplication,
    )

    result = run_policy_experiment(
        problem_name="ROSENBROCK-1",
        policy_name="Current",
        solver_class=ASTROMORF,
        budget=120,
        n_macroreps=3,
        solver_factors={"Record Diagnostics": False, "polynomial degree": 2},
    )

    assert result.policy_name == "Current"
    assert len(result.final_objectives) == 3
    assert len(result.best_objectives) == 3
    assert len(result.iterations_completed) == 3
    assert len(result.samples_used) == 3
    assert len(result.success_rates) == 3
    assert len(result.wall_clock_times) == 3
    assert len(result.dimensions_used) == 3
    assert result.mean_dimension == pytest.approx(2.3333333333333335)


if __name__ == "__main__":
    main()
