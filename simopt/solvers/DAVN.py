"""Displacement Adjusted Virtual Nesting (DAVN) solver.

This solver targets the ``VANRYZIN-2`` multistage problem by:

1. Solving a deterministic LP (DLP) relaxation using expected demand.
2. Extracting leg bid prices from the dual LP.
3. Computing displacement-adjusted product revenues.
4. Clustering classes into leg-wise buckets using adjusted leg revenues.
5. Building leg booking limits (high-to-low buckets with monotone limits).
6. Converting booking limits to theft-nesting protection levels.

The implementation focuses on the partitioning core of DAVN and produces a
single high-quality recommended solution within the solver budget.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field
from scipy.optimize import linprog
from scipy.stats import norm

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    MultistageProblem,
    ObjectiveType,
    ProblemLike,
    Solver,
    SolverConfig,
    SolverProblemType,
    VariableType,
)


class DAVNConfig(SolverConfig):
    """Configuration for DAVN solver."""

    n_virtual_classes: Annotated[
        int | None,
        Field(
            default=None,
            gt=1,
            description=(
                "Number of virtual classes used in the DAVN partition. "
                "If omitted, uses the model's n_virtual_classes."
            ),
        ),
    ]
    evaluation_reps: Annotated[
        int,
        Field(
            default=5,
            gt=0,
            description="Replications used to score the final DAVN decision.",
        ),
    ]


class DAVN(Solver):
    """Displacement Adjusted Virtual Nesting solver for ``VANRYZIN-2``."""

    config: DAVNConfig  # type: ignore[assignment]

    name: str = "DAVN"
    config_class: ClassVar[type[SolverConfig]] = DAVNConfig
    class_name_abbr: ClassVar[str] = "DAVN"
    class_name: ClassVar[str] = "Displacement Adjusted Virtual Nesting"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    supported_problem_type: ClassVar[SolverProblemType] = SolverProblemType.MULTISTAGE
    gradient_needed: ClassVar[bool] = False

    def solve(self, problem: ProblemLike) -> None:  # noqa: D102
        if not isinstance(problem, MultistageProblem):
            raise ValueError("DAVN can only solve multistage problems.")

        # Make standalone runs robust when experiment harness RNG attachment is
        # not used.
        if not self.solution_progenitor_rngs:
            self.solution_progenitor_rngs = [
                MRG32k3a(s_ss_sss_index=[0, i, 0]) for i in range(problem.model.n_rngs)
            ]

        model_factors = problem.model.factors
        required = [
            "ODF_leg_matrix",
            "capacity",
            "fares",
            "gamma_shape",
            "gamma_scale",
            "n_virtual_classes",
        ]
        missing = [name for name in required if name not in model_factors]
        if missing:
            raise ValueError(f"Problem model missing DAVN factors: {missing}")

        odf = np.asarray(model_factors["ODF_leg_matrix"], dtype=float)
        capacity = np.asarray(model_factors["capacity"], dtype=float)
        fares = np.asarray(model_factors["fares"], dtype=float)
        gamma_shape = np.asarray(model_factors["gamma_shape"], dtype=float)
        gamma_scale = np.asarray(model_factors["gamma_scale"], dtype=float)
        expected_demand = gamma_shape * gamma_scale
        demand_variance = gamma_shape * np.square(gamma_scale)

        if odf.shape[0] != fares.size:
            raise ValueError("ODF_leg_matrix row count must equal number of fares.")
        if odf.shape[1] != capacity.size:
            raise ValueError("ODF_leg_matrix column count must equal number of legs.")

        n_classes = fares.size
        n_virtual = int(
            self.config.n_virtual_classes
            if self.config.n_virtual_classes is not None
            else model_factors["n_virtual_classes"]
        )
        n_virtual = max(2, min(n_virtual, n_classes))

        lp_solution = self._solve_deterministic_lp(
            odf_leg_matrix=odf,
            capacity=capacity,
            fares=fares,
            demand_upper_bounds=expected_demand,
        )

        leg_bucket_data = self.compute_displacement_adjusted_leg_buckets_from_lp(
            lp_bid_prices=lp_solution["bid_prices"],
            odf_leg_matrix=odf,
            fares=fares,
            demand_means=expected_demand,
            demand_variances=demand_variance,
            capacity=capacity,
            n_virtual_classes=n_virtual,
        )

        protection = self._booking_limits_to_protection_levels(
            capacity=capacity,
            booking_limits=leg_bucket_data["booking_limits"],
        )

        decision = tuple(
            float(protection[l, k])
            for l in range(protection.shape[0])  # noqa: E741
            for k in range(protection.shape[1])
        )

        if not problem.check_deterministic_constraints(decision):
            raise ValueError("DAVN generated an infeasible protection-level decision.")

        final_solution = self.create_new_solution(decision, problem)
        self.budget.request(self.budget.total)
        self.recommended_solns.append(final_solution)
        self.intermediate_budgets.append(self.budget.used)

        problem.simulate(final_solution, num_macroreps=self.config.evaluation_reps)
        self.iterations.append(0)
        self.budget_history.append(self.budget.used)
        simulated_revenue = float(final_solution.objectives_mean.item())
        self.fn_estimates.append(simulated_revenue)

        # Convert the final protection decision back to booking limits and
        # report the implied booking-limit weighted revenue.
        implied_booking_limits = self._protection_levels_to_booking_limits(
            capacity=capacity,
            protection_levels=protection,
        )
        booking_limit_revenue = float(
            np.sum(implied_booking_limits * leg_bucket_data["leg_bucket_fare"])
        )
        print(
            "DAVN simulated revenue from final protection levels: "
            f"{simulated_revenue:.6f}"
        )
        print(
            "DAVN booking-limit weighted revenue (sum booking_limit * revenue): "
            f"{booking_limit_revenue:.6f}"
        )

        # ── Build open-loop policy (same protection levels at every stage) ─
        # Standard DAVN: compute protection levels once, use them unchanged
        # throughout the booking horizon.  The model's booking control logic
        # handles accept/reject decisions based on remaining capacity.
        n_stages = problem.model.n_stages
        decs = tuple(decision for _ in range(n_stages))
        ps = problem.create_policy_solution(decs)  # no policy callable → open-loop
        ps.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
        self.policy_solutions = [ps]

    def _solve_deterministic_lp(
        self,
        odf_leg_matrix: np.ndarray,
        capacity: np.ndarray,
        fares: np.ndarray,
        demand_upper_bounds: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Solve the DLP and return primal flow plus dual leg bid prices.

        Primal:
            max  r^T x
            s.t. x <= E[D]
                 A^T x <= C
                 x >= 0

        Dual (implemented directly):
            min  E[D]^T alpha + C^T pi
            s.t. alpha + A pi >= r
                 alpha, pi >= 0

        Here ``A[j, l]`` is the incidence of product ``j`` on leg ``l``.
        """
        n_products, n_legs = odf_leg_matrix.shape

        # Primal solve with explicit inequalities, matching the stated DAVN DLP.
        # First n_products rows enforce x <= E[D], next n_legs rows enforce
        # per-leg capacities.
        a_ub_primal = np.vstack([np.eye(n_products), odf_leg_matrix.T])
        b_ub_primal = np.concatenate([demand_upper_bounds, capacity])
        primal = linprog(
            c=-fares,
            A_ub=a_ub_primal,
            b_ub=b_ub_primal,
            bounds=[(0.0, None)] * n_products,
            method="highs",
        )
        if not primal.success:
            raise ValueError(f"DAVN primal LP failed: {primal.message}")

        # Dual solve (variables ordered as alpha then pi) to recover leg bid
        # prices pi associated with the capacity constraints.
        c_dual = np.concatenate([demand_upper_bounds, capacity])
        a_ub_dual = np.zeros((n_products, n_products + n_legs), dtype=float)
        b_ub_dual = -fares
        for j in range(n_products):
            a_ub_dual[j, j] = -1.0
            a_ub_dual[j, n_products:] = -odf_leg_matrix[j, :]

        dual = linprog(
            c=c_dual,
            A_ub=a_ub_dual,
            b_ub=b_ub_dual,
            bounds=[(0.0, None)] * (n_products + n_legs),
            method="highs",
        )
        if not dual.success:
            raise ValueError(f"DAVN dual LP failed: {dual.message}")

        bid_prices = np.asarray(dual.x[n_products:], dtype=float)
        return {
            "flows": np.asarray(primal.x, dtype=float),
            "bid_prices": bid_prices,
        }

    def compute_displacement_adjusted_partition_from_lp(
        self,
        lp_bid_prices: np.ndarray,
        odf_leg_matrix: np.ndarray,
        fares: np.ndarray,
        demand_weights: np.ndarray,
        n_virtual_classes: int,
    ) -> dict[str, np.ndarray | list[list[int]]]:
        """Find the optimal DAVN partition from LP bid prices.

        A product's displacement-adjusted value is:

            adjusted_j = fare_j - sum_l a_{j,l} * bid_price_l

        Products are sorted by ``adjusted_j`` (descending) and partitioned into
        ``n_virtual_classes`` contiguous groups via dynamic programming, using
        weighted within-class squared error as the objective.
        """
        adjusted = fares - (odf_leg_matrix @ lp_bid_prices)
        order = np.argsort(-adjusted)
        sorted_adjusted = adjusted[order]
        sorted_weights = np.maximum(1e-12, demand_weights[order])

        boundaries = self._optimal_weighted_contiguous_partition(
            values=sorted_adjusted,
            weights=sorted_weights,
            n_groups=n_virtual_classes,
        )

        class_assignment = np.zeros_like(order, dtype=int)
        classes: list[list[int]] = []
        for class_idx, (start, end) in enumerate(boundaries, start=1):
            idx = order[start:end]
            class_assignment[idx] = class_idx
            classes.append(idx.tolist())

        class_means = np.zeros(n_virtual_classes, dtype=float)
        for class_idx, (start, end) in enumerate(boundaries, start=1):
            seg_vals = sorted_adjusted[start:end]
            seg_w = sorted_weights[start:end]
            class_means[class_idx - 1] = float(np.sum(seg_w * seg_vals) / np.sum(seg_w))

        return {
            "adjusted_revenues": adjusted,
            "order": order,
            "class_assignment": class_assignment,
            "class_boundaries": np.asarray(boundaries, dtype=int),
            "class_means": class_means,
            "classes": classes,
        }

    def compute_displacement_adjusted_leg_buckets_from_lp(
        self,
        lp_bid_prices: np.ndarray,
        odf_leg_matrix: np.ndarray,
        fares: np.ndarray,
        demand_means: np.ndarray,
        demand_variances: np.ndarray,
        capacity: np.ndarray,
        n_virtual_classes: int,
    ) -> dict[str, np.ndarray]:
        """Compute leg-wise DAVN buckets and booking limits.

        For each leg ``l`` and product ``j`` that uses leg ``l``, define adjusted
        leg revenue

            rbar_{j,l} = fare_j - sum_{m != l} a_{j,m} * pi_m

        Then, per leg, sort classes by ``rbar_{j,l}`` (high to low), partition into
        contiguous revenue buckets, and set booking limits with

            b_{l,1} >= b_{l,2} >= ... >= b_{l,K}

        before converting to theft-nesting protection levels.
        """
        n_products, n_legs = odf_leg_matrix.shape
        K = n_virtual_classes  # noqa: N806

        # 1) Compute adjusted leg revenue matrix (n_products x n_legs).
        full_displacement = odf_leg_matrix @ lp_bid_prices
        adjusted_leg_revenue = np.full((n_products, n_legs), -np.inf, dtype=float)
        for l in range(n_legs):  # noqa: E741
            # Remove only displacement on all other legs, keeping leg l local.
            adjusted_l = fares - (
                full_displacement - odf_leg_matrix[:, l] * lp_bid_prices[l]
            )
            uses_l = odf_leg_matrix[:, l] > 0.5
            adjusted_leg_revenue[uses_l, l] = adjusted_l[uses_l]

        # 2) Per-leg partition into buckets and demand aggregation.
        leg_class_assignment = np.zeros((n_products, n_legs), dtype=int)
        leg_bucket_mean = np.zeros((n_legs, K), dtype=float)
        leg_bucket_var = np.zeros((n_legs, K), dtype=float)
        leg_bucket_fare = np.zeros((n_legs, K), dtype=float)
        booking_limits = np.zeros((n_legs, K), dtype=float)

        for l in range(n_legs):  # noqa: E741
            on_leg = np.where(odf_leg_matrix[:, l] > 0.5)[0]
            if on_leg.size == 0:
                continue

            vals = adjusted_leg_revenue[on_leg, l]
            wts = np.maximum(1e-12, demand_means[on_leg])
            order_local = np.argsort(-vals)
            sorted_vals = vals[order_local]
            sorted_wts = wts[order_local]

            boundaries = self._optimal_weighted_contiguous_partition(
                values=sorted_vals,
                weights=sorted_wts,
                n_groups=K,
            )

            # Assign local classes and aggregate expected demand by bucket.
            for class_idx, (start, end) in enumerate(boundaries, start=1):
                if end <= start:
                    continue
                local_product_idx = on_leg[order_local[start:end]]
                leg_class_assignment[local_product_idx, l] = class_idx
                seg_means = demand_means[local_product_idx]
                seg_vars = demand_variances[local_product_idx]
                seg_adj_fares = adjusted_leg_revenue[local_product_idx, l]

                leg_bucket_mean[l, class_idx - 1] = float(np.sum(seg_means))
                leg_bucket_var[l, class_idx - 1] = float(np.sum(seg_vars))
                if np.sum(seg_means) > 0:
                    leg_bucket_fare[l, class_idx - 1] = float(
                        np.sum(seg_means * seg_adj_fares) / np.sum(seg_means)
                    )
                else:
                    leg_bucket_fare[l, class_idx - 1] = float(np.mean(seg_adj_fares))

            # 3) Compute booking limits with EMSR-BL on the leg buckets.
            booking_limits[l, :] = self._emsr_bl_booking_limits_for_leg(
                capacity=float(capacity[l]),
                bucket_means=leg_bucket_mean[l, :],
                bucket_variances=leg_bucket_var[l, :],
                bucket_fares=leg_bucket_fare[l, :],
            )

        return {
            "adjusted_leg_revenue": adjusted_leg_revenue,
            "leg_class_assignment": leg_class_assignment,
            "leg_bucket_mean": leg_bucket_mean,
            "leg_bucket_var": leg_bucket_var,
            "leg_bucket_fare": leg_bucket_fare,
            "booking_limits": booking_limits,
        }

    def _emsr_bl_booking_limits_for_leg(
        self,
        capacity: float,
        bucket_means: np.ndarray,
        bucket_variances: np.ndarray,
        bucket_fares: np.ndarray,
    ) -> np.ndarray:
        """Compute EMSR-BL booking limits for one leg.

        Buckets are ordered high-to-low by adjusted fare. For class k (k>=2),
        protect aggregate demand of buckets 1..k-1 against bucket k using
        Littlewood/EMSR-B critical fractile.
        """
        K = int(bucket_means.size)  # noqa: N806
        b = np.zeros(K, dtype=float)
        if K == 0:
            return b

        b[0] = float(capacity)
        eps = 1e-12

        for k in range(1, K):
            mu_h = float(np.sum(bucket_means[:k]))
            var_h = float(np.sum(bucket_variances[:k]))
            sigma_h = float(np.sqrt(max(var_h, 0.0)))

            # Demand-weighted average adjusted fare for the aggregated high set.
            mu_weights = np.maximum(bucket_means[:k], 0.0)
            if float(np.sum(mu_weights)) > eps:
                fare_h = float(
                    np.sum(mu_weights * bucket_fares[:k]) / np.sum(mu_weights)
                )
            else:
                fare_h = (
                    float(np.mean(bucket_fares[:k]))
                    if k > 0
                    else float(bucket_fares[0])
                )
            fare_l = float(bucket_fares[k])

            if fare_h <= eps or sigma_h <= eps:
                y = mu_h
            else:
                fractile = 1.0 - (fare_l / max(fare_h, eps))
                fractile = float(np.clip(fractile, 1e-8, 1.0 - 1e-8))
                y = mu_h + sigma_h * float(norm.ppf(fractile))

            y = float(np.clip(y, 0.0, capacity))
            b[k] = float(capacity - y)

        # Enforce monotone limits and theft-nesting terminal condition.
        b = np.clip(np.minimum.accumulate(b), 0.0, capacity)
        b[-1] = 0.0
        return b

    def _optimal_weighted_contiguous_partition(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        n_groups: int,
    ) -> list[tuple[int, int]]:
        """Dynamic programming for optimal weighted contiguous segmentation."""
        n = values.size
        g = max(1, min(n_groups, n))

        w_prefix = np.zeros(n + 1, dtype=float)
        v_prefix = np.zeros(n + 1, dtype=float)
        vv_prefix = np.zeros(n + 1, dtype=float)
        w_prefix[1:] = np.cumsum(weights)
        v_prefix[1:] = np.cumsum(weights * values)
        vv_prefix[1:] = np.cumsum(weights * values * values)

        def seg_cost(i: int, j: int) -> float:
            w_sum = w_prefix[j] - w_prefix[i]
            if w_sum <= 0:
                return 0.0
            v_sum = v_prefix[j] - v_prefix[i]
            vv_sum = vv_prefix[j] - vv_prefix[i]
            return float(vv_sum - (v_sum * v_sum) / w_sum)

        inf = float("inf")
        dp = np.full((g + 1, n + 1), inf, dtype=float)
        back = np.full((g + 1, n + 1), -1, dtype=int)
        dp[0, 0] = 0.0

        for k in range(1, g + 1):
            for j in range(k, n + 1):
                best_val = inf
                best_i = -1
                for i in range(k - 1, j):
                    cand = dp[k - 1, i] + seg_cost(i, j)
                    if cand < best_val:
                        best_val = cand
                        best_i = i
                dp[k, j] = best_val
                back[k, j] = best_i

        boundaries_rev: list[tuple[int, int]] = []
        j = n
        for k in range(g, 0, -1):
            i = int(back[k, j])
            if i < 0:
                i = k - 1
            boundaries_rev.append((i, j))
            j = i
        boundaries = list(reversed(boundaries_rev))

        if g < n_groups:
            boundaries.extend([(n, n)] * (n_groups - g))
        return boundaries

    def _build_protection_levels_from_partition(
        self,
        capacity: np.ndarray,
        odf_leg_matrix: np.ndarray,
        demand_weights: np.ndarray,
        class_assignment: np.ndarray,
        n_virtual_classes: int,
    ) -> np.ndarray:
        """Convert class partition to nested protection levels per leg."""
        n_legs = capacity.size
        aggregated = np.zeros((n_legs, n_virtual_classes), dtype=float)

        for j in range(odf_leg_matrix.shape[0]):
            cls = int(class_assignment[j])
            if cls <= 0:
                continue
            for l in range(n_legs):  # noqa: E741
                if odf_leg_matrix[j, l] > 0.5:
                    aggregated[l, cls - 1] += float(demand_weights[j])

        protection = np.zeros((n_legs, n_virtual_classes), dtype=float)
        for l in range(n_legs):  # noqa: E741
            cap = float(capacity[l])
            for k in range(1, n_virtual_classes + 1):
                lower_class_demand = float(np.sum(aggregated[l, k:]))
                y = cap - lower_class_demand
                protection[l, k - 1] = float(np.clip(y, 0.0, cap))

            # Enforce monotonicity and ensure top level equals capacity.
            protection[l, :] = np.maximum.accumulate(protection[l, :])
            protection[l, -1] = cap

        return protection

    def _booking_limits_to_protection_levels(
        self,
        capacity: np.ndarray,
        booking_limits: np.ndarray,
    ) -> np.ndarray:
        """Convert monotone booking limits to theft-nesting protection levels.

        Booking limits are expected to be high-to-low by bucket index. We map

            y_{l,k} = C_l - b_{l,k}

        then enforce monotonicity and ``y_{l,K}=C_l`` for compatibility with the
        model's protection-level representation.
        """
        n_legs, K = booking_limits.shape  # noqa: N806
        protection = np.zeros((n_legs, K), dtype=float)
        for l in range(n_legs):  # noqa: E741
            cap = float(capacity[l])
            b = np.asarray(booking_limits[l], dtype=float)

            # If limits are normalized, scale to leg capacity.
            if np.max(b) <= 1.0 + 1e-12:
                b = cap * b

            # Ensure required ordering and bounds before conversion.
            b = np.clip(np.minimum.accumulate(b), 0.0, cap)
            if K > 0:
                b[-1] = 0.0

            y = cap - b
            y = np.maximum.accumulate(np.clip(y, 0.0, cap))
            if K > 0:
                y[-1] = cap
            protection[l, :] = y

        return protection

    def _protection_levels_to_booking_limits(
        self,
        capacity: np.ndarray,
        protection_levels: np.ndarray,
    ) -> np.ndarray:
        """Convert theft-nesting protection levels to monotone booking limits."""
        n_legs, K = protection_levels.shape  # noqa: N806
        booking_limits = np.zeros((n_legs, K), dtype=float)
        for l in range(n_legs):  # noqa: E741
            cap = float(capacity[l])
            y = np.asarray(protection_levels[l], dtype=float)

            # Ensure monotone protection levels and terminal y_{l,K}=C_l.
            y = np.maximum.accumulate(np.clip(y, 0.0, cap))
            if K > 0:
                y[-1] = cap

            b = cap - y
            b = np.clip(np.minimum.accumulate(b), 0.0, cap)
            if K > 0:
                b[-1] = 0.0
            booking_limits[l, :] = b

        return booking_limits
