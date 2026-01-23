from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")


# from simopt.solvers.active_subspaces.basis import *
# from simopt.solvers.active_subspaces.polyridge import *
# from simopt.solvers.active_subspaces.subspace import *
# from simopt.solvers.active_subspaces.index_set import IndexSet
from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import norm, qr
from scipy.optimize import linprog, minimize
from scipy.special import factorial

from simopt.base import Problem

if TYPE_CHECKING:
    from simopt.solvers.TrustRegion.TrustRegion import TrustRegionBase

__all__ = ["AstroDFGeometry", "OMoRFGeometry", "TrustRegionGeometry"]


class TrustRegionGeometry:
    def __init__(self, problem: Problem):
        self.problem = problem

    def standard_basis(self, index: int) -> list[float]:
        """Creates a standard basis vector e_i in the space of dimension equal to the problem dimension. Where i is at the index of the index parameter
        Args:
                index (int): the location of the value 1 in the standard basis vector

        Returns:
                np.array: a standard basis vector of the form (0,0,...,0,1,0,...,0), where the 1 is in the location of index
        """
        arr = np.zeros(self.problem.dim)
        arr[index] = 1.0
        return arr

    def interpolation_points(self, x_k: np.ndarray, delta: float) -> list[np.ndarray]:
        """Constructs an interpolation set of

        Args:
                delta (TYPE): Description

        Returns:
                [np.array]: Description
        """
        d = self.problem.dim

        Y = [x_k]
        epsilon = 0.01
        for i in range(0, d):
            plus = Y[0] + delta * self.standard_basis(i)
            minus = Y[0] - delta * self.standard_basis(i)

            if sum(x_k) != 0:  # check if x_k is not the origin
                # block constraints
                if minus[i] <= self.problem.lower_bounds[i]:
                    minus[i] = self.problem.lower_bounds[i] + epsilon
                if plus[i] >= self.problem.upper_bounds[i]:
                    plus[i] = self.problem.upper_bounds[i] - epsilon

            Y.append(plus)
            Y.append(minus)
        return Y


class AstroDFGeometry(TrustRegionGeometry):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)

    # generate the basis (rotated coordinate) (the first vector comes from the visited design points (origin basis)
    def get_rotated_basis(
        self, first_basis: np.ndarray, rotate_index: np.ndarray
    ) -> np.ndarray:
        rotate_matrix = np.array(first_basis)
        rotation = np.matrix([[0, -1], [1, 0]])

        # rotate the coordinate basis based on the first basis vector (first_basis)
        # choose two dimensions which we use for the rotation (0,i)
        for i in range(1, len(rotate_index)):
            v1 = np.array(
                [[first_basis[rotate_index[0]]], [first_basis[rotate_index[i]]]]
            )
            v2 = np.dot(rotation, v1)
            rotated_basis = np.copy(first_basis)
            rotated_basis[rotate_index[0]] = v2[0][0]
            rotated_basis[rotate_index[i]] = v2[1][0]
            # stack the rotated vector
            rotate_matrix = np.vstack((rotate_matrix, rotated_basis))
        return rotate_matrix

    # compute the interpolation points (2d+1) using the rotated coordinate basis (reuse one design point)
    def get_rotated_basis_interpolation_points(
        self,
        x_k: np.ndarray,
        delta: float,
        rotate_matrix: np.ndarray,
        reused_x: np.ndarray,
    ) -> list[np.ndarray]:
        Y = [x_k]
        epsilon = 0.01
        for i in range(self.problem.dim):
            if i == 0:
                plus = np.array(reused_x)
            else:
                plus = Y[0] + delta * rotate_matrix[i]
            minus = Y[0] - delta * rotate_matrix[i]

            if sum(x_k) != 0:
                # block constraints
                for j in range(self.problem.dim):
                    if minus[j] <= self.problem.lower_bounds[j]:
                        minus[j] = self.problem.lower_bounds[j] + epsilon
                    elif minus[j] >= self.problem.upper_bounds[j]:
                        minus[j] = self.problem.upper_bounds[j] - epsilon
                    if plus[j] <= self.problem.lower_bounds[j]:
                        plus[j] = self.problem.lower_bounds[j] + epsilon
                    elif plus[j] >= self.problem.upper_bounds[j]:
                        plus[j] = self.problem.upper_bounds[j] - epsilon

            Y.append(plus)
            Y.append(minus)
        return Y


"""
	Class that represents the geometry of the solution space. It is able to construct an interpolation set and handle geometric behaviours of the space
"""


class OMoRFGeometry(TrustRegionGeometry):
    def __init__(self, problem: Problem, tr: TrustRegionBase, index_set, **kwargs):
        # print(kwargs)
        self.problem = problem
        self.tr = tr
        self.random_initial = kwargs["random_directions"]
        self.n = kwargs["n"]
        self.epsilon_1 = kwargs["epsilon_1"]
        self.epsilon_2 = kwargs["epsilon_2"]
        self.rho_min = kwargs["rho_min"]
        self.alpha_1 = kwargs["alpha_1"]
        self.alpha_2 = kwargs["alpha_2"]
        self.d = kwargs["d"]
        self.q = kwargs["q"]
        self.p = kwargs["p"]
        self.index_set = index_set

        super().__init__(problem)

    def generate_set(self, num, s_old, delta_k):
        """Generates an initial set of samples using either coordinate directions or orthogonal, random directions
        """
        bounds_l = np.maximum(
            np.array(self.problem.lower_bounds).reshape(s_old.shape), s_old - delta_k
        )
        bounds_u = np.minimum(
            np.array(self.problem.upper_bounds).reshape(s_old.shape), s_old + delta_k
        )

        if self.random_initial:
            direcs = self.random_directions(
                num, bounds_l - s_old, bounds_u - s_old, delta_k
            )
        else:
            direcs = self.coordinate_directions(
                num, bounds_l - s_old, bounds_u - s_old, delta_k
            )
        S = np.zeros((num, self.n))
        S[0, :] = s_old
        for i in range(1, num):
            S[i, :] = s_old + np.minimum(
                np.maximum(bounds_l - s_old, direcs[i, :]), bounds_u - s_old
            )
        return S

    def coordinate_directions(self, num_pnts, lower, upper, delta_k):
        """Generates coordinate directions
        """
        at_lower_boundary = lower > -1.0e-8 * delta_k
        at_upper_boundary = upper < 1.0e-8 * delta_k
        direcs = np.zeros((num_pnts, self.n))
        for i in range(1, num_pnts):
            if 1 <= i < self.n + 1:
                dirn = i - 1
                step = delta_k if not at_upper_boundary[dirn] else -delta_k
                direcs[i, dirn] = step
            elif self.n + 1 <= i < 2 * self.n + 1:
                dirn = i - self.n - 1
                step = -delta_k
                if at_lower_boundary[dirn]:
                    step = min(2.0 * delta_k, upper[dirn])
                if at_upper_boundary[dirn]:
                    step = max(-2.0 * delta_k, lower[dirn])
                direcs[i, dirn] = step
            else:
                itemp = (i - self.n - 1) // self.n
                q = i - itemp * self.n - self.n
                p = q + itemp
                if p > self.n:
                    p, q = q, p - self.n
                direcs[i, p - 1] = direcs[p, p - 1]
                direcs[i, q - 1] = direcs[q, q - 1]
        return direcs

    def get_scale(self, dirn, delta, lower, upper):
        """Sets the max distance for each direction in the current trust region
        """
        scale = delta
        for j in range(len(dirn)):
            if dirn[j] < 0.0:
                scale = min(scale, lower[j] / dirn[j])
            elif dirn[j] > 0.0:
                scale = min(scale, upper[j] / dirn[j])
        return scale

    def random_directions(self, num_pnts, lower, upper, delta_k):
        """Generates orthogonal, random directions
        """
        direcs = np.zeros((self.n, max(2 * self.n + 1, num_pnts)))
        idx_l = lower == 0
        idx_u = upper == 0
        active = np.logical_or(idx_l, idx_u)
        inactive = np.logical_not(active)
        nactive = np.sum(active)
        ninactive = self.n - nactive
        if ninactive > 0:
            A = np.random.normal(size=(ninactive, ninactive))
            Qred = qr(A)[0]
            Q = np.zeros((self.n, ninactive))
            Q[inactive, :] = Qred
            for i in range(ninactive):
                scale = self.get_scale(Q[:, i], delta_k, lower, upper)
                direcs[:, i] = scale * Q[:, i]
                scale = self.get_scale(-Q[:, i], delta_k, lower, upper)
                direcs[:, self.n + i] = -scale * Q[:, i]
        idx_active = np.where(active)[0]
        for i in range(nactive):
            idx = idx_active[i]
            direcs[idx, ninactive + i] = 1.0 if idx_l[idx] else -1.0
            direcs[:, ninactive + i] = (
                self.get_scale(direcs[:, ninactive + i], delta_k, lower, upper)
                * direcs[:, ninactive + i]
            )
            sign = 1.0 if idx_l[idx] else -1.0
            if upper[idx] - lower[idx] > delta_k:
                direcs[idx, self.n + ninactive + i] = 2.0 * sign * delta_k
            else:
                direcs[idx, self.n + ninactive + i] = (
                    0.5 * sign * (upper[idx] - lower[idx])
                )
            direcs[:, self.n + ninactive + i] = (
                self.get_scale(direcs[:, self.n + ninactive + i], 1.0, lower, upper)
                * direcs[:, self.n + ninactive + i]
            )
        for i in range(num_pnts - 2 * self.n):
            dirn = np.random.normal(size=(self.n,))
            for j in range(nactive):
                idx = idx_active[j]
                sign = 1.0 if idx_l[idx] else -1.0
                if dirn[idx] * sign < 0.0:
                    dirn[idx] *= -1.0
            dirn = dirn / norm(dirn)
            scale = self.get_scale(dirn, delta_k, lower, upper)
            direcs[:, 2 * self.n + i] = dirn * scale
        # Ensure first row is zero direction to match generate_set expectations
        R = np.zeros((num_pnts, self.n))
        cols = min(max(num_pnts - 1, 0), direcs.shape[1])
        if cols > 0:
            R[1 : 1 + cols, :] = direcs[:, :cols].T
        return R

    def update_geometry_omorf(
        self,
        s_old,
        f_old,
        delta_k,
        rho_k,
        U,
        S_full,
        f_full,
        S_red,
        f_red,
        unsuccessful_iteration_counter,
        ratio,
    ):
        dist = max(self.epsilon_1 * delta_k, self.epsilon_2 * rho_k)

        as_matrix = U

        if max(norm(S_full - s_old, axis=1, ord=np.inf)) > dist:
            S_full, f_full = self.sample_set(
                "improve", s_old, delta_k, rho_k, f_old, U, S=S_full, f=f_full
            )  # f_full is not needed to be evaluated
            try:
                self.tr.calculate_subspace(S_full, f_full, delta_k)
                as_matrix = self.tr.U
            except:
                pass

        elif max(norm(S_red - s_old, axis=1, ord=np.inf)) > dist:
            S_red, f_red = self.sample_set(
                "improve",
                s_old,
                delta_k,
                rho_k,
                f_old,
                U,
                S=S_red,
                f=f_red,
                full_space=False,
            )

        elif delta_k == rho_k:
            delta_k = self.alpha_2 * rho_k

            if unsuccessful_iteration_counter >= 3 and ratio < 0:
                # print('UNSUCCESSFUL 3 OR MORE TIMES')
                if rho_k >= 250 * self.rho_min:
                    rho_k = self.alpha_1 * rho_k

                elif 16 * self.rho_min < rho_k < 250 * self.rho_min:
                    rho_k = np.sqrt(rho_k * self.rho_min)

                else:
                    rho_k = self.rho_min

        return S_full, f_full, S_red, f_red, delta_k, rho_k, as_matrix

    def sample_set(
        self,
        method,
        s_old,
        delta_k,
        rho_k,
        f_old,
        U,
        S=None,
        f=None,
        s_new=None,
        f_new=None,
        full_space=True,
    ):
        q = self.p if full_space else self.q

        dist = max(self.epsilon_1 * delta_k, self.epsilon_2 * rho_k)

        if method == "replace":
            S_hat = np.vstack((S, s_new))
            f_hat = np.vstack((f, f_new))
            if S_hat.shape != np.unique(S_hat, axis=0).shape:
                S_hat, indices = np.unique(S_hat, axis=0, return_index=True)
                f_hat = f_hat[indices]
            elif f_hat.size > q and max(norm(S_hat - s_old, axis=1, ord=np.inf)) > dist:
                S_hat, f_hat = self.remove_furthest_point(S_hat, f_hat, s_old)
            S_hat, f_hat = self.remove_point_from_set(S_hat, f_hat, s_old)
            S = np.zeros((q, self.n))
            f = np.zeros((q, 1))
            S[0, :] = s_old
            f[0, :] = f_old
            S, f = self.LU_pivoting(
                S,
                f,
                s_old,
                delta_k,
                S_hat,
                f_hat,
                full_space,
                U,
                evaluate_f_flag=full_space,
            )

        elif method == "improve":
            S_hat = np.copy(S)
            f_hat = np.copy(f)
            if max(norm(S_hat - s_old, axis=1, ord=np.inf)) > dist:
                S_hat, f_hat = self.remove_furthest_point(S_hat, f_hat, s_old)
            S_hat, f_hat = self.remove_point_from_set(S_hat, f_hat, s_old)
            S = np.zeros((q, self.n))
            f = np.zeros((q, 1))
            S[0, :] = s_old
            f[0, :] = f_old
            S, f = self.LU_pivoting(
                S,
                f,
                s_old,
                delta_k,
                S_hat,
                f_hat,
                full_space,
                U,
                evaluate_f_flag=full_space,
                method="improve",
            )

        elif method == "new":
            S_hat = f_hat = np.array([])
            S = np.zeros((q, self.n))
            f = np.zeros((q, 1))
            S[0, :] = s_old
            f[0, :] = f_old
            S, f = self.LU_pivoting(
                S,
                f,
                s_old,
                delta_k,
                S_hat,
                f_hat,
                full_space,
                U,
                evaluate_f_flag=full_space,
                method="new",
            )

        return S, f

    def LU_pivoting(
        self,
        S,
        f,
        s_old,
        delta_k,
        S_hat,
        f_hat,
        full_space,
        active_subspace,
        evaluate_f_flag=True,
        method=None,
    ):
        psi_1 = 1.0e-4
        psi_2 = 1.0 if full_space else 0.25

        phi_function, phi_function_deriv = self.get_phi_function_and_derivative(
            S_hat, s_old, delta_k, full_space, active_subspace
        )
        q = self.p if full_space else self.q

        # use class helper to compute |phi(X) @ v| in chunks

        # Initialise U matrix of LU factorisation of M matrix (see Conn et al.)
        U = np.zeros((q, q))
        U[0, :] = phi_function(s_old)

        # Perform the LU factorisation algorithm for the rest of the points
        for k in range(1, q):
            flag = True
            v = np.zeros(q)
            for j in range(k):
                den = U[j, j]
                if abs(den) < 1e-12:
                    den = 1e-12
                v[j] = -U[j, k] / den
            v[k] = 1.0

            # If there are still points to choose from, find if points meet criterion. If so, use the index to choose
            # point with given index to be next point in regression/interpolation set
            if f_hat.size > 0:
                M = self._abs_phi_dot_v(phi_function, S_hat, v)
                index = np.argmax(M)
                if (
                    M[index] < psi_1
                    or (method == "improve" and (k == q - 1 and M[index] < psi_2))
                    or (method == "new" and M[index] < psi_2)
                ):
                    flag = False
            else:
                flag = False

            # If index exists, choose the point with that index and delete it from possible choices
            if flag:
                s = S_hat[index, :]
                S[k, :] = s
                f[k, :] = f_hat[index]
                S_hat = np.delete(S_hat, index, 0)
                f_hat = np.delete(f_hat, index, 0)

            # If index doesn't exist, solve an optimisation problem to find the point in the range which best satisfies criterion
            else:
                try:
                    s = self.find_new_point(
                        v, phi_function, phi_function_deriv, s_old, delta_k, full_space
                    )
                    if np.unique(np.vstack((S[:k, :], s)), axis=0).shape[0] != k + 1:
                        s = self.find_new_point_alternative(
                            v, phi_function, S[:k, :], s_old, delta_k
                        )
                except:
                    s = self.find_new_point_alternative(
                        v, phi_function, S[:k, :], s_old, delta_k
                    )
                if f_hat.size > 0 and M[index] >= abs(np.dot(v, phi_function(s))):
                    s = S_hat[index, :]
                    S[k, :] = s
                    f[k, :] = f_hat[index]
                    S_hat = np.delete(S_hat, index, 0)
                    f_hat = np.delete(f_hat, index, 0)
                else:
                    S[k, :] = s
                    f[k, :] = self.tr.blackbox_evaluation(s, self.problem)

            # Update U factorisation in LU algorithm
            phi = phi_function(s)
            U[k, k] = np.dot(v, phi)
            for i in range(k + 1, q):
                U[k, i] += phi[i]
                for j in range(k):
                    den2 = U[j, j]
                    if abs(den2) < 1e-12:
                        den2 = 1e-12
                    U[k, i] -= (phi[j] * U[j, i]) / den2
        return S, f

    def get_phi_function_and_derivative(
        self, S_hat, s_old, delta_k, full_space, active_subspace
    ):
        Del_S = delta_k

        if full_space:
            if S_hat.size > 0:
                norms = norm(S_hat - s_old, axis=1, ord=np.inf)
                Del_S = max(np.max(norms), 1e-12)

            def phi_function(s):
                s_tilde = np.divide((s - s_old), Del_S)
                try:
                    m, n = s_tilde.shape
                except:
                    m = 1
                    s_tilde = s_tilde.reshape(1, -1)
                phi = np.zeros((m, self.p))
                phi[:, 0] = 1.0
                phi[:, 1:] = s_tilde
                if m == 1:
                    return phi.flatten()
                return phi

            phi_function_deriv = None

        else:
            if S_hat.size > 0:
                # active_subspace expected as shape (n, d): map full-space differences to reduced coords
                norms = norm(np.dot(S_hat - s_old, active_subspace), axis=1)
                Del_S = max(np.max(norms), 1e-12)

            def phi_function(s):
                # compute reduced coordinates u = (s - s_old) @ active_subspace
                u = np.divide(np.dot((s - s_old), active_subspace), Del_S)
                # print(f'shape of u: {u.shape}')
                try:
                    m, n = u.shape
                except:
                    m = 1
                    u = u.reshape(1, -1)
                phi = np.zeros((m, self.q))
                for k in range(self.q):
                    phi[:, k] = np.prod(
                        np.divide(
                            np.power(u, self.index_set[k, :]),
                            factorial(self.index_set[k, :]),
                        ),
                        axis=1,
                    )
                if m == 1:
                    return phi.flatten()
                return phi

            def phi_function_deriv(s):
                u = np.divide(np.dot((s - s_old), active_subspace), Del_S)
                phi_deriv = np.zeros((self.d, self.q))
                for i in range(self.d):
                    for k in range(1, self.q):
                        if self.index_set[k, i] != 0.0:
                            tmp = np.zeros(self.d)
                            tmp[i] = 1
                            phi_deriv[i, k] = self.index_set[k, i] * np.prod(
                                np.divide(
                                    np.power(u, self.index_set[k, :] - tmp),
                                    factorial(self.index_set[k, :]),
                                )
                            )
                phi_deriv = np.divide(phi_deriv.T, Del_S).T
                # active_subspace is (n, d); phi_deriv is (d, q) -> result is (n, q)
                return np.dot(active_subspace, phi_deriv)

        return phi_function, phi_function_deriv

    def _abs_phi_dot_v(self, phi_function, X, v, chunk_size=64):
        """Compute |phi(X) @ v| in chunks to limit peak memory usage.

        - `phi_function` should accept an (m,n) array and return (m,q) phi values.
        - `X` may be 1D (single point) or 2D (many points).
        - `chunk_size` controls memory vs. speed trade-off.
        """
        if X is None or X.size == 0:
            return np.array([])
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        m = X.shape[0]
        res = np.empty(m)
        for start in range(0, m, chunk_size):
            end = min(m, start + chunk_size)
            X_chunk = X[start:end, :]
            try:
                phi_chunk = phi_function(X_chunk)
                # phi_chunk may be flattened for single row; ensure 2D
                phi_chunk = np.asarray(phi_chunk)
                if phi_chunk.ndim == 1:
                    phi_chunk = phi_chunk.reshape(1, -1)
                vals = np.abs(np.dot(phi_chunk, v).flatten())
            except Exception:
                vals = np.zeros(end - start)
            res[start:end] = vals
        return res

    def find_new_point(
        self, v, phi_function, phi_function_deriv, s_old, delta_k, full_space=False
    ):
        # change bounds to be defined using the problem and delta_k
        bounds_l = np.maximum(
            np.array(self.problem.lower_bounds).reshape(s_old.shape), s_old - delta_k
        )
        bounds_u = np.minimum(
            np.array(self.problem.upper_bounds).reshape(s_old.shape), s_old + delta_k
        )

        bounds = []
        for i in range(self.n):
            bounds.append((bounds_l[i], bounds_u[i]))

        if full_space:
            c = v[1:]
            res1 = linprog(c, bounds=bounds)
            res2 = linprog(-c, bounds=bounds)
            val1 = -np.inf
            val2 = -np.inf
            if getattr(res1, "success", False):
                try:
                    val1 = abs(np.dot(v, phi_function(res1.x)))
                except Exception:
                    val1 = -np.inf
            if getattr(res2, "success", False):
                try:
                    val2 = abs(np.dot(v, phi_function(res2.x)))
                except Exception:
                    val2 = -np.inf
            if val1 >= val2 and val1 > -np.inf:
                s = res1.x
            elif val2 > -np.inf:
                s = res2.x
            else:
                s = s_old
        else:
            obj1 = lambda s: np.dot(v, phi_function(s))
            jac1 = lambda s: np.dot(phi_function_deriv(s), v)
            obj2 = lambda s: -np.dot(v, phi_function(s))
            jac2 = lambda s: -np.dot(phi_function_deriv(s), v)
            res1 = minimize(
                obj1,
                s_old,
                method="TNC",
                jac=jac1,
                bounds=bounds,
                options={"disp": False},
            )
            res2 = minimize(
                obj2,
                s_old,
                method="TNC",
                jac=jac2,
                bounds=bounds,
                options={"disp": False},
            )
            # FIX: Don't want to rely on this (possibly)
            if abs(res1["fun"]) > abs(res2["fun"]):
                s = res1["x"]
            else:
                s = res2["x"]
        return s

    def find_new_point_alternative(self, v, phi_function, S, s_old, delta_k):
        S_tmp = self.generate_set(
            int(0.5 * (self.n + 1) * (self.n + 2)), s_old, delta_k
        )
        # Compute |phi(S_tmp) @ v| in chunks to limit peak memory
        if S_tmp.size == 0:
            return s_old
        M = self._abs_phi_dot_v(phi_function, S_tmp, v)
        indices = np.argsort(M)[::-1][: M.size]
        for index in indices:
            s = S_tmp[index, :]
            if np.unique(np.vstack((S, s)), axis=0).shape[0] == S.shape[0] + 1:
                return s
        return S_tmp[indices[0], :]

    @staticmethod
    def remove_point_from_set(S, f, s):
        ind_current = np.where(norm(S - s, axis=1, ord=np.inf) == 0.0)[0]
        S = np.delete(S, ind_current, 0)
        f = np.delete(f, ind_current, 0)
        return S, f

    @staticmethod
    def remove_furthest_point(S, f, s):
        ind_distant = np.argmax(norm(S - s, axis=1, ord=np.inf))
        S = np.delete(S, ind_distant, 0)
        f = np.delete(f, ind_distant, 0)
        return S, f

    def remove_points_outside_limits(self, S, f, s_old, delta_k, rho_k):
        ind_inside = np.where(
            norm(S - s_old, axis=1, ord=np.inf)
            <= max(self.epsilon_1 * delta_k, self.epsilon_2 * rho_k)
        )[0]
        S = S[ind_inside, :]
        f = f[ind_inside]
        return S, f
