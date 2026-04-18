"""Techniques for subsampling a mesh."""

from copy import deepcopy

import numpy as np
from scipy.linalg import cholesky, lstsq, lu, qr, svd


class Subsampling:
    """Returns subsampling methods for pruning down the number of quadrature points.

    Choices include:

     - `qr`: QR column pivoting.
     - `svd`: QR column pivoting on right singular vectors.
     - `newton`: Determinant maximization.
     - `lu`: LU row pivoting.
     - `random`: Randomly subsample.

    :param string subsampling_algorithm: Choose between `qr`, `svd`, `newton`, `lu`,
    `random`.
    """

    def __init__(self, subsampling_algorithm) -> None:  # noqa: ANN001, D107
        self.subsampling_algorithm = subsampling_algorithm
        if self.subsampling_algorithm is None:
            self.algorithm = _get_all_pivots
        elif self.subsampling_algorithm.lower() == "qr":
            self.algorithm = get_qr_column_pivoting
        elif self.subsampling_algorithm.lower() == "svd":
            self.algorithm = get_svd_subset_selection
        elif self.subsampling_algorithm.lower() == "newton":
            self.algorithm = get_newton_determinant_maximization
        elif self.subsampling_algorithm.lower() == "lu":
            self.algorithm = get_lu_row_pivoting
        elif self.subsampling_algorithm.lower() == "random":
            # Is this a placeholder?
            self.algorithm = 0  # np.random.choice(int(m), m_refined, replace=False)

    def get_subsampling_method(self):  # noqa: ANN201, D102
        return self.algorithm


def _get_all_pivots(Ao, number_of_subsamples):  # noqa: ANN001, ANN202, ARG001, N803
    """A dummy case where we return all the subsamples."""
    return np.arange(1, len(number_of_subsamples))


def get_qr_column_pivoting(Ao, number_of_subsamples):  # noqa: ANN001, ANN201, N803
    """Pivoted QR factorization, where the pivots are used as a heuristic for.

    subsampling.
    """
    A = deepcopy(Ao)  # noqa: N806
    _, _, pvec = qr(A.T, pivoting=True)
    return pvec[0:number_of_subsamples]


def get_svd_subset_selection(Ao, number_of_subsamples):  # noqa: ANN001, ANN201, N803
    """Singular value decomposition and pivoted QR factorization, where the pivots.

    are used as a heuristic for subsampling.
    """
    A = deepcopy(Ao)  # noqa: N806
    _, _, V = svd(A.T)  # noqa: N806
    _, _, pvec = qr(V[:, 0:number_of_subsamples].T, pivoting=True)
    return pvec[0:number_of_subsamples]


def get_lu_row_pivoting(Ao, number_of_subsamples):  # noqa: ANN001, ANN201, N803
    """Retain rows with largest pivots in LU factorisation. AKA Leja sequence."""
    A = Ao.copy()  # noqa: N806
    P = lu(A)[0]  # noqa: N806
    return np.where(P == 1)[1][:number_of_subsamples]


def get_newton_determinant_maximization(Ao, number_of_subsamples):  # noqa: ANN001, ANN201, N803
    """A convex relaxation technique for determinant maximization---akin to optimal.

    experiment of.

    design (D). Based on the work of Joshi and Boyd [1].

    **References**
        1. Joshi, S., Boyd, S., (2009) Sensor Selection via Convex Optimization. IEEE
        Transactions on Signal Processing, 57(2). `Paper
        <https://ieeexplore.ieee.org/document/4663892>`__

    """
    A = deepcopy(Ao)  # noqa: N806
    maxiter = 50
    n_tol = 1e-12
    gap = 1.005

    # For backtracking line search parameters
    alpha = 0.01
    beta = 0.5

    # Assuming the input matrix is an np.matrix()
    m, n = A.shape
    if m < n:
        raise ValueError(
            "maxdet(): requires the number of columns to be greater than the number of rows!"
        )
    z = np.ones((m, 1)) * float(number_of_subsamples) / float(m)
    g = np.zeros((m, 1))
    ones_m = np.ones((m, 1))
    ones_m_transpose = np.ones((1, m))
    kappa = np.log(gap) * n / m

    # Objective function
    Z = _diag(z)  # noqa: N806
    fz = -np.log(np.linalg.det(A.T * Z * A)) - kappa * np.sum(
        np.log(z) + np.log(1.0 - z)
    )

    # Optimization loop!
    for _i in range(0, maxiter):
        Z = _diag(z)  # noqa: N806
        W = np.linalg.inv(A.T * Z * A)  # noqa: N806
        V = A * W * A.T  # noqa: N806
        vo = np.matrix(np.diag(V))
        vo = vo.T

        # define some z operations
        one_by_z = ones_m / z
        one_by_one_minus_z = ones_m / (ones_m - z)
        one_by_z2 = ones_m / z**2
        one_by_one_minus_z2 = ones_m / (ones_m - z) ** 2
        g = -vo - kappa * (one_by_z - one_by_one_minus_z)
        H = np.multiply(V, V) + kappa * _diag(one_by_z2 + one_by_one_minus_z2)  # noqa: N806

        # Textbook Newton's method -- compute inverse of Hessian
        R = np.matrix(cholesky(H))  # noqa: N806
        u = lstsq(R.T, g)
        Hinvg = lstsq(R, u[0])  # noqa: N806
        Hinvg = Hinvg[0]  # noqa: N806
        v = lstsq(R.T, ones_m)
        Hinv1 = lstsq(R, v[0])  # noqa: N806
        Hinv1 = Hinv1[0]  # noqa: N806
        dz = (
            -Hinvg
            + (np.dot(ones_m_transpose, Hinvg) / np.dot(ones_m_transpose, Hinv1))
            * Hinv1
        )

        deczi = _indices(dz, lambda x: x < 0)
        inczi = _indices(dz, lambda x: x > 0)
        a1 = 0.99 * -z[deczi, 0] / dz[deczi, 0]
        a2 = (1 - z[inczi, 0]) / dz[inczi, 0]
        s = np.min(np.vstack([1.0, np.vstack(a1), np.vstack(a2)]))
        flag = 1

        while flag == 1:
            zp = z + s * dz
            Zp = _diag(zp)  # noqa: N806
            fzp = -np.log(np.linalg.det(A.T * Zp * A)) - kappa * np.sum(
                np.log(zp) + np.log(1 - zp)
            )
            const = fz + alpha * s * g.T * dz
            if fzp <= const[0, 0]:
                flag = 2
            if flag != 2:
                s = beta * s
        z = zp
        fz = fzp
        sig = -g.T * dz * 0.5
        if sig[0, 0] <= n_tol:
            break
        zsort = np.sort(z, axis=0)
        thres = zsort[m - number_of_subsamples - 1]
        zhat, not_used = _find(z, thres)  # noqa: RUF059

    zsort = np.sort(z, axis=0)
    thres = zsort[m - number_of_subsamples - 1]
    zhat, _not_used = _find(z, thres)
    _p, _q = zhat.shape
    Zhat = _diag(zhat)  # noqa: N806
    np.log(np.linalg.det(A.T * Zhat * A))
    np.log(np.linalg.det(A.T * _diag(z) * A)) + 2 * m * kappa
    return _binary2indices(zhat)


def _binary2indices(zhat):  # noqa: ANN001, ANN202
    """Simple utility that converts a binary array into one with indices!"""
    pvec = []
    m, _n = zhat.shape
    for i in range(0, m):
        if zhat[i, 0] == 1:
            pvec.append(i)
    return pvec


def _indices(a, func):  # noqa: ANN001, ANN202
    return [i for (i, val) in enumerate(a) if func(val)]


def _diag(vec):  # noqa: ANN001, ANN202
    m = len(vec)
    D = np.zeros((m, m))  # noqa: N806
    for i in range(0, m):
        D[i, i] = vec[i, 0]
    return D


def _find(vec, thres):  # noqa: ANN001, ANN202
    t = []
    vec_new = []
    for i in range(0, len(vec)):
        if vec[i] > thres:
            t.append(i)
            vec_new.append(1.0)
        else:
            vec_new.append(0.0)
    vec_new = np.matrix(vec_new)
    vec_new = vec_new.T
    return vec_new, t
