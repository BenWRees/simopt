# type: ignore
"""Descriptions of various bases."""

from abc import abstractmethod

__all__ = [
    "ArnoldiPolynomialBasis",  # FIX: THIS ONE
    "AstroDFBasis",
    "Basis",
    "ChebyshevTensorBasis",
    "HermiteTensorBasis",
    "LagrangePolynomialBasis",
    "LaguerreTensorBasis",
    "LegendreTensorBasis",
    "MonomialPolynomialBasis",
    "MonomialTensorBasis",
    "NFPPolynomialBasis",
    "NaturalPolynomialBasis",
    "PolynomialTensorBasis",  # THIS IS A BASE CLASS TO BE INHERITED BY SPECIFIC TENSOR BASIS CLASSES
]


from math import comb, factorial

import numpy as np
from numpy.polynomial.chebyshev import chebder, chebroots, chebvander
from numpy.polynomial.hermite_e import hermeder as hermder
from numpy.polynomial.hermite_e import hermeroots as hermroots
from numpy.polynomial.hermite_e import hermevander as hermvander
from numpy.polynomial.laguerre import lagder, lagroots, lagvander
from numpy.polynomial.legendre import legder, legroots, legvander
from numpy.polynomial.polynomial import polyder, polyroots, polyvander


class Basis:  # noqa: D101
    pass


################################################################################
# Indexing utility functions for total degree
################################################################################

# Module-level cache for index sets to avoid repeated computation
_INDEX_SET_CACHE: dict[tuple[int, int], np.ndarray] = {}
_FULL_INDEX_SET_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _full_index_set(n: int, d: int) -> np.ndarray:
    """A helper function for index_set with caching.

    Parameters
    ----------
    n : int
            degree of polynomial
    d : int
            number of variables, dimension
    """
    cache_key = (n, d)
    if cache_key in _FULL_INDEX_SET_CACHE:
        return _FULL_INDEX_SET_CACHE[cache_key]

    if d == 1:
        result = np.array([[n]])
    else:
        II = _full_index_set(n, d - 1)  # noqa: N806
        m = II.shape[0]
        result = np.hstack((np.zeros((m, 1)), II))
        for i in range(1, n + 1):
            II = _full_index_set(n - i, d - 1)  # noqa: N806
            m = II.shape[0]
            T = np.hstack((i * np.ones((m, 1)), II))  # noqa: N806
            result = np.vstack((result, T))

    _FULL_INDEX_SET_CACHE[cache_key] = result
    return result


def index_set(n: int, d: int) -> np.ndarray:
    """Enumerate multi-indices for a total degree of order `n` in `d` variables.

    Uses caching to avoid repeated computation for the same (n, d) pairs.

    Parameters
    ----------
    n : int
            degree of polynomial
    d : int
            number of variables, dimension

    Returns:
    -------
    I : ndarray
            multi-indices ordered as columns
    """
    # If the full polynomial basis would be extremely large relative to
    # the problem size, fall back to a diagonal-style model to avoid
    # explosive memory usage. This mirrors the safeguard used elsewhere
    # in the code base (e.g., ASTROMoRF) and keeps this module safe when
    # called with large `d` and `n`.
    from math import comb

    # number of terms for full total-degree polynomial
    full_poly_terms = comb(d + n, n)
    # a minimal reasonable number of design points for a quadratic model
    design_points = 2 * d + 1

    # Threshold: if full polynomial terms exceed 10x a small design-point
    # proxy, switch to diagonal model (constant + linear + per-variable
    # quadratic terms). This avoids creating enormous index sets.
    if n >= 2 and full_poly_terms > 10 * design_points:
        cache_key = ("diagonal", n, d)
        if cache_key in _INDEX_SET_CACHE:
            return _INDEX_SET_CACHE[cache_key].copy()

        indices = [np.zeros(d, dtype=np.int64)]
        # linear terms
        for i in range(d):
            idx = np.zeros(d, dtype=np.int64)
            idx[i] = 1
            indices.append(idx)
        # diagonal quadratic terms
        for i in range(d):
            idx = np.zeros(d, dtype=np.int64)
            idx[i] = 2
            indices.append(idx)

        result = np.array(indices, dtype=np.int64)
        _INDEX_SET_CACHE[cache_key] = result
        return result.copy()

    cache_key = (n, d)
    if cache_key in _INDEX_SET_CACHE:
        return _INDEX_SET_CACHE[cache_key].copy()  # Return copy to prevent mutation

    I = np.zeros((1, d), dtype=np.int64)  # noqa: E741, N806
    for i in range(1, n + 1):
        II = _full_index_set(i, d)  # noqa: N806
        I = np.vstack((I, II))  # noqa: E741, N806
    result = I[:, ::-1].astype(int)

    _INDEX_SET_CACHE[cache_key] = result
    return result.copy()


class PolynomialTensorBasis(Basis):
    r"""Generic tensor product basis of fixed total degree.

    This class constructs a tensor product basis of dimension :math:`n`
    of fixed given degree :math:`p` given a basis for polynomials
    in one variable. Namely, this basis is composed of elements:

    This is a base class

    .. math::

            \psi_j(\mathbf x) := \prod_{i=1}^n \phi_{[\boldsymbol \alpha_j]_i}(x_i)
                    \quad \sum_{i=1}^n [\boldsymbol \alpha_j]_i \le p;
                    \quad \phi_i \in \mathcal{P}_{i}(\mathbb{R})


    Parameters
    ----------
    dim: int
            The input dimension of the space
    degree: int
            The total degree of polynomials
    polyvander: function
            Function providing the scalar Vandermonde matrix (i.e.,
            numpy.polynomial.polynomial.polyvander)
    polyder: function
            Function providing the derivatives of scalar polynomials (i.e.,
            numpy.polynomial.polynomial.polyder)

    """

    def __init__(self, degree, dim) -> None:  # noqa: ANN001, D107
        self.degree = int(degree)
        self.dim = int(dim)

        self.indices = index_set(self.degree, self.dim).astype(int)
        self._build_Dmat()

        # Initialize scaling parameters to None - will be set on first scale() call
        self._lb = None
        self._ub = None

    def __len__(self) -> int:  # noqa: D105
        return len(self.indices)

    def __name__(self):  # noqa: ANN204, D105
        raise NotImplementedError

    def assign_interpolation_set(self, X) -> None:  # noqa: ANN001, D102, N803
        self.X = X

    def _build_Dmat(self) -> None:  # noqa: N802
        """Constructs the (scalar) derivative matrix."""
        self.Dmat = np.zeros((self.degree + 1, self.degree))
        I = np.eye(self.degree + 1)  # noqa: E741, N806
        for j in range(self.degree + 1):
            self.Dmat[j, :] = self.polyder(I[:, j])

    def set_scale(self, X) -> None:  # noqa: ANN001, N803
        r"""Construct an affine transformation of the domain to improve the conditioning."""
        self._set_scale(np.array(X))

    def _set_scale(self, X) -> None:  # noqa: ANN001, N803
        r"""Default scaling to [-1,1]."""
        self._lb = np.min(X, axis=0)
        self._ub = np.max(X, axis=0)

    def scale(self, X):  # noqa: ANN001, ANN201, N803
        r"""Apply the scaling to the input coordinates."""
        # Auto-initialize scaling on first call if not already set
        if self._lb is None or self._ub is None:
            self._set_scale(X)
        return 2 * (X - self._lb[None, :]) / (self._ub[None, :] - self._lb[None, :]) - 1

    def dscale(self):  # noqa: ANN201
        r"""Returns the scaling associated with the scaling transform."""
        try:
            return 2.0 / (self._ub - self._lb)
        except AttributeError:
            raise NotImplementedError  # noqa: B904

    def V(self, X=None):  # noqa: ANN001, ANN201, N802, N803
        r"""Builds the Vandermonde matrix associated with this basis.

        Given points :math:`\mathbf x_i \in \mathbb{R}^n`,
        this creates the Vandermonde matrix

        .. math::

                [\mathbf{V}]_{i,j} = \phi_j(\mathbf x_i)

        where :math:`\phi_j` is a multivariate polynomial as defined in the class
        definition.

        Parameters
        ----------
        X: array-like (M, n)
                Points at which to evaluate the basis at where :code:`X[i]` is one such
                point in
                :math:`\mathbf{R}^n`.

        Returns:
        -------
        V: np.array
                Vandermonde matrix of shape (M, N) where M is the number of desired
                points and N is the number of Basis Elements
        """
        if X is None:
            raise NotImplementedError

        dim = X.shape[1]

        self.indices = index_set(self.degree, dim).astype(int)
        X = X.reshape(-1, dim)  # noqa: N806
        X = self.scale(np.array(X))  # noqa: N806
        M = X.shape[0]  # noqa: N806
        # print(f'the number of rows in V is {M} and the number of columns is {len(self.indices)}')
        assert X.shape[1] == dim, "Expected %d dimensions, got %d" % (dim, X.shape[1])  # noqa: UP031
        V_coordinate = [self.vander(X[:, k], self.degree) for k in range(dim)]  # noqa: N806

        V = np.ones((M, len(self.indices)), dtype=X.dtype)  # noqa: N806

        for j, alpha in enumerate(self.indices):
            for k in range(dim):
                V[:, j] *= V_coordinate[k][:, alpha[k]]
        return V

    def VC(self, X, c, dim):  # noqa: ANN001, ANN201, N802, N803
        r"""Evaluate the product of the Vandermonde matrix and a vector.

        This evaluates the product :math:`\mathbf{V}\mathbf{c}`
        where :math:`\mathbf{V}` is the Vandermonde matrix defined in :code:`V`.
        This is done without explicitly constructing the Vandermonde matrix to save
        memory.

        Parameters
        ----------
        X: array-like (M,n)
                Points at which to evaluate the basis at where :code:`X[i]` is one such
                point in
                :math:`\mathbf{R}^n`.
        c: array-like
                The vector to take the inner product with.

        Returns:
        -------
        Vc: np.array (M,)
                Product of Vandermonde matrix and :math:`\mathbf c`

        Note:
        ----
        This is an optimisation technique not currently implemented in the simopt
        library
        """
        if dim is None:
            raise NotImplementedError
        X = X.reshape(-1, dim)  # noqa: N806
        X = self.scale(np.array(X))  # noqa: N806
        M = X.shape[0]  # noqa: N806
        c = np.array(c)
        self.indices = index_set(self.degree, dim).astype(int)
        assert len(self.indices) == c.shape[0]

        if len(c.shape) == 2:
            oneD = False  # noqa: N806
        else:
            c = c.reshape(-1, 1)
            oneD = True  # noqa: N806

        V_coordinate = [self.vander(X[:, k], self.degree) for k in range(dim)]  # noqa: N806
        out = np.zeros((M, c.shape[1]))
        for j, alpha in enumerate(self.indices):
            # If we have a non-zero coefficient
            if np.max(np.abs(c[j, :])) > 0.0:
                col = np.ones(M)
                for ell in range(dim):
                    col *= V_coordinate[ell][:, alpha[ell]]

                for k in range(c.shape[1]):
                    out[:, k] += c[j, k] * col
        if oneD:
            out = out.flatten()
        return out

    def DV(self, X):  # noqa: ANN001, ANN201, N802, N803
        r"""Column-wise derivative of the Vandermonde matrix.

        Given points :math:`\mathbf x_i \in \mathbb{R}^n`,
        this creates the Vandermonde-like matrix whose entries
        correspond to the derivatives of each of basis elements;
        i.e.,

        .. math::

                [\mathbf{V}]_{i,j} = \left. \frac{\partial}{\partial x_k}
                \psi_j(\mathbf{x})
                        \right|_{\mathbf{x} = \mathbf{x}_i}.

        Parameters
        ----------
        X: array-like (M, n)
                Points at which to evaluate the basis at where :code:`X[i]` is one such
                point in
                :math:`\mathbf{R}^n`.

        Returns:
        -------
        Vp: np.array (M, N, n)
                Derivative of Vandermonde matrix where :code:`Vp[i,j,:]`
                is the gradient of :code:`V[i,j]`.
        """
        dim = X.shape[1]
        self.indices = index_set(self.degree, dim).astype(int)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)  # noqa: N806
        # X = X.reshape(-1, self.dim)
        X = self.scale(np.array(X))  # noqa: N806
        M = X.shape[0]  # noqa: N806
        V_coordinate = [self.vander(X[:, k], self.degree) for k in range(dim)]  # noqa: N806

        N = len(self.indices)  # noqa: N806
        DV = np.ones((M, N, dim), dtype=X.dtype)  # noqa: N806

        try:
            dscale = self.dscale()
        except NotImplementedError:
            dscale = np.ones(X.shape[1])

        for k in range(dim):
            for j, alpha in enumerate(self.indices):
                for q in range(dim):
                    if q == k:
                        DV[:, j, k] *= np.dot(
                            V_coordinate[q][:, 0:-1], self.Dmat[alpha[q], :]
                        )
                    else:
                        DV[:, j, k] *= V_coordinate[q][:, alpha[q]]
            # Correct for transform
            DV[:, :, k] *= dscale[k]

        return DV

    def DDV(self, X, dim):  # noqa: ANN001, ANN201, N802, N803
        r"""Column-wise second derivative of the Vandermonde matrix.

        Given points :math:`\mathbf x_i \in \mathbb{R}^n`,
        this creates the Vandermonde-like matrix whose entries
        correspond to the derivatives of each of basis elements;
        i.e.,

        .. math::

                [\mathbf{V}]_{i,j} = \left. \frac{\partial^2}{\partial x_k\partial
                x_\ell} \psi_j(\mathbf{x})
                        \right|_{\mathbf{x} = \mathbf{x}_i}.

        Parameters
        ----------
        X: array-like (M, n)
                Points at which to evaluate the basis at where :code:`X[i]` is one such
                point in
                :math:`\mathbf{R}^m`.

        Returns:
        -------
        Vpp: np.array (M, N, n, n)
                Second derivative of Vandermonde matrix where :code:`Vpp[i,j,:,:]`
                is the Hessian of :code:`V[i,j]`.
        """
        self.indices = index_set(self.degree, dim).astype(int)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)  # noqa: N806
        # X = X.reshape(-1, self.dim)
        X = self.scale(np.array(X))  # noqa: N806
        M = X.shape[0]  # noqa: N806
        V_coordinate = [self.vander(X[:, k], self.degree) for k in range(dim)]  # noqa: N806

        N = len(self.indices)  # noqa: N806
        DDV = np.ones((M, N, dim, dim), dtype=X.dtype)  # noqa: N806

        try:
            dscale = self.dscale()
        except NotImplementedError:
            dscale = np.ones(X.shape[1])

        for k in range(dim):
            for ell in range(k, dim):
                for j, alpha in enumerate(self.indices):
                    for q in range(dim):
                        if q == k == ell:
                            # We need the second derivative
                            eq = np.zeros(self.degree + 1)
                            eq[alpha[q]] = 1.0
                            der2 = self.polyder(eq, 2)
                            DDV[:, j, k, ell] *= V_coordinate[q][:, 0 : len(der2)].dot(
                                der2
                            )
                        elif q in (k, ell):
                            DDV[:, j, k, ell] *= np.dot(
                                V_coordinate[q][:, 0:-1], self.Dmat[alpha[q], :]
                            )
                        else:
                            DDV[:, j, k, ell] *= V_coordinate[q][:, alpha[q]]

                # Correct for transform
                DDV[:, :, k, ell] *= dscale[k] * dscale[ell]
                DDV[:, :, ell, k] = DDV[:, :, k, ell]
        return DDV

    def roots(self, coef, dim):  # noqa: ANN001, ANN201, D102
        if dim > 1:
            raise NotImplementedError
        r = self.polyroots(coef)
        return r * (self._ub[0] - self._lb[0]) / 2.0 + (self._ub[0] + self._lb[0]) / 2.0


class MonomialTensorBasis(PolynomialTensorBasis):
    """A tensor product basis of bounded total degree built from the monomials."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D107
        self.vander = polyvander
        self.polyder = polyder
        self.polyroots = polyroots
        PolynomialTensorBasis.__init__(self, *args, **kwargs)

    def __name__(self):  # noqa: ANN204, D105
        return "MonomialTensorBasis"


class LegendreTensorBasis(PolynomialTensorBasis):
    """A tensor product basis of bounded total degree built from the Legendre.

    polynomials.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D107
        self.vander = legvander
        self.polyder = legder
        self.polyroots = legroots
        PolynomialTensorBasis.__init__(self, *args, **kwargs)

    def __name__(self):  # noqa: ANN204, D105
        return "LegendreTensorBasis"


class ChebyshevTensorBasis(PolynomialTensorBasis):
    """A tensor product basis of bounded total degree built from the Chebyshev.

    polynomials.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D107
        self.vander = chebvander
        self.polyder = chebder
        self.polyroots = chebroots
        PolynomialTensorBasis.__init__(self, *args, **kwargs)

    def __name__(self):  # noqa: ANN204, D105
        return "LegendreTensorBasis"


class LaguerreTensorBasis(PolynomialTensorBasis):
    """A tensor product basis of bounded total degree built from the Laguerre.

    polynomials.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D107
        self.vander = lagvander
        self.polyder = lagder
        self.polyroots = lagroots
        PolynomialTensorBasis.__init__(self, *args, **kwargs)

    def __name__(self):  # noqa: ANN204, D105
        return "LaguerreTensorBasis"


class NFPTensorBasis(PolynomialTensorBasis):
    """A tensor product basis of bounded total degree built from the Newton Fundamental.

    Polynomials.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.vander = self.vander_fn
        self.polyder = self.vander_der_fn
        self.polyroots = self.vander_roots_fn
        PolynomialTensorBasis.__init__(self, *args, **kwargs)

    def __name__(self):  # noqa: ANN204
        return "NFPTensorBasis"

    def vander_fn(self, x, deg):  # noqa: ANN001, ANN202
        """Generate a Vandermonde matrix for Newton Fundamental Polynomials.

        The Newton basis polynomials are:
        - N_0(x) = 1
        - N_k(x) = ∏_{i=0}^{k-1} (x - x_i) for k > 0

        where x_i are the interpolation nodes stored in self.X.

        Parameters
        ----------
        x : array_like
                Array of points at which to evaluate the basis.
        deg : int
                Degree of the resulting matrix.

        Returns:
        -------
        vander : ndarray
                The Vandermonde matrix. Shape is x.shape + (deg + 1,).
        """
        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1)

        n = len(x)
        V = np.ones((n, deg + 1))  # noqa: N806

        # If we have interpolation nodes stored, use them
        if hasattr(self, "X") and self.X is not None:
            nodes = self.X.flatten() if self.X.ndim > 1 else self.X
        else:
            # Default: use equally spaced nodes from min to max of x
            nodes = np.linspace(x.min(), x.max(), deg + 1)

        # Build Newton basis: N_k(x) = ∏_{i=0}^{k-1} (x - nodes[i])
        for k in range(1, deg + 1):
            for i in range(k):
                if i < len(nodes):
                    V[:, k] *= x - nodes[i]

        return V

    def vander_der_fn(self, c, m=1):  # noqa: ANN001, ANN202
        """Differentiate a Newton Fundamental Polynomial series.

        Given coefficients c for a Newton basis representation, compute the
        coefficients of the m-th derivative.

        Parameters
        ----------
        c : array_like
                Array of Newton series coefficients.
        m : int, optional
                Number of derivatives taken, must be non-negative. (Default: 1)

        Returns:
        -------
        der : ndarray
                Newton series coefficients of the derivative.
        """
        c = np.array(c, ndmin=1, copy=True)

        if m < 0:
            raise ValueError("Order of derivative must be non-negative")
        if m == 0:
            return c

        # Get interpolation nodes
        if hasattr(self, "X") and self.X is not None:
            nodes = self.X.flatten() if self.X.ndim > 1 else self.X
        else:
            # Use zero nodes as default
            nodes = np.zeros(len(c))

        # Convert Newton form to standard polynomial form
        # Then differentiate and convert back
        poly_coef = self._newton_to_poly(c, nodes[: len(c)])

        # Differentiate in standard form
        for _ in range(m):
            if len(poly_coef) <= 1:
                return np.array([0.0])
            poly_coef = polyder(poly_coef)

        # Convert back to Newton form
        if len(poly_coef) == 0:
            return np.array([0.0])

        return self._poly_to_newton(poly_coef, nodes[: len(poly_coef)])

    def vander_roots_fn(self, c):  # noqa: ANN001, ANN202
        """Compute the roots of a Newton Fundamental Polynomial series.

        Parameters
        ----------
        c : 1-D array_like
                1-D array of Newton series coefficients.

        Returns:
        -------
        out : ndarray
                Array of the roots of the series.
        """
        c = np.array(c, ndmin=1, copy=True)

        # Get interpolation nodes
        if hasattr(self, "X") and self.X is not None:
            nodes = self.X.flatten() if self.X.ndim > 1 else self.X
        else:
            nodes = np.zeros(len(c))

        # Convert Newton form to standard polynomial form
        poly_coef = self._newton_to_poly(c, nodes[: len(c)])

        # Find roots using standard polynomial root finding
        return polyroots(poly_coef)

    def _newton_to_poly(self, newton_coef, nodes):  # noqa: ANN001, ANN202
        """Convert Newton form coefficients to standard polynomial form."""
        newton_coef = np.asarray(newton_coef)
        nodes = np.asarray(nodes)

        if len(newton_coef) == 0:
            return np.array([0.0])

        # Start with the highest degree term
        poly = np.array([newton_coef[-1]])

        # Work backwards through the Newton coefficients
        for i in range(len(newton_coef) - 2, -1, -1):
            # Multiply by (x - nodes[i])
            if i < len(nodes):
                poly = np.convolve(poly, [1, -nodes[i]])
            else:
                poly = np.convolve(poly, [1, 0])
            # Add the next coefficient
            poly[-1] += newton_coef[i]

        return poly

    def _poly_to_newton(self, poly_coef, nodes):  # noqa: ANN001, ANN202
        """Convert standard polynomial form to Newton form coefficients."""
        poly_coef = np.asarray(poly_coef, dtype=float)
        nodes = np.asarray(nodes)

        if len(poly_coef) == 0:
            return np.array([0.0])

        n = len(poly_coef)
        newton_coef = np.zeros(n)

        # Use divided differences to compute Newton coefficients
        # Create synthetic evaluation points
        if len(nodes) < n:
            # Extend nodes if needed
            extended_nodes = np.concatenate([nodes, np.zeros(n - len(nodes))])
        else:
            extended_nodes = nodes[:n]

        # Evaluate polynomial at nodes
        y_vals = np.polyval(poly_coef[::-1], extended_nodes)

        # Compute divided differences
        div_diff = y_vals.copy()
        for i in range(n):
            newton_coef[i] = div_diff[i]
            for j in range(n - 1, i, -1):
                denom = extended_nodes[j] - extended_nodes[j - i - 1]
                if abs(denom) > 1e-14:
                    div_diff[j] = (div_diff[j] - div_diff[j - 1]) / denom
                else:
                    div_diff[j] = 0.0

        return newton_coef


class HermiteTensorBasis(PolynomialTensorBasis):
    """A tensor product basis of bounded total degree built from the Hermite.

    polynomials.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D107
        self.vander = hermvander
        self.polyder = hermder
        self.polyroots = hermroots
        PolynomialTensorBasis.__init__(self, *args, **kwargs)

    def __name__(self):  # noqa: ANN204, D105
        return "HermiteTensorBasis"

    def _set_scale(self, X) -> None:  # noqa: ANN001, N803
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)

    def scale(self, X):  # noqa: ANN001, ANN201, D102, N803
        try:
            return (X - self._mean[None, :]) / self._std[None, :] / np.sqrt(2)
        except AttributeError:
            return X

    def dscale(self):  # noqa: ANN201, D102
        try:
            return 1.0 / self._std / np.sqrt(2)
        except AttributeError:
            raise NotImplementedError  # noqa: B904

    def roots(self, coef, dim):  # noqa: ANN001, ANN201, D102
        if dim > 1:
            raise NotImplementedError
        r = hermroots(coef)
        return r * self._std[0] * np.sqrt(2) + self._mean[0]


# TODO: Fix
class ArnoldiPolynomialBasis(Basis):
    r"""Construct a stable polynomial basis for arbitrary points using Vandermonde+Arnoldi."""

    def __init__(self, degree, problem, X=None, dim=None) -> None:  # noqa: ANN001, ARG002, D107, N803
        # Allow X to be optional; defer to provided dim if X is None
        self.X = np.copy(np.atleast_2d(X)) if X is not None else None
        self.dim = (
            dim
            if dim is not None
            else (self.X.shape[1] if self.X is not None else None)
        )
        self.degree = int(degree)
        # indices depend on dim; if dim is still None, defer and set on first use
        self.indices = (
            index_set(self.degree, self.dim).astype(int)
            if self.dim is not None
            else None
        )

        # self.Q, self.R = self.arnoldi()

    def __name__(self):  # noqa: ANN204, D105
        return "ArnoldiPolynomialBasis"

    def __len__(self) -> int:  # noqa: D105
        return len(self.indices)

    def set_X(self, X) -> None:  # noqa: ANN001, D102, N802, N803
        self.X = np.copy(np.atleast_2d(X)) if X is not None else None
        if self.X is not None and (self.dim is None):
            self.dim = self.X.shape[1]
        if self.indices is None and self.dim is not None:
            self.indices = index_set(self.degree, self.dim).astype(int)

    def assign_interpolation_set(self, X) -> None:  # noqa: ANN001, D102, N803
        # set_X mutates in-place; do not reassign its return value
        self.set_X(X)

    def _update_vec(self, ids):  # noqa: ANN001, ANN202
        # Determine which column to multiply by
        diff = self.indices - ids
        # Here we pick the most recent column that is one off
        cond = (np.sum(np.abs(diff), axis=1) <= 1) & (np.min(diff, axis=1) == -1)
        inds = np.nonzero(cond)[0]
        if inds.size == 0:
            raise ValueError(f"Unable to find update vector for ids={ids}")
        j = int(inds.max())
        i_candidates = np.nonzero(diff[j] == -1)[0]
        if i_candidates.size == 0:
            raise ValueError(f"Unable to find multiplicative index for ids={ids}")
        i = int(i_candidates[0])
        return i, j

    def arnoldi(self, X):  # noqa: ANN001, ANN201, N803
        """Apply the Arnoldi proceedure to build up columns of the Vandermonde matrix.

        Args:
                X (np.ndarray): an (M,n) array of M interpolation points

        Returns:
                (np.ndarray, np.ndarray): Elements Q and R
                        Q - An (M, n) array where the columns are an orthonormal basis
                        of the Krylov subspace
                        R - An (n, n) array where X is a basis on R, this is upper
                        Hessenberg
                        n is the length of the index set
        """
        # Ensure indices are available
        if self.indices is None:
            if getattr(self, "dim", None) is None:
                raise ValueError("Dimension not set for ArnoldiPolynomialBasis")
            self.indices = index_set(self.degree, self.dim).astype(int)

        idx = self.indices
        M = X.shape[0]  # noqa: N806

        # Allocate memory for matrices
        Q = np.zeros((M, len(idx)))  # noqa: N806
        R = np.zeros((len(idx), len(idx)))  # noqa: N806

        # First column: constant term
        Q[:, 0] = 1.0 / np.sqrt(M)
        R[0, 0] = np.sqrt(M)

        # Now work on the remaining columns (k from 1..N-1)
        N = len(idx)  # noqa: N806
        for k in range(1, N):
            ids = idx[k]
            i, j = self._update_vec(ids)
            i = int(i)
            j = int(j)
            # Form new column
            q = X[:, i] * Q[:, j]

            for s in range(k):
                R[s, k] = float(Q[:, s].T @ q)
                q = q - R[s, k] * Q[:, s]

            R[k, k] = np.linalg.norm(q)
            if R[k, k] != 0:
                Q[:, k] = q / R[k, k]
            else:
                # fallback: zero vector (cannot orthonormalize)
                Q[:, k] = np.zeros_like(q)

        self.Q = Q
        self.R = R
        return Q, R

    def arnoldi_X(self, X):  # noqa: ANN001, ANN201, N802, N803
        r"""Generate a Vandermonde matrix corresponding to a different set of points."""
        _Q, R = self.arnoldi(X)  # noqa: N806
        W = np.zeros((X.shape[0], len(self.indices)), dtype=X.dtype)  # noqa: N806

        # First column
        if R[0, 0] != 0:
            W[:, 0] = 1.0 / R[0, 0]
        else:
            W[:, 0] = 0.0

        N = len(self.indices)  # noqa: N806
        for k in range(1, N):
            ids = self.indices[k]
            i, j = self._update_vec(ids)
            i = int(i)
            j = int(j)
            w = X[:, i] * W[:, j]
            for s in range(k):
                w = w - R[s, k] * W[:, s]
            if R[k, k] != 0:
                W[:, k] = w / R[k, k]
            else:
                W[:, k] = w
        return W

    def V(self, X=None):  # noqa: ANN001, ANN201, D102, N802, N803
        # If no X provided, return cached Q (build if necessary)
        if X is None or (self.X is not None and np.array_equal(X, self.X)):
            if not hasattr(self, "Q") or self.Q is None:
                if self.X is None:
                    raise ValueError(
                        "No interpolation points available to build Vandermonde"
                    )
                self.arnoldi(self.X)
            return self.Q
        return self.arnoldi_X(X)

    def DV(self, X=None):  # noqa: ANN001, ANN201, D102, N802, N803
        if X is None or np.array_equal(X, self.X):
            X = self.X  # noqa: N806
            V = self.Q  # noqa: N806
        else:
            V = self.arnoldi_X(X)  # noqa: N806

        M = X.shape[0]  # noqa: N806
        N = self.Q.shape[1]  # noqa: N806
        n = self.X.shape[1]
        DV = np.zeros((M, N, n), dtype=self.Q.dtype)  # noqa: N806

        for ell in range(n):
            enumerate(self.indices)
            # skip first column
            # use explicit indexing to avoid iterator state issues
            for k in range(1, len(self.indices)):
                ids = self.indices[k]
                i, j = self._update_vec(ids)
                # Q[:,k] = X[:,i] * Q[:,j] - sum_s Q[:,s] * R[s, k]
                if i == ell:
                    DV[:, k, ell] = (
                        V[:, j]
                        + X[:, i] * DV[:, j, ell]
                        - DV[:, 0:k, ell] @ self.R[0:k, k]
                    )
                else:
                    DV[:, k, ell] = (
                        X[:, i] * DV[:, j, ell] - DV[:, 0:k, ell] @ self.R[0:k, k]
                    )
                den = self.R[k, k] if abs(self.R[k, k]) > 1e-12 else 1e-12
                DV[:, k, ell] /= den

        return DV

    def DDV(self, X=None):  # noqa: ANN001, ANN201, D102, N802, N803
        raise NotImplementedError


class PolynomialBasis(Basis):
    # polyder - derivative of the polynomial basis series
    # vander - vandermonde matrix of the polynomial basis series
    # polyroots - the roots of the of the polynomial basis series

    def __init__(self, degree, dim) -> None:  # noqa: ANN001
        self.degree = int(degree)
        self.dim = int(dim)

        self._build_Dmat()

    def __name__(self):  # noqa: ANN204
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.indices)

    @abstractmethod
    def poly_basis_fn(
        self, interpolation_set: list[np.ndarray], row_num: int, col_num: int
    ) -> np.float64:
        raise NotImplementedError

    @abstractmethod
    def polyder(self, coeff: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def polyroots(self, coeff: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def vander(self, interpolation_set: np.ndarray) -> np.ndarray:
        """Generate Vandermonde-like matrix.

        Uses vectorized computation via vander_vectorized if implemented,
        otherwise falls back to element-wise poly_basis_fn.

        Args:
            interpolation_set: Input points, shape (M,) for 1D.

        Returns:
            Vandermonde matrix of shape (M, degree+1).
        """
        interpolation_set = np.asarray(interpolation_set)

        # Use vectorized implementation if available
        if hasattr(self, "vander_vectorized"):
            return self.vander_vectorized(interpolation_set)

        # Fallback to element-wise construction
        X_shape = (*interpolation_set.shape, self.degree + 1)  # noqa: N806
        X = np.zeros(X_shape)  # noqa: N806

        if len(interpolation_set.shape) == 1:
            # 1D case: shape (M,) -> output shape (M, degree+1)
            M = X_shape[0]  # noqa: N806
            for i in range(M):
                for j in range(X_shape[1]):
                    X[i, j] = self.poly_basis_fn(interpolation_set, i, j)
        else:
            # Multi-D case: construct matrix for each row
            for i in range(len(interpolation_set)):
                for j in range(X_shape[-1]):
                    X[i, j] = self.poly_basis_fn(interpolation_set, i, j)
        return X

    # @property
    # def X(self) :
    # 	return self.X

    def assign_interpolation_set(self, X) -> None:  # noqa: ANN001, N803
        self.X = X

    @abstractmethod
    def solve_no_cols(self, interpolation_set) -> int:  # noqa: ANN001
        raise NotImplementedError

    def _build_Dmat(self) -> None:  # noqa: N802
        """Constructs the (scalar) derivative matrix.

        For polynomial degree p, creates a (p+1, p) matrix where Dmat[j, :] contains
        the coefficients of the derivative of the j-th basis polynomial.
        """
        self.Dmat = np.zeros((self.degree + 1, self.degree))
        I = np.eye(self.degree + 1)  # noqa: E741, N806
        for j in range(self.degree + 1):
            der = self.polyder(I[:, j])
            # Ensure derivative coefficients fit in the Dmat row
            # polyder returns array of length (degree) for input of length (degree+1)
            der = np.asarray(der).flatten()
            n = min(len(der), self.degree)
            self.Dmat[j, :n] = der[:n]

    def set_scale(self, X) -> None:  # noqa: ANN001, N803
        r"""Construct an affine transformation of the domain to improve the conditioning."""
        self._set_scale(np.array(X))

    def _set_scale(self, X) -> None:  # noqa: ANN001, N803
        r"""Default scaling to [-1,1]."""
        self._lb = np.min(X, axis=0)
        self._ub = np.max(X, axis=0)

    def scale(self, X):  # noqa: ANN001, ANN202, N803
        r"""Apply the scaling to the input coordinates."""
        try:
            return (
                2 * (X - self._lb[None, :]) / (self._ub[None, :] - self._lb[None, :])
                - 1
            )
        except AttributeError:
            return X

    def dscale(self):  # noqa: ANN202
        r"""Returns the scaling associated with the scaling transform."""
        try:
            return 2.0 / (self._ub - self._lb)
        except AttributeError:
            raise NotImplementedError  # noqa: B904

    # Construct a matrix Row by Row
    def V(self, X: np.ndarray) -> np.ndarray:  # noqa: N802, N803
        dim = X.shape[1]
        self.indices = index_set(self.degree, dim).astype(int)
        # print(f'dimension: {self.dim}')
        # print(f'shape of X: {X.shape}')
        # X = X.reshape(-1, dim)
        self.X = X
        X = self.scale(np.array(X))  # noqa: N806
        M = X.shape[0]  # noqa: N806
        assert X.shape[1] == dim, "Expected %d dimensions, got %d" % (dim, X.shape[1])  # noqa: UP031
        V_coordinate = [self.vander(X[:, k]) for k in range(dim)]  # noqa: N806

        V = np.ones((M, len(self.indices)), dtype=X.dtype)  # noqa: N806

        for j, alpha in enumerate(self.indices):
            for k in range(dim):
                V[:, j] *= V_coordinate[k][:, alpha[k]]

        return V

    def DV(self, X):  # noqa: ANN001, ANN202, N802, N803
        dim = X.shape[1]
        self.indices = index_set(self.degree, dim).astype(int)
        # X = X.reshape(-1, dim)
        X = self.scale(np.array(X))  # noqa: N806
        M = X.shape[0]  # noqa: N806
        V_coordinate = [self.vander(X[:, k]) for k in range(dim)]  # noqa: N806

        N = len(self.indices)  # noqa: N806
        DV = np.ones((M, N, dim), dtype=X.dtype)  # noqa: N806

        try:
            dscale = self.dscale()
        except NotImplementedError:
            dscale = np.ones(X.shape[1])

        for k in range(dim):
            for j, alpha in enumerate(self.indices):
                for q in range(dim):
                    if q == k:
                        DV[:, j, k] *= np.dot(
                            V_coordinate[q][:, 0:-1], self.Dmat[alpha[q], :]
                        )
                    else:
                        DV[:, j, k] *= V_coordinate[q][:, alpha[q]]
            # Correct for transform
            DV[:, :, k] *= dscale[k]

        return DV

    def DDV(self, X):  # noqa: ANN001, ANN202, N802, N803
        dim = X.shape[1]
        self.indices = index_set(self.degree, dim).astype(int)
        # X = X.reshape(-1, dim)
        X = self.scale(np.array(X))  # noqa: N806
        M = X.shape[0]  # noqa: N806
        V_coordinate = [self.vander(X[:, k]) for k in range(dim)]  # noqa: N806

        N = len(self.indices)  # noqa: N806
        DDV = np.ones((M, N, dim, dim), dtype=X.dtype)  # noqa: N806

        try:
            dscale = self.dscale()
        except NotImplementedError:
            dscale = np.ones(X.shape[1])

        for k in range(dim):
            for ell in range(k, dim):
                for j, alpha in enumerate(self.indices):
                    for q in range(dim):
                        if q == k == ell:
                            # We need the second derivative
                            eq = np.zeros(self.degree + 1)
                            eq[alpha[q]] = 1.0
                            der2 = self.polyder(self.polyder(eq))
                            DDV[:, j, k, ell] *= V_coordinate[q][:, 0 : len(der2)].dot(
                                der2
                            )
                        elif q in (k, ell):
                            DDV[:, j, k, ell] *= np.dot(
                                V_coordinate[q][:, 0:-1], self.Dmat[alpha[q], :]
                            )
                        else:
                            DDV[:, j, k, ell] *= V_coordinate[q][:, alpha[q]]

                # Correct for transform
                DDV[:, :, k, ell] *= dscale[k] * dscale[ell]
                DDV[:, :, ell, k] = DDV[:, :, k, ell]
        return DDV

    def roots(self, coef):  # noqa: ANN001, ANN202
        # if dim > 1:
        # 	raise NotImplementedError
        r = self.polyroots(coef)
        return r * (self._ub[0] - self._lb[0]) / 2.0 + (self._ub[0] + self._lb[0]) / 2.0


class NaturalPolynomialBasis(PolynomialBasis):  # noqa: D101
    # def __init__(self, degree, problem, X=None, dim=None) :
    # 	super().__init__(degree, problem, X, dim)

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D107
        PolynomialBasis.__init__(self, *args, **kwargs)

    def __name__(self):  # noqa: ANN204, D105
        return "NaturalPolynomialBasis"

    def vander_vectorized(self, x: np.ndarray) -> np.ndarray:
        """Vectorized Vandermonde matrix for natural polynomial basis.

        Natural polynomial basis: [1, x, x^2/2!, x^3/3!, ..., x^p/p!]

        Args:
            x: Input array of shape (M,)

        Returns:
            Vandermonde matrix of shape (M, degree+1)
        """
        x = np.asarray(x)
        M = len(x)  # noqa: N806
        V = np.zeros((M, self.degree + 1))  # noqa: N806

        # Compute powers of x divided by factorials
        # V[:, j] = x^j / j!
        V[:, 0] = 1.0
        for j in range(1, self.degree + 1):
            V[:, j] = V[:, j - 1] * x / j  # x^j/j! = (x^(j-1)/(j-1)!) * (x/j)

        return V

    # TODO: the factorial(degree) in this function can lead to a stackoverflow due to the recursive limit being reached
    def poly_basis_fn(self, interpolation_set, row_num, col_num):  # noqa: ANN001, ANN201, D102
        # val = interpolation_set[row_num]
        # val_dim = len(val) if not isinstance(val, float) or isinstance(val, int) else 1
        point = interpolation_set[
            row_num
        ]  # The row corresponds to the n-dimensional point in X

        if isinstance(point, (np.float64, np.int64)):
            point = [point]

        if col_num == 0:
            return 1  # Constant term for the first column

        # Determine which term this column corresponds to
        degree = 1
        while True:
            terms_for_degree = (degree + 1) * degree // 2
            if col_num <= terms_for_degree:
                break
            degree += 1

        term_position = col_num - (degree * (degree - 1) // 2) - 1

        if term_position < degree:
            # Diagonal terms like x_i^d / d!
            component = term_position
            if len(point) == 1:
                component = 0
            return (point[component] ** degree) / factorial(degree)
        # Cross terms like x_i * x_j
        i = term_position - degree
        j = degree - 1 - i
        return point[i] * point[j]

    """def poly_basis_fn(self, interpolation_set, row_num, col_num) : 
		val = interpolation_set[row_num]
		p = len(val) if not isinstance(val, float) or isinstance(val, int) else 1
		
		# Define the full list of basis terms, up to max_degree
		basis_terms = []
	
		# Add the constant term 1
		basis_terms.append(lambda v: 1)
		
		# Add all first-order terms x_1, x_2, ..., x_p
		for i in range(p):
			if isinstance(val, float) or isinstance(val, int) :
				basis_terms.append(lambda v : v)
			else :
				basis_terms.append(lambda v : v[i])

		
		# Add higher-order terms up to degree max_degree
		for degree in range(2, self.degree+2):
			for comb in combinations_with_replacement(range(p), degree):
				def term_func(v, comb=comb, degree=degree):
					result = 1
					for idx in comb:
						result *= v[idx]
					inner_result = result / factorial(degree)
					return inner_result
				def term_func_float(v, comb=comb, degree=degree):
					result = 1
					for idx in comb:
						result *= v
					inner_result = result / factorial(degree)
					return inner_result
				if isinstance(val, float) or isinstance(val, int) :
					basis_terms.append(term_func_float)
				else :
					basis_terms.append(term_func)
		# Evaluate the basis term at the given row (vector)
		print('length of basis terms: ', len(basis_terms))
		res = basis_terms[col_num](val)
		# return basis_terms[col_num](val)
		return res"""

    def polyder(self, coeff: np.ndarray, m: int = 1) -> np.ndarray:
        """Compute the derivative of a polynomial with given coefficients.

        Fast NumPy-based implementation that avoids symbolic computation.
        For coefficients [a_0, a_1, a_2, ...] representing a_0 + a_1*x + a_2*x^2 + ...,
        returns the coefficients of the derivative.

        Args:
            coeff (np.ndarray): Polynomial coefficients in ascending power order.
            m (int): Number of derivatives to take. Default is 1.

        Returns:
            np.ndarray: Coefficients of the derivative polynomial.
        """
        coeff = np.asarray(coeff)

        if m < 0:
            raise ValueError("Order of derivative must be non-negative")
        if m == 0:
            return coeff

        # For each derivative, multiply by power and shift
        for _ in range(m):
            if len(coeff) <= 1:
                return np.array([0.0])
            # Derivative: a_i * x^i -> i * a_i * x^(i-1)
            # New coefficients: [1*a_1, 2*a_2, 3*a_3, ...]
            n = len(coeff)
            powers = np.arange(1, n)
            coeff = coeff[1:] * powers

        if len(coeff) == 0:
            return np.array([0.0])
        return coeff

    def polyroots(self, coeff: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.roots(coeff)

    def solve_no_cols(self, interpolation_set):  # noqa: ANN001, ANN201, D102
        # val = interpolation_set[0]
        if isinstance(interpolation_set[0], float) or isinstance(  # noqa: SIM101
            interpolation_set[0], int
        ):
            sample_dim = 1
        else:
            sample_dim = len(interpolation_set[0])

        return comb(sample_dim + self.degree, self.degree)


class MonomialPolynomialBasis(PolynomialBasis):  # noqa: D101
    # def __init__(self, degree, problem, X=None, dim=None):
    # 	super().__init__(degree, problem, X, dim)

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D107
        PolynomialBasis.__init__(self, *args, **kwargs)

    def __name__(self):  # noqa: ANN204, D105
        return "MonomialPolynomialBasis"

    def vander_vectorized(self, x: np.ndarray) -> np.ndarray:
        """Vectorized Vandermonde matrix for monomial basis.

        Monomial basis: [1, x, x^2, x^3, ..., x^p]

        Args:
            x: Input array of shape (M,)

        Returns:
            Vandermonde matrix of shape (M, degree+1)
        """
        x = np.asarray(x)
        len(x)
        # Use broadcasting: x[:, None] ** powers[None, :]
        powers = np.arange(self.degree + 1)
        return x[:, np.newaxis] ** powers[np.newaxis, :]

    def poly_basis_fn(self, interpolation_set, row_num, col_num):  # noqa: ANN001, ANN201, D102
        interpolation_set = np.array(interpolation_set)
        val = interpolation_set[row_num]
        # calculate the whole row,
        row = [1]
        for exp in range(1, col_num + 1):
            row = np.append(row, val**exp)

        return row[col_num]

    def polyder(self, coeff: np.ndarray, m: int = 1) -> np.ndarray:
        """Compute the derivative of a polynomial with given coefficients.

        Fast NumPy-based implementation that avoids symbolic computation.
        For coefficients [a_0, a_1, a_2, ...] representing a_0 + a_1*x + a_2*x^2 + ...,
        returns the coefficients of the derivative.

        Args:
            coeff (np.ndarray): Polynomial coefficients in ascending power order.
            m (int): Number of derivatives to take. Default is 1.

        Returns:
            np.ndarray: Coefficients of the derivative polynomial.
        """
        coeff = np.asarray(coeff)

        if m < 0:
            raise ValueError("Order of derivative must be non-negative")
        if m == 0:
            return coeff

        # For each derivative, multiply by power and shift
        for _ in range(m):
            if len(coeff) <= 1:
                return np.array([0.0])
            # Derivative: a_i * x^i -> i * a_i * x^(i-1)
            # New coefficients: [1*a_1, 2*a_2, 3*a_3, ...]
            n = len(coeff)
            powers = np.arange(1, n)
            coeff = coeff[1:] * powers

        if len(coeff) == 0:
            return np.array([0.0])
        return coeff

    def polyroots(self, coeff: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.roots(coeff)

    def solve_no_cols(self, interpolation_set):  # noqa: ANN001, ANN201, D102
        val = interpolation_set[0]
        return 1 + (len(val) * self.degree)


class LagrangePolynomialBasis(PolynomialBasis):  # noqa: D101
    # def __init__(self, degree, problem, X=None, dim=None) :
    # 	super().__init__(degree,  X, dim)

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D107
        PolynomialBasis.__init__(self, *args, **kwargs)

    def __name__(self):  # noqa: ANN204, D105
        return "LagrangePolynomialBasis"

    # lagrange polynomial function for each element in the matrix
    def poly_basis_fn(self, interpolation_set, row_num, col_num):  # noqa: ANN001, ANN201
        """Evaluate the col_num-th Lagrange basis function at.

        interpolation_set[row_num].

        For multidimensional case, this evaluates the tensor product of 1D Lagrange
        polynomials.
        """
        # For the tensor product case, we need the stored interpolation nodes
        if not hasattr(self, "X") or self.X is None:
            raise ValueError(
                "Interpolation nodes self.X must be set before evaluating basis"
            )

        # Get the evaluation point
        x = np.atleast_1d(interpolation_set[row_num])
        dim = len(x)

        # Get the node corresponding to this basis function
        nodes = self.X
        M = len(nodes)  # noqa: N806

        if col_num >= M:
            raise IndexError(
                f"Column index {col_num} out of range for {M} basis functions"
            )

        # For each dimension, compute the 1D Lagrange polynomial
        result = 1.0
        for d in range(dim):
            lagrange_1d = 1.0
            x_d = x[d]
            node_d = nodes[col_num, d]

            for k in range(M):
                if k == col_num:
                    continue
                numerator = x_d - nodes[k, d]
                denominator = node_d - nodes[k, d]

                if abs(denominator) < 1e-14:
                    # Duplicate nodes - handle carefully
                    if abs(numerator) < 1e-14:
                        lagrange_1d *= 1.0
                    else:
                        lagrange_1d = 0.0
                        break
                else:
                    lagrange_1d *= numerator / denominator

            result *= lagrange_1d

        return result

    def V(self, X):  # noqa: ANN001, ANN201, N802, N803
        """Construct Vandermonde matrix for Lagrange polynomial basis.

        V[i, j] = L_j(X[i]) where L_j is the j-th Lagrange basis polynomial.

        Parameters
        ----------
        X : np.ndarray
                Evaluation points of shape (N, dim)

        Returns:
        -------
        V : np.ndarray
                Vandermonde matrix of shape (N, M) where M = len(self.X)
        """
        X = np.atleast_2d(X)  # noqa: N806
        N, _dim = X.shape  # noqa: N806

        if self.X is None:
            raise ValueError("Interpolation nodes self.X must be set")

        M = len(self.X)  # Number of basis functions  # noqa: N806
        V = np.zeros((N, M))  # noqa: N806

        # Check if evaluating at the interpolation nodes (optimization)
        if np.allclose(X, self.X) and N == M:
            return np.eye(N, M)

        # General case: compute Lagrange polynomials
        for i in range(N):
            for j in range(M):
                V[i, j] = self.poly_basis_fn(X, i, j)

        return V

    def _build_Dmat(self) -> None:  # noqa: N802
        return None

    def DV(self, X=None):  # noqa: ANN001, ANN201, N802, N803
        """Compute the column-wise derivative of the Vandermonde matrix for Lagrange.

        basis.

        For Lagrange polynomials L_j(x), the derivative at node x_i is:
        dL_j/dx|_{x=x_i} = sum_{k≠j} 1/(x_i - x_k) * product_{m≠j,k} (x_i - x_m)/(x_j -
        x_m)

        Parameters
        ----------
        X : np.ndarray, optional
                Points at which to evaluate derivatives. If None, uses self.X

        Returns:
        -------
        DV : np.ndarray
                Derivative matrix of shape (N, M, dim) where N is number of points,
                M is number of basis functions, dim is the dimension
        """
        if X is None:
            X = self.X  # noqa: N806

        if X is None:
            raise ValueError("No interpolation points provided")

        X = np.atleast_2d(X)  # noqa: N806
        N, dim = X.shape  # noqa: N806
        M = len(self.X)  # Number of Lagrange basis functions  # noqa: N806

        DV = np.zeros((N, M, dim))  # noqa: N806

        # For each dimension
        for d in range(dim):
            # Get the interpolation nodes for this dimension
            nodes = self.X[:, d]

            # For each basis function j
            for j in range(M):
                x_j = nodes[j]

                # For each evaluation point i
                for i in range(N):
                    x_i = X[i, d]

                    # Compute derivative of L_j at x_i
                    derivative = 0.0

                    # For Lagrange polynomial L_j(x) = prod_{k≠j} (x - x_k)/(x_j - x_k)
                    # The derivative is: sum_{k≠j} [1/(x_j - x_k) * prod_{m≠j,k} (x - x_m)/(x_j - x_m)]

                    for k in range(M):
                        if k == j:
                            continue

                        # Compute the term for index k
                        numerator_deriv = 1.0
                        for m in range(M):
                            if m in (j, k):
                                continue
                            numerator_deriv *= (x_i - nodes[m]) / (x_j - nodes[m])

                        denominator_k = x_j - nodes[k]
                        if abs(denominator_k) > 1e-14:  # Avoid division by zero
                            derivative += numerator_deriv / denominator_k

                    DV[i, j, d] = derivative

        return DV

    def DDV(self, X=None):  # noqa: ANN001, ANN201, D102, N802, N803
        raise NotImplementedError

    def polyroots(self, coeff: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.roots(coeff)

    def solve_no_cols(self, interpolation_set):  # noqa: ANN001, ANN201, D102
        return len(interpolation_set)


class NFPPolynomialBasis(PolynomialBasis):  # noqa: D101
    # def __init__(self, degree, X=None, dim=None) :
    # 	super().__init__(degree, X, dim)

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D107
        PolynomialBasis.__init__(self, *args, **kwargs)

    def __name__(self):  # noqa: ANN204, D105
        return "NFPPolynomialBasis"

    def vander_vectorized(self, x: np.ndarray) -> np.ndarray:
        """Vectorized Vandermonde matrix for Newton Fundamental Polynomial basis.

        NFP basis: φ_0(x) = 1, φ_j(x) = ∏_{k=0}^{j-1} (x - x_k) for j > 0
        where x_k are the interpolation nodes.

        Args:
            x: Input array of shape (M,)

        Returns:
            Vandermonde matrix of shape (M, degree+1)
        """
        x = np.asarray(x)
        M = len(x)  # noqa: N806
        V = np.zeros((M, self.degree + 1))  # noqa: N806

        # Get interpolation nodes (the input x itself serves as nodes for NFP)
        # For 1D case, use x values as nodes
        nodes = x[: self.degree + 1] if len(x) >= self.degree + 1 else x

        # φ_0(x) = 1
        V[:, 0] = 1.0

        # φ_j(x) = ∏_{k=0}^{j-1} (x - nodes[k])
        # Build iteratively: φ_j(x) = φ_{j-1}(x) * (x - nodes[j-1])
        for j in range(1, self.degree + 1):
            if j - 1 < len(nodes):
                V[:, j] = V[:, j - 1] * (x - nodes[j - 1])
            else:
                # If we don't have enough nodes, use 0
                V[:, j] = V[:, j - 1] * x

        return V

    # TODO: Fix the polynomial basis function
    def poly_basis_fn(self, interpolation_set, row_num, col_num):  # noqa: ANN001, ANN201, D102
        if col_num == 0:
            return 1
        val = np.array(interpolation_set[row_num])
        res = np.prod([(val - np.array(a)) for a in interpolation_set[:col_num]])
        return float(res)

    def _build_Dmat(self) -> None:  # noqa: N802
        return None

    def DV(self, X=None):  # noqa: ANN001, ANN201, N802, N803
        """Compute the column-wise derivative of the Vandermonde matrix for NFP basis.

        For NFP (Newton Forward Polynomial) basis, the basis functions are:
        - φ_0(x) = 1
        - φ_j(x) = ∏_{k=0}^{j-1} (x - x_k) for j > 0

        The derivative of φ_j(x) is computed using the product rule:
        dφ_j/dx = ∑_{i=0}^{j-1} [∏_{k=0, k≠i}^{j-1} (x - x_k)]

        Parameters
        ----------
        X : np.ndarray, optional
                Points at which to evaluate derivatives. If None, uses self.X

        Returns:
        -------
        DV : np.ndarray
                Derivative matrix of shape (N, M, dim) where N is number of points,
                M is number of basis functions, dim is the dimension
        """
        if X is None:
            X = self.X  # noqa: N806

        if X is None:
            raise ValueError("No interpolation points provided")

        X = np.atleast_2d(X)  # noqa: N806
        N, dim = X.shape  # noqa: N806

        if self.X is None:
            raise ValueError("Interpolation nodes self.X must be set")

        M = len(self.X)  # Number of basis functions  # noqa: N806
        DV = np.zeros((N, M, dim))  # noqa: N806

        # For each dimension
        for d in range(dim):
            # Get the interpolation nodes for this dimension
            nodes = self.X[:, d]

            # For each basis function j
            for j in range(M):
                # For each evaluation point i
                for i in range(N):
                    x_i = X[i, d]

                    if j == 0:
                        # Derivative of constant is 0
                        DV[i, j, d] = 0.0
                    else:
                        # φ_j(x) = ∏_{k=0}^{j-1} (x - x_k)
                        # dφ_j/dx = ∑_{m=0}^{j-1} [∏_{k=0, k≠m}^{j-1} (x - x_k)]
                        derivative = 0.0

                        for m in range(j):
                            # Compute the product excluding term m
                            term = 1.0
                            for k in range(j):
                                if k != m:
                                    term *= x_i - nodes[k]
                            derivative += term

                        DV[i, j, d] = derivative

        return DV

    def DDV(self, X=None):  # noqa: ANN001, ANN201, D102, N802, N803
        raise NotImplementedError

    def polyroots(self, coeff: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.roots(coeff)

    def solve_no_cols(self, interpolation_set):  # noqa: ANN001, ANN201, D102
        return len(interpolation_set)


class AstroDFBasis:  # noqa: D101
    def __init__(self, degree, problem, X=None, dim=None) -> None:  # noqa: ANN001, ARG002, D107, N803
        self.degree = int(degree)
        if X is not None:
            self.X = np.atleast_2d(X)
            self.dim = self.X.shape[1]
            self.set_scale(self.X)
        elif dim is not None:
            self.dim = int(dim)
            self.X = None

    def __name__(self):  # noqa: ANN204, D105
        return "AstroDFBasis"

    def assign_interpolation_set(self, X) -> None:  # noqa: ANN001, D102, N803
        self.X = X

    def V(self, interpolation_set: np.ndarray, dim: int) -> np.ndarray:  # noqa: ARG002, D102, N802
        X_shape = (len(interpolation_set), self.degree * len(interpolation_set[0]) + 1)  # noqa: N806
        # calculate the whole row,
        X = np.zeros(X_shape)  # noqa: N806
        for i in range(X_shape[0]):
            for j in range(X_shape[1]):
                X[i, j] = self.poly_basis_fn(interpolation_set, i, j)
        return X

    def poly_basis_fn(self, interpolation_set, row_num, col_num):  # noqa: ANN001, ANN201, D102
        interpolation_set = np.array(interpolation_set)
        val = interpolation_set[row_num]
        # calculate the whole row,
        row = [1]
        for exp in range(1, self.degree + 1):
            row = np.append(row, val**exp)
        return row[col_num]


#! UNFINISHED CLASSES
class BasisCombination(PolynomialBasis):
    """Class combines any multiple of the polynomial bases together when constructing.

    the vandermonde matrix.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        PolynomialBasis.__init__(self, *args, **kwargs)

    def combine_bases(self, *bases) -> None:  # noqa: ANN002
        self.bases = bases

    def poly_basis_fn(self, interpolation_set, row_num, col_num) -> None:  # noqa: ANN001
        pass


class BasisFittingGA:
    """A genetic algorithm that chooses the best polynomial basis to use."""

    def __init__(self) -> None:
        pass
