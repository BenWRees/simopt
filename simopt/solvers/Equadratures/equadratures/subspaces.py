import warnings  # noqa: D100

# import equadratures.plot as plot
import numpy as np
import scipy
import scipy.io
from scipy.linalg import orth, sqrtm
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.special import comb

from equadratures.basis import Basis
from equadratures.parameter import Parameter
from equadratures.poly import Poly
from equadratures.scalers import scaler_custom, scaler_minmax


class Subspaces:
    """This class defines a subspaces object. It can be used for polynomial-based
    subspace dimension reduction.

    Parameters
    ----------
    method : str
        The method to be used for subspace-based dimension reduction. Two options:

        - ``active-subspace``, which uses ideas in [1] and [2] to compute a dimension-
        reducing subspace with a global polynomial approximant. Gradients evaluations of
        the polynomial approximation are used to compute the averaged outer product of
        the gradient covariance matrix. The polynomial approximation in the original
        full-space can be provided via ``full_space_poly``. Otherwise, it is fit
        internally to the data provided via ``sample_points`` and ``sample_outputs``.
        - ``variable-projection`` [3], where a Gauss-Newton optimisation problem is
        solved to compute both the polynomial coefficients and its subspace, with the
        data provided via ``sample_points`` and ``sample_outputs``.

    full_space_poly : Poly, optional
        An instance of Poly fitted to the full-space data, to use for the AS
        computation.
    sample_points : numpy.ndarray, optional
        Array with shape (number_of_observations, dimensions) that corresponds to a set
        of sample points over the parameter space.
    sample_outputs : numpy.ndarray, optional
        Array with shape (number_of_observations, 1) that corresponds to model
        evaluations at the sample points.
    subspace_dimension : int, optional
        The dimension of the *active* subspace.
    param_args : dict, optional
        Arguments passed to parameters of the AS polynomial. (see
        :class:`~equadratures.parameter.Parameter`)
    poly_args : dict , optional
        Arguments passed to constructing polynomial used for AS computation. (see
        :class:`~equadratures.poly.Poly`)
    dr_args : dict, optional
        Arguments passed to customise the VP optimiser. See documentation for
        :meth:`~equadratures.subspaces.Subspaces._get_variable_projection` in source.

    Examples:
    --------
    Obtaining a 2D subspace via active subspaces on user data
        >>> mysubspace = Subspaces(method='active-subspace', sample_points=X, sample_outputs=Y)
        >>> eigs = mysubspace.get_eigenvalues()
        >>> W = mysubspace.get_subspace()[:, :2]
        >>> e = mysubspace.get_eigenvalues()

    Obtaining a 2D subspace via active subspaces with a Poly object (remember to call
    set_model() on Poly first)
        >>> mysubspace = Subspaces(method='active-subspace', full_space_poly=my_poly)
        >>> eigs = mysubspace.get_eigenvalues()
        >>> W = mysubspace.get_subspace()[:, :2]
        >>> e = mysubspace.get_eigenvalues()

    Obtaining a 2D subspace via variable projection on user data
        >>> mysubspace = Subspaces(method='variable-projection', sample_points=X, sample_outputs=Y)
        >>> W = mysubspace.get_subspace()[:, :2]

    References:
    ----------
    1. Constantine, P., (2015) Active Subspaces: Emerging Ideas for Dimension Reduction
    in Parameter Studies. SIAM Spotlights.
    2. Seshadri, P., Shahpar, S., Constantine, P., Parks, G., Adams, M. (2018)
    Turbomachinery Active Subspace Performance Maps. Journal of Turbomachinery, 140(4),
    041003. `Paper <http://turbomachinery.asmedigitalcollection.asme.org/article.aspx?ar
    ticleid=2668256>`__.
    3. Hokanson, J., Constantine, P., (2018) Data-driven Polynomial Ridge Approximation
    Using Variable Projection. SIAM Journal of Scientific Computing, 40(3), A1566-A1589.
    `Paper <https://epubs.siam.org/doi/abs/10.1137/17M1117690>`__.
    """

    def __init__(  # noqa: D107
        self,
        method,  # noqa: ANN001
        full_space_poly=None,  # noqa: ANN001
        sample_points=None,  # noqa: ANN001
        sample_outputs=None,  # noqa: ANN001
        subspace_dimension=2,  # noqa: ANN001
        polynomial_degree=2,  # noqa: ANN001
        param_args=None,  # noqa: ANN001
        poly_args=None,  # noqa: ANN001
        dr_args=None,  # noqa: ANN001
    ) -> None:
        self.full_space_poly = full_space_poly
        self.sample_points = sample_points
        self.Y = None  # for the zonotope vertices
        self.sample_outputs = sample_outputs
        self.method = method
        self.subspace_dimension = subspace_dimension
        self.polynomial_degree = polynomial_degree

        my_poly_args = {"method": "least-squares", "solver_args": {}}
        if poly_args is not None:
            my_poly_args.update(poly_args)
        self.poly_args = my_poly_args

        my_param_args = {
            "distribution": "uniform",
            "order": self.polynomial_degree,
            "lower": -1,
            "upper": 1,
        }
        if param_args is not None:
            my_param_args.update(param_args)

        # I suppose we can detect if lower and upper is present to decide between these categories?
        bounded_distrs = [
            "analytical",
            "beta",
            "chebyshev",
            "arcsine",
            "truncated-gaussian",
            "uniform",
        ]
        unbounded_distrs = [
            "gaussian",
            "normal",
            "gumbel",
            "logistic",
            "students-t",
            "studentst",
        ]

        if dr_args is not None and "standardize" in dr_args:
            dr_args["standardise"] = dr_args["standardize"]

        if (
            self.method.lower() == "active-subspace"
            or self.method.lower() == "active-subspaces"
        ):
            self.method = "active-subspace"
            if dr_args is not None:
                self.standardise = getattr(dr_args, "standardise", True)
            else:
                self.standardise = True

            if self.full_space_poly is None:
                # user provided input/output data
                _N, d = self.sample_points.shape  # noqa: N806
                if self.standardise:
                    self.data_scaler = scaler_minmax()
                    self.data_scaler.fit(self.sample_points)
                    self.std_sample_points = self.data_scaler.transform(
                        self.sample_points
                    )
                else:
                    self.std_sample_points = self.sample_points.copy()
                param = Parameter(**my_param_args)
                if param_args is not None:  # noqa: SIM102
                    if (
                        hasattr(dr_args, "lower") or hasattr(dr_args, "upper")
                    ) and self.standardise:
                        warnings.warn(
                            "Points standardised but parameter range provided. Overriding default ([-1,1])...",
                            UserWarning, stacklevel=2,
                        )
                myparameters = [param for _ in range(d)]
                mybasis = Basis("total-order")
                mypoly = Poly(
                    myparameters,
                    mybasis,
                    sampling_args={
                        "sample-points": self.std_sample_points,
                        "sample-outputs": self.sample_outputs,
                    },
                    **my_poly_args,
                )
                mypoly.set_model()
                self.full_space_poly = mypoly
            else:
                # User provided polynomial
                # Standardise according to distribution specified. Only care about the scaling (not shift)
                # TODO: user provided callable with parameters?
                user_params = self.full_space_poly.parameters
                d = len(user_params)
                self.sample_points = self.full_space_poly.get_points()
                if self.standardise:
                    scale_factors = np.zeros(d)
                    centers = np.zeros(d)
                    for dd, p in enumerate(user_params):
                        if p.name.lower() in bounded_distrs:
                            scale_factors[dd] = (p.upper - p.lower) / 2.0
                            centers[dd] = (p.upper + p.lower) / 2.0
                        elif p.name.lower() in unbounded_distrs:
                            scale_factors[dd] = np.sqrt(p.variance)
                            centers[dd] = p.mean
                        else:
                            scale_factors[dd] = np.sqrt(p.variance)
                            centers[dd] = 0.0
                    self.param_scaler = scaler_custom(centers, scale_factors)
                    self.std_sample_points = self.param_scaler.transform(
                        self.sample_points
                    )
                else:
                    self.std_sample_points = self.sample_points.copy()
                if not hasattr(self.full_space_poly, "coefficients"):
                    raise ValueError("Please call set_model() first on poly.")

            self.sample_outputs = self.full_space_poly.get_model_evaluations()
            # TODO: use dr_args for resampling of gradient points
            as_args = {"grad_points": None}
            if dr_args is not None:
                as_args.update(dr_args)
            self._get_active_subspace(**as_args)
        elif self.method == "variable-projection":
            self.data_scaler = scaler_minmax()
            self.data_scaler.fit(self.sample_points)
            self.std_sample_points = self.data_scaler.transform(self.sample_points)

            if dr_args is not None:
                vp_args = {
                    "gamma": 0.1,
                    "beta": 1e-4,
                    "tol": 1e-7,
                    "maxiter": 1000,
                    "U0": None,
                    "verbose": False,
                }
                vp_args.update(dr_args)
                self._get_variable_projection(**vp_args)
            else:
                self._get_variable_projection()

    def get_subspace_polynomial(self):  # noqa: ANN201
        """Returns a polynomial defined over the dimension reducing subspace.

        Returns:
        -------
        Poly
            A Poly object that defines a polynomial over the subspace. The distribution
            of parameters
            is assumed to be uniform and the maximum and minimum bounds for each
            parameter are defined by the maximum
            and minimum values of the project samples.
        """
        # TODO: Try correlated poly here
        active_subspace = self._subspace[:, 0 : self.subspace_dimension]
        projected_points = np.dot(self.std_sample_points, active_subspace)
        myparameters = []
        for i in range(0, self.subspace_dimension):
            param = Parameter(
                distribution="uniform",
                lower=np.min(projected_points[:, i]),
                upper=np.max(projected_points[:, i]),
                order=self.polynomial_degree,
            )
            myparameters.append(param)
        mybasis = Basis("total-order")
        subspacepoly = Poly(
            myparameters,
            mybasis,
            method="least-squares",
            sampling_args={
                "sample-points": projected_points,
                "sample-outputs": self.sample_outputs,
            },
        )
        subspacepoly.set_model()
        return subspacepoly

    def get_eigenvalues(self):  # noqa: ANN201
        """Returns the eigenvalues of the dimension reducing subspace. Note: this option
        is.

        currently only valid for method ``active-subspace``.

        Returns:
        -------
        numpy.ndarray
            Array of shape (dimensions,) corresponding to the eigenvalues of the above mentioned covariance matrix.
        """
        if self.method == "active-subspace":
            return self._eigenvalues
        print("Only the active-subspace method yields eigenvalues.")
        return None

    def get_subspace(self):  # noqa: ANN201
        """Returns the dimension reducing subspace.

        Returns:
        -------
        numpy.ndarray
            Array of shape (dimensions, dimensions) where the first
            ``subspace_dimension`` columns
            contain the dimension reducing subspace, while the remaining columns contain its orthogonal complement.
        """
        return self._subspace

    def _get_active_subspace(self, grad_points=None, **kwargs) -> None:  # noqa: ANN001, ANN003, ARG002
        """Private method to compute active subspaces."""
        if grad_points is None:
            X = self.full_space_poly.get_points()  # noqa: N806
        else:
            if hasattr(self, "data_scaler"):
                X = self.data_scaler.transform(grad_points)  # noqa: N806
            else:
                # Either no standardisation, or user provided poly + param scaling
                X = grad_points.copy()  # noqa: N806

        M, d = X.shape  # noqa: N806
        if d != self.sample_points.shape[1]:
            raise ValueError(
                "In _get_active_subspace: dimensions of gradient evaluation points mismatched with input dimension!"
            )

        alpha = 2.0
        num_grad_lb = alpha * self.subspace_dimension * np.log(d)
        if num_grad_lb > M:
            warnings.warn(
                "Number of gradient evaluation points is likely to be insufficient. Consider resampling!",
                UserWarning, stacklevel=2,
            )

        polygrad = self.full_space_poly.get_polyfit_grad(X)
        if hasattr(self, "param_scaler"):
            # Evaluate gradient in transformed coordinate space
            polygrad = self.param_scaler.div[:, np.newaxis] * polygrad
        weights = np.ones((M, 1)) / M
        R = polygrad.transpose() * weights  # noqa: N806
        C = np.dot(polygrad, R)  # noqa: N806

        # Compute eigendecomposition!
        e, W = np.linalg.eigh(C)  # noqa: N806
        idx = e.argsort()[::-1]
        eigs = e[idx]
        eigVecs = W[:, idx]  # noqa: N806
        if hasattr(self, "data_scaler"):
            scale_factors = 2.0 / (self.data_scaler.Xmax - self.data_scaler.Xmin)
            eigVecs = scale_factors[:, np.newaxis] * eigVecs  # noqa: N806
            eigVecs = np.linalg.qr(eigVecs)[0]  # noqa: N806

        self._subspace = eigVecs
        self._eigenvalues = eigs

    def _get_variable_projection(
        self, gamma=0.1, beta=1e-4, tol=1e-7, maxiter=1000, U0=None, verbose=False  # noqa: ANN001, N803
    ) -> None:
        """Private method to obtain an active subspace in inputs design space via variable projection.

        Note: It may help to standardize outputs to zero mean and unit variance

        Parameters
        ----------
        gamma : float, optional
            Step length reduction factor (0,1).
        beta : float, optional
            Armijo tolerance for backtracking line search (0,1).
        tol : float, optional
            Tolerance for convergence, measured in the norm of residual over norm of f.
        maxiter : int, optional
            Maximum number of optimisation iterations.
        U0 : numpy.ndarray, optional
            Initial guess for active subspace.
        verbose : bool, optional
            Set to ``True`` for debug messages.
        """
        # NOTE: How do we know these are the best values of gamma and beta?
        M, m = self.std_sample_points.shape  # noqa: N806
        if U0 is None:
            Z = np.random.randn(m, self.subspace_dimension)  # noqa: N806
            U, _ = np.linalg.qr(Z)  # noqa: N806
        else:
            U = orth(U0)  # noqa: N806
        y = np.dot(self.std_sample_points, U)
        minmax = np.zeros((2, self.subspace_dimension))
        minmax[0, :] = np.amin(y, axis=0)
        minmax[1, :] = np.amax(y, axis=0)
        # Construct the affine transformation
        eta = 2 * np.divide((y - minmax[0, :]), (minmax[1, :] - minmax[0, :])) - 1

        # Construct the Vandermonde matrix step 6

        V, poly_obj = vandermonde(eta, self.polynomial_degree)  # noqa: N806
        V_plus = np.linalg.pinv(V)  # noqa: N806
        coeff = np.dot(V_plus, self.sample_outputs)
        res = self.sample_outputs - np.dot(V, coeff)
        # R = np.linalg.norm(res)
        # TODO: convergence criterion??

        for iteration in range(0, maxiter):
            # Construct the Jacobian step 9
            J = jacobian_vp(  # noqa: N806
                V,
                V_plus,
                U,
                self.sample_outputs,
                poly_obj,
                eta,
                minmax,
                self.std_sample_points,
            )
            # Calculate the gradient of Jacobian (step 10)
            G = np.zeros((m, self.subspace_dimension))  # noqa: N806
            # NOTE: Can be vectorised
            for i in range(0, M):
                G += res[i] * J[i, :, :]  # noqa: N806

            # conduct the SVD for J_vec
            vec_J = np.reshape(J, (M, m * self.subspace_dimension))  # noqa: N806
            Y, S, Z = np.linalg.svd(vec_J, full_matrices=False)  # step 11  # noqa: N806

            # obtain delta
            delta = np.dot(Y[:, : -(self.subspace_dimension**2)].T, res)
            delta = np.dot(np.diag(1 / S[: -(self.subspace_dimension**2)]), delta)
            delta = -np.dot(Z[: -(self.subspace_dimension**2), :].T, delta).reshape(
                U.shape
            )

            # carry out Gauss-Newton step
            vec_delta = delta.flatten()  # step 12

            # vectorize G step 13
            vec_G = G.flatten()  # noqa: N806
            alpha = np.dot(vec_G.T, vec_delta)
            norm_G = np.dot(vec_G.T, vec_G)  # noqa: N806

            # check alpha step 14
            if alpha >= 0:
                delta = -G
                alpha = -norm_G

            # SVD on delta step 17

            Y, S, Z = np.linalg.svd(delta, full_matrices=False)  # noqa: N806

            UZ = np.dot(U, Z.T)  # noqa: N806
            t = 1
            for _iter2 in range(0, 20):
                U_new = np.dot(UZ, np.diag(np.cos(S * t))) + np.dot(  # noqa: N806
                    Y, np.diag(np.sin(S * t))
                )  # step 19
                U_new = orth(U_new)  # noqa: N806
                # Update the values with the new U matrix
                y = np.dot(self.std_sample_points, U_new)
                minmax[0, :] = np.amin(y, axis=0)
                minmax[1, :] = np.amax(y, axis=0)
                eta = (
                    2 * np.divide((y - minmax[0, :]), (minmax[1, :] - minmax[0, :])) - 1
                )

                V_new, poly_obj = vandermonde(eta, self.polynomial_degree)  # noqa: N806
                V_plus_new = np.linalg.pinv(V_new)  # noqa: N806
                coeff_new = np.dot(V_plus_new, self.sample_outputs)
                res_new = self.sample_outputs - np.dot(V_new, coeff_new)
                np.linalg.norm(res_new)

                if (
                    np.linalg.norm(res_new) <= np.linalg.norm(res) + alpha * beta * t
                    or t < 1e-10
                ):  # step 21
                    break
                t = t * gamma
            dist_change = subspace_dist(U, U_new)
            U = U_new  # noqa: N806
            V = V_new  # noqa: N806
            # coeff = coeff_new
            V_plus = V_plus_new  # noqa: N806
            res = res_new
            # R = R_new
            if dist_change < tol:
                if verbose:
                    print("VP finished with %d iterations" % iteration)  # noqa: UP031
                break
        if iteration == maxiter - 1 and verbose:
            print("VP finished with %d iterations" % iteration)  # noqa: UP031
        active_subspace = U
        inactive_subspace = _null_space(active_subspace.T)
        self._subspace = np.hstack([active_subspace, inactive_subspace])

    def get_zonotope_vertices(self, num_samples=10000, max_count=100000):  # noqa: ANN001, ANN201
        """Returns the vertices of the zonotope -- the projection of the high-dimensional space over the computed.

        subspace.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples per iteration to check.
        max_count : int, optional
            Maximum number of iteration.

        Returns:
        -------
        numpy.ndarray
            Array of shape (number of vertices, ``subspace_dimension``).

        Note:
        ----
        This routine has been adapted from Paul Constantine's zonotope_vertices()
        function; see reference below.

        Constantine, P., Howard, R., Glaws, A., Grey, Z., Diaz, P., Fletcher, L., (2016) Python Active-Subspaces Utility Library. Journal of Open Source Software, 1(5), 79. `Paper <http://joss.theoj.org/papers/10.21105/joss.00079>`__.
        """
        m = self._subspace.shape[0]
        n = self.subspace_dimension
        W = self._subspace[:, :n]  # noqa: N806
        if n == 1:
            y0 = np.dot(W.T, np.sign(W))[0]
            if y0 < -y0:
                yl, yu = y0, -y0
                xl, xu = np.sign(W), -np.sign(W)
            else:
                yl, yu = -y0, y0
                xl, xu = -np.sign(W), np.sign(W)
            Y = np.array([yl, yu]).reshape((2, 1))  # noqa: N806
            X = np.vstack((xl.reshape((1, m)), xu.reshape((1, m))))  # noqa: N806
            self.Y = Y
            return Y
        total_vertices = 0
        for i in range(n):
            total_vertices += comb(m - 1, i)
        total_vertices = int(2 * total_vertices)

        Z = np.random.normal(size=(num_samples, n))  # noqa: N806
        X = get_unique_rows(np.sign(np.dot(Z, W.transpose())))  # noqa: N806
        X = get_unique_rows(np.vstack((X, -X)))  # noqa: N806
        N = X.shape[0]  # noqa: N806

        count = 0
        while total_vertices > N:
            Z = np.random.normal(size=(num_samples, n))  # noqa: N806
            X0 = get_unique_rows(np.sign(np.dot(Z, W.transpose())))  # noqa: N806
            X0 = get_unique_rows(np.vstack((X0, -X0)))  # noqa: N806
            X = get_unique_rows(np.vstack((X, X0)))  # noqa: N806
            N = X.shape[0]  # noqa: N806
            count += 1
            if count > max_count:
                break

        num_vertices = X.shape[0]
        if total_vertices > num_vertices:
            print(f"Warning: {num_vertices} of {total_vertices} vertices found.")

        Y = np.dot(X, W)  # noqa: N806
        self.Y = Y.reshape((num_vertices, n))
        return self.Y

    def get_linear_inequalities(self):  # noqa: ANN201
        """Returns the linear inequalities defining the zonotope vertices, i.e., Ax<=b.

        Returns:
        -------
        tuple
            Tuple (A,b), containing the numpy.ndarray's A and b; where A is the matrix for setting the linear inequalities,
            and b is the right-hand-side vector for setting the linear inequalities.
        """
        if self.Y is None:
            self.Y = self.get_zonotope_vertices()
        n = self.Y.shape[1]
        if n == 1:
            A = np.array([[1], [-1]])  # noqa: N806
            b = np.array([[max(self.Y)], [min(self.Y)]])
            return A, b
        convexHull = ConvexHull(self.Y)  # noqa: N806
        A = convexHull.equations[:, :n]  # noqa: N806
        b = -convexHull.equations[:, n]
        return A, b

    def get_samples_constraining_active_coordinates(  # noqa: ANN201
        self, inactive_samples, active_coordinates  # noqa: ANN001
    ):
        """A hit and run type sampling strategy for generating samples at a given coordinate in the active subspace.

        by varying its coordinates along the inactive subspace.

        Parameters
        ----------
        inactive_samples : int
            The number of inactive samples required.
        active_coordiantes : numpy.ndarray
            The active subspace coordinates.

        Returns:
        -------
        numpy.ndarray
            Array containing the full-space coordinates.

        Note:
        ----
        This routine has been adapted from Paul Constantine's hit_and_run() function; see reference below.

        Constantine, P., Howard, R., Glaws, A., Grey, Z., Diaz, P., Fletcher, L., (2016) Python Active-Subspaces Utility Library. Journal of Open Source Software, 1(5), 79. `Paper <http://joss.theoj.org/papers/10.21105/joss.00079>`__.

        """
        y = active_coordinates
        N = inactive_samples  # noqa: N806
        W1 = self._subspace[:, : self.subspace_dimension]  # noqa: N806
        W2 = self._subspace[:, self.subspace_dimension :]  # noqa: N806
        m, n = W1.shape

        s = np.dot(W1, y).reshape((m, 1))
        normW2 = np.sqrt(np.sum(np.power(W2, 2), axis=1)).reshape((m, 1))  # noqa: N806
        A = np.hstack((np.vstack((W2, -W2.copy())), np.vstack((normW2, normW2.copy()))))  # noqa: N806
        b = np.vstack((1 - s, 1 + s)).reshape((2 * m, 1))
        c = np.zeros((m - n + 1, 1))
        c[-1] = -1.0
        # print()

        zc = linear_program_ineq(c, -A, -b)
        z0 = zc[:-1].reshape((m - n, 1))

        # define the polytope A >= b
        s = np.dot(W1, y).reshape((m, 1))
        A = np.vstack((W2, -W2))  # noqa: N806
        b = np.vstack((-1 - s, -1 + s)).reshape((2 * m, 1))

        # tolerance
        ztol = 1e-6
        eps0 = ztol / 4.0

        Z = np.zeros((N, m - n))  # noqa: N806
        for i in range(N):
            # random direction
            bad_dir = True
            count, maxcount = 0, 50
            while bad_dir:
                d = np.random.normal(size=(m - n, 1))
                bad_dir = np.any(np.dot(A, z0 + eps0 * d) <= b)
                count += 1
                if count >= maxcount:
                    Z[i:, :] = np.tile(z0, (1, N - i)).transpose()
                    yz = np.vstack([np.repeat(y[:, np.newaxis], N, axis=1), Z.T])
                    return np.dot(self._subspace, yz).T

            # find constraints that impose lower and upper bounds on eps
            f, g = b - np.dot(A, z0), np.dot(A, d)

            # find an upper bound on the step
            min_ind = np.logical_and(g <= 0, f < -np.sqrt(np.finfo(np.float).eps))
            eps_max = np.amin(f[min_ind] / g[min_ind])

            # find a lower bound on the step
            max_ind = np.logical_and(g > 0, f < -np.sqrt(np.finfo(np.float).eps))
            eps_min = np.amax(f[max_ind] / g[max_ind])

            # randomly sample eps
            eps1 = np.random.uniform(eps_min, eps_max)

            # take a step along d
            z1 = z0 + eps1 * d
            Z[i, :] = z1.reshape((m - n,))

            # update temp var
            z0 = z1.copy()

        yz = np.vstack([np.repeat(y[:, np.newaxis], N, axis=1), Z.T])
        return np.dot(self._subspace, yz).T

    # ""def plot_sufficient_summary(self, ax=None, X_test=None, y_test=None, show=True, poly=True, uncertainty=False, legend=False, scatter_kwargs={}, plot_kwargs={}):
    #     """ Generates a sufficient summary plot for 1D or 2D polynomial ridge approximations.
    #     See :meth:`~equadratures.plot.plot_sufficient_summary` for full description. """
    #     return plot.plot_sufficient_summary(self, ax, X_test, y_test, show, poly, uncertainty, legend, scatter_kwargs, plot_kwargs)

    # def plot_2D_contour_zonotope(self, mysubspace, minmax=[- 3.5, 3.5], grid_pts=180, show=True, ax=None):
    #     """ Generates a 2D contour plot of the polynomial ridge approximation.
    #     See :meth:`~equadratures.plot.plot_2D_contour_zonotope` for full description. """
    #     return plot.plot_2D_contour_zonotope(self,minmax,grid_pts,show,ax)

    # def plot_samples_from_second_subspace_over_first(self, mysubspace_2, axs=None, no_of_samples=500, minmax=[- 3.5, 3.5], grid_pts=180, show=True):
    #     """
    #     Generates a zonotope plot where samples from the second subspace are projected over the first.
    #     See :meth:`~equadratures.plot.plot_samples_from_second_subspace_over_first` for full description.
    #     """
    #     return plot.plot_samples_from_second_subspace_over_first(self,mysubspace_2, axs, no_of_samples, minmax, grid_pts, show)""


def vandermonde(eta, p):  # noqa: ANN001, ANN201, D103
    # TODO: Try using a "correlated" basis here?
    _, n = eta.shape
    listing = []
    for _i in range(0, n):
        listing.append(p)
    Object = Basis("total-order", listing)  # noqa: N806
    # Establish n Parameter objects
    params = []
    P = Parameter(order=p, lower=-1, upper=1, distribution="uniform")  # noqa: N806
    for _i in range(0, n):
        params.append(P)
    # Use the params list to establish the Poly object
    poly_obj = Poly(params, Object, method="least-squares")
    V = poly_obj.get_poly(eta)  # noqa: N806
    V = V.T  # noqa: N806
    return V, poly_obj


def vector_AS(  # noqa: ANN201, D103, N802
    list_of_polys,  # noqa: ANN001
    R=None,  # noqa: ANN001, N803
    alpha=None,  # noqa: ANN001
    k=None,  # noqa: ANN001
    samples=None,  # noqa: ANN001
    bootstrap=False,  # noqa: ANN001
    bs_trials=50,  # noqa: ANN001
    J=None,  # noqa: ANN001, N803
    save_path=None,  # noqa: ANN001
):
    # Find AS directions to vector val func
    # analogous to computeActiveSubspace
    # Since we are dealing with *one* vector val func we should have just one input space
    # Take the first of the polys.
    poly = list_of_polys[0]
    if samples is None:
        d = poly.dimensions
        if alpha is None:
            alpha = 4
        if k is None or k > d:
            k = d
        M = int(alpha * k * np.log(d))  # noqa: N806
        X = np.zeros((M, d))  # noqa: N806
        for j in range(0, d):
            X[:, j] = np.reshape(poly.parameters[j].getSamples(M), M)
    else:
        X = samples  # noqa: N806
        M, d = X.shape  # noqa: N806
    n = len(list_of_polys)  # number of outputs
    if R is None:
        R = np.eye(n)  # noqa: N806
    elif len(R.shape) == 1:
        R = np.diag(R)  # noqa: N806
    if J is None:
        J = jacobian_vec(list_of_polys, X)  # noqa: N806
        if save_path is not None:
            np.save(save_path, J)

    J_new = np.matmul(sqrtm(R), np.transpose(J, [2, 0, 1]))  # noqa: N806
    JtJ = np.matmul(np.transpose(J_new, [0, 2, 1]), J_new)  # noqa: N806
    H = np.mean(JtJ, axis=0)  # noqa: N806

    # Compute P_r by solving generalized eigenvalue problem...
    # Assume sigma = identity for now
    e, W = np.linalg.eigh(H)  # noqa: N806
    eigs = np.flipud(e)
    eigVecs = np.fliplr(W)  # noqa: N806
    if bootstrap:
        all_bs_eigs = np.zeros((bs_trials, d))
        all_bs_W = []  # noqa: N806
        for t in range(bs_trials):
            print("Starting bootstrap trial %d" % t)  # noqa: UP031
            bs_samples = X[np.random.randint(0, M, size=M), :]
            J_bs = jacobian_vec(list_of_polys, bs_samples)  # noqa: N806
            J_new_bs = np.matmul(sqrtm(R), np.transpose(J_bs, [2, 0, 1]))  # noqa: N806
            JtJ_bs = np.matmul(np.transpose(J_new_bs, [0, 2, 1]), J_new_bs)  # noqa: N806
            H_bs = np.mean(JtJ_bs, axis=0)  # noqa: N806

            # Compute P_r by solving generalized eigenvalue problem...
            # Assume sigma = identity for now
            e_bs, W_bs = np.linalg.eigh(H_bs)  # noqa: N806
            all_bs_eigs[t, :] = np.flipud(e_bs)
            eigVecs_bs = np.fliplr(W_bs)  # noqa: N806
            all_bs_W.append(eigVecs_bs)

        eigs_bs_lower = np.min(all_bs_eigs, axis=0)
        eigs_bs_upper = np.max(all_bs_eigs, axis=0)
        return eigs, eigVecs, eigs_bs_lower, eigs_bs_upper, all_bs_W
    return eigs, eigVecs


def jacobian_vp(V, V_plus, U, f, Polybasis, eta, minmax, X):  # noqa: ANN001, ANN201, D103, N803
    M, N = V.shape  # noqa: N806
    m, n = U.shape
    Gradient = Polybasis.get_poly_grad(eta)  # noqa: N806
    sub = (minmax[1, :] - minmax[0, :]).T
    vectord = np.reshape(2.0 / sub, (n, 1))
    # Initialize the tensor
    J = np.zeros((M, m, n))  # noqa: N806
    # Obtain the derivative of this tensor
    dV = np.zeros((m, n, M, N))  # noqa: N806
    for l in range(0, n):  # noqa: E741
        for j in range(0, N):
            current = Gradient[l].T
            if n == 1:
                current = Gradient.T
            dV[:, l, :, j] = np.asscalar(vectord[l]) * (X.T * current[:, j])

    # Get the P matrix
    P = np.identity(M) - np.matmul(V, V_plus)  # noqa: N806
    V_minus = scipy.linalg.pinv(V)  # noqa: N806

    # Calculate entries for the tensor
    for j in range(0, m):
        for k in range(0, n):
            temp1 = np.linalg.multi_dot([P, dV[j, k, :, :], V_minus])
            J[:, j, k] = (-np.matmul((temp1 + temp1.T), f)).reshape((M,))  # Eqn 15

    return J


def jacobian_vec(list_of_poly, X):  # noqa: ANN001, ANN201, D103, N803
    m = len(list_of_poly)
    [N, d] = X.shape  # noqa: N806
    J = np.zeros((m, d, N))  # noqa: N806
    for p in range(len(list_of_poly)):
        J[p, :, :] = list_of_poly[p].get_polyfit_grad(X)
    return J


def subspace_dist(U, V):  # noqa: ANN001, ANN201, D103, N803
    if len(U.shape) == 1:
        return np.linalg.norm(np.outer(U, U) - np.outer(V, V), ord=2)
    return np.linalg.norm(np.dot(U, U.T) - np.dot(V, V.T), ord=2)


def linear_program_ineq(c, A, b):  # noqa: ANN001, ANN201, D103, N803
    c = c.reshape((c.size,))
    b = b.reshape((b.size,))

    # make unbounded bounds
    bounds = []
    for _i in range(c.size):
        bounds.append((None, None))

    A_ub, b_ub = -A, -b  # noqa: N806
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        options={"disp": False},
        method="simplex",
    )
    if res.success:
        return res.x.reshape((c.size, 1))
    np.savez(
        f"bad_scipy_lp_ineq_{np.random.randint(int(1e9)):010d}", c=c, A=A, b=b, res=res
    )
    raise Exception("Scipy did not solve the LP. Blame Scipy.")


def get_unique_rows(X0):  # noqa: ANN001, ANN201, D103, N803
    X1 = X0.view(np.dtype((np.void, X0.dtype.itemsize * X0.shape[1])))  # noqa: N806
    return np.unique(X1).view(X0.dtype).reshape(-1, X0.shape[1])


def _null_space(A, rcond=None):  # noqa: ANN001, ANN202, N803
    """Null space method adapted from scipy."""
    u, s, vh = scipy.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]  # noqa: N806
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    return vh[num:, :].T.conj()
