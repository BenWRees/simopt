import numpy as np  # noqa: D100

import equadratures.plot as plot
from equadratures import Weight
from equadratures.basis import Basis
from equadratures.parameter import Parameter
from equadratures.poly import Poly


class PolyTree:
    """Definition of a polynomial tree object.

    Parameters
    ----------
    splitting_criterion : str, optional
            The type of splitting_criterion to use in the fit function. Options include
            ``model_aware`` which fits polynomials for each candidate split,
            ``model_agnostic`` which uses a standard deviation based model-agnostic
            split criterion [1], and ``loss_gradient`` which uses a gradient based
            splitting criterion similar to that in [2].
    max_depth : int, optional
            The maximum depth which the tree will grow to.
    min_samples_leaf : int, optional
            The minimum number of samples per leaf node.
    order : int, optional
            The order of the generated orthogonal polynomials.
    basis : str, optional
            The type of index set used for the basis. Options include: ``univariate``,
            ``total-order``, ``tensor-grid``, ``sparse-grid`` and ``hyperbolic-basis``.
    search : str, optional
            The method of search to be used. Options are ``grid`` or ``exhaustive``.
    samples : int, optional
            The interval between splits if ``grid`` search is chosen.
    verbose : bool, optional
            For debugging.
    all_data : bool, optional
            Store data at all nodes in :class:`~PolyTree` (instead of only leaf nodes).
    split_dims : list, optional
            List of dimensions along which to make splits.
    k : float, optional
            The smoothing parameter. Range from 0.0 to 1.0, with 0 giving no smoothing,
            and 1 giving maximum smoothing.
    distribution : str, optional
            The type of input parameter distributions. Either ``uniform`` or ``data``.

    Example:
    -------
    >>> tree = polytree.PolyTree()
    >>> X = np.loadtxt('inputs.txt')
    >>> Xtest = np.loadtxt('inputs_test.txt')
    >>> y = np.loadtxt('outputs.txt')
    >>> tree.fit(X,y)
    >>> y_test = tree.predict(X_test)

    References:
    ----------
    1. Wang, Y., Witten, I. H., (1997) Inducing Model Trees for Continuous Classes. In
    Proc. of the 9th European Conf. on Machine Learning Poster Papers. 128-137. `Paper
    <https://researchcommons.waikato.ac.nz/handle/10289/1183>`__
    2. Broelemann, K., Kasneci, G., (2019) A Gradient-Based Split Criterion for Highly
    Accurate and Transparent Model Trees. In Int. Joint Conf. on Artificial Intelligence
    (IJCAI). 2030-2037. `Paper <https://www.ijcai.org/Proceedings/2019/0281.pdf>`__
    3. Chan, T. F., Golub, G. H., LeVeque, R. J., (1983) Algorithms for computing the
    sample variance: Analysis and recommendations. The American Statistician. 37(3):
    242-247. `Paper  # noqa: RUF002
    <https://www.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115>`__
    """

    def __init__(  # noqa: D107
        self,
        splitting_criterion="model_aware",  # noqa: ANN001
        max_depth=5,  # noqa: ANN001
        min_samples_leaf=None,  # noqa: ANN001
        order=1,  # noqa: ANN001
        basis="total-order",  # noqa: ANN001
        search="exhaustive",  # noqa: ANN001
        samples=50,  # noqa: ANN001
        verbose=False,  # noqa: ANN001
        poly_method="least-squares",  # noqa: ANN001
        poly_solver_args=None,  # noqa: ANN001
        all_data=False,  # noqa: ANN001
        split_dims=None,  # noqa: ANN001
        k=0.05,  # noqa: ANN001
        distribution="uniform",  # noqa: ANN001
    ) -> None:
        if poly_solver_args is None:
            poly_solver_args = {}
        self.splitting_criterion = splitting_criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.order = order
        self.basis = basis
        self.tree = None
        self.search = search
        self.samples = samples
        self.verbose = verbose
        self.cardinality = None
        self.poly_method = poly_method
        self.poly_solver_args = poly_solver_args
        self.actual_max_depth = 0
        self.all_data = all_data
        self.k = k
        self.distribution = distribution
        if split_dims is not None:
            split_dims = (
                [split_dims] if not isinstance(split_dims, list) else split_dims
            )
            assert all(isinstance(dim, int) for dim in split_dims), (
                "split_dims should be a list if ints"
            )
        self.split_dims = split_dims

        assert max_depth >= 0, "max_depth must be >= 0"
        assert order > 0, "order must be a postive integer"
        assert samples > 0, "samples must be a postive integer"
        assert k > 0, "k must be a positive number"

    def get_splits(self):  # noqa: ANN201
        """Returns all of the data splits made.

        Returns:
        -------
        list
            A list of splits made in the format of a nested list: [[split, dimension],
            ...]
        """

        def _search_tree(node, splits):  # noqa: ANN001, ANN202
            if node["children"]["left"] is not None:
                if [node["threshold"], node["j_feature"]] not in splits:
                    splits.append([node["threshold"], node["j_feature"]])
                splits = _search_tree(node["children"]["left"], splits)

            if node["children"]["right"] is not None:
                if [node["threshold"], node["j_feature"]] not in splits:
                    splits.append([node["threshold"], node["j_feature"]])
                splits = _search_tree(node["children"]["right"], splits)

            return splits

        return _search_tree(self.tree, [])

    def _split_data(self, j_feature, threshold, X, y):  # noqa: ANN001, ANN202, N803
        idx_left = np.where(X[:, j_feature] <= threshold)[0]
        idx_right = np.delete(np.arange(0, len(X)), idx_left)
        assert len(idx_left) + len(idx_right) == len(X)
        return (X[idx_left], y[idx_left]), (X[idx_right], y[idx_right])

    def get_polys(self):  # noqa: ANN201
        """Returns all of the polynomials fitted at each node in the tree.

        Returns:
        -------
        list
            A list of Poly objects.
        """

        def _search_tree(node, polys):  # noqa: ANN001, ANN202
            if node["children"]["left"] is None and node["children"]["right"] is None:
                polys.append(node["poly"])

            if node["children"]["left"] is not None:
                polys = _search_tree(node["children"]["left"], polys)

            if node["children"]["right"] is not None:
                polys = _search_tree(node["children"]["right"], polys)

            return polys

        return _search_tree(self.tree, [])

    def fit(self, X, y) -> None:  # noqa: ANN001, N803
        """Fits the PolyTree to the provided data.

        Parameters
        ----------
        X : numpy.ndarray
                Training input data
        y : numpy.ndarray
                Training output data
        """

        def _build_tree():  # noqa: ANN202
            global index_node_global

            def _splitter(node):  # noqa: ANN001, ANN202
                # Extract data
                X, y = node["data"]  # noqa: N806
                depth = node["depth"]
                N, d = X.shape  # noqa: N806

                # Dimensions to split along
                if self.split_dims is None:
                    self.split_dims = range(d)

                # Find feature splits that might improve loss
                did_split = False
                if self.splitting_criterion == "model_aware":
                    loss_best = node["loss"]
                elif (
                    self.splitting_criterion == "model_agnostic"
                    or self.splitting_criterion == "loss_gradient"
                ):
                    loss_best = np.inf
                else:
                    raise Exception("invalid splitting_criterion")
                data_best = None
                polys_best = None
                j_feature_best = None
                threshold_best = None

                if self.verbose:
                    polys_fit = 0

                # Perform threshold split search only if node has not hit max depth
                if (depth >= 0) and (depth < self.max_depth):
                    if self.splitting_criterion != "loss_gradient":
                        for j_feature in range(d):
                            if self.search == "exhaustive":
                                threshold_search = X[:, j_feature]
                            elif self.search == "grid":
                                samples = N if self.samples > N else self.samples
                                threshold_search = np.linspace(
                                    np.min(X[:, j_feature]),
                                    np.max(X[:, j_feature]),
                                    num=samples,
                                )
                            else:
                                raise Exception(
                                    "Incorrect search type! Must be 'exhaustive' or 'grid'"
                                )

                            # Perform threshold split search on j_feature
                            for threshold in np.unique(np.sort(threshold_search)):
                                # Split data based on threshold
                                (X_left, y_left), (X_right, y_right) = self._split_data(  # noqa: N806
                                    j_feature, threshold, X, y
                                )
                                # print(j_feature, threshold, X_left, X_right)
                                N_left, N_right = len(X_left), len(X_right)  # noqa: N806

                                # Do not attempt to split if split conditions not satisfied
                                if not (
                                    N_left >= self.min_samples_leaf
                                    and N_right >= self.min_samples_leaf
                                ):
                                    continue

                                # Compute weight loss function
                                if self.splitting_criterion == "model_aware":
                                    loss_left, poly_left = _fit_poly(X_left, y_left)
                                    loss_right, poly_right = _fit_poly(X_right, y_right)

                                    loss_split = (
                                        N_left * loss_left + N_right * loss_right
                                    ) / N

                                    if self.verbose:
                                        polys_fit += 2

                                elif self.splitting_criterion == "model_agnostic":
                                    loss_split = (
                                        np.std(y)
                                        - (
                                            N_left * np.std(y_left)
                                            + N_right * np.std(y_right)
                                        )
                                        / N
                                    )

                                # Update best parameters if loss is lower
                                if loss_split < loss_best:
                                    did_split = True
                                    loss_best = loss_split
                                    if self.splitting_criterion == "model_aware":
                                        polys_best = [poly_left, poly_right]
                                    data_best = [(X_left, y_left), (X_right, y_right)]
                                    j_feature_best = j_feature
                                    threshold_best = threshold

                    # Gradient based splitting criterion from ref. [2]
                    else:
                        # Fit a single poly to parent node
                        _loss, poly = _fit_poly(X, y)

                        # Now run the splitting algo using gradients from this poly
                        did_split, j_feature_best, threshold_best = (
                            self._find_split_from_grad(poly, X, y.reshape(-1, 1))
                        )

                # If model_agnostic or gradient based, fit poly's to children now we have split
                if self.splitting_criterion != "model_aware" and did_split:
                    (X_left, y_left), (X_right, y_right) = self._split_data(  # noqa: N806
                        j_feature_best, threshold_best, X, y
                    )
                    loss_left, poly_left = _fit_poly(X_left, y_left)
                    loss_right, poly_right = _fit_poly(X_right, y_right)
                    N_left, N_right = len(X_left), len(X_right)  # noqa: N806
                    loss_best = (N_left * loss_left + N_right * loss_right) / N
                    polys_best = [poly_left, poly_right]
                    if self.splitting_criterion == "loss_gradient":
                        data_best = [(X_left, y_left), (X_right, y_right)]

                    if self.verbose:
                        polys_fit += 2

                if self.verbose and did_split:
                    print(
                        f"Node (X.shape = {X.shape}) fitted with {polys_fit} polynomials generated"
                    )
                elif self.verbose:
                    print(
                        f"Node (X.shape = {X.shape}) failed to fit after {polys_fit} polynomials generated"
                    )

                if did_split and depth > self.actual_max_depth:
                    self.actual_max_depth = depth

                # Return the best result
                return {
                    "did_split": did_split,
                    "loss": loss_best,
                    "polys": polys_best,
                    "data": data_best,
                    "j_feature": j_feature_best,
                    "threshold": threshold_best,
                    "N": N,
                }

            def _fit_poly(X, y):  # noqa: ANN001, ANN202, N803
                #                                try:

                N, d = X.shape  # noqa: N806
                myParameters = []  # noqa: N806

                for dimension in range(d):
                    values = X[:, dimension]
                    values_min = np.amin(values)
                    values_max = np.amax(values)

                    if (values_min - values_max) ** 2 < 0.01:
                        values_min -= 0.01
                        values_max += 0.01
                        myParameters.append(
                            Parameter(
                                distribution="Uniform",
                                lower=values_min,
                                upper=values_max,
                                order=self.order,
                            )
                        )
                    else:
                        if self.distribution == "uniform":
                            myParameters.append(
                                Parameter(
                                    distribution="Uniform",
                                    lower=values_min,
                                    upper=values_max,
                                    order=self.order,
                                )
                            )
                        elif self.distribution == "data":
                            input_dist = Weight(
                                values, support=[values_min, values_max], pdf=False
                            )
                            myParameters.append(
                                Parameter(
                                    distribution="data",
                                    weight_function=input_dist,
                                    order=self.order,
                                )
                            )

                if self.basis == "hyperbolic-basis":
                    myBasis = Basis(  # noqa: N806
                        self.basis, orders=[self.order for _ in range(d)], q=0.5
                    )
                else:
                    myBasis = Basis(self.basis, orders=[self.order for _ in range(d)])  # noqa: N806

                container["index_node_global"] += 1
                poly = Poly(
                    myParameters,
                    myBasis,
                    method=self.poly_method,
                    sampling_args={"sample-points": X, "sample-outputs": y},
                    solver_args=self.poly_solver_args,
                )
                poly.set_model()

                mse = np.linalg.norm(y - poly.get_polyfit(X).reshape(-1)) ** 2 / N
                #                                except Exception as e:
                #                                        print("Warning fitting of Poly failed:", e)
                #                                        print(d, values_min, values_max)
                #                                        mse, poly = np.inf, None

                return mse, poly

            def _create_node(X, y, depth, container):  # noqa: ANN001, ANN202, N803
                poly_loss, poly = _fit_poly(X, y)

                node = {
                    "name": "node",
                    "index": container["index_node_global"],
                    "loss": poly_loss,
                    "poly": poly,
                    "data": (X, y),
                    "n_samples": len(X),
                    "j_feature": None,
                    "threshold": None,
                    "children": {"left": None, "right": None},
                    "depth": depth,
                    "flag": False,
                }
                container["index_node_global"] += 1

                return node

            def _split_traverse_node(node, container) -> None:  # noqa: ANN001
                result = _splitter(node)
                if not result["did_split"]:
                    return

                node["j_feature"] = result["j_feature"]
                node["threshold"] = result["threshold"]

                if not self.all_data:
                    del node["data"]

                (X_left, y_left), (X_right, y_right) = result["data"]  # noqa: N806
                poly_left, poly_right = result["polys"]

                node["children"]["left"] = _create_node(
                    X_left, y_left, node["depth"] + 1, container
                )
                node["children"]["right"] = _create_node(
                    X_right, y_right, node["depth"] + 1, container
                )
                node["children"]["left"]["poly"] = poly_left
                node["children"]["right"]["poly"] = poly_right

                # Split nodes
                _split_traverse_node(node["children"]["left"], container)
                _split_traverse_node(node["children"]["right"], container)

            container = {"index_node_global": 0}
            root = _create_node(X, y, 0, container)
            _split_traverse_node(root, container)

            return root

        _N, d = X.shape  # noqa: N806
        if self.basis == "hyperbolic-basis":
            self.cardinality = Basis(
                self.basis, orders=[self.order for _ in range(d)], q=0.5
            ).get_cardinality()
        else:
            self.cardinality = Basis(
                self.basis, orders=[self.order for _ in range(d)]
            ).get_cardinality()
        if self.min_samples_leaf is None or self.min_samples_leaf == self.cardinality:
            self.min_samples_leaf = int(np.ceil(self.cardinality * 1.25))
        elif self.cardinality > self.min_samples_leaf:
            print(
                f"WARNING: Basis cardinality ({self.cardinality}) greater than the minimum samples per leaf ({self.min_samples_leaf}). This may cause reduced performance."
            )

        self.k *= self.min_samples_leaf

        self.tree = _build_tree()

    def prune(self, X, y, tol=0.0, percent=False) -> None:  # noqa: ANN001, N803
        """Prunes the tree that you have fitted.

        Parameters
        ----------
        X : numpy.ndarray
                Training input data
        y : numpy.ndarray
                Training output data
        tol : float, optional
                Pruning tolerance (%). Prune nodes if they only improve loss by less
                than this tolerance.
        percent : bool, optional
                If true, tol is taken as a percentage of the parent node's error.
                Otherwise, tol is taken to be an absolute value.
        """
        if percent:
            tol /= 100.0

        def pruner(node, X_subset, y_subset):  # noqa: ANN001, ANN202, N803
            if X_subset.shape[0] < 1:
                node["test_loss"] = 0
                node["n_samples"] = 0
                return node

            node["test_loss"] = (
                np.linalg.norm(
                    y_subset - node["poly"].get_polyfit(X_subset).reshape(-1)
                )
                ** 2
                / X_subset.shape[0]
            )

            is_left = node["children"]["left"] is not None
            is_right = node["children"]["right"] is not None

            if is_left and is_right:
                (X_left, y_left), (X_right, y_right) = self._split_data(  # noqa: N806
                    node["j_feature"], node["threshold"], X_subset, y_subset
                )

                node["children"]["left"] = pruner(
                    node["children"]["left"], X_left, y_left
                )
                node["children"]["right"] = pruner(
                    node["children"]["right"], X_right, y_right
                )

                lower_loss = (
                    node["children"]["left"]["test_loss"]
                    * node["children"]["left"]["n_samples"]
                    + node["children"]["right"]["test_loss"]
                    * node["children"]["right"]["n_samples"]
                ) / (
                    node["children"]["left"]["n_samples"]
                    + node["children"]["right"]["n_samples"]
                )

                node["lower_loss"] = lower_loss

                loss_eps = tol * node["test_loss"] if percent else tol
                print(lower_loss + loss_eps, node["test_loss"])
                if lower_loss + loss_eps > node["test_loss"]:
                    if self.verbose:
                        print(
                            "prune",
                            lower_loss,
                            node["test_loss"],
                            node["children"]["left"]["test_loss"],
                            node["children"]["left"]["n_samples"],
                            node["children"]["right"]["test_loss"],
                            node["children"]["right"]["n_samples"],
                        )
                    node["children"]["left"] = None
                    node["children"]["right"] = None

            return node

        assert self.tree is not None, "Run fit() before prune()"
        (X_left, y_left), (X_right, y_right) = self._split_data(  # noqa: N806
            self.tree["j_feature"], self.tree["threshold"], X, y
        )

        self.tree["children"]["left"] = pruner(
            self.tree["children"]["left"], X_left, y_left
        )
        self.tree["children"]["right"] = pruner(
            self.tree["children"]["right"], X_right, y_right
        )

    def predict(self, X):  # noqa: ANN001, ANN201, N803
        """Evaluates the the polynomial tree approximation of the data.

        Parameters
        ----------
        X : numpy.ndarray
            An ndarray with shape (number_of_observations, dimensions) at which the tree
            fit must be evaluated at.

        Returns:
        -------
        numpy.ndarray
            Array with shape (1, number_of_observations) corresponding to the polynomial
            approximations of the tree.
        """

        def _predict(node, indexes) -> None:  # noqa: ANN001
            y_pred[indexes, node["depth"], 0] = (
                node["poly"].get_polyfit(X[indexes]).reshape(-1)
            )
            y_pred[indexes, node["depth"], 1] = np.full(
                fill_value=node["n_samples"], shape=len(indexes)
            )

            no_children = (
                node["children"]["left"] is None and node["children"]["right"] is None
            )
            if no_children:
                return

            idx_left = np.where(X[indexes, node["j_feature"]] <= node["threshold"])[0]
            idx_right = np.where(X[indexes, node["j_feature"]] > node["threshold"])[0]

            _predict(node["children"]["left"], indexes[idx_left])
            _predict(node["children"]["right"], indexes[idx_right])

        assert self.tree is not None
        y_pred = np.empty(shape=(X.shape[0], self.actual_max_depth + 2, 2)) * np.nan

        _predict(self.tree, np.arange(0, X.shape[0]))

        smoothed_y_pred = np.zeros(shape=(X.shape[0]))

        for y in range(0, X.shape[0]):
            i = self.actual_max_depth + 1

            while np.isnan(y_pred[y][i][0]) and i > 0:
                i -= 1

            smoothed_y = y_pred[y][i][0]

            # print(y_pred[i])
            while i > 0:
                n_i = y_pred[y][i][1]
                if n_i == 0:
                    break
                # print(smoothed_y)
                smoothed_y = (smoothed_y * n_i + y_pred[y][i][0] * self.k) / (
                    self.k + n_i
                )
                i -= 1

            # print("\n")
            smoothed_y_pred[y] = smoothed_y

        return smoothed_y_pred

    def apply(self, X):  # noqa: ANN001, ANN201, N803
        """Returns the leaf node index for each observation in the data.

        Parameters
        ----------
        X : numpy.ndarray
            Array with shape (number_of_observations, dimensions) at which the tree fit
            must be evaluated at.

        Returns:
        -------
        numpy.ndarray
            A numpy.ndarray of shape (number_of_observations,1) corresponding to the
            node indices for each observation in X.
        """

        def _apply(node, indexes) -> None:  # noqa: ANN001
            no_children = (
                node["children"]["left"] is None and node["children"]["right"] is None
            )
            if no_children:
                inode[indexes] = node["index"]
                return

            idx_left = np.where(X[indexes, node["j_feature"]] <= node["threshold"])[0]
            idx_right = np.where(X[indexes, node["j_feature"]] > node["threshold"])[0]
            _apply(node["children"]["left"], indexes[idx_left])
            _apply(node["children"]["right"], indexes[idx_right])

        if X.ndim == 1:
            X = X.reshape(1, -1)  # noqa: N806
        inode = np.zeros(shape=X.shape[0], dtype=int)
        _apply(self.tree, np.arange(0, X.shape[0]))
        return inode

    def get_leaves(self):  # noqa: ANN201
        """Returns the node indices for all leaf nodes.

        Returns:
        -------
        list
            Contains the node indices of all leaf nodes.
        """

        def _recurse(node, leaf_list) -> None:  # noqa: ANN001
            no_children = (
                node["children"]["left"] is None and node["children"]["right"] is None
            )
            if no_children:
                leaf_list.append(node["index"])
                return
            _recurse(node["children"]["left"], leaf_list)
            _recurse(node["children"]["right"], leaf_list)

        leaf_list = []
        _recurse(self.tree, leaf_list)
        return leaf_list

    def get_mean_and_variance(self):  # noqa: ANN201
        """Computes the mean and variance of the polynomial tree model.

        Returns:
        -------
        tuple
            Tuple (mean,variance) containing two floats; the approximated mean and
            variance from the fitted PolyTree.
        """
        # Get volume of polytree domain
        root_poly = self.tree["poly"]
        root_vol = self._calc_domain_vol(root_poly)

        # Get leaf nodes
        leaves = self.get_leaves()

        # Summation over all leaf nodes in the tree
        mean = 0.0
        var = 0.0
        for leaf in leaves:
            leaf_poly = self.get_node(leaf)["poly"]
            leaf_vol = self._calc_domain_vol(leaf_poly)
            coeffs = leaf_poly.coefficients

            # Compute mean
            mean += (leaf_vol / root_vol) * float(coeffs[0])

            # Compute variance
            tmp = 0.0
            for i in range(0, len(coeffs)):
                tmp += float(coeffs[i] ** 2)
            var += (leaf_vol / root_vol) * tmp
        var -= mean**2

        return mean, var

    def get_graphviz(self, X=None, feature_names=None, file_name=None):  # noqa: ANN001, ANN201, N803
        """Generates a graphviz visualisation of the PolyTree.

        Parameters
        ----------
        X : numpy.ndarray, optional
                An ndarray with shape (dimensions) containing an input vector for a
                given sample, to highlight in the tree.
        feature_names : list, optional
                A list of the names of the features used in the training data.
        filename : str, optional
                Filename to write graphviz data to. If ``None`` (default) then rendered
                in-place, if ``'source'``, the raw graphviz string is returned.

        """
        from graphviz import Digraph

        g = Digraph("g", node_attr={"shape": "record", "height": ".1"})

        if feature_names is None:
            dim = self.tree["poly"].dimensions
            feature_names = ["x_%d" % i for i in range(dim)]  # noqa: UP031

        def _build_graphviz_recurse(
            node,  # noqa: ANN001
            parent_node_index=0,  # noqa: ANN001
            parent_depth=0,  # noqa: ANN001
            edge_label="",  # noqa: ANN001
            labelangle=0,  # noqa: ANN001
        ) -> None:
            # Empty node
            if node is None:
                return

            # Create node
            node_index = node["index"]
            if node["children"]["left"] is None and node["children"]["right"] is None:
                threshold_str = ""
                leaf = True
            else:
                threshold_str = "{} <= {:.3f}\\n".format(
                    feature_names[node["j_feature"]], node["threshold"]
                )
                leaf = False

            if "lower_loss" in node:
                label_str = "node {} \\n {} n_samples = {}\\n loss = {:.6f}\\n lower_loss = {}".format(
                    node_index,
                    threshold_str,
                    node["n_samples"],
                    node["test_loss"],
                    node["lower_loss"],
                )
            elif "test_loss" in node:
                label_str = "node {} \\n {} n_samples = {}\\n loss = {:.6f}".format(
                    node_index, threshold_str, node["n_samples"], node["test_loss"]
                )
            else:
                label_str = "node {} \\n {} n_samples = {}\\n loss = {:.6f}".format(
                    node_index, threshold_str, node["n_samples"], node["loss"]
                )
            # Create node
            if leaf:
                nodeshape = "rectangle"
                style = ["rounded"]
                fillcolor = "#E4fEE4"
            else:
                nodeshape = "rectangle"
                style = ["filled"]
                fillcolor = "#EBFAFF"
            if node["flag"]:
                style.append("bold")
            bordercolor = "black"
            fontcolor = "black"
            g.attr("node", label=label_str, shape=nodeshape)
            g.node(
                f"{node_index}",
                color=bordercolor,
                style=", ".join(style),
                fillcolor=fillcolor,
                fontcolor=fontcolor,
            )

            # Create edge
            if parent_depth > 0:
                if node["flag"]:
                    edgecolor = "orange"
                    style = "bold"
                else:
                    edgecolor = "black"
                    style = "solid"
                if parent_depth > 1:
                    edge_label = ""  # Only label True/False for root node
                g.edge(
                    f"{parent_node_index}",
                    f"{node_index}",
                    headlabel=edge_label,
                    color=edgecolor,
                    style=style,
                    labeldistance="2.5",
                    labelangle=labelangle,
                )

            # Traverse child or append leaf value
            _build_graphviz_recurse(
                node["children"]["left"],
                parent_node_index=node_index,
                parent_depth=parent_depth + 1,
                edge_label="True",
                labelangle="45",
            )
            _build_graphviz_recurse(
                node["children"]["right"],
                parent_node_index=node_index,
                parent_depth=parent_depth + 1,
                edge_label="False",
                labelangle="-45",
            )

        def _flag_tree_walk(node, X):  # noqa: ANN001, ANN202, N803
            node["flag"] = True
            if node["children"]["left"] is None and node["children"]["right"] is None:
                return None
            if X[node["j_feature"]] <= node["threshold"]:
                return _flag_tree_walk(node["children"]["left"], X)
            if X[node["j_feature"]] > node["threshold"]:
                return _flag_tree_walk(node["children"]["right"], X)
            return None

        # Flag the node path to highlight later
        if X is not None:
            _flag_tree_walk(self.tree, X)

        # Build graph
        _build_graphviz_recurse(
            self.tree, parent_node_index=0, parent_depth=0, edge_label=""
        )

        if file_name == "source":
            return g.source

        if file_name is None:
            try:
                g.render(view=True)
            except Exception:
                file_name = "tree.dot"
                print(
                    "GraphViz source file written to "
                    + file_name
                    + " and can be viewed using an online renderer. Alternatively you can install graphviz on your system to render locally"
                )

        if (
            file_name is not None
        ):  # not elif here as file_name might be updated in try-except above
            with open(file_name, "w") as file:  # noqa: PTH123
                file.write(str(g.source))
        return None

    def get_node(self, inode):  # noqa: ANN001, ANN201
        """Returns the node corresponding to a given node number.

        Parameters
        ----------
        inode : int
            The node number.

        Returns:
        -------
        dict
            Dictionary containing the data for the requested node.
        """

        # Find node with given index inode. Traverse all children until correct node found.
        def _get_node_from_n(node):  # noqa: ANN001, ANN202
            if (
                node is not None
            ):  # Need to check if node is None here as below _get_node_from_n() calls on children will result in None if leaf node
                if node["index"] == inode:
                    return node
                result = _get_node_from_n(node["children"]["right"])
                if result is None:
                    result = _get_node_from_n(node["children"]["left"])
                return result
            return None

        return _get_node_from_n(self.tree)

    def get_paths(self, X=None):  # noqa: ANN001, ANN201, N803
        """Returns the tree paths for the leaf nodes in the tree.

        Parameters
        ----------
        X : numpy.ndarray, optional
            Array with shape (number_of_observations, dimensions) to apply the tree to.
            If given, paths will only be returned for leaves which contain observations.

        Returns:
        -------
        dict
            Dictionary containing a dict for each leaf node. Indexed by the node indices
            for the leaf nodes.
        """

        def _find_path(node, path, i) -> bool:  # noqa: ANN001
            """Private recursive function to find path through a tree for a given leaf.

            node.
            """
            node_index = node["index"]
            info = {
                "node": node_index,
                "j": node["j_feature"],
                "thresh": node["threshold"],
            }
            path.append(info)
            if node_index == i:
                return True
            left = False
            right = False
            if node["children"]["left"] is not None:
                left = _find_path(node["children"]["left"], path, i)
            if node["children"]["right"] is not None:
                right = _find_path(node["children"]["right"], path, i)
            if left or right:
                return True
            path.remove(info)
            return False

        # Get leaf node id's
        if X is None:  # noqa: SIM108
            leave_id = self.get_leaves()
        else:
            # Get leaf nodes
            leave_id = self.apply(X)

        # Loop through leaves and find path for each.
        paths = {}
        for leaf in np.unique(leave_id):
            path_leaf = []
            _find_path(self.tree, path_leaf, leaf)

            # Set split info to None for leaf node
            path_leaf[-1]["j"] = None
            path_leaf[-1]["thresh"] = None

            # Save in dict
            paths[leaf] = path_leaf

        return paths

    def plot_decision_surface(  # noqa: ANN201
        self,
        ij,  # noqa: ANN001
        ax=None,  # noqa: ANN001
        X=None,  # noqa: ANN001, N803
        y=None,  # noqa: ANN001
        max_depth=None,  # noqa: ANN001
        label=True,  # noqa: ANN001
        color="data",  # noqa: ANN001
        colorbar=True,  # noqa: ANN001
        show=True,  # noqa: ANN001
        kwargs=None,  # noqa: ANN001
    ):
        """Plots the decision boundaries of the PolyTree over a 2D surface. See.

        :meth:`~equadratures.plot.plot_decision_surface` for full description.
        """
        if kwargs is None:
            kwargs = {}
        return plot.plot_decision_surface(
            self, ij, ax, X, y, max_depth, label, color, colorbar, show, kwargs
        )

    def _find_split_from_grad(self, model, X, y):  # noqa: ANN001, ANN202, N803
        """Private method to find the optimal split point for a tree node based on the.

        training data in that node.

        Parameters
        ----------
        model : Poly
            An instance of the Poly class, corresponding to the Poly belonging to the
            tree node.
        X : numpy.ndarray
                An ndarray with shape (number_of_observations, dimensions) containing
                the input data belonging to the tree node.
        y : numpy.ndarray
                An ndarray with shape (number_of_observations, 1) containing the
                response data belonging to the tree node.

        Returns:
        -------
        tuple
            Tuple (did_split, split_dim, split_val), where:
                did_split (bool): True if a split was found, otherwise False.
                split_dim (int): The dimension in X within which the best split was
                found.
                split_val (float): The location of the best split.
        """
        renorm = True
        N, _D = np.shape(X)  # noqa: N806

        # Gradient of loss wrt model coefficients
        P = model.get_poly(X).T  # noqa: N806
        r = y - model.get_polyfit(X)
        g = r * P

        # Sum of gradients
        gsum = g.sum(axis=0)

        # Loop through all dimensions in X
        gain_max = -np.inf
        for d in self.split_dims:
            # Sort along feature i
            sort = np.argsort(X[:, d])
            Xd = X[sort, d]  # noqa: N806

            # Find unique values along one column. #TODO - grid search option
            _, splits = np.unique(Xd, return_index=True)
            splits = splits[1:]

            # Number of samples on left and right split
            N_l = splits  # noqa: N806
            N_r = N - N_l  # noqa: N806

            # Only take splits where both children have more than `min_samples_leaf` samples
            idx = np.minimum(N_l, N_r) >= self.min_samples_leaf
            splits = splits[idx]
            N_l = N_l[idx].reshape(-1, 1)  # noqa: N806
            N_r = N_r[idx].reshape(-1, 1)  # noqa: N806

            # If we've run out of candidate spilts, skip
            if len(splits) <= 1:
                continue

            # Sums of gradients for left and right
            gsum_left = g[sort, :].cumsum(axis=0)
            gsum_left = gsum_left[splits - 1, :]
            gsum_right = gsum - gsum_left

            # Renorm. gradients to zero mean and unit std
            if renorm:
                mu_l, mu_r, sigma_l, sigma_r = self._get_mean_and_sigma(
                    P[:, 1:], splits, N_l, N_r, sort
                )
                gsum_left = self._renormalise(gsum_left, 1 / sigma_l, -mu_l / sigma_l)
                gsum_right = self._renormalise(gsum_right, 1 / sigma_r, -mu_r / sigma_r)

            # Compute the Gain (see Eq. (6) in [1])
            gain = (gsum_left**2).sum(axis=1) / N_l.reshape(-1) + (gsum_right**2).sum(
                axis=1
            ) / N_r.reshape(-1)

            # Find best gain and compare with previous best
            best_idx = np.argmax(gain)
            gain = gain[best_idx]
            if gain > gain_max:
                gain_max = gain
                best_split_dim = d
                best_split_val = 0.5 * (Xd[splits[best_idx] - 1] + Xd[splits[best_idx]])

        # If gain_max stilll == -np.inf, we must have passed through all features w/o finding a split
        # so return False. Otherwise return True and the spilt details.
        if gain_max == -np.inf:
            return False, None, None
        return True, best_split_dim, best_split_val

    @staticmethod
    def _get_mean_and_sigma(X, splits, N_l, N_r, sort):  # noqa: ANN001, ANN205, N803
        """Computes mean and standard deviation of the data in array X, when it is.

        split in two by the threshold values in the splits array. The data is offset by
        its mean to avoid catastrophic cancellation when computing the variance (see
        ref. [3]).

        Parameters
        ----------
        X : numpy.ndarray
            Arrray with dimensions (N,ndim) containing the orthogonal polynomials P.
        splits : numpy.ndarray
            Array of split locations.
        N_l : numpy.ndarray
            Array containing info on number of samples to left of splits.
        N_r : numpy.ndarray
            Array containing info on number of samples to right of splits.
        sort : numpy.ndarray
            Index array to reorder X.
        """
        # Min value of sigma (for stability later)
        epsilon = 0.001

        # Reorder, and shift X by mean
        mu = np.reshape(np.mean(X, axis=0), (1, -1))
        Xshift = X[sort] - mu  # noqa: N806

        # Cumulative sums (and sums of squares) for left and right splits
        Xsum_l = Xshift.cumsum(axis=0)  # noqa: N806
        Xsum_r = Xsum_l[-1:, :] - Xsum_l  # noqa: N806
        X2sum_l = (Xshift**2).cumsum(axis=0)  # noqa: N806
        X2sum_r = X2sum_l[-1:, :] - X2sum_l  # noqa: N806

        # Compute mean of left and right side for all splits
        mu_l = Xsum_l[splits - 1, :] / N_l
        mu_r = Xsum_r[splits - 1, :] / N_r

        # Compute standard deviation of left and right side for all splits
        sigma_l = np.sqrt(
            np.maximum(X2sum_l[splits - 1, :] / (N_l - 1) - mu_l**2, epsilon**2)
        )
        sigma_r = np.sqrt(
            np.maximum(X2sum_r[splits - 1, :] / (N_r - 1) - mu_r**2, epsilon**2)
        )

        # Correct for previous shift
        mu_l = mu_l + mu
        mu_r = mu_r + mu

        return mu_l, mu_r, sigma_l, sigma_r

    @staticmethod
    def _renormalise(gradients, a, c):  # noqa: ANN001, ANN205
        """Renormalises gradients according to according to eq. (14) of [1].

        Parameters.
        ----------
        gradients : numpy.ndarray
            Array with shape (n_samples, n_params), containing the gradients.
        a : numpy.ndarray
            Array with shape (n_samples, n_params-1) containing the normalisation
            factors.
        c: numpy.ndarray
            Array with shape (n_samples, n_params-1) containing the normalisation
            offsets.

        Returns:
        -------
        gradients : numpy.ndarray
            Array with shape (n_samples, n_params) containing the renormalised
            gradients.
        """
        c = c * gradients[:, 0].reshape(-1, 1)
        gradients[:, 1:] = gradients[:, 1:] * a + c
        return gradients

    @staticmethod
    def _calc_domain_vol(Polynomial):  # noqa: ANN001, ANN205, N803
        params = Polynomial.parameters
        vol = 1.0
        for param in params:
            vol *= param.upper - param.lower
        return vol
