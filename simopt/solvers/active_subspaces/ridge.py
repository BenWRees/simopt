# type: ignore
"""Ridge function approximation from function values."""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

from copy import deepcopy

import numpy as np
from matplotlib.axes import Axes

# from .pgf
from matplotlib.path import Path

from .poly import BaseFunction
from .subspace import SubspaceBasedDimensionReduction


class PGF:  # noqa: D101
    def __init__(self) -> None:  # noqa: D107
        self.column_names = []
        self.columns = []

    def add(self, name, column) -> None:  # noqa: ANN001, D102
        if len(self.columns) > 1:
            assert len(self.columns[0]) == len(column)

        self.columns.append(deepcopy(column))
        self.column_names.append(name)

    def keys(self):  # noqa: ANN201, D102
        return self.column_names

    def __getitem__(self, key):  # noqa: ANN001, ANN204, D105
        i = self.column_names.index(key)
        return self.columns[i]

    def write(self, filename) -> None:  # noqa: ANN001, D102
        f = open(filename, "w")  # noqa: PTH123, SIM115

        for name in self.column_names:
            f.write(name + "\t")
        f.write("\n")

        for j in range(len(self.columns[0])):
            for col in self.columns:
                f.write(f"{float(col[j])}\t")
            f.write("\n")

        f.close()

    def read(self, filename) -> None:  # noqa: ANN001, D102
        with open(filename) as f:  # noqa: PTH123
            for i, line in enumerate(f):
                # Remove the newline and trailing tab if present
                line = line.replace("\t\n", "").replace("\n", "")
                if i == 0:
                    self.column_names = line.split("\t")
                    self.columns = [[] for name in self.column_names]
                else:
                    cols = line.split("\t")
                    for j, col in enumerate(cols):
                        self.columns[j].append(float(col))


def save_contour(fname, cs, fmt="matlab", simplify=1e-3, **kwargs) -> None:  # noqa: ANN001, ANN003, ARG001
    """Save a contour plot to a file for pgfplots.

    Additional arguments are passed to iter_segements
    Important, simplify = True will remove invisible points
    """

    def write_path_matlab(fout, x_vec, y_vec, z) -> None:  # noqa: ANN001
        # Now dump this data back out
        # Header is level followed by number of rows
        fout.write("%15.15e\t%15d\n" % (z, len(x_vec)))  # noqa: UP031
        for x, y in zip(x_vec, y_vec, strict=False):
            fout.write(f"{x:15.15e}\t{y:15.15e}\n")

    def write_path_prepared(fout, x_vec, y_vec, z) -> None:  # noqa: ANN001
        fout.write(f"{x_vec:15.15e}\t{y_vec:15.15e}\t{z:15.15e}\n")
        fout.write("\t\t\t\n")

    if fmt == "matlab":
        write_path = write_path_matlab
    elif fmt == "prepared":
        write_path = write_path_prepared
    else:
        raise NotImplementedError

    with open(fname, "w") as fout:  # noqa: PTH123
        for col, z in zip(cs.collections, cs.levels, strict=False):
            for path in col.get_paths():
                path.simplify_threshold = simplify
                x_vec = []
                y_vec = []
                for _i, ((x, y), code) in enumerate(path.iter_segments(simplify=True)):
                    if code == Path.MOVETO:
                        if len(x_vec) != 0:
                            write_path(fout, x_vec, y_vec, z)
                            x_vec = []
                            y_vec = []
                        x_vec.append(x)
                        y_vec.append(y)

                    elif code == Path.LINETO:
                        x_vec.append(x)
                        y_vec.append(y)

                    elif code == Path.CLOSEPOLY:
                        x_vec.append(x_vec[0])
                        y_vec.append(y_vec[0])
                    else:
                        print("received code", code)

                write_path(fout, x_vec, y_vec, z)


class RidgeFunction(BaseFunction, SubspaceBasedDimensionReduction):  # noqa: D101
    # @property
    # def U(self):
    # 	return self._U

    def shadow_plot(  # noqa: ANN201, D102
        self,
        X=None,  # noqa: ANN001, N803
        fX=None,  # noqa: ANN001, N803
        dim: int | None = None,
        U=None,  # noqa: ANN001, N803
        ax="auto",  # noqa: ANN001
        pgfname=None,  # noqa: ANN001
    ):
        if dim is None and U is not None:
            dim = U.shape[1]
        else:
            assert dim == U.shape[1]

        ax = SubspaceBasedDimensionReduction.shadow_plot(
            self, X=X, fX=fX, dim=dim, ax=ax, pgfname=pgfname
        )

        # Draw the response surface
        if dim == 1:
            Y = np.dot(U.T, X.T).T  # noqa: N806
            lb = np.min(Y)
            ub = np.max(Y)

            xx = np.linspace(lb, ub, 500)
            Uxx = np.hstack([U * xxi for xxi in xx]).T  # noqa: N806
            yy = self.eval(Uxx)

            if ax is not None and isinstance(ax, Axes):
                ax.plot(xx, yy, "r-")

            if pgfname is not None:
                pgfname2 = (
                    pgfname[: pgfname.rfind(".")]
                    + "_response"
                    + pgfname[pgfname.rfind(".") :]
                )
                pgf = PGF()
                pgf.add("x", xx)
                pgf.add("fx", yy)
                pgf.write(pgfname2)

        elif dim == 2 and isinstance(ax, Axes):
            Y = np.dot(U.T, X.T).T  # noqa: N806
            lb0 = np.min(Y[:, 0])
            ub0 = np.max(Y[:, 0])

            lb1 = np.min(Y[:, 1])
            ub1 = np.max(Y[:, 1])

            # Constuct mesh on the domain
            xx0 = np.linspace(lb0, ub0, 50)
            xx1 = np.linspace(lb1, ub1, 50)
            XX0, XX1 = np.meshgrid(xx0, xx1)  # noqa: N806
            UXX = np.vstack([XX0.flatten(), XX1.flatten()])  # noqa: N806
            XX = np.dot(U, UXX).T  # noqa: N806
            YY = self.eval(XX).reshape(XX0.shape)  # noqa: N806

            ax.contour(
                xx0,
                xx1,
                YY,
                levels=np.linspace(np.min(fX), np.max(fX), 20),
                vmin=np.min(fX),
                vmax=np.max(fX),
                linewidths=0.5,
            )

        else:
            raise NotImplementedError(
                "Cannot draw shadow plots in more than two dimensions"
            )

        return ax
