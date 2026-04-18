# ---  # noqa: D100
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys

sys.path.append("..")

# %%
import random

import matplotlib.pyplot as plt
import numpy as np

from simopt.experiment_base import (
    PlotType,
    ProblemSolver,
    plot_progress_curves,
    post_normalize,
)
from simopt.input_models import InputModel


# %%
class DemandInputModel(InputModel):  # noqa: D101
    def set_rng(self, rng: random.Random) -> None:  # noqa: D102
        self.rng = rng

    def unset_rng(self) -> None:  # noqa: D102
        self.rng = None

    def random(self, burr_c: float, burr_k: float) -> float:  # noqa: ARG002, D102
        mean = 10
        std = 0.5
        return self.rng.normalvariate(mean, std)


# %%
class FileInputModel(InputModel):  # noqa: D101
    def __init__(self, filename) -> None:  # noqa: ANN001, D107
        self.data = np.load(filename)

    def set_rng(self, rng: random.Random) -> None:  # noqa: D102
        self.rng = rng

    def unset_rng(self) -> None:  # noqa: D102
        self.rng = None

    def random(self, burr_c: float, burr_k: float) -> float:  # noqa: ARG002, D102
        return np.random.choice(self.data, size=1, replace=True)[0]


# %%
class Experiment(ProblemSolver):  # noqa: D101
    def model_created(self, model) -> None:  # noqa: ANN001, D102
        # model.demand_model = DemandInputModel()
        model.demand_model = FileInputModel("demand.npy")

    def before_replicate(self, model, rng_list) -> None:  # noqa: ANN001, D102
        model.demand_model.set_rng(rng_list[0])


# %%
# Run 10 macroreplications of ASTRO-DF on the continuous newsvendor problem.
experiment = Experiment("ASTRODF", "CNTNEWS-1")
experiment.run(n_macroreps=10)

# Post-process the results.
experiment.post_replicate(n_postreps=200)
post_normalize(experiments=[experiment], n_postreps_init_opt=200)

# Record the results and plot the mean progress curve.
experiment.log_experiment_results()
plot_progress_curves(
    experiments=[experiment],
    plot_type=PlotType.MEAN,
    normalize=False,
)
plt.show()
