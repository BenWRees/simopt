"""The Pareto distribution."""

import numpy as np
from scipy.stats import pareto

from equadratures.distributions.template import Distribution

RECURRENCE_PDF_SAMPLES = 50000


class Pareto(Distribution):
    """The class defines a Pareto object. It is the child of Distribution.

    :param int shape_parameter:
    The shape parameter associated with the Pareto distribution.
    """

    def __init__(self, shape_parameter) -> None:  # noqa: ANN001, D107
        if shape_parameter is None:
            self.shape_parameter = 1.0
        else:
            self.shape_parameter = shape_parameter

        self.bounds = np.array([0.999, np.inf])
        if self.shape_parameter < 0:
            raise ValueError(
                "Invalid parameters in Pareto distribution. Scale should be positive."
            )
        self.parent = pareto(self.shape_parameter)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(
            moments="mvsk"
        )
        self.x_range_for_pdf = np.linspace(
            0.999, self.shape_parameter + 20.0, RECURRENCE_PDF_SAMPLES
        )

    def get_description(self):  # noqa: ANN201
        """A description of the Pareto distribution.

        :param Pareto self:
            An instance of the Pareto class.
        :return:
            A string describing the Pareto distribution.
        """
        return (
            "is a pareto distribution which is characterised by its shape parameter, which here is"
            + str(self.shape_parameter)
            + ". While the distribution can be characterized by a shape parameter and a scale parameter, in Effective Quadratures we use only the one, that is the scale parameter is set to 1. "
        )

    def get_pdf(self, points=None):  # noqa: ANN001, ANN201
        """A Pareto probability density function.

        :param Pareto self:
            An instance of the Pareto class.
        :param points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the Pareto
            distribution.
        :return:
            Probability density values along the support of the Pareto distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        raise ValueError("Please digit an input for get_pdf method")

    def get_cdf(self, points=None):  # noqa: ANN001, ANN201
        """A Pareto cumulative density function.

        :param Pareto self:
            An instance of the Pareto class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the Pareto
            distribution.
        :return:
            Cumulative density values along the support of the Pareto distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        raise ValueError("Please digit an input for get_cdf method")

    def get_icdf(self, xx):  # noqa: ANN001, ANN201
        """A Pareto inverse cumulative density function.

        :param Pareto:
            An instance of Pareto class
        :param matrix xx:
            A matrix of points at which the inverse cumulative density function need to
            be evaluated.
        :return:
            Inverse cumulative density function values of the Pareto distribution.
        """
        return self.parent.ppf(xx)

    def get_samples(self, m=None):  # noqa: ANN001, ANN201
        """Generates samples from the Pareto distribution.

        :param Pareto self:
            An instance of Pareto class
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e05 is
            assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        number = m if m is not None else 500000
        return self.parent.rvs(size=number)
