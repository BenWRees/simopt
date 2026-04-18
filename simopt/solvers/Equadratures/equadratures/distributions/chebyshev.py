"""The Chebyshev / Arcsine distribution."""

import numpy as np
from scipy.stats import arcsine

from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
from equadratures.distributions.template import Distribution

RECURRENCE_PDF_SAMPLES = 8000


class Chebyshev(Distribution):
    """The class defines a Chebyshev object. It is the child of Distribution.

    :param double lower:
    Lower bound of the support of the Chebyshev (arcsine) distribution.
    :param double upper:
    Upper bound of the support of the Chebyshev (arcsine) distribution.
    """

    def __init__(self, lower, upper) -> None:  # noqa: ANN001, D107
        if lower is None:
            self.lower = 0.0
        else:
            self.lower = lower
        if upper is None:
            self.upper = 1.0
        else:
            self.upper = upper

        if self.lower > self.upper:
            raise ValueError(
                "Invalid Beta distribution parameters. Lower should be smaller than upper."
            )

        self.bounds = np.array([self.lower, self.upper])
        self.x_range_for_pdf = np.linspace(
            self.lower, self.upper, RECURRENCE_PDF_SAMPLES
        )
        loc = self.lower
        scale = self.upper - self.lower

        self.parent = arcsine(loc=loc, scale=scale)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(
            moments="mvsk"
        )
        self.shape_parameter_A = -0.5
        self.shape_parameter_B = -0.5

    def get_description(self):  # noqa: ANN201
        """A description of the Chebyshev (arcsine) distribution.

        :param Chebyshev self:
            An instance of the Chebyshev (arcsine) class.
        :return:
            A string describing the Chebyshev (arcsine) distribution.
        """
        return (
            "is a Chebyshev or arcsine distribution that is characterised by its lower bound, which is"
            + str(self.lower)
            + " and its upper bound, which is"
            + str(self.upper)
            + "."
        )

    def get_pdf(self, points=None):  # noqa: ANN001, ANN201
        """A Chebyshev probability density function.

        :param Chebyshev self:
            An instance of the Chebyshev (arcsine) class.
        :param points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N the support of the Chebyshev (arcsine) distribution.
        :return:
            Probability density values along the support of the Chebyshev (arcsine)
            distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        raise ValueError("Please digit an input for getPDF method")

    def get_cdf(self, points=None):  # noqa: ANN001, ANN201
        """A Chebyshev cumulative density function.

        :param Chebyshev self:
            An instance of the Chebyshev class.
        :param points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N values over the support of the Chebyshev (arcsine)
            distribution.
        :return:
            Cumulative density values along the support of the Chebyshev (arcsine)
            distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        raise ValueError("Please digit an input for getCDF method")

    def get_recurrence_coefficients(self, order):  # noqa: ANN001, ANN201
        """Recurrence coefficients for the Chebyshev distribution.

        :param Chebyshev self:
            An instance of the Chebyshev class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the Chebyshev distribution.
        """
        return jacobi_recurrence_coefficients(
            self.shape_parameter_A,
            self.shape_parameter_B,
            self.lower,
            self.upper,
            order,
        )

    def get_icdf(self, xx):  # noqa: ANN001, ANN201
        """A Arcisine inverse cumulative density function.

        :param Arcsine self:
            An instance of Arcisine class.
        :param xx:
            A matrix of points at which the inverse cumulative density function needs to
            be evaluated.
        :return:
            Inverse cumulative density function values of the Arcisine distribution.
        """
        return self.parent.ppf(xx)

    def get_samples(self, m=None):  # noqa: ANN001, ANN201
        """Generates samples from the Arcsine distribution.

        :param arcsine self:
            An instance of Arcsine class.
        :param integer m:
            Number of random samples. If not provided, a default of 5e05 is assumed.

        """
        number = m if m is not None else 500000
        return self.parent.rvs(size=number)
