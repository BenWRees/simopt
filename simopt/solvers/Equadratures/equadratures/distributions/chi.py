"""The Chebyshev / Arcsine distribution."""

import numpy as np
from scipy.stats import chi

from equadratures.distributions.template import Distribution

RECURRENCE_PDF_SAMPLES = 8000


class Chi(Distribution):
    """The class defines a Chi object. It is the child of Distribution.

    :param int dofs:
    Degrees of freedom for the chi-squared distribution.
    """

    def __init__(self, dofs) -> None:  # noqa: ANN001, D107
        if dofs is None:
            self.dofs = 1
        else:
            self.dofs = dofs

        if self.dofs < 0:
            raise ValueError(
                "Invalid parameter in chi distribution: dofs must be positive."
            )

        if self.dofs == 1:
            self.bounds = np.array([1e-15, np.inf])
        else:
            self.bounds = np.array([0.0, np.inf])

        mean, var, skew, kurt = chi.stats(dofs, moments="mvsk")
        self.mean = mean
        self.variance = var
        self.skewness = skew
        self.kurtosis = kurt
        self.x_range_for_pdf = np.linspace(
            0.0, 10.0 * self.mean, RECURRENCE_PDF_SAMPLES
        )
        self.parent = chi(self.dofs)

    def get_description(self):  # noqa: ANN201
        """A description of the Chi-squared distribution.

        :param Chi-squared self:
            An instance of the Chi-squared class.
        :return:
            A string describing the Chi-squared distribution.
        """
        return (
            "is a chi distribution which is characterised by its degrees of freedom, which here is"
            + str(self.dofs)
            + "."
        )

    def get_pdf(self, points=None):  # noqa: ANN001, ANN201
        """A Chi  probability density function.

        :param Chi  self:
            An instance of the Chi  class.
        :param points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the Chi distribution.
        :return:
            Probability density values along the support of the Chi distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        raise ValueError("Please digit an input for get_pdf method")

    def get_cdf(self, points=None):  # noqa: ANN001, ANN201
        """A Chi cumulative density function.

        :param Chi self:
            An instance of the Chi class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the Chi distribution.
        :return:
            Cumulative density values along the support of the Chi distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        raise ValueError("Please digit an input for get_cdf method")

    def get_icdf(self, xx):  # noqa: ANN001, ANN201
        """A Chi inverse cumulative density function.

        :param Chi:
            An instance of Chi class
        :param matrix xx:
            A matrix of points at which the inverse cumulative density function need to
            be evaluated.
        :return:
            Inverse cumulative density function values of the Chi distribution.
        """
        return self.parent.ppf(xx)

    def get_samples(self, m=None):  # noqa: ANN001, ANN201
        """Generates samples from the Chi distribution.

        :param Chi self:
            An instance of Chi class
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e05 is
            assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        number = m if m is not None else 500000
        return self.parent.rvs(size=number)
