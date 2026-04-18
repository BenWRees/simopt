"""The Chi-squared distribution."""

import numpy as np
from scipy.stats import chi2

from equadratures.distributions.template import Distribution

RECURRENCE_PDF_SAMPLES = 50000


class Chisquared(Distribution):
    """The class defines a Chi-squared object. It is the child of Distribution.

    :param int dofs:
    Degrees of freedom for the chi-squared distribution.
    """

    def __init__(self, dofs) -> None:  # noqa: ANN001, D107
        if dofs is None:
            self.dofs = 1
        else:
            self.dofs = int(dofs)

        if not isinstance(self.dofs, int) or self.dofs < 1:
            raise ValueError(
                "Invalid parameter in chi2 distribution: dofs must be positive integer."
            )

        if self.dofs == 1:
            self.bounds = np.array([1e-15, np.inf])
        else:
            self.bounds = np.array([0.0, np.inf])

        self.parent = chi2(self.dofs)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(
            moments="mvsk"
        )
        self.x_range_for_pdf = np.linspace(
            0.0, 10.0 * self.mean, RECURRENCE_PDF_SAMPLES
        )

    def get_description(self):  # noqa: ANN201
        """A description of the Chi-squared distribution.

        :param Chisquared self:
            An instance of the Chi-squared class.
        :return:
            A string describing the Chi-squared distribution.
        """
        return (
            "is a chi-squared distribution; characterised by its degrees of freedom, which here is"
            + str(self.dofs)
            + "."
        )

    def get_pdf(self, points=None):  # noqa: ANN001, ANN201
        """A Chi-squared  probability density function.

        :param Chisquared self:
            An instance of the Chi-squared  class.
        :param points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the Chi-squared
            distribution.
        :return:
            Probability density values along the support of the Chi-squared
            distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        raise ValueError("Please digit an input for getPDF method")

    def get_cdf(self, points=None):  # noqa: ANN001, ANN201
        """A Chi-squared cumulative density function.

        :param Chisquared self:
            An instance of the Chi-squared class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the Chi-squared
            distribution.
        :return:
            Cumulative density values along the support of the Chi-squared distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        raise ValueError("Please digit an input for getCDF method")

    def get_icdf(self, xx):  # noqa: ANN001, ANN201
        """A Chi-squared inverse cumulative density function.

        :param Chisquared self:
            An instance of Chi-squared class
        :param matrix xx:
            A matrix of points at which the inverse cumulative density function need to
            be evaluated.
        :return:
            Inverse cumulative density function values of the Chi-squared distribution.
        """
        return self.parent.ppf(xx)

    def get_samples(self, m=None):  # noqa: ANN001, ANN201
        """Generates samples from the Chi-squared distribution.

        :param Chisquared self:
            An instance of Chi-squared class
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e05 is
            assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        number = m if m is not None else 500000
        return self.parent.rvs(size=number)
