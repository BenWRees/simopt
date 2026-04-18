"""The Exponential distribution."""

import numpy as np
from scipy.stats import expon

from equadratures.distributions.template import Distribution

RECURRENCE_PDF_SAMPLES = 8000


class Exponential(Distribution):
    """The class defines a Exponential object. It is the child of Distribution.

    :param double rate:
    Rate parameter of the Exponential distribution.
    """

    def __init__(self, rate=None) -> None:  # noqa: ANN001, D107
        if rate is None:
            self.rate = 1.0
        else:
            self.rate = rate

        if (self.rate is not None) and (self.rate > 0.0):
            self.parent = expon(scale=1.0 / rate)
            self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(
                moments="mvsk"
            )
            self.bounds = np.array([0.0, np.inf])
            self.x_range_for_pdf = np.linspace(
                0.0, 20 * self.rate, RECURRENCE_PDF_SAMPLES
            )
        else:
            raise ValueError(
                "Invalid parameters in exponential distribution. Rate should be positive."
            )

    def get_description(self):  # noqa: ANN201
        """A description of the Exponential distribution.

        :param Exponential self:
            An instance of the Exponential class.
        :return:
            A string describing the Exponential distribution.
        """
        return (
            "is an exponential distribution with a rate parameter of"
            + str(self.rate)
            + "."
        )

    def get_pdf(self, points=None):  # noqa: ANN001, ANN201
        """An exponential probability density function.

        :param Exponential self:
            An instance of the Exponential class.
        :param matrix points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N values over the support of the distribution.
        :return:
            Probability density values along the support of the exponential
            distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        raise ValueError("Please digit an input for getPDF method")

    def get_icdf(self, xx):  # noqa: ANN001, ANN201
        """An inverse exponential cumulative density function.

        :param Exponential self:
            An instance of the Exponential class.
        :param array xx:
            A numpy array of uniformly distributed samples between [0,1].
        :return:
            Inverse CDF samples associated with the exponential distribution.
        """
        return self.parent.ppf(xx)

    def get_cdf(self, points=None):  # noqa: ANN001, ANN201
        """An exponential cumulative density function.

        :param Exponential self:
            An instance of the Exponential class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N values over the support of the distribution.
        :return:
            Cumulative density values along the support of the exponential distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        raise ValueError("Please digit an input for getCDF method")

    def get_samples(self, m=None):  # noqa: ANN001, ANN201
        """Generates samples from the Exponential distribution.

        :param Expon self:
            An instance of the Exponential class.
        :param integer m:
             Number of random samples. If no value is provided, a default of 5e05 is
             assumed
        :return:
            A N-by-1 vector that contains the samples.
        """
        number = m if m is not None else 500000
        return self.parent.rvs(size=number)
