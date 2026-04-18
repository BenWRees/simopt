"""The Distribution template."""

import numpy as np

from equadratures.distributions.recurrence_utils import custom_recurrence_coefficients

PDF_SAMPLES = 500000


class Distribution:
    """The class defines a Distribution object. It serves as a template for all.

    distributions.

    :param double lower:
        Lower bound of the support of the distribution.
    :param double upper:
        Upper bound of the support of the distribution.
    """

    def __init__(  # noqa: D107
        self,
        mean=None,  # noqa: ANN001
        variance=None,  # noqa: ANN001
        lower=None,  # noqa: ANN001
        upper=None,  # noqa: ANN001
        shape=None,  # noqa: ANN001, ARG002
        scale=None,  # noqa: ANN001
        rate=None,  # noqa: ANN001
    ) -> None:
        self.mean = mean
        self.variance = variance
        self.lower = lower
        self.upper = upper
        self.rate = rate
        self.scale = scale
        self.x_range_for_pdf = []

    def get_description(self) -> None:
        """Returns the description of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        pass

    def get_pdf(self, points=None) -> None:  # noqa: ANN001
        """Returns the PDF of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        pass

    def get_cdf(self, points=None) -> None:  # noqa: ANN001
        """Returns the CDF of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        pass

    def get_icdf(self, xx) -> None:  # noqa: ANN001
        """An inverse cumulative density function.

        :param Distribution self:
                An instance of the distribution class.
        :param xx:
                A numpy array of uniformly distributed samples between [0,1].
        :return:
                Inverse CDF samples associated with the gamma distribution.
        """
        pass

    def get_recurrence_coefficients(self, order):  # noqa: ANN001, ANN201
        """Recurrence coefficients for the distribution.

        :param Distribution self:
            An instance of the distribution class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the distribution.
        """
        w_pdf = self.get_pdf(self.x_range_for_pdf)
        return custom_recurrence_coefficients(self.x_range_for_pdf, w_pdf, order)

    def get_samples(self, m=None):  # noqa: ANN001, ANN201
        """Generates samples from the distribution.

        :param Distribution self:
            An instance of the distribution class.
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e5 is
            assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        number_of_random_samples = PDF_SAMPLES if m is None else m
        uniform_samples = np.random.random((number_of_random_samples, 1))
        return self.get_icdf(uniform_samples)
