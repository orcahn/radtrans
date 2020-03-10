import numpy as np


class Absorption:
    """
    Class that represents some types of constant and
    spatially varying absorption coefficients
    """

    def __init__(self, L, abs_type):
        self.L = L
        self.abs_type = abs_type

    def __call__(self, x):
        if self.abs_type == 'Const':
            return self.const_abs(x)
        elif self.abs_type == 'PosGrad':
            return self.pos_grad_abs(x)
        elif self.abs_type == 'Gaussian':
            return self.gaussian_abs(x)
        elif self.abs_type == 'Step':
            return self.step_abs(x)
        else:
            return self.no_abs(x)

    # test case: no absorption
    def no_abs(self, x):

        return np.zeros(x.shape)

    # constant absorption throughout the domain. Recovers the case of
    # homogeneous medium in the domain
    def const_abs(self, x):

        return np.full(x.shape, 1.0)

    # positive gradient from 0.0 at 0 to 1.0 at L
    def pos_grad_abs(self, x):

        return x / self.L

    # gaussian with stddev 1.0, centered at L/2
    def gaussian_abs(self, x):

        return np.exp(-0.5 * (x - 0.5 * self.L) * (x - 0.5 * self.L)) / np.sqrt(2.0 * np.pi)

    # discontinuous absorption coefficient
    def step_abs(self, x):

        return np.heaviside(x - self.L / 2.0, 1.0)
