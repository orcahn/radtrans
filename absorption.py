import numpy as np


# test case: no absorption
def no_abs(x, L):

    return np.zeros(x.shape)


# constant absorption throughout the domain. Recovers the case of
# homogeneous medium in the domain
def const_abs(x, L):

    return np.full(x.shape, 1.0)


# positive gradient from 0.0 at 0 to 1.0 at L
def pos_grad_abs(x, L):

    return x / L


# gaussian with stddev 1.0, centered at L/2
def gaussian_abs(x, L):

    return np.exp(-0.5 * (x - 0.5 * L) * (x - 0.5 * L)) / np.sqrt(2.0 * np.pi)


# discontinuous absorption coefficient
def step_abs(x, L):

    return np.heaviside(x - L / 2.0, 1.0)


class Absorption:
    """
    Class that represents some types of constant and
    spatially varying absorption coefficients
    """

    def __init__(self, abs_fun_type, domain_length):

        assert abs_fun_type in ['none', 'const', 'posGrad', 'gaussian',
                                'step'], 'Absorption type ' + abs_fun_type + \
            ' currently not supported.'

        self.abs_fun = None

        if abs_fun_type == 'none':

            self.abs_fun = lambda x: no_abs(x, domain_length)

        elif abs_fun_type == 'const':

            self.abs_fun = lambda x: const_abs(x, domain_length)

        elif abs_fun_type == 'posGrad':

            self.abs_fun = lambda x: pos_grad_abs(x, domain_length)

        elif abs_fun_type == 'gaussian':

            self.abs_fun = lambda x: gaussian_abs(x, domain_length)

        elif abs_fun_type == 'step':

            self.abs_fun = lambda x: step_abs(x, domain_length)
