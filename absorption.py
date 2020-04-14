import numpy as np

from numpy.linalg import norm


# test case: no absorption
def no_abs(x):

    return 0.0


# constant absorption throughout the domain. Recovers the case of
# homogeneous medium in the domain
def const_abs(x):

    return 1.0


# positive gradient from 0.0 at 0 to 1.0 at L
def pos_grad_abs(x, L):

    return x[0] / L[0]


# gaussian with stddev 1.0, centered at L/2
def gaussian_abs(x, L):

    res = np.exp(-0.5 * (x[0] - 0.5 * L[0]) *
                 (x[0] - 0.5 * L[0])) / np.sqrt(2.0 * np.pi)

    if len(x) == 2:

        res *= np.exp(-0.5 * (x[1] - 0.5 * L[1]) *
                      (x[1] - 0.5 * L[1])) / np.sqrt(2.0 * np.pi)

    return res


# discontinuous absorption coefficient
def step_abs(x, L):

    dist = [x[d] - L[d] / 2.0 for d in range(len(x))]
    if norm(dist) < min(L) / 4.0:
        return 1.0
    else:
        return 0.0


# piecewise constant absorption with random values
# following a gaussian distribution
def gaussian_random_piecewise(x, constants):

    bool_array1 = [x >= 0.5 * i for i in range(len(constants))]
    bool_array2 = [x < 0.5 * (i + 1) for i in range(len(constants))]

    return np.piecewise(x, np.logical_and(bool_array1, bool_array2), constants)


class Absorption:
    """
    Wrapper for the spatially varying absorption coefficient in the radiative
    transfer problem.

    Attributes
    ----------
    abs_fun : callable
        Callable object, representing the absorption coefficient. Takes
        values in the domain as argument.
    """

    def __init__(self, abs_fun_type, domain_length):
        """
        Paramters
        ---------
        abs_fun_type : string
            Type of absorption coefficient as parsed from the .ini file
        domain_length : float
            Right boundary L of the domain (0, L)
        """

        assert abs_fun_type in ['none', 'const', 'posGrad', 'gaussian',
                                'step', 'gaussianRandomPiecewise'], \
            'Absorption type ' + abs_fun_type + ' currently not supported.'

        self.abs_fun = None

        if abs_fun_type == 'none':

            self.abs_fun = lambda x: no_abs(x)

        elif abs_fun_type == 'const':

            self.abs_fun = lambda x: const_abs(x)

        elif abs_fun_type == 'posGrad':

            self.abs_fun = lambda x: pos_grad_abs(x, domain_length)

        elif abs_fun_type == 'gaussian':

            self.abs_fun = lambda x: gaussian_abs(x, domain_length)

        elif abs_fun_type == 'step':

            self.abs_fun = lambda x: step_abs(x, domain_length)

        elif abs_fun_type == 'gaussianRandomPiecewise':

            num_intervals = int(domain_length * 2)

            # Samples from a Gaussian distribution with mean 0.5
            # and standard deviation 0.15
            constants = np.array(
                [0.15 * np.random.randn() + 0.5
                 for i in range(num_intervals)]).clip(min=0, max=1)

            self.abs_fun = lambda x: gaussian_random_piecewise(x, constants)
