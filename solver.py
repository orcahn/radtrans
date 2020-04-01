import timeit

import numpy as np
import scipy.sparse.linalg as spsla


def invert_transport(M, x, n_dof, n_ord):

    nc = n_dof // n_ord

    prec_vec = np.empty(n_dof)

    # invert the nc x nc diagonal blocks
    for m in range(n_ord):
        prec_vec[m * nc: (m + 1) * nc] = spsla.spsolve(
            M[m * nc: (m + 1) * nc, m * nc: (m + 1) * nc],
            x[m * nc: (m + 1) * nc])

    return prec_vec


def invert_diagonal(M, x):

    return np.multiply(np.reciprocal(M.diagonal()), x)


class solve_counter(object):
    """
    Simple class that counts the number of
    iterations within a scipy iterative solver.
    """

    def __init__(self):
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1


class Preconditioner:
    """
    For radiative transfer problems, it is beneficial
    to use the Lambda iteration as a preconditioner
    """

    def __init__(self, disc, type):

        if type == 'lambdaIteration':
            self.M = spsla.LinearOperator(
                (disc.n_dof, disc.n_dof),
                lambda x: invert_transport(disc.lambda_prec, x,
                                           disc.n_dof, disc.n_ord))

        elif type == 'diagonal':
            self.M = spsla.LinearOperator(
                (disc.n_dof, disc.n_dof),
                lambda x: invert_diagonal(disc.stiff_mat, x))

        else:
            self.M = None


class Solver:
    """
    Class for the linear solvers.
    It can be specified as a direct or iterative GMRES solver.
    In case GMRES or BiCGSTAB is selected, a preconditioner can
    be specified. Additionally, one can supply an initial guess.
    """

    def __init__(self, name, preconditioner):

        self.name = name
        self.prec = preconditioner

    def solve(self, A, b, x_in=None):

        if self.name == "SparseDirect":

            start_time = timeit.default_timer()

            x = spsla.spsolve(A, b)
            elapsed_time = timeit.default_timer() - start_time

            print('Sparse direct solver:    ' +
                  "% 10.3e" % (elapsed_time) + ' s')

            return x, None, elapsed_time

        elif self.name == "GMRES":

            counter = solve_counter()
            start_time = timeit.default_timer()

            x, exit_code = spsla.gmres(
                A=A, b=b, M=self.prec.M, x0=x_in, callback=counter, tol=1e-8)

            elapsed_time = timeit.default_timer() - start_time

            print('GMRES ended with exit code ' + str(exit_code) + ' after ' +
                  str(counter.niter) + ' iterations in ' +
                  "% 10.3e" % (elapsed_time) + ' s')

            return x, counter.niter, elapsed_time

        elif self.name == "BiCGSTAB":

            counter = solve_counter()
            start_time = timeit.default_timer()

            x, exit_code = spsla.bicgstab(
                A=A, b=b, M=self.prec.M, x0=x_in, callback=counter, tol=1e-8)

            elapsed_time = timeit.default_timer() - start_time

            print('BiCGSTAB ended with exit code ' + str(exit_code) +
                  ' after ' + str(counter.niter) + ' iterations in ' +
                  "% 10.3e" % (elapsed_time) + ' s')

            return x, counter.niter, elapsed_time
