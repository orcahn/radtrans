import timeit

import numpy as np
import scipy.sparse.linalg as spsla

from scipy.sparse import identity


def invert_transport(M, x, n_ord):
    """
    Solver specialized in inverting the discretized transport and absorption
    terms. Technically it solves M*p=x, where M is a block-diagonal matrix.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        Explicit sparse representation of the linear preconditioner used in
        the lambda iteration.
    x : numpy.ndarray
        Dummy for a vector to which the preconditioner is applied
    n_ord : integer
        Total number of discrete ordinates used in discretization

    Returns
    -------
    numpy.ndarray
        Array representing the application of the preconditioner
    """

    if isinstance(M, type(identity(x.size))):

        return x

    else:

        nc = x.size // n_ord

        prec_vec = np.empty(x.size)

        # Invert the nc x nc diagonal blocks
        for m in range(n_ord):
            prec_vec[m * nc: (m + 1) * nc] = spsla.spsolve(
                M[m * nc: (m + 1) * nc, m * nc: (m + 1) * nc],
                x[m * nc: (m + 1) * nc])

        return prec_vec


def invert_diagonal(M, x):
    """
    Preconditioner inverting the diagonal

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        Explicit sparse representation of the linear preconditioner
    x : numpy.ndarray
        Dummy for a vector to which the preconditioner is applied

    Returns
    -------
    numpy.ndarray
        Array representing the application of the preconditioner
    """

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
    Preconditioner used in the iterative solution of the discretized
    system

    Attributes
    ----------
    M : scipy.sparse.csr_matrix
        Explicit sparse representation of the linear preconditioner.
    """

    def __init__(self, disc, type):
        """
        Parameters
        ----------
        disc : discretization.FiniteVolume
            Discretization of the continuous model problem
        type : string
            Type of preconditioner to use
        """

        if type == 'lambdaIteration':
            self.M = spsla.LinearOperator(
                (disc.n_dof, disc.n_dof),
                lambda x: invert_transport(disc.lambda_prec, x,
                                           disc.n_ord))

        elif type == 'diagonal':
            self.M = spsla.LinearOperator(
                (disc.n_dof, disc.n_dof),
                lambda x: invert_diagonal(disc.stiff_mat, x))

        else:
            self.M = None


class Solver:
    """
    Class for the linear solvers for the discretized system

    Attributes
    ----------
    name : string
        Type of the solver to be used
    prec : solver.Preconditioner
        Preconditioner to be used for iterative solvers

    Methods
    -------
    solve(A, b, x_in=None)
        Solve the linear system with Matrix A, right hand side b and initial
        guess x_in
    """

    def __init__(self, name, preconditioner):
        """
        Parameters
        ----------
        name : string
            Type of the solver to be used
        preconditioner : solver.Preconditioner
            Preconditioner to be used for iterative solvers
        """

        self.name = name
        self.prec = preconditioner

    def solve(self, A, b, x_in=None):
        """
        Solve the linear system

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Sparse stiffness matrix of the system
        b : numpy.ndarray
            Load vector of the system
        x_in : None or numpy.ndarray
            Initial guess for the solution using an iterative solver

        Returns
        -------
        numpy.ndarray
            solution to the linear system
        integer
            Number of iterations that were performed
        float
            Time it took solving the system

        """

        if self.name == "SparseDirect":

            start_time = timeit.default_timer()

            x = spsla.spsolve(A, b)
            elapsed_time = timeit.default_timer() - start_time

            print('Sparse direct solver:    ' +
                  "% 10.3e" % (elapsed_time) + ' s')

            return x, 1, elapsed_time

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
