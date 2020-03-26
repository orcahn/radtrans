import time
import scipy.sparse.linalg as spsla


class solve_counter(object):
    """
    Simple class that counts the number of
    iterations within a scipy iterative solver.
    """

    def __init__(self):
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1


class LambdaPreconditioner:
    """
    For radiative transfer problems, it is beneficial
    to use the Lambda iteration as a preconditioner
    """

    def __init__(self, discretization):

        self.disc = discretization

        self.M = spsla.LinearOperator(
            (self.disc.n_dof, self.disc.n_dof), lambda x: spsla.spsolve(
                self.disc.lambda_prec, x))


class Solver:
    """
    Class for the linear solvers.
    It can be specified as a direct or iterative GMRES solver.
    In case GMRES or BiCGSTAB is selected, a preconditioner can
    be specified. Additionally, one can supply an initial guess.
    """

    def __init__(self, name, preconditioner):

        self.name = name
        self.preconditioner = preconditioner

    def solve(self, A, b, x_in=None):

        if self.name == "SparseDirect":

            t = time.process_time()

            x = spsla.spsolve(A, b)
            elapsed_time = time.process_time() - t

            print("Sparse direct solver ended after " + str(elapsed_time) + "s")

            return x, None, elapsed_time

        elif self.name == "GMRES":

            if isinstance(self.preconditioner, LambdaPreconditioner):

                M = self.preconditioner.M

            else:

                M = None

            counter = solve_counter()
            t = time.process_time()

            x, exit_code = spsla.gmres(
                A=A, b=b, M=M, x0=x_in, callback=counter, tol=1e-8)

            elapsed_time = time.process_time() - t

            print("GMRES ended with exit code " + str(exit_code)+" after " +
                  str(counter.niter)+" iterations in "+str(elapsed_time)+"s")

            return x, counter.niter, elapsed_time

        elif self.name == "BiCGSTAB":

            if isinstance(self.preconditioner, LambdaPreconditioner):

                M = self.preconditioner.M

            else:

                M = None

            counter = solve_counter()
            t = time.process_time()

            x, exit_code = spsla.bicgstab(
                A=A, b=b, M=M, x0=x_in, callback=counter, tol=1e-8)

            elapsed_time = time.process_time() - t

            print("BiCGSTAB ended with exit code " + str(exit_code)+" after " +
                  str(counter.niter)+" iterations in "+str(elapsed_time)+"s")

            return x, counter.niter, elapsed_time

        elif self.name == "CG":

            if isinstance(self.preconditioner, LambdaPreconditioner):

                M = self.preconditioner.M

            else:

                M = None

            counter = solve_counter()
            t = time.process_time()

            x, exit_code = spsla.cg(
                A=A, b=b, M=M, x0=x_in, callback=counter, tol=1e-8)

            elapsed_time = time.process_time() - t

            print("CG ended with exit code " + str(exit_code)+" after " +
                  str(counter.niter)+" iterations in "+str(elapsed_time)+"s")

            return x, counter.niter, elapsed_time

