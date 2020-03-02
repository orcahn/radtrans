import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt

import modelProblem
import discretization

"""
For radiative transfer problems, it is beneficial
to use the \Lambda iteration as a preconditioner
"""
class LambdaPreconditioner:
    
    def __init__(self,discretization):
        self.disc = discretization
        self.M = spsla.LinearOperator((self.disc.n_dof, self.disc.n_dof), lambda x: spsla.spsolve(self.disc.lambda_prec, x))

"""
Class for the linear solvers.
It can be specified as a direct or iterative GMRES solver.
In case GMRES is selected, a preconditioner can be specified.
Additionally, one can supply an initial guess.
"""
class Solver:

    def __init__(self,name,preconditioner):
        self.name = name
        self.preconditioner = preconditioner
    
    def solve(self,A,b,x_in=0):
        if self.name == "SparseDirect":
            x = spsla.spsolve(A,b)
            return x
        elif self.name == "GMRES":
            if isinstance(self.preconditioner,LambdaPreconditioner):
                M = self.preconditioner.M
            x, exit_code = spsla.gmres(A=A,b=b,M=M,x0=x_in,tol=1e-8)
            print("GMRES ended with exit code " + str(exit_code))
            return x
