import numpy as np
from scipy.sparse import diags


class FVDiscretization:
    """
    This class represents a one-dimensional finite-volume discretization of
    the continuous model equations.

    Attributes
    ------
    N_dof : integer
        The total number of cells used to subdivide the domain
    h : float
        The length of a single cell
    A : scipy.sparse.csc.csc_matrix
        The sparse stiffness matrix of the system
    b : np.ndarray
        The dense load vector of the system
    """

    def __init__(self, model_problem, N_dof):
        """
        Parameter
        ---------
        model_problem : ModelProblem.ModelProblem
            The model problem to be discretized
        N_dof : integer
            The total number of cells used to subdivide the domain
        """

        self.N_dof = N_dof
        self.h = model_problem.domain_length / N_dof

        main_diag = np.full(N_dof, 1.0 + self.h)
        off_diag = np.full(N_dof-1, -1.0)

        # Use the csc format as this is later required for the solvers
        self.A = diags([main_diag, off_diag], [0, -1], format='csc')

        self.b = np.full(N_dof, model_problem.s_eps * self.h)
        self.b[0] += model_problem.inflow_bc
