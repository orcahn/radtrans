import numpy as np
import scipy.sparse as sps

from scipy.integrate import fixed_quad


class FiniteVolume1d:
    """
    This class represents a one-dimensional finite-volume discretization of
    the continuous model equations.

    Attributes
    ------
    n_dof : integer
        Total number of degrees of freedom in the complete system.
    n_ord : integer
        Total number of discrete ordinates.
    h : float
        Length of a single cell.
    mesh : one-dimensional np.ndarray
        Uniform mesh used for the FV discretization.
    alpha : one-dimensional np.ndarray
        Array with same number of entries as there are cells. The entries
        consist of the L2 scalar product of the absorption coefficient with
        the basis function corresponding to the cell in the mesh.
    stiff_mat : scipy.sparse.csr.csr_matrix
        Sparse stiffness matrix of the system.
    lambda_prec : scipy.sparse.csr.csr_matrix
        Explicit sparse representation of the linear preconditioner used in
        the lambda iteration.
    load_vec : np.ndarray
        Dense load vector of the system.
    """

    def __init__(self, mp, n_cells, do_weights=0):
        """
        Parameters
        ----------
        mp : modelProblem.ModelProblem1d
            Radiative Transfer problem to be discretized
        n_cells : integer
            Total number of cells used to subdivide the domain
        do_weights : tuple of length 2
            Weights for the quadrature of the discrete ordinates
        """

        self.n_ord = 2
        self.n_dof = self.n_ord * n_cells
        self.mesh, self.h = np.linspace(
            0.0, mp.dom_len, num=n_cells + 1, endpoint=True, retstep=True)

        self.stiff_mat = sps.csr_matrix((self.n_dof, self.n_dof))

        # scattering coefficients for the chosen process
        sig = np.zeros((self.n_ord, self.n_ord))

        if mp.scat == 'isotropic':

            assert do_weights != 0, \
                'For isotropic scattering, quadrature weights for the ' + \
                'discrete ordinates must be provided.'

            assert len(do_weights) == 2, \
                'Number of quadrature weights provided and number of ' + \
                'discrete ordinates do not match.'

            # scattering probability for all discrete ordinates
            scat_prob = 1.0 / float(self.n_ord)

            for i in range(self.n_ord):
                for j in range(self.n_ord):
                    sig[i, j] = do_weights[i] * scat_prob

        # Compute the L2 scalar product of the absorption coefficient with the
        # basis functions corresponding to each cell. Order 4 gaussian
        # quadrature is used, which translates to 2 quadrature nodes per cell.
        self.alpha = np.array([fixed_quad(
            mp.abs_fun, self.mesh[i], self.mesh[i+1], n=4)[0]
            for i in range(n_cells)])

        # diagonals of the transport and absorption part of the
        # complete FV stiffness matrix
        ta_main = np.array([1.0 + mp.xip1 * self.alpha[k]
                            for k in range(n_cells)])

        ta_off = np.full(n_cells - 1, -1.0)

        ta_diag_blocks = []

        for m in range(self.n_ord):
            ta_diag_blocks += [sps.diags([ta_main, ta_off],
                                         [0, (-1)**(m+1)], format='csr')]

        # explicit representation of preconditioner used in the
        # lambda iteration
        lambda_prec = sps.block_diag(ta_diag_blocks, format='csr')

        if mp.scat == 'none':

            self.lambda_prec = sps.csr_matrix((self.n_dof, self.n_dof))
            self.stiff_mat = lambda_prec

        else:

            # block diagonals occuring in the scattering part s_mat of the
            # complete FV stiffness matrix
            s_diags = []

            for i in range(self.n_ord):

                block_row = []

                for j in range(self.n_ord):

                    block_row_diag = np.array(
                        [-sig[i, j] * mp.xi * self.alpha[k]
                         for k in range(n_cells)])
                    block_row += [sps.diags([block_row_diag],
                                            [0], format='csr')]

                s_diags += [block_row]

            s_mat = sps.bmat(s_diags, format='csr')

            self.lambda_prec = lambda_prec
            self.stiff_mat = lambda_prec + s_mat

        self.load_vec = np.full(self.n_dof, mp.s_eps * self.h)

        self.load_vec[0] += mp.inflow_bc[0]
        self.load_vec[-1] += mp.inflow_bc[1]
