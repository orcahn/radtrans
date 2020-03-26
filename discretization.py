import numpy as np
import scipy.sparse as sps

import mesh
from mesh import Direction as Dir


def upwind_index(sameDirection, cell):

    # orthogonal direction does not matter because
    # corresponding entry is zero
    if sameDirection:

        return [cell]

    else:

        return [cell + 1]


def centered_index(sameDirection, cell):

    return [cell, cell + 1]


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


    Implementation Notes
    --------------------
    COO is a fast format for constructing sparse matrices. Once a matrix has
    been constructed, it will be converted to CSR format for fast arithmetic
    and matrix vector operations needed in the solvers.
    """

    def __init__(self, mp, n_cells, n_ordinates, do_weights=0,
                 numerical_flux='upwind'):
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

        assert numerical_flux in ['upwind', 'centered'], \
            'Numerical flux ' + numerical_flux + ' not implemented.'

        print('Discretization:\n' +
              '    - number of cells: ' + str(n_cells) + '\n' +
              '    - number of discrete ordinates: ' + str(n_ordinates) + '\n'
              '    - numerical flux: ' + numerical_flux +
              '\n\n\n')

        # --------------------------------------------------------------------
        #               DISCRETE ORDINATES AND SCATTERING
        # --------------------------------------------------------------------

        # in one dimension there are only two possible discrete ordinates
        self.n_ord = n_ordinates if mp.dim == 2 else 2

        # list of directions of the discrete ordinates
        self.ord_dir = [np.array([1.0]), np.array([-1.0])]

        if mp.dim == 1 and n_ordinates != 2:
            print('Warning: In one dimension two discrete ordinates' +
                  '(+1.0, -1.0) will be used!')

        # scattering coefficients for the chosen process
        sig = np.zeros((self.n_ord, self.n_ord))

        if mp.scat == 'isotropic':

            assert do_weights != 0, \
                'For isotropic scattering, quadrature weights for the ' + \
                'discrete ordinates must be provided.'

            assert len(do_weights) == n_ordinates, \
                'Number of quadrature weights provided and number of ' + \
                'discrete ordinates do not match.'

            # scattering probability for all discrete ordinates
            scat_prob = 1.0 / ((2.0 * np.pi)**(mp.dim - 1) * 2.0)

            for i in range(self.n_ord):
                for j in range(self.n_ord):
                    sig[i, j] = do_weights[i] * scat_prob

        # --------------------------------------------------------------------
        #               MESH GENERATION AND MATRIX ASSEMBLY
        # --------------------------------------------------------------------

        self.n_dof = self.n_ord * n_cells

        self.mesh = mesh.Mesh(mp.dom_len, n_cells)

        # TODO: do not store entire array but compute in place when needed
        self.alpha = self.mesh.integrate_cellwise(mp.abs_fun)

        # coo-format matrices representing the contributions to the stiffness
        # matrix. By default when converting to CSR or CSC format, duplicate
        # (i,j) entries will be summed together
        ta_mat = self.__assemble_transport__(numerical_flux)
        a_data = self.__assemble_absorption__(mp.xip1)
        s_data = self.__assemble_scattering__(sig, n_cells, mp.xi)

        ta_mat.data = np.concatenate((ta_mat.data, a_data[0]))
        ta_mat.row = np.concatenate((ta_mat.row, a_data[1]))
        ta_mat.col = np.concatenate((ta_mat.col, a_data[2]))

        if mp.scat == 'none':

            self.lambda_prec = sps.csr_matrix((self.n_dof, self.n_dof))
            self.stiff_mat = ta_mat.tocsr()

        else:

            # concatenate indices and data of transport and scattering part
            data = np.concatenate((ta_mat.data, s_data[0]))
            row = np.concatenate((ta_mat.row, s_data[1]))
            col = np.concatenate((ta_mat.col, s_data[2]))

            self.lambda_prec = ta_mat.tocsr()

            self.stiff_mat = sps.coo_matrix((data, (row, col)),
                                            shape=(self.n_dof, self.n_dof))

            self.stiff_mat = self.stiff_mat.tocsr()

        # --------------------------------------------------------------------
        #                       LOAD VECTOR ASSEMBLY
        # --------------------------------------------------------------------

        self.load_vec = np.tile(mp.emiss * mp.s_e * self.alpha, reps=2)

        # add boundary conditions
        for m in range(n_ordinates):
            self.load_vec[self.mesh.inflow_boundary(m, self.n_dof)] += \
                mp.inflow_bc[m]

    def __assemble_transport__(self, num_flux):

        # For the transport part, there is no coupling between
        # the different ordinates. The corresponding matrix thus
        # has block diagonal structure.
        block_diag = []

        nfi_fun = upwind_index if num_flux == 'upwind' else centered_index

        for m in range(self.n_ord):

            n_prod = np.dot(self.mesh.outer_normal[Dir.E], self.ord_dir[m])

            def num_flux_index(p): return nfi_fun(n_prod > 0.0, p)

            num_flux_value = n_prod if num_flux == 'upwind' else 0.5 * n_prod

            row = []
            col = []
            data = []

            for p in range(self.mesh.n_cells):

                # first, deal with the domain boudaries
                # then with the interior cells
                if p == 0:

                    colIndex = num_flux_index(p)

                    col += [*colIndex]
                    row += len(colIndex) * [p]
                    data += len(colIndex) * [num_flux_value]

                    if self.mesh.outflow_boundary_cell(p, m):

                        col += [p]
                        row += [p]
                        data += [-n_prod]

                elif p == self.mesh.n_cells - 1:

                    colIndex = num_flux_index(p - 1)

                    col += [*colIndex]
                    row += len(colIndex) * [p]
                    data += len(colIndex) * [-num_flux_value]

                    if self.mesh.outflow_boundary_cell(p, m):

                        col += [p]
                        row += [p]
                        data += [n_prod]

                else:

                    colIndex0 = num_flux_index(p)
                    colIndex1 = num_flux_index(p - 1)
                    col += [*colIndex0, *colIndex1]

                    row += len(colIndex0) * [p]
                    row += len(colIndex1) * [p]

                    data += len(colIndex0) * [num_flux_value]
                    data += len(colIndex1) * [-num_flux_value]

            block_diag += [sps.coo_matrix(
                (data, (row, col)),
                shape=(self.mesh.n_cells, self.mesh.n_cells))]

        return sps.block_diag(block_diag)

    def __assemble_absorption__(self, xip1):

        data = np.tile(xip1 * self.alpha, reps=2)
        row = np.arange(float(self.n_dof))
        col = row

        return data, row, col

    def __assemble_scattering__(self, sig, nc, xi):

        data = np.concatenate((-xi * sig[0, 0] * self.alpha,
                               -xi * sig[0, 1] * self.alpha,
                               -xi * sig[1, 0] * self.alpha,
                               -xi * sig[1, 1] * self.alpha))

        row = np.tile(np.arange(float(nc)), reps=self.n_ord * self.n_ord)

        col = np.concatenate((np.arange(float(nc)),
                              np.arange(float(nc)) + nc,
                              np.arange(float(nc)),
                              np.arange(float(nc)) + nc), axis=0)

        return data, row, col
