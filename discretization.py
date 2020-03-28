import timeit

import numpy as np
import scipy.sparse as sps

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

    def __init__(self, mp, mesh, n_ordinates, do_weights=0,
                 numerical_flux='upwind', quadrature='midpoint'):
        """
        Parameters
        ----------
        mp : modelProblem.ModelProblem1d
            Radiative Transfer problem to be discretized
        n_cells : integer
            Total number of cells used to subdivide the domain
        do_weights : tuple of length 2
            Weights for the quadrature of the discrete ordinates
        numerical_flux : string
            Numerical flux function used for the discretization of the
            transport terms.
        quadrature : string
            Quadrature method to be used in computation of matrix entries
        """

        # in one dimension there are only two possible discrete ordinates
        self.n_ord = n_ordinates if mp.dim == 2 else 2

        self.n_dof = self.n_ord * mesh.n_cells

        assert numerical_flux in ['upwind', 'centered'], \
            'Numerical flux ' + numerical_flux + ' not implemented.'

        assert quadrature in ['midpoint', 'trapezoidal'], \
            'Quadrature method ' + quadrature + ' not implemented.'

        print('Discretization:\n' +
              '---------------\n' +
              '    - number of discrete ordinates: ' + str(n_ordinates) +
              '\n    - number of degrees of freedom: ' + str(self.n_dof) +
              '\n    - quadrature method: ' + quadrature + ' rule\n' +
              '    - numerical flux: ' + numerical_flux +
              '\n\n')

        # --------------------------------------------------------------------
        #               DISCRETE ORDINATES AND SCATTERING
        # --------------------------------------------------------------------

        t0 = timeit.default_timer()

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

        t1 = timeit.default_timer() - t0
        print('scattering coefficients: ' + "% 10.3e" % (t1))
        # --------------------------------------------------------------------
        #               MESH GENERATION AND MATRIX ASSEMBLY
        # --------------------------------------------------------------------

        t0 = timeit.default_timer()
        alpha_tiled = np.tile(mesh.integrate_cellwise(mp.abs_fun, quadrature),
                              reps=self.n_ord)
        t1 = timeit.default_timer() - t0
        print('alpha: ' + "% 10.3e" % (t1))

        # coo-format matrices representing the contributions to the stiffness
        # matrix.
        t0 = timeit.default_timer()
        t_mat = self.__assemble_transport__(mesh, numerical_flux)
        t1 = timeit.default_timer() - t0
        print('transport: ' + "% 10.3e" % (t1))
        t0 = timeit.default_timer()
        a_mat = self.__assemble_absorption__(mesh, alpha_tiled, mp.xip1)
        t1 = timeit.default_timer() - t0
        print('absorption: ' + "% 10.3e" % (t1))

        if mp.scat == 'none':

            self.lambda_prec = None

            # Combine transport and absorption parts. By default
            # when converting to CSR or CSC format, duplicate
            # (i,j) entries will be summed together
            self.stiff_mat = sps.coo_matrix((
                np.concatenate((t_mat.data, a_mat.data)),
                (np.concatenate((t_mat.row, a_mat.row)),
                 np.concatenate((t_mat.col, a_mat.col)))),
                shape=(self.n_dof, self.n_dof)).tocsr()

        else:
            t0 = timeit.default_timer()
            s_mat = self.__assemble_scattering__(
                mesh, alpha_tiled[:mesh.n_cells], sig, mp.xi)
            t1 = timeit.default_timer() - t0
            print('scattering: ' + "% 10.3e" % (t1))

            t0 = timeit.default_timer()
            # Combine transport and absorption parts. By default
            # when converting to CSR or CSC format, duplicate
            # (i,j) entries will be summed together
            self.lambda_prec = sps.coo_matrix((
                np.concatenate((t_mat.data, mp.xip1 * a_mat.data)),
                (np.concatenate((t_mat.row, a_mat.row)),
                 np.concatenate((t_mat.col, a_mat.col)))),
                shape=(self.n_dof, self.n_dof))
            t1 = timeit.default_timer() - t0
            print('preconditioner: ' + "% 10.3e" % (t1))

            t0 = timeit.default_timer()
            self.stiff_mat = sps.coo_matrix((
                np.concatenate((self.lambda_prec.data, s_mat.data)),
                (np.concatenate((self.lambda_prec.row, s_mat.row)),
                 np.concatenate((self.lambda_prec.col, s_mat.col)))),
                shape=(self.n_dof, self.n_dof)).tocsr()

            self.lambda_prec = self.lambda_prec.tocsr()
            t1 = timeit.default_timer() - t0
            print('stiffness matrix: ' + "% 10.3e" % (t1))

        # --------------------------------------------------------------------
        #                       LOAD VECTOR ASSEMBLY
        # --------------------------------------------------------------------
        t0 = timeit.default_timer()
        self.load_vec = mp.emiss * mp.s_e * alpha_tiled

        # add boundary conditions
        for m in range(n_ordinates):
            self.load_vec[mesh.inflow_boundary_cells(m)] += \
                mp.inflow_bc[m]

        t1 = timeit.default_timer() - t0
        print('load vector: ' + "% 10.3e" % (t1) + '\n')

    def __assemble_transport__(self, mesh, num_flux):

        # For the transport part, there is no coupling between
        # the different ordinates. The corresponding matrix thus
        # has block diagonal structure.
        block_diag = []

        # numerical flux index function
        nfi_fun = upwind_index if num_flux == 'upwind' else centered_index

        for m in range(self.n_ord):

            # scalar product of ordinate direction m with Direction E
            n_prod = np.dot(mesh.outer_normal[Dir.E], self.ord_dir[m])

            def num_flux_index(p): return nfi_fun(n_prod > 0.0, p)

            if num_flux == 'upwind':
                num_flux_value = [n_prod]
            else:
                [0.5 * n_prod, 0.5 * n_prod]

            row = []
            col = []
            data = []

            # Loop over boundary cells
            for boundary in mesh.outer_normal:

                for p in mesh.boundary_cells(boundary):

                    if p == 0:

                        colIndex = num_flux_index(p)

                        col += colIndex
                        row += len(colIndex) * [p]
                        data += num_flux_value

                        if mesh.is_outflow_boundary_cell(p, m):

                            col += [p]
                            row += [p]
                            data += [-n_prod]

                    if p == mesh.n_cells - 1:

                        colIndex = num_flux_index(p - 1)

                        col += colIndex
                        row += len(colIndex) * [p]
                        data += [-value for value in num_flux_value]

                        if mesh.is_outflow_boundary_cell(p, m):

                            col += [p]
                            row += [p]
                            data += [n_prod]

            # Loop over interior cells
            for p in mesh.interior_cells():

                colIndex0 = num_flux_index(p)
                colIndex1 = num_flux_index(p - 1)
                col += colIndex0 + colIndex1

                row += len(colIndex0) * [p]
                row += len(colIndex1) * [p]

                data += num_flux_value
                data += [-value for value in num_flux_value]

            block_diag += [sps.coo_matrix((data, (row, col)),
                                          shape=(mesh.n_cells, mesh.n_cells))]

        return sps.block_diag(block_diag)

    def __assemble_absorption__(self, mesh, alpha_tiled, xip1):

        # for the pure absorption part, there is neither a coupling between
        # neighbouring cells nor between different ordinates, the the resulting
        # matrix is diagonal.
        return sps.diags(alpha_tiled, format='coo',
                         shape=(self.n_dof, self.n_dof))

    def __assemble_scattering__(self, mesh, alpha, sig, xi):

        # For the scattering part there is no coupling between
        # neighbouring cells, however there is coupling between
        # one ordinate and all others. Thus, the corresponding
        # matrix consists of blocks of diagonal matrices.
        blocks = []

        for m in range(self.n_ord):

            block_row = []

            for n in range(self.n_ord):
                block_row += [sps.diags(-xi * sig[m, n] * alpha, format='coo')]

            blocks += [block_row]

        return sps.bmat(blocks, format='coo')
