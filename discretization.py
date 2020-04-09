import timeit

import numpy as np
import scipy.sparse as sps

from mesh import Direction as Dir


def upwind_flux_minus(scalar_prod, h):

    if scalar_prod > 0.0:
        return -h * scalar_prod
    else:
        return 0.0


def upwind_flux_null(scalar_prod, h):

    if scalar_prod > 0.0:
        return h * scalar_prod
    else:
        return -h * scalar_prod


def upwind_flux_plus(scalar_prod, h):

    if scalar_prod > 0.0:
        return 0.0
    else:
        return h * scalar_prod


def centered_flux_minus(scalar_prod, h):

    return -0.5 * h * scalar_prod


def centered_flux_null(scalar_prod, h):

    return 0.0


def centered_flux_plus(scalar_prod, h):

    return 0.5 * h * scalar_prod


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
        mesh : mesh.UniformMesh
            Uniform mesh used for the FV discretization.
        n_ordinates : integer
            Number of discrete ordinates. The directions are chosen equidistant
            on the unit circle.
        do_weights : tuple of length 2
            Weights for the quadrature of the discrete ordinates
        numerical_flux : string
            Numerical flux function used in the discretization of the
            transport term.
        quadrature : string
            Quadrature method to be used in computation of matrix entries
        """

        # in one dimension there are only two possible discrete ordinates
        self.n_ord = n_ordinates if mesh.dim == 2 else 2

        self.n_dof = self.n_ord * mesh.n_tot

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
        ord_dir = [np.array([1.0]), np.array([-1.0])]

        if mesh.dim == 1 and n_ordinates != 2:
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
            scat_prob = 1.0 / ((2.0 * np.pi)**(mesh.dim - 1) * 2.0)

            for i in range(self.n_ord):
                for j in range(self.n_ord):
                    sig[i, j] = do_weights[i] * scat_prob

        t1 = timeit.default_timer() - t0
        print('scattering coefficients: ' + "% 10.3e" % (t1))

        # --------------------------------------------------------------------
        #                           MATRIX ASSEMBLY
        # --------------------------------------------------------------------

        t0 = timeit.default_timer()
        print(mesh.integrate_cellwise(mp.abs_fun, quadrature).shape)
        alpha_tiled = np.tile(mesh.integrate_cellwise(mp.abs_fun, quadrature),
                              reps=(self.n_ord, 1))

        t1 = timeit.default_timer() - t0
        print('alpha: ' + "% 10.3e" % (t1))

        # timing and assembly of the discretized transport term
        t0 = timeit.default_timer()
        t_mat = self.__assemble_transport__(mesh, ord_dir, numerical_flux)
        t1 = timeit.default_timer() - t0
        print('transport: ' + "% 10.3e" % (t1))

        # timing and assembly of the discretized absorption term
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

            # timing and assembly of the discretize scattering terms
            t0 = timeit.default_timer()
            s_mat = self.__assemble_scattering__(
                mesh, alpha_tiled[:mesh.n_cells[0]], sig, mp.xi)
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

    def __assemble_transport__(self, mesh, ord_dir, num_flux):

        # For the transport part, there is no coupling between
        # the different ordinates. The corresponding matrix thus
        # has block diagonal structure.
        block_diag = []

        flux_minus = None
        flux_null = None
        flux_plus = None

        if num_flux == 'upwind':

            flux_minus = upwind_flux_minus
            flux_null = upwind_flux_null
            flux_plus = upwind_flux_plus

        else:

            flux_minus = centered_flux_minus
            flux_null = centered_flux_null
            flux_plus = centered_flux_plus

        for m in range(self.n_ord):

            outflow_boundary = mesh.outflow_boundary()

            h_h = mesh.h[0]
            h_v = mesh.h[1]
            n_0 = mesh.n_cells[0]

            # scalar product of eastern outer normal with ordinate direction m
            n_prod_E = np.dot(mesh.outer_normal[Dir.E], ord_dir[m])
            n_prod_N = np.dot(mesh.outer_normal[Dir.N], ord_dir[m])
            n_prod_W = -n_prod_E
            n_prod_S = -n_prod_N

            row = []
            col = []
            data = []

            for i in mesh.interior_cells():

                # within one ordinate, for the transport part there is only
                # coupling between one cell and its direct neighbours in the
                # four directions. This translates to 5 nonzero entries.
                row += 5 * [i]
                col += [i, i-1, i+1, i-n_0, i+n_0]
                data += [flux_null(n_prod_E, h_v) +
                         flux_null(n_prod_N, mesh.h[0]),
                         flux_minus(n_prod_E, h_v),
                         flux_plus(n_prod_E, h_v),
                         flux_minus(n_prod_N, h_h),
                         flux_plus(n_prod_N, h_h)]

            # the boundary cells exclude cells in the corners. These are
            # treated separately below
            for i in mesh.outflow_boundary_cells(Dir.E):

                row += 4 * [i]
                col += [i, i-1, i-n_0, i+n_0]
                data += [h_v * n_prod_E -
                         flux_plus(n_prod_E, h_v) +
                         flux_null(n_prod_N, h_h),
                         flux_minus(n_prod_E, h_v),
                         flux_minus(n_prod_N, h_h),
                         flux_plus(n_prod_N, h_h)]

            for i in mesh.outflow_boundary_cells(Dir.N):

                row += 4 * [i]
                col += [i, i-1, i+1, i-n_0]
                data += [h_h * n_prod_N -
                         flux_plus(n_prod_N, h_h) +
                         flux_null(n_prod_E, h_v),
                         flux_minus(n_prod_E, h_v),
                         flux_plus(n_prod_E, h_v),
                         flux_minus(n_prod_N, h_h)]

            for i in mesh.outflow_boundary_cells(Dir.W):

                row += 4 * [i]
                col += [i, i+1, i-n_0, i+n_0]
                data += [h_v * n_prod_W -
                         flux_minus(n_prod_E, h_v) +
                         flux_null(n_prod_N, h_h),
                         flux_plus(n_prod_E, h_v),
                         flux_minus(n_prod_N, h_h),
                         flux_plus(n_prod_N, h_h)]

            for i in mesh.outflow_boundary_cells(Dir.S):

                row += 4 * [i]
                col += [i, i-1, i+1, i+n_0]
                data += [h_h * n_prod_S -
                         flux_minus(n_prod_N, h_h) +
                         flux_null(n_prod_E, h_v),
                         flux_minus(n_prod_E, h_v),
                         flux_plus(n_prod_E, h_v),
                         flux_plus(n_prod_N, h_h)]

            # # treat the corner cells separately
            # sw_index = mesh.south_west_corner()

            # row += 2 * [sw_index]
            # col += [sw_index+1, sw_index+n_0]
            # data += [flux_plus(n_prod_E, h_v),
            #          flux_plus(n_prod_N, h_h)]

            # if Dir.W in outflow_boundary():

            #     row += [sw_index]
            #     col += [sw_index]
            #     data += [h_v * n_prod_E - flux_minus(n_prod_E, h_v)]

            # if Dir.S in outflow_boundary():

            #     row += [sw_index]
            #     col += [sw_index]
            #     data += [h_h * n_prod_E - flux_minus(n_prod_E, h_v)]

            # se_index = mesh.south_east_corner()

            # row += 2 * [se_index]
            # col += [se_index-1, se_index+n_0]
            # data += [flux_minus(n_prod_E, h_v),
            #          flux_plus(n_prod_N, h_h)]

            # if Dir.S in outflow_boundary():

            #     row += [se_index]
            #     col += [se_index]
            #     data += []

            # if Dir.E in outflow_boundary():
            #     pass

            # ne_index = mesh.north_east_corner()
            # if Dir.N in outflow_boundary():
            #     pass
            # if Dir.E in outflow_boundary():
            #     pass

            # nw_index = mesh.north_west_corner()
            # if Dir.N in outflow_boundary():
            #     pass
            # if Dir.W in outflow_boundary():
            #     pass

            # TODO: define sparse matrix and remove zeros
            block_diag += [sps.coo_matrix(
                (data, (row, col)),
                shape=(n_0, n_0))]

        return sps.block_diag(block_diag)

    def __assemble_absorption__(self, mesh, alpha_tiled, xip1):

        # for the pure absorption part, there is neither a coupling between
        # neighbouring cells nor between different ordinates, the the resulting
        # matrix is diagonal.
        print(alpha_tiled.shape)
        return sps.diags(np.ravel(alpha_tiled), format='coo',
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
                block_row += [sps.diags(-xi * sig[m, n]
                                        * np.ravel(alpha), format='coo')]

            blocks += [block_row]

        return sps.bmat(blocks, format='coo')
