import numpy as np
import scipy.sparse as sps

from mesh import Direction as Dir

# -----------------------------------------------------------------------------
#                           NUMERICAL FLUXES
# -----------------------------------------------------------------------------

# The possible numerical fluxes used in the discretization of the transport
# term. When computing entry T_{ij} we need the value of the numerical flux
# function on the cell in negative direction of the face-normal we consider
# (""_flux_minus), as well as the values on the same cell (""_flux_null) and
# the cell in positive normal direction (""_flux_plus).


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


# -----------------------------------------------------------------------------
#                       FINITE VOLUME DISCRETIZATION
# -----------------------------------------------------------------------------

class FiniteVolume:
    """
    This class represents a finite-volume discretization of the continuous
    model equations in one or two dimensions.

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
    load_vec : numpy.ndarray
        Load vector of the system.

    Methods
    -------
    __compute_scalar_product__(outer_normals, ord_dir)
        Computes the scalar product between all discrete ordinates directions
        and the outer normals of the domain.
    __assemble_transport__(mesh, n_dot_n, ord_dir, num_flux)
        Assembles the part of the stiffness matrix corresponding to the
        transport term.
    __assemble_absorption__(xip1)
        Assembles the part of the stiffness matrix corresponding to the
        absorption term.
    __assemble_scattering__(sig, xi)
        Assembles the part of the stiffness matrix corresponding to the
        scattering term.
    __assemble_source__(mp, mesh, n_dot_n)
        Assembles the load vector corresponding to the source terms.
    __assemble_diffusion__(smesh, abs_f, scattering, xip1, n_boundaries)
        Assembles the matrix for the diffusion limit.
    __assemble_dirichlet__(self, mesh, abs_f, xip1, n_boundaries)
        Assembles the dirichlet boundary conditions for the diffusion
        limit.


    Implementation Notes
    --------------------
    COO is a fast format for constructing sparse matrices. Once a matrix has
    been constructed, it will be converted to CSR format for fast arithmetic
    and matrix vector operations needed in the solvers.
    """

    def __init__(self, mp, mesh, n_ordinates, inflow_bc,
                 numerical_flux='upwind', quadrature='midpoint', output=True):
        """
        Parameters
        ----------
        mp : modelProblem.ModelProblem
            Radiative Transfer problem to be discretized
        mesh : mesh.UniformMesh
            Uniform mesh used for the FV discretization.
        n_ordinates : integer
            Number of discrete ordinates. The directions are chosen equidistant
            on the unit circle.
        inflow_bc : list of floats
            The inflow boundary conditions for each discrete ordinate
        numerical_flux : string
            Numerical flux function to be used in the discretization of the
            transport term.
        quadrature : string
            Quadrature method to be used in computation of matrix entries
        output : bool
            Whether to print the relevant quantities of the discretization
        """

        self.n_ord = n_ordinates
        self.n_dof = self.n_ord * mesh.n_tot
        self.inflow_bc = inflow_bc

        if mesh.dim == 1:
            assert len(inflow_bc) == 2, \
                'Invalid inflow boundary conditions. ' + \
                'For a 1d problem there must be exactly 2 conditions.'

        assert numerical_flux in ['upwind', 'centered', 'diffusion'], \
            'Numerical flux ' + numerical_flux + ' not implemented.'

        assert quadrature in ['midpoint', 'trapezoidal'], \
            'Quadrature method ' + quadrature + ' not implemented.'

        # ---------------------------------------------------------------------
        #               DISCRETE ORDINATES AND SCATTERING
        # ---------------------------------------------------------------------

        # List of directions of the discrete ordinates
        if mesh.dim == 1:
            ord_dir = [np.array([1.0]), np.array([-1.0])]

        else:

            # 'Equidistant' on the unit circle
            piM = 2.0 * np.pi / float(n_ordinates)

            ord_dir = [np.array([np.cos(m * piM), np.sin(m * piM)])
                       for m in np.arange(n_ordinates, dtype=np.single)]

        n_dot_n = self.__compute_scalar_product__(mesh.outer_normal, ord_dir)

        if numerical_flux == 'diffusion':

            self.n_dof = mesh.n_cells[0]*mesh.n_cells[1]
            self.inflow_bc = inflow_bc

            n_diri = np.count_nonzero(self.inflow_bc)

            if mesh.dim == 2:

                # If n_diri == 1, just set the eastern boundary
                if n_diri != 1:
                    n_diri = 4

            if output:

                print('Discretization:\n' +
                      '---------------\n' +
                      '\n    - number of degrees of freedom: ' +
                      str(self.n_dof) +
                      '\n    - quadrature method: ' + quadrature + ' rule\n' +
                      '    - numerical flux: ' + numerical_flux +
                      '\n\n')

            # --------------------------------------------------------------------
            #                           MATRIX ASSEMBLY
            # --------------------------------------------------------------------

            # Assembly of the discretized diffusion term
            d_mat = self.__assemble_diffusion__(
                mesh, mp.abs_fun, mp.scat, mp.xip1, n_diri)

            # Assembly of the discretized absorption term.
            # In the diffusion case, it does not contain the
            # scattering coefficient, unlike the transport case.
            alpha = mesh.integrate_cellwise(mp.abs_fun, quadrature)
            a_mat = self.__assemble_absorption__(alpha, 1.0)

            if mp.scat == 'none':

                self.lambda_prec = sps.identity(self.n_dof)

                # Combine diffusion and absorption parts. By default
                # when converting to CSR or CSC format, duplicate
                # (i,j) entries will be summed together
                self.stiff_mat = sps.coo_matrix((
                    np.concatenate((d_mat.data, a_mat.data)),
                    (np.concatenate((d_mat.row, a_mat.row)),
                     np.concatenate((d_mat.col, a_mat.col)))),
                    shape=(self.n_dof, self.n_dof)).tocsr()

            else:

                # Combine diffusion and absorption parts. By default
                # when converting to CSR or CSC format, duplicate
                # (i,j) entries will be summed together
                self.lambda_prec = sps.coo_matrix((
                    np.concatenate((d_mat.data, a_mat.data)),
                    (np.concatenate((d_mat.row, a_mat.row)),
                     np.concatenate((d_mat.col, a_mat.col)))),
                    shape=(self.n_dof, self.n_dof)).tocsr()

                self.stiff_mat = self.lambda_prec
            # --------------------------------------------------------------------
            #                       LOAD VECTOR ASSEMBLY
            # --------------------------------------------------------------------

            # Assemble the vector of Dirichlet boundary conditions
            diri_bc = self.__assemble_dirichlet__(
                mesh, mp.abs_fun, mp.xip1, n_diri)

            # Assemble the load vector
            self.load_vec = mp.emiss * mp.s_e * alpha + diri_bc

        else:

            if output:

                print('Discretization:\n' +
                      '---------------\n' +
                      '    - number of discrete ordinates: ' +
                      str(n_ordinates) +
                      '\n    - number of degrees of freedom: ' +
                      str(self.n_dof) +
                      '\n    - quadrature method: ' + quadrature + ' rule\n' +
                      '    - numerical flux: ' + numerical_flux +
                      '\n\n')

            # Scattering coefficients for the chosen process
            sig = np.empty((self.n_ord, self.n_ord))
            do_weights = np.zeros(n_ordinates)
            scat_prob = np.zeros(n_ordinates)

            if mp.scat == 'isotropic':

                # Ordinate weights could in principle be chosen
                # arbitrarily. However we chose them such, that
                # energy conservation is satisfied
                if mesh.dim == 1:
                    do_weights = np.array([1.0, 1.0])
                    scat_prob = np.array([0.5, 0.5])

                else:
                    do_weights = np.full(n_ordinates,
                                         2.0 * np.pi / float(n_ordinates))
                    scat_prob = np.full(n_ordinates, 0.5 / np.pi)

            for i in range(self.n_ord):
                for j in range(self.n_ord):
                    sig[i, j] = do_weights[i] * scat_prob[j]

            # --------------------------------------------------------------------
            #                           ASSEMBLY
            # --------------------------------------------------------------------

            alpha = mesh.integrate_cellwise(mp.abs_fun, quadrature)

            # Assembly of the discretized transport term
            t_mat = self.__assemble_transport__(
                mesh, n_dot_n, ord_dir, numerical_flux)

            # Assembly of the discretized absorption term
            a_mat = self.__assemble_absorption__(alpha, mp.xip1)

            if mp.scat == 'none':

                self.lambda_prec = sps.identity(self.n_dof)

                # Combine transport and absorption parts. By default
                # when converting to CSR or CSC format, duplicate
                # (i,j) entries will be summed together
                self.stiff_mat = sps.coo_matrix((
                    np.concatenate((t_mat.data, a_mat.data)),
                    (np.concatenate((t_mat.row, a_mat.row)),
                     np.concatenate((t_mat.col, a_mat.col)))),
                    shape=(self.n_dof, self.n_dof)).tocsr()

            else:

                # Assembly of the discretized scattering terms
                s_mat = self.__assemble_scattering__(alpha, sig, mp.xi)

                # Combine transport and absorption parts. By default
                # when converting to CSR or CSC format, duplicate
                # (i,j) entries will be summed together
                self.lambda_prec = sps.coo_matrix((
                    np.concatenate((t_mat.data, a_mat.data)),
                    (np.concatenate((t_mat.row, a_mat.row)),
                     np.concatenate((t_mat.col, a_mat.col)))),
                    shape=(self.n_dof, self.n_dof))

                self.stiff_mat = sps.coo_matrix((
                    np.concatenate((self.lambda_prec.data, s_mat.data)),
                    (np.concatenate((self.lambda_prec.row, s_mat.row)),
                     np.concatenate((self.lambda_prec.col, s_mat.col)))),
                    shape=(self.n_dof, self.n_dof)).tocsr()

                self.lambda_prec = self.lambda_prec.tocsr()

            self.load_vec = self.__assemble_source__(mp, mesh, alpha, n_dot_n)

    def __compute_scalar_product__(self, outer_normals, ord_dir):
        """
        Computes the scalar product between all discrete ordinate directions
        and all outer normals. Accounts for errors is floating point
        representation.

        Parameters
        ----------
        outer_normals : <mesh.Direction : numpy.ndarray> dictionary
            The four outer normals of the rectangular domain
        ord_dir : list of numpy.ndarray
            List of all the discrete ordinate directions

        Returns
        -------
        <mesh.Direction : list of floats> dictionary
            A dictionary with the directions as keys and a list of the scalar
            products of all the discrete ordinate directions with the outer
            normal in this direction as values.
        """

        nn = {Dir.E: [], Dir.N: [], Dir.W: [], Dir.S: []}

        for d, nn_list in nn.items():
            for nm in ord_dir:

                prod = np.dot(outer_normals[d], nm)

                # Even for 1e6 ordinate directions, the smallest absolute value
                # of a scalar product between ordinate and normal, is on the
                # order 1e-6. Values smaller than 1e-12 are thus guaranteed to
                # stem from errors in floating point calculations.
                if np.abs(prod) < 1e-12:
                    prod = 0.0

                nn_list += [prod]

        return nn

    def __assemble_transport__(self, mesh, n_dot_n, ord_dir, num_flux):
        """
        Assembles the matrix corresponding to the transport term in sparse
        coordinate format.

        Parameters
        ----------
        mesh : mesh.UniformMesh
            Uniform mesh used for the discretization.
        n_dot_n : <mesh.Direction : list of floats> dictionary
            Dictionary containing all the relevant scala products, as returned
            by __compute_scalar_products__.
        ord_dir : list of numpy.ndarray
            List of all the discrete ordinate directions.
        num_flux : string
            Numerical flux function to be used in discretization

        Returns
        -------
        scipy.sparse.coo_matrix

        Sparse matrix representing the discretized transport term in coordinate
        format.
        """

        h_h = mesh.h[0]
        h_v = mesh.h[1]
        v_offset = mesh.n_cells[0] if mesh.dim == 2 else 0

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

            out_bndry = []

            for d, n_list in n_dot_n.items():
                if n_list[m] > 0.0:
                    out_bndry += [d]

            # Scalar product of outer normals with ordinate direction m
            n_prod_E = n_dot_n[Dir.E][m]
            n_prod_N = n_dot_n[Dir.N][m]
            n_prod_W = -n_prod_E
            n_prod_S = -n_prod_N

            row = []
            col = []
            data = []

            for i in mesh.interior_cells():

                # Within one ordinate, for the transport part there is only
                # coupling between one cell and its direct neighbours in the
                # four directions. This translates to 5 nonzero entries.
                row += 5 * [i]
                col += [i, i-1, i+1, i-v_offset, i+v_offset]
                data += [flux_null(n_prod_E, h_v) +
                         flux_null(n_prod_N, h_h),
                         flux_minus(n_prod_E, h_v),
                         flux_plus(n_prod_E, h_v),
                         flux_minus(n_prod_N, h_h),
                         flux_plus(n_prod_N, h_h)]

            # Treat the corner cells separately
            if mesh.dim == 2:

                sw_index = mesh.south_west_corner()

                row += 3 * [sw_index]
                col += [sw_index, sw_index+1, sw_index+v_offset]
                data += [-flux_minus(n_prod_E, h_v) -
                         flux_minus(n_prod_N, h_h),
                         flux_plus(n_prod_E, h_v),
                         flux_plus(n_prod_N, h_h)]

                se_index = mesh.south_east_corner()

                row += 3 * [se_index]
                col += [se_index, se_index-1, se_index+v_offset]
                data += [-flux_plus(n_prod_E, h_v) -
                         flux_minus(n_prod_N, h_h),
                         flux_minus(n_prod_E, h_v),
                         flux_plus(n_prod_N, h_h)]

                ne_index = mesh.north_east_corner()

                row += 3 * [ne_index]
                col += [ne_index, ne_index-1, ne_index-v_offset]
                data += [-flux_plus(n_prod_E, h_v) -
                         flux_plus(n_prod_N, h_h),
                         flux_minus(n_prod_E, h_v),
                         flux_minus(n_prod_N, h_h)]

                nw_index = mesh.north_west_corner()

                row += 3 * [nw_index]
                col += [nw_index, nw_index+1, nw_index-v_offset]
                data += [-flux_minus(n_prod_E, h_v) -
                         flux_plus(n_prod_N, h_h),
                         flux_plus(n_prod_E, h_v),
                         flux_minus(n_prod_N, h_h)]

                if Dir.E in out_bndry:

                    row += [se_index, ne_index]
                    col += [se_index, ne_index]
                    data += 2 * [h_v * n_prod_E]

                elif Dir.W in out_bndry:

                    row += [sw_index, nw_index]
                    col += [sw_index, nw_index]
                    data += 2 * [h_v * n_prod_W]

                if Dir.N in out_bndry:

                    row += [ne_index, nw_index]
                    col += [ne_index, nw_index]
                    data += 2 * [h_h * n_prod_N]

                elif Dir.S in out_bndry:

                    row += [se_index, sw_index]
                    col += [se_index, sw_index]
                    data += 2 * [h_h * n_prod_S]

            # The boundary cells excluding cells in the corners
            for i in mesh.boundary_cells(Dir.E):

                row += 4 * [i]
                col += [i, i-1, i-v_offset, i+v_offset]
                data += [-flux_plus(n_prod_E, h_v) +
                         flux_null(n_prod_N, h_h),
                         flux_minus(n_prod_E, h_v),
                         flux_minus(n_prod_N, h_h),
                         flux_plus(n_prod_N, h_h)]

                if Dir.E in out_bndry:

                    if mesh.dim == 1:

                        row += [mesh.n_cells[0] - 1]
                        col += [mesh.n_cells[0] - 1]
                        data += [h_v * n_prod_E]

                    else:

                        row += [i]
                        col += [i]
                        data += [h_v * n_prod_E]

            for i in mesh.boundary_cells(Dir.N):

                row += 4 * [i]
                col += [i, i-1, i+1, i-v_offset]
                data += [-flux_plus(n_prod_N, h_h) +
                         flux_null(n_prod_E, h_v),
                         flux_minus(n_prod_E, h_v),
                         flux_plus(n_prod_E, h_v),
                         flux_minus(n_prod_N, h_h)]

                if Dir.N in out_bndry:

                    if mesh.dim == 2:

                        row += [i]
                        col += [i]
                        data += [h_h * n_prod_N]

            for i in mesh.boundary_cells(Dir.W):

                row += 4 * [i]
                col += [i, i+1, i-v_offset, i+v_offset]
                data += [-flux_minus(n_prod_E, h_v) +
                         flux_null(n_prod_N, h_h),
                         flux_plus(n_prod_E, h_v),
                         flux_minus(n_prod_N, h_h),
                         flux_plus(n_prod_N, h_h)]

                if Dir.W in out_bndry:

                    if mesh.dim == 1:

                        row += [0]
                        col += [0]
                        data += [h_v * n_prod_W]

                    else:

                        row += [i]
                        col += [i]
                        data += [h_v * n_prod_W]

            for i in mesh.boundary_cells(Dir.S):

                row += 4 * [i]
                col += [i, i-1, i+1, i+v_offset]
                data += [-flux_minus(n_prod_N, h_h) +
                         flux_null(n_prod_E, h_v),
                         flux_minus(n_prod_E, h_v),
                         flux_plus(n_prod_E, h_v),
                         flux_plus(n_prod_N, h_h)]

                if Dir.S in out_bndry:

                    if mesh.dim == 2:

                        row += [i]
                        col += [i]
                        data += [h_h * n_prod_S]

            block_diag += [sps.coo_matrix((data, (row, col)),
                                          shape=(mesh.n_tot, mesh.n_tot))]

        return sps.block_diag(block_diag)

    def __assemble_absorption__(self, alpha, xip1):
        """
        Assembles the matrix corresponding to the absorption term in sparse
        coordinate format.

        Parameters
        ----------
        alpha : one dimensional numpy.ndarray
            The integrals of the absorption coefficient over the cells of the
            mesh.
        xip1 : float
            The ratio between scattering coefficient and absorption coefficient
            plus 1.0.

        Returns
        -------
        scipy.sparse.coo_matrix

        Sparse matrix representing the discretized absorption term in
        coordinate format.
        """

        alpha_tiled = xip1 * \
            np.ravel(np.tile(alpha, reps=(1, self.n_ord)))

        # For the pure absorption part, there is neither a coupling between
        # neighbouring cells nor between different ordinates. The resulting
        # matrix is diagonal.
        return sps.diags(alpha_tiled, format='coo',
                         shape=(self.n_dof, self.n_dof))

    def __assemble_scattering__(self, alpha, sig, xi):
        """
        Assembles the matrix corresponding to the scattering term in sparse
        coordinate format.

        Parameters
        ----------
        alpha : one dimensional numpy.ndarray
        The integrals of the absorption coefficient over the cells of the
        mesh.
        xip1 : float
            The ratio between scattering coefficient and absorption coefficient
            plus 1.0.

        Returns
        -------
        scipy.sparse.coo_matrix

        Sparse matrix representing the discretized scattering term in
        coordinate format.
        """

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

    def __assemble_source__(self, mp, mesh, alpha, n_dot_n):
        """
        Assembles the load vector corresponding to the source terms.

        Parameters
        ----------
        mp : modelProblem.ModelProblem
            Radiative Transfer problem to be discretized
        mesh : mesh.UniformMesh
            Uniform mesh used for the FV discretization.
        alpha : one dimensional numpy.ndarray
            The integrals of the absorption coefficient over the cells of the
            mesh.
        n_dot_n : <mesh.Direction : list of floats> dictionary
            Dictionary containing all the relevant scala products, as returned
            by __compute_scalar_products__.

        Returns
        -------
        numpy.ndarray
            Load vector of the system
        """

        alpha_tiled = np.ravel(np.tile(alpha, reps=(1, self.n_ord)))
        load_vec = mp.emiss * mp.s_e * alpha_tiled

        # Add boundary conditions
        for m in range(self.n_ord):

            in_bndry = []

            for d, n_list in n_dot_n.items():
                if n_list[m] < 0.0:
                    in_bndry += [d]

            offset = m * mesh.n_tot

            if mesh.dim == 1:

                for d in in_bndry:
                    load_vec[offset + mesh.boundary_cells(d)] -= \
                        mesh.h[1] * n_dot_n[d][m] * self.inflow_bc[m]

            else:

                if Dir.E in in_bndry:

                    load_vec[offset + mesh.boundary_cells(Dir.E)] -= \
                        mesh.h[1] * n_dot_n[Dir.E][m] * self.inflow_bc[m]

                    load_vec[[offset + mesh.south_east_corner(),
                              offset + mesh.north_east_corner()]] -= \
                        mesh.h[1] * n_dot_n[Dir.E][m] * self.inflow_bc[m]

                if Dir.W in in_bndry:

                    load_vec[offset + mesh.boundary_cells(Dir.W)] -= \
                        mesh.h[1] * n_dot_n[Dir.W][m] * self.inflow_bc[m]

                    load_vec[[offset + mesh.south_west_corner(),
                              offset + mesh.north_west_corner()]] -= \
                        mesh.h[1] * n_dot_n[Dir.W][m] * self.inflow_bc[m]

                if Dir.N in in_bndry:

                    load_vec[offset + mesh.boundary_cells(Dir.N)] -= \
                        mesh.h[0] * n_dot_n[Dir.N][m] * self.inflow_bc[m]

                    load_vec[[offset + mesh.north_west_corner(),
                              offset + mesh.north_east_corner()]] -= \
                        mesh.h[0] * n_dot_n[Dir.N][m] * self.inflow_bc[m]

                if Dir.S in in_bndry:

                    load_vec[offset + mesh.boundary_cells(Dir.S)] -= \
                        mesh.h[0] * n_dot_n[Dir.S][m] * self.inflow_bc[m]

                    load_vec[[offset + mesh.south_west_corner(),
                              offset + mesh.south_east_corner()]] -= \
                        mesh.h[0] * n_dot_n[Dir.S][m] * self.inflow_bc[m]

        return load_vec

    def __assemble_diffusion__(self, mesh, abs_f, scattering, xip1,
                               n_boundaries):
        """
        Assembles the diffusion matrix.

        Parameters
        ----------
        mesh : mesh.UniformMesh
            Uniform mesh used for the discretization.
        abs_f : callable
        Callable object, representing the absorption coefficient.
        Takes values in the domain as argument.
        scattering : string
        Type of scattering process assumed for the medium
        xip1 : float
        The ratio between scattering coefficient and absorption
        coefficient plus 1.0.
        alpha_scat = xi * alpha_abs.
        n_boundaries : int
        The number of Dirichlet boundaries.

        Returns
        -------
        scipy.sparse.coo_matrix

        Sparse matrix representing the discretized diffusion term in
        coordinate format.
        """

        row = []
        col = []
        data = []

        # Cut-off value for absorption coefficient
        eps = 1.e-6

        # Modified absorption function
        def abs_fun(x):

            if abs(abs_f(x)) < eps:

                return eps

            else:

                return abs_f(x)

        if mesh.dim == 1:

            # Get the mesh size h
            h = mesh.dom_len[0] / mesh.n_cells[0]

            # Save cell centers for efficiency reasons
            cell_centers = mesh.cell_centers_1d()

            # Loop over boundary cells
            for boundary in mesh.outer_normal:

                for p in mesh.boundary_cells(boundary):

                    if p == 0:

                        row += 2 * [p]
                        col += [p, p + 1]

                        d_diri = 2. / h * 1. / \
                            (xip1 * abs_fun([cell_centers[p]]))
                        d_E = 2. / h * 1. / \
                            (xip1 * (abs_fun([cell_centers[p + 1]]
                                             ) + abs_fun([cell_centers[p]])))

                        data += [d_diri + d_E, -d_E]

                    if p == mesh.n_cells[0] - 1:

                        row += 2 * [p]
                        col += [p - 1, p]

                        d_W = 2. / h * 1. / \
                            (xip1 * (abs_fun([cell_centers[p - 1]]
                                             ) + abs_fun([cell_centers[p]])))

                        # 'inc_west' boundary condition
                        if n_boundaries == 1:

                            data += [-d_W, d_W]

                        # 'uniform' boundary condition
                        else:

                            d_diri = 2. / h * 1. / \
                                (xip1 * abs_fun([cell_centers[p]]))
                            data += [-d_W, d_diri + d_W]

            for p in mesh.interior_cells():

                row += 3 * [p]
                col += [p - 1, p, p + 1]

                d_E = 2. / h * 1. / \
                    (xip1 * (abs_fun([cell_centers[p + 1]]) +
                             abs_fun([cell_centers[p]])))
                d_W = 2. / h * 1. / \
                    (xip1 * (abs_fun([cell_centers[p - 1]]) +
                             abs_fun([cell_centers[p]])))

                data += [-d_W, d_E + d_W, -d_E]

        else:

            # Save cell centers for efficiency reasons
            cell_centers = mesh.cell_centers_2d()

            # 'inc_west' boundary condition
            if n_boundaries == 1:

                # Lower left corner
                p = mesh.south_west_corner()

                row += 3 * [p]
                col += [p, p + 1, p + mesh.n_cells[0]]

                d_diri = 2. / mesh.h[0] * mesh.h[1] * \
                    1. / (xip1 * abs_fun(cell_centers[p]))
                d_E = 2. / mesh.h[0] * mesh.h[1] / \
                    (xip1 *
                     (abs_fun(cell_centers[p + 1]) + abs_fun(cell_centers[p])))
                d_N = 2. / mesh.h[1] * mesh.h[0] / (
                    xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                            abs_fun(cell_centers[p])))

                d_p = d_diri + d_E + d_N

                data += [d_p, -d_E, -d_N]

                # Lower right corner
                p = mesh.south_east_corner()

                row += 3 * [p]
                col += [p, p - 1, p + mesh.n_cells[0]]

                d_W = 2. / mesh.h[0] * mesh.h[1] / \
                    (xip1 *
                     (abs_fun(cell_centers[p - 1]) + abs_fun(cell_centers[p])))
                d_N = 2. / mesh.h[1] * mesh.h[0] / (
                    xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                            abs_fun(cell_centers[p])))

                d_p = d_W + d_N

                data += [d_p, -d_W, -d_N]

                # Upper left corner
                p = mesh.north_west_corner()

                row += 3 * [p]
                col += [p, p + 1, p - mesh.n_cells[0]]

                d_diri = 2. / mesh.h[0] * mesh.h[1] * \
                    1. / (xip1 * abs_fun(cell_centers[p]))
                d_E = 2. / mesh.h[0] * mesh.h[1] / \
                    (xip1 *
                     (abs_fun(cell_centers[p + 1]) + abs_fun(cell_centers[p])))
                d_S = 2. / mesh.h[1] * mesh.h[0] / (
                    xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                            abs_fun(cell_centers[p])))

                d_p = d_diri + d_E + d_S

                data += [d_p, -d_E, -d_S]

                # Upper right corner
                p = mesh.north_east_corner()

                row += 3 * [p]
                col += [p, p - 1, p - mesh.n_cells[0]]

                d_W = 2. / mesh.h[0] * mesh.h[1] / \
                    (xip1 *
                     (abs_fun(cell_centers[p - 1]) + abs_fun(cell_centers[p])))
                d_S = 2. / mesh.h[1] * mesh.h[0] / (
                    xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                            abs_fun(cell_centers[p])))

                d_p = d_W + d_S

                data += [d_p, -d_W, -d_S]

                # Right boundary
                for p in mesh.boundary_cells(0):

                    row += 4 * [p]
                    col += [p, p - 1, p - mesh.n_cells[0], p + mesh.n_cells[0]]

                    d_W = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p - 1]) +
                          abs_fun(cell_centers[p])))
                    d_S = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))
                    d_N = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))

                    d_p = d_W + d_S + d_N
                    data += [d_p, -d_W, -d_S, -d_N]

                # Left boundary
                for p in mesh.boundary_cells(1):

                    row += 4 * [p]
                    col += [p, p + 1, p - mesh.n_cells[0], p + mesh.n_cells[0]]

                    d_diri = 2. / mesh.h[0] * mesh.h[1] / \
                        (xip1 * abs_fun(cell_centers[p]))
                    d_E = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p + 1]) +
                          abs_fun(cell_centers[p])))
                    d_S = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))
                    d_N = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))

                    d_p = d_diri + d_E + d_S + d_N
                    data += [d_p, -d_E, -d_S, -d_N]

                # Upper boundary
                for p in mesh.boundary_cells(2):

                    row += 4 * [p]
                    col += [p, p - 1, p + 1, p - mesh.n_cells[0]]

                    d_E = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p + 1]) +
                          abs_fun(cell_centers[p])))
                    d_W = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p - 1]) +
                          abs_fun(cell_centers[p])))
                    d_S = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))

                    d_p = d_E + d_W + d_S
                    data += [d_p, -d_W, -d_E, -d_S]

                # Lower boundary
                for p in mesh.boundary_cells(3):

                    row += 4 * [p]
                    col += [p, p - 1, p + 1, p + mesh.n_cells[0]]

                    d_E = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p + 1]) +
                          abs_fun(cell_centers[p])))
                    d_W = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p - 1]) +
                          abs_fun(cell_centers[p])))
                    d_N = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))

                    d_p = d_E + d_W + d_N
                    data += [d_p, -d_W, -d_E, -d_N]

                # Interior cells
                for p in mesh.interior_cells():

                    row += 5 * [p]
                    col += [p, p - 1, p + 1, p +
                            mesh.n_cells[0], p - mesh.n_cells[0]]

                    d_E = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p + 1]) +
                          abs_fun(cell_centers[p])))
                    d_W = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p - 1]) +
                          abs_fun(cell_centers[p])))
                    d_N = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))
                    d_S = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))

                    d_p = d_E + d_W + d_N + d_S
                    data += [d_p, -d_W, -d_E, -d_N, -d_S]

            # 'uniform' boundary condition
            else:

                # Lower left corner
                p = mesh.south_west_corner()

                row += 3 * [p]
                col += [p, p + 1, p + mesh.n_cells[0]]

                d_diri = 2. / mesh.h[1] * mesh.h[0] * \
                    1. / (xip1 * abs_fun(cell_centers[p]))
                d_diri += 2. / mesh.h[0] * mesh.h[1] * \
                    1. / (xip1 * abs_fun(cell_centers[p]))
                d_E = 2. / mesh.h[0] * mesh.h[1] / \
                    (xip1 *
                     (abs_fun(cell_centers[p + 1]) + abs_fun(cell_centers[p])))
                d_N = 2. / mesh.h[1] * mesh.h[0] / (
                    xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                            abs_fun(cell_centers[p])))

                d_p = d_diri + d_E + d_N

                data += [d_p, -d_E, -d_N]

                # Lower right corner
                p = mesh.south_east_corner()

                row += 3 * [p]
                col += [p, p - 1, p + mesh.n_cells[0]]

                d_diri = 2. / mesh.h[1] * mesh.h[0] * \
                    1. / (xip1 * abs_fun(cell_centers[p]))
                d_diri += 2. / mesh.h[0] * mesh.h[1] * \
                    1. / (xip1 * abs_fun(cell_centers[p]))
                d_W = 2. / mesh.h[0] * mesh.h[1] / \
                    (xip1 *
                     (abs_fun(cell_centers[p - 1]) + abs_fun(cell_centers[p])))
                d_N = 2. / mesh.h[1] * mesh.h[0] / (
                    xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                            abs_fun(cell_centers[p])))

                d_p = d_diri + d_W + d_N

                data += [d_p, -d_W, -d_N]

                # Upper left corner
                p = mesh.north_west_corner()

                row += 3 * [p]
                col += [p, p + 1, p - mesh.n_cells[0]]

                d_diri = 2. / mesh.h[1] * mesh.h[0] * \
                    1. / (xip1 * abs_fun(cell_centers[p]))
                d_diri += 2. / mesh.h[0] * mesh.h[1] * \
                    1. / (xip1 * abs_fun(cell_centers[p]))
                d_E = 2. / mesh.h[0] * mesh.h[1] / \
                    (xip1 *
                     (abs_fun(cell_centers[p + 1]) + abs_fun(cell_centers[p])))
                d_S = 2. / mesh.h[1] * mesh.h[0] / (
                    xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                            abs_fun(cell_centers[p])))

                d_p = d_diri + d_E + d_S

                data += [d_p, -d_E, -d_S]

                # Upper right corner
                p = mesh.north_east_corner()

                row += 3 * [p]
                col += [p, p - 1, p - mesh.n_cells[0]]

                d_diri = 2. / mesh.h[1] * mesh.h[0] * \
                    1. / (xip1 * abs_fun(cell_centers[p]))
                d_diri += 2. / mesh.h[0] * mesh.h[1] * \
                    1. / (xip1 * abs_fun(cell_centers[p]))
                d_W = 2. / mesh.h[0] * mesh.h[1] / \
                    (xip1 *
                     (abs_fun(cell_centers[p - 1]) + abs_fun(cell_centers[p])))
                d_S = 2. / mesh.h[1] * mesh.h[0] / (
                    xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                            abs_fun(cell_centers[p])))

                d_p = d_diri + d_W + d_S

                data += [d_p, -d_W, -d_S]

                # Right boundary
                for p in mesh.boundary_cells(0):

                    row += 4 * [p]
                    col += [p, p - 1, p - mesh.n_cells[0], p + mesh.n_cells[0]]

                    d_diri = 2. / mesh.h[0] * mesh.h[1] / \
                        (xip1 * abs_fun(cell_centers[p]))
                    d_W = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p - 1]) +
                          abs_fun(cell_centers[p])))
                    d_S = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))
                    d_N = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))

                    d_p = d_diri + d_W + d_S + d_N
                    data += [d_p, -d_W, -d_S, -d_N]

                # Left boundary
                for p in mesh.boundary_cells(1):

                    row += 4 * [p]
                    col += [p, p + 1, p - mesh.n_cells[0], p + mesh.n_cells[0]]

                    d_diri = 2. / mesh.h[0] * mesh.h[1] / \
                        (xip1 * abs_fun(cell_centers[p]))
                    d_E = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p + 1]) +
                          abs_fun(cell_centers[p])))
                    d_S = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))
                    d_N = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))

                    d_p = d_diri + d_E + d_S + d_N
                    data += [d_p, -d_E, -d_S, -d_N]

                # Upper boundary
                for p in mesh.boundary_cells(2):

                    row += 4 * [p]
                    col += [p, p - 1, p + 1, p - mesh.n_cells[0]]

                    d_diri = 2. * mesh.h[0] / mesh.h[1] / \
                        (xip1 * abs_fun(cell_centers[p]))
                    d_E = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p + 1]) +
                          abs_fun(cell_centers[p])))
                    d_W = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p - 1]) +
                          abs_fun(cell_centers[p])))
                    d_S = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))

                    d_p = d_diri + d_E + d_W + d_S
                    data += [d_p, -d_W, -d_E, -d_S]

                # Lower boundary
                for p in mesh.boundary_cells(3):

                    row += 4 * [p]
                    col += [p, p - 1, p + 1, p + mesh.n_cells[0]]

                    d_diri = 2. * mesh.h[0] / mesh.h[1] / \
                        (xip1 * abs_fun(cell_centers[p]))
                    d_E = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p + 1]) +
                          abs_fun(cell_centers[p])))
                    d_W = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p - 1]) +
                          abs_fun(cell_centers[p])))
                    d_N = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))

                    d_p = d_diri + d_E + d_W + d_N
                    data += [d_p, -d_W, -d_E, -d_N]

                # Interior cells
                for p in mesh.interior_cells():

                    row += 5 * [p]
                    col += [p, p - 1, p + 1, p +
                            mesh.n_cells[0], p - mesh.n_cells[0]]

                    d_E = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p + 1]) +
                          abs_fun(cell_centers[p])))
                    d_W = 2. / \
                        mesh.h[0] * mesh.h[1] / \
                        (xip1 *
                         (abs_fun(cell_centers[p - 1]) +
                          abs_fun(cell_centers[p])))
                    d_N = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p + mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))
                    d_S = 2. / mesh.h[1] * mesh.h[0] / (
                        xip1 * (abs_fun(cell_centers[p - mesh.n_cells[0]]) +
                                abs_fun(cell_centers[p])))

                    d_p = d_E + d_W + d_N + d_S
                    data += [d_p, -d_W, -d_E, -d_N, -d_S]

        return sps.coo_matrix((np.array(data) * 1./3., (row, col)),
                              shape=(mesh.n_cells[0] * mesh.n_cells[1],
                                     mesh.n_cells[0] * mesh.n_cells[1]))

    def __assemble_dirichlet__(self, mesh, abs_f, xip1, n_boundaries):
        """
        Assembles the vector of Dirichlet boundary conditions for the
        diffusion problem.

        Parameters
        ----------
        mesh : mesh.UniformMesh
            Uniform mesh used for the discretization.
        abs_f : callable
        Callable object, representing the absorption coefficient.
        Takes values in the domain as argument.
        scattering : string
        Type of scattering process assumed for the medium
        xip1 : float
        The ratio between scattering coefficient and absorption
        coefficient plus 1.0.
        alpha_scat = xi * alpha_abs.
        n_boundaries : int
        The number of Dirichlet boundaries.

        Returns
        -------
        numpy.ndarray
            Load vector of the system
        """

        # Assemble a vector containing the Dirichlet boundary conditions
        # given by diri_fun

        # Cut-off value for absorption coefficient
        eps = 1.e-6

        # Modified absorption function
        def abs_fun(x):

            if abs(abs_f(x)) < eps:

                return eps

            else:

                return abs_f(x)

        # Define a Dirichlet boundary function
        def diri_fun(x):

            return float(self.inflow_bc[0])

        diri_bc = np.zeros(self.n_dof)

        if mesh.dim == 1:

            # Get the mesh size h
            h = mesh.dom_len[0] / mesh.n_cells[0]

            # Save cell centers for efficiency reasons
            cell_centers = mesh.cell_centers_1d()

            # Get the distanceto a horizontal Dirichlet boundary
            # with respect to an adjacent cell center.
            distance = 0.5 * h

            # 'inc_west' boundary condition: left-most vertex
            diri_bc[0] = 2. / h * 1. / (xip1 * abs_fun([cell_centers[0]])) * \
                diri_fun(cell_centers[0] - distance)

            # 'uniform boundary condition'
            if n_boundaries == 2:

                # Right-most vertex
                diri_bc[mesh.n_cells[0] - 1] = 2. / h * 1. / (xip1 * abs_fun(
                    [cell_centers[mesh.n_cells[0] - 1]])) * \
                    diri_fun(cell_centers[mesh.n_cells[0] - 1] + distance)

        else:

            # Save cell centers for efficiency reasons
            cell_centers = mesh.cell_centers_2d()

            # Get the distances to a horizontal and vertical Dirichlet boundary
            # with respect to an adjacent cell center.
            distances = np.array([[0.5*mesh.h[0], 0], [0, 0.5*mesh.h[1]]])

            # 'inc_west' boundary condition
            if n_boundaries == 1:

                # Lower left corner
                p = mesh.south_west_corner()

                diri_bc[p] = 2. / mesh.h[0] * mesh.h[1] * 1. / \
                    (xip1 * abs_fun(cell_centers[p])) * \
                    diri_fun(cell_centers[p] - distances[0])

                # Upper left corner
                p = mesh.north_west_corner()

                diri_bc[p] = 2. / mesh.h[0] * mesh.h[1] * 1. / \
                    (xip1 * abs_fun(cell_centers[p])) * \
                    diri_fun(cell_centers[p] - distances[0])

                # Left boundary
                for p in mesh.boundary_cells(1):

                    diri_bc[p] = 2. / mesh.h[0] * mesh.h[1] / (xip1 * abs_fun(
                        cell_centers[p])) * diri_fun(cell_centers[p] -
                                                     distances[0])

            # 'uniform' boundary condition
            else:

                # Lower left corner
                p = mesh.south_west_corner()

                diri_bc[p] = 2. / mesh.h[1] * mesh.h[0] * 1. / \
                    (xip1 * abs_fun(cell_centers[p])) * \
                    diri_fun(cell_centers[p] - distances[1])
                diri_bc[p] += 2. / mesh.h[0] * mesh.h[1] * 1. / \
                    (xip1 * abs_fun(cell_centers[p])) * \
                    diri_fun(cell_centers[p] - distances[0])

                # Lower right corner
                p = mesh.south_east_corner()

                diri_bc[p] = 2. / mesh.h[1] * mesh.h[0] * 1. / \
                    (xip1 * abs_fun(cell_centers[p])) * \
                    diri_fun(cell_centers[p] - distances[1])
                diri_bc[p] += 2. / mesh.h[0] * mesh.h[1] * 1. / \
                    (xip1 * abs_fun(cell_centers[p])) * \
                    diri_fun(cell_centers[p] + distances[0])

                # Upper left corner
                p = mesh.north_west_corner()

                diri_bc[p] = 2. / mesh.h[1] * mesh.h[0] * 1. / \
                    (xip1 * abs_fun(cell_centers[p])) * \
                    diri_fun(cell_centers[p] + distances[1])
                diri_bc[p] += 2. / mesh.h[0] * mesh.h[1] * 1. / \
                    (xip1 * abs_fun(cell_centers[p])) * \
                    diri_fun(cell_centers[p] - distances[0])

                # Upper right corner
                p = mesh.north_east_corner()

                diri_bc[p] = 2. / mesh.h[1] * mesh.h[0] * 1. / \
                    (xip1 * abs_fun(cell_centers[p])) * \
                    diri_fun(cell_centers[p] + distances[1])
                diri_bc[p] += 2. / mesh.h[0] * mesh.h[1] * 1. / \
                    (xip1 * abs_fun(cell_centers[p])) * \
                    diri_fun(cell_centers[p] + distances[0])

                # Right boundary
                for p in mesh.boundary_cells(0):

                    diri_bc[p] = 2. / mesh.h[0] * mesh.h[1] * 1. / \
                        (xip1 * abs_fun(cell_centers[p])) * \
                        diri_fun(cell_centers[p] + distances[0])

                # Left boundary
                for p in mesh.boundary_cells(1):

                    diri_bc[p] = 2. / mesh.h[0] * mesh.h[1] * 1. / \
                        (xip1 * abs_fun(cell_centers[p])) * \
                        diri_fun(cell_centers[p] - distances[0])

                # Upper boundary
                for p in mesh.boundary_cells(2):

                    diri_bc[p] = 2. * mesh.h[0] / mesh.h[1] * 1. / \
                        (xip1 * abs_fun(
                            cell_centers[p])) * \
                        diri_fun(cell_centers[p] + distances[1])

                # Lower boundary
                for p in mesh.boundary_cells(3):

                    diri_bc[p] = 2. * mesh.h[0] / mesh.h[1] * 1. / \
                        (xip1 * abs_fun(
                            cell_centers[p])) * \
                        diri_fun(cell_centers[p] - distances[1])

        return 1./3. * diri_bc
