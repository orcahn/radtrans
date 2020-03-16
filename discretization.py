import numpy as np
import scipy.sparse as sps

import mesh


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

    def __init__(self, mp, n_cells, n_ordinates, do_weights=0):
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

        print('Discretization:\n' +
              '    - number of cells: ' + str(n_cells) + '\n' +
              '    - number of discrete ordinates: ' + str(n_ordinates) +
              '\n\n\n')

        # --------------------------------------------------------------------
        #               DISCRETE ORDINATES AND SCATTERING
        # --------------------------------------------------------------------

        # in one dimension there are only two possible discrete ordinates
        self.n_ord = n_ordinates if mp.dim == 2 else 2

        if mp.dim == 1 and n_ordinates != 2:
            print('Warning: In one dimension two discrete ordinates' +
                  '(+1.0, -1.0) will be used!')

        # list of directions of the discrete ordinates
        self.ord_dir = [np.array([1.0]), np.array([-1.0])]

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

        self.alpha = self.mesh.integrate_cellwise(mp.abs_fun)

        # matrix storing the dot products of relevant outer normals with each
        # discrete ordinate direction. Entry (m, l) is the dot product of
        # direction m with outer normal l
        ndotn = np.dot(self.ord_dir, np.transpose(self.mesh.outer_normals))

        # coo-format matrices representing the contributions to the stiffness
        # matrix. By default when converting to CSR or CSC format, duplicate
        # (i,j) entries will be summed together
        t_data = self.__assemble_transport__(n_cells, ndotn)
        a_data = self.__assemble_absorption__(mp.xip1)
        s_data = self.__assemble_scattering__(sig, n_cells, mp.xi)

        ta_data = [np.concatenate((t_data[r], a_data[r]), axis=0)
                   for r in range(3)]

        if mp.scat == 'none':

            self.stiff_mat = sps.coo_matrix(
                (ta_data[0], (ta_data[1], ta_data[2])),
                shape=(self.n_dof, self.n_dof))

            self.lambda_prec = sps.csr_matrix((self.n_dof, self.n_dof))
            self.stiff_mat = self.stiff_mat.tocsr()

        else:

            # concatenate indices and data of transport and scattering part
            data = [np.concatenate((ta_data[r], s_data[r]), axis=0)
                    for r in range(3)]
            # print(ta_data[0].shape, ta_data[1].shape, ta_data[2].shape)
            self.lambda_prec = sps.coo_matrix((
                ta_data[0], (ta_data[1], ta_data[2])),
                shape=(self.n_dof, self.n_dof))

            self.stiff_mat = sps.coo_matrix((data[0], (data[1], data[2])),
                                            shape=(self.n_dof, self.n_dof))

            self.lambda_prec = self.lambda_prec.tocsr()
            self.stiff_mat = self.stiff_mat.tocsr()

        # --------------------------------------------------------------------
        #                       LOAD VECTOR ASSEMBLY
        # --------------------------------------------------------------------

        self.load_vec = np.tile(np.array([mp.emiss * mp.s_e * self.alpha[m]
                                          for m in range(n_cells)]), reps=2)

        # add boundary conditions
        for m in range(n_ordinates):
            self.load_vec[self.mesh.inflow_boundary_cells(m, self.n_dof)] += \
                mp.inflow_bc[m]

    def __assemble_transport__(self, nc, ndotn):

        data = []
        row = []
        col = []

        def upwind_index(global_index, nn):

            if nn < 0.0:

                return global_index

            elif nn > 0.0:

                return global_index - 1

            else:  # value here does not matter

                return 0

        # loop over each degree of freedom
        for m in range(self.n_ord):
            for i in range(nc):

                global_index = m * nc + i
                nn = ndotn[m, 0]

                if global_index in \
                        self.mesh.inflow_boundary_cells(m, self.n_dof):

                    row += [global_index]

                    if nn > 0:
                        data += [nn]
                        col += [upwind_index(global_index + 1, nn)]
                    else:
                        data += [-nn]
                        col += [upwind_index(global_index, nn)]

                else:

                    data += [nn, -nn]
                    row += [global_index, global_index]
                    col += [upwind_index(global_index + 1, nn),
                            upwind_index(global_index, nn)]

        return np.array(data), np.array(row), np.array(col)

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
