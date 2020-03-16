import numpy as np


from scipy.integrate import fixed_quad


class Mesh:

    def __init__(self, length, n_cells):

        self.length = length
        self.n_cells = n_cells

        self.outer_normals = [1.0, -1.0]

        self.cell_boundaries, self.h = np.linspace(
            0.0, length, num=n_cells + 1, endpoint=True, retstep=True)

    def inflow_boundary_cells(self, ordIndex, n_dof):

        ib_cells = []

        if ordIndex == 0:
            ib_cells += [0]
        elif ordIndex == 1:
            ib_cells += [n_dof - 1]

        return ib_cells

    def outflow_boundary_cells(self, ordIndex, n_dof):

        ob_cells = []

        if ordIndex == 0:
            ob_cells += [n_dof - 1]
        elif ordIndex == 1:
            ob_cells += [0]

    # Compute the L2 scalar product of the absorption coefficient with the
    # basis functions corresponding to each cell. Order 4 gaussian
    # quadrature is used, which translates to 2 quadrature nodes per cell.
    def integrate_cellwise(self, fun):
        """
        Compute the L2 scalar product of a function with the basis functions
        corresponding to each cell. Order 4 gaussian quadrature is used, which
        translates to 2 quadrature nodes per cell.

        Parameters
        ----------
        fun : callable
            the function to be used as argument in the scalar product

        Returns
        -------
        np.ndarray of shape (n_cells,)
            Array with same number of entries as there are cells. Each entry k
            corresponds to the L2 scalar product of fun with basis function k.
        """

        return np.array([fixed_quad(
            fun, self.cell_boundaries[i], self.cell_boundaries[i+1], n=4)[0]
            for i in range(self.n_cells)])
