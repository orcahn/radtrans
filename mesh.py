import numpy as np

from scipy.integrate import fixed_quad
from enum import IntEnum


class Direction(IntEnum):

    E = 0       # east
    W = 1       # west
    N = 2       # north
    S = 3       # south


class Mesh:

    def __init__(self, domain_length, n_cells):
        """
        Parameters
        ----------
        domain_length : float
            Length of the one-dimensional domain. The domain itself is then
            defined as D = (0, dom_len).
        n_cells : integer
            Number of cells, the domain is to be partitioned into.
        """

        self.n_cells = n_cells

        # Outer normal vectors
        self.outer_normal = {Direction.E: np.array([1.0]),
                             Direction.W: np.array([-1.0])}

        self.cell_boundaries, self.h = np.linspace(
            0.0, domain_length, num=n_cells + 1, endpoint=True, retstep=True)

        print('Mesh:\n' +
              '    - domain: (0.0, ' + str(domain_length) + ')\n' +
              '    - number of cells: ' + str(n_cells) +
              '\n\n')

    def inflow_boundary(self, ordIndex, n_dof):

        cells = []

        if ordIndex == 0:

            cells += [0]

        elif ordIndex == 1:

            cells += [n_dof - 1]

        else:
            raise Exception('Invalid ordinate index')

        return np.array([cells])

    def outflow_boundary_cell(self, cell, ord_index):

        if ord_index == 0:

            if cell == self.n_cells - 1:
                return True

            else:
                return False

        elif ord_index == 1:

            if cell == 0:
                return True

            else:
                return False

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

    def cell_centers(self):

        return np.arange(0.5 * self.h, self.n_cells * self.h, self.h)
