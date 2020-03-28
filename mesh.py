import numpy as np

from enum import IntEnum


class Direction(IntEnum):

    E = 0       # east
    W = 1       # west
    N = 2       # north
    S = 3       # south


def midpoint(a, b, fun, h):

    return h * fun(0.5 * (a + b))


def trapezoidal(a, b, fun, h):

    return 0.5 * h * (fun(a) + fun(b))


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

        self.dom_len = domain_length
        self.n_cells = n_cells
        self.h = domain_length / float(n_cells)

        # Outer normal vectors
        self.outer_normal = {Direction.E: np.array([1.0]),
                             Direction.W: np.array([-1.0])}

        print('Mesh:\n' +
              '-----\n' +
              '    - domain: (0.0, ' + str(domain_length) + ')\n' +
              '    - number of cells: ' + str(n_cells) +
              '\n\n')

    def integrate_cellwise(self, abs_fun, quad_method):
        """
        Compute the L2 scalar product of a function with the basis functions
        corresponding to each cell.

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

        # In a typical application, cell sizes are so small that low order
        # quadrature is of acceptable accuracy. Tests showed, that higher order
        # methods did not significantly improve accuracy, but imposed a
        # high performance overhead.
        quadrature_fun = None

        if quad_method == 'midpoint':
            def quadrature_fun(a, b): return midpoint(a, b, abs_fun, self.h)
        else:
            def quadrature_fun(a, b): return trapezoidal(a, b, abs_fun, self.h)

        boundaries = np.linspace(0.0, self.dom_len, num=self.n_cells + 1,
                                 endpoint=True, retstep=False)

        return np.array([quadrature_fun(boundaries[i], boundaries[i + 1])
                         for i in range(self.n_cells)])

    def inflow_boundary_cells(self, ordIndex):

        if ordIndex == 0:

            return [0]

        elif ordIndex == 1:

            return [-1]

        else:
            raise Exception('Invalid ordinate index')

    def is_outflow_boundary_cell(self, cell, ord_index):

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

    def boundary_cells(self, direction):

        if direction == Direction.E:
            return range(self.n_cells - 1, self.n_cells)

        elif direction == Direction.W:
            return range(1)

        else:
            return range(0)

    def interior_cells(self):

        return range(1, self.n_cells - 1)

    def cell_centers(self):

        return np.arange(0.5 * self.h, self.n_cells * self.h, self.h)
