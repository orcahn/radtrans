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


class UniformMesh:
    """
    Uniform mesh in 1 or 2 dimensions

    Attributes
    ----------
    dim : integer
        The dimension of the domain.
    dom_len : tuple of floats
        Length of the domain.
    n_cells : tuple of integers
        tuple indicating the number of cells along each dimension.
    n_tot : integer
        Total number of cells the domain is partitioned into.
    h : float
        Length of a single cell.
    outer_normal : {mesh.Direction: np.ndarray} dict
        Outer normals to corresponding directions

    Methods
    -------
    integrate_cellwise(abs_fun, quad_method)
        Compute the L2 scalar product of a function with the basis functions
        corresponding to each cell.
    inflow_boundary_cells(ord_index)
        Compute the cells with part on the inflow boundary of specific ordinate
    is_outflow_boundary_cell(cell, ord_index)
        Test for a cell to have part on the outflow boundary of specific
        ordinate
    boundary_cells(direction)
        Compute the cells with part on the domain boundary in specific
        direction
    interior_cells()
        Compute indices of the interior cells
    cell_centers()
        Compute the coordinates of the cell centers
    """

    def __init__(self, dimension, domain_length, n_cells):
        """
        Parameters
        ----------
        dimension : integer
            The dimension of the domain
        domain_length : float
            Length of the one-dimensional domain
        n_cells : integer
            Number of cells
        """

        assert dimension in [1, 2], \
            'Dimension ' + dimension + ' of the domain is not supported. ' + \
            'Currently only 1 and 2 dimensions are supported.'
        self.dim = dimension

        for length in domain_length:
            assert length > 0.0, 'Invalid domain length. Must be positive'
        self.dom_len = domain_length[:dimension]

        domain_str = '(0.0, ' + str(domain_length[0]) + ')'
        for d in range(dimension - 1):
            domain_str += ' x (0.0, ' + str(domain_length[d + 1]) + ')'

        self.n_cells = n_cells[:dimension]

        # compute total number of cells
        self.n_tot = 1

        for nc in self.n_cells:
            self.n_tot *= nc

        self.h = tuple(
            map(lambda a, b: a / float(b), self.dom_len, self.n_cells))

        # Outer normal vectors
        self.outer_normal = None

        if dimension == 1:
            self.outer_normal = {Direction.E: np.array([1.0]),
                                 Direction.W: np.array([-1.0])}
        elif dimension == 2:
            self.outer_normal = {Direction.E: np.array([1.0, 0.0]),
                                 Direction.N: np.array([0.0, 1.0]),
                                 Direction.W: np.array([-1.0, 0.0]),
                                 Direction.S: np.array([0.0, -1.0])}

        print('Mesh:\n' +
              '-----\n' +
              '    - dimension: ' + str(dimension) + '\n' +
              '    - domain: ' + domain_str + '\n' +
              '    - cells per dimension: ' + str(self.n_cells) + '\n' +
              '    - total number of cells: ' + str(self.n_tot) + '\n' +
              '    - mesh size per dimension: ' + str(self.h) +
              '\n\n')

    def integrate_cellwise(self, abs_fun, quad_method):
        """
        Compute the L2 scalar product of a function with the basis functions
        corresponding to each cell.

        Parameters
        ----------
        fun : callable
            The function to be used as argument in the scalar product
        quad_method : string
            Specifies which quadrature method to use in the computation of
            the scalar product.

        Returns
        -------
        np.ndarray of shape (n_tot,)
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

    def inflow_boundary_cells(self, ord_index):
        """
        List of all cell indices corresponding to a cell, which has at
        least one face on the inflow boundary of ordinate ord_index.

        Parameters
        ----------
        ord_index : integer
            Index of the desired ordinate

        Returns
        -------
        list of integers
            List of the indices of the cells with part in the
            inflow boundary of ord_index
        """

        if ord_index == 0:

            return [0]

        elif ord_index == 1:

            return [-1]

        else:
            raise Exception('Invalid ordinate index')

    def is_outflow_boundary_cell(self, cell, ord_index):
        """
        Test for a cell to be part of the outflow boundary

        Parameters
        ----------
        cell : integer
            Index of the cell to be tested
        ord_index : integer
            Index of the desired ordinate

        Returns
        -------
        bool
            True if cell has a part on the outflow boundary of ordinate
            ord_index, False otherwise.
        """

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
        """
        Cells with part on the domain boudary in a certain direction

        Parameters
        ----------
        direction : mesh.Direction
            Direction, indicating the boudary

        Returns
        -------
        range
            Range of indices of the cells with part on the boundary
            specified by the direction.
        """

        if direction == Direction.E:
            return range(self.n_cells - 1, self.n_cells)

        elif direction == Direction.W:
            return range(1)

        else:
            return range(0)

    def interior_cells(self):
        """
        Indices of the interior cells

        Returns
        -------
        range
            Range of indices of the interior cells of the domain.
        """

        return range(1, self.n_cells - 1)

    def cell_centers(self):
        """
        Coordinates of the cell centers

        Returns
        -------
        np.ndarray
            Array containing the coordinates of the cell centers with the same
            indexing as for the corresponding cell indices.
        """

        return np.arange(0.5 * self.h, self.n_cells * self.h, self.h)
