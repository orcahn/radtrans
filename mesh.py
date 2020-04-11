import numpy as np

from enum import IntEnum


class Direction(IntEnum):

    E = 0       # east
    W = 1       # west
    N = 2       # north
    S = 3       # south


def midpoint(a, b, fun, h):

    return (h[0] * h[1]) * fun(0.5 * (a + b))


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
    h : tuple of floats
        Length of a single cell along each dimension
    outer_normal : {mesh.Direction: np.ndarray} dict
        Outer normals to corresponding directions
    grid : tuple of np.ndarray
        Cartesian Coordinates of the gridpoints.

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

        self.outer_normal = None
        self.n_cells = None
        self.h = None

        if dimension == 1:

            self.outer_normal = {Direction.E: np.array([1.0]),
                                 Direction.N: np.array([0.0]),
                                 Direction.W: np.array([-1.0]),
                                 Direction.S: np.array([0.0])}

            self.n_cells = (n_cells[0], 1)

            # this is a hack for discretization in 1 dimension to
            # work properly. In this case h does not have the
            # meaning of mesh size.
            self.h = (0.0, 1.0)

        else:

            self.outer_normal = {Direction.E: np.array([1.0, 0.0]),
                                 Direction.N: np.array([0.0, 1.0]),
                                 Direction.W: np.array([-1.0, 0.0]),
                                 Direction.S: np.array([0.0, -1.0])}

            self.n_cells = n_cells[:dimension]

            self.h = tuple(
                map(lambda a, b: a / float(b), self.dom_len, self.n_cells))

        # compute total number of cells
        self.n_tot = 1
        for nc in self.n_cells:
            self.n_tot *= nc

        # i-th x-coordinate is given by entry [i, 0] in first tuple entry.
        # j-th y-coordinate is given by entry [0, j] in second tuple entry.
        self.grid = np.meshgrid(
            *[np.linspace(0.0, self.dom_len[n],
                          num=self.n_cells[n] + 1, endpoint=True)
              for n in range(dimension)],
            indexing='ij', sparse=True, copy=False)

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
            def quadrature_fun(a, b): return midpoint(a, b, abs_fun, self.h[0])
        else:
            def quadrature_fun(a, b): return trapezoidal(
                a, b, abs_fun, self.h[0])

        boundaries = np.linspace(0.0, self.dom_len, num=self.n_cells[0] + 1,
                                 endpoint=True, retstep=False)

        return np.array([quadrature_fun(boundaries[i], boundaries[i + 1])
                         for i in range(self.n_cells[0])])

    def outflow_boundary(self, ord_dir):

        boundaries = []

        for d in self.outer_normal:

            if np.dot(self.outer_normal[d], ord_dir) > 0.0:

                boundaries += [d]

        return boundaries

    def interior_cells(self):

        # start with the lowest interior row
        interior_row = np.arange(self.n_cells[0] + 1, 2 * self.n_cells[0] - 1)
        interior_indices = interior_row

        if self.dim == 1:

            return np.add(interior_indices, -self.n_cells[0])

        else:

            for row in range(self.n_cells[1] - 3):

                # get indices of row above
                interior_row = np.add(interior_row, self.n_cells[0])

                # append to array storing all indices
                interior_indices = np.concatenate(
                    (interior_indices, interior_row))

            return interior_indices

    def boundary_cells(self, direction):

        if direction == Direction.E:

            return np.arange(start=2 * self.n_cells[0] - 1,
                             stop=self.n_tot - 1,
                             step=self.n_cells[0])

        elif direction == Direction.N:

            if self.dim == 1:
                return []
            else:
                return np.arange(
                    start=(self.n_cells[1] - 1) * self.n_cells[0] + 1,
                    stop=self.n_tot - 1)

        elif direction == Direction.W:

            return np.arange(start=self.n_cells[0],
                             stop=(self.n_cells[1] - 1) * self.n_cells[0],
                             step=self.n_cells[0])

        elif direction == Direction.S:

            if self.dim == 1:
                return []
            else:
                return np.arange(start=1, stop=self.n_cells[0] - 1)

        else:
            raise Exception('Unknown direction ' + str(direction))

    def cell_centers(self):
        """
        Coordinates of the cell centers

        Returns
        -------
        np.ndarray
            Array containing the coordinates of the cell centers with the same
            indexing as for the corresponding cell indices.
        """

        return np.arange(
            0.5 * self.h[0], self.n_cells[0] * self.h[0], self.h[0])

    def south_west_corner(self):

        return 0

    def south_east_corner(self):

        return self.n_cells[0] - 1

    def north_west_corner(self):

        return (self.n_cells[1] - 1) * self.n_cells[0]

    def north_east_corner(self):

        return self.n_tot - 1
