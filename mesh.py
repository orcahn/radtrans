import numpy as np

from enum import IntEnum


class Direction(IntEnum):

    E = 0       # east
    W = 1       # west
    N = 2       # north
    S = 3       # south


def quadrature(fun, w, qn):
    """
    General quadrature in one or two dimensions

    Parameters
    ----------
    fun : callable
        The function to be integrated
    w : list of lists of floats
        Quadrature weights. First entry corresponds to a quadrature node,
        second entry to a dimension.
    qn : list of lists of floats
        Quadrature nodes. First entry corresponds to a node, second to
        a dimension.

    Returns
    float
        The integral of fun over the domain implied by the quadrature nodes.
    """

    integral = 0.0

    for q in range(len(w)):
        for p in range(len(w)):
            integral += w[q][0] * w[p][1] * fun([qn[q][0], qn[p][1]])

    return integral


class UniformMesh:
    """
    Uniform mesh in 1 or 2 dimensions

    Attributes
    ----------
    dim : integer
        The dimension of the domain.
    dom_len : tuple of floats
        Length of the domain along each dimension
    outer_normal : <mesh.Direction: np.ndarray> dictionary
        Outer normals to corresponding directions
    n_cells : tuple of integers
        tuple indicating the number of cells along each dimension.
    n_tot : integer
        Total number of cells the domain is partitioned into.
    h : tuple of floats
        Length of a single cell along each dimension
    grid : numpy.meshgrid
        Cartesian Coordinates of the gridpoints.

    Methods
    -------
    __integrate_cellwise__1d__(abs_fun, quad_method)
        Compute the L2 scalar product of a function with the basis functions
        corresponding to each cell in one dimension.
    __integrate_cellwise__2d__(abs_fun, quad_method)
        Compute the L2 scalar product of a function with the basis functions
        corresponding to each cell in two dimensions.
    integrate_cellwise(abs_fun, quad_method)
        Compute the L2 scalar product of a function with the basis functions
        corresponding to each cell.
    interior_cells()
        Compute the indices of the interior cells
    boundary_cells(direction)
        Compute the indices of the cells with part on the domain boundary
        in a given direction. Corner cells are excluded.
    cell_centers_1d()
        Compute the cell centers for a one dimensional mesh.
    south_west_corner()
        Compute index of the cell in the south west corner of the mesh
    south_east_corner()
        Compute index of the cell in the south east corner of the mesh
    north_west_corner()
        Compute index of the cell in the north west corner of the mesh
    north_east_corner()
        Compute index of the cell in the north east corner of the mesh
    """

    def __init__(self, dimension, domain_length, n_cells):
        """
        Parameters
        ----------
        dimension : integer
            The dimension of the domain
        domain_length : tuple of floats
            Length of the domain along each dimension
        n_cells : tuple of integers
            Number of cells along each dimension
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
            indexing='xy', sparse=True, copy=False)

        print('Mesh:\n' +
              '-----\n' +
              '    - dimension: ' + str(dimension) + '\n' +
              '    - domain: ' + domain_str + '\n' +
              '    - cells per dimension: ' + str(self.n_cells) + '\n' +
              '    - total number of cells: ' + str(self.n_tot) + '\n' +
              '    - mesh size per dimension: ' + str(self.h) +
              '\n\n')

    def __integrate_cellwise_1d__(self, abs_fun, quad_method):
        """
        Computes the L2 scalar product of a function with the basis functions
        corresponding to each cell in one dimension.

        Parameters
        ----------
        abs_fun : callable
            The absorption function of the model problem
        quad_method : string
            The quadrature method to be used

        Returns
        -------
        numpy.ndarray
            An array, where each entry is given by the integral of the
            absorption coefficient over the cell with according index.
        """

        # In a typical application, cell sizes are so small that low order
        # quadrature is of acceptable accuracy. Tests showed, that higher order
        # methods did not significantly improve accuracy, but imposed a
        # high performance overhead.
        if quad_method == 'midpoint':

            def quadrature(a, b, fun, h):

                return h * fun([0.5 * (a + b)])

        else:

            def quadrature(a, b, fun, h):

                return 0.5 * h * (fun([a]) + fun([b]))

        def q_fun(a, b): return quadrature(a, b, abs_fun,
                                           self.dom_len[0] / self.n_cells[0])

        return np.array([q_fun(self.grid[0][i], self.grid[0][i+1])
                         for i in range(self.n_cells[0])])

    def __integrate_cellwise_2d__(self, abs_fun, quad_method):
        """
        Computes the L2 scalar product of a function with the basis functions
        corresponding to each cell in a two dimensional mesh.

        Parameters
        ----------
        abs_fun : callable
            The absorption function of the model problem
        quad_method : string
            The quadrature method to be used

        Returns
        -------
        numpy.ndarray
            An array, where each entry is given by the integral of the
            absorption coefficient over the cell with according index.
        """

        # In a typical application, cell sizes are so small that low order
        # quadrature is of acceptable accuracy. Tests showed, that higher order
        # methods did not significantly improve accuracy, but imposed a
        # high performance overhead.
        if quad_method == 'midpoint':

            weights = [[self.h[0], self.h[1]]]

            def nodes(cell):

                # 2d indices of global indexing of cell
                i = cell % self.n_cells[0]
                j = (cell - i) // self.n_cells[0]

                return [[0.5 * (self.grid[0][0, i] +
                                self.grid[0][0, i+1]),
                         0.5 * (self.grid[1][j, 0] +
                                self.grid[1][j+1, 0])]]

        else:   # quad_method == 'trapezoidal'

            weights = [2 * [0.5 * self.h[0], 0.5 * self.h[1]]]

            def nodes(cell):

                # 2d indices of global indexing of cell
                i = cell % self.n_cells[0]
                j = (cell - i) // self.n_cells[0]

                return [[self.grid[0][0, i],
                         self.grid[1][j, 0]],
                        [self.grid[0][0, i+1],
                         self.grid[1][j+1, 0]]]

        def q_fun(cell): return quadrature(abs_fun, weights, nodes(cell))

        return np.array([q_fun(cell) for cell in range(self.n_tot)])

    def integrate_cellwise(self, abs_fun, quad_method):
        """
        Computes the L2 scalar product of a function with the basis functions
        corresponding to each cell for the given mesh.

        Parameters
        ----------
        abs_fun : callable
            The absorption function of the model problem
        quad_method : string
            The quadrature method to be used

        Returns
        -------
        numpy.ndarray
            An array, where each entry is given by the integral of the
            absorption coefficient over the cell with according index.
        """

        if self.dim == 1:
            return self.__integrate_cellwise_1d__(abs_fun, quad_method)
        else:
            return self.__integrate_cellwise_2d__(abs_fun, quad_method)

    def interior_cells(self):
        """
        Computes the indices of the interior cells of the mesh

        Returns
        -------
        numpy.ndarray
            An array with the indices of the interior cells
        """

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
        """
        Computes the indices of the cells with at least one face on the domain
        boundary in a given direction.

        Parameters
        ----------
        direction : mesh.Direction
            The direction of the domain boundary that is of interest

        Returns
        -------
        numpy.ndarray
            An array with the indices of the boundary cells
        """

        if direction == Direction.E:

            if self.dim == 1:
                return np.array([self.n_cells[0] - 1])
            else:
                return np.arange(start=2 * self.n_cells[0] - 1,
                                 stop=self.n_tot - 1,
                                 step=self.n_cells[0])

        elif direction == Direction.N:

            if self.dim == 1:
                return np.array([])
            else:
                return np.arange(
                    start=(self.n_cells[1] - 1) * self.n_cells[0] + 1,
                    stop=self.n_tot - 1)

        elif direction == Direction.W:

            if self.dim == 1:
                return np.array([0])
            else:
                return np.arange(start=self.n_cells[0],
                                 stop=(self.n_cells[1] - 1) * self.n_cells[0],
                                 step=self.n_cells[0])

        elif direction == Direction.S:

            if self.dim == 1:
                return np.array([])
            else:
                return np.arange(start=1, stop=self.n_cells[0] - 1)

        else:
            raise Exception('Unknown direction ' + str(direction))

    def cell_centers_1d(self):
        """
        Coordinates of the cell centers in a one-dimensional mesh

        Returns
        -------
        np.ndarray
            Array containing the coordinates of the cell centers with the same
            indexing as for the corresponding cell indices.
        """

        if self.dim == 1:

            h = self.dom_len[0] / self.n_cells[0]

            return np.arange(0.5 * h, self.n_cells[0] * h, h)

        else:

            raise Exception('cell_centers_1d() only works for in 1 dimension')

    def south_west_corner(self):
        """
        Computes the index of the cell in the south western corner of the mesh.

        Returns
        -------
        integer
            index of the south western corner cell.
        """

        return 0

    def south_east_corner(self):
        """
        Computes the index of the cell in the south eastern corner of the mesh.

        Returns
        -------
        integer
            index of the south eastern corner cell.
        """

        return self.n_cells[0] - 1

    def north_west_corner(self):
        """
        Computes the index of the cell in the north western corner of the mesh.

        Returns
        -------
        integer
            index of the north western corner cell.
        """

        return (self.n_cells[1] - 1) * self.n_cells[0]

    def north_east_corner(self):
        """
        Computes the index of the cell in the north eastern corner of the mesh.

        Returns
        -------
        integer
            index of the north eastern corner cell.
        """

        return self.n_tot - 1
