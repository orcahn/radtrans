import numpy as np
import matplotlib.pyplot as plt


def visualize(sol, mesh, n_ord, outputType):
    """
    Visualize the numerical results

    Parameters
    ----------
    sol : numpy.ndarray
        Numerical approximation to the solution of the model_problem
    mesh : mesh.UniformMesh
        Uniform mesh used to partition the domain of the model problem
    n_ord : integer
        Total number of discrete ordinates used in the discretization
    outputType : string
        String specifying which part or function of the solution to visualize
    """

    if mesh.dim == 1:

        if outputType == "firstOrdinate":

            plt.step(mesh.cell_centers_1d(),
                     sol[:mesh.n_cells[0]], where='mid')

        elif outputType == "meanIntensity":

            plt.step(mesh.cell_centers_1d(), np.mean(
                (sol[-mesh.n_cells[0]:],
                    sol[:mesh.n_cells[0]]),
                axis=0), where='mid')

        elif outputType == "diffusion":

            plt.step(mesh.cell_centers_1d(), sol)

        else:

            raise Exception('Unknown output type: ' + outputType)

    else:   # mesh.dim == 2

        if outputType == "firstOrdinate":

            z = sol[:mesh.n_tot].reshape(
                mesh.n_cells[1], mesh.n_cells[0], order='C')

        elif outputType == "meanIntensity":

            z = sol.reshape((n_ord, mesh.n_tot), order='C')
            z = np.mean(z, axis=0).reshape((mesh.n_cells[1], mesh.n_cells[0]))

        elif outputType == "totalIntensity":

            z = sol.reshape((n_ord, mesh.n_tot), order='C')
            z = np.sum(z, axis=0).reshape((mesh.n_cells[1], mesh.n_cells[0]))

        elif outputType == "diffusion":

            z = sol.reshape((mesh.n_cells[1], mesh.n_cells[0]), order='C')

        plt.figure(1)
        plt.pcolormesh(mesh.grid[0], mesh.grid[1], z)
        plt.colorbar()

    plt.show()
