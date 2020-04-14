import numpy as np
import matplotlib.pyplot as plt


def visualize(sol, mesh, n_ord, outputType):

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

            z = np.flipud(sol[:mesh.n_tot].reshape(
                (mesh.n_cells[0], mesh.n_cells[1]), order='C'))

        elif outputType == "meanIntensity":

            z = np.mean(sol.reshape((mesh.n_tot, n_ord), order='C'), axis=1
                        ).reshape((mesh.n_cells[0], mesh.n_cells[1]),
                                  order='C')

        elif outputType == "diffusion":

            z = sol.reshape(
                (mesh.n_cells[0], mesh.n_cells[1]))

        plt.figure(1)
        plt.pcolormesh(mesh.grid[0][:, 0], mesh.grid[1][0, :], z)
        plt.colorbar()

    plt.show()
