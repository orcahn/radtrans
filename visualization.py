import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    """
    Given a computed solution and a mesh, this class produces visual output in
    1 and 2 spatial dimensions.
    """

    def __init__(self, dimension, method, solution, unif_mesh, n_ordinates,
                 outputType):

        self.dim = dimension
        self.method = method
        self.x = solution
        self.mesh = unif_mesh
        self.n_ord = n_ordinates
        self.outputType = outputType

    def visualize(self):

        if self.dim == 1:

            if self.outputType == "firstOrdinate":

                plt.step(self.mesh.cell_centers(),
                         self.x[:self.mesh.n_cells[0]], where='mid')

            elif self.outputType == "meanIntensity":

                plt.step(self.mesh.cell_centers(), np.mean(
                    (self.x[-self.mesh.n_cells[0]:],
                        self.x[:self.mesh.n_cells[0]]),
                    axis=0), where='mid')

            elif self.outputType == "diffusion":

                plt.step(self.mesh.cell_centers(), self.x)

            else:

                raise Exception('Unknown output type: ' + self.outputType)

        else:   # self.dim == 2

            if self.outputType == "firstOrdinate":

                X, Y = self.mesh.centers
                Z = self.x[:self.mesh.n_cells[0]*self.mesh.n_cells[1]
                           ].reshape((self.mesh.n_cells[0], self.mesh.n_cells[1]))

            elif self.outputType == "meanIntensity":

                X, Y = self.mesh.centers
                Z = np.mean(np.array_split(self.x, self.n_ord), axis=0).reshape(
                    (self.mesh.n_cells[0], self.mesh.n_cells[1]))

            elif self.outputType == "diffusion":

                X, Y = self.mesh.centers
                Z = self.x.reshape(
                    (self.mesh.n_cells[0], self.mesh.n_cells[1]))

            plt.figure(1)
            plt.contourf(X, Y, Z)
            plt.colorbar()
            plt.figure(2)
            plt.plot(np.arange(
                0.5 * self.mesh.h[0], self.mesh.n_cells[0] * self.mesh.h[0], self.mesh.h[0]), Z[:, 0])

        plt.show()
