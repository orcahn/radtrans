import numpy as np
import matplotlib.pyplot as plt


def visualize(sol, abs_fun, mesh, n_ord, outputType):
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

    fig, (ax0, ax1) = plt.subplots(1, 2)

    if mesh.dim == 1:

        domain = mesh.cell_centers_1d()

        abs_coeff_val = np.array([abs_fun([x]) for x in domain])

        if outputType == "firstOrdinate":

            z = sol[:mesh.n_cells[0]]

        elif outputType == "meanIntensity":

            z = np.mean((sol[-mesh.n_cells[0]:], sol[:mesh.n_cells[0]]),
                        axis=0)

        elif outputType == "totalIntensity":

            z = np.sum((sol[-mesh.n_cells[0]:], sol[:mesh.n_cells[0]]),
                       axis=0)

        elif outputType == "diffusion":

            z = sol

        else:

            raise Exception('Unknown output type: ' + outputType)

        try:

            from matplotlib import rc

            font = {'family': 'serif',
                    'size': 16}

            rc('text', usetex=True)
            rc('font', **font)

            ax0.plot(domain, abs_coeff_val)
            ax0.set_title(r"Absorption Coefficient")
            ax0.set_xlabel(r"x [m]")
            ax0.set_ylabel(r"$\alpha \; [m^{-1}]$")

            ax1.step(domain, z, where='mid')
            ax1.set_title('Numerical Solution')
            ax1.set_xlabel(r"x [m]")
            ax1.set_ylabel(
                r"intensity per $\mathcal{I} = \frac{1}{2} \: c^3 h^{-2} " +
                r"\nu^{-3}$ *")
            ax1.text(0.5, 0.0,
                     r"*speed of light c, planck constant h, frequency $\nu$",
                     size=12)

        except (ImportError, RuntimeError) as Err:

            print(Err)

            print('Tex output could not be rendered. Using bare output.')

            ax0.plot(domain, abs_coeff_val)
            ax0.set_title('Absorption Coefficient')
            ax0.set_xlabel('x [m]')
            ax0.set_ylabel('alpha [1 / m]')

            ax1.step(domain, z, where='mid')
            ax1.set_title('Numerical Solution')
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel(
                'intensity per I = c^3 / (2 * h^2 * (nu)^3) *')
            ax1.text(0.5, 0.0,
                     '*speed of light c, planck constant h, frequency nu',
                     size=12)

    else:   # mesh.dim == 2

        abs_coeff_val = [[abs_fun([x, y]) for x in mesh.grid[0][0, :]]
                         for y in mesh.grid[1][:, 0]]

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

        else:

            raise Exception('Unknown output type ' + outputType)

        try:

            from matplotlib import rc

            font = {'family': 'serif',
                    'size': 16}

            rc('text', usetex=True)
            rc('font', **font)

            if min(mesh.h) > 0.05:
                abs_plot = ax0.pcolormesh(mesh.grid[0], mesh.grid[1],
                                          abs_coeff_val, cmap='Blues', ec='k',
                                          linewidths=0.01)
            else:
                abs_plot = ax0.pcolormesh(mesh.grid[0], mesh.grid[1],
                                          abs_coeff_val, cmap='Blues')

            sol_plot = ax1.pcolormesh(mesh.grid[0], mesh.grid[1], z)

            abs_cbar = fig.colorbar(abs_plot, ax=ax0)
            sol_cbar = fig.colorbar(sol_plot, ax=ax1)

            ax0.set_title(r"Absorption Coefficient")
            ax0.set_xlabel(r"x [m]")
            ax0.set_ylabel(r"y [m]")

            ax1.set_title(r"Numerical Solution")
            ax1.set_xlabel(r"x [m]")
            ax1.set_ylabel(r"y [m]")

            abs_cbar.ax.set_ylabel(r"$\alpha \; [m^{-1}]$",
                                   labelpad=26, rotation=270)
            sol_cbar.ax.set_ylabel(
                r"intensity per $\mathcal{I} = \frac{1}{2} \: c^3 h^{-2} " +
                r"\nu^{-3}$ *", labelpad=26, rotation=270)
            ax1.text(0.5, -0.2,
                     r"*speed of light c, planck constant h, frequency $\nu$",
                     size=12)

        except (ImportError, RuntimeError) as Err:

            print(Err)

            print('Tex output could not be rendered. Using bare output.')

            if min(mesh.h) > 0.05:
                abs_plot = ax0.pcolormesh(mesh.grid[0], mesh.grid[1],
                                          abs_coeff_val, cmap='Blues', ec='k',
                                          linewidths=0.01)
            else:
                abs_plot = ax0.pcolormesh(mesh.grid[0], mesh.grid[1],
                                          abs_coeff_val, cmap='Blues')

            sol_plot = ax1.pcolormesh(mesh.grid[0], mesh.grid[1], z)

            abs_cbar = fig.colorbar(abs_plot, ax=ax0)
            sol_cbar = fig.colorbar(sol_plot, ax=ax1)

            ax0.set_title('Absorption Coefficient')
            ax0.set_xlabel('x [m]')
            ax0.set_ylabel('y [m]')

            ax1.set_title('Numerical Solution')
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel('y [m]')

            abs_cbar.ax.set_ylabel(
                'alpha [1 / m]', labelpad=26, rotation=270)
            sol_cbar.ax.set_ylabel('intensity per I = c^3 / (h^2 * (nu)^3) *',
                                   labelpad=26, rotation=270)
            ax1.text(0.5, -0.2,
                     '*speed of light c, planck constant h, frequency nu')

    plt.show()
