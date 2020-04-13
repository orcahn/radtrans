import sys
import timeit
import configparser

import numpy as np
import matplotlib.pyplot as plt

import modelProblem
import absorption
import mesh
import discretization
import solver


class RadiativeTransfer:
    """
    Main class of the project.
    Reads in parameter .ini file, specifies model, discretization method
    and solver parameters and discretizes and solves the radiative transfer
    problem for the given parameters. The numerical solution is then
    visualized.
    """

    def main(self, argv):

        # parse parameter file
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(argv[1:])

        dimension = int(config['MODEL']['dimension'])
        temperature = float(config['MODEL']['temperature'])
        frequency = float(config['MODEL']['frequency'])
        albedo = float(config['MODEL']['albedo'])
        emissivity = float(config['MODEL']['emissivity'])

        domain = tuple(
            map(float,
                config['MODEL']['domain'].strip().split(',')))

        scattering = str(config['MODEL']['scattering'])
        assert scattering in ['none', 'isotropic'], \
            'Scattering type ' + scattering + ' currently not supported.'

        abs_type = config['MODEL']['absorptionType']
        absorption_coeff = absorption.Absorption(abs_type, domain)

        n_cells = tuple(
            map(int,
                config['DISCRETIZATION']['n_cells'].strip().split(',')))

        # in one dimension there are only two possible discrete ordinates
        n_ordinates = 2 if dimension == 1 else \
            int(config['DISCRETIZATION']['n_ordinates'])

        flux = str(config['DISCRETIZATION']['flux'])

        boundary_values = None

        if config['BOUNDARY_VALUES']['type'] == 'uniform':

            boundary_values = tuple(config['BOUNDARY_VALUES']['value']
                                    for m in range(n_ordinates))

        elif config['BOUNDARY_VALUES']['type'] == 'inc_east':

            boundary_values = (1.0, *[0.0 for m in range(n_ordinates - 1)])

        elif config['BOUNDARY_VALUES']['type'] == 'manual':

            boundary_values = tuple(
                map(float,
                    config['BOUNDARY_VALUES']['valArray'].strip().split(',')))

        else:

            n_bndry_val = len(list(
                map(float,
                    config['BOUNDARY_VALUES']['valArray'].strip().split(','))))

            sys.exit('Option \'manual\' was chosen for the boundary\n' +
                     'values. Provided number of boundary values (' +
                     str(n_ordinates) + ')\ndid not match provided number' +
                     'of discrete ordinates, which is ' + str(n_bndry_val) +
                     '.')

        self.method = str(config['DISCRETIZATION']['method'])

        solver_name = str(config['SOLVER']['solver'])
        assert solver_name in ['SparseDirect', 'GMRES', 'BiCGSTAB'], \
            'Solver ' + solver_name + ' not implemented.'

        initial_guess = str(config['SOLVER']['initialGuess'])
        assert initial_guess in ['thermalEmission', 'noScattering'], \
            'Initial guess ' + initial_guess + ' unknown.'

        prec_type = str(config['SOLVER']['preconditioner'])
        assert prec_type in ['none', 'lambdaIteration', 'diagonal'], \
            'Preconditioner ' + prec_type + ' currently not supported.'

        self.outputType = str(config['OUTPUT']['type'])

        # define model problem and discretization
        model_problem = modelProblem.ModelProblem(
            temperature, frequency, albedo, emissivity, scattering,
            absorption_coeff.abs_fun, boundary_values)

        # time mesh generation
        start_time = timeit.default_timer()
        self.mesh = mesh.UniformMesh(dimension, domain, n_cells)
        elapsed_time = timeit.default_timer() - start_time
        mesh_time = elapsed_time

        assert(self.method == 'finiteVolume')

        # time matrix and load vector assembly
        start_time = timeit.default_timer()

        if scattering == 'isotropic':

            disc = discretization.FiniteVolume1d(
                model_problem, self.mesh, n_ordinates, boundary_values, flux)

        else:

            disc = discretization.FiniteVolume1d(
                model_problem, self.mesh, n_ordinates, boundary_values, flux)

        elapsed_time = timeit.default_timer() - start_time

        print('Timings:')
        print('--------')
        print('Mesh generation: ' +
              "% 10.3e" % (mesh_time) + ' s')
        print('Matrix and rhs assembly: ' +
              "% 10.3e" % (elapsed_time) + ' s')

        prec = None
        x_in = None

        if not solver_name == 'SparseDirect':

            # time precoditioner setup
            start_time = timeit.default_timer()

            prec = solver.Preconditioner(disc, prec_type)

            elapsed_time = timeit.default_timer() - start_time
            print('Preconditioner setup:    ' +
                  "% 10.3e" % (elapsed_time) + ' s')

            # time initial guess setup
            start_time = timeit.default_timer()

            if initial_guess == "thermalEmission":

                x_in = np.full(disc.n_dof, model_problem.s_e)

            elif initial_guess == "noScattering":

                sol1 = disc.inflow_bc[0] * \
                    np.exp(-self.mesh.cell_centers()) + model_problem.s_e * \
                    (1 - np.exp(-self.mesh.cell_centers()))

                sol2 = disc.inflow_bc[1] * \
                    np.exp(-self.mesh.cell_centers()[::-1]) + \
                    model_problem.s_e * \
                    (1 - np.exp(-self.mesh.cell_centers()[::-1]))

                x_in = np.concatenate((sol1, sol2), axis=0)

            else:

                x_in = None

            elapsed_time = timeit.default_timer() - start_time
            print('Initial guess setup:     ' +
                  "% 10.3e" % (elapsed_time) + ' s')

        A, b = disc.stiff_mat, disc.load_vec

        linear_solver = solver.Solver(solver_name, prec)

        self.x, self.iters, self.elapsed_time = linear_solver.solve(A, b, x_in)

    def output_results(self):

        if self.outputType == "firstOrdinate":

            if self.method == "finiteVolume":

                plt.step(self.mesh.cell_centers(),
                         self.x[:self.mesh.n_cells[0]])

            else:

                plt.plot(self.mesh.cell_centers(),
                         self.x[:self.mesh.n_cells[0]])

        elif self.outputType == "meanIntensity":

            if self.method == "finiteVolume":

                plt.step(self.mesh.cell_centers(), np.mean(
                    (self.x[self.mesh.n_cells[0]:],
                     self.x[:self.mesh.n_cells[0]]),
                    axis=0))

            else:

                plt.plot(self.mesh.cell_centers(), np.mean(
                    (self.x[self.mesh.n_cells[0]:],
                     self.x[:self.mesh.n_cells[0]]),
                    axis=0))

        plt.show()


if __name__ == "__main__":

    radtrans = RadiativeTransfer()
    radtrans.main(sys.argv)
    radtrans.output_results()
