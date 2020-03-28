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
        config = configparser.ConfigParser()
        config.read(argv[1:])

        dimension = int(config['MODEL']['dimension'])
        temperature = float(config['MODEL']['temperature'])
        frequency = float(config['MODEL']['frequency'])
        albedo = float(config['MODEL']['albedo'])
        emissivity = float(config['MODEL']['emissivity'])
        domain = float(config['MODEL']['domain'])
        abs_type = config['MODEL']['absorptionType']

        scattering = str(config['MODEL']['scattering'])
        assert scattering in ['none', 'isotropic'], \
            'Scattering type ' + scattering + ' currently not supported.'

        absorption_coeff = absorption.Absorption(abs_type, domain)

        boundary_values = [
            float(
                e.strip()) for e in config.get(
                'MODEL',
                'boundaryValues').split(',')]

        quadrature_weights = [
            float(
                e.strip()) for e in config.get(
                'MODEL',
                'quadratureWeights').split(',')]

        self.method = str(config['DISCRETIZATION']['method'])

        n_cells = int(config['DISCRETIZATION']['n_cells'])
        n_ordinates = int(config['DISCRETIZATION']['n_ordinates'])
        flux = str(config['DISCRETIZATION']['flux'])

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
            dimension, temperature, frequency, albedo, emissivity, scattering,
            absorption_coeff.abs_fun, boundary_values)

        self.mesh = mesh.Mesh(domain, n_cells)

        assert(self.method == 'finiteVolume')

        # time matrix and load vector assembly
        start_time = timeit.default_timer()

        if scattering == 'isotropic':

            disc = discretization.FiniteVolume1d(
                model_problem, self.mesh, n_ordinates, quadrature_weights,
                flux)

        else:

            disc = discretization.FiniteVolume1d(
                model_problem, self.mesh, n_ordinates, quadrature_weights,
                flux)

        elapsed_time = timeit.default_timer() - start_time
        print('Timings:')
        print('--------')
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

                sol1 = model_problem.inflow_bc[0] * \
                    np.exp(-self.mesh.cell_centers()) + model_problem.s_e * \
                    (1 - np.exp(-self.mesh.cell_centers()))

                sol2 = model_problem.inflow_bc[1] * \
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

                plt.step(self.mesh.cell_centers(), self.x[:self.mesh.n_cells])

            else:

                plt.plot(self.mesh.cell_centers(), self.x[:self.mesh.n_cells])

        elif self.outputType == "meanIntensity":

            if self.method == "finiteVolume":

                plt.step(self.mesh.cell_centers(), np.mean(
                    (self.x[self.mesh.n_cells:], self.x[:self.mesh.n_cells]),
                    axis=0))

            else:

                plt.plot(self.mesh.cell_centers(), np.mean(
                    (self.x[self.mesh.n_cells:], self.x[:self.mesh.n_cells]),
                    axis=0))

        plt.show()


if __name__ == "__main__":

    radtrans = RadiativeTransfer()
    radtrans.main(sys.argv)
    radtrans.output_results()
