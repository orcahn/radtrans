import sys
import configparser
import numpy as np
import matplotlib.pyplot as plt

import modelProblem
import discretization
import solver
import absorption


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
        self.flux = str(config['DISCRETIZATION']['flux'])
        self.n_cells = int(config['DISCRETIZATION']['n_cells'])
        self.n_ordinates = int(config['DISCRETIZATION']['n_ordinates'])

        solver_name = str(config['SOLVER']['solver'])
        initial_guess = str(config['SOLVER']['initialGuess'])

        preconditioner = str(config['SOLVER']['Preconditioner'])
        assert preconditioner in ['none', 'LambdaIteration'], \
            'Preconditioner ' + preconditioner + ' currently not supported.'

        self.outputType = str(config['OUTPUT']['type'])

        # define model problem and discretization
        model_problem = modelProblem.ModelProblem(
            dimension, temperature, frequency, albedo, emissivity, scattering,
            absorption_coeff.abs_fun, domain, boundary_values)

        assert(self.method == 'finiteVolume')

        if scattering == 'isotropic':

            self.disc = discretization.FiniteVolume1d(
              generalizedInterface
                model_problem, self.n_cells, self.n_ordinates,
                quadrature_weights)

        else:

            self.disc = discretization.FiniteVolume1d(
                model_problem, self.n_cells, self.n_ordinates)

        # define stiffness matrix, load vector, solver and preconditioner
        if preconditioner == 'LambdaIteration':
            preconditioner = solver.LambdaPreconditioner(self.disc)

        A, b = self.disc.stiff_mat, self.disc.load_vec

        self.dom = np.arange(
            0.5 * self.disc.mesh.h, self.n_cells * self.disc.mesh.h,
            self.disc.mesh.h)

        if initial_guess == "thermalEmission":

            x_in = np.full(self.disc.n_dof, model_problem.s_e)

        elif initial_guess == "noScattering":

            sol1 = model_problem.inflow_bc[0] * \
                np.exp(-self.dom) + model_problem.s_e * \
                (1 - np.exp(-self.dom))

            sol2 = model_problem.inflow_bc[1] * \
                np.exp(-self.dom[::-1]) + model_problem.s_e * \
                (1 - np.exp(-self.dom[::-1]))

            x_in = np.concatenate((sol1, sol2), axis=0)

        else:

            x_in = None

        linear_solver = solver.Solver(solver_name, preconditioner)

        self.x, self.iters, self.elapsed_time = linear_solver.solve(A, b, x_in)

    def output_results(self):

        if self.outputType == "firstOrdinate":

            if self.method == "finiteVolume":

                plt.step(self.dom, self.x[:self.n_cells])

            else:

                plt.plot(self.dom, self.x[:self.n_cells])

        elif self.outputType == "meanIntensity":

            if self.method == "finiteVolume":

                plt.step(self.dom, np.mean(
                    (self.x[self.n_cells:], self.x[:self.n_cells]), axis=0))

            else:

                plt.plot(self.dom, np.mean(
                    (self.x[self.n_cells:], self.x[:self.n_cells]), axis=0))

        plt.show()


if __name__ == "__main__":

    radtrans = RadiativeTransfer()
    radtrans.main(sys.argv)
    radtrans.output_results()
