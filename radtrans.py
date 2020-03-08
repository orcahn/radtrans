import sys
import configparser
import numpy as np
import matplotlib.pyplot as plt

import modelProblem
import discretization
import solver


class RadiativeTransfer:
    """
    Main class of the project.
    Reads in parameter .ini file, specifies model, discretization method
    and solver parameters and discretizes and solves the radiative transfer
    problem for the given parameters. The numerical solution is then visualized.
    """

    def main(self, argv):
        # parse parameter file
        config = configparser.ConfigParser()
        config.read(argv[1:])
        dimension = int(config['Model']['Dimension'])
        temperature = float(config['Model']['Temperature'])
        frequency = float(config['Model']['Frequency'])
        albedo = float(config['Model']['Albedo'])
        scattering = str(config['Model']['Scattering'])
        domain = config['Model']['Domain']
        boundary_values = [
            float(
                e.strip()) for e in config.get(
                'Model',
                'BoundaryValues').split(',')]
        quadrature_weights = [
            float(
                e.strip()) for e in config.get(
                'Model',
                'QuadratureWeights').split(',')]
        self.method = str(config['Discretization']['Name'])
        self.n_cells = int(config['Discretization']['n_cells'])
        solver_name = str(config['Solver']['Name'])
        preconditioner = str(config['Solver']['Preconditioner'])
        initial_guess = str(config['Solver']['InitialGuess'])
        # define model problem and discretization
        domain = float(domain)  # 1d domain only has one length
        model_problem = modelProblem.ModelProblem1d(
            temperature, frequency, albedo, scattering, domain, boundary_values)
        assert(self.method == "FiniteVolume")
        if scattering == "isotropic":
            self.disc = discretization.FiniteVolume1d(
                model_problem, self.n_cells, quadrature_weights)
        else:
            self.disc = discretization.FiniteVolume1d(model_problem, self.n_cells)

        # define stiffness matrix, load vector, solver and preconditioner
        if preconditioner == "LambdaIteration":
            preconditioner = solver.LambdaPreconditioner(self.disc)
        A, b = self.disc.stiff_mat, self.disc.load_vec

        self.dom = np.arange(0.5 * self.disc.h, self.n_cells * self.disc.h, self.disc.h)

        if initial_guess == "Inflow":
            x_in = np.full(self.disc.n_dof, model_problem.s_eps)
        elif initial_guess == "Analytical":
            sol = 0.5 * (model_problem.inflow_bc[0] * \
                np.exp(-self.dom) + model_problem.s_eps * (1 - np.exp(-self.dom)) + model_problem.inflow_bc[1] * \
                np.exp(-self.dom[::-1]) + model_problem.s_eps * (1 - np.exp(-self.dom[::-1]))) 
            x_in = np.concatenate((sol,np.zeros(self.n_cells)),axis=0)
        else:
            x_in = None
        linear_solver = solver.Solver(solver_name, preconditioner)
        self.x,self.iters = linear_solver.solve(A, b, x_in)

    def output_results(self):
        if self.method == "FiniteVolume":
            plt.step(self.dom, self.x[:self.n_cells])
        else:
            plt.plot(self.dom, self.x)
        plt.show()


if __name__ == "__main__":
    radtrans = RadiativeTransfer()
    radtrans.main(sys.argv)
    radtrans.output_results()
