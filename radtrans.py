import sys
import configparser
import numpy as np
import matplotlib.pyplot as plt

import modelProblem
import discretization
import solver

"""
Main class of the project.
Reads in parameter .ini file, specifies model, discretization method
and solver parameters and discretizes and solves the radiative transfer
problem for the given parameters. The numerical solution is then visualized.
"""


class RadiativeTransfer:

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
        method = str(config['Discretization']['Name'])
        n_cells = int(config['Discretization']['n_cells'])
        solver_name = str(config['Solver']['Name'])
        preconditioner = str(config['Solver']['Preconditioner'])

        # define model problem and discretization
        domain = float(domain)  # 1d domain only has one length
        model_problem = modelProblem.ModelProblem1d(
            temperature, frequency, albedo, scattering, domain, boundary_values)
        assert(method == "FiniteVolume")
        if scattering == "isotropic":
            disc = discretization.FiniteVolume1d(
                model_problem, n_cells, quadrature_weights)
        else:
            disc = discretization.FiniteVolume1d(model_problem, n_cells)

        # define stiffness matrix, load vector, solver and preconditioner
        if preconditioner == "LambdaIteration":
            preconditioner = solver.LambdaPreconditioner(disc)
        A, b = disc.stiff_mat, disc.load_vec

        # physical initial guess in case iterative solver is used
        x_in = np.full(disc.n_dof, model_problem.s_eps)
        linear_solver = solver.Solver(solver_name, preconditioner)
        x = linear_solver.solve(A, b, x_in)

        # output solution
        dom = np.arange(0.5 * disc.h, n_cells * disc.h, disc.h)
        if method == "FiniteVolume":
            plt.step(dom, x[:n_cells])
        else:
            plt.plot(dom, x)
        plt.show()


if __name__ == "__main__":
    radtrans = RadiativeTransfer()
    radtrans.main(sys.argv)
