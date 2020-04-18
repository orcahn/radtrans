import sys
import timeit
import configparser

import numpy as np

import modelProblem
import absorption
import mesh
import discretization
import solver
import visualization


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

        self.n_ord = config.getint('DISCRETIZATION', 'n_ordinates')
        n_cells = tuple(
            map(int,
                config['DISCRETIZATION']['n_cells'].strip().split(',')))

        # in one dimension there are only two possible discrete ordinates
        if dimension == 1 and self.n_ord != 2:

            print('\nWarning: In one dimension two discrete ordinates' +
                  '(+1.0, -1.0) will be used!')
            self.n_ord = 2

        flux = str(config['DISCRETIZATION']['flux'])

        boundary_values = None

        if config['BOUNDARY_VALUES']['type'] == 'uniform':

            boundary_values = tuple(float(config['BOUNDARY_VALUES']['value'])
                                    for m in range(self.n_ord))

        elif config['BOUNDARY_VALUES']['type'] == 'inc_west':

            boundary_values = (config.getfloat('BOUNDARY_VALUES', 'value'),
                               *[0.0 for m in range(self.n_ord - 1)])

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
                     str(n_bndry_val) + ')\ndid not match provided number' +
                     'of discrete ordinates, which is ' + str(self.n_ord) +
                     '.')

        method = str(config['DISCRETIZATION']['method'])
        quad_method = str(config['DISCRETIZATION']['quadrature'])

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
            absorption_coeff.abs_fun)

        # time mesh generation
        start_time = timeit.default_timer()
        self.mesh = mesh.UniformMesh(dimension, domain, n_cells)
        elapsed_time = timeit.default_timer() - start_time
        mesh_time = elapsed_time

        assert(method == 'finiteVolume')

        # time matrix and load vector assembly
        start_time = timeit.default_timer()

        if scattering == 'isotropic':

            disc = discretization.FiniteVolume(
                model_problem, self.mesh, self.n_ord, boundary_values, flux,
                quad_method)

        else:

            disc = discretization.FiniteVolume(
                model_problem, self.mesh, self.n_ord, boundary_values, flux,
                quad_method)

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

                if self.mesh.dim == 1:

                    sol1 = disc.inflow_bc[0] * \
                        np.exp(-self.mesh.cell_centers()) + \
                        model_problem.s_e * \
                        (1 - np.exp(-self.mesh.cell_centers()))

                    sol2 = disc.inflow_bc[1] * \
                        np.exp(-self.mesh.cell_centers()[::-1]) + \
                        model_problem.s_e * \
                        (1 - np.exp(-self.mesh.cell_centers()[::-1]))

                    x_in = np.concatenate((sol1, sol2), axis=0)

                else:

                    # create model problem without scattering
                    ns_mp = model_problem
                    ns_mp.scat = 'none'

                    ns_disc = discretization.FiniteVolume(
                        ns_mp, self.mesh, self.n_ord, boundary_values, flux,
                        quad_method, False)

                    # one step of lambda iteration
                    x_in = solver.invert_transport(ns_disc.stiff_mat,
                                                   ns_disc.load_vec,
                                                   ns_disc.n_dof,
                                                   self.n_ord)

            else:

                x_in = None

            elapsed_time = timeit.default_timer() - start_time
            print('Initial guess setup:     ' +
                  "% 10.3e" % (elapsed_time) + ' s')

        A, b = disc.stiff_mat, disc.load_vec

        linear_solver = solver.Solver(solver_name, prec)

        self.x = linear_solver.solve(A, b, x_in)[0]

    def visualize(self):

        visualization.visualize(
            self.x, self.mesh, self.n_ord, self.outputType)


if __name__ == "__main__":

    radtrans = RadiativeTransfer()
    radtrans.main(sys.argv)
    radtrans.visualize()
