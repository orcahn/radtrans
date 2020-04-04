import numpy as np
import configparser
import matplotlib.pyplot as plt

from radtrans import RadiativeTransfer


config = configparser.ConfigParser()
radtrans = RadiativeTransfer()
fig, axs = plt.subplots(2, 2)


# iteration number benchmark
k = 0

for precond in ['none', 'lambdaIteration']:
    for solver in ['GMRES', 'BiCGSTAB']:
        iters = []
        for alb in np.linspace(0, 0.999999, 100):
            config['MODEL'] = {'dimension': '1',
                               'temperature': '300',
                               'frequency': '440e11',
                               'albedo': alb,
                               'emissivity': '0.7',
                               'scattering': 'isotropic',
                               'domain': '5.0',
                               'absorptionType': 'step',
                               'boundaryValues': '1.0,0.0',
                               'quadratureWeights': '1.0,1.0'}

            config['DISCRETIZATION'] = {'method': 'finiteVolume',
                                        'flux': 'upwind',
                                        'n_ordinates': '2',
                                        'n_cells': '1000'}
            config['SOLVER'] = {'solver': solver,
                                'Preconditioner': precond,
                                'initialGuess': 'thermalEmission'}

            config['OUTPUT'] = {'type': 'firstOrdinate'}

            with open('benchmarks.ini', 'w') as configfile:
                config.write(configfile)
            radtrans.main(['file: ', 'benchmarks.ini'])
            iters.append(radtrans.iters)

        axs[0, k].set_title(
            r'1D, N=1000, iso. scat., BVs: 1.0,0.0, Step, '+precond)
        axs[0, k].plot(np.linspace(0, 1, 100), iters, label=solver)
        axs[0, k].set_xlabel(r'$\eta$')
        axs[0, k].set_ylabel('#iters')
        axs[0, k].legend()
    k += 1

# execution time benchmarks
k = 0
Ns = [10*x for x in [2**j for j in range(7)]]

for precond in ['diagonal', 'lambdaIteration']:
    for solver in ['GMRES', 'BiCGSTAB']:
        times = []
        for n in Ns:
            config['MODEL'] = {'dimension': '1',
                               'temperature': '300',
                               'frequency': '440e11',
                               'albedo': '0.999',
                               'emissivity': '0.7',
                               'scattering': 'isotropic',
                               'domain': '5.0',
                               'absorptionType': 'gaussianRandomPiecewise',
                               'boundaryValues': '0.0,0.0',
                               'quadratureWeights': '1.0,1.0'}

            config['DISCRETIZATION'] = {'method': 'finiteVolume',
                                        'flux': 'upwind',
                                        'n_ordinates': '2',
                                        'n_cells': n}
            config['SOLVER'] = {'solver': solver,
                                'Preconditioner': precond,
                                'initialGuess': 'thermalEmission'}

            config['OUTPUT'] = {'type': 'firstOrdinate'}

            with open('benchmarks.ini', 'w') as configfile:
                config.write(configfile)
            radtrans.main(['file: ', 'benchmarks.ini'])
            times.append(radtrans.elapsed_time)

        axs[1, k].set_title(r'1D, iso. scat., BVs: 1.0,2.0, Step, '+precond)
        axs[1, k].plot(Ns, times, label=solver)
        axs[1, k].set_xlabel('n_cells')
        axs[1, k].set_ylabel('time (s)')
        axs[1, k].legend()
    k += 1

plt.show()
