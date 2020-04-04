import configparser

from radtrans import RadiativeTransfer
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
radtrans = RadiativeTransfer()
plt.figure()

# iteration number benchmark

k = 1
"""
for precond in ['none', 'LambdaIteration']:
    plt.subplot(1, 2, k)
    for solver in ['GMRES', 'BiCGSTAB']:
        iters = []
        for alb in np.linspace(0, 0.999999, 100):
            config['MODEL'] = {'dimension': '1',
                               'temperature': '300',
                               'frequency': '440e11',
                               'albedo': alb,
                               'scattering': 'isotropic',
                               'domain': '5.0',
                               'absorptionType': 'step',
                               'boundaryValues': '1.0,0.0',
                               'quadratureWeights': '1.0,1.0'}

            config['DISCRETIZATION'] = {'method': 'finiteVolume',
                                        'n_cells': '1000'}
            config['SOLVER'] = {'solver': solver,
                                'Preconditioner': precond,
                                'initialGuess': 'thermalEmission'}

            config['OUTPUT'] = {'type': 'firstOrdinate'}
            with open('benchmarks.ini', 'w') as configfile:
                config.write(configfile)
            radtrans.main(['file: ', 'benchmarks.ini'])
            iters.append(radtrans.iters)

        plt.title(r'1D, N=1000, iso. scat., BVs: 1.0,0.0, Step, '+precond)
        plt.plot(np.linspace(0, 1, 100), iters, label=solver)
        plt.xlabel(r'$\eta$')
        plt.ylabel('#iters')
        plt.legend()
    k += 1
"""
# execution time benchmarks
Ns = [10*x for x in [2**j for j in range(11)]]

# for precond in ['NoPreconditioning', 'LambdaIteration']:
for precond in ['LambdaIteration']:
    plt.subplot(2, 2, k)
    for solver in ['CG', 'GMRES', 'BiCGSTAB']:
        times = []
        for n in Ns:
            config['MODEL'] = {'dimension': '1',
                               'temperature': '300',
                               'frequency': '440e11',
                               'albedo': '0.999',
                               'scattering': 'isotropic',
                               'domain': '5.0',
                               'absorptionType': 'gaussianRandomPiecewise',
                               'boundaryValues': '0.0,0.0',
                               'quadratureWeights': '1.0,1.0'}

            config['DISCRETIZATION'] = {'method': 'finiteVolume',
                                        'flux': 'centered',
                                        'n_cells': n}
            config['SOLVER'] = {'solver': solver,
                                'Preconditioner': precond,
                                'initialGuess': 'thermalEmission'}

            config['OUTPUT'] = {'type': 'firstOrdinate'}

            config['OUTPUT'] = {'type': 'firstOrdinate'}

            with open('benchmarks.ini', 'w') as configfile:
                config.write(configfile)
            radtrans.main(['file: ', 'benchmarks.ini'])
            times.append(radtrans.elapsed_time)

        plt.title(r'1D, iso. scat., BVs: 1.0,2.0, Step, '+precond)
        plt.plot(Ns, times, label=solver)
        plt.xlabel('n_cells')
        plt.ylabel('time (s)')
        plt.legend()
    k += 1

plt.show()
