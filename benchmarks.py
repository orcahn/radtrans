import numpy as np
import configparser
import time

from radtrans import RadiativeTransfer
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
radtrans = RadiativeTransfer()
plt.figure()

# iteration number benchmark
k = 1
for precond in ['NoPreconditioning','LambdaIteration']:
    plt.subplot(2,2,k)
    for solver in ['GMRES','BiCGSTAB']:
        iters = []
        for alb in np.linspace(0,0.999999,10):
            config['Model'] = {'Dimension': '1',
                               'Temperature': '300',
                               'Frequency': '440e11',
                               'Albedo': alb,
                               'Scattering': 'isotropic',
                               'Domain': '5.0',
                               'Absorption' : 'Step',
                               'BoundaryValues': '1.0,0.0',
                               'QuadratureWeights': '1.0,1.0'}

            config['Discretization'] = {'Name': 'FiniteVolume',
                                        'n_cells': '100'}
            config['Solver'] = {'Name': solver,
                                'Preconditioner': precond,
                                'InitialGuess': 'Inflow'}

            with open('benchmarks.ini', 'w') as configfile:
                config.write(configfile)
            radtrans.main(['file: ','benchmarks.ini'])
            iters.append(radtrans.iters) 

        plt.title(r'1D, N=100, iso. scat., BVs: 1.0,0.0, Step, '+precond)
        plt.plot(np.linspace(0,1,10),iters,label=solver)
        plt.xlabel(r'$\eta$')
        plt.ylabel('#iters')
        plt.legend()
    k+=1

# execution time benchmarks
Ns = [10*x for x in [2**j for j in range(10)]] 

for precond in ['NoPreconditioning','LambdaIteration']:
    plt.subplot(2,2,k)
    for solver in ['GMRES','BiCGSTAB']:
        times = []
        for n in Ns:
            config['Model'] = {'Dimension': '1',
                               'Temperature': '300',
                               'Frequency': '440e11',
                               'Albedo': '0.5',
                               'Scattering': 'isotropic',
                               'Domain': '5.0',
                               'Absorption' : 'Step',
                               'BoundaryValues': '1.0,0.0',
                               'QuadratureWeights': '1.0,1.0'}

            config['Discretization'] = {'Name': 'FiniteVolume',
                                        'n_cells': n}
            config['Solver'] = {'Name': solver,
                                'Preconditioner': precond,
                                'InitialGuess': 'Inflow'}

            with open('benchmarks.ini', 'w') as configfile:
                config.write(configfile)
            radtrans.main(['file: ','benchmarks.ini'])
            times.append(radtrans.elapsed_time) 

        plt.title(r'1D, iso. scat., BVs: 1.0,0.0, Step, '+precond)
        plt.plot(Ns,times,label=solver)
        plt.xlabel('n_cells')
        plt.ylabel('time (s)')
        plt.legend()
    k+=1

plt.show()

