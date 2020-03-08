import numpy as np
import configparser
from radtrans import RadiativeTransfer
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
radtrans = RadiativeTransfer()
plt.figure()
k = 1
for precond in ['NoPreconditioning','LambdaIteration']:
    plt.subplot(1,2,k)
    for solver in ['GMRES','BiCGSTAB']:
        iters = []
        for alb in np.linspace(0,1,10):
            config['Model'] = {'Dimension': '1',
                               'Temperature': '300',
                               'Frequency': '440e11',
                               'Albedo': alb,
                               'Scattering': 'isotropic',
                               'Domain': '5.0',
                               'BoundaryValues': '1.0,0.0',
                               'QuadratureWeights': '1.0,1.0'}

            config['Discretization'] = {'Name': 'FiniteVolume',
                                        'n_cells': 10}
            config['Solver'] = {'Name': solver,
                                'Preconditioner': precond,
                                'InitialGuess': 'Analytical'}

            with open('benchmarks.ini', 'w') as configfile:
                config.write(configfile)
            radtrans.main(['file: ','benchmarks.ini'])
            iters.append(radtrans.iters) 

        plt.title(r'1D, iso. scat., BVs: 1.0,0.0, '+precond)
        plt.plot(iters,np.linspace(0,1,10),label=solver)
        plt.ylabel(r'$\eta$')
        plt.xlabel('#iters')
        plt.legend()
    k+=1

plt.show()

