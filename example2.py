import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt

import modelProblem
import discretization

N = 1000
L = 5.0


# constant absorption throughout the domain. Recovers the case of
# homogeneous medium in the domain
def const_abs(x):

    return np.full(x.shape, 1.0)


# positive gradient from 0.0 at 0 to 1.0 at L
def pos_grad_abs(x):

    return x / L


# gaussian with stddev 1.0, centered at L/2
def gaussian_abs(x):

    return np.exp(-0.5 * (x - 0.5 * L) * (x - 0.5 * L)) / np.sqrt(2.0 * np.pi)


# discontinuous absorption coefficient
def step_abs(x):

    return np.heaviside(x - L / 2.0, 1.0)


testModel1 = modelProblem.ModelProblem1d(
    300.0, 440e11, 0.2, 'isotropic', const_abs, L, [2.0, 1.0])
testModel2 = modelProblem.ModelProblem1d(
    300.0, 440e11, 0.2, 'isotropic', pos_grad_abs, L, [2.0, 1.0])
testModel3 = modelProblem.ModelProblem1d(
    300.0, 440e11, 0.2, 'isotropic', gaussian_abs, L, [2.0, 1.0])
testModel4 = modelProblem.ModelProblem1d(
    300.0, 440e11, 0.2, 'isotropic', step_abs, L, [2.0, 1.0])

testDisc1 = discretization.FiniteVolume1d(testModel1, N, [1.0, 1.0])
testDisc2 = discretization.FiniteVolume1d(testModel2, N, [1.0, 1.0])
testDisc3 = discretization.FiniteVolume1d(testModel3, N, [1.0, 1.0])
testDisc4 = discretization.FiniteVolume1d(testModel4, N, [1.0, 1.0])

x1 = spsla.spsolve(testDisc1.stiff_mat, testDisc1.load_vec)
x2 = spsla.spsolve(testDisc2.stiff_mat, testDisc2.load_vec)
x3 = spsla.spsolve(testDisc3.stiff_mat, testDisc3.load_vec)
x4 = spsla.spsolve(testDisc4.stiff_mat, testDisc4.load_vec)

dom = np.arange(0.5*testDisc1.h, N * testDisc1.h, testDisc1.h)

abs1 = const_abs(dom)
abs2 = pos_grad_abs(dom)
abs3 = gaussian_abs(dom)
abs4 = step_abs(dom)

plt.figure()

plt.subplot(2, 2, 1)
plt.step(dom, x1[:N], label='FV_solution', where='mid')
plt.plot(dom, abs1, label='absorption coefficient')
plt.legend()

plt.subplot(2, 2, 2)
plt.step(dom, x2[:N], label='FV_solution', where='mid')
plt.plot(dom, abs2, label='absorption coefficient')
plt.legend()

plt.subplot(2, 2, 3)
plt.step(dom, x3[:N], label='FV_solution', where='mid')
plt.plot(dom, abs3, label='absorption coefficient')
plt.legend()

plt.subplot(2, 2, 4)
plt.step(dom, x4[:N], label='FV_solution', where='mid')
plt.step(dom, abs4)
plt.legend()

plt.show()
