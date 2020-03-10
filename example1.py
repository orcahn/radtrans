import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt

import discretization
import modelProblem


testModel1 = modelProblem.ModelProblem1d(
    300.0, 440e11, 0.2, 'none', 5.0, [1.0, 0.0])
testModel2 = modelProblem.ModelProblem1d(
    300.0, 440e11, 0.2, 'none', 5.0, [1.0, 2.0])
testModel3 = modelProblem.ModelProblem1d(
    300.0, 440e11, 0.2, 'isotropic', 5.0, [1.0, 0.0])
testModel4 = modelProblem.ModelProblem1d(
    300.0, 440e11, 0.2, 'isotropic', 5.0, [1.0, 2.0])

N = 10

testDisc1 = discretization.FiniteVolume1d(testModel1, N)
testDisc2 = discretization.FiniteVolume1d(testModel2, N)
testDisc3 = discretization.FiniteVolume1d(testModel3, N, [1.0, 1.0])
testDisc4 = discretization.FiniteVolume1d(testModel4, N, [1.0, 1.0])

dom = np.arange(0.5*testDisc1.h, N * testDisc1.h, testDisc1.h)

x1 = spsla.spsolve(testDisc1.stiff_mat, testDisc1.load_vec)
x2 = spsla.spsolve(testDisc2.stiff_mat, testDisc2.load_vec)
x3 = spsla.spsolve(testDisc3.stiff_mat, testDisc3.load_vec)
x4 = spsla.spsolve(testDisc4.stiff_mat, testDisc4.load_vec)

sol1 = testModel1.inflow_bc[0] * \
    np.exp(-dom) + testModel1.s_eps * (1 - np.exp(-dom))

sol2 = testModel1.inflow_bc[1] * \
    np.exp(-dom) + testModel1.s_eps * (1 - np.exp(-dom))

# lambda iteration


def M_x(x): return spsla.spsolve(testDisc4.lambda_prec, x)


M = spsla.LinearOperator((testDisc4.n_dof, testDisc4.n_dof), M_x)

# 0 indicates successful convergence
# initial guess motivated by physics
x5, exitCode0 = spsla.gmres(A=testDisc4.stiff_mat, b=testDisc4.load_vec, M=M,
                            x0=np.full(testDisc4.n_dof, testModel4.s_eps),
                            tol=1e-8)

# initial guess: analytic solution, i.e. no scattering
x6, exitCode1 = spsla.gmres(A=testDisc3.stiff_mat, b=testDisc3.load_vec, M=M,
                            x0=np.concatenate(
                                sol1, np.flip(sol2, axis=0), axis=0),
                            tol=1e-8)

print("gmres exit code: " + str(exitCode0))
print("gmres exit code: " + str(exitCode1))

allclose = np.allclose(x4, x5) and np.allclose(x3, x6)

if allclose:
    print("GMRES close to direct solve.")
else:
    print("GMRES differs from direct solve")

plt.step(dom, x1[:N], label='LeftBC_noScat', where='mid')
plt.step(dom, x2[:N], label='bothBC_noScat', where='mid')
plt.step(dom, x3[:N], label='LeftBC_isoScat', where='mid')
plt.step(dom, x4[:N], label='bothBC_isoScat', where='mid')
plt.step(dom, x5[:N], label='GMRES_bothBC_isoScat_sEpsIniGuess', where='mid')
plt.step(dom, x6[:N], label='GMRES_bothBC_isoScat_noScatIniGuess', where='mid')
plt.plot(dom, sol, label='ANALYTIC_LeftBC_noScat')
plt.legend()
plt.show()
