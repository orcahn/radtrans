# radtrans

Programming project as part of the lecture 'Computational Methods for the
Interaction of Light and Matter' held in conjunction by Prof. G. Kanschat and
Prof. C. Dullemond at Heidelberg University.

The project contains software that solves the (1D or 2D) radiative transfer equation
for a homogeneous but anisotropic medium, in which isotropic scattering takes place.
Radiation wavelength, as well as the temperature, emissivity and albedo of the medium
are constant per simulation, but can be chosen freely from one simulation to the 
next. This allows for the computation of temperature as well as line profiles.
There are several options for boundary conditons provided and custom ones can be implemented 
with little overhead.

The project uses a Finite Volume Method on a uniform, quadrilateral grid for the discretization of the equations. Furthermore it 
uses a variable number of discrete ordinates in the approximation of the scattering process. The 
quadrature weights were chosen such, that energy conservation is satisfied. In solving the discretized
system, there is the option for using either a generalization of the Lambda Iteration or the approximation
by an elliptic operator corresponding to the diffusion limit. Benchmarks suggest using the Lambda Iteration
for albedo values below a threshold value of 0.9 and the diffusion limit for values larger than the threshold.
Throughout the project, sparse matrix formats provided by the scipy.sparse software library are used.

As for linear solvers, there are options for a sparse direct solver (using SuperLU), as well as BiCGSTAB and 
GMRES iterative solvers. For the latter two, initial guesses corresponding to the zero vector, thermal emission
of the medium and a problem absent of scattering can be chosen. Benchmarks suggest the use of the direct solver
for less than or about 1e6 degrees of freedom, if memory permits. For larger systems the iterative solvers provide
substantial benefits, particularly in terms of memory.

# Dependencies

# Structure
% how to use
# Authors

Jonas Roller and Robert Kutri
