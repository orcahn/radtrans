# radtrans

Programming project as part of the lecture 'Computational Methods for the
Interaction of Light and Matter' held in conjunction by Prof. G. Kanschat and
Prof. C. Dullemond at Heidelberg University.


## About

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
substantial benefits, particularly in terms of memory consumption.


## Dependencies

We made extensive use of the **numpy** (https://numpy.org/) and **scipy** (https://scipy.org/) open-source software
packages for numerical calculations. The project was tested using version 1.17 of numpy and 1.2.3 of scipy.

Properly rendering the text in the images displaying the results of the simulation requires a working **LaTeX** installation,
**dvipng** (which may be included with your LaTeX installation), and **Ghostscript** (GPL Ghostscript 9.0 or later is
required). Alternatively, omitting these, the annotations and labels will be displayed using plain text.


## Usage

In order to perform a simulation, the parameters of the problem along with the desired options for discretization and solvers 
have to be set in the **setup.ini** file. Once this is done, the simulation is performed by passing the setup.ini file as a command
line argument to **radtrans.py**, which contains the driver for the simulation.

E.g. in a Linux shell, a simulation with setup determined by setup.ini is run with 
```
python3 radtrans.py setup.ini
```
or
```
ipython3 radtrans.py setup.ini
```


## Authors

Jonas Roller and Robert Kutri
