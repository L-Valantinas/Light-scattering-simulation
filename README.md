# Light scattering simulation using the beam propagation method

To run the code download all files in the repository and execute the main.py file. Prerequisite Python libraries include 
numpy, scipy, matplotlib and joblib. 

The program simulates forward light propagation, as the beam passes through scattering objects, which are placed within the refractive
index grid. Built-in scattering objects include a diffusive layer of random refractive index at every pixel and randomly 
placed spheres of uniform refractive index.

The simulation also measures the transmission matrix of the system, which is a mathematical tool for algebraic control over input and 
output electric fields. Using it, the program constructs an input wavefront, which can focus through the scattering system. 
It also allows for indepth analysis of the propagation (e.g. it is possible to single-out high-transmission modes and boost the 
overall transmission of light through the medium).

There are also included functions for the analysis of tilt/tilt and shift/shift optical memory effects using a standard deviation method.
It allows the user to investigate the output scattering correlations of the optical system, as the input beam is tilted 
or shifted by a small amount.




