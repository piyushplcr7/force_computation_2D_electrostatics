# Electrostatic Force computation using BEM

* This code implements the shape derivative of the electrostatic field energy, given in eq. 4.17 in the report, along with the boundary based shape derivative from eq. 3.9. The code is meant for 2D problems and is based on the library 2DParametricBEM (url: https://gitlab.ethz.ch/ppanchal/2dparametricbem).

* Lead developer: Piyush Panchal

## Requirements
* Requires Eigen3, CMake, GNU compiler
* If Eigen 3 unavailable, use "sudo apt-get install -y libeigen3-dev".
* Get the submodule files by "git submodule update --init --recursive".

## Using CMake
* Create a "build" directory in the project directory. Use "cmake .." from inside the build directory. It is recommended to specify the macro "-DCMAKE_BUILD_TPE=Release" to compile in release mode for faster runs.

## Building targets for shape derivative
* From the build folder, execute "make target_name -e mm=i nn=j vel=k" to compile target_name. The environment variables
mm, nn and vel have to be defined in shape derivative computations.
* The environment variables mm and nn represent the m and n values in the polynomial fields x^m y^n
and the sinusoidal fields sin(mx) sin(ny)
* The environment variable vel specifies the direction of the velocity field. For vel=1, the field
is in x direction. For vel=2 it is in the y direction. For torque computations, vel=3 needs to be specified
for using the rotational fields.
* All the compiled executables lie in the folder build/examples

## Target names for net force and torque computations
* For square-shaped D: square_cond (mm=0, nn=0, vel = 1/2/3)
* For kite-shaped D: sq_kite_gip (mm=0, nn=0, vel = 1/2/3)

## Target names for general shape derivatives using sin/poly fields
* Square-shaped D + sin : mp5sin (mm=1..5, nn=1..5, vel=1/2)
* Square-shaped D + poly : mp5poly (mm=1..5, nn=1..5, vel=1/2)
* Square-shaped D (non smooth g) + poly : sq_cond_g (mm=1..5, nn=1..5, vel=1/2)
* Kite-shaped D + sin : sq_kite_gip_sin (mm=1..5, nn=1..5, vel=1/2)
* Kite-shaped D + poly : sq_kite_gip (mm=1..5, nn=1..5, vel=1/2)

## Gramian matrix
* Gramian matrix needs to be computed when evaluating the dual norm errors. There is a program available for that
in the file "gramat.cpp" which can be compiled using "make gramat". The velocity fields used for this computation
are explicitly specified inside the cpp file.
