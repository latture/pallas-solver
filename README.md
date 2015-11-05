Pallas Solver
=============

This project is a suite of global optimization algorithms for C++ inspired by SciPy's global optimization package.
Currently supported functions are basinhopping, brute, differential evolution, and simulated annealing.

Dependencies
------------
 * C++11 compatible compiler.
 * [glog](https://github.com/google/glog)
 * [CMake](http://www.cmake.org/)
 * [Ceres](http://ceres-solver.org/)

Usage
-----
To use this library first install glog and CMake. Pallas is based off of the Google Ceres project which has extensive
use of glog for logging and debugging features, and this functionality is carried over into Pallas. Follow the
instructions to install Ceres at <a href="http://ceres-solver.org/building.html">ceres-solver.org/building.html</a>.
Once CMake, Ceres, and glog are built and installed use the following steps to build Pallas:
    * Navigate to the pallas root directory.
    * On the same level as the `README.md`, create a folder named `build`.
    * In the terminal, navigate to the newly created `build` folder.
    * Execute the following command: `cmake .. -DCERES_DIR:PATH=/path/to/CeresConfig.cmake`, where
      `/path/to/CeresConfig.cmake` denotes the folder where the file `CeresConfig.cmake` is located
      Currently on Linux, Ceres by default will place this file at `/usr/local/share/Ceres`, though
      this may change in the future and may be different on your machine.
    * From within the same build directory execute `make` in the terminal.
This should build Pallas. The folder `build/lib` will hold the library

For more information please see the documentation at: http://latture.github.io/pallas-solver

Contributor(s)
------------
 * Ryan Latture

Credits
-------
This libary uses the local minimization algorithms from Google's Ceres solver.
Implementations of the global optimization algorithms are based on Scipy's
optimize package. Because of the similarities between the Pallas algorithms
and scipy.optimize, much of the documentation was taken from their source.
