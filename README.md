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

Building and Installation
-------------------------
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

Example
-------

```cpp
#include "glog/logging.h"

// Each solver is defined in its own header file.
// include the solver you wish you use:
#include "pallas/basinhopping.h"

// define a problem you wish to solve by inheriting
// from the pallas::GradientCostFunction interface
// and implementing the Evaluate and NumParameters methods.
class Rosenbrock : public pallas::GradientCostFunction {
public:
    virtual ~Rosenbrock() {}

    virtual bool Evaluate(const double* parameters,
                          double* cost,
                          double* gradient) const {
        const double x = parameters[0];
        const double y = parameters[1];

        cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
        if (gradient != NULL) {
            gradient[0] = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
            gradient[1] = 200.0 * (y - x * x);
        }
        return true;
    }

    virtual int NumParameters() const { return 2; }
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    // define the starting point for the optimization
    double parameters[2] = {-1.2, 0.0};

    // set up global optimizer options only initialization
    // is need to accept the default options
    pallas::Basinhopping::Options options;

    // initialize a summary object to hold the
    // optimization details
    pallas::Basinhopping::Summary summary;

    // create a problem from your cost function
    pallas::GradientProblem problem(new Rosenbrock());

    // solve the problem and store the optimal position
    // in parameters and the optimization details in
    // the summary
    pallas::Solve(options, problem, parameters, &summary);

    std::cout << summary.FullReport() << std::endl;
    std::cout << "Global minimum found at:" << std::endl;
    std::cout << "\tx: " << parameters[0] << "\ty: " << parameters[1] << std::endl;

    return 0;
    }
```

After compiling and running, the console should display the following:

```
Solver Summary

Parameters                                  2
Line search direction                   LBFGS

Cost:
  Initial                        2.122000e+02
  Final                          3.300841e-27
  Change                         2.122000e+02

Minimizer iterations                       21

Time (in seconds):
  Cost evaluation                      0.0000
  Local minimization                   0.0015
  Step function                        0.0000
  Total                                0.0015

Termination: CONVERGENCE (Maximum number of stagnant iterations reached.)

Global minimum found at:
	x: 1	y: 1
```
This example (among others) can be found in the examples folder.
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
