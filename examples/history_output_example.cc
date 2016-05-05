// Pallas Solver
// Copyright 2015. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: ryan.latture@gmail.com (Ryan Latture)

#include <cmath>
#include <fstream>
#include "glog/logging.h"
#include "rapidjson/stringbuffer.h"

// Each solver is defined in its own header file.
// include the solver you wish you use:
#include "pallas/differential_evolution.h"

// define a problem you wish to solve by inheriting
// from the pallas::GradientCostFunction interface
// and implementing the Evaluate and NumParameters methods.
class Peaks : public pallas::GradientCostFunction {
public:
    virtual ~Peaks() {}

    virtual bool Evaluate(const double* parameters,
                          double* cost,
                          double* gradient) const {
        const double x = parameters[0];
        const double y = parameters[1];
        cost[0] = 3.0 * pow(1.0 - x, 2) * exp(- pow(x, 2) - pow(y + 1, 2))
                  - 10.0 * (x/5.0 - pow(x, 3) - pow(y, 5)) * exp(-pow(x, 2) - pow(y, 2))
                  - 1.0/3.0 * exp(-pow(x+1, 2) - pow(y, 2));

        if (gradient != NULL) {
            gradient[0] = -6.0 * exp(-pow(x, 2) - pow(1.0 + y, 2)) * (1.0 - x)
                          - 6.0 * exp(-pow(x, 2) - pow(1.0 + y, 2)) * pow(1.0 - x, 2) * x
                          + 2.0/3.0 * exp(-pow(1.0 + x, 2) - pow(y, 2)) * (1.0 + x)
                          - 10 * exp(-pow(x, 2) - pow(y, 2)) * (1.0/5.0 - 3.0 * pow(x, 2))
                          + 20.0 * exp(-pow(x, 2) - pow(y, 2)) * x * (x/5.0 - pow(x, 3) - pow(y, 5));
            gradient[1] = 2.0/3.0 * exp(-pow(1.0 + x, 2) - pow(y, 2)) * y + 50.0 * exp(-pow(x, 2) - pow(y, 2)) * pow(y, 4)
                          - 6.0 * exp(-pow(x, 2) - pow(1.0 + y, 2)) * pow(1.0 - x, 2) * (1.0 + y)
                          + 20.0 * exp(-pow(x, 2) - pow(y, 2)) * y * (x/5.0 - pow(x, 3) - pow(y, 5));
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
    pallas::DifferentialEvolution::Options options;

    // save history output every 10 iterations
    options.history_save_frequency = 1;

    // differential evolution requires that upper
    // and lower bounds are placed on the parameters
    // in the form of pallas::Vector which is a
    // just a typedef of an Eigen::VectorXd.
    pallas::Vector upper_bounds(2);
    upper_bounds << 5, 5;

    pallas::Vector lower_bounds(2);
    lower_bounds << -5, -5;

    options.upper_bounds = upper_bounds;
    options.lower_bounds = lower_bounds;

    // initialize a summary object to hold the
    // optimization details
    pallas::DifferentialEvolution::Summary summary;

    // create a problem from your cost function
    pallas::GradientProblem problem(new Peaks());

    // solve the problem and store the optimal position
    // in parameters and the optimization details in
    // the summary
    pallas::Solve(options, problem, parameters, &summary);

    std::cout << summary.FullReport() << "\n";
    std::cout << "Global minimum found at:" << "\n";
    std::cout << "\tx: " << parameters[0] << "\ty: " << parameters[1] << "\n\n";

    // create a string buffer to hold history data
    rapidjson::StringBuffer sb;

    // create the writer that will translate a series
    // of history outputs into a string of JSON data
    // and store the results in our string buffer
    pallas::HistoryWriter writer(sb);

    // dump the history series to the string buffer
    pallas::dump(summary.history, writer);

    // save the history to a disk
    std::string history_filename("history.json");
    std::cout << "Saving history data to: " << history_filename << "\n";
    std::ofstream history_stream(history_filename);
    history_stream << sb.GetString();
    history_stream.close();
    std::cout << "File saved." << std::endl;

    return 0;
}