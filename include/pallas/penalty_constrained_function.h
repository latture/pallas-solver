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

#ifndef PALLAS_PENALTY_CONSTRAINED_FUNCTION_H
#define PALLAS_PENALTY_CONSTRAINED_FUNCTION_H

#include "pallas/types.h"
#include "pallas/scoped_ptr.h"

namespace pallas {

    class PenaltyConstrainedFunction : GradientCostFunction
    {
    public:
        PenaltyConstrainedFunction(GradientCostFunction* cost_function,
                                   const double* upper_bounds,
                                   const double* lower_bounds);

        virtual bool Evaluate(double* parameters,
                              double* cost,
                              double* gradient) const;

        virtual int NumParameters() const;

    private:
        void clamp_(double* parameters, double* gradient) const;

        scoped_ptr<GradientCostFunction> cost_function_;
        Vector upper_bounds_;
        Vector lower_bounds_;
        const unsigned int num_parameters_;
        const double tolerance_;
    };


} // namespace pallas

#endif // PALLAS_PENALTY_CONSTRAINED_FUNCTION_H