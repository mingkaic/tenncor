/*!
 *
 *  futils.hpp
 *  cnnet
 *
 *  Purpose:
 *  define commonly used activation functions
 *  and useful graph operations
 *
 *  Created by Mingkai Chen on 2016-09-30.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/operations/operations.hpp"

#ifndef TENNCOR_FUTILS_HPP
#define TENNCOR_FUTILS_HPP

namespace nnet
{

//! sigmoid function: f(x) = 1/(1+e^-x)
varptr sigmoid (varptr x);

//! tanh function: f(x) = (e^(2*x)+1)/(e^(2*x)-1)
varptr tanh (varptr x);

//! softmax function: f(x) = e^x / sum(e^x)
varptr softmax (varptr x);

}

#endif /* TENNCOR_FUTILS_HPP */
