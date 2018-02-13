//
// Created by Mingkai Chen on 2017-04-27.
//

#include "include/utils/futils.hpp"

#ifdef TENNCOR_FUTILS_HPP

namespace nnet
{

varptr sigmoid (varptr x)
{
	return 1 / (1 + exp(-x));
}

varptr tanh (varptr x)
{
	varptr etx = exp(2 * x);
	return (etx - 1) / (etx + 1);
}

varptr softmax (varptr x)
{
	varptr e = exp(x);
	return e / reduce_sum(e);
}

}

#endif
