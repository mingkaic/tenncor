#include <cstdarg>
#include <sstream>

#include "tensor/tensorshape.hpp"

#ifndef TTEST_CHECK_HPP
#define TTEST_CHECK_HPP

namespace testutils
{

bool tensorshape_equal (
	const nnet::tensorshape& ts1,
	const nnet::tensorshape& ts2);

bool tensorshape_equal (
	const nnet::tensorshape& ts1,
	std::vector<size_t>& ts2);

}

#endif
