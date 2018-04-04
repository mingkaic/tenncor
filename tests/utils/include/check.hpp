#include <cstdarg>
#include <sstream>

#include "tensor/tensorshape.hpp"

#ifndef TTEST_CHECK_HPP
#define TTEST_CHECK_HPP

namespace testutils
{

#define ASSERT_SHAPEQ(shape, shape2) \
ASSERT_TRUE(tensorshape_equal(shape, shape2)) << \
testutils::sprintf("expect shape %p, got shape %p", &shape, &shape2);

#define EXPECT_SHAPEQ(shape, shape2) \
EXPECT_TRUE(tensorshape_equal(shape, shape2)) << \
testutils::sprintf("expect shape %p, got shape %p", &shape, &shape2);

bool tensorshape_equal (
	const nnet::tensorshape& ts1,
	const nnet::tensorshape& ts2);

bool tensorshape_equal (
	const nnet::tensorshape& ts1,
	std::vector<size_t>& ts2);

}

#endif
