#include <cstdarg>
#include <sstream>

#include "tensor/tshape.hpp"

#ifndef TTEST_CHECK_HPP
#define TTEST_CHECK_HPP

namespace testutils
{

#define ASSERT_SHAPEQ(shape, shape2) \
ASSERT_TRUE(tshape_equal(shape, shape2)) << \
testutils::sprintf("expect shape %p, got shape %p", &shape, &shape2);

#define EXPECT_SHAPEQ(shape, shape2) \
EXPECT_TRUE(tshape_equal(shape, shape2)) << \
testutils::sprintf("expect shape %p, got shape %p", &shape, &shape2);

bool tshape_equal (
	const nnet::tshape& ts1,
	const nnet::tshape& ts2);

bool tshape_equal (
	const nnet::tshape& ts1,
	std::vector<size_t>& ts2);

}

#endif
