#include "ioutil/stream.hpp"

#ifndef TESTUTIL_CHECK_HPP
#define TESTUTIL_CHECK_HPP

namespace testutil
{

#define ASSERT_ARREQ(arr, arr2) \
ASSERT_TRUE(std::equal(arr.begin(), arr.end(), arr2.begin())) << \
std::string(ioutil::Stream() << "expect list " << arr << ", got " << arr2 << " instead")

#define EXPECT_ARREQ(arr, arr2) \
EXPECT_TRUE(std::equal(arr.begin(), arr.end(), arr2.begin())) << \
std::string(ioutil::Stream() << "expect list " << arr << ", got " << arr2 << " instead")

#define ASSERT_SHAPEQ(shape, shape2) {\
auto l1 = shape.as_list(); \
auto l2 = shape2.as_list(); \
ASSERT_ARREQ(l1, l2); }

#define EXPECT_SHAPEQ(shape, shape2) {\
auto l1 = shape.as_list(); \
auto l2 = shape2.as_list(); \
EXPECT_ARREQ(l1, l2); }

}

#endif
