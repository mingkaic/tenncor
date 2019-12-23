
#ifndef DISABLE_CONV_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "layr/api.hpp"


TEST(CONV, Connection)
{
	auto conv = layr::conv<float>({6, 5}, 4, 3);

	auto x = eteq::make_variable_scalar<float>(
		0, teq::Shape({4, 10, 9, 2}), "x");
	auto y = conv.connect(eteq::ETensor<float>(x));

	EXPECT_GRAPHEQ(
		"(ADD[3\\6\\4\\2\\1\\1\\1\\1])\n"
		" `--(PERMUTE[3\\6\\4\\2\\1\\1\\1\\1])\n"
		" |   `--(CONV[1\\6\\4\\2\\3\\1\\1\\1])\n"
		" |       `--(PAD[4\\10\\9\\2\\5\\1\\1\\1])\n"
		" |       |   `--(variable:x[4\\10\\9\\2\\1\\1\\1\\1])\n"
		" |       `--(REVERSE[3\\4\\5\\6\\1\\1\\1\\1])\n"
		" |           `--(variable:weight[3\\4\\5\\6\\1\\1\\1\\1])\n"
		" `--(EXTEND[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"     `--(variable:bias[3\\1\\1\\1\\1\\1\\1\\1])", y);
}


#endif // DISABLE_CONV_TEST
