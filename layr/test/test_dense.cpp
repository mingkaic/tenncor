
#ifndef DISABLE_DENSE_TEST


#include "gtest/gtest.h"

#include "dbg/stream/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "eteq/serialize.hpp"

#include "layr/api.hpp"

#include <google/protobuf/util/json_util.h>


TEST(DENSE, Connection)
{
    teq::Shape shape({6});
    teq::Shape shape2({7});
	auto biased_dense = layr::dense<float>(shape, {5}, layr::unif_xavier_init<float>(2), layr::unif_xavier_init<float>(4));
    auto dense = layr::dense<float>(shape2, {6}, layr::unif_xavier_init<float>(3), layr::InitF<float>());

	auto x = eteq::make_variable_scalar<float>(
		0, teq::Shape({6, 2}), "x");
	auto x2 = eteq::make_variable_scalar<float>(
		0, teq::Shape({7, 2}), "x2");
	auto biasedy = biased_dense.connect(eteq::ETensor<float>(x));
	auto y = dense.connect(eteq::ETensor<float>(x2));

	EXPECT_GRAPHEQ(
		"(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:bias[5\\1\\1\\1\\1\\1\\1\\1])", biasedy);

	EXPECT_GRAPHEQ(
		"(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:x2[7\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])", y);
}


#endif // DISABLE_DENSE_TEST
