
#ifndef DISABLE_API_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "layr/api.hpp"


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


TEST(RBM, Connection)
{
	auto rrbm = layr::rbm<float>(5, 6, layr::unif_xavier_init<float>(2), layr::unif_xavier_init<float>(4));
	auto nobias = layr::rbm<float>(6, 7, layr::unif_xavier_init<float>(3), layr::InitF<float>());

	auto x = eteq::make_variable_scalar<float>(0, teq::Shape({6, 2}), "x");
	auto x2 = eteq::make_variable_scalar<float>(0, teq::Shape({7, 2}), "x2");
	auto biasedy = rrbm.fwd_.connect(eteq::ETensor<float>(x));
	auto y = nobias.fwd_.connect(eteq::ETensor<float>(x2));

	EXPECT_GRAPHEQ(
		"(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:hbias[5\\1\\1\\1\\1\\1\\1\\1])", biasedy);

	EXPECT_GRAPHEQ(
		"(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:x2[7\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])", y);
}


TEST(RBM, BackwardConnection)
{
	auto rrbm = layr::rbm<float>(5, 6, layr::unif_xavier_init<float>(2), layr::unif_xavier_init<float>(4));
	auto nobias = layr::rbm<float>(6, 7, layr::unif_xavier_init<float>(3), layr::InitF<float>());

	auto y = eteq::make_variable_scalar<float>(0, teq::Shape({5, 2}), "y");
	auto y2 = eteq::make_variable_scalar<float>(0, teq::Shape({6, 2}), "y2");
	auto biasedx = rrbm.bwd_.connect(eteq::ETensor<float>(y));
	auto x = nobias.bwd_.connect(eteq::ETensor<float>(y2));

	EXPECT_GRAPHEQ(
		"(ADD[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:y[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(PERMUTE[6\\5\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:vbias[6\\1\\1\\1\\1\\1\\1\\1])", biasedx);

	EXPECT_GRAPHEQ(
		"(MATMUL[7\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:y2[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(PERMUTE[7\\6\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])", x);
}


#endif // DISABLE_API_TEST
