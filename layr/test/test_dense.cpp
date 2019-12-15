
#ifndef DISABLE_DENSE_TEST


#include "gtest/gtest.h"

#include "dbg/stream/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "layr/api.hpp"


TEST(DENSE, Connection)
{
	auto x = eteq::make_variable_scalar<float>(
		0, teq::Shape({6, 2}), "x");
	auto x2 = eteq::make_variable_scalar<float>(
		0, teq::Shape({7, 2}), "x2");
	auto biasedy = layr::dense(eteq::to_link<float>(x),
		{5}, layr::unif_xavier_init<float>(2),
		layr::unif_xavier_init<float>(4));
	auto y = layr::dense(eteq::to_link<float>(x2),
		{6}, layr::unif_xavier_init<float>(3),
		layr::InitF<float>());

	EXPECT_GRAPHEQ(
		"(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:bias[5\\1\\1\\1\\1\\1\\1\\1])",
		biasedy->get_root());

	EXPECT_GRAPHEQ(
		"(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:x2[7\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])",
		y->get_root());
}


// TEST(DENSE, Serialization)
// {
// 	std::stringstream ss;
// 	std::string label = "very_dense";
// 	teq::DimT ninput = 6, noutput = 5;
// 	std::vector<float> weight_data;
// 	std::vector<float> bias_data;
// 	{
// 		// save
// 		layr::Dense dense(noutput, teq::Shape({ninput}),
// 			layr::unif_xavier_init<float>(2),
// 			layr::unif_xavier_init<float>(4),
// 			nullptr,
// 			label);

// 		auto contents = dense.get_contents();
// 		ASSERT_EQ(3, contents.size());
// 		auto weight = contents[0];
// 		auto bias = contents[1];
// 		auto params = contents[2];
// 		ASSERT_EQ(nullptr, params);
// 		float* w = eteq::to_link<float>(weight)->data();
// 		float* b = eteq::to_link<float>(bias)->data();
// 		weight_data = std::vector<float>(w, w + weight->shape().n_elems());
// 		bias_data = std::vector<float>(b, b + bias->shape().n_elems());

// 		auto x = eteq::make_variable_scalar<float>(
// 			0, teq::Shape({6, 2}), "x");
// 		auto y = dense.connect(eteq::to_link<float>(x));

// 		layr::save_layer(ss, dense, {y->get_tensor()});
// 	}
// 	ASSERT_EQ(noutput * ninput, weight_data.size());
// 	ASSERT_EQ(noutput, bias_data.size());
// 	{
// 		// load
// 		teq::TensptrsT roots;
// 		auto dense = layr::load_layer(ss, roots,
// 			layr::dense_layer_key, label);

// 		// verify layer
// 		auto contents = dense->get_contents();
// 		ASSERT_EQ(3, contents.size());
// 		auto weight = contents[0];
// 		auto bias = contents[1];
// 		teq::Shape exwshape({noutput, ninput});
// 		teq::Shape exbshape({noutput});
// 		auto wshape = weight->shape();
// 		auto bshape = bias->shape();
// 		ASSERT_ARREQ(exwshape, wshape);
// 		ASSERT_ARREQ(exbshape, bshape);
// 		float* w = eteq::to_link<float>(weight)->data();
// 		float* b = eteq::to_link<float>(bias)->data();
// 		std::vector<float> gotw(w, w + weight->shape().n_elems());
// 		std::vector<float> gotb(b, b + bias->shape().n_elems());
// 		EXPECT_VECEQ(weight_data, gotw);
// 		EXPECT_VECEQ(bias_data, gotb);

// 		// verify root
// 		ASSERT_EQ(2, roots.size());
// 		EXPECT_GRAPHEQ("(variable:bias[5\\1\\1\\1\\1\\1\\1\\1])", roots[0]);
// 		EXPECT_GRAPHEQ("(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])", roots[1]);
// 	}
// }


#endif // DISABLE_DENSE_TEST
