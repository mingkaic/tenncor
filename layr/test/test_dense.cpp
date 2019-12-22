
#ifndef DISABLE_DENSE_TEST


#include "gtest/gtest.h"

#include "dbg/stream/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "eteq/serialize.hpp"

#include "layr/api.hpp"

#include <google/protobuf/util/json_util.h>


TEST(DENSE, Connection)
{
	auto x = eteq::make_variable_scalar<float>(
		0, teq::Shape({6, 2}), "x");
	auto x2 = eteq::make_variable_scalar<float>(
		0, teq::Shape({7, 2}), "x2");
	auto biasedy = layr::dense(eteq::ETensor<float>(x),
		{5}, layr::unif_xavier_init<float>(2),
		layr::unif_xavier_init<float>(4));
	auto y = layr::dense(eteq::ETensor<float>(x2),
		{6}, layr::unif_xavier_init<float>(3),
		layr::InitF<float>());

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


TEST(DENSE, Serialization)
{
	onnx::ModelProto model;

	teq::DimT ninput = 6, noutput = 5;
	std::vector<float> weight_data;
	std::vector<float> bias_data;
	{
		auto x = eteq::make_variable_scalar<float>(
			0, teq::Shape({ninput, 2}), "x");
		auto y = layr::dense(eteq::ETensor<float>(x), {noutput},
			layr::unif_xavier_init<float>(2),
			layr::unif_xavier_init<float>(4));

		auto contents = layr::calc_storage<float>(y, x);
		ASSERT_EQ(2, contents.size());
		teq::TensptrT weight, bias;
		if (contents[0]->to_string() == layr::weight_key)
		{
			weight = contents[0];
			bias = contents[1];
		}
		else
		{
			bias = contents[0];
			weight = contents[1];
		}
		float* w = (float*) weight->data();
		float* b = (float*) bias->data();
		weight_data = std::vector<float>(w, w + weight->shape().n_elems());
		bias_data = std::vector<float>(b, b + bias->shape().n_elems());
		EXPECT_GRAPHEQ(
			"(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
			" `--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
			" |   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
			" |   `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
			" `--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"     `--(variable:bias[5\\1\\1\\1\\1\\1\\1\\1])", y);

		eteq::save_model(model, {y});
	}
	ASSERT_EQ(noutput * ninput, weight_data.size());
	ASSERT_EQ(noutput, bias_data.size());
	{
		// load
		onnx::TensptrIdT ids;
		teq::TensptrsT roots = eteq::load_model(ids, model);
		ASSERT_EQ(1, roots.size());

		EXPECT_GRAPHEQ(
			"(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
			" `--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
			" |   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
			" |   `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
			" `--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"     `--(variable:bias[5\\1\\1\\1\\1\\1\\1\\1])", roots.front());
	}
}


#endif // DISABLE_DENSE_TEST
