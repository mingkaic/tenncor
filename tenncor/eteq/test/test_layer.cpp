
#ifndef DISABLE_LAYER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "tenncor/tenncor.hpp"


TEST(LAYER, Dense)
{
	teq::DimT ninput = 6, noutput = 5, ninput2 = 7;
	auto x = eteq::make_variable_scalar<float>(0, teq::Shape({ninput, 2}), "x");
	auto x2 = eteq::make_variable_scalar<float>(0, teq::Shape({ninput2, 2}), "x2");

	teq::TensptrT weight = eteq::make_variable_scalar<float>(0, teq::Shape({noutput, ninput}), "weight");
	teq::TensptrT bias = eteq::make_variable_scalar<float>(0, teq::Shape({noutput}), "bias");

	teq::TensptrT weight2 = eteq::make_variable_scalar<float>(0, teq::Shape({6, ninput2}), "weight");

	auto biasedy = tenncor().nn.dense(eteq::ETensor(x),
		eteq::ETensor(weight), eteq::ETensor(bias));
	auto y = tenncor().nn.dense(eteq::ETensor(x2),
		eteq::ETensor(weight2));

	EXPECT_GRAPHEQ(
		"(IDENTITY<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(ADD<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(MATMUL<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:x<FLOAT>[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:weight<FLOAT>[5\\6\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(variable:bias<FLOAT>[5\\1\\1\\1\\1\\1\\1\\1])", biasedy);

	EXPECT_GRAPHEQ(
		"(IDENTITY<FLOAT>[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MATMUL<FLOAT>[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:x2<FLOAT>[7\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:weight<FLOAT>[6\\7\\1\\1\\1\\1\\1\\1])", y);
}


TEST(LAYER, DenseSerialization)
{
	onnx::ModelProto model;

	teq::DimT ninput = 6, noutput = 5;
	std::vector<float> weight_data;
	std::vector<float> bias_data;
	{
		auto x = eteq::make_variable_scalar<float>(0, teq::Shape({ninput, 2}), "x");
		eteq::VarptrT<float> weight = eteq::make_variable_scalar<float>(
			0, teq::Shape({noutput, ninput}), "weight");
		eteq::VarptrT<float> bias = eteq::make_variable_scalar<float>(
			0, teq::Shape({noutput}), "bias");
		auto y = tenncor().nn.dense(eteq::ETensor(x),
			eteq::ETensor(weight), eteq::ETensor(bias));
		eteq::VarptrsT<float> contents = layr::get_storage<float>(y);
		ASSERT_EQ(2, contents.size());
		EXPECT_ARRHAS(contents, weight);
		EXPECT_ARRHAS(contents, bias);

		float* w = (float*) weight->device().data();
		float* b = (float*) bias->device().data();
		weight_data = std::vector<float>(w, w + weight->shape().n_elems());
		bias_data = std::vector<float>(b, b + bias->shape().n_elems());
		EXPECT_GRAPHEQ(
			"(IDENTITY<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_`--(ADD<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____`--(MATMUL<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____|___`--(variable:x<FLOAT>[6\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____|___`--(variable:weight<FLOAT>[5\\6\\1\\1\\1\\1\\1\\1])\n"
			"_____`--(EXTEND<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_________`--(variable:bias<FLOAT>[5\\1\\1\\1\\1\\1\\1\\1])", y);

		tcr::save_model(model, {y});
	}
	ASSERT_EQ(noutput * ninput, weight_data.size());
	ASSERT_EQ(noutput, bias_data.size());
	{
		// load
		onnx::TensptrIdT ids;
		auto roots = tcr::load_model(ids, model);
		ASSERT_EQ(1, roots.size());

		EXPECT_GRAPHEQ(
			"(IDENTITY<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_`--(ADD<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____`--(MATMUL<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____|___`--(variable:x<FLOAT>[6\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____|___`--(variable:weight<FLOAT>[5\\6\\1\\1\\1\\1\\1\\1])\n"
			"_____`--(EXTEND<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_________`--(variable:bias<FLOAT>[5\\1\\1\\1\\1\\1\\1\\1])", roots.front());
	}
}


TEST(LAYER, Conv)
{
	auto x = eteq::make_variable_scalar<float>(
		0, teq::Shape({4, 10, 9, 2}), "x");

	std::pair<teq::DimT,teq::DimT> filters = {6, 5};
	teq::DimT indim = 4;
	teq::DimT outdim = 3;
	teq::TensptrT weight = eteq::make_variable_scalar<float>(0,
		teq::Shape({outdim, indim, filters.second, filters.first}), "weight");
	teq::TensptrT bias = eteq::make_variable_scalar<float>(0, teq::Shape({outdim}), "bias");
	auto y = tenncor().nn.conv(eteq::ETensor(x),
		eteq::ETensor(weight), eteq::ETensor(bias));

	EXPECT_GRAPHEQ(
		"(IDENTITY<FLOAT>[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_`--(ADD<FLOAT>[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_____`--(PERMUTE<FLOAT>[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_____|___`--(CONV<FLOAT>[1\\6\\4\\2\\3\\1\\1\\1])\n"
		"_____|_______`--(PAD<FLOAT>[4\\10\\9\\2\\5\\1\\1\\1])\n"
		"_____|_______|___`--(variable:x<FLOAT>[4\\10\\9\\2\\1\\1\\1\\1])\n"
		"_____|_______`--(REVERSE<FLOAT>[3\\4\\5\\6\\1\\1\\1\\1])\n"
		"_____|___________`--(variable:weight<FLOAT>[3\\4\\5\\6\\1\\1\\1\\1])\n"
		"_____`--(EXTEND<FLOAT>[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_________`--(variable:bias<FLOAT>[3\\1\\1\\1\\1\\1\\1\\1])", y);
}


#endif
