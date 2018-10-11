#include "gtest/gtest.h"

#include "anteroc/testcase.hpp"

#include "llo/api.hpp"


static ade::Shape get_shape (GeneratedCase& gcase)
{
	auto& inputs = gcase.inputs();
	auto it = inputs.find("shape");
	if (inputs.end() == it)
	{
		throw std::runtime_error("shape not found");
	}
	assert(testify::INT64 == it->second.dtype());
	testify::Int64s arr;
	it->second.data().UnpackTo(&arr);
	auto temp = arr.data();
	std::vector<ade::DimT> slist(temp.begin(), temp.end());
	return ade::Shape(slist);
}


static std::vector<double> get_data (GeneratedCase& gcase, std::string key)
{
	auto& inputs = gcase.inputs();
	auto it = inputs.find(key);
	if (inputs.end() == it)
	{
		throw std::runtime_error(key + " not found");
	}
	assert(testify::DOUBLE == it->second.dtype());
	testify::Doubles arr;
	it->second.data().UnpackTo(&arr);
	auto temp = arr.data();
	return std::vector<double>(temp.begin(), temp.end());
}


static void unary_op (std::string tname,
	std::function<ade::Tensorptr(ade::Tensorptr&)> op)
{
	GeneratedCase gcase = get("REGRESS::" + tname);
	ade::Shape shape = get_shape(gcase);
	std::vector<double> data = get_data(gcase, "data");
	std::vector<double> resdata = get_data(gcase, "unary_out");
	std::vector<double> gresdata = get_data(gcase, "unary_ga");

	auto leaf = llo::Source<double>::get(shape, data);
	auto res = op(leaf);
	auto gres = res->gradient(leaf);

	std::vector<double> resd = llo::evaluate(llo::DOUBLE, res.get());
	std::vector<double> gresd = llo::evaluate(llo::DOUBLE, gres.get());

	EXPECT_ARREQ(resdata, resd);
	EXPECT_ARREQ(gresdata, gresd);
}


static void binary_op (std::string tname,
	std::function<ade::Tensorptr(ade::Tensorptr&,ade::Tensorptr&)> op)
{
	GeneratedCase gcase = get("REGRESS::" + tname);
	ade::Shape shape = get_shape(gcase);
	std::vector<double> data = get_data(gcase, "data");
	std::vector<double> data2 = get_data(gcase, "data2");
	std::vector<double> resdata = get_data(gcase, "binary_out");
	std::vector<double> gresdata = get_data(gcase, "binary_ga");
	std::vector<double> gresdata2 = get_data(gcase, "binary_gb");

	auto leaf = llo::Source<double>::get(shape, data);
	auto leaf2 = llo::Source<double>::get(shape, data2);
	auto res = op(leaf, leaf2);
	auto gres = res->gradient(leaf);
	auto gres2 = res->gradient(leaf2);

	std::vector<double> resd = llo::evaluate(llo::DOUBLE, res.get());
	std::vector<double> gresd = llo::evaluate(llo::DOUBLE, gres.get());
	std::vector<double> gresd2 = llo::evaluate(llo::DOUBLE, gres2.get());

	EXPECT_ARREQ(resdata, resd);
	EXPECT_ARREQ(gresdata, gresd);
	EXPECT_ARREQ(gresdata2, gresd2);
}


int main (int argc, char** argv)
{
	char* gen = getenv("GENERATE_MODE");
	antero::INIT(":32768", gen != nullptr);

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();

	antero::SHUTDOWN();
	return ret;
}


struct REGRESS : public Testament {};


TEST_F(REGRESS, Abs)
{
	unary_op("Abs", [](ade::Tensorptr& a)
	{
		return llo::abs(a);
	});
}


TEST_F(REGRESS, Neg)
{
	unary_op("Neg", [](ade::Tensorptr& a)
	{
		return llo::neg(a);
	});
}


TEST_F(REGRESS, Sin)
{
	unary_op("Sin", [](ade::Tensorptr& a)
	{
		return llo::sin(a);
	});
}


TEST_F(REGRESS, Cos)
{
	unary_op("Cos", [](ade::Tensorptr& a)
	{
		return llo::cos(a);
	});
}

TEST_F(REGRESS, Tan)
{
	unary_op("Tan", [](ade::Tensorptr& a)
	{
		return llo::tan(a);
	});
}


TEST_F(REGRESS, Exp)
{
	unary_op("Exp", [](ade::Tensorptr& a)
	{
		return llo::exp(a);
	});
}


TEST_F(REGRESS, Log)
{
	unary_op("Log", [](ade::Tensorptr& a)
	{
		return llo::log(a);
	});
}


TEST_F(REGRESS, Sqrt)
{
	unary_op("Sqrt", [](ade::Tensorptr& a)
	{
		return llo::sqrt(a);
	});
}


TEST_F(REGRESS, Pow)
{
	binary_op("Pow", [](ade::Tensorptr& a, ade::Tensorptr& b)
	{
		return llo::pow(a, b);
	});
}


TEST_F(REGRESS, Add)
{
	binary_op("Add", [](ade::Tensorptr& a, ade::Tensorptr& b)
	{
		return llo::add(a, b);
	});
}


TEST_F(REGRESS, Sub)
{
	binary_op("Sub", [](ade::Tensorptr& a, ade::Tensorptr& b)
	{
		return llo::sub(a, b);
	});
}


TEST_F(REGRESS, Mul)
{
	binary_op("Mul", [](ade::Tensorptr& a, ade::Tensorptr& b)
	{
		return llo::mul(a, b);
	});
}


TEST_F(REGRESS, Div)
{
	binary_op("Div", [](ade::Tensorptr& a, ade::Tensorptr& b)
	{
		return llo::div(a, b);
	});
}


TEST_F(REGRESS, Matmul)
{
	GeneratedCase gcase = get("REGRESS::Matmul");
	ade::Shape ashape;
	ade::Shape bshape;
	{
		auto& inputs = gcase.inputs();
		auto it = inputs.find("ashape");
		if (inputs.end() == it)
		{
			throw std::runtime_error("ashape not found");
		}
		assert(testify::INT64 == it->second.dtype());
		testify::Int64s arr;
		it->second.data().UnpackTo(&arr);
		auto temp = arr.data();
		std::vector<ade::DimT> slist(temp.begin(), temp.end());
		ashape = ade::Shape(slist);

		auto bit = inputs.find("bdim");
		if (inputs.end() == bit)
		{
			throw std::runtime_error("bdim not found");
		}
		assert(testify::INT64 == bit->second.dtype());
		bit->second.data().UnpackTo(&arr);
		ade::DimT bdim = arr.data()[0];
		bshape = ade::Shape({bdim, slist[0]});
	}

	std::vector<double> data = get_data(gcase, "data");
	std::vector<double> data2 = get_data(gcase, "data2");
	std::vector<double> resdata = get_data(gcase, "matmul_out");
	std::vector<double> gresdata = get_data(gcase, "matmul_ga");
	std::vector<double> gresdata2 = get_data(gcase, "matmul_gb");

	auto leaf = llo::Source<double>::get(ashape, data);
	auto leaf2 = llo::Source<double>::get(bshape, data2);
	auto res = llo::matmul(leaf, leaf2);
	auto gres = res->gradient(leaf);
	auto gres2 = res->gradient(leaf2);

	std::vector<double> resd = llo::evaluate(llo::DOUBLE, res.get());
	std::vector<double> gresd = llo::evaluate(llo::DOUBLE, gres.get());
	std::vector<double> gresd2 = llo::evaluate(llo::DOUBLE, gres2.get());

	EXPECT_ARREQ(resdata, resd);
	EXPECT_ARREQ(gresdata, gresd);
	EXPECT_ARREQ(gresdata2, gresd2);
}
