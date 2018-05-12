//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_KILN_MODULE_TESTS

#include "gtest/gtest.h"

#include "testutil/fuzz.hpp"
#include "testutil/sgen.hpp"
#include "testutil/check.hpp"

#include "clay/shape.hpp"
#include "clay/dtype.hpp"
#include "kiln/validator.hpp"
#include "kiln/const_init.hpp"
#include "kiln/unif_init.hpp"
#include "kiln/norm_init.hpp"


#ifndef DISABLE_ALL_INIT_TEST


class ALL_INIT : public testutil::fuzz_test {};


using namespace testutil;


TEST_F(ALL_INIT, InvalidSet_A000)
{
	// default
	kiln::ConstInit ci;
	kiln::UnifInit ui;
	kiln::NormInit ni;

	clay::Shape shape = random_shape(this);
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	kiln::Validator valid(shape, {dtype});
	// validated
	kiln::ConstInit vci(valid);
	kiln::UnifInit vui(valid);
	kiln::NormInit vni(valid);

	// string initialized
	unsigned short bsize = clay::type_size(dtype);
	size_t n = get_int(1, "n", {1, 256})[0];
	std::string data = get_string(n * bsize, "data");
	std::string scalar = get_string(bsize, "scalar");
	kiln::ConstInit sci(data, dtype);
	kiln::UnifInit sui(scalar, scalar, dtype);
	kiln::NormInit sni(scalar, scalar, dtype);

	EXPECT_THROW(ci.set<std::string>(data), std::exception);
	EXPECT_THROW(vci.set<std::string>(data), std::exception);
	EXPECT_THROW(sci.set<std::string>(data), std::exception);
	EXPECT_THROW(ci.set<std::string>(std::vector<std::string>{data}), std::exception);
	EXPECT_THROW(vci.set<std::string>(std::vector<std::string>{data}), std::exception);
	EXPECT_THROW(sci.set<std::string>(std::vector<std::string>{data}), std::exception);

	EXPECT_THROW(ui.set<std::string>(data, data), std::exception);
	EXPECT_THROW(vui.set<std::string>(data, data), std::exception);
	EXPECT_THROW(sui.set<std::string>(data, data), std::exception);

	EXPECT_THROW(ni.set<std::string>(data, data), std::exception);
	EXPECT_THROW(vni.set<std::string>(data, data), std::exception);
	EXPECT_THROW(sni.set<std::string>(data, data), std::exception);
}


TEST_F(ALL_INIT, UnsetGet_A001)
{
	// default
	kiln::ConstInit ci;
	kiln::UnifInit ui;
	kiln::NormInit ni;

	clay::Shape shape = random_shape(this);
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	kiln::Validator valid(shape, {dtype});
	// validated
	kiln::ConstInit vci(valid);
	kiln::UnifInit vui(valid);
	kiln::NormInit vni(valid);
	
	clay::Shape cshape = random_def_shape(this);
	ASSERT_EQ(nullptr, ci.get(cshape));
	ASSERT_EQ(nullptr, ui.get(cshape));
	ASSERT_EQ(nullptr, ni.get(cshape));
	ASSERT_EQ(nullptr, vci.get(cshape));
	ASSERT_EQ(nullptr, vui.get(cshape));
	ASSERT_EQ(nullptr, vni.get(cshape));
}


TEST_F(ALL_INIT, RejectType_A002)
{
	clay::Shape shape = random_def_shape(this);
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	kiln::Validator valid(shape, {dtype});
	// validated
	kiln::ConstInit vci(valid);
	kiln::UnifInit vui(valid);
	kiln::NormInit vni(valid);

	unsigned short bsize = clay::type_size(dtype);
	std::string scalar = get_string(bsize, "scalar");
	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double d = *((double*) &scalar[0]);
			vci.set<double>(d);
			vui.set<double>(d, d);
			vni.set<double>(d, d);
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float d = *((float*) &scalar[0]);
			vci.set<float>(d);
			vui.set<float>(d, d);
			vni.set<float>(d, d);
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t d = *((int8_t*) &scalar[0]);
			vci.set<int8_t>(d);
			vui.set<int8_t>(d, d);
			vni.set<int8_t>(d, d);
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t d = *((uint8_t*) &scalar[0]);
			vci.set<uint8_t>(d);
			vui.set<uint8_t>(d, d);
			vni.set<uint8_t>(d, d);
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t d = *((int16_t*) &scalar[0]);
			vci.set<int16_t>(d);
			vui.set<int16_t>(d, d);
			vni.set<int16_t>(d, d);
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t d = *((uint16_t*) &scalar[0]);
			vci.set<uint16_t>(d);
			vui.set<uint16_t>(d, d);
			vni.set<uint16_t>(d, d);
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t d = *((int32_t*) &scalar[0]);
			vci.set<int32_t>(d);
			vui.set<int32_t>(d, d);
			vni.set<int32_t>(d, d);
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t d = *((uint32_t*) &scalar[0]);
			vci.set<uint32_t>(d);
			vui.set<uint32_t>(d, d);
			vni.set<uint32_t>(d, d);
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t d = *((int64_t*) &scalar[0]);
			vci.set<int64_t>(d);
			vui.set<int64_t>(d, d);
			vni.set<int64_t>(d, d);
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t d = *((uint64_t*) &scalar[0]);
			vci.set<uint64_t>(d);
			vui.set<uint64_t>(d, d);
			vni.set<uint64_t>(d, d);
		}
		break;
		default:
		break;
	}

	// string initialized
	size_t n = get_int(1, "n", {1, 256})[0];
	std::string data = get_string(n * bsize, "data");
	kiln::ConstInit sci(data, dtype, valid);
	kiln::UnifInit sui(scalar, scalar, dtype, valid);
	kiln::NormInit sni(scalar, scalar, dtype, valid);
	
	ASSERT_EQ(nullptr, vci.get());
	ASSERT_EQ(nullptr, vui.get());
	ASSERT_EQ(nullptr, vni.get());
	ASSERT_EQ(nullptr, sci.get());
	ASSERT_EQ(nullptr, sui.get());
	ASSERT_EQ(nullptr, sni.get());
	ASSERT_EQ(nullptr, vci.get(shape));
	ASSERT_EQ(nullptr, vui.get(shape));
	ASSERT_EQ(nullptr, vni.get(shape));
	ASSERT_EQ(nullptr, sci.get(shape));
	ASSERT_EQ(nullptr, sui.get(shape));
	ASSERT_EQ(nullptr, sni.get(shape));
}


TEST_F(ALL_INIT, RejectShape_A003)
{
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape shape = clist;
	clay::Shape badshape = make_incompatible(clist);

	size_t typeidx = get_int(1, "dtype", {0, clay::DTYPE::_SENTINEL - 2})[0];
	clay::DTYPE dtype = (clay::DTYPE) (typeidx + 1);
	clay::DTYPE badtype = (clay::DTYPE) ((typeidx + 1) % (clay::DTYPE::_SENTINEL - 1) + 1);
	kiln::Validator valid(badshape, {badtype});
	// validated
	kiln::ConstInit vci(valid);
	kiln::UnifInit vui(valid);
	kiln::NormInit vni(valid);

	unsigned short bsize = clay::type_size(dtype);
	std::string scalar = get_string(bsize, "scalar");
	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double d = *((double*) &scalar[0]);
			vci.set<double>(d);
			vui.set<double>(d, d);
			vni.set<double>(d, d);
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float d = *((float*) &scalar[0]);
			vci.set<float>(d);
			vui.set<float>(d, d);
			vni.set<float>(d, d);
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t d = *((int8_t*) &scalar[0]);
			vci.set<int8_t>(d);
			vui.set<int8_t>(d, d);
			vni.set<int8_t>(d, d);
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t d = *((uint8_t*) &scalar[0]);
			vci.set<uint8_t>(d);
			vui.set<uint8_t>(d, d);
			vni.set<uint8_t>(d, d);
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t d = *((int16_t*) &scalar[0]);
			vci.set<int16_t>(d);
			vui.set<int16_t>(d, d);
			vni.set<int16_t>(d, d);
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t d = *((uint16_t*) &scalar[0]);
			vci.set<uint16_t>(d);
			vui.set<uint16_t>(d, d);
			vni.set<uint16_t>(d, d);
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t d = *((int32_t*) &scalar[0]);
			vci.set<int32_t>(d);
			vui.set<int32_t>(d, d);
			vni.set<int32_t>(d, d);
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t d = *((uint32_t*) &scalar[0]);
			vci.set<uint32_t>(d);
			vui.set<uint32_t>(d, d);
			vni.set<uint32_t>(d, d);
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t d = *((int64_t*) &scalar[0]);
			vci.set<int64_t>(d);
			vui.set<int64_t>(d, d);
			vni.set<int64_t>(d, d);
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t d = *((uint64_t*) &scalar[0]);
			vci.set<uint64_t>(d);
			vui.set<uint64_t>(d, d);
			vni.set<uint64_t>(d, d);
		}
		break;
		default:
		break;
	}

	// string initialized
	size_t n = get_int(1, "n", {1, 256})[0];
	std::string data = get_string(n * bsize, "data");
	kiln::ConstInit sci(data, dtype, valid);
	kiln::UnifInit sui(scalar, scalar, dtype, valid);
	kiln::NormInit sni(scalar, scalar, dtype, valid);

	ASSERT_EQ(nullptr, vci.get(shape));
	ASSERT_EQ(nullptr, vui.get(shape));
	ASSERT_EQ(nullptr, vni.get(shape));
	ASSERT_EQ(nullptr, sci.get(shape));
	ASSERT_EQ(nullptr, sui.get(shape));
	ASSERT_EQ(nullptr, sni.get(shape));
}


TEST_F(ALL_INIT, NoShapeGet_A004)
{
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape shape = clist;
	clay::Shape badshape = make_partial(this, clist);

	kiln::Validator valid(shape, {});
	kiln::Validator badvalid(badshape, {});
	// validated
	kiln::ConstInit vci(valid);
	kiln::UnifInit vui(valid);
	kiln::NormInit vni(valid);
	kiln::ConstInit bvci(badvalid);
	kiln::UnifInit bvui(badvalid);
	kiln::NormInit bvni(badvalid);

	double cdata = get_double(1, "ci")[0];
	std::vector<double> udata = get_double(2, "ui", {-12, 24});
	double umin = std::min(udata[0], udata[1]);
	double umax = std::max(udata[0], udata[1]);
	std::vector<double> ndata = get_double(2, "ni", {0.5, 2});
	double nmean = ndata[0];
	double nstdev = ndata[1];
	vci.set<double>(cdata);
	vui.set<double>(umin, umax);
	vni.set<double>(nmean, nstdev);
	bvci.set<double>(cdata);
	bvui.set<double>(umin, umax);
	bvni.set<double>(nmean, nstdev);

	// string initialized
	kiln::ConstInit sci(
		std::string((char*) &cdata, sizeof(double)), clay::DTYPE::DOUBLE, valid);
	kiln::UnifInit sui(
		std::string((char*) &umin, sizeof(double)), 
		std::string((char*) &umax, sizeof(double)), clay::DTYPE::DOUBLE, valid);
	kiln::NormInit sni(
		std::string((char*) &nmean, sizeof(double)), 
		std::string((char*) &nstdev, sizeof(double)), clay::DTYPE::DOUBLE, valid);
	kiln::ConstInit bsci(
		std::string((char*) &cdata, sizeof(double)), clay::DTYPE::DOUBLE, badvalid);
	kiln::UnifInit bsui(
		std::string((char*) &umin, sizeof(double)), 
		std::string((char*) &umax, sizeof(double)), clay::DTYPE::DOUBLE, badvalid);
	kiln::NormInit bsni(
		std::string((char*) &nmean, sizeof(double)), 
		std::string((char*) &nstdev, sizeof(double)), clay::DTYPE::DOUBLE, badvalid);

	ASSERT_EQ(nullptr, bvci.get());
	ASSERT_EQ(nullptr, bvui.get());
	ASSERT_EQ(nullptr, bvni.get());
	ASSERT_EQ(nullptr, bsci.get());
	ASSERT_EQ(nullptr, bsui.get());
	ASSERT_EQ(nullptr, bsni.get());

	clay::Tensor* vcten = vci.get();
	ASSERT_NE(nullptr, vcten);
	clay::Tensor* vuten = vui.get();
	ASSERT_NE(nullptr, vuten);
	clay::Tensor* vnten = vni.get();
	ASSERT_NE(nullptr, vnten);
	clay::Tensor* scten = sci.get();
	ASSERT_NE(nullptr, scten);
	clay::Tensor* suten = sui.get();
	ASSERT_NE(nullptr, suten);
	clay::Tensor* snten = sni.get();
	ASSERT_NE(nullptr, snten);

	EXPECT_SHAPEQ(shape, vcten->get_shape());
	EXPECT_SHAPEQ(shape, vuten->get_shape());
	EXPECT_SHAPEQ(shape, vnten->get_shape());
	EXPECT_SHAPEQ(shape, scten->get_shape());
	EXPECT_SHAPEQ(shape, suten->get_shape());
	EXPECT_SHAPEQ(shape, snten->get_shape());

	delete vcten;
	delete vuten;
	delete vnten;
	delete scten;
	delete suten;
	delete snten;
}


#endif /* DISABLE_ALL_INIT_TEST */


#endif /* DISABLE_KILN_MODULE_TESTS */
