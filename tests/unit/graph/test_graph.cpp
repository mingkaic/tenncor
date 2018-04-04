//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATE_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "fuzz.hpp"
#include "sgen.hpp"
#include "check.hpp"
#include "print.hpp"

#include "graph/graph.hpp"
#include "graph/constant.hpp"
#include "graph/variable.hpp"
#include "graph/placeholder.hpp"


#ifndef DISABLE_GRAPH_TEST // compound node functions


struct GRAPH : public testutils::fuzz_test {};


using namespace testutils;


TEST_F(GRAPH, GraphSerialize_G000)
{
}


TEST_F(GRAPH, SerialConst_G001)
{
	double c = get_double(1, "c")[0];

	nnet::tensorshape shape = random_def_shape(this, {2, 5});
	size_t n = shape.n_elems();
	std::vector<double> v = get_double(n, "v");

	nnet::varptr res = nnet::constant::get<double>(c);
	nnet::varptr res2 = nnet::constant::get<double>(v, shape);

	google::protobuf::Any proto_dest;
	google::protobuf::Any proto_dest2;
	google::protobuf::Any proto_dest3; // double serialized (test overwriting)

	res->serialize_detail(&proto_dest);
	res->serialize_detail(&proto_dest3);
	res2->serialize_detail(&proto_dest2);
	res2->serialize_detail(&proto_dest3);

	tenncor::tensor_proto proto_src;
	tenncor::tensor_proto proto_src2;
	proto_dest.UnpackTo(&proto_src);
	proto_dest2.UnpackTo(&proto_src2);

	tenncor::tensor_proto proto_src3;
	proto_dest3.UnpackTo(&proto_src3); // expect eq to proto_dest2

	// with optimization
	nnet::varptr out = nnet::constant::get(proto_src, "outcome");
	EXPECT_EQ(res.get(), out.get());

	// without optimization
	// todo: make scalar without optimization
	nnet::varptr out2 = nnet::constant::get(proto_src2, "outcome2");
	nnet::tensor* ten = out2->get_tensor();
	ASSERT_NE(nullptr, ten);
	std::vector<double> vec = nnet::expose<double>(ten);
	nnet::tensorshape oshape = ten->get_shape();
	ASSERT_SHAPEQ(shape, oshape);
	for (size_t i = 0, n = vec.size(); i < n; ++i)
	{
		EXPECT_EQ(v[i], vec[i]);
	}

	// overwrite check
	EXPECT_EQ(proto_src2.type(), proto_src3.type());
	EXPECT_STREQ(proto_src2.data().c_str(), proto_src3.data().c_str());

	auto low2 = proto_src2.allowed_shape();
	auto loc2 = proto_src2.alloced_shape();
	auto low3 = proto_src3.allowed_shape();
	auto loc3 = proto_src3.alloced_shape();
	nnet::tensorshape lows(std::vector<size_t>(low2.begin(), low2.end()));
	nnet::tensorshape locs(std::vector<size_t>(loc2.begin(), loc2.end()));
	nnet::tensorshape lows2(std::vector<size_t>(low3.begin(), low3.end()));
	nnet::tensorshape locs2(std::vector<size_t>(loc3.begin(), loc3.end()));

	EXPECT_SHAPEQ(lows,  lows2);
	EXPECT_SHAPEQ(locs,  locs2);
}


TEST_F(GRAPH, SerialPlace_G002)
{
	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape = clist;
	nnet::tensorshape pshape = make_partial(this, clist);

	google::protobuf::Any proto_dest;
	google::protobuf::Any proto_dest2;
	google::protobuf::Any proto_dest3; // double serialized (test overwriting)

	nnet::placeholder place(pshape, label1);
	place.serialize_detail(&proto_dest);

	std::vector<double> raw = get_double(shape.n_elems(), "raw");
	place = raw;
	place.serialize_detail(&proto_dest2);
	place.serialize_detail(&proto_dest3);

	tenncor::place_proto place_src;
	tenncor::place_proto place_src2;
	proto_dest.UnpackTo(&place_src);
	proto_dest2.UnpackTo(&place_src2);

	auto vec = place_src.allowed_shape();
	auto vec2 = place_src2.allowed_shape();
	nnet::tensorshape outshape(std::vector<size_t>(vec.begin(), vec.end()));
	nnet::tensorshape outshape2(std::vector<size_t>(vec2.begin(), vec2.end()));

	EXPECT_SHAPEQ(pshape, outshape);
	EXPECT_SHAPEQ(pshape, outshape2);

	// overwrite check
	nnet::tensorshape shape2 = random_def_shape(this);
	nnet::placeholder place2(shape2, label2);
	place2.serialize_detail(&proto_dest3);
	tenncor::place_proto place_src3;
	proto_dest3.UnpackTo(&place_src3);
	auto vec3 = place_src3.allowed_shape();
	nnet::tensorshape outshape3(std::vector<size_t>(vec3.begin(), vec3.end()));
	EXPECT_SHAPEQ(shape2, outshape3);
}


TEST_F(GRAPH, SerialVar_G003)
{
	std::vector<size_t> strns = get_int(3, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	std::string label3 = get_string(strns[2], "label3");
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape = clist;
	nnet::tensorshape pshape = make_partial(this, clist);
	double c = get_double(1, "c")[0];
	std::vector<double> minmax = get_double(2, "min-max", {-24, 26});
	double min = *std::min_element(minmax.begin(), minmax.end());
	double max = *std::max_element(minmax.begin(), minmax.end());

	std::shared_ptr<nnet::const_init> cinit = std::make_shared<nnet::const_init>();
	cinit->set<double>(c);
	std::shared_ptr<nnet::r_uniform_init> rinit = std::make_shared<nnet::r_uniform_init>();
	rinit->set<double>(min, max);

	nnet::variable cinitv(pshape, cinit, label1);
	nnet::variable rinitv(pshape, rinit, label2);
	
	google::protobuf::Any proto_dest;
	google::protobuf::Any proto_dest2;
	google::protobuf::Any proto_dest3; // double serialized (test overwriting)

	cinitv.serialize_detail(&proto_dest);
	rinitv.serialize_detail(&proto_dest2);
	rinitv.serialize_detail(&proto_dest3);

	tenncor::variable_proto var_src;
	tenncor::variable_proto var_src2;
	proto_dest.UnpackTo(&var_src);
	proto_dest2.UnpackTo(&var_src2);
	
	std::string data_ep = var_src.varpos();
	std::string data_ep2 = var_src2.varpos();
	std::string cep = cinitv.get_varpos();
	std::string rep = rinitv.get_varpos();

	EXPECT_STREQ(data_ep.c_str(), cep.c_str());
	EXPECT_STREQ(data_ep2.c_str(), rep.c_str());

	tenncor::source_proto csrc = var_src.source();
	tenncor::source_proto rsrc = var_src2.source();

	std::string csetting = csrc.settings(0);
	double* gotc = (double*) &csetting[0];
	EXPECT_EQ(c, *gotc);
	EXPECT_EQ(nnet::CSRC_T, csrc.src());
	EXPECT_EQ(nnet::DOUBLE, csrc.dtype());

	ASSERT_EQ(2, rsrc.settings_size());
	std::string minsetting = rsrc.settings(0);
	std::string maxsetting = rsrc.settings(1);
	double* gotmin = (double*) &minsetting[0];
	double* gotmax = (double*) &maxsetting[0];
	EXPECT_EQ(min, *gotmin);
	EXPECT_EQ(max, *gotmax);
	EXPECT_EQ(nnet::USRC_T, rsrc.src());
	EXPECT_EQ(nnet::DOUBLE, rsrc.dtype());

	auto vec = var_src.allowed_shape();
	auto vec2 = var_src.allowed_shape();
	nnet::tensorshape outshape(std::vector<size_t>(vec.begin(), vec.end()));
	nnet::tensorshape outshape2(std::vector<size_t>(vec2.begin(), vec2.end()));

	EXPECT_SHAPEQ(pshape, outshape);
	EXPECT_SHAPEQ(pshape, outshape2);

	// overwrite check
	nnet::tensorshape shape2 = random_def_shape(this);
	nnet::variable var(shape2, cinit, label2);
	var.serialize_detail(&proto_dest3);
	tenncor::variable_proto var_src3;
	proto_dest3.UnpackTo(&var_src3);

	std::string data_ep3 = var_src3.varpos();
	std::string vep = var.get_varpos();
	EXPECT_STREQ(data_ep3.c_str(), vep.c_str());

	tenncor::source_proto vsrc = var_src3.source();

	std::string csetting2 = csrc.settings(0);
	double* gotc2 = (double*) &csetting2[0];
	EXPECT_EQ(c, *gotc2);
	EXPECT_EQ(nnet::CSRC_T, vsrc.src());
	EXPECT_EQ(nnet::DOUBLE, vsrc.dtype());

	auto vec3 = var_src3.allowed_shape();
	nnet::tensorshape outshape3(std::vector<size_t>(vec3.begin(), vec3.end()));
	EXPECT_SHAPEQ(shape2, outshape3);
}


TEST_F(GRAPH, SerialFunc_G004)
{
}


TEST_F(GRAPH, SerialData_G005)
{
}


#endif /* DISABLE_GRAPH_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
