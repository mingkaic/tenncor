//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATE_MODULE_TESTS

#include <algorithm>
#include <fstream>

#include "gtest/gtest.h"

#include "fuzz.hpp"
#include "sgen.hpp"
#include "check.hpp"
#include "print.hpp"

#include "graph/graph.hpp"
#include "graph/constant.hpp"
#include "graph/variable.hpp"
#include "graph/placeholder.hpp"
#include "graph/functor.hpp"


#ifndef DISABLE_GRAPH_TEST // compound node functions


struct GRAPH : public testutils::fuzz_test {};


using namespace testutils;


const std::string SAMPLE_DIR = "tests/unit/samples";


TEST_F(GRAPH, GraphSerialize_A000)
{
	std::fstream rgraph(SAMPLE_DIR + "/random.graph",
		std::ios::in | std::ios::binary);
	ASSERT_TRUE((bool) rgraph);

	tenncor::graph_proto src;
	ASSERT_TRUE(src.ParseFromIstream(&rgraph));
	std::unique_ptr<nnet::graph> temp = nnet::graph::get_temp();
	nnet::LEAF_SET leaves;
	nnet::ROOT_STR roots;
	temp->register_proto(leaves, roots, src);
	EXPECT_EQ(src.gid(), temp->get_gid());

	EXPECT_EQ(1, roots.size());
	// check leave set and create order

	tenncor::graph_proto dest;
	temp->serialize(dest);
	// expect src and dest are the same
	EXPECT_EQ(src.gid(), dest.gid());
	// map old to id to new id
	std::unordered_map<std::string, std::string> idmap;
	size_t nnodes = src.create_order_size();
	std::string srcid, destid;
	tenncor::node_proto srcnode, destnode;
	nnet::inode* tempnode;
	auto srcmap = src.node_map();
	auto destmap = dest.node_map();
	ASSERT_EQ(nnodes, dest.create_order_size());
	for (size_t i = 0; i < nnodes; ++i)
	{
		srcid = src.create_order(i);
		destid = dest.create_order(i);
		idmap[srcid] = destid;
		srcnode = srcmap[srcid];
		destnode = destmap[destid];
		tempnode = temp->get_inst(destid);
		ASSERT_NE(nullptr, tempnode);

		// type equal
		nnet::NODE_TYPE ntype = srcnode.type();
		ASSERT_EQ(ntype, destnode.type());
		ASSERT_EQ(ntype, tempnode->node_type());

		// label equal
		EXPECT_EQ(srcnode.label(), destnode.label());
		EXPECT_EQ(srcnode.label(), tempnode->get_label());

		auto srcany = srcnode.detail();
		auto destany = destnode.detail();
		switch (ntype)
		{
			case nnet::PLACEHOLDER_T:
			{
				tenncor::place_proto srcplace;
				tenncor::place_proto destplace;
				srcany.UnpackTo(&srcplace);
				destany.UnpackTo(&destplace);
				nnet::placeholder* tempplace = 
					dynamic_cast<nnet::placeholder*>(tempnode);
				ASSERT_NE(nullptr, tempplace);
				size_t nshape = srcplace.allowed_shape_size();
				nnet::tensorshape tempshape = tempplace->get_tensor()->get_allowed();
				ASSERT_EQ(nshape, destplace.allowed_shape_size());
				ASSERT_EQ(nshape, tempshape.rank());
				for (size_t j = 0; j < nshape; j++)
				{
					EXPECT_EQ(srcplace.allowed_shape(j), destplace.allowed_shape(j));
					EXPECT_EQ(srcplace.allowed_shape(j), tempshape[j]);
				}
			}
			break;
			case nnet::CONSTANT_T:
			{
				tenncor::tensor_proto srcconst;
				tenncor::tensor_proto destconst;
				srcany.UnpackTo(&srcconst);
				destany.UnpackTo(&destconst);
				nnet::constant* tempconst = 
					dynamic_cast<nnet::constant*>(tempnode);
				ASSERT_NE(nullptr, tempconst);
				nnet::tensor* tempten = tempconst->get_tensor();
				nnet::tensorshape tempshape = tempten->get_allowed();
				nnet::tensorshape tempshape2 = tempten->get_shape();
		
				size_t nshape = srcconst.allowed_shape_size();
				ASSERT_EQ(nshape, destconst.allowed_shape_size());
				ASSERT_EQ(nshape, tempshape.rank());
				for (size_t j = 0; j < nshape; j++)
				{
					EXPECT_EQ(srcconst.allowed_shape(j), destconst.allowed_shape(j));
					EXPECT_EQ(srcconst.allowed_shape(j), tempshape[j]);
				}
				
				size_t nshape2 = srcconst.alloced_shape_size();
				ASSERT_EQ(nshape2, destconst.alloced_shape_size());
				ASSERT_EQ(nshape2, tempshape2.rank());
				for (size_t j = 0; j < nshape2; j++)
				{
					EXPECT_EQ(srcconst.alloced_shape(j), destconst.alloced_shape(j));
					EXPECT_EQ(srcconst.alloced_shape(j), tempshape2[j]);
				}

				TENS_TYPE srctype = srcconst.type();
				EXPECT_EQ(srctype, destconst.type());
				EXPECT_EQ(srctype, tempten->get_type());
				EXPECT_EQ(nnet::INT32, srctype);

				tenncor::int32_arr srcarr;
				tenncor::int32_arr destarr;
				srcconst.data().UnpackTo(&srcarr);
				destconst.data().UnpackTo(&destarr);
				auto srcfields = srcarr.data();
				auto destfields = destarr.data();
				std::vector<int32_t> tempdata = nnet::expose<int32_t>(tempten);
				size_t nsrcfields = srcfields.size();
				ASSERT_EQ(nsrcfields, destfields.size());
				ASSERT_EQ(nsrcfields, tempdata.size());
				for (size_t j = 0; j < nsrcfields; ++j)
				{
					EXPECT_EQ(srcfields[j], destfields[j]);
					EXPECT_EQ(srcfields[j], tempdata[j]);
				}
			}
			break;
			case nnet::VARIABLE_T:
			{
				tenncor::variable_proto srcvar;
				tenncor::variable_proto destvar;
				srcany.UnpackTo(&srcvar);
				destany.UnpackTo(&destvar);
				nnet::variable* tempvar = 
					dynamic_cast<nnet::variable*>(tempnode);
				ASSERT_NE(nullptr, tempvar);

				EXPECT_EQ(srcvar.varpos(), destvar.varpos());
				EXPECT_EQ(srcvar.varpos(), tempvar->get_varpos());

				nnet::tensorshape tempshape = tempvar->get_tensor()->get_allowed();
				size_t nshape = srcvar.allowed_shape_size();
				ASSERT_EQ(nshape, destvar.allowed_shape_size());
				ASSERT_EQ(nshape, tempshape.rank());
				for (size_t j = 0; j < nshape; j++)
				{
					EXPECT_EQ(srcvar.allowed_shape(j), destvar.allowed_shape(j));
					EXPECT_EQ(srcvar.allowed_shape(j), tempshape[j]);
				}

				EXPECT_EQ(srcvar.source().src(), destvar.source().src());
			}
			break;
			case nnet::FUNCTOR_T:
			{
				tenncor::functor_proto srcfunc;
				tenncor::functor_proto destfunc;
				srcany.UnpackTo(&srcfunc);
				destany.UnpackTo(&destfunc);
				nnet::functor* tempfunc = 
					dynamic_cast<nnet::functor*>(tempnode);
				ASSERT_NE(nullptr, tempfunc);

				EXPECT_EQ(srcfunc.opcode(), destfunc.opcode());
			
				auto srcargs = srcfunc.args();
				auto destargs = destfunc.args();
				auto tempargs = tempfunc->get_arguments();

				size_t nsrcs = srcargs.size();
				EXPECT_EQ(nsrcs, destargs.size());
				EXPECT_EQ(nsrcs, tempargs.size());
				for (size_t j = 0; j < nsrcs; ++j)
				{
					std::string mappedid = idmap[srcfunc.args(j)];
					EXPECT_EQ(mappedid, destfunc.args(j));
					EXPECT_EQ(mappedid, tempargs[j]->get_uid());
				}
			}
			break;
			default:
				ASSERT_FALSE(true) << "unrecognized type " << ntype;
		}
	}
}


TEST_F(GRAPH, SerialConst_A001)
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
	TENS_TYPE p2type = proto_src2.type();
	TENS_TYPE p3type = proto_src3.type();
	EXPECT_EQ(p2type, p3type);
	std::shared_ptr<void> c2data = nnet::deserialize_data(proto_src2.data(), p2type);
	std::shared_ptr<void> c3data = nnet::deserialize_data(proto_src3.data(), p3type);
	size_t nb = n * sizeof(double);
	std::string c2str((char*) c2data.get(), nb);
	std::string c3str((char*) c3data.get(), nb);
	EXPECT_STREQ(c2str.c_str(), c3str.c_str());

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


TEST_F(GRAPH, SerialPlace_A002)
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


TEST_F(GRAPH, SerialVar_A003)
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

	TENS_TYPE rctype = csrc.dtype();
	std::shared_ptr<void> csptr = nnet::deserialize_data(csrc.settings(), rctype);
	double* gotc = (double*) csptr.get();
	EXPECT_EQ(c, *gotc);
	EXPECT_EQ(nnet::CSRC_T, csrc.src());
	EXPECT_EQ(nnet::DOUBLE, rctype);

	size_t nsettings;
	TENS_TYPE rrtype = rsrc.dtype();
	std::shared_ptr<void> rsptr = nnet::deserialize_data(rsrc.settings(), rrtype, &nsettings);
	ASSERT_EQ(2, nsettings);
	double* gotmin = (double*) rsptr.get();
	double* gotmax = (double*) ((char*) rsptr.get() + nnet::type_size(rctype));
	EXPECT_EQ(min, *gotmin);
	EXPECT_EQ(max, *gotmax);
	EXPECT_EQ(nnet::USRC_T, rsrc.src());
	EXPECT_EQ(nnet::DOUBLE, rrtype);

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

	TENS_TYPE rvtype = vsrc.dtype();
	std::shared_ptr<void> vsptr = nnet::deserialize_data(vsrc.settings(), rvtype);
	double* gotv = (double*) vsptr.get();
	EXPECT_EQ(c, *gotv);
	EXPECT_EQ(nnet::CSRC_T, vsrc.src());
	EXPECT_EQ(nnet::DOUBLE, vsrc.dtype());

	auto vec3 = var_src3.allowed_shape();
	nnet::tensorshape outshape3(std::vector<size_t>(vec3.begin(), vec3.end()));
	EXPECT_SHAPEQ(shape2, outshape3);
}


TEST_F(GRAPH, SerialFunc_A004)
{
}


TEST_F(GRAPH, SerialData_A005)
{
}


#endif /* DISABLE_GRAPH_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
