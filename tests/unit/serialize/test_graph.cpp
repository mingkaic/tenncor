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


#define OP2STR(OP) vec[OP] = #OP;


std::vector<std::string> opcodename = []()
{
	std::vector<std::string> vec(_OP_SENTINEL);
	OP2STR(ABS)
	OP2STR(NEG)
	OP2STR(NOT)
	OP2STR(SIN)
	OP2STR(COS)
	OP2STR(TAN)
	OP2STR(EXP)
	OP2STR(LOG)
	OP2STR(SQRT)
	OP2STR(ROUND)
	OP2STR(POW)
	OP2STR(ADD)
	OP2STR(SUB)
	OP2STR(MUL)
	OP2STR(DIV)
	OP2STR(EQ)
	OP2STR(NE)
	OP2STR(GT)
	OP2STR(LT)
	OP2STR(BINO)
	OP2STR(UNIF)
	OP2STR(NORM)
	OP2STR(TRANSPOSE)
	OP2STR(FLIP)
	OP2STR(ARGMAX)
	OP2STR(RMAX)
	OP2STR(RSUM)
	OP2STR(EXPAND)
	OP2STR(N_ELEMS)
	OP2STR(N_DIMS)
	OP2STR(MATMUL)
	return vec;
}();


using namespace testutils;


const std::string SAMPLE_DIR = "tests/unit/samples/";

const std::string RANDOM_PROTO = "RANDOM.graph";


// covers graph get_temp, serialize, register_proto
// assumes test/unit/sample/*.graph are correct
TEST_F(GRAPH, GraphSerialize_A000)
{
	std::fstream rgraph(SAMPLE_DIR + RANDOM_PROTO,
		std::ios::in | std::ios::binary);
	ASSERT_TRUE((bool) rgraph);

	tenncor::GraphPb src;
	ASSERT_TRUE(src.ParseFromIstream(&rgraph));
	std::unique_ptr<nnet::graph> temp = nnet::graph::get_temp();
	nnet::LEAF_SET leaves;
	nnet::ROOT_STR roots;
	temp->register_proto(leaves, roots, src);
	EXPECT_EQ(src.gid(), temp->get_gid());

	EXPECT_EQ(1, roots.size());
	// check leave set and create order

	tenncor::GraphPb dest;
	temp->serialize(dest);
	// expect src and dest are the same
	EXPECT_EQ(src.gid(), dest.gid());
	// map old to id to new id
	std::unordered_map<std::string, std::string> idmap;
	size_t nnodes = src.create_order_size();
	std::string srcid, destid;
	tenncor::NodePb srcnode, destnode;
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
				tenncor::PlacePb srcplace;
				tenncor::PlacePb destplace;
				srcany.UnpackTo(&srcplace);
				destany.UnpackTo(&destplace);
				nnet::placeholder* tempplace = 
					dynamic_cast<nnet::placeholder*>(tempnode);
				ASSERT_NE(nullptr, tempplace);
				size_t nshape = srcplace.allowed_shape_size();
				nnet::tshape tempshape = tempplace->get_tensor()->get_allowed();
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
				tenncor::TensorPb srcconst;
				tenncor::TensorPb destconst;
				srcany.UnpackTo(&srcconst);
				destany.UnpackTo(&destconst);
				nnet::constant* tempconst = 
					dynamic_cast<nnet::constant*>(tempnode);
				ASSERT_NE(nullptr, tempconst);
				nnet::tensor* tempten = tempconst->get_tensor();
				nnet::tshape tempshape = tempten->get_allowed();
				nnet::tshape tempshape2 = tempten->get_shape();
		
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

				tenncor::Int32Arr srcarr;
				tenncor::Int32Arr destarr;
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
				tenncor::VariablePb srcvar;
				tenncor::VariablePb destvar;
				srcany.UnpackTo(&srcvar);
				destany.UnpackTo(&destvar);
				nnet::variable* tempvar = 
					dynamic_cast<nnet::variable*>(tempnode);
				ASSERT_NE(nullptr, tempvar);

				EXPECT_EQ(srcvar.varpos(), destvar.varpos());
				EXPECT_EQ(srcvar.varpos(), tempvar->get_varpos());

				nnet::tshape tempshape = tempvar->get_tensor()->get_allowed();
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
				tenncor::FunctorPb srcfunc;
				tenncor::FunctorPb destfunc;
				srcany.UnpackTo(&srcfunc);
				destany.UnpackTo(&destfunc);
				nnet::functor* tempfunc = 
					dynamic_cast<nnet::functor*>(tempnode);
				ASSERT_NE(nullptr, tempfunc);

				EXPECT_EQ(srcfunc.opcode(), destfunc.opcode());
				EXPECT_EQ(srcfunc.opcode(), tempfunc->get_opcode());
			
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


// covers constant::get(tenncor::TensorPb&,std::string), serialize_detail, node_type
TEST_F(GRAPH, SerialConst_A001)
{
	double c = get_double(1, "c")[0];

	nnet::tshape shape = random_def_shape(this, {2, 5});
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

	tenncor::TensorPb proto_src;
	tenncor::TensorPb proto_src2;
	proto_dest.UnpackTo(&proto_src);
	proto_dest2.UnpackTo(&proto_src2);

	tenncor::TensorPb proto_src3;
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
	nnet::tshape oshape = ten->get_shape();
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
	nnet::tshape lows(std::vector<size_t>(low2.begin(), low2.end()));
	nnet::tshape locs(std::vector<size_t>(loc2.begin(), loc2.end()));
	nnet::tshape lows2(std::vector<size_t>(low3.begin(), low3.end()));
	nnet::tshape locs2(std::vector<size_t>(loc3.begin(), loc3.end()));

	EXPECT_SHAPEQ(lows,  lows2);
	EXPECT_SHAPEQ(locs,  locs2);
}


// covers placeholder serialize_detail, node_type
TEST_F(GRAPH, SerialPlace_A002)
{
	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tshape shape = clist;
	nnet::tshape pshape = make_partial(this, clist);

	google::protobuf::Any proto_dest;
	google::protobuf::Any proto_dest2;
	google::protobuf::Any proto_dest3; // double serialized (test overwriting)

	nnet::placeholder place(pshape, label1);
	place.serialize_detail(&proto_dest);

	std::vector<double> raw = get_double(shape.n_elems(), "raw");
	place = raw;
	place.serialize_detail(&proto_dest2);
	place.serialize_detail(&proto_dest3);

	tenncor::PlacePb place_src;
	tenncor::PlacePb place_src2;
	proto_dest.UnpackTo(&place_src);
	proto_dest2.UnpackTo(&place_src2);

	auto vec = place_src.allowed_shape();
	auto vec2 = place_src2.allowed_shape();
	nnet::tshape outshape(std::vector<size_t>(vec.begin(), vec.end()));
	nnet::tshape outshape2(std::vector<size_t>(vec2.begin(), vec2.end()));

	EXPECT_SHAPEQ(pshape, outshape);
	EXPECT_SHAPEQ(pshape, outshape2);

	// overwrite check
	nnet::tshape shape2 = random_def_shape(this);
	nnet::placeholder place2(shape2, label2);
	place2.serialize_detail(&proto_dest3);
	tenncor::PlacePb place_src3;
	proto_dest3.UnpackTo(&place_src3);
	auto vec3 = place_src3.allowed_shape();
	nnet::tshape outshape3(std::vector<size_t>(vec3.begin(), vec3.end()));
	EXPECT_SHAPEQ(shape2, outshape3);
}


// covers variable serialize_detail, node_type
TEST_F(GRAPH, SerialVar_A003)
{
	std::vector<size_t> strns = get_int(3, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	std::string label3 = get_string(strns[2], "label3");
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tshape shape = clist;
	nnet::tshape pshape = make_partial(this, clist);
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

	tenncor::VariablePb var_src;
	tenncor::VariablePb var_src2;
	proto_dest.UnpackTo(&var_src);
	proto_dest2.UnpackTo(&var_src2);
	
	std::string data_ep = var_src.varpos();
	std::string data_ep2 = var_src2.varpos();
	std::string cep = cinitv.get_varpos();
	std::string rep = rinitv.get_varpos();

	EXPECT_STREQ(data_ep.c_str(), cep.c_str());
	EXPECT_STREQ(data_ep2.c_str(), rep.c_str());

	tenncor::SourcePb csrc = var_src.source();
	tenncor::SourcePb rsrc = var_src2.source();

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
	nnet::tshape outshape(std::vector<size_t>(vec.begin(), vec.end()));
	nnet::tshape outshape2(std::vector<size_t>(vec2.begin(), vec2.end()));

	EXPECT_SHAPEQ(pshape, outshape);
	EXPECT_SHAPEQ(pshape, outshape2);

	// overwrite check
	nnet::tshape shape2 = random_def_shape(this);
	nnet::variable var(shape2, cinit, label2);
	var.serialize_detail(&proto_dest3);
	tenncor::VariablePb var_src3;
	proto_dest3.UnpackTo(&var_src3);

	std::string data_ep3 = var_src3.varpos();
	std::string vep = var.get_varpos();
	EXPECT_STREQ(data_ep3.c_str(), vep.c_str());

	tenncor::SourcePb vsrc = var_src3.source();

	TENS_TYPE rvtype = vsrc.dtype();
	std::shared_ptr<void> vsptr = nnet::deserialize_data(vsrc.settings(), rvtype);
	double* gotv = (double*) vsptr.get();
	EXPECT_EQ(c, *gotv);
	EXPECT_EQ(nnet::CSRC_T, vsrc.src());
	EXPECT_EQ(nnet::DOUBLE, vsrc.dtype());

	auto vec3 = var_src3.allowed_shape();
	nnet::tshape outshape3(std::vector<size_t>(vec3.begin(), vec3.end()));
	EXPECT_SHAPEQ(shape2, outshape3);
}


// covers functor serialize_detail, node_type
// assumes test/unit/sample/*.graph are correct
TEST_F(GRAPH, SerialFunc_A004)
{
	std::string line;
	std::vector<std::string> ops;
	std::ifstream registry(SAMPLE_DIR + "registry.txt");
	if (registry.is_open())
	{
		while (!registry.eof())
		{
			std::getline(registry, line);
			if (line.size() > 0 &&
				0 != line.compare(RANDOM_PROTO))
			{
				ops.push_back(line);
			}
		}
		registry.close();
	}
	std::uniform_int_distribution<int> selector(0, ops.size() - 1);
	std::string opfile = ops[selector(nnutils::get_generator())];

	std::fstream rgraph(SAMPLE_DIR + opfile,
		std::ios::in | std::ios::binary);
	ASSERT_TRUE((bool) rgraph);

	tenncor::GraphPb src;
	ASSERT_TRUE(src.ParseFromIstream(&rgraph));
	std::unique_ptr<nnet::graph> temp = nnet::graph::get_temp();
	nnet::LEAF_SET leaves;
	nnet::ROOT_STR roots;
	temp->register_proto(leaves, roots, src);

	ASSERT_EQ(1, roots.size());
	nnet::inode* root = temp->get_inst(*(roots.begin()));

	nnet::functor* rfunc = dynamic_cast<nnet::functor*>(root);
	ASSERT_NE(nullptr, rfunc);

	std::string opstr = opcodename[rfunc->get_opcode()];
	EXPECT_EQ(0, opfile.find(opstr)) << "expecting " << opstr << " to match " << opfile;
}


// covers graph get_temp, replace_global, get_global, save_data, load_data
// assumes test/unit/sample/*.graph, test/unit/sample/*.data are correct
TEST_F(GRAPH, SerialData_A005)
{
	{
		tenncor::DataRepoPb emptypb;
		std::unique_ptr<nnet::graph> temp = nnet::graph::get_temp();
		EXPECT_TRUE(temp->save_data(emptypb));
		EXPECT_EQ(0, emptypb.data_map_size());
		nnet::graph::replace_global(std::move(temp));
	}
	nnet::graph& grf = nnet::graph::get_global();
	// test save_data
	{
		std::vector<size_t> strns = get_int(2, "strns", {14, 29});
		std::string varlabel = get_string(strns[0], "varlabel");
		std::vector<size_t> clist = random_def_shape(this);
		nnet::tshape cshape(clist);
		size_t n = cshape.n_elems();
		nnet::tshape varshape = make_partial(this, clist);
		double c = get_double(1, "c")[0];
		tenncor::DataRepoPb uninitvar;
		tenncor::DataRepoPb initvar;
		nnet::const_init* csrc = new nnet::const_init();
		std::shared_ptr<nnet::data_src> src = 
			std::shared_ptr<nnet::data_src>(csrc);
		csrc->set<double>(c);
		nnet::variable* var = new nnet::variable(varshape, src, varlabel);
		ASSERT_TRUE(grf.has_node(var));
		EXPECT_FALSE(grf.save_data(uninitvar));
		EXPECT_EQ(0, uninitvar.data_map_size());
		ASSERT_TRUE(var->initialize(cshape));
		EXPECT_TRUE(grf.save_data(initvar));
		EXPECT_EQ(1, initvar.data_map_size());
		// verify initvar
		auto varmap = initvar.data_map();
		auto it = varmap.find(var->get_varpos());
		ASSERT_TRUE(varmap.end() != it);
		tenncor::TensorPb tp = it->second;
		nnet::tshape allow(std::vector<size_t>(
			tp.allowed_shape().begin(),
			tp.allowed_shape().end()));
		nnet::tshape alloc(std::vector<size_t>(
			tp.alloced_shape().begin(),
			tp.alloced_shape().end()));
		EXPECT_SHAPEQ(varshape, allow);
		EXPECT_SHAPEQ(cshape, alloc);
		ASSERT_EQ(nnet::DOUBLE, tp.type());
		tenncor::DoubleArr dbarr;
		tp.data().UnpackTo(&dbarr);
		ASSERT_EQ(n, dbarr.data_size());
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(c, dbarr.data(i));
		}
		delete var;
		ASSERT_FALSE(grf.has_node(var));
	
		std::unique_ptr<nnet::graph> temp = nnet::graph::get_temp();
		nnet::graph::replace_global(std::move(temp));
	}
	// test load_data
	{
		std::fstream rgraph(SAMPLE_DIR + RANDOM_PROTO,
			std::ios::in | std::ios::binary);
		ASSERT_TRUE((bool) rgraph);

		tenncor::GraphPb src;
		ASSERT_TRUE(src.ParseFromIstream(&rgraph));
		nnet::LEAF_SET leaves;
		nnet::ROOT_STR roots;
		grf.register_proto(leaves, roots, src);
		std::vector<nnet::tensor*> vartens;
		for (nnet::varptr v : leaves)
		{
			if (nnet::variable* var = dynamic_cast<nnet::variable*>(v.get()))
			{
				nnet::tensor* ten = var->get_tensor();
				ASSERT_NE(nullptr, ten);
				EXPECT_FALSE(ten->has_data());
				vartens.push_back(ten);
			}
		}
		
		std::fstream rdata(SAMPLE_DIR + "RANDOM.data",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE((bool) rdata);
		tenncor::DataRepoPb srcdata;
		ASSERT_TRUE(srcdata.ParseFromIstream(&rdata));
		grf.load_data(srcdata);
		for (nnet::tensor* ten : vartens)
		{
			EXPECT_TRUE(ten->has_data());
		}
	}
}


#endif /* DISABLE_GRAPH_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
