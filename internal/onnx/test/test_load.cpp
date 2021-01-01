
#ifndef DISABLE_ONNX_LOAD_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "dbg/print/teq.hpp"

#include "internal/onnx/load.hpp"


using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::Throw;


#ifdef CMAKE_SOURCE_DIR
const std::string testdir = std::string(CMAKE_SOURCE_DIR) + "models/test";
#else
const std::string testdir = "models/test";
#endif


struct MockUnmarshFuncs final : public onnx::iUnmarshFuncs
{
	MOCK_CONST_METHOD3(unmarsh_leaf, teq::TensptrT(const onnx::TensorProto&,teq::Usage,std::string));

	MOCK_CONST_METHOD3(unmarsh_func, teq::TensptrT(std::string,const teq::TensptrsT&,marsh::Maps&&));

	MOCK_CONST_METHOD4(unmarsh_layr, teq::TensptrT(std::string,const teq::TensptrT&,const teq::TensptrT&,marsh::Maps&&));
};


static auto handle_leaf (teq::Shape exshape, const std::string& id)
{
	return [exshape,id](const onnx::TensorProto& tens, teq::Usage, std::string name)
	{
		auto shape = onnx::unmarshal_shape(tens);
		EXPECT_ARREQ(exshape, shape);
		return make_var(onnx::unmarshal_shape(tens), id);
	};
}


static auto handle_func (const types::StringsT& args, const std::string& id)
{
	return [id,args](std::string opname, const teq::TensptrsT& edges, marsh::Maps&&)
	{
		types::StringsT edgenames;
		edgenames.reserve(edges.size());
		std::transform(edges.begin(), edges.end(), std::back_inserter(edgenames),
			[](teq::TensptrT tens){ return tens->to_string(); });
		EXPECT_VECEQ(args, edgenames);
		return make_fnc(id, 0, edges);
	};
}


static auto handle_layer (const std::string& root_id, const std::string& arg, const std::string& id)
{
	return [root_id,id,arg](std::string opname, const teq::TensptrT& root, const teq::TensptrT& child, marsh::Maps&&)
	{
		EXPECT_STREQ(root_id.c_str(), root->to_string().c_str());
		EXPECT_STREQ(arg.c_str(), child->to_string().c_str());
		return make_fnc(id, 0, teq::TensptrsT{root});
	};
}


TEST(LOAD, BadGraph)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);

	teq::Shape exshape({3, 1, 7});
	teq::Shape exshape1({3, 7, 1});
	teq::Shape exshape2({3, 3, 1});
	teq::Shape exshape3({7, 3, 1});

	{
		onnx::ModelProto model;
		std::fstream inputstr(testdir + "/bad_onnx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));

		MockUnmarshFuncs unmarsh;
		EXPECT_CALL(unmarsh, unmarsh_leaf(_,_,_)).Times(7).
			WillOnce(Invoke(handle_leaf(exshape3,"a"))).
			WillOnce(Invoke(handle_leaf(exshape,"b"))).
			WillOnce(Invoke(handle_leaf(exshape1,"c"))).
			WillOnce(Invoke(handle_leaf(exshape1,"d"))).
			WillOnce(Invoke(handle_leaf(exshape2,"e"))).
			WillOnce(Invoke(handle_leaf(exshape2,"f"))).
			WillOnce(Invoke(handle_leaf(exshape2,"g")));
		EXPECT_CALL(unmarsh, unmarsh_func(_,_,_)).Times(2).
			WillOnce(Invoke(handle_func({"c"}, "h"))).
			WillOnce(Invoke(handle_func({"d"}, "i")));
		EXPECT_CALL(unmarsh, unmarsh_layr(_,_,_,_)).Times(0);

		onnx::TensptrIdT ids;
		std::string fatalmsg = "unknown onnx attribute type of `peanut`";
		EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
		EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
		EXPECT_FATAL(onnx::load_graph(ids, model.graph(), unmarsh), fatalmsg.c_str());
	}
	{
		onnx::ModelProto model;
		std::fstream inputstr(testdir + "/bad_onnx2.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));

		MockUnmarshFuncs unmarsh;
		EXPECT_CALL(unmarsh, unmarsh_leaf(_,_,_)).Times(7).
			WillOnce(Invoke(handle_leaf(exshape3, "a"))).
			WillOnce(Invoke(handle_leaf(exshape,"b"))).
			WillOnce(Invoke(handle_leaf(exshape1,"c"))).
			WillOnce(Invoke(handle_leaf(exshape1,"d"))).
			WillOnce(Invoke(handle_leaf(exshape2,"e"))).
			WillOnce(Invoke(handle_leaf(exshape2,"f"))).
			WillOnce(Invoke(handle_leaf(exshape2,"g")));
		EXPECT_CALL(unmarsh, unmarsh_func(_,_,_)).Times(2).
			WillOnce(Invoke(handle_func({"c"}, "h"))).
			WillOnce(Invoke(handle_func({"d"}, "i")));
		EXPECT_CALL(unmarsh, unmarsh_layr(_,_,_,_)).Times(0);

		onnx::TensptrIdT ids;
		std::string fatalmsg = "unknown graph attribute `peanut`";
		EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
		EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
		EXPECT_FATAL(onnx::load_graph(ids, model.graph(), unmarsh), fatalmsg.c_str());
	}

	global::set_logger(new exam::NoSupportLogger());
}


TEST(LOAD, SimpleGraph)
{
	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/simple_onnx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	teq::Shape exshape({3, 1, 7});
	teq::Shape exshape1({3, 7, 1});
	teq::Shape exshape2({3, 3, 1});
	teq::Shape exshape3({7, 3, 1});

	MockUnmarshFuncs unmarsh;
	EXPECT_CALL(unmarsh, unmarsh_leaf(_,_,_)).Times(7).
		WillOnce(Invoke(handle_leaf(exshape,"a"))).
		WillOnce(Invoke(handle_leaf(exshape1,"b"))).
		WillOnce(Invoke(handle_leaf(exshape1,"c"))).
		WillOnce(Invoke(handle_leaf(exshape3,"d"))).
		WillOnce(Invoke(handle_leaf(exshape2,"e"))).
		WillOnce(Invoke(handle_leaf(exshape2,"f"))).
		WillOnce(Invoke(handle_leaf(exshape2,"g")));
	EXPECT_CALL(unmarsh, unmarsh_func(_,_,_)).Times(11).
		WillOnce(Invoke(handle_func({"b"}, "i"))).
		WillOnce(Invoke(handle_func({"c"}, "j"))).
		WillOnce(Invoke(handle_func({"j", "c"}, "k"))).
		WillOnce(Invoke(handle_func({"i", "k"}, "l"))).
		WillOnce(Invoke(handle_func({"l", "d"}, "m"))).
		WillOnce(Invoke(handle_func({"a", "m"}, "n"))).
		WillOnce(Invoke(handle_func({"e"}, "o"))).
		WillOnce(Invoke(handle_func({"f"}, "p"))).
		WillOnce(Invoke(handle_func({"g"}, "q"))).
		WillOnce(Invoke(handle_func({"o","p","q"}, "r"))).
		WillOnce(Invoke(handle_func({"e","r"}, "s")));
	EXPECT_CALL(unmarsh, unmarsh_layr(_,_,_,_)).Times(0);

	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = onnx::load_graph(ids, model.graph(), unmarsh);
	ASSERT_EQ(2, graph_roots.size());

	EXPECT_STREQ("n", graph_roots.front()->to_string().c_str());
	EXPECT_STREQ("s", graph_roots.back()->to_string().c_str());
}


TEST(LOAD, LayerGraph)
{
	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/layer_onnx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	teq::Shape exshape({3, 1, 7});
	teq::Shape exshape1({3, 7, 1});
	teq::Shape exshape2({3, 3, 1});
	teq::Shape exshape3({7, 3, 1});
	teq::Shape exshape0;

	MockUnmarshFuncs unmarsh;
	EXPECT_CALL(unmarsh, unmarsh_leaf(_,_,_)).Times(7).
		WillOnce(Invoke(handle_leaf(exshape3,"a"))).
		WillOnce(Invoke(handle_leaf(exshape,"b"))).
		WillOnce(Invoke(handle_leaf(exshape1,"c"))).
		WillOnce(Invoke(handle_leaf(exshape1,"d"))).
		WillOnce(Invoke(handle_leaf(exshape2,"e"))).
		WillOnce(Invoke(handle_leaf(exshape2,"f"))).
		WillOnce(Invoke(handle_leaf(exshape2,"g")));
	EXPECT_CALL(unmarsh, unmarsh_func(_,_,_)).Times(11).
		WillOnce(Invoke(handle_func({"c"}, "h"))).
		WillOnce(Invoke(handle_func({"d"}, "i"))).
		WillOnce(Invoke(handle_func({"i","d"}, "j"))).
		WillOnce(Invoke(handle_func({"h","j"}, "k"))).
		WillOnce(Invoke(handle_func({"k","a"}, "l"))).
		WillOnce(Invoke(handle_func({"b","l"}, "m"))).
		WillOnce(Invoke(handle_func({"e"}, "n"))).
		WillOnce(Invoke(handle_func({"g"}, "o"))).
		WillOnce(Invoke(handle_func({"f"}, "p"))).
		WillOnce(Invoke(handle_func({"n","o","p"}, "q"))).
		WillOnce(Invoke(handle_func({"e","q"}, "r")));
	EXPECT_CALL(unmarsh, unmarsh_layr(_,_,_,_)).Times(1).
		WillOnce(Invoke(handle_layer("j", "i", "s")));

	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = onnx::load_graph(ids, model.graph(), unmarsh);
	ASSERT_EQ(2, graph_roots.size());

	EXPECT_STREQ("m", graph_roots.front()->to_string().c_str());
	EXPECT_STREQ("r", graph_roots.back()->to_string().c_str());
}


TEST(LOAD, ReplaceLayerGraph)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);

	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/layer_onnx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	teq::Shape exshape0;
	teq::Shape exshape({3, 1, 7});
	teq::Shape exshape1({3, 7, 1});
	teq::Shape exshape2({3, 3, 1});
	teq::Shape exshape3({7, 3, 1});

	MockUnmarshFuncs unmarsh;
	EXPECT_CALL(unmarsh, unmarsh_leaf(_,_,_)).Times(13).
		WillOnce(Invoke(handle_leaf(exshape3,"a"))).
		WillOnce(Invoke(handle_leaf(exshape,"b"))).
		WillOnce(Invoke(handle_leaf(exshape1,"c"))).
		WillOnce(Invoke(handle_leaf(exshape1,"d"))).
		WillOnce(Invoke(handle_leaf(exshape2,"e"))).
		WillOnce(Invoke(handle_leaf(exshape2,"f"))).
		WillOnce(Invoke(handle_leaf(exshape2,"g"))).
		WillOnce(Invoke(handle_leaf(exshape3,"h"))).
		WillOnce(Invoke(handle_leaf(exshape1,"i"))).
		WillOnce(Invoke(handle_leaf(exshape1,"j"))).
		WillOnce(Invoke(handle_leaf(exshape2,"k"))).
		WillOnce(Invoke(handle_leaf(exshape2,"l"))).
		WillOnce(Invoke(handle_leaf(exshape2,"m")));
	EXPECT_CALL(unmarsh, unmarsh_func(_,_,_)).Times(12).
		WillOnce(Invoke(handle_func({"c"}, "n"))).
		WillOnce(Invoke(handle_func({"i"}, "o"))).
		WillOnce(Invoke(handle_func({"j"}, "p"))).
		WillOnce(Invoke(handle_func({"p", "j"}, "q"))).
		WillOnce(Invoke(handle_func({"o","q"}, "r"))).
		WillOnce(Invoke(handle_func({"r","h"}, "s"))).
		WillOnce(Invoke(handle_func({"replaced","s"}, "t"))).
		WillOnce(Invoke(handle_func({"k"}, "u"))).
		WillOnce(Invoke(handle_func({"m"}, "v"))).
		WillOnce(Invoke(handle_func({"l"}, "w"))).
		WillOnce(Invoke(handle_func({"u","v","w"}, "x"))).
		WillOnce(Invoke(handle_func({"k","x"}, "y")));
	EXPECT_CALL(unmarsh, unmarsh_layr(_,_,_,_)).Times(1).
		WillOnce(Invoke(handle_layer("q", "p", "z")));

	auto badm = std::make_shared<MockLeaf>();
	onnx::TensptrIdT badids;
	badids.insert({badm, "5"});
	std::string fatalmsg = "duplicate id 5";
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(onnx::load_graph(badids, model.graph(), unmarsh), fatalmsg.c_str());

	auto m = make_var(teq::Shape(), "replaced");
	onnx::TensptrIdT ids;
	ids.insert({m, "1"});
	teq::TensptrsT graph_roots = onnx::load_graph(ids, model.graph(), unmarsh);
	EXPECT_EQ(2, graph_roots.size());

	ASSERT_HAS(ids.right, "root1");
	ASSERT_HAS(ids.right, "root2");

	global::set_logger(new exam::NoSupportLogger());
}


TEST(LOAD, SimpleGraphEarlyStop)
{
	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/simple_stop.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	teq::Shape exshape({3, 1, 7});
	teq::Shape exshape1({3, 7, 1});
	teq::Shape exshape2({3, 3, 1});
	teq::Shape exshape3({7, 3, 1});

	MockUnmarshFuncs unmarsh;
	EXPECT_CALL(unmarsh, unmarsh_leaf(_,_,_)).Times(8).
		WillOnce(Invoke(handle_leaf(exshape,"a"))).
		WillOnce(Invoke(handle_leaf(exshape1,"b"))).
		WillOnce(Invoke(handle_leaf(exshape2,"c"))).
		WillOnce(Invoke(handle_leaf(exshape2,"d"))).
		WillOnce(Invoke(handle_leaf(exshape1,"e"))).
		WillOnce(Invoke(handle_leaf(exshape3,"f"))).
		WillOnce(Invoke(handle_leaf(exshape2,"g"))).
		WillOnce(Invoke(handle_leaf(exshape2,"h")));
	EXPECT_CALL(unmarsh, unmarsh_func(_,_,_)).Times(7).
		WillOnce(Invoke(handle_func({"e"}, "i"))).
		WillOnce(Invoke(handle_func({"i","b"}, "j"))).
		WillOnce(Invoke(handle_func({"j","f"}, "k"))).
		WillOnce(Invoke(handle_func({"a","k"}, "l"))).
		WillOnce(Invoke(handle_func({"h"}, "m"))).
		WillOnce(Invoke(handle_func({"c","m","d"}, "n"))).
		WillOnce(Invoke(handle_func({"g","n"}, "o")));
	EXPECT_CALL(unmarsh, unmarsh_layr(_,_,_,_)).Times(0);

	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = onnx::load_graph(ids, model.graph(), unmarsh);
	ASSERT_EQ(2, graph_roots.size());

	EXPECT_STREQ("l", graph_roots[0]->to_string().c_str());
	EXPECT_STREQ("o", graph_roots[1]->to_string().c_str());
}


#endif // DISABLE_ONNX_LOAD_TEST
