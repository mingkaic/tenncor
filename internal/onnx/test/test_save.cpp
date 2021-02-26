
#ifndef DISABLE_ONNX_SAVE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/global/mock/mock.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/onnx/save.hpp"


using ::testing::_;
using ::testing::Const;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::Throw;


#ifdef CMAKE_SOURCE_DIR
const std::string testdir = std::string(CMAKE_SOURCE_DIR) + "models/test";
#else
const std::string testdir = "models/test";
#endif


struct MockMarshFuncs final : public onnx::iMarshFuncs
{
	size_t get_typecode (const teq::iTensor&) const override { return 0; }

	void marsh_leaf (
		InitBuildF build_init, SInitBuildF build_sinit,
		const teq::iLeaf& leaf) const override {}
};


TEST(SAVE, BadMarshal)
{
	auto* logger = new exam::MockLogger();
	global::set_logger(logger);
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));

	onnx::AttributeProto proto;
	teq::CTensMapT<std::string> tensid;
	onnx::OnnxAttrMarshaler marshal(&proto, tensid);

	marsh::ObjTuple tup;
	std::string fatalmsg = "onnx does not support tuple attributes";
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(tup.accept(marshal), fatalmsg.c_str());

	marsh::Maps mm;
	std::string fatalmsg1 = "onnx does not support map attributes";
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(mm.accept(marshal), fatalmsg1.c_str());

	auto var = make_var(teq::Shape(), "m");
	teq::LayerObj layer("banana_split", var);
	std::string fatalmsg2 = "onnx does not support layer attributes";
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg2, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg2)));
	EXPECT_FATAL(layer.accept(marshal), fatalmsg2.c_str());

	{
		auto nllog = new exam::NoSupportLogger();
		global::set_logger(nllog);
		tup.accept(marshal);
		EXPECT_FALSE(nllog->called_);
	}
	{
		auto nllog = new exam::NoSupportLogger();
		global::set_logger(nllog);
		mm.accept(marshal);
		EXPECT_FALSE(nllog->called_);
	}
	{
		auto nllog = new exam::NoSupportLogger();
		global::set_logger(nllog);
		layer.accept(marshal);
		EXPECT_FALSE(nllog->called_);
	}
}


TEST(SAVE, SimpleGraph)
{
	auto gen = std::make_shared<MockGenerator>();
	global::set_generator(gen);

	std::string expect_pbfile = testdir + "/simple_onnx.onnx";
#ifdef EXPORT_TESTDATA
	std::string got_pbfile = "/tmp/simple_onnx.onnx";
#else
	std::string got_pbfile = "got_simple_onnx.onnx";
#endif

	size_t counter = 0;
	auto incr_id = [&]{ return fmts::to_string(++counter); };

	EXPECT_CALL(*gen, get_str()).
		WillRepeatedly(Invoke(incr_id));

	{
		onnx::ModelProto model;
		std::vector<teq::TensptrT> roots;
		onnx::TensIdT ids;

		// subtree one
		teq::Shape shape({3, 7});
		teq::Shape shape2({7, 3});
		auto osrc = make_var(shape, "osrc");
		auto osrc2 = make_var(shape2, "osrc2");

		{
			teq::Shape shape3({3, 1, 7});
			auto src = make_var(shape, "src");
			auto src2 = make_var(shape3, "src2");

			auto f1 = make_fnc("neg", 3, teq::TensptrsT{osrc});
			auto f2 = make_fnc("sin", 5, teq::TensptrsT{src});
			auto f3 = make_fnc("+", 4, teq::TensptrsT{f2,src});
			auto f4 = make_fnc("/", 2, teq::TensptrsT{f1,f3});
			auto f5 = make_fnc("@", 1, teq::TensptrsT{f4,osrc2});
			auto dest = make_fnc("-", 0, teq::TensptrsT{src2,f5});
			EXPECT_CALL(*dest, shape()).WillRepeatedly(Return(teq::Shape({3, 1, 7})));
			roots.push_back(dest);
			ids.insert({dest.get(), "root1"});
		}

		// subtree two
		{
			teq::Shape mshape({3, 3});
			auto src = make_var(mshape, "s2src");
			auto src2 = make_var(mshape, "s2src2");
			auto src3 = make_var(mshape, "s2src3");

			auto f1 = make_fnc("abs", 7, teq::TensptrsT{src});
			auto f2 = make_fnc("exp", 8, teq::TensptrsT{src2});
			auto f3 = make_fnc("neg", 3, teq::TensptrsT{src3});
			auto f4 = make_fnc("*", 6, teq::TensptrsT{f1,f2,f3});
			auto dest = make_fnc("-", 0, teq::TensptrsT{src,f4});
			EXPECT_CALL(*dest, shape()).WillRepeatedly(Return(teq::Shape({3, 3})));
			roots.push_back(dest);
			ids.insert({dest.get(), "root2"});
		}

		MockMarshFuncs marsh;
		onnx::save_graph(*model.mutable_graph(), roots, marsh, ids);

		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(model.SerializeToOstream(&gotstr));
	}

	{
		std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
		std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
		ASSERT_TRUE(expect_ifs.is_open());
		ASSERT_TRUE(got_ifs.is_open());

		onnx::ModelProto expect_model;
		onnx::ModelProto got_model;
		ASSERT_TRUE(expect_model.ParseFromIstream(&expect_ifs));
		ASSERT_TRUE(got_model.ParseFromIstream(&got_ifs));

		google::protobuf::util::MessageDifferencer differ;
		std::string report;
		differ.ReportDifferencesToString(&report);
		EXPECT_TRUE(differ.Compare(expect_model, got_model)) << report;
	}
	global::set_generator(nullptr);
}


TEST(SAVE, LayerGraph)
{
	std::string expect_pbfile = testdir + "/layer_onnx.onnx";
#ifdef EXPORT_TESTDATA
	std::string got_pbfile = "/tmp/layer_onnx.onnx";
#else
	std::string got_pbfile = "got_layer_onnx.onnx";
#endif

	size_t counter = 0;
	auto incr_id = [&]{ return fmts::to_string(++counter); };

	auto gen = std::make_shared<MockGenerator>();
	global::set_generator(gen);
	EXPECT_CALL(*gen, get_str()).
		WillRepeatedly(Invoke(incr_id));

	{
		onnx::ModelProto model;
		std::vector<teq::TensptrT> roots;
		onnx::TensIdT ids;

		// subtree one
		teq::Shape shape({3, 7});
		teq::Shape shape2({7, 3});
		auto osrc = make_var(shape, "osrc");
		auto osrc2 = make_var(shape2, "osrc2");
		EXPECT_CALL(*osrc2, get_usage()).WillRepeatedly(Return(teq::PLACEHOLDER));

		marsh::Number<size_t> numobj(54);
		marsh::NumArray<double> arrobj(std::vector<double>{3.3, 2.1, 7.6});
		marsh::Number<double> numobj2(4.3);
		marsh::NumArray<size_t> arrobj2(std::vector<size_t>{4, 6, 2});
		std::shared_ptr<teq::LayerObj> layrobj = nullptr;

		auto attrib1 = make_fnc("cos", 10, teq::TensptrsT{osrc});
		auto attrib2 = make_fnc("cos", 10, teq::TensptrsT{osrc2});
		teq::LayerObj sololayr("solo", attrib1);
		teq::TensorObj hantens(attrib2);
		EXPECT_CALL(*attrib1, shape()).WillRepeatedly(Return(shape));
		EXPECT_CALL(*attrib2, shape()).WillRepeatedly(Return(shape));
		{
			teq::Shape shape3({3, 1, 7});
			auto src = make_var(shape, "src");
			auto src2 = make_var(shape3, "src2");

			auto f0 = make_fnc("sin", 5, teq::TensptrsT{src});
			EXPECT_CALL(*f0, size()).WillRepeatedly(Return(2));
			EXPECT_CALL(*f0, ls_attrs()).WillRepeatedly(Return(types::StringsT{"array","num"}));
			EXPECT_CALL(*f0, get_attr("array")).WillRepeatedly(Return(&arrobj));
			EXPECT_CALL(*f0, get_attr("num")).WillRepeatedly(Return(&numobj));
			EXPECT_CALL(Const(*f0), get_attr("array")).WillRepeatedly(Return(&arrobj));
			EXPECT_CALL(Const(*f0), get_attr("num")).WillRepeatedly(Return(&numobj));

			EXPECT_CALL(*f0, shape()).Times(1).WillRepeatedly(Return(teq::Shape({3, 7})));

			layrobj = std::make_shared<teq::LayerObj>("onion", f0);

			auto f1 = make_fnc("+", 4, teq::TensptrsT{f0, src});
			EXPECT_CALL(*f1, size()).WillRepeatedly(Return(3));
			EXPECT_CALL(*f1, ls_attrs()).WillRepeatedly(Return(types::StringsT{"array",teq::layer_attr,"num"}));
			EXPECT_CALL(*f1, get_attr("array")).WillRepeatedly(Return(&arrobj2));
			EXPECT_CALL(*f1, get_attr(teq::layer_attr)).WillRepeatedly(Return(layrobj.get()));
			EXPECT_CALL(*f1, get_attr("num")).WillRepeatedly(Return(&numobj2));
			EXPECT_CALL(Const(*f1), get_attr("array")).WillRepeatedly(Return(&arrobj2));
			EXPECT_CALL(Const(*f1), get_attr(teq::layer_attr)).WillRepeatedly(Return(layrobj.get()));
			EXPECT_CALL(Const(*f1), get_attr("num")).WillRepeatedly(Return(&numobj2));

			EXPECT_CALL(*f1, shape()).Times(1).WillRepeatedly(Return(teq::Shape({3, 7})));

			auto f2 = make_fnc("neg", 3, teq::TensptrsT{osrc});
			EXPECT_CALL(*f2, size()).WillRepeatedly(Return(1));
			EXPECT_CALL(*f2, ls_attrs()).WillRepeatedly(Return(types::StringsT{teq::layer_attr}));
			EXPECT_CALL(*f2, get_attr(teq::layer_attr)).WillRepeatedly(Return(&sololayr));
			EXPECT_CALL(Const(*f2), get_attr(teq::layer_attr)).WillRepeatedly(Return(&sololayr));

			EXPECT_CALL(*f2, shape()).Times(1).WillRepeatedly(Return(teq::Shape({3, 7})));

			auto f3 = make_fnc("/", 2, teq::TensptrsT{f2,f1});
			EXPECT_CALL(*f3, size()).WillRepeatedly(Return(1));
			EXPECT_CALL(*f3, ls_attrs()).WillRepeatedly(Return(types::StringsT{"tens_stuff"}));
			EXPECT_CALL(*f3, get_attr("tens_stuff")).WillRepeatedly(Return(&hantens));
			EXPECT_CALL(Const(*f3), get_attr("tens_stuff")).WillRepeatedly(Return(&hantens));

			EXPECT_CALL(*f3, shape()).WillRepeatedly(Return(teq::Shape({3, 7})));

			auto f4 = make_fnc("@", 1, teq::TensptrsT{f3,osrc2});
			auto dest = make_fnc("-", 0, teq::TensptrsT{src2,f4});
			EXPECT_CALL(*dest, shape()).WillRepeatedly(Return(teq::Shape({3, 1, 7})));

			roots.push_back(dest);
			ids.insert({dest.get(), "root1"});
		}

		marsh::PtrArray<marsh::String> strsobj;
		strsobj.contents_.insert(strsobj.contents_.end(),
			std::make_unique<marsh::String>("world"));
		strsobj.contents_.insert(strsobj.contents_.end(),
			std::make_unique<marsh::String>("food"));
		marsh::PtrArray<teq::TensorObj> tensorsobj;
		tensorsobj.contents_.insert(tensorsobj.contents_.end(),
			std::make_unique<teq::TensorObj>(osrc));
		marsh::String strobj("hello");
		teq::TensorObj tensorobj(osrc);

		// subtree two
		{
			teq::Shape mshape({3, 3});
			auto src = make_var(mshape, "s2src");
			auto src2 = make_var(mshape, "s2src2");
			auto src3 = make_var(mshape, "s2src3");

			tensorsobj.contents_.insert(tensorsobj.contents_.end(),
				std::make_unique<teq::TensorObj>(src3));

			auto f0 = make_fnc("abs", 7, teq::TensptrsT{src});
			EXPECT_CALL(*f0, size()).WillRepeatedly(Return(2));
			EXPECT_CALL(*f0, ls_attrs()).WillRepeatedly(Return(types::StringsT{"strs","tensors"}));
			EXPECT_CALL(*f0, get_attr("strs")).WillRepeatedly(Return(&strsobj));
			EXPECT_CALL(*f0, get_attr("tensors")).WillRepeatedly(Return(&tensorsobj));
			EXPECT_CALL(Const(*f0), get_attr("strs")).WillRepeatedly(Return(&strsobj));
			EXPECT_CALL(Const(*f0), get_attr("tensors")).WillRepeatedly(Return(&tensorsobj));

			auto f2 = make_fnc("exp", 8, teq::TensptrsT{src2});
			auto f3 = make_fnc("neg", 3, teq::TensptrsT{src3});

			auto f1 = make_fnc("*", 6, teq::TensptrsT{f0,f2,f3});
			EXPECT_CALL(*f1, size()).WillRepeatedly(Return(1));
			EXPECT_CALL(*f1, ls_attrs()).WillRepeatedly(Return(types::StringsT{"str"}));
			EXPECT_CALL(*f1, get_attr("str")).WillRepeatedly(Return(&strobj));
			EXPECT_CALL(Const(*f1), get_attr("str")).WillRepeatedly(Return(&strobj));

			auto dest = make_fnc("-", 0, teq::TensptrsT{src, f1});
			EXPECT_CALL(*dest, size()).WillRepeatedly(Return(1));
			EXPECT_CALL(*dest, ls_attrs()).WillRepeatedly(Return(types::StringsT{"pineapple"}));
			EXPECT_CALL(*dest, get_attr("pineapple")).WillRepeatedly(Return(&tensorobj));
			EXPECT_CALL(Const(*dest), get_attr("pineapple")).WillRepeatedly(Return(&tensorobj));
			EXPECT_CALL(*dest, shape()).WillRepeatedly(Return(teq::Shape({3, 3})));

			roots.push_back(dest);
			ids.insert({dest.get(), "root2"});
		}

		MockMarshFuncs marsh;
		onnx::save_graph(*model.mutable_graph(), roots, marsh, ids);

		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(model.SerializeToOstream(&gotstr));
	}

	{
		std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
		std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
		ASSERT_TRUE(expect_ifs.is_open());
		ASSERT_TRUE(got_ifs.is_open());

		onnx::ModelProto expect_model;
		onnx::ModelProto got_model;
		ASSERT_TRUE(expect_model.ParseFromIstream(&expect_ifs));
		ASSERT_TRUE(got_model.ParseFromIstream(&got_ifs));

		google::protobuf::util::MessageDifferencer differ;
		std::string report;
		differ.ReportDifferencesToString(&report);
		EXPECT_TRUE(differ.Compare(expect_model, got_model)) << report;
	}
	global::set_generator(nullptr);
}


TEST(SAVE, SimpleGraphEarlyStop)
{
	std::string expect_pbfile = testdir + "/simple_stop.onnx";
#ifdef EXPORT_TESTDATA
	std::string got_pbfile = "/tmp/simple_stop.onnx";
#else
	std::string got_pbfile = "got_simple_stop.onnx";
#endif

	size_t counter = 0;
	auto incr_id = [&]{ return fmts::to_string(++counter); };

	auto gen = std::make_shared<MockGenerator>();
	global::set_generator(gen);
	EXPECT_CALL(*gen, get_str()).
		WillRepeatedly(Invoke(incr_id));

	{
		onnx::ModelProto model;
		std::vector<teq::TensptrT> roots;
		onnx::TensIdT ids;
		teq::TensSetT stops;

		// subtree one
		teq::Shape shape({3, 7});
		teq::Shape shape2({7, 3});
		auto osrc = make_var(shape, "osrc");
		auto osrc2 = make_var(shape2, "osrc2");

		{
			teq::Shape shape3({3, 1, 7});
			auto src = make_var(shape, "src");
			auto src2 = make_var(shape3, "src2");

			auto f0 = make_fnc("sin", 5, teq::TensptrsT{src});
			auto f = make_fnc("+", 4, teq::TensptrsT{f0,src});
			auto f1 = make_fnc("neg", 3, teq::TensptrsT{osrc});
			auto f2 = make_fnc("/", 2, teq::TensptrsT{f1,f});
			auto f3 = make_fnc("@", 1, teq::TensptrsT{f2,osrc2});
			auto dest = make_fnc("-", 0, teq::TensptrsT{src2,f3});
			EXPECT_CALL(*dest, shape()).WillRepeatedly(Return(teq::Shape({3, 1, 7})));
			EXPECT_CALL(*f, shape()).WillRepeatedly(Return(teq::Shape({3, 7})));

			stops.emplace(src2.get());
			stops.emplace(f.get());
			roots.push_back(dest);
			ids.insert({dest.get(), "root1"});
		}

		// subtree two
		{
			teq::Shape mshape({3, 3});
			auto src = make_var(mshape, "s2src");
			auto src2 = make_var(mshape, "s2src2");
			auto src3 = make_var(mshape, "s2src3");

			auto f = make_fnc("abs", 7, teq::TensptrsT{src});
			auto f2 = make_fnc("neg", 3, teq::TensptrsT{src3});
			auto f3 = make_fnc("exp", 8, teq::TensptrsT{src2});
			auto f4 = make_fnc("*", 6, teq::TensptrsT{f,f3,f2});
			auto dest = make_fnc("-", 0, teq::TensptrsT{src,f4});
			EXPECT_CALL(*dest, shape()).WillRepeatedly(Return(teq::Shape({3, 3})));
			EXPECT_CALL(*f, shape()).WillRepeatedly(Return(teq::Shape({3, 3})));
			EXPECT_CALL(*f2, shape()).WillRepeatedly(Return(teq::Shape({3, 3})));

			stops.emplace(f.get());
			stops.emplace(f2.get());
			roots.push_back(dest);
			ids.insert({dest.get(), "root2"});
		}

		MockMarshFuncs marsh;
		onnx::save_graph(*model.mutable_graph(),
			roots, marsh, ids, stops);

		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(model.SerializeToOstream(&gotstr));
	}

	{
		std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
		std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
		ASSERT_TRUE(expect_ifs.is_open());
		ASSERT_TRUE(got_ifs.is_open());

		onnx::ModelProto expect_model;
		onnx::ModelProto got_model;
		ASSERT_TRUE(expect_model.ParseFromIstream(&expect_ifs));
		ASSERT_TRUE(got_model.ParseFromIstream(&got_ifs));

		google::protobuf::util::MessageDifferencer differ;
		std::string report;
		differ.ReportDifferencesToString(&report);
		EXPECT_TRUE(differ.Compare(expect_model, got_model)) << report;
	}
}


#endif // DISABLE_ONNX_SAVE_TEST
