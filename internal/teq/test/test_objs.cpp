
#ifndef DISABLE_TEQ_OBJS_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/marsh/mock/mock.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/teq/objs.hpp"
#include "internal/teq/findattr.hpp"


using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::An;


struct OBJ : public tutil::TestcaseWithLogger<> {};


TEST_F(OBJ, TensorObj)
{
	marsh::String str;
	teq::Shape ashape({1, 2, 3});
	teq::Shape bshape({4, 2, 3});
	auto a = make_var(ashape, "A");
	auto b = make_var(bshape, "B");

	teq::TensorObj tensobj(a);
	auto direct_clone = tensobj.clone();
	auto buildout = tensobj.copynreplace(b);

	EXPECT_EQ(a, tensobj.get_tensor());
	EXPECT_EQ(a, direct_clone->get_tensor());
	EXPECT_EQ(b, buildout->get_tensor());

	EXPECT_EQ(typeid(teq::TensorObj).hash_code(), tensobj.class_code());
	EXPECT_STREQ("A", tensobj.to_string().c_str());
	EXPECT_STREQ("B", buildout->to_string().c_str());

	EXPECT_TRUE(tensobj.equals(tensobj));
	EXPECT_TRUE(direct_clone->equals(tensobj));
	EXPECT_FALSE(buildout->equals(tensobj));
	EXPECT_FALSE(buildout->equals(str));

	const teq::TensorObj* captens = nullptr;
	auto capture_tens = [&](const teq::TensorObj& arg){ captens = &arg; };

	MockTeqMarsh mmarsh;
	EXPECT_CALL(mmarsh, marshal(An<const teq::TensorObj&>())).Times(1).WillOnce(Invoke(capture_tens));
	tensobj.accept(mmarsh);
	EXPECT_EQ(&tensobj, captens);

	EXPECT_CALL(*logger_, supports_level(logs::warn_level)).Times(1).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::warn_level, "non-teq marshaler cannot marshal tensor-typed objects", _)).Times(1);

	MockMarsh gmarsh;
	tensobj.accept(gmarsh);

	delete direct_clone;
	delete buildout;
}


TEST_F(OBJ, LayerObj)
{
	marsh::String str;
	auto a = make_var(teq::Shape({1, 2, 3}), "A");
	auto b = make_var(teq::Shape({4, 2, 3}), "B");

	teq::LayerObj layerobj("lasagna", a);
	teq::LayerObj layerobj2("nacho_dip", a);
	auto direct_clone = layerobj.clone();
	auto buildout = layerobj.copynreplace(b);

	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));
	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, "cannot `sandwich` with null input", _)).Times(1);
	teq::LayerObj("sandwich", nullptr);

	EXPECT_EQ(a, layerobj.get_tensor());
	EXPECT_EQ(a, direct_clone->get_tensor());
	EXPECT_EQ(b, buildout->get_tensor());

	EXPECT_EQ(typeid(teq::LayerObj).hash_code(), layerobj.class_code());
	EXPECT_STREQ("lasagna", layerobj.to_string().c_str());
	EXPECT_STREQ("lasagna", buildout->to_string().c_str());

	EXPECT_TRUE(layerobj.equals(layerobj));
	EXPECT_TRUE(direct_clone->equals(layerobj));
	EXPECT_FALSE(buildout->equals(layerobj));
	EXPECT_FALSE(buildout->equals(layerobj2));
	EXPECT_FALSE(buildout->equals(str));

	const teq::LayerObj* captens = nullptr;
	auto capture_layr = [&](const teq::LayerObj& arg){ captens = &arg; };

	MockTeqMarsh mmarsh;
	EXPECT_CALL(mmarsh, marshal(An<const teq::LayerObj&>())).Times(1).WillOnce(Invoke(capture_layr));
	layerobj.accept(mmarsh);
	EXPECT_EQ(&layerobj, captens);

	EXPECT_CALL(*logger_, supports_level(logs::warn_level)).Times(1).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::warn_level, "non-teq marshaler cannot marshal layer-typed objects", _)).Times(1);

	MockMarsh gmarsh;
	layerobj.accept(gmarsh);

	delete direct_clone;
	delete buildout;
}


TEST_F(OBJ, FindAttrs)
{
	auto a = make_var(teq::Shape({1, 2, 3}), "A");
	auto b = make_var(teq::Shape({4, 2, 3}), "B");
	auto c = make_var(teq::Shape({4, 1, 3}), "C");

	marsh::Maps root;
	root.add_attr("obj1",
		std::make_unique<marsh::PtrArray<teq::TensorObj>>());
	root.add_attr("obj2",
		std::make_unique<marsh::Number<float>>(2.3));
	root.add_attr("obj3",
		std::make_unique<marsh::ObjTuple>());
	root.add_attr("obj4",
		std::make_unique<teq::LayerObj>("onion", b));
	root.add_attr("null",std::unique_ptr<marsh::String>());

	auto arr = static_cast<marsh::PtrArray<teq::TensorObj>*>(
		root.get_attr("obj1"));
	auto tup = static_cast<marsh::ObjTuple*>(root.get_attr("obj3"));

	arr->contents_.insert(arr->contents_.end(),
		std::make_unique<teq::TensorObj>(a));
	arr->contents_.insert(arr->contents_.end(),
		std::make_unique<teq::TensorObj>(c));

	tup->contents_.insert(tup->contents_.end(),
		std::make_unique<marsh::String>("zzz"));
	tup->contents_.insert(tup->contents_.end(),
		std::make_unique<teq::TensorObj>(b));

	teq::FindTensAttr finder;
	root.accept(finder);
	EXPECT_EQ(4, finder.tens_.size());

	EXPECT_ARRHAS(finder.tens_, a);
	EXPECT_ARRHAS(finder.tens_, b);
	EXPECT_ARRHAS(finder.tens_, c);

	auto d = make_var(teq::Shape({4, 1, 3}), "D");
	auto e = make_var(teq::Shape({4, 1, 3}), "E");
	teq::LayerObj layer("sundae", d);
	teq::TensorObj tensor(e);

	EXPECT_STREQ("sundae", layer.get_opname().c_str());

	teq::FindTensAttr finder2;
	layer.accept(finder2);
	tensor.accept(finder2);
	EXPECT_EQ(2, finder2.tens_.size());

	EXPECT_ARRHAS(finder2.tens_, d);
	EXPECT_ARRHAS(finder2.tens_, e);
}


#endif // DISABLE_TEQ_OBJ_TEST
