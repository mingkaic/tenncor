
#ifndef DISABLE_TEQ_OBJ_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/teq/mock/leaf.hpp"
#include "internal/teq/mock/functor.hpp"
#include "internal/teq/mock/marshal.hpp"

#include "internal/marsh/mock/marshal.hpp"

#include "internal/teq/objs.hpp"


TEST(OBJ, TensorObj)
{
	marsh::String str;
	teq::TensptrT a(new MockLeaf(teq::Shape({1, 2, 3}), "A"));
	teq::TensptrT b(new MockLeaf(teq::Shape({4, 2, 3}), "B"));

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

	MockTeqMarsh mmarsh;
	tensobj.accept(mmarsh);
	EXPECT_HAS(mmarsh.visited_, &tensobj);

	MockMarsh gmarsh;
	tensobj.accept(gmarsh);
	EXPECT_HASNOT(gmarsh.visited_, &tensobj);

	delete direct_clone;
	delete buildout;
}


TEST(OBJ, LayerObj)
{
	marsh::String str;
	teq::TensptrT a(new MockLeaf(teq::Shape({1, 2, 3}), "A"));
	teq::TensptrT b(new MockLeaf(teq::Shape({4, 2, 3}), "B"));

	teq::LayerObj layerobj("lasagna", a);
	teq::LayerObj layerobj2("nacho_dip", a);
	auto direct_clone = layerobj.clone();
	auto buildout = layerobj.copynreplace(b);

	EXPECT_FATAL(teq::LayerObj("sandwich", nullptr),
		"cannot `sandwich` with null input");

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

	MockTeqMarsh mmarsh;
	layerobj.accept(mmarsh);
	EXPECT_HAS(mmarsh.visited_, &layerobj);

	MockMarsh gmarsh;
	layerobj.accept(gmarsh);
	EXPECT_HASNOT(gmarsh.visited_, &layerobj);

	delete direct_clone;
	delete buildout;
}


#endif // DISABLE_OBJ_TEST
