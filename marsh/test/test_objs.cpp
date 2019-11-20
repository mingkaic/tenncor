
#ifndef DISABLE_OBJS_TEST


#include "gtest/gtest.h"

#include "marsh/objs.hpp"

#include "marsh/test/json_marsh.hpp"


TEST(OBJS, Number)
{
	marsh::Number<double> numba_wun(1.11);
	marsh::Number<size_t> numba_deux(2);
	marsh::Number<float> numba_tres(3.3);

	marsh::Number<double> numba_tree(3.3);
	marsh::Number<size_t> diff(3);
	marsh::Number<size_t> same(2);

	EXPECT_DOUBLE_EQ(1.11, numba_wun.to_float64());
	EXPECT_DOUBLE_EQ(2, numba_deux.to_float64());
	EXPECT_DOUBLE_EQ(3.3f, numba_tres.to_float64());

	EXPECT_EQ(1, numba_wun.to_int64());
	EXPECT_EQ(2, numba_deux.to_int64());
	EXPECT_EQ(3, numba_tres.to_int64());

	marsh::JsonMarshaler parser;
	EXPECT_STREQ("1.11", parser.parse(numba_wun).c_str());
	EXPECT_STREQ("2", parser.parse(numba_deux).c_str());
	EXPECT_STREQ("3.3", parser.parse(numba_tres).c_str());

	EXPECT_FALSE(numba_tree.equals(numba_tres));
	EXPECT_FALSE(numba_tres.equals(numba_tree));
	EXPECT_FALSE(diff.equals(numba_deux));
	EXPECT_FALSE(numba_deux.equals(diff));
	EXPECT_TRUE(same.equals(numba_deux));
	EXPECT_TRUE(numba_deux.equals(same));
}


TEST(OBJS, ObjArray)
{
	marsh::ObjArray root;
	root.contents_.emplace(root.contents_.end(),
		std::make_unique<marsh::ObjArray>());
	root.contents_.emplace(root.contents_.end(),
		std::make_unique<marsh::Number<double>>(3.3));

	auto sub = static_cast<marsh::ObjArray*>(root.contents_[0].get());
	sub->contents_.emplace(sub->contents_.end(),
		std::make_unique<marsh::Number<size_t>>(2));
	sub->contents_.emplace(sub->contents_.end(),
		std::make_unique<marsh::Number<float>>(1.11));

	EXPECT_EQ(2, root.size());
	EXPECT_EQ(2, sub->size());

	std::vector<marsh::iObject*> root_refs;
	root.foreach([&root_refs](marsh::ObjptrT& obj) { root_refs.push_back(obj.get()); });
	ASSERT_EQ(2, root_refs.size());
	EXPECT_EQ(sub, root_refs[0]);
	EXPECT_TRUE(root_refs[1]->equals(marsh::Number<double>(3.3)));

	std::vector<marsh::iObject*> sub_refs;
	sub->foreach([&sub_refs](marsh::ObjptrT& obj) { sub_refs.push_back(obj.get()); });
	ASSERT_EQ(2, sub_refs.size());
	EXPECT_TRUE(sub_refs[0]->equals(marsh::Number<size_t>(2)));
	EXPECT_TRUE(sub_refs[1]->equals(marsh::Number<float>(1.11)));

	marsh::ObjArray root_clone;
	{
		root_clone.contents_.emplace(root_clone.contents_.end(),
			std::make_unique<marsh::ObjArray>());
		root_clone.contents_.emplace(root_clone.contents_.end(),
			std::make_unique<marsh::Number<double>>(3.3));

		auto sub = static_cast<marsh::ObjArray*>(root_clone.contents_[0].get());
		sub->contents_.emplace(sub->contents_.end(),
			std::make_unique<marsh::Number<size_t>>(2));
		sub->contents_.emplace(sub->contents_.end(),
			std::make_unique<marsh::Number<float>>(1.11));
	}
	EXPECT_TRUE(root.equals(root_clone));
	EXPECT_TRUE(root_clone.equals(root));

	marsh::ObjArray imperfect_clone; // demonstrate order matters
	{
		imperfect_clone.contents_.emplace(imperfect_clone.contents_.end(),
			std::make_unique<marsh::Number<double>>(3.3));
		imperfect_clone.contents_.emplace(imperfect_clone.contents_.end(),
			std::make_unique<marsh::ObjArray>());

		auto sub = static_cast<marsh::ObjArray*>(imperfect_clone.contents_[1].get());
		sub->contents_.emplace(sub->contents_.end(),
			std::make_unique<marsh::Number<float>>(1.11));
		sub->contents_.emplace(sub->contents_.end(),
			std::make_unique<marsh::Number<size_t>>(2));
	}
	EXPECT_FALSE(root.equals(imperfect_clone));
	EXPECT_FALSE(imperfect_clone.equals(root));

	marsh::ObjArray empty;
	marsh::JsonMarshaler parser;
	std::string parsed_root = parser.parse(root, false);
	fmts::trim(parsed_root);
	EXPECT_STREQ("{\"\":[\"2\",\"1.11\"],\"\":\"3.3\"}", parsed_root.c_str());
	EXPECT_STREQ("[]", parser.parse(empty, false).c_str());
}


// todo: test these
// TEST(OBJS, NumArray)
// {}


// TEST(OBJS, Maps)
// {}


#endif // DISABLE_OBJS_TEST
