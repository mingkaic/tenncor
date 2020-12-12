
#ifndef DISABLE_MARSH_OBJS_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/marsh/objs.hpp"

#include "internal/marsh/mock/mock.hpp"


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

	EXPECT_FALSE(numba_wun.is_integral());
	ASSERT_TRUE(numba_deux.is_integral());
	EXPECT_FALSE(numba_tres.is_integral());

	auto one = numba_wun.clone();
	auto two = numba_deux.clone();
	auto three = numba_tres.clone();

	ASSERT_TRUE(one->equals(numba_wun));
	ASSERT_TRUE(two->equals(numba_deux));
	ASSERT_TRUE(three->equals(numba_tres));

	delete one;
	delete two;
	delete three;
}


TEST(OBJS, ObjTuple)
{
	marsh::ObjTuple root;
	root.contents_.emplace(root.contents_.end(),
		std::make_unique<marsh::ObjTuple>());
	root.contents_.emplace(root.contents_.end(),
		std::make_unique<marsh::Number<double>>(3.3));

	auto sub = static_cast<marsh::ObjTuple*>(root.contents_[0].get());
	sub->contents_.emplace(sub->contents_.end(),
		std::make_unique<marsh::Number<size_t>>(2));
	sub->contents_.emplace(sub->contents_.end(),
		std::make_unique<marsh::Number<float>>(1.11));

	EXPECT_EQ(2, root.size());
	EXPECT_EQ(2, sub->size());

	EXPECT_STREQ("[[2\\1.11]\\3.3]", root.to_string().c_str());

	std::vector<marsh::iObject*> root_refs;
	root.foreach([&root_refs](size_t, marsh::iObject* obj)
		{ root_refs.push_back(obj); });
	ASSERT_EQ(2, root_refs.size());
	EXPECT_EQ(sub, root_refs[0]);
	EXPECT_TRUE(root_refs[1]->equals(marsh::Number<double>(3.3)));

	std::vector<marsh::iObject*> sub_refs;
	sub->foreach([&sub_refs](size_t i, marsh::iObject* obj)
		{ sub_refs.push_back(obj); });
	ASSERT_EQ(2, sub_refs.size());
	EXPECT_TRUE(sub_refs[0]->equals(marsh::Number<size_t>(2)));
	EXPECT_TRUE(sub_refs[1]->equals(marsh::Number<float>(1.11)));

	marsh::ObjTuple root_clone;
	{
		root_clone.contents_.emplace(root_clone.contents_.end(),
			std::make_unique<marsh::ObjTuple>());
		root_clone.contents_.emplace(root_clone.contents_.end(),
			std::make_unique<marsh::Number<double>>(3.3));

		auto sub = static_cast<marsh::ObjTuple*>(root_clone.contents_[0].get());
		sub->contents_.emplace(sub->contents_.end(),
			std::make_unique<marsh::Number<size_t>>(2));
		sub->contents_.emplace(sub->contents_.end(),
			std::make_unique<marsh::Number<float>>(1.11));
	}
	EXPECT_TRUE(root.equals(root_clone));
	EXPECT_TRUE(root_clone.equals(root));

	marsh::ObjTuple big_root;
	{
		big_root.contents_.emplace(big_root.contents_.end(),
			std::make_unique<marsh::ObjTuple>());
		big_root.contents_.emplace(big_root.contents_.end(),
			std::make_unique<marsh::Number<double>>(3.3));
		big_root.contents_.emplace(big_root.contents_.end(),
			std::make_unique<marsh::Number<double>>(3.4));

		auto sub = static_cast<marsh::ObjTuple*>(big_root.contents_[0].get());
		sub->contents_.emplace(sub->contents_.end(),
			std::make_unique<marsh::Number<size_t>>(2));
		sub->contents_.emplace(sub->contents_.end(),
			std::make_unique<marsh::Number<float>>(1.11));
	}
	EXPECT_FALSE(root.equals(big_root));
	EXPECT_FALSE(big_root.equals(root));

	marsh::ObjTuple imperfect_clone; // demonstrate order matters
	{
		imperfect_clone.contents_.emplace(imperfect_clone.contents_.end(),
			std::make_unique<marsh::Number<double>>(3.3));
		imperfect_clone.contents_.emplace(imperfect_clone.contents_.end(),
			std::make_unique<marsh::ObjTuple>());

		auto sub = static_cast<marsh::ObjTuple*>(imperfect_clone.contents_[1].get());
		sub->contents_.emplace(sub->contents_.end(),
			std::make_unique<marsh::Number<float>>(1.11));
		sub->contents_.emplace(sub->contents_.end(),
			std::make_unique<marsh::Number<size_t>>(2));
	}
	EXPECT_FALSE(root.equals(imperfect_clone));
	EXPECT_FALSE(imperfect_clone.equals(root));

	marsh::ObjTuple empty;
	marsh::JsonMarshaler parser;
	std::string parsed_root = parser.parse(root, false);
	fmts::trim(parsed_root);
	EXPECT_STREQ("{\"\":[\"2\",\"1.11\"],\"\":\"3.3\"}", parsed_root.c_str());
	EXPECT_STREQ("[]", parser.parse(empty, false).c_str());

	auto r = root.clone();
	auto c = root_clone.clone();
	auto big = big_root.clone();
	auto imperfect = imperfect_clone.clone();

	ASSERT_TRUE(r->equals(root));
	ASSERT_TRUE(c->equals(root_clone));
	ASSERT_TRUE(big->equals(big_root));
	ASSERT_TRUE(imperfect->equals(imperfect_clone));

	delete r;
	delete c;
	delete big;
	delete imperfect;
}


TEST(OBJS, NumArray)
{
	marsh::NumArray<double> root({2, 3.3});
	EXPECT_EQ(0, root.subclass_code());

	EXPECT_EQ(2, root.size());
	EXPECT_STREQ("[2\\3.3]", root.to_string().c_str());

	marsh::NumArray<size_t> empty;
	EXPECT_FALSE(root.equals(empty));
	EXPECT_FALSE(empty.equals(root));

	marsh::NumArray<double> root_clone;
	root_clone.contents_ = {2, 3.3};
	EXPECT_TRUE(root.equals(root_clone));
	EXPECT_TRUE(root_clone.equals(root));

	marsh::NumArray<double> big_root;
	big_root.contents_ = {2, 3.3, 5.2};
	EXPECT_FALSE(root.equals(big_root));
	EXPECT_FALSE(big_root.equals(root));

	marsh::NumArray<double> imperfect_clone;
	imperfect_clone.contents_ = {2, 3};
	EXPECT_FALSE(root.equals(imperfect_clone));
	EXPECT_FALSE(imperfect_clone.equals(root));

	std::vector<double> values;
	root.foreach(
		[&values](size_t i, marsh::iObject* obj)
		{
			ASSERT_EQ(typeid(marsh::Number<double>).hash_code(),
				obj->class_code());
			auto num = static_cast<marsh::Number<double>*>(obj);
			values.push_back(num->val_);
		});
	ASSERT_EQ(2, values.size());
	EXPECT_DOUBLE_EQ(2, values[0]);
	EXPECT_DOUBLE_EQ(3.3, values[1]);

	marsh::JsonMarshaler parser;
	std::string parsed_root = parser.parse(root, false);
	fmts::trim(parsed_root);
	EXPECT_STREQ("{\"\":\"2\",\"\":\"3.3\"}", parsed_root.c_str());
	EXPECT_STREQ("[]", parser.parse(empty, false).c_str());

	EXPECT_TRUE(empty.is_primitive());
	EXPECT_TRUE(empty.is_integral());

	auto r = root.clone();
	auto e = empty.clone();
	auto big = big_root.clone();
	auto imperfect = imperfect_clone.clone();

	ASSERT_TRUE(r->equals(root));
	ASSERT_TRUE(e->equals(empty));
	ASSERT_TRUE(big->equals(big_root));
	ASSERT_TRUE(imperfect->equals(imperfect_clone));

	delete r;
	delete e;
	delete big;
	delete imperfect;
}


TEST(OBJS, PtrArray)
{
	marsh::PtrArray<marsh::Number<double>> root;
	root.contents_.insert(root.contents_.end(),
		std::make_unique<marsh::Number<double>>(2));
	root.contents_.insert(root.contents_.end(),
		std::make_unique<marsh::Number<double>>(3.3));
	EXPECT_EQ(typeid(marsh::Number<double>).hash_code(),
		root.subclass_code());

	marsh::String str;
	EXPECT_FALSE(root.equals(str));

	EXPECT_EQ(2, root.size());
	EXPECT_STREQ("[2\\3.3]", root.to_string().c_str());

	marsh::PtrArray<marsh::Number<double>> empty;
	EXPECT_FALSE(root.equals(empty));
	EXPECT_FALSE(empty.equals(root));

	marsh::PtrArray<marsh::Number<double>> root_clone;
	root_clone.contents_.insert(root_clone.contents_.end(),
		std::make_unique<marsh::Number<double>>(2));
	root_clone.contents_.insert(root_clone.contents_.end(),
		std::make_unique<marsh::Number<double>>(3.3));
	EXPECT_TRUE(root.equals(root_clone));
	EXPECT_TRUE(root_clone.equals(root));

	marsh::PtrArray<marsh::Number<double>> big_root;
	big_root.contents_.insert(big_root.contents_.end(),
		std::make_unique<marsh::Number<double>>(2));
	big_root.contents_.insert(big_root.contents_.end(),
		std::make_unique<marsh::Number<double>>(3.3));
	big_root.contents_.insert(big_root.contents_.end(),
		std::make_unique<marsh::Number<double>>(5.2));
	EXPECT_FALSE(root.equals(big_root));
	EXPECT_FALSE(big_root.equals(root));

	marsh::PtrArray<marsh::Number<double>> imperfect_clone;
	imperfect_clone.contents_.insert(imperfect_clone.contents_.end(),
		std::make_unique<marsh::Number<double>>(2));
	imperfect_clone.contents_.insert(imperfect_clone.contents_.end(),
		std::make_unique<marsh::Number<double>>(3));
	EXPECT_FALSE(root.equals(imperfect_clone));
	EXPECT_FALSE(imperfect_clone.equals(root));

	std::vector<double> values;
	root.foreach(
		[&values](size_t i, marsh::iObject* obj)
		{
			ASSERT_EQ(typeid(marsh::Number<double>).hash_code(),
				obj->class_code());
			auto num = static_cast<marsh::Number<double>*>(obj);
			values.push_back(num->val_);
		});
	ASSERT_EQ(2, values.size());
	EXPECT_DOUBLE_EQ(2, values[0]);
	EXPECT_DOUBLE_EQ(3.3, values[1]);

	marsh::JsonMarshaler parser;
	std::string parsed_root = parser.parse(root, false);
	fmts::trim(parsed_root);
	EXPECT_STREQ("{\"\":\"2\",\"\":\"3.3\"}", parsed_root.c_str());
	EXPECT_STREQ("[]", parser.parse(empty, false).c_str());

	EXPECT_FALSE(empty.is_primitive());
	EXPECT_FALSE(empty.is_integral());

	auto r = root.clone();
	auto e = empty.clone();
	auto big = big_root.clone();
	auto imperfect = imperfect_clone.clone();

	ASSERT_TRUE(r->equals(root));
	ASSERT_TRUE(e->equals(empty));
	ASSERT_TRUE(big->equals(big_root));
	ASSERT_TRUE(imperfect->equals(imperfect_clone));

	delete r;
	delete e;
	delete big;
	delete imperfect;
}


TEST(OBJS, Maps)
{
	marsh::Maps root;
	root.add_attr("obj1",
		std::make_unique<marsh::NumArray<size_t>>());
	root.add_attr("obj2",
		std::make_unique<marsh::Number<float>>(2.3));
	EXPECT_STREQ("[obj1:[]\\obj2:2.3]", root.to_string().c_str());
	EXPECT_EQ(2, root.size());

	marsh::Maps empty;
	EXPECT_EQ(0, empty.size());

	EXPECT_FALSE(root.equals(empty));
	EXPECT_FALSE(empty.equals(root));

	marsh::Maps root_clone;
	{
		root_clone.add_attr("obj1",
			std::make_unique<marsh::NumArray<size_t>>());
		root_clone.add_attr("obj2",
			std::make_unique<marsh::Number<float>>(2.3));
	}
	EXPECT_TRUE(root.equals(root_clone));
	EXPECT_TRUE(root_clone.equals(root));

	marsh::Maps big_root;
	{
		big_root.add_attr("obj1",
			std::make_unique<marsh::NumArray<size_t>>());
		big_root.add_attr("obj2",
			std::make_unique<marsh::Number<float>>(2.3));
		big_root.add_attr("obj3",
			std::make_unique<marsh::ObjTuple>());
	}
	EXPECT_FALSE(root.equals(big_root));
	EXPECT_FALSE(big_root.equals(root));

	auto numar = big_root.get_attr("obj1");
	auto numf = big_root.get_attr("obj2");
	auto objtup = big_root.get_attr("obj3");
	EXPECT_EQ(typeid(marsh::NumArray<size_t>).hash_code(), numar->class_code());
	EXPECT_EQ(typeid(marsh::Number<float>).hash_code(), numf->class_code());
	EXPECT_EQ(typeid(marsh::ObjTuple).hash_code(), objtup->class_code());

	marsh::Maps imperfect_clone;
	{
		imperfect_clone.add_attr("obj1",
			std::make_unique<marsh::Number<float>>(2.3));
		imperfect_clone.add_attr("obj2",
			std::make_unique<marsh::NumArray<size_t>>());
	}
	EXPECT_FALSE(root.equals(imperfect_clone));
	EXPECT_FALSE(imperfect_clone.equals(root));

	marsh::Number<size_t> wun(3);
	EXPECT_FALSE(root.equals(wun));
	EXPECT_FALSE(wun.equals(root));

	marsh::JsonMarshaler parser;
	std::string parsed_root = parser.parse(root, false);
	fmts::trim(parsed_root);
	EXPECT_STREQ("{\"obj1\":\"\",\"obj2\":\"2.3\"}", parsed_root.c_str());
	EXPECT_STREQ("[]", parser.parse(empty, false).c_str());

	auto r = root.clone();
	auto c = root_clone.clone();
	auto big = big_root.clone();
	auto imperfect = imperfect_clone.clone();

	ASSERT_TRUE(r->equals(root));
	ASSERT_TRUE(c->equals(root_clone));
	ASSERT_TRUE(big->equals(big_root));
	ASSERT_TRUE(imperfect->equals(imperfect_clone));

	big_root.rm_attr("obj2");
	auto keys = big_root.ls_attrs();
	ASSERT_EQ(2, keys.size());
	EXPECT_ARRHAS(keys, "obj1");
	EXPECT_ARRHAS(keys, "obj3");

	marsh::Maps dump;
	marsh::get_attrs(dump, big_root);

	auto numar_dump = dump.get_attr("obj1");
	auto numf_dump = dump.get_attr("obj2");
	auto objtup_dump = dump.get_attr("obj3");
	ASSERT_NE(nullptr, numar_dump);
	ASSERT_EQ(nullptr, numf_dump);
	ASSERT_NE(nullptr, objtup_dump);
	EXPECT_EQ(typeid(marsh::NumArray<size_t>).hash_code(), numar_dump->class_code());
	EXPECT_EQ(typeid(marsh::ObjTuple).hash_code(), objtup_dump->class_code());

	delete r;
	delete c;
	delete big;
	delete imperfect;
}


TEST(OBJS, String)
{
	marsh::String empty;
	marsh::String content("stuff happening");

	ASSERT_EQ(typeid(marsh::String).hash_code(), empty.class_code());
	ASSERT_EQ(typeid(marsh::String).hash_code(), content.class_code());

	EXPECT_FALSE(empty.equals(content));
	EXPECT_FALSE(content.equals(empty));

	marsh::Number<size_t> notstr;
	marsh::String es;
	marsh::String exact("stuff happening");
	marsh::String notexact("stuf hppenig");

	EXPECT_TRUE(es.equals(empty));
	EXPECT_TRUE(exact.equals(content));
	EXPECT_TRUE(empty.equals(es));
	EXPECT_TRUE(content.equals(exact));

	EXPECT_FALSE(empty.equals(notstr));
	EXPECT_FALSE(empty.equals(content));
	EXPECT_FALSE(content.equals(empty));
	EXPECT_FALSE(notexact.equals(content));
	EXPECT_FALSE(content.equals(notexact));

	EXPECT_STREQ("stuff happening", content.to_string().c_str());

	marsh::JsonMarshaler parser;
	std::string parsed_root = parser.parse(content, false);
	fmts::trim(parsed_root);
	EXPECT_STREQ("", parser.parse(empty, false).c_str());
	EXPECT_STREQ("stuff happening", parsed_root.c_str());

	auto e = empty.clone();
	auto c = content.clone();

	ASSERT_TRUE(e->equals(empty));
	ASSERT_TRUE(c->equals(content));

	delete e;
	delete c;
}


#endif // DISABLE_MARSH_OBJS_TEST
