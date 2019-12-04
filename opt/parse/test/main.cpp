#include "gtest/gtest.h"

#include "exam/exam.hpp"

extern "C" {
#include "opt/parse/def.h"
}


static const char* sample_cfg = "cfg/optimizations.rules";


static std::vector<::Conversion*> to_conversions (::PtrList& arr)
{
	std::vector<::Conversion*> outs;
	for (auto it = arr.head_; nullptr != it; it = it->next_)
	{
		outs.push_back((::Conversion*) it->val_);
	}
	return outs;
}


static std::vector<::Arg*> to_arguments (::PtrList& arr)
{
	std::vector<::Arg*> outs;
	for (auto it = arr.head_; nullptr != it; it = it->next_)
	{
		outs.push_back((::Arg*) it->val_);
	}
	return outs;
}


static std::vector<::KeyVal*> to_attrs (::PtrList& arr)
{
	std::vector<::KeyVal*> outs;
	for (auto it = arr.head_; nullptr != it; it = it->next_)
	{
		outs.push_back((::KeyVal*) it->val_);
	}
	return outs;
}


static std::vector<double> to_nums (::NumList& lst)
{
	std::vector<double> outs;
	for (auto it = lst.head_; nullptr != it; it = it->next_)
	{
		outs.push_back(it->val_);
	}
	return outs;
}


static void test_edge (std::vector<::KeyVal*>& out,
	::Conversion* converter, size_t id)
{
	ASSERT_NE(nullptr, converter);

	auto matcher = converter->matcher_;
	auto target = converter->target_;
	ASSERT_NE(nullptr, matcher);
	ASSERT_NE(nullptr, target);
	ASSERT_EQ(::TreeNode::SCALAR, target->type_);

	EXPECT_EQ(id, target->val_.scalar_);

	EXPECT_STREQ("F", matcher->name_);
	EXPECT_STREQ("", matcher->variadic_);
	EXPECT_EQ(FALSE, matcher->commutative_);

	ASSERT_EQ(::ARGUMENT, matcher->args_.type_);
	auto args = to_arguments(matcher->args_);
	ASSERT_EQ(1, args.size());
	auto arg = args[0];

	ASSERT_EQ(::TreeNode::ANY, arg->node_->type_);
	EXPECT_STREQ("X", arg->node_->val_.any_);

	ASSERT_EQ(::KV_PAIR, arg->attrs_.type_);
	out = to_attrs(arg->attrs_);
}


TEST(PARSE, Fails)
{
	const char* single_graph = "F(X);";
	const char* empty_matcher = "=>F(X);";
	const char* empty_target = "F(X)=>;";
	const char* long_symb = "F(X)=>reallylongsymbolreallylongsymbol;";
	const char* comm_target = "F(X)=>comm F(X);";
	const char* invalid_func = "F()=>X;";
	const char* invalid_comm = "comm X=>X;";
	const char* scalar_matcher = "2=>F(X);";
	const char* symbol_matcher = "C=>G(X);";

	::PtrList* arr = nullptr;
	EXPECT_EQ(1, ::parse_str(&arr, single_graph));
	ASSERT_EQ(nullptr, arr);

	::PtrList* arr2 = nullptr;
	EXPECT_EQ(1, ::parse_str(&arr2, empty_matcher));
	ASSERT_EQ(nullptr, arr2);

	::PtrList* arr3 = nullptr;
	EXPECT_EQ(1, ::parse_str(&arr3, empty_target));
	ASSERT_EQ(nullptr, arr3);

	::PtrList* arr4 = nullptr;
	EXPECT_EQ(1, ::parse_str(&arr4, long_symb));
	ASSERT_EQ(nullptr, arr4);

	::PtrList* arr5 = nullptr;
	EXPECT_EQ(1, ::parse_str(&arr5, comm_target));
	ASSERT_EQ(nullptr, arr5);

	::PtrList* arr6 = nullptr;
	EXPECT_EQ(1, ::parse_str(&arr6, invalid_func));
	ASSERT_EQ(nullptr, arr6);

	::PtrList* arr7 = nullptr;
	EXPECT_EQ(1, ::parse_str(&arr7, invalid_comm));
	ASSERT_EQ(nullptr, arr7);

	::PtrList* arr8 = nullptr;
	EXPECT_EQ(1, ::parse_str(&arr8, scalar_matcher));
	ASSERT_EQ(nullptr, arr8);

	::PtrList* arr9 = nullptr;
	EXPECT_EQ(1, ::parse_str(&arr9, symbol_matcher));
	ASSERT_EQ(nullptr, arr9);
}


TEST(PARSE, Basics)
{
	const char* rules =
		"F(X,Y)=>0;\n"
		"F(X)=>X;\n"
		"F(X)=>G(X);\n";

	::PtrList* arr = nullptr;
	ASSERT_EQ(0, ::parse_str(&arr, rules));
	ASSERT_NE(nullptr, arr);
	ASSERT_EQ(::CONVERSION, arr->type_);
	auto conversions = to_conversions(*arr);
	ASSERT_EQ(3, conversions.size());

	// todo: check
	::cversions_free(arr);
}


TEST(PARSE, Commutative)
{
	const char* rules = "comm F(X,Y)=>0;";

	::PtrList* arr = nullptr;
	ASSERT_EQ(0, ::parse_str(&arr, rules));
	ASSERT_NE(nullptr, arr);
	ASSERT_EQ(::CONVERSION, arr->type_);
	auto conversions = to_conversions(*arr);
	ASSERT_EQ(1, conversions.size());

	// todo: check
	::cversions_free(arr);
}


TEST(PARSE, Variadic)
{
	const char* rules =
		"F(X,..Y)=>0;\n"
		"F(X)=>G(..X);";

	::PtrList* arr = nullptr;
	ASSERT_EQ(0, ::parse_str(&arr, rules));
	ASSERT_NE(nullptr, arr);
	ASSERT_EQ(::CONVERSION, arr->type_);
	auto conversions = to_conversions(*arr);
	ASSERT_EQ(2, conversions.size());

	// todo: check
	::cversions_free(arr);
}


TEST(PARSE, EdgeAttrs)
{
	const char* rules =
		"F(X={a:[4,5,6,7,8,9,10,11]})=>0;\n"
		"F(X={b:2})=>1;\n"
		"F(X={c:[8],d:12})=>2;";

	::PtrList* arr = nullptr;
	ASSERT_EQ(0, ::parse_str(&arr, rules));
	ASSERT_NE(nullptr, arr);
	ASSERT_EQ(::CONVERSION, arr->type_);
	auto conversions = to_conversions(*arr);
	ASSERT_EQ(3, conversions.size());

	std::vector<::KeyVal*> kps;
	test_edge(kps, conversions[0], 0);
	{
		ASSERT_EQ(1, kps.size());
		auto a = kps[0];

		EXPECT_STREQ("a", a->key_);
		EXPECT_EQ(FALSE, a->val_scalar_);
		std::vector<double> expect = {4,5,6,7,8,9,10,11};
		auto nums = to_nums(a->val_);
		EXPECT_VECEQ(expect, nums);
	}
	kps.clear();

	test_edge(kps, conversions[1], 1);
	{
		ASSERT_EQ(1, kps.size());
		auto s = kps[0];

		EXPECT_STREQ("b", s->key_);
		EXPECT_EQ(TRUE, s->val_scalar_);
		std::vector<double> expect = {2};
		auto nums = to_nums(s->val_);
		EXPECT_VECEQ(expect, nums);
	}
	kps.clear();

	test_edge(kps, conversions[2], 2);
	{
		ASSERT_EQ(2, kps.size());

		auto a = kps[0];
		EXPECT_STREQ("c", a->key_);
		EXPECT_EQ(FALSE, a->val_scalar_);
		std::vector<double> expect = {8};
		auto nums = to_nums(a->val_);
		EXPECT_VECEQ(expect, nums);

		auto s = kps[1];
		EXPECT_STREQ("d", s->key_);
		EXPECT_EQ(TRUE, s->val_scalar_);
		expect = {12};
		nums = to_nums(s->val_);
		EXPECT_VECEQ(expect, nums);
	}

	::cversions_free(arr);
}


TEST(PARSE, OptimizationRules)
{
	FILE* file = std::fopen(sample_cfg, "r");
	ASSERT_NE(nullptr, file);

	::PtrList* arr = nullptr;
	int status = ::parse_file(&arr, file);
	EXPECT_EQ(0, status);

	ASSERT_NE(nullptr, arr);
	EXPECT_EQ(::CONVERSION, arr->type_);

	::cversions_free(arr);
}


int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
