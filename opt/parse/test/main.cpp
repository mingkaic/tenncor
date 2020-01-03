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


static std::vector<::TreeNode*> to_arguments (::PtrList& arr)
{
	std::vector<::TreeNode*> outs;
	for (auto it = arr.head_; nullptr != it; it = it->next_)
	{
		outs.push_back((::TreeNode*) it->val_);
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
	out = to_attrs(matcher->attrs_);

	ASSERT_EQ(::ARGUMENT, matcher->args_.type_);
	auto args = to_arguments(matcher->args_);
	ASSERT_EQ(1, args.size());
	auto arg = args[0];

	ASSERT_EQ(::TreeNode::ANY, arg->type_);
	EXPECT_STREQ("X", arg->val_.any_);
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
	const char* bad_token = "F(C)=>~;";

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

	::PtrList* arr10 = nullptr;
	EXPECT_EQ(1, ::parse_str(&arr10, bad_token));
	ASSERT_EQ(nullptr, arr10);
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
		"F{a:[4,5,6,7,8,9,10,11]}(X)=>0;\n"
		"F{b:2}(X)=>1;\n"
		"F{c:[8],d:12}(X)=>2;";

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
	ASSERT_EQ(0, ::parse_file(&arr, file));

	ASSERT_NE(nullptr, arr);
	EXPECT_EQ(::CONVERSION, arr->type_);

	::cversions_free(arr);
}


TEST(LIST, NumListMove)
{
	::NumList* lst = ::new_numlist();
	ASSERT_EQ(NULL, lst->head_);
	ASSERT_EQ(NULL, lst->tail_);
	::numlist_pushback(lst, 4.);
	::numlist_pushback(lst, 5.);
	::numlist_pushback(NULL, 6.);

	::NumNode* head = lst->head_;
	::NumNode* tail = lst->tail_;
	ASSERT_EQ(NULL, tail->next_);
	ASSERT_EQ(4., head->val_);
	ASSERT_EQ(5., tail->val_);

	// failed move should not change lst
	::numlist_move(NULL, lst);
	ASSERT_EQ(head, lst->head_);
	ASSERT_EQ(tail, lst->tail_);
	// ensure the list structure is not changed
	ASSERT_EQ(tail, head->next_);
	ASSERT_EQ(NULL, tail->next_);

	// failed move should not change lst
	::numlist_move(lst, NULL);
	ASSERT_EQ(head, lst->head_);
	ASSERT_EQ(tail, lst->tail_);
	// ensure the list structure is not changed
	ASSERT_EQ(tail, head->next_);
	ASSERT_EQ(NULL, tail->next_);

	::NumList* lst2 = ::new_numlist();
	ASSERT_EQ(NULL, lst2->head_);
	ASSERT_EQ(NULL, lst2->tail_);

	// move lst to lst2
	::numlist_move(lst2, lst);
	ASSERT_EQ(head, lst2->head_);
	ASSERT_EQ(tail, lst2->tail_);
	// ensure the list structure is not changed after move
	ASSERT_EQ(tail, head->next_);
	ASSERT_EQ(NULL, tail->next_);
	// ensure lst is cleared after moving
	ASSERT_EQ(NULL, lst->head_);
	ASSERT_EQ(NULL, lst->tail_);

	::numlist_free(lst);
	::numlist_free(lst2);
}


TEST(LIST, NumListClear)
{
	::NumList* lst = ::new_numlist();
	ASSERT_EQ(NULL, lst->head_);
	ASSERT_EQ(NULL, lst->tail_);
	::numlist_pushback(lst, 4.);
	::numlist_pushback(lst, 5.);
	::numlist_pushback(NULL, 6.);

	ASSERT_EQ(lst->tail_, lst->head_->next_);
	ASSERT_EQ(NULL, lst->tail_->next_);
	ASSERT_EQ(4., lst->head_->val_);
	ASSERT_EQ(5., lst->tail_->val_);

	::numlist_clear(NULL);
	::numlist_clear(lst);

	ASSERT_EQ(NULL, lst->head_);
	ASSERT_EQ(NULL, lst->tail_);

	::numlist_free(lst);
	::numlist_free(NULL);
}


TEST(LIST, PtrListMove)
{
	int a = 3, b = 5;
	void* obj = &a;
	void* obj2 = &b;

	::PtrList* lst = ::new_ptrlist(4);
	ASSERT_EQ(4, lst->type_);
	ASSERT_EQ(NULL, lst->head_);
	ASSERT_EQ(NULL, lst->tail_);
	::ptrlist_pushback(lst, obj);
	::ptrlist_pushback(lst, obj2);
	::ptrlist_pushback(NULL, obj);

	::PtrNode* head = lst->head_;
	::PtrNode* tail = lst->tail_;
	ASSERT_EQ(NULL, tail->next_);
	ASSERT_EQ(3, *((int*) head->val_));
	ASSERT_EQ(5, *((int*) tail->val_));

	// failed move should not change lst
	::ptrlist_move(NULL, lst);
	ASSERT_EQ(head, lst->head_);
	ASSERT_EQ(tail, lst->tail_);
	// ensure the list structure is not changed
	ASSERT_EQ(tail, head->next_);
	ASSERT_EQ(NULL, tail->next_);

	// failed move should not change lst
	::ptrlist_move(lst, NULL);
	ASSERT_EQ(head, lst->head_);
	ASSERT_EQ(tail, lst->tail_);
	// ensure the list structure is not changed
	ASSERT_EQ(tail, head->next_);
	ASSERT_EQ(NULL, tail->next_);

	::PtrList* lst2 = ::new_ptrlist(2);
	ASSERT_EQ(2, lst2->type_);
	ASSERT_EQ(NULL, lst2->head_);
	ASSERT_EQ(NULL, lst2->tail_);

	// move lst to lst2
	::ptrlist_move(lst2, lst);
	ASSERT_EQ(4, lst2->type_);
	ASSERT_EQ(head, lst2->head_);
	ASSERT_EQ(tail, lst2->tail_);
	// ensure the list structure is not changed after move
	ASSERT_EQ(tail, head->next_);
	ASSERT_EQ(NULL, tail->next_);
	// ensure lst is cleared after moving
	ASSERT_EQ(NULL, lst->head_);
	ASSERT_EQ(NULL, lst->tail_);

	::ptrlist_free(lst, NULL);
	::ptrlist_free(lst2, NULL);
}


TEST(LIST, PtrListClear)
{
	int a = 3, b = 5;
	int* obj = new int;
	int* obj2 = new int;
	*obj = a;
	*obj2 = b;

	void (*delobjs)(void*) = [](void* ptr)
	{
		auto i = (int*) ptr;
		delete i;
	};

	::PtrList* lst = ::new_ptrlist(4);
	{
		ASSERT_EQ(4, lst->type_);
		ASSERT_EQ(NULL, lst->head_);
		ASSERT_EQ(NULL, lst->tail_);
		::ptrlist_pushback(lst, obj);
		::ptrlist_pushback(lst, obj2);
		::ptrlist_pushback(NULL, obj);

		::PtrNode* head = lst->head_;
		::PtrNode* tail = lst->tail_;
		ASSERT_EQ(NULL, tail->next_);
		ASSERT_EQ(3, *((int*) head->val_));
		ASSERT_EQ(5, *((int*) tail->val_));

		::ptrlist_clear(NULL, NULL);
		::ptrlist_clear(lst, NULL);

		ASSERT_EQ(NULL, lst->head_);
		ASSERT_EQ(NULL, lst->tail_);
	}
	{
		::ptrlist_pushback(lst, obj);
		::ptrlist_pushback(lst, obj2);

		::ptrlist_clear(lst, delobjs);

		ASSERT_EQ(NULL, lst->head_);
		ASSERT_EQ(NULL, lst->tail_);
	}

	::ptrlist_free(lst, NULL);
	::ptrlist_free(NULL, NULL);
}


int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
