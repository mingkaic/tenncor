#include "gtest/gtest.h"

#include "exam/exam.hpp"

extern "C" {
#include "opt/parse/def.h"
}


static const char* sample_cfg = "cfg/optimizations.rules";


static std::vector<::Statement*> vectorize (::PtrList* stmts)
{
	assert(nullptr != stmts);
	std::vector<::Statement*> outs;
	for (auto it = stmts->head_; nullptr != it; it = it->next_)
	{
		outs.push_back((::Statement*) it->val_);
	}
	return outs;
}


static std::vector<double> vectorize (::NumList* lst)
{
	assert(nullptr != lst);
	std::vector<double> outs;
	for (auto it = lst->head_; nullptr != it; it = it->next_)
	{
		outs.push_back(it->val_);
	}
	return outs;
}


TEST(PARSE, SymbolFail)
{
	const char* symbs = "symbol Apple Banana;\nsymbol Citrus Zucchini;";

	::PtrList* stmts = nullptr;
	int status = ::parse_str(&stmts, symbs);
	EXPECT_EQ(1, status);
}


TEST(PARSE, PropFail)
{
	const char* props = "property Apple;\nproperty Citrus;";
	const char* props2 = "property Apple Banna Zucchini;\nproperty Citrus Grapefruit Lemon;";

	::PtrList* stmts = nullptr;
	int status = ::parse_str(&stmts, props);
	EXPECT_EQ(1, status);

	int status2 = ::parse_str(&stmts, props2);
	EXPECT_EQ(1, status2);
}


TEST(PARSE, SymbolDef)
{
	const char* symbs = "symbol Apple;\n"
		"symbol Banana;symbol Citrus;\n"
		"symbol Zucchini;";

	::PtrList* stmts = nullptr;
	int status = ::parse_str(&stmts, symbs);
	EXPECT_EQ(0, status);

	ASSERT_NE(nullptr, stmts);
	EXPECT_EQ(::STATEMENT, stmts->type_);

	auto vstmts = vectorize(stmts);
	ASSERT_EQ(4, vstmts.size());

	std::vector<const char*> labels;
	labels.reserve(4);
	for (::Statement* stmt : vstmts)
	{
		ASSERT_EQ(::SYMBOL_DEF, stmt->type_);
		labels.push_back((char*) stmt->val_);
	}

	EXPECT_STREQ("Apple", labels[0]);
	EXPECT_STREQ("Banana", labels[1]);
	EXPECT_STREQ("Citrus", labels[2]);
	EXPECT_STREQ("Zucchini", labels[3]);

	::statements_free(stmts);
}


TEST(PARSE, PropertyDef)
{
	const char* groups = "property Owl spooky;\n"
		"property Bat spooky;property group:Skeleton doot;\n"
		"property group:Skeleton spooky;"
		"property Casper friendly;\n";

	::PtrList* stmts = nullptr;
	int status = ::parse_str(&stmts, groups);
	EXPECT_EQ(0, status);

	ASSERT_NE(nullptr, stmts);
	EXPECT_EQ(::STATEMENT, stmts->type_);

	auto vstmts = vectorize(stmts);
	ASSERT_EQ(5, vstmts.size());

	std::vector<::Property*> props;
	props.reserve(5);
	for (::Statement* stmt : vstmts)
	{
		ASSERT_EQ(::PROPERTY_DEF, stmt->type_);
		props.push_back((::Property*) stmt->val_);
	}

	EXPECT_STREQ("Owl", props[0]->label_);
	EXPECT_STREQ("Bat", props[1]->label_);
	EXPECT_STREQ("Skeleton", props[2]->label_);
	EXPECT_STREQ("Skeleton", props[3]->label_);
	EXPECT_STREQ("Casper", props[4]->label_);

	EXPECT_STREQ("spooky", props[0]->property_);
	EXPECT_STREQ("spooky", props[1]->property_);
	EXPECT_STREQ("doot", props[2]->property_);
	EXPECT_STREQ("spooky", props[3]->property_);
	EXPECT_STREQ("friendly", props[4]->property_);

	EXPECT_FALSE(props[0]->is_group_);
	EXPECT_FALSE(props[1]->is_group_);
	EXPECT_TRUE(props[2]->is_group_);
	EXPECT_TRUE(props[3]->is_group_);
	EXPECT_FALSE(props[4]->is_group_);

	::statements_free(stmts);
}


TEST(PARSE, EdgeDef)
{
	const char* shape_edge = "F(X={shaper:[4,5,6,7,8,9,10,11]})=>1;\n";
	const char* coord_edge = "F(X={coorder:[8,8,8,8,8,8,8,8]})=>2;\n";
	const char* both_edges = "F(X={coorder:[8,8,8,8,8,8,8,8],shaper:[4,5,6,7,8,9,10,11]})=>3;\n";
	std::vector<double> expect_shaper = {4,5,6,7,8,9,10,11};
	std::vector<double> expect_coorder = {8,8,8,8,8,8,8,8};

	::PtrList* stmts = nullptr;
	ASSERT_EQ(0, ::parse_str(&stmts, shape_edge));
	EXPECT_EQ(nullptr, stmts->head_->next_);
	auto stmt = (::Statement*) stmts->head_->val_;
	ASSERT_EQ(::CONVERSION, stmt->type_);
	auto conv = (::Conversion*) stmt->val_;
	auto src = conv->source_;
	auto dest = conv->dest_;
	ASSERT_EQ(::SCALAR, dest->type_);
	EXPECT_EQ(1, dest->val_.scalar_);
	ASSERT_EQ(::BRANCH, src->type_);
	auto branch = src->val_.branch_;
	EXPECT_STREQ("F", branch->label_);
	EXPECT_EQ(nullptr, branch->args_->head_->next_);
	auto arg = (::Arg*) branch->args_->head_->val_;
	ASSERT_EQ(::ANY, arg->subgraph_->type_);
	EXPECT_STREQ("X", arg->subgraph_->val_.any_);
	ASSERT_NE(nullptr, arg->shaper_);
	auto shaper = vectorize(arg->shaper_);
	EXPECT_ARREQ(expect_shaper, shaper);
	EXPECT_EQ(nullptr, arg->coorder_);

	ASSERT_EQ(0, ::parse_str(&stmts, coord_edge));
	EXPECT_EQ(nullptr, stmts->head_->next_);
	stmt = (::Statement*) stmts->head_->val_;
	ASSERT_EQ(::CONVERSION, stmt->type_);
	conv = (::Conversion*) stmt->val_;
	src = conv->source_;
	dest = conv->dest_;
	ASSERT_EQ(::SCALAR, dest->type_);
	EXPECT_EQ(2, dest->val_.scalar_);
	ASSERT_EQ(::BRANCH, src->type_);
	branch = src->val_.branch_;
	EXPECT_STREQ("F", branch->label_);
	EXPECT_EQ(nullptr, branch->args_->head_->next_);
	arg = (::Arg*) branch->args_->head_->val_;
	ASSERT_EQ(::ANY, arg->subgraph_->type_);
	EXPECT_STREQ("X", arg->subgraph_->val_.any_);
	EXPECT_EQ(nullptr, arg->shaper_);
	ASSERT_NE(nullptr, arg->coorder_);
	auto coorder = vectorize(arg->coorder_);
	EXPECT_ARREQ(expect_coorder, coorder);

	ASSERT_EQ(0, ::parse_str(&stmts, both_edges));
	EXPECT_EQ(nullptr, stmts->head_->next_);
	stmt = (::Statement*) stmts->head_->val_;
	ASSERT_EQ(::CONVERSION, stmt->type_);
	conv = (::Conversion*) stmt->val_;
	src = conv->source_;
	dest = conv->dest_;
	ASSERT_EQ(::SCALAR, dest->type_);
	EXPECT_EQ(3, dest->val_.scalar_);
	ASSERT_EQ(::BRANCH, src->type_);
	branch = src->val_.branch_;
	EXPECT_STREQ("F", branch->label_);
	EXPECT_EQ(nullptr, branch->args_->head_->next_);
	arg = (::Arg*) branch->args_->head_->val_;
	ASSERT_EQ(::ANY, arg->subgraph_->type_);
	EXPECT_STREQ("X", arg->subgraph_->val_.any_);
	ASSERT_NE(nullptr, arg->shaper_);
	ASSERT_NE(nullptr, arg->coorder_);
	shaper = vectorize(arg->shaper_);
	coorder = vectorize(arg->coorder_);
	EXPECT_ARREQ(expect_shaper, shaper);
	EXPECT_ARREQ(expect_coorder, coorder);
}


TEST(PARSE, OptimizationRules)
{
	FILE* file = std::fopen(sample_cfg, "r");
	assert(nullptr != file);

	::PtrList* stmts = nullptr;
	int status = ::parse_file(&stmts, file);
	EXPECT_EQ(0, status);

	ASSERT_NE(nullptr, stmts);
	EXPECT_EQ(::STATEMENT, stmts->type_);

	::statements_free(stmts);
}


int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
