#include "gtest/gtest.h"

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
