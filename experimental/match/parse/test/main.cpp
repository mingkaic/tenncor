#include "gtest/gtest.h"

extern "C" {
#include "experimental/opt/parse/def.h"
}

TEST(PARSE, RuleCall)
{
    PtrList* stmts = nullptr;
    int status = parse_rule(&stmts, "experimental/opt/optimizations.rules");
    EXPECT_EQ(0, status);

    ASSERT_NE(nullptr, stmts);
    EXPECT_EQ(::STATEMENT, stmts->type_);

    statements_free(stmts);
}


int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}