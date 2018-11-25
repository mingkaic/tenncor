
#ifndef DISABLE_GRADER_TEST


#include "gtest/gtest.h"

#include "age/test/grader_dep.hpp"
#include "age/generated/grader.hpp"


TEST(AGE, RulesData)
{
	age::RuleSet rule;
	ade::LeafptrT data = rule.data(45, ade::Shape({4, 6}));
	MockTensor* mockdata = dynamic_cast<MockTensor*>(data.get());
	ASSERT_NE(nullptr, mockdata);

	ade::Shape shape = mockdata->shape();
	EXPECT_EQ(45, mockdata->scalar_);
	EXPECT_EQ(4, shape.at(0));
	EXPECT_EQ(6, shape.at(1));
	EXPECT_EQ(24, shape.n_elems());
}


TEST(AGE, RulesSum)
{
	age::RuleSet rule;
	auto code = rule.sum_opcode();
	EXPECT_STREQ("EMINEM", code.name_.c_str());
	EXPECT_EQ(age::EMINEM, code.code_);
}


TEST(AGE, RulesProd)
{
	age::RuleSet rule;
	auto code = rule.prod_opcode();
	EXPECT_STREQ("KHALED", code.name_.c_str());
	EXPECT_EQ(age::KHALED, code.code_);
}


TEST(AGE, GraderEminem)
{
	age::RuleSet rule;
	auto mock = new MockTensor(1, ade::Shape());
	ade::TensptrT arg(mock);
	size_t idx = 42;
	rule.grad_rule(age::EMINEM, {arg}, idx);
	EXPECT_EQ(idx, mock->scalar_);
}


TEST(AGE, GraderKhaled)
{
	age::RuleSet rule;
	auto mock = new MockTensor(1, ade::Shape());
	ade::TensptrT arg(mock);
	size_t idx = 63;
	rule.grad_rule(age::KHALED, {arg}, idx);
	EXPECT_EQ(idx + khaled_constant, mock->scalar_);
}


#endif // DISABLE_GRADER_TEST
