
#ifndef DISABLE_PARSE_TEST


#include "gtest/gtest.h"

#include "subgraph_match/parse.hpp"


TEST(PARSE, TransformCreation)
{
	std::stringstream ss;
	ss << "// -> denotes preference of form in the direction of arrow\n"
		<< "// = denotes equivalent preference\n"
		<< "// graph pattern uses level-order tree traversal\n"
		<< "\n"
		<< "// ========== normalization rules ==========\n"
		<< "// pool constants to the left\n"
		<< "(ADD|MUL|MIN|MAX)\\(\\d*\\),,([\\w\\(\\)\\[\\]<>]+),(constant\\(\\d*\\)|"
		<< "scalar\\(\\d+(?:\\.\\d+)?\\)\\(\\d*\\)) -> $1(),,$3,$2\n"
		<< "// left > right leaning\n"
		<< "(ADD|MUL|MIN|MAX)\\(\\d*\\),,([\\w\\(\\)\\[\\]<>]+),\\1\\(\\d*\\),,(.*),?"
		<< "([\\w\\(\\)\\[\\]<>]+),([\\w\\(\\)\\[\\]<>]+) -> $1(),,$1(),$5,,$2,$4,$3 "
		<< "->// comment\n";

	auto transforms = opt::parse_lines(ss);

	ASSERT_EQ(2, transforms.size());

	ASSERT_EQ(2, transforms[0].pheight_);
	EXPECT_TRUE(std::regex_match("ADD(123),,abcde(12),scalar(2.3)(24)",
		transforms[0].pattern_));
	EXPECT_STREQ("$1(),,$3,$2", transforms[0].simplification_.c_str());
	EXPECT_TRUE(transforms[0].stop_);

	ASSERT_EQ(3, transforms[1].pheight_);
	EXPECT_TRUE(std::regex_match("ADD(123),,abcde(12),ADD(234),,variable(333),variable(444)",
		transforms[1].pattern_));
	EXPECT_STREQ("$1(),,$1(),$5,,$2,$4,$3", transforms[1].simplification_.c_str());
	EXPECT_FALSE(transforms[1].stop_);
}


TEST(PARSE, TransformWorks)
{
	std::stringstream ss;
	ss << "(ADD|MUL|MIN|MAX)\\(\\d*\\),,([\\w\\(\\)\\[\\]<>]+),(constant\\(\\d*\\)|"
		<< "scalar\\(\\d+(?:\\.\\d+)?\\)\\(\\d*\\)) -> $1(),,$3,$2\n"
		<< "(ADD|MUL|MIN|MAX)\\(\\d*\\),,([\\w\\(\\)\\[\\]<>]+),(constant\\(\\d*\\)|"
		<< "scalar\\(\\d+(?:\\.\\d+)?\\)\\(\\d*\\)) -> $1(),,$3,$2 ->\n"
		<< "(ADD|MUL|MIN|MAX)\\(\\d*\\),,([\\w\\(\\)\\[\\]<>]+),\\1\\(\\d*\\),,(.*),?"
		<< "([\\w\\(\\)\\[\\]<>]+),([\\w\\(\\)\\[\\]<>]+) -> $1(),,$1(),$5,,$2,$4,$3\n";

	auto transforms = opt::parse_lines(ss);

	ASSERT_EQ(3, transforms.size());

	// simplification by applying transformation
	opt::IdTokenMapT empty;
	{
		auto root = std::make_shared<opt::TokenNode>("ADD(123)");
		root->children_ = {
			std::make_shared<opt::TokenNode>("variable(345)"),
			std::make_shared<opt::TokenNode>("constant(567)"),
		};
		EXPECT_TRUE(transforms[0].simplify(root, empty));
		ASSERT_NE(nullptr, root);
		std::string rep = root->encode(3);
		EXPECT_STREQ("ADD(),,constant(567),variable(345)", rep.c_str());
	}

	// continue normalization after first match
	{
		auto root = std::make_shared<opt::TokenNode>("ADD(123)");
		root->children_ = {
			std::make_shared<opt::TokenNode>("variable(345)"),
			std::make_shared<opt::TokenNode>("constant(567)"),
		};
		EXPECT_FALSE(transforms[1].simplify(root, empty));
		ASSERT_NE(nullptr, root);
		std::string rep = root->encode(3);
		EXPECT_STREQ("ADD(),,constant(567),variable(345)", rep.c_str());
	}

	// no simplification since depth matrix doesn't match pattern
	{
		opt::TokenptrT orig;
		auto root = orig = std::make_shared<opt::TokenNode>("ADD(123)");
		root->children_ = {
			std::make_shared<opt::TokenNode>("variable(345)"),
			std::make_shared<opt::TokenNode>("constant(567)"),
		};
		EXPECT_FALSE(transforms[2].simplify(root, empty));
		ASSERT_EQ(orig, root);
		std::string rep = root->encode(3);
		EXPECT_STREQ("ADD(123),,variable(345),constant(567)", rep.c_str());
	}
}


#endif // DISABLE_PARSE_TEST
