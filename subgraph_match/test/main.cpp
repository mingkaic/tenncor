#include "gtest/gtest.h"

#include "subgraph_match/parse.hpp"

int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

TEST(PARSE, TransformCreation)
{
	std::stringstream ss;
	ss << "// -> denotes preference of form in the direction of arrow\n"
		<< "// = denotes equivalent preference\n"
		<< "// graph pattern uses level-order tree traversal\n"
		<< "\n"
		<< "// ========== normalization rules ==========\n"
		<< "// pool constants to the left\n"
		<< "(ADD|MUL|MIN|MAX)\\(\\d+\\),,([\\w\\(\\)]+),(constant\\(\\d+\\)|"
		<< "scalar\\(\\d+\\)) -> $1(0),,$3,$2\n"
		<< "// left > right leaning\n"
		<< "(ADD|MUL|MIN|MAX)\\(\\d+\\),,([\\w\\(\\)]+),\\1\\(\\d+\\),,(.*),?"
		<< "([\\w\\(\\)]+),([\\w\\(\\)]+) -> $1(0),,$1(0),$5,,$2,$4,$3 "
		<< "// comment\n";

	auto transforms = opt::parse_lines(ss);

	ASSERT_EQ(2, transforms.size());

	ASSERT_EQ(2, transforms[0].pheight_);
	EXPECT_TRUE(std::regex_match("ADD(123),,abcde(12),scalar(2)",
		transforms[0].pattern_));
	EXPECT_STREQ("$1(0),,$3,$2", transforms[0].simplification_.c_str());

	ASSERT_EQ(3, transforms[1].pheight_);
	EXPECT_TRUE(std::regex_match("ADD(123),,abcde(12),ADD(234),,variable(333),variable(444)",
		transforms[1].pattern_));
	EXPECT_STREQ("$1(0),,$1(0),$5,,$2,$4,$3", transforms[1].simplification_.c_str());
}

TEST(DEPTH_MATRIX, Simplify)
{
	std::stringstream ss;
	ss << "// -> denotes preference of form in the direction of arrow\n"
		<< "// = denotes equivalent preference\n"
		<< "// graph pattern uses level-order tree traversal\n"
		<< "\n"
		<< "// ========== normalization rules ==========\n"
		<< "// pool constants to the left\n"
		<< "(ADD|MUL|MIN|MAX)\\(\\d+\\),,([\\w\\(\\)]+),(constant\\(\\d+\\)|"
		<< "scalar\\(\\d+\\)) -> $1(0),,$3,$2\n"
		<< "// left > right leaning\n"
		<< "(ADD|MUL|MIN|MAX)\\(\\d+\\),,([\\w\\(\\)]+),\\1\\(\\d+\\),,(.*),?"
		<< "([\\w\\(\\)]+),([\\w\\(\\)]+) -> $1(0),,$1(0),$5,,$2,$4,$3 "
		<< "// comment\n";

	auto transforms = opt::parse_lines(ss);

	ASSERT_EQ(2, transforms.size());

	// simplification by applying transformation
	{
		opt::DepthMatrixT init = {
			{"ADD(123)"},
			{"variable(345)", "constant(567)"}
		};
		opt::DepthMatrixT simple = opt::simplify_depthmatrix(transforms, init);
		ASSERT_EQ(2, simple.size());
		ASSERT_EQ(1, simple[0].size());
		ASSERT_EQ(2, simple[1].size());
		EXPECT_STREQ("ADD(0)", simple[0][0].c_str());
		EXPECT_STREQ("constant(567)", simple[1][0].c_str());
		EXPECT_STREQ("variable(345)", simple[1][1].c_str());
	}

	// no simplification since depth matrix doesn't match pattern
	{
		opt::DepthMatrixT init = {
			{"ADD(123)"},
			{"variable(345)", "constant(567)"}
		};
		opt::TransformsT no_matches = {transforms[1]};
		opt::DepthMatrixT simple = opt::simplify_depthmatrix(no_matches, init);
		ASSERT_EQ(2, simple.size());
		ASSERT_EQ(1, simple[0].size());
		ASSERT_EQ(2, simple[1].size());
		EXPECT_STREQ("ADD(123)", simple[0][0].c_str());
		EXPECT_STREQ("variable(345)", simple[1][0].c_str());
		EXPECT_STREQ("constant(567)", simple[1][1].c_str());
	}
}
