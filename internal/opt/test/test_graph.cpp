
#ifndef DISABLE_OPT_GRAPH_TEST

#include <sstream>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/opt/mock/mock.hpp"


TEST(GRAPH, GetInfo)
{
	teq::Shape in_shape({2,3});
	teq::Shape w0_shape({4,2});
	teq::Shape b0_shape({4});
	teq::Shape w1_shape({3,4});
	teq::Shape b1_shape({3});
	teq::Shape out_shape({3,3});

	auto one = std::make_shared<MockLeaf>(std::vector<double>{1}, teq::Shape());
	auto two = std::make_shared<MockLeaf>(std::vector<double>{2}, teq::Shape());
	auto three = std::make_shared<MockLeaf>(std::vector<double>{3}, teq::Shape());
	auto in = std::make_shared<MockLeaf>(std::vector<double>{2, 8, 4, 5, 2, 1}, in_shape);
	auto w0 = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 5, 8, 1, 1, 0, 9}, w0_shape);
	auto b0 = std::make_shared<MockLeaf>(std::vector<double>{2, 1, 8, 4}, b0_shape);
	auto w1 = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 2, 7, 1, 5, 3, 7, 2, 7, 1, 5}, w1_shape);
	auto b1 = std::make_shared<MockLeaf>(std::vector<double>{2, 8, 4}, b1_shape);
	auto out = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 7, 5, 1, 5}, out_shape);

	auto m0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{in, w0},
		teq::Opcode{"MATMUL", 5});
	auto eb0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{b0}, teq::Opcode{"EXTEND", 2});
	auto layer0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{m0, eb0},
		teq::Opcode{"ADD", 4});
	auto neg0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{layer0}, teq::Opcode{"NEG", 3});
	auto exp0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{neg0}, teq::Opcode{"EXP", 1});
	auto denom0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, exp0}, teq::Opcode{"ADD", 4});
	auto sig0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, denom0}, teq::Opcode{"DIV", 0});

	auto m1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{sig0, w1},
		teq::Opcode{"MATMUL", 5});
	auto eb1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{b1}, teq::Opcode{"EXTEND", 2});
	auto layer1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{m1, eb1},
		teq::Opcode{"ADD", 4});
	auto neg1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{layer1}, teq::Opcode{"NEG", 3});
	auto exp1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{neg1}, teq::Opcode{"EXP", 1});
	auto denom1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, exp1}, teq::Opcode{"ADD", 4});
	auto sig1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, denom1}, teq::Opcode{"DIV", 0});

	auto sub = std::make_shared<MockFunctor>(
		teq::TensptrsT{out, sig1}, teq::Opcode{"SUB", 6});
	auto err = std::make_shared<MockFunctor>(
		teq::TensptrsT{sub, two}, teq::Opcode{"POW", 7});

	opt::GraphInfo graph({err});

	EXPECT_EQ(nullptr, graph.get_owner(three.get()));
	EXPECT_EQ(sig1, graph.get_owner(sig1.get()));

	auto roots = graph.get_roots();
	ASSERT_EQ(1, roots.size());
	EXPECT_EQ(err, roots.front());

	auto owners = graph.get_owners();

	EXPECT_EQ(24, owners.size());
	EXPECT_HAS(owners, err.get());
	EXPECT_HAS(owners, sub.get());
	EXPECT_HAS(owners, sig1.get());
	EXPECT_HAS(owners, sig0.get());
	EXPECT_EQ(err, owners.at(err.get()));
	EXPECT_EQ(sub, owners.at(sub.get()));
	EXPECT_EQ(sig1, owners.at(sig1.get()));
	EXPECT_EQ(sig0, owners.at(sig0.get()));
}


TEST(Graph, Find)
{
	teq::Shape in_shape({2,3});
	teq::Shape w0_shape({4,2});
	teq::Shape b0_shape({4});
	teq::Shape w1_shape({3,4});
	teq::Shape b1_shape({3});
	teq::Shape out_shape({3,3});

	auto one = std::make_shared<MockLeaf>(std::vector<double>{1}, teq::Shape(), "1", true);
	auto two = std::make_shared<MockLeaf>(std::vector<double>{2}, teq::Shape(), "2", true);
	auto in = std::make_shared<MockLeaf>(std::vector<double>{2, 8, 4, 5, 2, 1}, in_shape);
	auto w0 = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 5, 8, 1, 1, 0, 9}, w0_shape);
	auto b0 = std::make_shared<MockLeaf>(std::vector<double>{2, 1, 8, 4}, b0_shape);
	auto w1 = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 2, 7, 1, 5, 3, 7, 2, 7, 1, 5}, w1_shape);
	auto b1 = std::make_shared<MockLeaf>(std::vector<double>{2, 8, 4}, b1_shape);
	auto out = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 7, 5, 1, 5}, out_shape);

	auto m0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{in, w0},
		teq::Opcode{"MATMUL", 5});
	auto eb0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{b0}, teq::Opcode{"EXTEND", 2});
	auto layer0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{m0, eb0},
		teq::Opcode{"ADD", 4});
	auto neg0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{layer0}, teq::Opcode{"NEG", 3});
	auto exp0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{neg0}, teq::Opcode{"EXP", 1});
	auto denom0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, exp0}, teq::Opcode{"ADD", 4});
	auto sig0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, denom0}, teq::Opcode{"DIV", 0});

	auto m1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{sig0, w1},
		teq::Opcode{"MATMUL", 5});
	auto eb1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{b1}, teq::Opcode{"EXTEND", 2});
	auto layer1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{m1, eb1},
		teq::Opcode{"ADD", 4});
	auto neg1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{layer1}, teq::Opcode{"NEG", 3});
	auto exp1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{neg1}, teq::Opcode{"EXP", 1});
	auto denom1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, exp1}, teq::Opcode{"ADD", 4});
	auto sig1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, denom1}, teq::Opcode{"DIV", 0});

	auto sub = std::make_shared<MockFunctor>(
		teq::TensptrsT{out, sig1}, teq::Opcode{"SUB", 6});
	auto err = std::make_shared<MockFunctor>(
		teq::TensptrsT{sub, two}, teq::Opcode{"POW", 7});

	opt::GraphInfo graph({err});

	std::stringstream sigmoid_pattern;
	sigmoid_pattern <<
	"{"
		"\"op\":{"
			"\"opname\":\"DIV\","
			"\"args\":[{"
				"\"cst\":1"
			"},{"
				"\"op\":{"
					"\"opname\":\"ADD\","
					"\"args\":[{"
						"\"cst\":1"
					"},{"
						"\"op\":{"
							"\"opname\":\"EXP\","
							"\"args\":[{"
								"\"op\":{"
									"\"opname\":\"NEG\","
									"\"args\":[{"
										"\"symb\":\"X\""
									"}]"
								"}"
							"}]"
						"}"
					"}]"
				"}"
			"}]"
		"}"
	"}";
	query::Node sigmoid_cond;
	query::json_parse(sigmoid_cond, sigmoid_pattern);

	auto sigs = graph.find(sigmoid_cond);
	ASSERT_EQ(2, sigs.size());
	EXPECT_ARRHAS(sigs, sig0);
	EXPECT_ARRHAS(sigs, sig1);
}


TEST(Graph, Replace)
{
	teq::Shape in_shape({2,3});
	teq::Shape w0_shape({4,2});
	teq::Shape b0_shape({4});
	teq::Shape w1_shape({3,4});
	teq::Shape b1_shape({3});
	teq::Shape out_shape({3,3});

	auto one = std::make_shared<MockLeaf>(std::vector<double>{1}, teq::Shape(), "1", true);
	auto two = std::make_shared<MockLeaf>(std::vector<double>{2}, teq::Shape(), "2", true);
	auto in = std::make_shared<MockLeaf>(std::vector<double>{2, 8, 4, 5, 2, 1}, in_shape);
	auto w0 = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 5, 8, 1, 1, 0, 9}, w0_shape);
	auto b0 = std::make_shared<MockLeaf>(std::vector<double>{2, 1, 8, 4}, b0_shape);
	auto w1 = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 2, 7, 1, 5, 3, 7, 2, 7, 1, 5}, w1_shape);
	auto b1 = std::make_shared<MockLeaf>(std::vector<double>{2, 8, 4}, b1_shape);
	auto out = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 7, 5, 1, 5}, out_shape);

	auto m0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{in, w0},
		teq::Opcode{"MATMUL", 5});
	auto eb0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{b0}, teq::Opcode{"EXTEND", 2});
	auto layer0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{m0, eb0},
		teq::Opcode{"ADD", 4});
	auto neg0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{layer0}, teq::Opcode{"NEG", 3});
	auto exp0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{neg0}, teq::Opcode{"EXP", 1});
	auto denom0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, exp0}, teq::Opcode{"ADD", 4});
	auto sig0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, denom0}, teq::Opcode{"DIV", 0});

	auto s0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{layer0}, teq::Opcode{"SIGMOID", 10});

	auto m1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{sig0, w1},
		teq::Opcode{"MATMUL", 5});
	auto eb1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{b1}, teq::Opcode{"EXTEND", 2});
	auto layer1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{m1, eb1},
		teq::Opcode{"ADD", 4});
	auto neg1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{layer1}, teq::Opcode{"NEG", 3});
	auto exp1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{neg1}, teq::Opcode{"EXP", 1});
	auto denom1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, exp1}, teq::Opcode{"ADD", 4});
	auto sig1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{one, denom1}, teq::Opcode{"DIV", 0});

	auto s1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{layer1}, teq::Opcode{"SIGMOID", 10});

	auto sub = std::make_shared<MockFunctor>(
		teq::TensptrsT{out, sig1}, teq::Opcode{"SUB", 6});
	auto err = std::make_shared<MockFunctor>(
		teq::TensptrsT{sub, two}, teq::Opcode{"POW", 7});

	opt::GraphInfo graph({err, s0});
	graph.replace({
		{sig0.get(), s0},
		{sig1.get(), s1},
	});

	auto roots = graph.get_roots();
	ASSERT_EQ(2, roots.size());
	EXPECT_EQ(err, roots.front());
	EXPECT_EQ(s0, roots.back());

	EXPECT_GRAPHEQ(
		"(POW<no_type>[3\\3\\1\\1\\1\\1\\1\\1])\n"
		"_`--(SUB<no_type>[3\\3\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:<no_type>[3\\3\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(SIGMOID<no_type>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(ADD<no_type>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___________`--(MATMUL<no_type>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___________|___`--(SIGMOID<no_type>[2\\3\\1\\1\\1\\1\\1\\1])\n"
		"_|___________|___|___`--(ADD<no_type>[2\\3\\1\\1\\1\\1\\1\\1])\n"
		"_|___________|___|_______`--(MATMUL<no_type>[2\\3\\1\\1\\1\\1\\1\\1])\n"
		"_|___________|___|_______|___`--(constant:<no_type>[2\\3\\1\\1\\1\\1\\1\\1])\n"
		"_|___________|___|_______|___`--(constant:<no_type>[4\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___________|___|_______`--(EXTEND<no_type>[4\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___________|___|___________`--(constant:<no_type>[4\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___________|___`--(constant:<no_type>[3\\4\\1\\1\\1\\1\\1\\1])\n"
		"_|___________`--(EXTEND<no_type>[3\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|_______________`--(constant:<no_type>[3\\1\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:2<no_type>[1\\1\\1\\1\\1\\1\\1\\1])\n", roots.front());
	EXPECT_GRAPHEQ(
		"(SIGMOID<no_type>[2\\3\\1\\1\\1\\1\\1\\1])\n"
		"_`--(ADD<no_type>[2\\3\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(MATMUL<no_type>[2\\3\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(constant:<no_type>[2\\3\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(constant:<no_type>[4\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND<no_type>[4\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(constant:<no_type>[4\\1\\1\\1\\1\\1\\1\\1])\n", roots.back());
}


#endif // DISABLE_OPT_GRAPH_TEST
