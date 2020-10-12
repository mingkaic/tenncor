
#ifndef DISABLE_OPT_APPLY_TEST

#include <sstream>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/opt/mock/mock.hpp"


TEST(APPLY, Optimize)
{
	teq::Shape in_shape({2,3});
	teq::Shape w0_shape({4,2});
	teq::Shape b0_shape({4});
	teq::Shape w1_shape({3,4});
	teq::Shape b1_shape({3});
	teq::Shape out_shape({3,3});

	auto one = std::make_shared<MockLeaf>(std::vector<double>{1}, teq::Shape(), "1", true);
	auto two = std::make_shared<MockLeaf>(std::vector<double>{2}, teq::Shape(), "2", true);
	auto three = std::make_shared<MockLeaf>(std::vector<double>{3}, teq::Shape(), "3", true);
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

	std::string sigmoid_pattern = "{"
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

	opt::OptRulesT rules = {opt::OptRule()};
	auto& rule = rules.front();
	auto cond = rule.match_srcs_.Add();
	std::stringstream pattern(sigmoid_pattern);
	query::json_parse(*cond, pattern);
	rule.target_ = std::make_shared<ConditionalMockTarget>(
	[&](const query::SymbMapT& candidates) -> teq::TensptrT
	{
		auto sub = estd::try_get(candidates, "X", nullptr);
		if (nullptr == sub)
		{
			return three;
		}
		return std::make_shared<MockFunctor>(
			teq::TensptrsT{graph.get_owner(sub)}, teq::Opcode{"SIGMOID", 10});
	});

	bool success = opt::optimize(graph, rules);
	EXPECT_TRUE(success);

	auto roots = graph.get_roots();
	ASSERT_EQ(1, roots.size());
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
}


#endif // DISABLE_OPT_APPLY_TEST
