
#ifndef DISABLE_OPT_APPLY_TEST

#include <sstream>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/opt/mock/mock.hpp"


using ::testing::_;
using ::testing::Const;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::ReturnRef;


TEST(APPLY, Optimize)
{
	teq::Shape in_shape({2,3});
	teq::Shape w0_shape({4,2});
	teq::Shape b0_shape({4});
	teq::Shape w1_shape({3,4});
	teq::Shape b1_shape({3});
	teq::Shape out_shape({3,3});

	double s1 = 1, s2 = 2, s3 = 3;
	std::vector<double> indata{2, 8, 4, 5, 2, 1};
	std::vector<double> w0data{3, 7, 5, 8, 1, 1, 0, 9};
	std::vector<double> b0data{2, 1, 8, 4};
	std::vector<double> w1data{3, 7, 2, 7, 1, 5, 3, 7, 2, 7, 1, 5};
	std::vector<double> b1data{2, 8, 4};
	std::vector<double> outdata{3, 7, 7, 5, 1, 5};

	MockDeviceRef mockdev;
	MockDeviceRef mockdev2;
	MockDeviceRef mockdev3;
	MockDeviceRef mockdev4;
	MockDeviceRef mockdev5;
	MockDeviceRef mockdev6;
	MockDeviceRef mockdev7;
	MockDeviceRef mockdev8;
	MockDeviceRef mockdev9;

	auto one = make_cst<double>(s1, mockdev);
	auto two = make_cst<double>(s2, mockdev2);
	auto three = make_cst<double>(s3, mockdev3);
	auto in = make_var<double>(indata.data(), mockdev4, in_shape);
	auto w0 = make_var<double>(w0data.data(), mockdev5, w0_shape);
	auto b0 = make_var<double>(b0data.data(), mockdev6, b0_shape);
	auto w1 = make_var<double>(w1data.data(), mockdev7, w1_shape);
	auto b1 = make_var<double>(b1data.data(), mockdev8, b1_shape);
	auto out = make_var<double>(outdata.data(), mockdev9, out_shape);

	auto m0 = make_fnc("MATMUL", 5, teq::TensptrsT{in, w0});
	auto eb0 = make_fnc("EXTEND", 2, teq::TensptrsT{b0});
	auto layer0 = make_fnc("ADD", 4, teq::TensptrsT{m0, eb0});
	auto neg0 = make_fnc("NEG", 3, teq::TensptrsT{layer0});
	auto exp0 = make_fnc("EXP", 1, teq::TensptrsT{neg0});
	auto denom0 = make_fnc("ADD", 4, teq::TensptrsT{one, exp0});
	auto sig0 = make_fnc("DIV", 0, teq::TensptrsT{one, denom0});

	auto replace1 = make_fnc("SIGMOID", 0, teq::TensptrsT{layer0});

	auto m1 = make_fnc("MATMUL", 5, teq::TensptrsT{sig0, w1});
	auto eb1 = make_fnc("EXTEND", 2, teq::TensptrsT{b1});
	auto layer1 = make_fnc("ADD", 4, teq::TensptrsT{m1, eb1});
	auto neg1 = make_fnc("NEG", 3, teq::TensptrsT{layer1});
	auto exp1 = make_fnc("EXP", 1, teq::TensptrsT{neg1});
	auto denom1 = make_fnc("ADD", 4, teq::TensptrsT{one, exp1});
	auto sig1 = make_fnc("DIV", 0, teq::TensptrsT{one, denom1});

	auto replace2 = make_fnc("SIGMOID", 0, teq::TensptrsT{layer1});

	auto sub = make_fnc("SUB", 6, teq::TensptrsT{out, sig1});
	auto err = make_fnc("POW", 7, teq::TensptrsT{sub, two});

	EXPECT_CALL(*m1, update_child(teq::TensptrT(replace1), 0)).Times(1);
	EXPECT_CALL(*sub, update_child(teq::TensptrT(replace2), 1)).Times(1);

	opt::GraphInfo graph(teq::TensptrsT{err});

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
	auto mtarg = std::make_shared<MockTarget>();
	rule.target_ = mtarg;

	query::SymbMapT cand0;
	query::SymbMapT cand1;
	EXPECT_CALL(*mtarg, convert(_)).Times(2).
		WillOnce(Invoke(
		[&](const query::SymbMapT& candidates) -> teq::TensptrT
		{
			cand1 = candidates;
			return replace2;
		})).
		WillOnce(Invoke(
		[&](const query::SymbMapT& candidates) -> teq::TensptrT
		{
			cand0 = candidates;
			return replace1;
		}));

	bool success = opt::optimize(graph, rules);
	EXPECT_TRUE(success);

	ASSERT_HAS(cand0, "X");
	EXPECT_EQ(layer0.get(), cand0.at("X"));
	ASSERT_HAS(cand1, "X");
	EXPECT_EQ(layer1.get(), cand1.at("X"));

	auto roots = graph.get_roots();
	ASSERT_EQ(1, roots.size());
}


#endif // DISABLE_OPT_APPLY_TEST
