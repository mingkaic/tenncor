
#ifndef DISABLE_OPT_GRAPH_TEST

#include <sstream>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/opt/mock/mock.hpp"


using ::testing::_;
using ::testing::Const;
using ::testing::Return;
using ::testing::ReturnRef;


TEST(GRAPH, GetInfo)
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

	auto m1 = make_fnc("MATMUL", 5, teq::TensptrsT{sig0, w1});
	auto eb1 = make_fnc("EXTEND", 2, teq::TensptrsT{b1});
	auto layer1 = make_fnc("ADD", 4, teq::TensptrsT{m1, eb1});
	auto neg1 = make_fnc("NEG", 3, teq::TensptrsT{layer1});
	auto exp1 = make_fnc("EXP", 1, teq::TensptrsT{neg1});
	auto denom1 = make_fnc("ADD", 4, teq::TensptrsT{one, exp1});
	auto sig1 = make_fnc("DIV", 0, teq::TensptrsT{one, denom1});

	auto sub = make_fnc("SUB", 6, teq::TensptrsT{out, sig1});
	auto err = make_fnc("POW", 7, teq::TensptrsT{sub, two});

	opt::GraphInfo graph(teq::TensptrsT{err});

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


TEST(GRAPH, Find)
{
	teq::Shape in_shape({2,3});
	teq::Shape w0_shape({4,2});
	teq::Shape b0_shape({4});
	teq::Shape w1_shape({3,4});
	teq::Shape b1_shape({3});
	teq::Shape out_shape({3,3});

	double s1 = 1, s2 = 2; 
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

	auto m1 = make_fnc("MATMUL", 5, teq::TensptrsT{sig0, w1});
	auto eb1 = make_fnc("EXTEND", 2, teq::TensptrsT{b1});
	auto layer1 = make_fnc("ADD", 4, teq::TensptrsT{m1, eb1});
	auto neg1 = make_fnc("NEG", 3, teq::TensptrsT{layer1});
	auto exp1 = make_fnc("EXP", 1, teq::TensptrsT{neg1});
	auto denom1 = make_fnc("ADD", 4, teq::TensptrsT{one, exp1});
	auto sig1 = make_fnc("DIV", 0, teq::TensptrsT{one, denom1});

	auto sub = make_fnc("SUB", 6, teq::TensptrsT{out, sig1});
	auto err = make_fnc("POW", 7, teq::TensptrsT{sub, two});

	opt::GraphInfo graph(teq::TensptrsT{err});

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


TEST(GRAPH, Replace)
{
	teq::Shape in_shape({2,3});
	teq::Shape w0_shape({4,2});
	teq::Shape b0_shape({4});
	teq::Shape w1_shape({3,4});
	teq::Shape b1_shape({3});
	teq::Shape out_shape({3,3});

	double s1 = 1, s2 = 2; 
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

	auto real_sig0 = make_fnc("SIGMOID", 10, teq::TensptrsT{layer0});

	auto m1 = make_fnc("MATMUL", 5, teq::TensptrsT{sig0, w1});
	auto eb1 = make_fnc("EXTEND", 2, teq::TensptrsT{b1});
	auto layer1 = make_fnc("ADD", 4, teq::TensptrsT{m1, eb1});
	auto neg1 = make_fnc("NEG", 3, teq::TensptrsT{layer1});
	auto exp1 = make_fnc("EXP", 1, teq::TensptrsT{neg1});
	auto denom1 = make_fnc("ADD", 4, teq::TensptrsT{one, exp1});
	auto sig1 = make_fnc("DIV", 0, teq::TensptrsT{one, denom1});

	auto real_sig1 = make_fnc("SIMGOID", 10, teq::TensptrsT{layer1});

	auto sub = make_fnc("SUB", 6, teq::TensptrsT{out, sig1});
	auto err = make_fnc("POW", 7, teq::TensptrsT{sub, two});

	EXPECT_CALL(*m1, update_child(teq::TensptrT(real_sig0), 0)).Times(1);
	EXPECT_CALL(*sub, update_child(teq::TensptrT(real_sig1), 1)).Times(1);

	opt::GraphInfo graph(teq::TensptrsT{err, sig0});
	graph.replace({
		{sig0.get(), real_sig0},
		{sig1.get(), real_sig1},
	});

	auto roots = graph.get_roots();
	ASSERT_EQ(2, roots.size());
	EXPECT_EQ(err, roots.front());
	EXPECT_EQ(real_sig0, roots.back()); // expect roots to be replaced
}


#endif // DISABLE_OPT_GRAPH_TEST
