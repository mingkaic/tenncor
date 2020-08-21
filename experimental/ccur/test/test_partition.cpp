
#ifndef DISABLE_PARTITION_TEST


#include <cmath>

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/eteq/eteq.hpp"

#include "ccur/partition.hpp"


TEST(PARTITION, Kpartition)
{
	teq::Shape in_shape({10, 3});
	teq::Shape weight0_shape({9, 10});
	teq::Shape bias0_shape({9});
	teq::Shape weight1_shape({5, 9});
	teq::Shape bias1_shape({5});
	teq::Shape out_shape({5,3});

	std::vector<double> in_data = {
		0.8575073725, 0.0910915775, 0.9133499042,
		0.0660953837, 0.2419061306, 0.3696410139,
		0.4013100896, 0.5172528430, 0.1323293907,
		0.4278464745, 0.0410668952, 0.1652450001,
		0.4190357348, 0.2008750679, 0.5067047954,
		0.0413809185, 0.3094381994, 0.2199267703,
		0.9221660000, 0.0781527711, 0.4804519704,
		0.1206410099, 0.1630566401, 0.6360934496,
		0.2976741228, 0.0709092288, 0.1560062294,
		0.2497900767, 0.2511943240, 0.5474749688,
	};
	std::vector<double> w0_data = {
		0.1613409462, 0.9457144276, 0.9495257985,
		0.2793930966, 0.2723075870, 0.3588235299,
		0.3938297525, 0.3393228095, 0.5716848928,

		0.6339794570, 0.6139023931, 0.7724697132,
		0.0698909799, 0.6535996814, 0.2244414703,
		0.4194435958, 0.6321126915, 0.2891770970,

		0.6457218252, 0.3446479912, 0.3171555503,
		0.2252455176, 0.7602351414, 0.9312997376,
		0.1333143817, 0.7155225995, 0.2032897111,

		0.0224006501, 0.9908721456, 0.0319914474,
		0.9704203846, 0.5274515737, 0.3339836660,
		0.7091134065, 0.5576000673, 0.7501829168,

		0.2442227058, 0.1842266311, 0.8504773433,
		0.3926588922, 0.2833117224, 0.9620642436,
		0.1147953593, 0.4177183136, 0.2914940248,

		0.0219832027, 0.4042951820, 0.3837337063,
		0.5981982488, 0.1894350758, 0.6036559792,
		0.1345821880, 0.8417718235, 0.6846826161,

		0.7122232912, 0.4294986009, 0.6729728379,
		0.4321375967, 0.0759832146, 0.6365364108,
		0.6262763516, 0.7468564758, 0.7312610352,

		0.9549105342, 0.1993684076, 0.4657970235,
		0.7518583439, 0.2239421519, 0.2273247980,
		0.5971696669, 0.7370904837, 0.9237708470,

		0.1546511078, 0.6033025992, 0.4691343777,
		0.1600327544, 0.4296355788, 0.6961808304,
		0.2007259045, 0.3526185965, 0.8712730678,

		0.6285644271, 0.0312853544, 0.7161966751,
		0.1366034209, 0.0015131718, 0.0057481276,
		0.0878447392, 0.4122209597, 0.2014009404,
	};
	std::vector<double> b0_data = {
		0.8801580962, 0.5008790402, 0.8270796004,
		0.7715771391, 0.2662051941, 0.5704192232,
		0.5373307027, 0.1856321630, 0.5669738908,
	};
	std::vector<double> w1_data = {
		0.4260408543, 0.5722656129, 0.9604073010, 0.2924135371, 0.7859575638,
		0.7289488993, 0.9683322421, 0.3507929923, 0.6774173641, 0.6407122174,
		0.8975668345, 0.4348200381, 0.6121628654, 0.8736230885, 0.0833758337,
		0.8422647036, 0.3981612935, 0.0784194187, 0.7461062031, 0.4919135980,
		0.8157699378, 0.4931650049, 0.6282318830, 0.6176567461, 0.8502403216,
		0.9404269220, 0.6854869637, 0.9051396941, 0.2966845031, 0.2721141527,
		0.2877237941, 0.0590600035, 0.6288776397, 0.1353232608, 0.9594369234,
		0.5920096937, 0.1026460668, 0.9349781326, 0.2640904799, 0.6960341493,
		0.5056684425, 0.6169691389, 0.8741161106, 0.5260663550, 0.8161608103,
	};
	std::vector<double> b1_data = {
		0.1993173272, 0.6008457459, 0.3355862244, 0.1906307583, 0.3078908360,
	};
	std::vector<double> out_data = {
		0.5301287168, 0.0816631236, 0.4232512930, 0.1983706018, 0.2941365700,
		0.9427764606, 0.0267634765, 0.4367877602, 0.1155584527, 0.6275693090,
		0.4350741570, 0.3949956178, 0.2341486792, 0.1348473539, 0.8681677362,
	};

	eteq::EVariable<double> in = eteq::make_variable<double>(in_data.data(), in_shape);
	eteq::EVariable<double> weight0 = eteq::make_variable<double>(w0_data.data(), weight0_shape);
	eteq::EVariable<double> bias0 = eteq::make_variable<double>(b0_data.data(), bias0_shape);
	eteq::EVariable<double> weight1 = eteq::make_variable<double>(w1_data.data(), weight1_shape);
	eteq::EVariable<double> bias1 = eteq::make_variable<double>(b1_data.data(), bias1_shape);
	eteq::EVariable<double> out = eteq::make_variable<double>(out_data.data(), out_shape);

	auto layer0 = tenncor<double>().matmul(in, weight0) + tenncor<double>().extend(bias0, 1, {3});
	auto sig0 = tenncor<double>().sigmoid(layer0);

	auto layer1 = tenncor<double>().matmul(sig0, weight1) + tenncor<double>().extend(bias1, 1, {3});
	auto sig1 = tenncor<double>().sigmoid(layer1);

	auto err = tenncor<double>().pow(out - sig1, 2.);

	auto dw0 = tcr::derive(err, weight0);
	auto db0 = tcr::derive(err, bias0);
	auto dw1 = tcr::derive(err, weight1);
	auto db1 = tcr::derive(err, bias1);

	auto groups = ccur::k_partition({dw0, db0, dw1, db1}, 2);

	ASSERT_EQ(2, groups.size());
	long ng0 = groups[0].size();
	long ng1 = groups[1].size();
	EXPECT_GE(1, std::abs(ng0 - ng1));
}


#endif // DISABLE_PARTITION_TEST
