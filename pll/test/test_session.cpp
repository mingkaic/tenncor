
#ifndef DISABLE_SESSION_TEST


#include <cmath>

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "ead/ead.hpp"

#include "pll/session.hpp"


TEST(SESSION, Update)
{
	ade::Shape in_shape({10, 3});
	ade::Shape weight0_shape({9, 10});
	ade::Shape bias0_shape({9});
	ade::Shape weight1_shape({5, 9});
	ade::Shape bias1_shape({5});
	ade::Shape out_shape({5,3});

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

	ead::NodeptrT<double> in = ead::make_variable<double>(in_data.data(), in_shape);
	ead::NodeptrT<double> weight0 = ead::make_variable<double>(w0_data.data(), weight0_shape);
	ead::NodeptrT<double> bias0 = ead::make_variable<double>(b0_data.data(), bias0_shape);
	ead::NodeptrT<double> weight1 = ead::make_variable<double>(w1_data.data(), weight1_shape);
	ead::NodeptrT<double> bias1 = ead::make_variable<double>(b1_data.data(), bias1_shape);
	ead::NodeptrT<double> out = ead::make_variable<double>(out_data.data(), out_shape);

	auto layer0 = tenncor::add(tenncor::matmul(in, weight0), tenncor::extend(bias0, 1, {3}));
	auto sig0 = tenncor::sigmoid(layer0);

	auto layer1 = tenncor::add(tenncor::matmul(sig0, weight1), tenncor::extend(bias1, 1, {3}));
	auto sig1 = tenncor::sigmoid(layer1);

	auto err = tenncor::pow(tenncor::sub(out, sig1), ead::make_constant_scalar<double>(2, out_shape));

	auto dw0 = ead::derive(err, weight0);
	auto db0 = ead::derive(err, bias0);
	auto dw1 = ead::derive(err, weight1);
	auto db1 = ead::derive(err, bias1);

	pll::Session sess(2);;
	sess.track({
		dw0->get_tensor(),
		db0->get_tensor(),
		dw1->get_tensor(),
		db1->get_tensor(),
	});
	sess.update();

	std::vector<double> expect_gw0 = {
		0.0027019915285570115, 0.0049524065585107684, 0.0025163812594914499,
		0.0038923591714517891, 0.0069661439225563049, 0.0031510414764447736,
		0.0022478768029587262, 0.0022765639443853786, 0.0033410980166779791,
		0.00097522116465253051, 0.001693896620321596, 0.00095117349853452403,
		0.0013105640666332885, 0.0020640852743794621, 0.00092385650660917521,
		0.00066472294682227138, 0.00077684718047769464, 0.0010573870695921599,
		0.0030538233109959514, 0.0059914873371482626, 0.002816933963813279,
		0.0046344494070597901, 0.0074484900533361223, 0.002923178818749005,
		0.0025888494003570082, 0.0024167440905060772, 0.0036775897657920385,
		0.0024904082366803493, 0.0038341910866663683, 0.002500010407779862,
		0.0030142629092504893, 0.0052389646418249634, 0.0027829501832842408,
		0.0014995699089375711, 0.0020951478239990117, 0.002647907962560701,
		0.0027345254951523104, 0.0047865076822633029, 0.0026658727484682914,
		0.0036954188898844099, 0.0057282961097162239, 0.0025178275492600422,
		0.0018635506556079224, 0.0021607566147704336, 0.0029495543716158779,
		0.00085640005061550983, 0.0017314042599025487, 0.00076741704372808913,
		0.0013502530638826596, 0.0023208864089261359, 0.0009271185274612394,
		0.0008027150040004346, 0.00070015113143881663, 0.0011081335206311336,
		0.0019621893481767188, 0.0036606508512356819, 0.00185644674133355,
		0.0028297033136024576, 0.004484638344372795, 0.0018499041696268298,
		0.0015164101514628795, 0.0015527185924082477, 0.0022527047367589602,
		0.0021165392753594199, 0.003915006058678872, 0.0019922410790477645,
		0.0030454607213162344, 0.0050629793776993106, 0.00217522488536227,
		0.0016780267061119055, 0.0017163284163560558, 0.0024973871364443096,
		0.0037214774412162085, 0.0065510651741015865, 0.0036557068873296914,
		0.0050194415379978593, 0.0073000722708336623, 0.0030647941129167272,
		0.0024288766857152181, 0.0028581925623700367, 0.0038626543124159161,
		0.0023858654073475815, 0.0040005201672014658, 0.0023078024962114858,
		0.0031512536042022608, 0.0056436665737315383, 0.0027835035219529766,
		0.0017163829072048768, 0.0020227502089074081, 0.0027598072650753232,
	};
	std::vector<double> expect_gb0 = {
		0.0073014851440853963, 0.012795691944764515, 0.0070609853996545418,
		0.0099353821844054037, 0.016126120980560758, 0.0072594252580560938,
		0.0051855631031577599, 0.0058893997089519293, 0.0081361796595888665,
	};
	std::vector<double> expect_gw1 = {
		0.0075364033117421396, 0.050253916292103726, 0.012521300175850935,
		0.071258507740799906, 0.011490047203544582, 0.0074889441884519962,
		0.049364053114276017, 0.012400168850027316, 0.070298267826285218,
		0.011199991977589233, 0.0078271335097107944, 0.052010295641316096,
		0.013009172329239052, 0.073906224147165633, 0.011831323397315638,
		0.007474817281334166, 0.049098796177996611, 0.012354849203709242,
		0.069981094983390488, 0.011130069952177708, 0.006659969695296075,
		0.045029505644527811, 0.01116649756842061, 0.063711311399035839,
		0.010289373247410607, 0.0074443608399844983, 0.050801211812765354,
		0.012559014313408341, 0.071776564373706303, 0.011602207271183348,
		0.0070458198331052385, 0.046265739369542938, 0.011635104124310208,
		0.065919115499176104, 0.010502662637079581, 0.0073178590190504517,
		0.048699319909912865, 0.012117165013920111, 0.068991625331410358,
		0.011180624845463449, 0.0076281958672005084, 0.050999158876468637,
		0.012744598043270331, 0.072449921542875759, 0.011571221536496201,
	};
	std::vector<double> expect_gb1 = {
		0.0083642760546030238, 0.055794840260732122, 0.013998520595663699, 0.079441131671069379, 0.012580279526871282,
	};

	{
		auto gotshape = dw0->shape();
		ASSERT_ARREQ(weight0_shape, gotshape);
	}
	double* gw0ptr = (double*) dw0->data();
	for (size_t i = 0, n = weight0_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw0[i], gw0ptr[i]);
	}
	{
		auto gotshape = db0->shape();
		ASSERT_ARREQ(bias0_shape, gotshape);
	}
	double* gb0ptr = (double*) db0->data();
	for (size_t i = 0, n = bias0_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb0[i], gb0ptr[i]);
	}
	{
		auto gotshape = dw1->shape();
		ASSERT_ARREQ(weight1_shape, gotshape);
	}

	double* gw1ptr = (double*) dw1->data();
	for (size_t i = 0, n = weight1_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw1[i], gw1ptr[i]);
	}
	{
		auto gotshape = db1->shape();
		ASSERT_ARREQ(bias1_shape, gotshape);
	}
	double* gb1ptr = (double*) db1->data();
	for (size_t i = 0, n = bias1_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb1[i], gb1ptr[i]);
	}
}


#endif // DISABLE_SESSION_TEST
