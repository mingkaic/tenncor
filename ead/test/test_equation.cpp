
#ifndef DISABLE_EQUATION_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "ead/generated/api.hpp"

#include "ead/session.hpp"
#include "ead/grader.hpp"
#include "ead/constant.hpp"
#include "ead/variable.hpp"


TEST(EQUATION, MatmulComplex)
{
	std::vector<ade::DimT> alist = {3, 2};
	std::vector<ade::DimT> blist = {4, 3};
	std::vector<ade::DimT> clist = {2, 4};
	ade::Shape ashape(alist);
	ade::Shape bshape(blist);
	ade::Shape cshape(clist);

	std::vector<int32_t> data = {
		40, 1, 23,
		18, 50, 77,
	};
	std::vector<int32_t> data2 = {
		62, 31, 90, 68,
		68, 78, 55, 95,
		16, 99, 97, 77,
	};
	std::vector<int32_t> data3 = {
		29, 75,
		39, 67,
		37, 57,
		48, 42,
	};
	std::vector<int32_t> expect_ga = {
		-154880684, -127906804, -105914132,
		-206505460, -164002948, -131588540,
	};
	std::vector<int32_t> expect_gb = {
		-55352996, 20961008, -56860896, -29599272,
		-58614174, 26705056, -60728512, -32475964,
		-108808856, 47268840, -112469200, -59707840,
	};
	std::vector<int32_t> expect_gc = {
		152732, 4310652,
		-73239126, -139902552,
		-56297930, -101235528,
		-79671648, -172118240,
	};

	ead::NodeptrT<int32_t> a = ead::make_variable<int32_t>(data.data(), ashape);
	ead::NodeptrT<int32_t> b = ead::make_variable<int32_t>(data2.data(), bshape);
	ead::NodeptrT<int32_t> c = ead::make_variable<int32_t>(data3.data(), cshape);

	auto d = age::matmul(a, b);
	auto e = age::matmul(c, d);
	auto f = age::matmul(age::transpose(d), age::transpose(c));
	auto dest = age::matmul(e, f);

	auto da = ead::derive(dest, a);
	auto db = ead::derive(dest, b);
	auto dc = ead::derive(dest, c);

	ead::Session<int32_t> session;
	session.track(dest);
	session.track(da);
	session.track(db);
	session.track(dc);
	session.update();

	{
		auto gotshape = da->shape();
		ASSERT_ARREQ(alist, gotshape);
	}
	int32_t* gaptr = (int32_t*) da->data();
	for (size_t i = 0, n = ashape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(expect_ga[i], gaptr[i]);
	}
	{
		auto gotshape = db->shape();
		ASSERT_ARREQ(blist, gotshape);
	}
	int32_t* gbptr = (int32_t*) db->data();
	for (size_t i = 0, n = bshape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(expect_gb[i], gbptr[i]);
	}
	{
		auto gotshape = dc->shape();
		ASSERT_ARREQ(clist, gotshape);
	}
	int32_t* gcptr = (int32_t*) dc->data();
	for (size_t i = 0, n = cshape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(expect_gc[i], gcptr[i]);
	}
}


TEST(EQUATION, SigmoidMLP_Precise)
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

	auto layer0 = age::add(age::matmul(in, weight0), age::extend(bias0, 1, {3}));
	auto sig0 = age::div(ead::make_constant_scalar<double>(1, ade::Shape({9, 3})),
		age::add(ead::make_constant_scalar<double>(1, ade::Shape({9, 3})),
			age::exp(age::neg(layer0))));

	auto layer1 = age::add(age::matmul(sig0, weight1), age::extend(bias1, 1, {3}));
	auto sig1 = age::div(ead::make_constant_scalar<double>(1, ade::Shape({5, 3})),
		age::add(ead::make_constant_scalar<double>(1, ade::Shape({5, 3})),
			age::exp(age::neg(layer1))));

	auto err = age::pow(age::sub(out, sig1), ead::make_constant_scalar<double>(2, out_shape));

	auto dw0 = ead::derive(err, weight0);
	auto db0 = ead::derive(err, bias0);
	auto dw1 = ead::derive(err, weight1);
	auto db1 = ead::derive(err, bias1);

	ead::Session<double> session;
	session.track(dw0);
	session.track(db0);
	session.track(dw1);
	session.track(db1);
	session.update();

	std::vector<double> expect_gw0 = {
		0.0027019915285569958557, 0.0049524065585107423376, 0.0025163812594914390132,
		0.003892359171451776137, 0.0069661439225562789279, 0.0031510414764447489296,
		0.0022478768029587292351, 0.0022765639443853690621, 0.0033410980166779665444,

		0.00097522116465252812242, 0.001693896620321594295, 0.00095117349853452229369,
		0.0013105640666332882532, 0.0020640852743794577476, 0.00092385650660917043898,
		0.00066472294682227062081, 0.0007768471804776914914, 0.0010573870695921579015,

		0.0030538233109959527292, 0.0059914873371482522263, 0.0028169339638132759145,
		0.0046344494070597866062, 0.0074484900533361162248, 0.0029231788187489967865,
		0.0025888494003570216781, 0.0024167440905060806784, 0.0036775897657920384939,

		0.0024904082366803271913, 0.0038341910866663535956, 0.0025000104077798506776,
		0.0030142629092504832564, 0.005238964641824933946, 0.0027829501832842135078,
		0.0014995699089375572402, 0.0020951478239989895953, 0.0026479079625606844958,

		0.002734525495152305679, 0.004786507682263300284, 0.0026658727484682879019,
		0.0036954188898844107444, 0.0057282961097162160558, 0.0025178275492600309463,
		0.0018635506556079210923, 0.0021607566147704261941, 0.0029495543716158744221,

		0.00085640005061550896273, 0.0017314042599025419519, 0.0007674170437280871792,
		0.0013502530638826559361, 0.0023208864089261311073, 0.00092711852746123484155,
		0.00080271500400043948025, 0.00070015113143881760587, 0.001108133520631132251,

		0.0019621893481767179596, 0.0036606508512356775338, 0.0018564467413335478119,
		0.0028297033136024567559, 0.0044846383443727889581, 0.0018499041696268233095,
		0.0015164101514628836216, 0.0015527185924082463966, 0.0022527047367589584335,

		0.0021165392753594151377, 0.0039150060586788616376, 0.001992241079047760155,
		0.00304546072131622967, 0.0050629793776992993576, 0.0021752248853622582411,
		0.0016780267061119089576, 0.0017163284163560521035, 0.0024973871364443048573,

		0.0037214774412162102135, 0.0065510651741015942656, 0.0036557068873296918142,
		0.0050194415379978670963, 0.0073000722708336622524, 0.0030647941129167211374,
		0.0024288766857152176322, 0.0028581925623700306024, 0.0038626543124159170087,

		0.0023858654073475628332, 0.0040005201672014458619, 0.0023078024962114745446,
		0.0031512536042022512919, 0.0056436665737315105246, 0.0027835035219529510181,
		0.0017163829072048709251, 0.0020227502089073924954, 0.0027598072650753084634,
	};
	std::vector<double> expect_gb0 = {
		0.0073014851440853728928, 0.012795691944764490261, 0.0070609853996545252769,
		0.0099353821844053950146, 0.016126120980560716689, 0.0072594252580560503924,
		0.0051855631031577564469, 0.0058893997089519050173, 0.0081361796595888474098,
	};
	std::vector<double> expect_gw1 = {
		0.0075364033117420892866, 0.050253916292103517627, 0.012521300175850822584, 0.071258507740799781338, 0.011490047203544696483,
		0.007488944188451942402, 0.049364053114275815992, 0.012400168850027197542, 0.070298267826285107396, 0.011199991977589342229,
		0.0078271335097107406359, 0.052010295641315887338, 0.013009172329238934129, 0.073906224147165494598, 0.011831323397315757912,
		0.0074748172813341130435, 0.049098796177996409384, 0.012354849203709123567, 0.069981094983390362829, 0.01113006995217781514,
		0.0066599696952960316457, 0.045029505644527637043, 0.011166497568420511236, 0.063711311399035727709, 0.010289373247410710938,
		0.0074443608399844506, 0.050801211812765167, 0.012559014313408234, 0.071776564373706178, 0.011602207271183464,
		0.007045819833105189, 0.046265739369542737, 0.011635104124310097, 0.065919115499175993, 0.010502662637079685,
		0.0073178590190504031, 0.048699319909912664, 0.012117165013920005, 0.068991625331410233, 0.011180624845463562,
		0.0076281958672004555, 0.050999158876468442, 0.012744598043270213, 0.072449921542875648, 0.011571221536496312,
	};
	std::vector<double> expect_gb1 = {
		0.0083642760546029649, 0.055794840260731907, 0.013998520595663564, 0.07944113167106924, 0.012580279526871407,
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


TEST(EQUATION, DISABLED_SigmoidMLP_Fast)
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

	auto layer0 = age::add(age::matmul(in, weight0), age::extend(bias0, 1, {3}));
	auto sig0 = age::sigmoid(layer0);

	auto layer1 = age::add(age::matmul(sig0, weight1), age::extend(bias1, 1, {3}));
	auto sig1 = age::sigmoid(layer1);

	auto err = age::pow(age::sub(out, sig1), ead::make_constant_scalar<double>(2, out_shape));

	auto dw0 = ead::derive(err, weight0);
	auto db0 = ead::derive(err, bias0);
	auto dw1 = ead::derive(err, weight1);
	auto db1 = ead::derive(err, bias1);

	ead::Session<double> session;
	session.track(dw0);
	session.track(db0);
	session.track(dw1);
	session.track(db1);
	session.update();

	std::vector<double> expect_gw0 = {
		0.0027019915285569958557, 0.0049524065585107423376, 0.0025163812594914390132,
		0.003892359171451776137, 0.0069661439225562789279, 0.0031510414764447489296,
		0.0022478768029587292351, 0.0022765639443853690621, 0.0033410980166779665444,

		0.00097522116465252812242, 0.001693896620321594295, 0.00095117349853452229369,
		0.0013105640666332882532, 0.0020640852743794577476, 0.00092385650660917043898,
		0.00066472294682227062081, 0.0007768471804776914914, 0.0010573870695921579015,

		0.0030538233109959527292, 0.0059914873371482522263, 0.0028169339638132759145,
		0.0046344494070597866062, 0.0074484900533361162248, 0.0029231788187489967865,
		0.0025888494003570216781, 0.0024167440905060806784, 0.0036775897657920384939,

		0.0024904082366803271913, 0.0038341910866663535956, 0.0025000104077798506776,
		0.0030142629092504832564, 0.005238964641824933946, 0.0027829501832842135078,
		0.0014995699089375572402, 0.0020951478239989895953, 0.0026479079625606844958,

		0.002734525495152305679, 0.004786507682263300284, 0.0026658727484682879019,
		0.0036954188898844107444, 0.0057282961097162160558, 0.0025178275492600309463,
		0.0018635506556079210923, 0.0021607566147704261941, 0.0029495543716158744221,

		0.00085640005061550896273, 0.0017314042599025419519, 0.0007674170437280871792,
		0.0013502530638826559361, 0.0023208864089261311073, 0.00092711852746123484155,
		0.00080271500400043948025, 0.00070015113143881760587, 0.001108133520631132251,

		0.0019621893481767179596, 0.0036606508512356775338, 0.0018564467413335478119,
		0.0028297033136024567559, 0.0044846383443727889581, 0.0018499041696268233095,
		0.0015164101514628836216, 0.0015527185924082463966, 0.0022527047367589584335,

		0.0021165392753594151377, 0.0039150060586788616376, 0.001992241079047760155,
		0.00304546072131622967, 0.0050629793776992993576, 0.0021752248853622582411,
		0.0016780267061119089576, 0.0017163284163560521035, 0.0024973871364443048573,

		0.0037214774412162102135, 0.0065510651741015942656, 0.0036557068873296918142,
		0.0050194415379978670963, 0.0073000722708336622524, 0.0030647941129167211374,
		0.0024288766857152176322, 0.0028581925623700306024, 0.0038626543124159170087,

		0.0023858654073475628332, 0.0040005201672014458619, 0.0023078024962114745446,
		0.0031512536042022512919, 0.0056436665737315105246, 0.0027835035219529510181,
		0.0017163829072048709251, 0.0020227502089073924954, 0.0027598072650753084634,
	};
	std::vector<double> expect_gb0 = {
		0.0073014851440853728928, 0.012795691944764490261, 0.0070609853996545252769,
		0.0099353821844053950146, 0.016126120980560716689, 0.0072594252580560503924,
		0.0051855631031577564469, 0.0058893997089519050173, 0.0081361796595888474098,
	};
	std::vector<double> expect_gw1 = {
		0.0075364033117420892866, 0.050253916292103517627, 0.012521300175850822584, 0.071258507740799781338, 0.011490047203544696483,
		0.007488944188451942402, 0.049364053114275815992, 0.012400168850027197542, 0.070298267826285107396, 0.011199991977589342229,
		0.0078271335097107406359, 0.052010295641315887338, 0.013009172329238934129, 0.073906224147165494598, 0.011831323397315757912,
		0.0074748172813341130435, 0.049098796177996409384, 0.012354849203709123567, 0.069981094983390362829, 0.01113006995217781514,
		0.0066599696952960316457, 0.045029505644527637043, 0.011166497568420511236, 0.063711311399035727709, 0.010289373247410710938,
		0.0074443608399844506, 0.050801211812765167, 0.012559014313408234, 0.071776564373706178, 0.011602207271183464,
		0.007045819833105189, 0.046265739369542737, 0.011635104124310097, 0.065919115499175993, 0.010502662637079685,
		0.0073178590190504031, 0.048699319909912664, 0.012117165013920005, 0.068991625331410233, 0.011180624845463562,
		0.0076281958672004555, 0.050999158876468442, 0.012744598043270213, 0.072449921542875648, 0.011571221536496312,
	};
	std::vector<double> expect_gb1 = {
		0.0083642760546029649, 0.055794840260731907, 0.013998520595663564, 0.07944113167106924, 0.012580279526871407,
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


#endif // DISABLE_EQUATION_TEST
