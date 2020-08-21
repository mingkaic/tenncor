
#ifndef DISABLE_EQUATION_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/teq/evaluator.hpp"

#include "tenncor/optimize.hpp"

#include "tenncor/tenncor.hpp"


using TensProcF = std::function<void(teq::TensptrsT&)>;


static void matmul_complex (TensProcF root_proc = TensProcF())
{
	eigen::Device device;
	std::vector<teq::DimT> alist = {3, 2};
	std::vector<teq::DimT> blist = {4, 3};
	std::vector<teq::DimT> clist = {2, 4};
	teq::Shape ashape(alist);
	teq::Shape bshape(blist);
	teq::Shape cshape(clist);

	std::vector<float> data = {
		40, 1, 23,
		18, 50, 77,
	};
	std::vector<float> data2 = {
		62, 31, 90, 68,
		68, 78, 55, 95,
		16, 99, 97, 77,
	};
	std::vector<float> data3 = {
		29, 75,
		39, 67,
		37, 57,
		48, 42,
	};
	std::vector<float> expect_ga = {
		245250818048, 287692324864, 309372977152,
		386310111232, 453162434560, 487312982016,
	};
	std::vector<float> expect_gb = {
		38305894400, 72401903616, 78513586176, 74675994624,
		44697538560, 84482736128, 91614191616, 87136288768,
		80860676096, 152834621440, 165735874560, 157635051520,
	};
	std::vector<float> expect_gc = {
		112505257984, 278567649280,
		112505257984, 278567649280,
		112505257984, 278567649280,
		112505257984, 278567649280,
	};

	eteq::EVariable<float> a =
		eteq::make_variable<float>(data.data(), ashape);
	eteq::EVariable<float> b =
		eteq::make_variable<float>(data2.data(), bshape);
	eteq::EVariable<float> c =
		eteq::make_variable<float>(data3.data(), cshape);

	auto d = tenncor<float>().matmul(a, b);
	auto e = tenncor<float>().matmul(c, d);
	auto f = tenncor<float>().matmul(tenncor<float>().transpose(d), tenncor<float>().transpose(c));
	auto dest = tenncor<float>().matmul(e, f);

	auto ders = tcr::derive(dest, {a, b, c});
	auto da = ders[0];
	auto db = ders[1];
	auto dc = ders[2];

	teq::TensptrsT roots = {dest, da, db, dc};
	if (root_proc)
	{
		root_proc(roots);
	}

	teq::Evaluator eval;
	teq::TensSetT targets;
	std::transform(roots.begin(), roots.end(),
		std::inserter(targets, targets.end()),
		[](teq::TensptrT g) { return g.get(); });
	eval.evaluate(device, targets);

	{
		auto gotshape = da->shape();
		ASSERT_ARREQ(ashape, gotshape);
	}
	float* gaptr = (float*) da->device().data();
	for (size_t i = 0, n = ashape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(expect_ga[i], gaptr[i]);
	}

	{
		auto gotshape = db->shape();
		ASSERT_ARREQ(bshape, gotshape);
	}
	float* gbptr = (float*) db->device().data();
	for (size_t i = 0, n = bshape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(expect_gb[i], gbptr[i]);
	}

	{
		auto gotshape = dc->shape();
		ASSERT_ARREQ(cshape, gotshape);
	}
	float* gcptr = (float*) dc->device().data();
	for (size_t i = 0, n = cshape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(expect_gc[i], gcptr[i]);
	}
}


static void sigmoid_MLP_slow (TensProcF root_proc = TensProcF())
{
	eigen::Device device;
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

	eteq::EVariable<double> in =
		eteq::make_variable<double>(in_data.data(), in_shape);
	eteq::EVariable<double> weight0 =
		eteq::make_variable<double>(w0_data.data(), weight0_shape);
	eteq::EVariable<double> bias0 =
		eteq::make_variable<double>(b0_data.data(), bias0_shape);
	eteq::EVariable<double> weight1 =
		eteq::make_variable<double>(w1_data.data(), weight1_shape);
	eteq::EVariable<double> bias1 =
		eteq::make_variable<double>(b1_data.data(), bias1_shape);
	eteq::EVariable<double> out =
		eteq::make_variable<double>(out_data.data(), out_shape);

	auto layer0 = tenncor<double>().matmul(in, weight0) + tenncor<double>().extend(bias0, 1, {3});
	auto sig0 = 1. / (1. + tenncor<double>().exp(-layer0));

	auto layer1 = tenncor<double>().matmul(sig0, weight1) + tenncor<double>().extend(bias1, 1, {3});
	auto sig1 = 1. / (1. + tenncor<double>().exp(-layer1));

	auto err = tenncor<double>().pow(out - sig1, 2.);

	auto ders = tcr::derive(err, {weight0, bias0, weight1, bias1});
	auto dw0 = ders[0];
	auto db0 = ders[1];
	auto dw1 = ders[2];
	auto db1 = ders[3];

	teq::TensptrsT roots = {dw0, db0, dw1, db1};
	if (root_proc)
	{
		root_proc(roots);
	}

	teq::Evaluator eval;
	teq::TensSetT targets;
	std::transform(roots.begin(), roots.end(),
		std::inserter(targets, targets.end()),
		[](teq::TensptrT g) { return g.get(); });
	eval.evaluate(device, targets);

	{
		auto gotshape = dw0->shape();
		ASSERT_ARREQ(weight0_shape, gotshape);
	}
	double* gw0ptr = (double*) dw0->device().data();
	for (size_t i = 0, n = weight0_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw0[i], gw0ptr[i]);
	}

	{
		auto gotshape = db0->shape();
		ASSERT_ARREQ(bias0_shape, gotshape);
	}
	double* gb0ptr = (double*) db0->device().data();
	for (size_t i = 0, n = bias0_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb0[i], gb0ptr[i]);
	}

	{
		auto gotshape = dw1->shape();
		ASSERT_ARREQ(weight1_shape, gotshape);
	}
	double* gw1ptr = (double*) dw1->device().data();
	for (size_t i = 0, n = weight1_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw1[i], gw1ptr[i]);
	}

	{
		auto gotshape = db1->shape();
		ASSERT_ARREQ(bias1_shape, gotshape);
	}
	double* gb1ptr = (double*) db1->device().data();
	for (size_t i = 0, n = bias1_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb1[i], gb1ptr[i]);
	}
}


static void sigmoid_MLP_fast (TensProcF root_proc = TensProcF())
{
	eigen::Device device;
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

	eteq::EVariable<double> in =
		eteq::make_variable<double>(in_data.data(), in_shape);
	eteq::EVariable<double> weight0 =
		eteq::make_variable<double>(w0_data.data(), weight0_shape);
	eteq::EVariable<double> bias0 =
		eteq::make_variable<double>(b0_data.data(), bias0_shape);
	eteq::EVariable<double> weight1 =
		eteq::make_variable<double>(w1_data.data(), weight1_shape);
	eteq::EVariable<double> bias1 =
		eteq::make_variable<double>(b1_data.data(), bias1_shape);
	eteq::EVariable<double> out =
		eteq::make_variable<double>(out_data.data(), out_shape);

	auto layer0 = tenncor<double>().matmul(in, weight0) + tenncor<double>().extend(bias0, 1, {3});
	auto sig0 = tenncor<double>().sigmoid(layer0);

	auto layer1 = tenncor<double>().matmul(sig0, weight1) + tenncor<double>().extend(bias1, 1, {3});
	auto sig1 = tenncor<double>().sigmoid(layer1);

	auto err = tenncor<double>().pow(out - sig1, 2.);

	auto ders = tcr::derive(err, {weight0, bias0, weight1, bias1});
	auto dw0 = ders[0];
	auto db0 = ders[1];
	auto dw1 = ders[2];
	auto db1 = ders[3];

	teq::TensptrsT roots = {dw0, db0, dw1, db1};
	if (root_proc)
	{
		root_proc(roots);
	}

	teq::Evaluator eval;
	teq::TensSetT targets;
	std::transform(roots.begin(), roots.end(),
		std::inserter(targets, targets.end()),
		[](teq::TensptrT g) { return g.get(); });
	eval.evaluate(device, targets);

	{
		auto gotshape = dw0->shape();
		ASSERT_ARREQ(weight0_shape, gotshape);
	}
	double* gw0ptr = (double*) dw0->device().data();
	for (size_t i = 0, n = weight0_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw0[i], gw0ptr[i]);
	}

	{
		auto gotshape = db0->shape();
		ASSERT_ARREQ(bias0_shape, gotshape);
	}
	double* gb0ptr = (double*) db0->device().data();
	for (size_t i = 0, n = bias0_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb0[i], gb0ptr[i]);
	}

	{
		auto gotshape = dw1->shape();
		ASSERT_ARREQ(weight1_shape, gotshape);
	}
	double* gw1ptr = (double*) dw1->device().data();
	for (size_t i = 0, n = weight1_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw1[i], gw1ptr[i]);
	}

	{
		auto gotshape = db1->shape();
		ASSERT_ARREQ(bias1_shape, gotshape);
	}
	double* gb1ptr = (double*) db1->device().data();
	for (size_t i = 0, n = bias1_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb1[i], gb1ptr[i]);
	}
}


static void tanh_RNN (TensProcF root_proc = TensProcF())
{
	eigen::Device device;
	teq::Shape in_shape({5, 3});
	teq::Shape weight_shape({5, 10});
	teq::Shape bias_shape({5});
	teq::Shape state_shape({5});
	teq::Shape out_shape({5, 3});

	std::vector<double> in_data = {
		0.8575073725, 0.0910915775, 0.9133499042, 0.0660953837, 0.2419061306,
		0.3696410139, 0.4013100896, 0.5172528430, 0.1323293907, 0.4278464745,
		0.0410668952, 0.1652450001, 0.4190357348, 0.2008750679, 0.5067047954,
	};
	std::vector<double> weight_data = {
		0.1613409462, 0.9457144276, 0.9495257985, 0.2793930966, 0.2723075870,
		0.3588235299, 0.3938297525, 0.3393228095, 0.5716848928, 0.6339794570,
		0.6139023931, 0.7724697132, 0.0698909799, 0.6535996814, 0.2244414703,
		0.4194435958, 0.6321126915, 0.2891770970, 0.6457218252, 0.3446479912,
		0.3171555503, 0.2252455176, 0.7602351414, 0.9312997376, 0.1333143817,
		0.7155225995, 0.2032897111, 0.0224006501, 0.9908721456, 0.0319914474,
		0.9704203846, 0.5274515737, 0.3339836660, 0.7091134065, 0.5576000673,
		0.7501829168, 0.2442227058, 0.1842266311, 0.8504773433, 0.3926588922,
		0.2833117224, 0.9620642436, 0.1147953593, 0.4177183136, 0.2914940248,
		0.0219832027, 0.4042951820, 0.3837337063, 0.5981982488, 0.1894350758,
	};
	std::vector<double> bias_data = {
		0.8801580962, 0.5008790402, 0.8270796004, 0.7715771391, 0.2662051941,
	};
	std::vector<double> state_data = {
		0.1993173272, 0.6008457459, 0.3355862244, 0.1906307583, 0.3078908360,
	};
	std::vector<double> out_data = {
		0.5301287168, 0.0816631236, 0.4232512930, 0.1983706018, 0.2941365700,
		0.9427764606, 0.0267634765, 0.4367877602, 0.1155584527, 0.6275693090,
		0.4350741570, 0.3949956178, 0.2341486792, 0.1348473539, 0.8681677362,
	};
	std::vector<double> expect_gw = {
		0.012832730484849566, 0.020665413892638967, 0.044297974765077949, 0.011199947435827553, 0.23221427250804694,
		0.0015635327458991598, 0.0041422544430273836, 0.019402241314062488, 0.0012707560367359271, 0.037639635777849893,
		0.014074278095615077, 0.024083876937220847, 0.067053962426625174, 0.012012401554196661, 0.25574625718565897,
		0.0012062411866441238, 0.0028174985838895759, 0.014657480646437735, 0.0009127396798470507, 0.023503570273553907,
		0.0041692909406491705, 0.0091141905411763757, 0.041922218230865907, 0.0032927494952498682, 0.081469894232265208,
		0.0041029832998777625, 0.012258101019880084, 0.074315292906842023, 0.0029076822315577708, 0.093741511447534259,
		0.010076544707145474, 0.021262033306889326, 0.090722915084789413, 0.0081235274938007002, 0.19704566770323834,
		0.0061205009797556456, 0.015240004827907053, 0.07927892619705644, 0.0046747767153010418, 0.12836943305677537,
		0.0039744284304135055, 0.012078611732748932, 0.074056104607064382, 0.0027954950855819194, 0.09163377907594053,
		0.0056874555383919466, 0.014238599249320296, 0.075673675846280597, 0.0042988939595802692, 0.11820195174875535,
	};
	std::vector<double> expect_gb = {
		0.016016579039489543, 0.030232782277125642, 0.10717897974006134, 0.013309371208968232, 0.29988019481964079,
	};
	std::vector<double> expect_gstate = {
		0.037213495617765324, 0.19248718495392608, 0.13617200776678728, 0.11084523350674103, 0.081535346039776163
	};

	eteq::EVariable<double> in =
		eteq::make_variable<double>(in_data.data(), in_shape);
	eteq::EVariable<double> weight =
		eteq::make_variable<double>(weight_data.data(), weight_shape);
	eteq::EVariable<double> bias =
		eteq::make_variable<double>(bias_data.data(), bias_shape);
	eteq::EVariable<double> istate =
		eteq::make_variable<double>(state_data.data(), state_shape);
	eteq::EVariable<double> out =
		eteq::make_variable<double>(out_data.data(), out_shape);

	teq::RankT seq_dim = 1;
	size_t nseq = in->shape().at(seq_dim);
	eteq::ETensor<double> state = istate;
	std::vector<eteq::ETensor<double>> states;
	for (size_t i = 0; i < nseq; ++i)
	{
		auto inslice = tenncor<double>().slice(in, i, 1, seq_dim);
		state = tenncor<double>().tanh(tenncor<double>().nn.fully_connect(
			{tenncor<double>().concat(inslice, state, 0)},
			{weight}, bias));
		states.push_back(state);
	}
	auto output = tenncor<double>().concat(states, seq_dim);

	auto err = tenncor<double>().pow(out - output, 2.);

	auto ders = tcr::derive(err, {weight, bias, istate});
	auto dw = ders[0];
	auto db = ders[1];
	auto dstate = ders[2];

	teq::TensptrsT roots = {dw, db, dstate};
	if (root_proc)
	{
		root_proc(roots);
	}

	teq::Evaluator eval;
	teq::TensSetT targets;
	std::transform(roots.begin(), roots.end(),
		std::inserter(targets, targets.end()),
		[](teq::TensptrT g) { return g.get(); });
	eval.evaluate(device, targets);

	{
		auto gotshape = dw->shape();
		ASSERT_ARREQ(weight_shape, gotshape);
	}
	double* gwptr = (double*) dw->device().data();
	for (size_t i = 0, n = weight_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw[i], gwptr[i]);
	}

	{
		auto gotshape = db->shape();
		ASSERT_ARREQ(bias_shape, gotshape);
	}
	double* gbptr = (double*) db->device().data();
	for (size_t i = 0, n = bias_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb[i], gbptr[i]);
	}

	{
		auto gotshape = dstate->shape();
		ASSERT_ARREQ(state_shape, gotshape);
	}
	double* gstateptr = (double*) dstate->device().data();
	for (size_t i = 0, n = state_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gstate[i], gstateptr[i]);
	}
}


static void tanh_RNN_layer (TensProcF root_proc = TensProcF())
{
	eigen::Device device;
	teq::Shape in_shape({5, 3});
	teq::Shape weight_shape({5, 10});
	teq::Shape bias_shape({5});
	teq::Shape state_shape({5});
	teq::Shape out_shape({5,3});

	std::vector<double> in_data = {
		0.8575073725, 0.0910915775, 0.9133499042, 0.0660953837, 0.2419061306,
		0.3696410139, 0.4013100896, 0.5172528430, 0.1323293907, 0.4278464745,
		0.0410668952, 0.1652450001, 0.4190357348, 0.2008750679, 0.5067047954
	};
	std::vector<double> weight_data = {
		0.1613409462, 0.9457144276, 0.9495257985, 0.2793930966, 0.2723075870,
		0.3588235299, 0.3938297525, 0.3393228095, 0.5716848928, 0.6339794570,
		0.6139023931, 0.7724697132, 0.0698909799, 0.6535996814, 0.2244414703,
		0.4194435958, 0.6321126915, 0.2891770970, 0.6457218252, 0.3446479912,
		0.3171555503, 0.2252455176, 0.7602351414, 0.9312997376, 0.1333143817,
		0.7155225995, 0.2032897111, 0.0224006501, 0.9908721456, 0.0319914474,
		0.9704203846, 0.5274515737, 0.3339836660, 0.7091134065, 0.5576000673,
		0.7501829168, 0.2442227058, 0.1842266311, 0.8504773433, 0.3926588922,
		0.2833117224, 0.9620642436, 0.1147953593, 0.4177183136, 0.2914940248,
		0.0219832027, 0.4042951820, 0.3837337063, 0.5981982488, 0.1894350758,
	};
	std::vector<double> bias_data = {
		0.8801580962, 0.5008790402, 0.8270796004, 0.7715771391, 0.2662051941,
	};
	std::vector<double> state_data = {
		0.1993173272, 0.6008457459, 0.3355862244, 0.1906307583, 0.3078908360,
	};
	std::vector<double> out_data = {
		0.5301287168, 0.0816631236, 0.4232512930, 0.1983706018, 0.2941365700,
		0.9427764606, 0.0267634765, 0.4367877602, 0.1155584527, 0.6275693090,
		0.4350741570, 0.3949956178, 0.2341486792, 0.1348473539, 0.8681677362,
	};
	std::vector<double> expect_gw = {
		0.012832730484849566, 0.020665413892638967, 0.044297974765077949, 0.011199947435827553, 0.23221427250804694,
		0.0015635327458991598, 0.0041422544430273836, 0.019402241314062488, 0.0012707560367359271, 0.037639635777849893,
		0.014074278095615077, 0.024083876937220847, 0.067053962426625174, 0.012012401554196661, 0.25574625718565897,
		0.0012062411866441238, 0.0028174985838895759, 0.014657480646437735, 0.0009127396798470507, 0.023503570273553907,
		0.0041692909406491705, 0.0091141905411763757, 0.041922218230865907, 0.0032927494952498682, 0.081469894232265208,
		0.0041029832998777625, 0.012258101019880084, 0.074315292906842023, 0.0029076822315577708, 0.093741511447534259,
		0.010076544707145474, 0.021262033306889326, 0.090722915084789413, 0.0081235274938007002, 0.19704566770323834,
		0.0061205009797556456, 0.015240004827907053, 0.07927892619705644, 0.0046747767153010418, 0.12836943305677537,
		0.0039744284304135055, 0.012078611732748932, 0.074056104607064382, 0.0027954950855819194, 0.09163377907594053,
		0.0056874555383919466, 0.014238599249320296, 0.075673675846280597, 0.0042988939595802692, 0.11820195174875535,
	};
	std::vector<double> expect_gb = {
		0.016016579039489543, 0.030232782277125642, 0.10717897974006134, 0.013309371208968232, 0.29988019481964079,
	};
	std::vector<double> expect_gstate = {
		0.037213495617765324, 0.19248718495392608, 0.13617200776678728, 0.11084523350674103, 0.081535346039776163,
	};

	eteq::EVariable<double> in =
		eteq::make_variable<double>(in_data.data(), in_shape);
	eteq::EVariable<double> weight =
		eteq::make_variable<double>(weight_data.data(), weight_shape);
	eteq::EVariable<double> bias =
		eteq::make_variable<double>(bias_data.data(), bias_shape);
	eteq::EVariable<double> istate =
		eteq::make_variable<double>(state_data.data(), state_shape);
	eteq::EVariable<double> out =
		eteq::make_variable<double>(out_data.data(), out_shape);

	teq::RankT seq_dim = 1;
	eteq::ETensor<double> cell_in(eteq::make_variable_scalar<double>(0, teq::Shape({10})));
	auto cell = tenncor<double>().nn.dense(cell_in, weight, bias);

	auto state = tenncor<double>().extend_like(istate,
		tenncor<double>().slice(in, 0, 1, seq_dim));

	auto output = tenncor<double>().nn.rnn(in, state, cell,
		[](const eteq::ETensor<double>& x)
		{ return tenncor<double>().tanh(x); }, seq_dim);

	auto err = tenncor<double>().pow(out - output, 2.);

	auto ders = tcr::derive(err, {weight, bias, istate});
	auto dw = ders[0];
	auto db = ders[1];
	auto dstate = ders[2];

	teq::TensptrsT roots = {dw, db, dstate};
	if (root_proc)
	{
		root_proc(roots);
	}

	teq::Evaluator eval;
	teq::TensSetT targets;
	std::transform(roots.begin(), roots.end(),
		std::inserter(targets, targets.end()),
		[](teq::TensptrT g) { return g.get(); });
	eval.evaluate(device, targets);

	{
		auto gotshape = dw->shape();
		ASSERT_ARREQ(weight_shape, gotshape);
	}
	double* gwptr = (double*) dw->device().data();
	for (size_t i = 0, n = weight_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw[i], gwptr[i]);
	}

	{
		auto gotshape = db->shape();
		ASSERT_ARREQ(bias_shape, gotshape);
	}
	double* gbptr = (double*) db->device().data();
	for (size_t i = 0, n = bias_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb[i], gbptr[i]);
	}

	{
		auto gotshape = dstate->shape();
		ASSERT_ARREQ(state_shape, gotshape);
	}
	double* gstateptr = (double*) dstate->device().data();
	for (size_t i = 0, n = state_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gstate[i], gstateptr[i]);
	}
}


static void tanh_RNN_layer_connect (TensProcF root_proc = TensProcF())
{
	eigen::Device device;
	teq::Shape in_shape({5, 3});
	teq::Shape weight_shape({5, 10});
	teq::Shape bias_shape({5});
	teq::Shape state_shape({5});
	teq::Shape out_shape({5,3});

	std::vector<double> in_data = {
		0.8575073725, 0.0910915775, 0.9133499042, 0.0660953837, 0.2419061306,
		0.3696410139, 0.4013100896, 0.5172528430, 0.1323293907, 0.4278464745,
		0.0410668952, 0.1652450001, 0.4190357348, 0.2008750679, 0.5067047954
	};
	std::vector<double> weight_data = {
		0.1613409462, 0.9457144276, 0.9495257985, 0.2793930966, 0.2723075870,
		0.3588235299, 0.3938297525, 0.3393228095, 0.5716848928, 0.6339794570,
		0.6139023931, 0.7724697132, 0.0698909799, 0.6535996814, 0.2244414703,
		0.4194435958, 0.6321126915, 0.2891770970, 0.6457218252, 0.3446479912,
		0.3171555503, 0.2252455176, 0.7602351414, 0.9312997376, 0.1333143817,
		0.7155225995, 0.2032897111, 0.0224006501, 0.9908721456, 0.0319914474,
		0.9704203846, 0.5274515737, 0.3339836660, 0.7091134065, 0.5576000673,
		0.7501829168, 0.2442227058, 0.1842266311, 0.8504773433, 0.3926588922,
		0.2833117224, 0.9620642436, 0.1147953593, 0.4177183136, 0.2914940248,
		0.0219832027, 0.4042951820, 0.3837337063, 0.5981982488, 0.1894350758,
	};
	std::vector<double> bias_data = {
		0.8801580962, 0.5008790402, 0.8270796004, 0.7715771391, 0.2662051941,
	};
	std::vector<double> state_data = {
		0.1993173272, 0.6008457459, 0.3355862244, 0.1906307583, 0.3078908360,
	};
	std::vector<double> out_data = {
		0.5301287168, 0.0816631236, 0.4232512930, 0.1983706018, 0.2941365700,
		0.9427764606, 0.0267634765, 0.4367877602, 0.1155584527, 0.6275693090,
		0.4350741570, 0.3949956178, 0.2341486792, 0.1348473539, 0.8681677362,
	};
	std::vector<double> expect_gw = {
		0.012832730484849566, 0.020665413892638967, 0.044297974765077949, 0.011199947435827553, 0.23221427250804694,
		0.0015635327458991598, 0.0041422544430273836, 0.019402241314062488, 0.0012707560367359271, 0.037639635777849893,
		0.014074278095615077, 0.024083876937220847, 0.067053962426625174, 0.012012401554196661, 0.25574625718565897,
		0.0012062411866441238, 0.0028174985838895759, 0.014657480646437735, 0.0009127396798470507, 0.023503570273553907,
		0.0041692909406491705, 0.0091141905411763757, 0.041922218230865907, 0.0032927494952498682, 0.081469894232265208,
		0.0041029832998777625, 0.012258101019880084, 0.074315292906842023, 0.0029076822315577708, 0.093741511447534259,
		0.010076544707145474, 0.021262033306889326, 0.090722915084789413, 0.0081235274938007002, 0.19704566770323834,
		0.0061205009797556456, 0.015240004827907053, 0.07927892619705644, 0.0046747767153010418, 0.12836943305677537,
		0.0039744284304135055, 0.012078611732748932, 0.074056104607064382, 0.0027954950855819194, 0.09163377907594053,
		0.0056874555383919466, 0.014238599249320296, 0.075673675846280597, 0.0042988939595802692, 0.11820195174875535,
	};
	std::vector<double> expect_gb = {
		0.016016579039489543, 0.030232782277125642, 0.10717897974006134, 0.013309371208968232, 0.29988019481964079,
	};
	std::vector<double> expect_gstate = {
		0.037213495617765324, 0.19248718495392608, 0.13617200776678728, 0.11084523350674103, 0.081535346039776163,
	};

	eteq::EVariable<double> in =
		eteq::make_variable<double>(in_data.data(), in_shape);
	eteq::EVariable<double> weight =
		eteq::make_variable<double>(weight_data.data(), weight_shape);
	eteq::EVariable<double> bias =
		eteq::make_variable<double>(bias_data.data(), bias_shape);
	eteq::EVariable<double> istate =
		eteq::make_variable<double>(state_data.data(), state_shape);
	eteq::EVariable<double> out =
		eteq::make_variable<double>(out_data.data(), out_shape);

	teq::RankT seq_dim = 1;
	eteq::ETensor<double> cell_in(eteq::make_variable_scalar<double>(0, teq::Shape({10})));
	auto cell = tenncor<double>().nn.dense(cell_in, weight, bias);

	auto state = tenncor<double>().extend_like(istate,
		tenncor<double>().slice(in, 0, 1, seq_dim));

	eteq::ETensor<double> layer_in(eteq::make_variable_scalar<double>(0, teq::Shape({5, 3})));
	auto layer = tenncor<double>().nn.rnn(layer_in, state, cell,
		[](const eteq::ETensor<double>& x)
		{ return tenncor<double>().tanh(x); }, seq_dim);
	auto output = layr::connect(layer, in);

	auto err = tenncor<double>().pow(out - output, 2.);

	auto ders = tcr::derive(err, {weight, bias, istate});
	auto dw = ders[0];
	auto db = ders[1];
	auto dstate = ders[2];

	teq::TensptrsT roots = {dw, db, dstate};
	if (root_proc)
	{
		root_proc(roots);
	}

	teq::Evaluator eval;
	teq::TensSetT targets;
	std::transform(roots.begin(), roots.end(),
		std::inserter(targets, targets.end()),
		[](teq::TensptrT g) { return g.get(); });
	eval.evaluate(device, targets);

	{
		auto gotshape = dw->shape();
		ASSERT_ARREQ(weight_shape, gotshape);
	}
	double* gwptr = (double*) dw->device().data();
	for (size_t i = 0, n = weight_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw[i], gwptr[i]);
	}

	{
		auto gotshape = db->shape();
		ASSERT_ARREQ(bias_shape, gotshape);
	}
	double* gbptr = (double*) db->device().data();
	for (size_t i = 0, n = bias_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb[i], gbptr[i]);
	}

	{
		auto gotshape = dstate->shape();
		ASSERT_ARREQ(state_shape, gotshape);
	}
	double* gstateptr = (double*) dstate->device().data();
	for (size_t i = 0, n = state_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gstate[i], gstateptr[i]);
	}
}


TEST(EQUATION, MatmulComplex)
{
	matmul_complex();
}


TEST(EQUATION, SlowSigmoidMLP)
{
	sigmoid_MLP_slow();
}


TEST(EQUATION, FastSigmoidMLP)
{
	sigmoid_MLP_fast();
}


TEST(EQUATION, TanhRNN)
{
	tanh_RNN();
}


TEST(EQUATION, TanhRNNLayer)
{
	tanh_RNN_layer();
}


TEST(EQUATION, TanhRNNLayerConnect)
{
	tanh_RNN_layer_connect();
}


TEST(EQUATION, OptimizedMatmulComplex)
{
	matmul_complex(
		[](teq::TensptrsT& roots)
		{
			std::ifstream file("cfg/optimizations.json");
			roots = eteq::optimize<float>(roots, file);
		});
}


TEST(EQUATION, OptimizedSlowSigmoidMLP)
{
	sigmoid_MLP_slow(
		[](teq::TensptrsT& roots)
		{
			std::ifstream file("cfg/optimizations.json");
			roots = eteq::optimize<double>(roots, file);
		});
}


TEST(EQUATION, OptimizedFastSigmoidMLP)
{
	sigmoid_MLP_fast(
		[](teq::TensptrsT& roots)
		{
			std::ifstream file("cfg/optimizations.json");
			roots = eteq::optimize<double>(roots, file);
		});
}


TEST(EQUATION, OptimizedTanhRNN)
{
	tanh_RNN(
		[](teq::TensptrsT& roots)
		{
			std::ifstream file("cfg/optimizations.json");
			roots = eteq::optimize<double>(roots, file);
		});
}


TEST(EQUATION, OptimizedTanhRNNLayer)
{
	tanh_RNN_layer(
		[](teq::TensptrsT& roots)
		{
			std::ifstream file("cfg/optimizations.json");
			roots = eteq::optimize<double>(roots, file);
		});
}


#endif // DISABLE_EQUATION_TEST
