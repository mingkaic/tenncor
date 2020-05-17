
#ifndef DISABLE_API_TEST

#include "dbg/print/teq.hpp"

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "layr/layer.hpp"

#include "generated/api.hpp"


template <typename T=float>
static eteq::ETensor<T> tc_tanh (const eteq::ETensor<T>& x)
{
	return tenncor<T>().tanh(x);
}


template <typename T=float>
static eteq::ETensor<T> tc_sigmoid (const eteq::ETensor<T>& x)
{
	return tenncor<T>().sigmoid(x);
}


TEST(DENSE, Connection)
{
	teq::Shape shape({6});
	teq::Shape shape2({7});
	auto biased_dense = tenncor<float>().layer.dense(shape, {5}, tenncor<float>().layer.unif_xavier_init(2), tenncor<float>().layer.unif_xavier_init(4));
	auto dense = tenncor<float>().layer.dense(shape2, {6}, tenncor<float>().layer.unif_xavier_init(3), layr::InitF<float>());

	auto x = eteq::make_variable_scalar<float>(
		0, teq::Shape({6, 2}), "x");
	auto x2 = eteq::make_variable_scalar<float>(
		0, teq::Shape({7, 2}), "x2");
	auto biasedy = eteq::connect(biased_dense, eteq::ETensor<float>(x));
	auto y = eteq::connect(dense, eteq::ETensor<float>(x2));

	EXPECT_GRAPHEQ(
		"(IDENTITY[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(variable:bias[5\\1\\1\\1\\1\\1\\1\\1])", biasedy);

	EXPECT_GRAPHEQ(
		"(IDENTITY[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:x2[7\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])", y);
}


TEST(CONV, Connection)
{
	auto conv = tenncor<float>().layer.conv({6, 5}, 4, 3,
		tenncor<float>().layer.unif_xavier_init(1), tenncor<float>().layer.zero_init());

	auto x = eteq::make_variable_scalar<float>(
		0, teq::Shape({4, 10, 9, 2}), "x");
	auto y = eteq::connect(conv, eteq::ETensor<float>(x));

	EXPECT_GRAPHEQ(
		"(IDENTITY[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_`--(ADD[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_____`--(PERMUTE[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_____|___`--(CONV[1\\6\\4\\2\\3\\1\\1\\1])\n"
		"_____|_______`--(PAD[4\\10\\9\\2\\5\\1\\1\\1])\n"
		"_____|_______|___`--(variable:x[4\\10\\9\\2\\1\\1\\1\\1])\n"
		"_____|_______`--(REVERSE[3\\4\\5\\6\\1\\1\\1\\1])\n"
		"_____|___________`--(variable:weight[3\\4\\5\\6\\1\\1\\1\\1])\n"
		"_____`--(EXTEND[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_________`--(variable:bias[3\\1\\1\\1\\1\\1\\1\\1])", y);
}


TEST(RBM, Connection)
{
	auto rrbm = tenncor<float>().layer.rbm(6, 5, tenncor<float>().layer.unif_xavier_init(2), tenncor<float>().layer.unif_xavier_init(4));
	auto nobias = tenncor<float>().layer.rbm(7, 6, tenncor<float>().layer.unif_xavier_init(3), layr::InitF<float>());

	auto x = eteq::make_variable_scalar<float>(0, teq::Shape({6, 2}), "x");
	auto x2 = eteq::make_variable_scalar<float>(0, teq::Shape({7, 2}), "x2");
	auto biasedy = rrbm.connect(eteq::ETensor<float>(x));
	auto y = nobias.connect(eteq::ETensor<float>(x2));

	EXPECT_GRAPHEQ(
		"(IDENTITY[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(variable:hbias[5\\1\\1\\1\\1\\1\\1\\1])", biasedy);

	EXPECT_GRAPHEQ(
		"(IDENTITY[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:x2[7\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])", y);
}


TEST(RBM, BackwardConnection)
{
	auto rrbm = tenncor<float>().layer.rbm(6, 5, tenncor<float>().layer.unif_xavier_init(2), tenncor<float>().layer.unif_xavier_init(4));
	auto nobias = tenncor<float>().layer.rbm(7, 6, tenncor<float>().layer.unif_xavier_init(3), layr::InitF<float>());

	auto y = eteq::make_variable_scalar<float>(0, teq::Shape({5, 2}), "y");
	auto y2 = eteq::make_variable_scalar<float>(0, teq::Shape({6, 2}), "y2");
	auto biasedx = rrbm.backward_connect(eteq::ETensor<float>(y));
	auto x = nobias.backward_connect(eteq::ETensor<float>(y2));

	EXPECT_GRAPHEQ(
		"(IDENTITY[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(ADD[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:y[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(PERMUTE[6\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(variable:vbias[6\\1\\1\\1\\1\\1\\1\\1])", biasedx);

	EXPECT_GRAPHEQ(
		"(IDENTITY[7\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MATMUL[7\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:y2[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(PERMUTE[7\\6\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])", x);
}


TEST(BIND, Sigmoid)
{
	auto sgm = tenncor<float>().layer.bind(tc_sigmoid<float>);

	auto x = eteq::make_variable_scalar<float>(0, teq::Shape({6, 2}), "x");
	auto s = eteq::connect(sgm, eteq::ETensor<float>(x));

	EXPECT_GRAPHEQ(
		"(IDENTITY[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(SIGMOID[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])", s);
}


TEST(BIND, Softmax)
{
	auto sft0 = tenncor<float>().layer.bind(
		[](eteq::ETensor<float> e)
		{
			return tenncor<float>().softmax(e, 0, 1);
		});

	auto sft1 = tenncor<float>().layer.bind(
		[](eteq::ETensor<float> e)
		{
			return tenncor<float>().softmax(e, 1, 1);
		});

	auto x = eteq::make_variable_scalar<float>(0, teq::Shape({6, 2}), "x");
	auto s0 = eteq::connect(sft0, eteq::ETensor<float>(x));
	auto s1 = eteq::connect(sft1, eteq::ETensor<float>(x));

	std::string eps_str = fmts::to_string(std::numeric_limits<float>::epsilon());
	auto expect_str0 = fmts::sprintf(
		"(IDENTITY[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(DIV[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXP[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(SUB[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________`--(REDUCE_MAX[1\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______________`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(ADD[1\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________`--(REDUCE_SUM[1\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___`--(EXP[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______`--(SUB[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___________`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___________`--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______________`--(REDUCE_MAX[1\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___________________`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________`--(EXTEND[1\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________________`--(constant:%s[1\\1\\1\\1\\1\\1\\1\\1])", eps_str.c_str());
	EXPECT_GRAPHEQ(expect_str0.c_str(), s0);

	auto expect_str1 = fmts::sprintf(
		"(IDENTITY[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(DIV[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXP[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(SUB[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________`--(REDUCE_MAX[6\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______________`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(ADD[6\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________`--(REDUCE_SUM[6\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___`--(EXP[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______`--(SUB[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___________`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___________`--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______________`--(REDUCE_MAX[6\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___________________`--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________`--(EXTEND[6\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________________`--(constant:%s[1\\1\\1\\1\\1\\1\\1\\1])", eps_str.c_str());
	EXPECT_GRAPHEQ(expect_str1.c_str(), s1);
}


#include "eteq/derive.hpp"


TEST(CONNECT, TanhRNN)
{
	teq::DimT indim = 5;
	teq::DimT hidden_dim = 5;
	teq::DimT nseq = 3;
	teq::RankT seq_dim = 1;

	teq::Shape in_shape({indim, nseq});
	teq::Shape out_shape({hidden_dim, nseq});

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
	std::vector<double> out_data = {
		0.5301287168, 0.0816631236, 0.4232512930, 0.1983706018, 0.2941365700,
		0.9427764606, 0.0267634765, 0.4367877602, 0.1155584527, 0.6275693090,
		0.4350741570, 0.3949956178, 0.2341486792, 0.1348473539, 0.8681677362,
	};
	std::vector<double> expect_gw = {
		0.085070019848660153, 0.084106757077574854, 0.085229994854534433, 0.10706263702973433, 0.37718999278376386,
		0.0092439416063910557, 0.011315144404878746, 0.025297977077133268, 0.011485447192218759, 0.054619453803546977,
		0.091019369485640267, 0.091810224051278908, 0.1112291027996378, 0.11412898908540463, 0.41071017142574917,
		0.0067767202442662166, 0.0078344867525577191, 0.018279725200380872, 0.0083108915713782268, 0.035134915536813964,
		0.024555136996221253, 0.027404937669598263, 0.054906669884766883, 0.030364485243587577, 0.12378854767068947,
		0.0011526316420559575, 0.0087368408183330359, 0.069182878918053803, 0.00039262332980391891, 0.044939725802614003,
		0.0011555108243066566, 0.0088986991500643459, 0.070014352025108639, 0.00040082607577537617, 0.046170513823455217,
		0.0011438653864017708, 0.0087925479246012544, 0.069231274295017575, 0.00039593714514908173, 0.045573655614657427,
		0.0011551873658234087, 0.0088361391930704032, 0.069712215293309454, 0.00039761593235744427, 0.045677785364467934,
		0.0011003202245395223, 0.0074140274388655603, 0.061686166142778734, 0.00032703236440574104, 0.035501314547524428,
	};
	std::vector<double> expect_gb = {
		0.10027097497967834, 0.10490989625885135, 0.15745263634658757, 0.12515186011624707, 0.47144630557362527,
	};
	std::vector<double> expect_gstate = {
		0.22953228264513739,0.50078716157858993,0.38645771008832386,0.30607972361570224,0.22923229162525588,
	};

	eteq::ETensor<double> in = eteq::ETensor<double>(
		eteq::make_variable<double>(in_data.data(), in_shape));
	eteq::ETensor<double> out = eteq::ETensor<double>(
		eteq::make_variable<double>(out_data.data(), out_shape));

	auto layer = tenncor<double>().layer.rnn(indim, hidden_dim, tc_tanh<double>, nseq,
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(weight_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(bias_data.data(), bshape, label);
		}, seq_dim);

	auto output = eteq::connect(layer, in);
	auto err = tenncor<double>().pow(out - output, 2.);
	auto contents = eteq::get_storage(layer);
	auto istate = contents[0];
	auto weight = contents[1];
	auto bias = contents[2];

	auto ders = eteq::derive(err, {
		eteq::ETensor<double>(weight),
		eteq::ETensor<double>(bias),
		eteq::ETensor<double>(istate),
	});
	auto dw = ders[0];
	auto db = ders[1];
	auto dstate = ders[2];

	auto session = eigen::get_session();
	session.track({dw, db, dstate});
	session.update();

	teq::Shape weight_shape({hidden_dim, (teq::DimT) (indim + hidden_dim)});
	teq::Shape bias_shape({hidden_dim});
	teq::Shape state_shape({hidden_dim});

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


TEST(CONNECT, DenseTanhRNN)
{
	teq::DimT indim = 2;
	teq::DimT hidden_dim = 5;
	teq::DimT nseq = 3;
	teq::RankT seq_dim = 1;

	teq::Shape in_shape({indim, nseq});
	teq::Shape out_shape({hidden_dim, nseq});

	std::vector<double> in_data = {
		0.8575073725, 0.0910915775, 0.9133499042,
		0.0660953837, 0.2419061306, 0.3696410139,
	};
	std::vector<double> w0_data = {
		0.9404269220, 0.6854869637, 0.9051396941, 0.2966845031, 0.2721141527,
		0.2877237941, 0.0590600035, 0.6288776397, 0.1353232608, 0.9594369234,
	};
	std::vector<double> b0_data = {
		0.0041691705, 0.0091563757, 0.0419265907, 0.0274944982, 0.0822265208,
	};
	std::vector<double> w1_data = {
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
	std::vector<double> b1_data = {
		0.8801580962, 0.5008790402, 0.8270796004, 0.7715771391, 0.2662051941,
	};
	std::vector<double> out_data = {
		0.5301287168, 0.0816631236, 0.4232512930, 0.1983706018, 0.2941365700,
		0.9427764606, 0.0267634765, 0.4367877602, 0.1155584527, 0.6275693090,
		0.4350741570, 0.3949956178, 0.2341486792, 0.1348473539, 0.8681677362,
	};
	std::vector<double> expect_gw1 = {
		0.049900389630410401, 0.043134629489864351, 0.05285797555648715, 0.03617284781383593, 0.2838792700695687,
		0.035878337114860631, 0.030944132497350757, 0.036837212775263042, 0.026031881751108807, 0.20389925116968016,
		0.05232377191314621, 0.045438665262010451, 0.058933665258365228, 0.037858532258749841, 0.29829544278629022,
		0.017572444127394664, 0.0152354397212844, 0.019378018307741274, 0.012722815492163121, 0.10010522074665991,
		0.024329027736336044, 0.021539319694469511, 0.034305882395864032, 0.017463363791159151, 0.13993780948676982,
		0.00091782796696175781, 0.0027999491304371298, 0.03249974907538751, 0.00018135952932522994, 0.023868623395203348,
		0.00091857550272562928, 0.0028106355086095565, 0.032627233214274985, 0.00018236676976160555, 0.024150751697762103,
		0.00091605302641236059, 0.0028011497245567707, 0.032516395450974581, 0.00018168523793404587, 0.024029194721247517,
		0.00091868354346336817, 0.0028109716229057229, 0.03263113725443078, 0.00018238878516974806, 0.024153765187252693,
		0.00090545144293347965, 0.0027156113428479954, 0.031501730179127212, 0.00017415086534201485, 0.022091071752949935,
	};
	std::vector<double> expect_gb = {
		0.060157304340203853, 0.05289306523568358, 0.07868571824260423, 0.043305125046571233, 0.34490978877452949,
	};
	std::vector<double> expect_gstate = {
		0.10657964191318281, 0.30857911834829199, 0.22768647658947433, 0.18169226315642575, 0.12571020804621758,
	};
	std::vector<double> expect_gw0 = {
		0.18796026997663043, 0.25806689630382468, 0.1581797352334042, 0.18792128360176938, 0.13835987729018656,
		0.030159219560069641, 0.032969331628495899, 0.018979408422524244, 0.024328206802583421, 0.022256401135160468,
	};
	std::vector<double> expect_gb0 = {
		0.24046239606060216, 0.31253918443616258, 0.18900470202872841, 0.22825661168260342, 0.17712428377873179,
	};

	eteq::ETensor<double> in = eteq::ETensor<double>(
		eteq::make_variable<double>(in_data.data(), in_shape));
	eteq::ETensor<double> out = eteq::ETensor<double>(
		eteq::make_variable<double>(out_data.data(), out_shape));

	auto indense = tenncor<double>().layer.dense(teq::Shape({indim}), {hidden_dim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w0_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b0_data.data(), bshape, label);
		});
	auto rnn = tenncor<double>().layer.rnn(hidden_dim, hidden_dim, tc_tanh<double>, nseq,
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w1_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b1_data.data(), bshape, label);
		}, seq_dim);

	auto layer = tenncor<double>().layer.link({indense, rnn});

	auto output = eteq::connect(layer, in);

	auto err = tenncor<double>().pow(out - output, 2.);
	auto contents = eteq::get_storage(layer);
	auto weight0 = contents[0];
	auto bias0 = contents[1];
	auto istate = contents[2];
	auto weight1 = contents[3];
	auto bias1 = contents[4];

	auto ders = eteq::derive(err, {
		eteq::ETensor<double>(weight1),
		eteq::ETensor<double>(bias1),
		eteq::ETensor<double>(istate),
		eteq::ETensor<double>(weight0),
		eteq::ETensor<double>(bias0),
	});
	auto dw1 = ders[0];
	auto db1 = ders[1];
	auto dstate = ders[2];
	auto dw0 = ders[3];
	auto db0 = ders[4];

	auto session = eigen::get_session();
	session.track({dw1, db1, dstate, dw0, db0});
	session.update();

	teq::Shape weight0_shape({hidden_dim, indim});
	teq::Shape bias0_shape({hidden_dim});
	teq::Shape weight1_shape({hidden_dim, (teq::DimT) (hidden_dim + hidden_dim)});
	teq::Shape bias1_shape({hidden_dim});
	teq::Shape state_shape({hidden_dim});

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
		EXPECT_DOUBLE_EQ(expect_gb[i], gb1ptr[i]);
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
}


TEST(CONNECT, TanhRNNFull)
{
	teq::DimT indim = 2;
	teq::DimT hidden_dim = 5;
	teq::DimT outdim = 4;
	teq::DimT nseq = 3;
	teq::RankT seq_dim = 1;

	teq::Shape in_shape({indim, nseq});
	teq::Shape out_shape({outdim, nseq});

	teq::Shape weight0_shape({hidden_dim, indim});
	teq::Shape bias0_shape({hidden_dim});
	teq::Shape weight1_shape({hidden_dim, (teq::DimT) (hidden_dim + hidden_dim)});
	teq::Shape bias1_shape({hidden_dim});
	teq::Shape state_shape({hidden_dim});
	teq::Shape w2_shape({outdim, hidden_dim});
	teq::Shape b2_shape({outdim});

	std::vector<double> in_data = {
		0.8575073725, 0.0910915775, 0.9133499042,
		0.0660953837, 0.2419061306, 0.3696410139,
	};
	std::vector<double> w0_data = {
		0.9404269220, 0.6854869637, 0.9051396941, 0.2966845031, 0.2721141527,
		0.2877237941, 0.0590600035, 0.6288776397, 0.1353232608, 0.9594369234,
	};
	std::vector<double> b0_data = {
		0.0041691705, 0.0091563757, 0.0419265907, 0.0274944982, 0.0822265208,
	};
	std::vector<double> w1_data = {
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
	std::vector<double> b1_data = {
		0.8801580962, 0.5008790402, 0.8270796004, 0.7715771391, 0.2662051941,
	};
	std::vector<double> w2_data = {
		0.7450313194, 0.2106727094, 0.8636567971, 0.4304782897, 0.6046344190,
		0.0096838345, 0.9544656921, 0.6086258218, 0.7407046375, 0.9247990543,
		0.4130239902, 0.1875074841, 0.2319260665, 0.9385046446, 0.3999650547,
		0.4904871600, 0.5629671843, 0.0174601624, 0.4782879485, 0.4765812224,
	};
	std::vector<double> b2_data = {
		0.9085104622, 0.5214807886, 0.1933926621, 0.4066651695,
	};
	std::vector<double> out_data = {
		0.5301287168, 0.0816631236, 0.4232512930, 0.1983706018, 0.2941365700,
		0.9427764606, 0.0267634765, 0.4367877602, 0.1155584527, 0.6275693090,
		0.4350741570, 0.3949956178, 0.2341486792, 0.1348473539, 0.8681677362,
	};
	std::vector<double> expect_gw0 = {
		0.017129098587432032, 0.02271495324975107, 0.014818342383768853, 0.017146867194678617, 0.014496199372674969,
		0.0027372201215958491, 0.0034088222126635267, 0.001953998720557916, 0.0024538231225036343, 0.0021429985426432287,
	};
	std::vector<double> expect_gb0 = {
		0.021890340641117105, 0.028567499639729755, 0.018073378028737853, 0.021315666808427773, 0.01816358175842913,
	};
	std::vector<double> expect_gw1 = {
		0.006866504340221893, 0.0027943627333801924, 0.0062002514819004487, 0.0040350532529357342, 0.022850304055400381,
		0.0049392064548147282, 0.0020026264231124912, 0.004389219420977592, 0.0029041792192727729, 0.016282971027585223,
		0.0071933725895132247, 0.0029496556795635086, 0.0067079733900522413, 0.0042220596995113649, 0.024399853911635065,
		0.0024166050290968877, 0.00098830130014656409, 0.0022284136895266869, 0.0014189958942532937, 0.0081425182672365329,
		0.0033317241384055499, 0.0014100408149359015, 0.0035255255117461197, 0.0019455183997297459, 0.012210623594520451,
		0.00010756864961694753, 0.00022286701630212493, 0.0019371455012163212, 9.0003197771625813e-06, 0.0050897423626272776,
		0.00010823016769968674, 0.00022352720862561467, 0.0019440423683257481, 9.0367385026375925e-06, 0.0051200554258282842,
		0.00010781228023964141, 0.00022281273200559673, 0.0019375853934425779, 9.0058063548215671e-06, 0.0051004929236757278,
		0.00010824327529943547, 0.00022355381450480985, 0.0019442745238257217, 9.0378205325293071e-06, 0.0051206748617375628,
		0.00010293780880097885, 0.00021720915527101081, 0.0018815477838088686, 8.7177606634764949e-06, 0.0048761005757784913,
	};
	std::vector<double> expect_gb1 = {
		0.0082497375362176207, 0.0034522669835862574, 0.0083558142189893159, 0.0048262601729992458, 0.029422524109408855,
	};
	std::vector<double> expect_gstate = {
		0.012174941708429383, 0.028698286163359815, 0.021707405073097825, 0.015237978764397881, 0.011424172179118194,
	};
	std::vector<double> expect_gw2 = {
		0.088682361894291545, 0.14449017806526632, 0.14132598901989094, 0.23238820801990973, 0.089115904424273903,
		0.14676961041612013, 0.14216706217058461, 0.23447409853763054, 0.088590086197043574, 0.14570261064266157,
		0.14142497395155604, 0.23305411105432836, 0.089156899980222554, 0.14681920262630568, 0.14221363855027216,
		0.23455861310287623, 0.084750184403512424, 0.12981300960846326, 0.13468499073981791, 0.21739413377416406,
	};
	std::vector<double> expect_gb2 = {
		0.08945967825003534, 0.14834500492997457, 0.14278589879926953, 0.23596898799883662,
	};

	eteq::ETensor<double> in = eteq::ETensor<double>(
		eteq::make_variable<double>(in_data.data(), in_shape));
	eteq::ETensor<double> out = eteq::ETensor<double>(
		eteq::make_variable<double>(out_data.data(), out_shape));

	auto indense = tenncor<double>().layer.dense(teq::Shape({indim}), {hidden_dim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w0_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b0_data.data(), bshape, label);
		});
	auto rnn = tenncor<double>().layer.rnn(hidden_dim, hidden_dim, tc_tanh<double>, nseq,
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w1_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b1_data.data(), bshape, label);
		}, seq_dim);
	auto outdense = tenncor<double>().layer.dense(teq::Shape({hidden_dim}), {outdim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w2_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b2_data.data(), bshape, label);
		});

	auto layer = tenncor<double>().layer.link({
		indense, rnn, outdense,
		tenncor<double>().layer.bind(tc_sigmoid<double>),
	});

	auto output = eteq::connect(layer, in);

	auto err = tenncor<double>().pow(out - output, 2.);
	auto contents = eteq::get_storage(layer);
	auto weight0 = contents[0];
	auto bias0 = contents[1];
	auto istate = contents[2];
	auto weight1 = contents[3];
	auto bias1 = contents[4];
	auto weight2 = contents[5];
	auto bias2 = contents[6];

	auto ders = eteq::derive(err, {
		eteq::ETensor<double>(weight0),
		eteq::ETensor<double>(bias0),
		eteq::ETensor<double>(istate),
		eteq::ETensor<double>(weight1),
		eteq::ETensor<double>(bias1),
		eteq::ETensor<double>(weight2),
		eteq::ETensor<double>(bias2),
	});

	auto dw0 = ders[0];
	auto db0 = ders[1];
	auto dstate = ders[2];
	auto dw1 = ders[3];
	auto db1 = ders[4];
	auto dw2 = ders[5];
	auto db2 = ders[6];

	auto session = eigen::get_session();
	session.track(teq::TensptrSetT(ders.begin(), ders.end()));
	session.update();

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

	{
		auto gotshape = dstate->shape();
		ASSERT_ARREQ(state_shape, gotshape);
	}
	double* gstateptr = (double*) dstate->device().data();
	for (size_t i = 0, n = state_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gstate[i], gstateptr[i]);
	}

	{
		auto gotshape = dw2->shape();
		ASSERT_ARREQ(w2_shape, gotshape);
	}
	double* gw2ptr = (double*) dw2->device().data();
	for (size_t i = 0, n = w2_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw2[i], gw2ptr[i]);
	}

	{
		auto gotshape = db2->shape();
		ASSERT_ARREQ(b2_shape, gotshape);
	}
	double* gb2ptr = (double*) db2->device().data();
	for (size_t i = 0, n = b2_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb2[i], gb2ptr[i]);
	}
}


TEST(CONNECT, TanhRNNCrossEntropyLoss)
{
	teq::DimT indim = 2;
	teq::DimT hidden_dim = 5;
	teq::DimT outdim = 4;
	teq::DimT nseq = 3;
	teq::RankT seq_dim = 1;

	teq::Shape in_shape({indim, nseq});
	teq::Shape out_shape({outdim, nseq});

	teq::Shape weight0_shape({hidden_dim, indim});
	teq::Shape bias0_shape({hidden_dim});
	teq::Shape weight1_shape({hidden_dim, (teq::DimT) (hidden_dim + hidden_dim)});
	teq::Shape bias1_shape({hidden_dim});
	teq::Shape state_shape({hidden_dim});
	teq::Shape w2_shape({outdim, hidden_dim});
	teq::Shape b2_shape({outdim});

	std::vector<double> in_data = {
		0.8575073725, 0.0910915775, 0.9133499042,
		0.0660953837, 0.2419061306, 0.3696410139,
	};
	std::vector<double> w0_data = {
		0.9404269220, 0.6854869637, 0.9051396941, 0.2966845031, 0.2721141527,
		0.2877237941, 0.0590600035, 0.6288776397, 0.1353232608, 0.9594369234,
	};
	std::vector<double> b0_data = {
		0.0041691705, 0.0091563757, 0.0419265907, 0.0274944982, 0.0822265208,
	};
	std::vector<double> w1_data = {
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
	std::vector<double> b1_data = {
		0.8801580962, 0.5008790402, 0.8270796004, 0.7715771391, 0.2662051941,
	};
	std::vector<double> w2_data = {
		0.7450313194, 0.2106727094, 0.8636567971, 0.4304782897, 0.6046344190,
		0.0096838345, 0.9544656921, 0.6086258218, 0.7407046375, 0.9247990543,
		0.4130239902, 0.1875074841, 0.2319260665, 0.9385046446, 0.3999650547,
		0.4904871600, 0.5629671843, 0.0174601624, 0.4782879485, 0.4765812224,
	};
	std::vector<double> b2_data = {
		0.9085104622, 0.5214807886, 0.1933926621, 0.4066651695,
	};
	std::vector<double> out_data = {
		0.5301287168, 0.0816631236, 0.4232512930, 0.1983706018, 0.2941365700,
		0.9427764606, 0.0267634765, 0.4367877602, 0.1155584527, 0.6275693090,
		0.4350741570, 0.3949956178, 0.2341486792, 0.1348473539, 0.8681677362,
	};
	std::vector<double> expect_gw0 = {
		0.014797737314603383, 0.019809508094808137, 0.012726737767362032, 0.014752687006304955, 0.011969597037686727,
		0.0026054547985398307, 0.0032369703210988667, 0.0017840737119467005, 0.0022846830975954708, 0.0019490497907674786,
	};
	std::vector<double> expect_gb0 = {
		0.019413369266302032, 0.02546466622728763, 0.015743259623184538, 0.018701392526435153, 0.015372453970754069,
	};
	std::vector<double> expect_gw1 = {
		0.006165773483209569, 0.0024932906312297395, 0.0052546474592286381, 0.0028117740857448959, 0.020634876025227535,
		0.0044341797439199125, 0.0017848533968123504, 0.0036999663935924991, 0.0020236893602289213, 0.014664650340104108,
		0.0064622225808361666, 0.0026378760658597976, 0.0057445845034463528, 0.002942237157240219, 0.022153248268727936,
		0.0021706287724747911, 0.00088313094728142342, 0.0019015454558405055, 0.00098884140785094319, 0.007378993517026834,
		0.0029988663194220164, 0.0012727730653916852, 0.0031329404587705314, 0.0013560741124794087, 0.011316364028471752,
		0.00012823603513773628, 0.00025001430427735592, 0.0022493207663570915, 7.9432850991111401e-06, 0.0058169416882857729,
		0.0001290391702350124, 0.00025076252557950518, 0.0022587267588952968, 7.9804270607877205e-06, 0.0058515782857296592,
		0.00012853790049406036, 0.00024995939762490865, 0.0022509314905259422, 7.9520625123512739e-06, 0.0058292223708224875,
		0.00012905480749006285, 0.00025079237820971475, 0.0022589974112820358, 7.9813858926396374e-06, 0.0058522862180471705,
		0.00012263500847396684, 0.00024362508420165566, 0.0021770180753695298, 7.6662010440785008e-06, 0.0055728170645181407,
	};
	std::vector<double> expect_gb1 = {
		0.0074203735699537934, 0.0031060021839102378, 0.0073358040252805646, 0.0033637561369648681, 0.027077551378799015,
	};
	std::vector<double> expect_gstate = {
		0.0099140532486462554, 0.024476364847262143, 0.018280014646555043, 0.012976170907960768, 0.0092840403627420694,
	};
	std::vector<double> expect_gw2 = {
		0.16460665311080808, 0.092681953727134239, 0.16537499975838424, 0.14406101700467783, 0.16533575492277533,
		0.094132729700748902, 0.1662787705726147, 0.14528967385955674, 0.16436766188618063, 0.093447823031179683,
		0.16542183371112462, 0.14441569673390947, 0.1654130013359732, 0.094164960304463641, 0.16633405194255468,
		0.14534311208074319, 0.15769762100726498, 0.083319886475115318, 0.15802919419189415, 0.13509635539524256,
	};
	std::vector<double> expect_gb2 = {
		0.1659255387826683, 0.095136378429847401, 0.16695085289770198, 0.14617531983628954,
	};

	eteq::ETensor<double> in = eteq::ETensor<double>(
		eteq::make_variable<double>(in_data.data(), in_shape));
	eteq::ETensor<double> out = eteq::ETensor<double>(
		eteq::make_variable<double>(out_data.data(), out_shape));

	auto indense = tenncor<double>().layer.dense(teq::Shape({indim}), {hidden_dim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w0_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b0_data.data(), bshape, label);
		});
	auto rnn = tenncor<double>().layer.rnn(hidden_dim, hidden_dim, tc_tanh<double>, nseq,
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w1_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b1_data.data(), bshape, label);
		}, seq_dim);
	auto outdense = tenncor<double>().layer.dense(teq::Shape({hidden_dim}), {outdim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w2_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b2_data.data(), bshape, label);
		});

	auto layer = tenncor<double>().layer.link({
		indense, rnn, outdense,
		tenncor<double>().layer.bind(tc_sigmoid<double>),
	});

	auto output = eteq::connect(layer, in);

	double epsilon = 1e-5;
	auto common = output + epsilon;
	auto err = tenncor<double>().reduce_mean(-(out * tenncor<double>().log(common) + (1. - out) * tenncor<double>().log(1. - common)));

	auto contents = eteq::get_storage(layer);
	auto weight0 = contents[0];
	auto bias0 = contents[1];
	auto istate = contents[2];
	auto weight1 = contents[3];
	auto bias1 = contents[4];
	auto weight2 = contents[5];
	auto bias2 = contents[6];

	auto ders = eteq::derive(err, {
		eteq::ETensor<double>(weight0),
		eteq::ETensor<double>(bias0),
		eteq::ETensor<double>(istate),
		eteq::ETensor<double>(weight1),
		eteq::ETensor<double>(bias1),
		eteq::ETensor<double>(weight2),
		eteq::ETensor<double>(bias2)
	});

	auto dw0 = ders[0];
	auto db0 = ders[1];
	auto dstate = ders[2];
	auto dw1 = ders[3];
	auto db1 = ders[4];
	auto dw2 = ders[5];
	auto db2 = ders[6];

	auto session = eigen::get_session();
	session.track(teq::TensptrSetT(ders.begin(), ders.end()));
	session.update();

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

	{
		auto gotshape = dstate->shape();
		ASSERT_ARREQ(state_shape, gotshape);
	}
	double* gstateptr = (double*) dstate->device().data();
	for (size_t i = 0, n = state_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gstate[i], gstateptr[i]);
	}

	{
		auto gotshape = dw2->shape();
		ASSERT_ARREQ(w2_shape, gotshape);
	}
	double* gw2ptr = (double*) dw2->device().data();
	for (size_t i = 0, n = w2_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw2[i], gw2ptr[i]);
	}

	{
		auto gotshape = db2->shape();
		ASSERT_ARREQ(b2_shape, gotshape);
	}
	double* gb2ptr = (double*) db2->device().data();
	for (size_t i = 0, n = b2_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb2[i], gb2ptr[i]);
	}
}


TEST(CONNECT, TanhRNNTraining)
{
	teq::DimT indim = 2;
	teq::DimT hidden_dim = 5;
	teq::DimT outdim = 4;
	teq::DimT nseq = 3;
	teq::RankT seq_dim = 1;

	teq::Shape in_shape({indim, nseq});
	teq::Shape out_shape({outdim, nseq});

	double lmbd = 0.5;
	double learning_rate = 0.05;
	double momentum_term = 0.80;
	double eps = 1e-6;

	teq::Shape w0_shape({hidden_dim, indim});
	teq::Shape b0_shape({hidden_dim});
	teq::Shape w1_shape({hidden_dim, (teq::DimT) (hidden_dim + hidden_dim)});
	teq::Shape b1_shape({hidden_dim});
	teq::Shape state_shape({hidden_dim});
	teq::Shape w2_shape({outdim, hidden_dim});
	teq::Shape b2_shape({outdim});

	std::vector<double> in_data = {
		0.8575073725, 0.0910915775, 0.9133499042,
		0.0660953837, 0.2419061306, 0.3696410139,
	};
	std::vector<double> w0_data = {
		0.9404269220, 0.6854869637, 0.9051396941, 0.2966845031, 0.2721141527,
		0.2877237941, 0.0590600035, 0.6288776397, 0.1353232608, 0.9594369234,
	};
	std::vector<double> b0_data = {
		0.0041691705, 0.0091563757, 0.0419265907, 0.0274944982, 0.0822265208,
	};
	std::vector<double> w1_data = {
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
	std::vector<double> b1_data = {
		0.8801580962, 0.5008790402, 0.8270796004, 0.7715771391, 0.2662051941,
	};
	std::vector<double> w2_data = {
		0.7450313194, 0.2106727094, 0.8636567971, 0.4304782897, 0.6046344190,
		0.0096838345, 0.9544656921, 0.6086258218, 0.7407046375, 0.9247990543,
		0.4130239902, 0.1875074841, 0.2319260665, 0.9385046446, 0.3999650547,
		0.4904871600, 0.5629671843, 0.0174601624, 0.4782879485, 0.4765812224,
	};
	std::vector<double> b2_data = {
		0.9085104622, 0.5214807886, 0.1933926621, 0.4066651695,
	};
	std::vector<double> out_data = {
		0.5301287168, 0.0816631236, 0.4232512930, 0.1983706018, 0.2941365700,
		0.9427764606, 0.0267634765, 0.4367877602, 0.1155584527, 0.6275693090,
		0.4350741570, 0.3949956178, 0.2341486792, 0.1348473539, 0.8681677362,
	};
	std::vector<double> state_data = {
		0, 0, 0, 0, 0
	};
	std::vector<std::vector<double>> expect_der1 {
		{
			0.014797737314603383, 0.019809508094808137, 0.012726737767362032, 0.014752687006304955, 0.011969597037686727,
			0.0026054547985398307, 0.0032369703210988667, 0.0017840737119467005, 0.0022846830975954708, 0.0019490497907674786,
		},
		{
			0.019413369266302032, 0.02546466622728763, 0.015743259623184538, 0.018701392526435153, 0.015372453970754069,
		},
		{
			0.006165773483209569, 0.0024932906312297395, 0.0052546474592286381, 0.0028117740857448959, 0.020634876025227535,
			0.0044341797439199125, 0.0017848533968123504, 0.0036999663935924991, 0.0020236893602289213, 0.014664650340104108,
			0.0064622225808361666, 0.0026378760658597976, 0.0057445845034463528, 0.002942237157240219, 0.022153248268727936,
			0.0021706287724747911, 0.00088313094728142342, 0.0019015454558405055, 0.00098884140785094319, 0.007378993517026834,
			0.0029988663194220164, 0.0012727730653916852, 0.0031329404587705314, 0.0013560741124794087, 0.011316364028471752,
			0.00012823603513773628, 0.00025001430427735592, 0.0022493207663570915, 7.9432850991111401e-06, 0.0058169416882857729,
			0.0001290391702350124, 0.00025076252557950518, 0.0022587267588952968, 7.9804270607877205e-06, 0.0058515782857296592,
			0.00012853790049406036, 0.00024995939762490865, 0.0022509314905259422, 7.9520625123512739e-06, 0.0058292223708224875,
			0.00012905480749006285, 0.00025079237820971475, 0.0022589974112820358, 7.9813858926396374e-06, 0.0058522862180471705,
			0.00012263500847396684, 0.00024362508420165566, 0.0021770180753695298, 7.6662010440785008e-06, 0.0055728170645181407,
		},
		{
			0.0074203735699537934, 0.0031060021839102378, 0.0073358040252805646, 0.0033637561369648681, 0.027077551378799015,
		},
		{
			0.0099140532486462554, 0.024476364847262143, 0.018280014646555043, 0.012976170907960768, 0.0092840403627420694,
		},
		{
			0.16460665311080808, 0.092681953727134239, 0.16537499975838424, 0.14406101700467783, 0.16533575492277533,
			0.094132729700748902, 0.1662787705726147, 0.14528967385955674, 0.16436766188618063, 0.093447823031179683,
			0.16542183371112462, 0.14441569673390947, 0.1654130013359732, 0.094164960304463641, 0.16633405194255468,
			0.14534311208074319, 0.15769762100726498, 0.083319886475115318, 0.15802919419189415, 0.13509635539524256,
		},
		{
			0.1659255387826683, 0.095136378429847401, 0.16695085289770198, 0.14617531983628954,
		},
	};
	std::vector<std::vector<double>> expect_g1 = {
		{
			0.940426922, 0.68548696369999995, 0.90513969409999995, 0.2966845031, 0.2721141527,
			0.28772379409999999, 0.0590600035, 0.62887763969999999, 0.13532326080000001, 0.95943692339999997,
		},
		{
			0.0041691705000000004, 0.0091563757000000003, 0.041926590700000002, 0.027494498199999998, 0.082226520799999994,
		},
		{
			0.1613409462, 0.94571442760000002, 0.94952579849999996, 0.2793930966, 0.27230758700000002,
			0.35882352989999999, 0.39382975250000002, 0.33932280949999999, 0.57168489280000001, 0.63397945700000002,
			0.61390239310000005, 0.77246971320000002, 0.069890979899999997, 0.65359968140000002, 0.2244414703,
			0.41944359580000001, 0.63211269150000005, 0.28917709699999999, 0.64572182519999999, 0.3446479912,
			0.31715555029999998, 0.2252455176, 0.76023514140000004, 0.93129973759999995, 0.13331438170000001,
			0.71552259949999997, 0.20328971109999999, 0.022400650099999999, 0.99087214560000003, 0.031991447399999998,
			0.97042038460000002, 0.52745157369999995, 0.33398366600000001, 0.70911340649999999, 0.5576000673,
			0.75018291680000004, 0.2442227058, 0.18422663110000001, 0.85047734330000002, 0.3926588922,
			0.28331172240000002, 0.96206424359999998, 0.1147953593, 0.4177183136, 0.29149402479999997,
			0.021983202699999999, 0.40429518199999998, 0.38373370629999998, 0.59819824880000005, 0.18943507579999999,
		},
		{
			0.88015809619999996, 0.50087904019999996, 0.82707960039999995, 0.77157713910000003, 0.26620519409999999,
		},
		{
			0, 0, 0, 0, 0,
		},
		{
			0.7450313194, 0.21067270939999999, 0.86365679709999998, 0.43047828970000002, 0.60463441900000003,
			0.0096838345000000003, 0.95446569209999998, 0.60862582180000002, 0.74070463750000004, 0.92479905429999998,
			0.41302399020000002, 0.18750748410000001, 0.2319260665, 0.93850464460000005, 0.39996505469999999,
			0.49048715999999998, 0.56296718430000003, 0.0174601624, 0.4782879485, 0.4765812224,
		},
		{
			0.90851046219999998, 0.52148078860000002, 0.1933926621, 0.40666516949999998,
		},
	};
	std::vector<std::vector<double>> expect_der2 = {
		{
			0.014797737314603383, 0.019809508094808137, 0.012726737767362032, 0.014752687006304955, 0.011969597037686727,
			0.0026054547985398307, 0.0032369703210988667, 0.0017840737119467005, 0.0022846830975954708, 0.0019490497907674786,
		},
		{
			0.019413369266302032, 0.02546466622728763, 0.015743259623184538, 0.018701392526435153, 0.015372453970754069,
		},
		{
			0.006165773483209569, 0.0024932906312297395, 0.0052546474592286381, 0.0028117740857448959, 0.020634876025227535,
			0.0044341797439199125, 0.0017848533968123504, 0.0036999663935924991, 0.0020236893602289213, 0.014664650340104108,
			0.0064622225808361666, 0.0026378760658597976, 0.0057445845034463528, 0.002942237157240219, 0.022153248268727936,
			0.0021706287724747911, 0.00088313094728142342, 0.0019015454558405055, 0.00098884140785094319, 0.007378993517026834,
			0.0029988663194220164, 0.0012727730653916852, 0.0031329404587705314, 0.0013560741124794087, 0.011316364028471752,
			0.00012823603513773628, 0.00025001430427735592, 0.0022493207663570915, 7.9432850991111401e-06, 0.0058169416882857729,
			0.0001290391702350124, 0.00025076252557950518, 0.0022587267588952968, 7.9804270607877205e-06, 0.0058515782857296592,
			0.00012853790049406036, 0.00024995939762490865, 0.0022509314905259422, 7.9520625123512739e-06, 0.0058292223708224875,
			0.00012905480749006285, 0.00025079237820971475, 0.0022589974112820358, 7.9813858926396374e-06, 0.0058522862180471705,
			0.00012263500847396684, 0.00024362508420165566, 0.0021770180753695298, 7.6662010440785008e-06, 0.0055728170645181407,
		},
		{
			0.0074203735699537934, 0.0031060021839102378, 0.0073358040252805646, 0.0033637561369648681, 0.027077551378799015,
		},
		{
			0.0099140532486462554, 0.024476364847262143, 0.018280014646555043, 0.012976170907960768, 0.0092840403627420694,
		},
		{
			0.16460665311080808, 0.092681953727134239, 0.16537499975838424, 0.14406101700467783, 0.16533575492277533,
			0.094132729700748902, 0.1662787705726147, 0.14528967385955674, 0.16436766188618063, 0.093447823031179683,
			0.16542183371112462, 0.14441569673390947, 0.1654130013359732, 0.094164960304463641, 0.16633405194255468,
			0.14534311208074319, 0.15769762100726498, 0.083319886475115318, 0.15802919419189415, 0.13509635539524256,
		},
		{
			0.1659255387826683, 0.095136378429847401, 0.16695085289770198, 0.14617531983628954,
		},
	};
	std::vector<std::vector<double>> expect_g2 = {
		{
			0.00010948651481600267, 0.00019620830547913454, 8.098492709959956e-05, 0.00010882088695299954, 7.1635626622299433e-05,
			3.3941973536171148e-06, 5.23898842983745e-06, 1.5914595048296393e-06, 2.6098884282192178e-06, 1.8993975434453761e-06,
		},
		{
			0.00018843945313490014, 0.0003242246130335816, 0.00012392511178149627, 0.0001748710412139023, 0.00011815617054147627,
		},
		{
			1.900838132312513e-05, 3.1082490858889965e-06, 1.380565996038899e-05, 3.9530367546332726e-06, 0.00021289905428825505,
			9.8309750006948308e-06, 1.5928508240562929e-06, 6.8448756568569425e-06, 2.0476593133518702e-06, 0.00010752598479875776,
			2.0880160342134423e-05, 3.4791950694179814e-06, 1.6500125558617991e-05, 4.3283797447225023e-06, 0.00024538320442794864,
			2.3558146339477092e-06, 3.8996013502309212e-07, 1.807937560313838e-06, 4.8890366494031769e-07, 2.7224772662162024e-05,
			4.4965996008818753e-06, 8.099756379932734e-07, 4.9076579591006539e-06, 9.1946849926840793e-07, 6.4030047412444707e-05,
			8.2222403539233661e-09, 3.1253576171645159e-08, 2.5297219549826267e-06, 3.1547889082880537e-11, 1.6918405302458468e-05,
			8.3255537274702558e-09, 3.1440922117505994e-08, 2.5509232856748262e-06, 3.184360803627647e-11, 1.7120484217011429e-05,
			8.2609959317104806e-09, 3.1239850230503591e-08, 2.5333462875206697e-06, 3.1617649100171226e-11, 1.6989916724248673e-05,
			8.3275716681485909e-09, 3.1448408484042301e-08, 2.5515346520894695e-06, 3.1851260383613509e-11, 1.7124626988972426e-05,
			7.5196726517049594e-09, 2.9676590826131903e-08, 2.3697038502428259e-06, 2.9385319224115146e-11, 1.5528145017292294e-05,
		},
		{
			2.7530971958834403e-05, 4.823624783227583e-06, 2.6907010348661268e-05, 5.6574276744844068e-06, 0.00036659689433575022,
		},
		{
			4.9144225908496685e-05, 0.00029954621806814496, 0.00016707946773913344, 8.4190505716303691e-05, 4.3096702728511946e-05,
		},
		{
			0.013547675124170952, 0.0042949722733393262, 0.013674445272542794, 0.010376788310211037, 0.013667955927942014,
			0.0044304854004571272, 0.013824314771570118, 0.010554544665108183, 0.013508364136964898, 0.0043662478146333384,
			0.013682191534175483, 0.010427946731570254, 0.013680730505487335, 0.0044335198745706064, 0.013833508417814239,
			0.010562310114657738, 0.01243426983567549, 0.0034711017411130523, 0.012486613108469696, 0.0091255126205388414,
		},
		{
			0.013765642210159381, 0.0045254652503735667, 0.013936293641635063, 0.010683612064620771,
		},
	};
	std::vector<std::vector<double>> expect_g3 = {
		{
			-0.070703920974516499, -0.070705630398087124, -0.070702821518564676, -0.070703900342190443, -0.070702324605496913,
			-0.070672317925052491, -0.070679798524545506, -0.070654671019098766, -0.070666935448772797, -0.070659408267082002,
		},
		{
			-0.070705527404504184, -0.070706751326691922, -0.070704326764473857, -0.070705331327588036, -0.070704173574788434,
		},
		{
			-0.070694463272453087, -0.070670593216343072, -0.070691652466707472, -0.070675131262433064, -0.070705832286441561,
			-0.070688133223662436, -0.07065469546554097, -0.070683661172651763, -0.0706612979282812, -0.070703859656917195,
			-0.070695206952315917, -0.070672789144948581, -0.070693274703700265, -0.070676706704447256, -0.07070616439608822,
			-0.070664638514170486, -0.070597625665906494, -0.070658128397497841, -0.070609694092068492, -0.070697128731730563,
			-0.070677347902063134, -0.07063219671984336, -0.070678773624602856, -0.070637012660115184, -0.070701842462233377,
			-0.069939372232220964, -0.070312950761156887, -0.070666248187591277, -0.060024061575325374, -0.070693491132900202,
			-0.069944120770051854, -0.070314130838599165, -0.07066643309092481, -0.060066311398179278, -0.070693592841335154,
			-0.069941163896234546, -0.070312863886971166, -0.070666279960850439, -0.060034076318515872, -0.070693527332706632,
			-0.069944212645139711, -0.070314177776227671, -0.070666438388642366, -0.06006739766818301, -0.070693594907587101,
			-0.069904546501233539, -0.07030258031184379, -0.070664773547506013, -0.059697965116656981, -0.070692738425267215,
		},
		{
			-0.070697204276109413, -0.070678497043700869, -0.070697048975191287, -0.070680961940992507, -0.070706985215406332,
		},
		{
			-0.070700592865449982, -0.070706592780693675, -0.070705208086702373, -0.070702972524812488, -0.070699908586675658,
		},
		{
			-0.070710070615015205, -0.070709599176421403, -0.070710073437505863, -0.070709983975156307, -0.070710073293977257,
			-0.070709615804856144, -0.070710076724086121, -0.070709989845194027, -0.070710069731710123, -0.070709608018962686,
			-0.070710073608700713, -0.070709985679930948, -0.070710073576422644, -0.070709616168457431, -0.070710076923958823,
			-0.070709990098248687, -0.07071004399936634, -0.070709477945359658, -0.070710045329848786, -0.070709937913984555,
		},
		{
			-0.070710075443813844, -0.070709627011652787, -0.070710079145057517, -0.070709994015259986,
		},
		{
			0.8697230010254835, 0.61478133330191287, 0.83443687258143528, 0.22598060275780957, 0.20141182809450309,
			0.21705147617494749, -0.011619795024545507, 0.55822296868090127, 0.064656325351227215, 0.88877751513291803,
		},
		{
			-0.066536356904504185, -0.061550375626691925, -0.028777736064473855, -0.043210833127588041, 0.01152234722521156,
		},
		{
			0.090646482927546915, 0.87504383438365696, 0.87883414603329246, 0.20871796533756692, 0.20160175471355846,
			0.28813539667633759, 0.32317505703445903, 0.26863914832734825, 0.50102359487171877, 0.56327559734308286,
			0.54320718614768415, 0.70179692405505145, -0.00080229480370026807, 0.58292297469555276, 0.15373530590391177,
			0.34877895728582953, 0.5615150658340935, 0.21851896860250214, 0.57511213110793147, 0.27395086246826944,
			0.24647820239793683, 0.15461332088015664, 0.68955636777539719, 0.86066272493988483, 0.062612539237766635,
			0.64558322726777906, 0.13297676033884309, -0.048265598087591274, 0.93084808402467467, -0.038702043732900204,
			0.90047626382994816, 0.45713744286140079, 0.26331723290907522, 0.64904709510182068, 0.48690647445866486,
			0.68024175290376554, 0.17390984191302883, 0.11356035113914957, 0.79044326698148415, 0.32196536486729338,
			0.2133675097548603, 0.89175006582377225, 0.04412892091135763, 0.35765091593181697, 0.22080042989241289,
			-0.047921343801233543, 0.3339926016881562, 0.31306893275249398, 0.53850028368334302, 0.11874233737473278,
		},
		{
			0.80946089192389059, 0.43020054315629908, 0.7563825514248087, 0.70089617715900754, 0.19549820888459365,
		},
		{
			-0.070700592865449982, -0.070706592780693675, -0.070705208086702373, -0.070702972524812488, -0.070699908586675658,
		},
		{
			0.67432124878498478, 0.1399631102235786, 0.79294672366249408, 0.3597683057248437, 0.53392434570602276,
			-0.061025781304856141, 0.88375561537591385, 0.53791583195480597, 0.66999456776828992, 0.85408944628103733,
			0.34231391659129928, 0.11679749842006906, 0.16121599292357736, 0.86779502843154266, 0.32925497777604118,
			0.41977716990175129, 0.49225714030063372, -0.053249315545359659, 0.40757790317015119, 0.40587128448601545,
		},
		{
			0.83780038675618618, 0.45077116158834724, 0.12268258295494248, 0.33595517548473997,
		},
	};

	eteq::ETensor<double> in = eteq::ETensor<double>(
		eteq::make_variable<double>(in_data.data(), in_shape, "input"));
	eteq::ETensor<double> out = eteq::ETensor<double>(
		eteq::make_variable<double>(out_data.data(), out_shape, "output"));

	auto indense = tenncor<double>().layer.dense(teq::Shape({indim}), {hidden_dim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w0_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b0_data.data(), bshape, label);
		});
	auto rnn = tenncor<double>().layer.rnn(hidden_dim, hidden_dim, tc_tanh<double>, nseq,
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w1_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b1_data.data(), bshape, label);
		}, seq_dim);
	auto outdense = tenncor<double>().layer.dense(teq::Shape({hidden_dim}), {outdim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w2_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b2_data.data(), bshape, label);
		});

	auto layer = tenncor<double>().layer.link({
		indense, rnn, outdense,
		tenncor<double>().layer.bind(tc_sigmoid<double>),
	});

	auto output = eteq::connect(layer, in);

	double epsilon = 1e-5;
	auto common = output + epsilon;
	auto err = tenncor<double>().reduce_mean(-(out * tenncor<double>().log(common) + (1. - out) * tenncor<double>().log(1. - common)));

	auto contents = eteq::get_storage(layer);
	auto weight0 = contents[0];
	auto bias0 = contents[1];
	auto istate = contents[2];
	auto weight1 = contents[3];
	auto bias1 = contents[4];
	auto weight2 = contents[5];
	auto bias2 = contents[6];

	auto ders = eteq::derive(err, {
		eteq::ETensor<double>(weight0),
		eteq::ETensor<double>(bias0),
		eteq::ETensor<double>(weight1),
		eteq::ETensor<double>(bias1),
		eteq::ETensor<double>(istate),
		eteq::ETensor<double>(weight2),
		eteq::ETensor<double>(bias2),
	});
	size_t nders = ders.size();

	eteq::VarptrsT<double> targets = {weight0, bias0, weight1, bias1, istate, weight2, bias2};
	eteq::VarptrsT<double> momentums;
	eteq::VarptrsT<double> mvavg_sqrs;
	for (auto root : ders)
	{
		momentums.push_back(eteq::make_variable_like<double>(0, root, "momentum_" + root->to_string()));
		mvavg_sqrs.push_back(eteq::make_variable_like<double>(0, root, "mvavg_sqrs_" + root->to_string()));
	}

	teq::TensptrSetT to_track;

	// group 1
	eteq::ETensorsT<double> momentum_tmps;
	for (auto mom : momentums)
	{
		momentum_tmps.push_back(eteq::ETensor<double>(mom) * momentum_term);
	}
	eteq::VarptrsT<double> group1_left;
	eteq::ETensorsT<double> group1_right;
	for (size_t i = 0; i < nders; ++i)
	{
		auto right = eteq::ETensor<double>(targets[i]) + momentum_tmps[i];
		group1_left.push_back(targets[i]);
		group1_right.push_back(right);
		to_track.emplace(right);
	}

	// group 2
	eteq::VarptrsT<double> group2_left;
	eteq::ETensorsT<double> group2_right;
	for (size_t i = 0; i < nders; ++i)
	{
		auto right = lmbd * eteq::ETensor<double>(mvavg_sqrs[i]) +
			(1. - lmbd) * tenncor<double>().pow(ders[i], 2.);
		group2_left.push_back(mvavg_sqrs[i]);
		group2_right.push_back(right);
		to_track.emplace(right);
	}

	// group 3
	eteq::VarptrsT<double> group3_left;
	eteq::ETensorsT<double> group3_right;
	{
		eteq::ETensorsT<double> pgrad_norm_nodes;
		for (size_t i = 0; i < nders; ++i)
		{
			pgrad_norm_nodes.push_back((learning_rate * ders[i]) /
				(tenncor<double>().sqrt(eteq::ETensor<double>(mvavg_sqrs[i])) + eps));
		}

		for (size_t i = 0; i < nders; ++i)
		{
			auto right = momentum_tmps[i] - pgrad_norm_nodes[i];
			group3_left.push_back(momentums[i]);
			group3_right.push_back(right);
			to_track.emplace(right);
		}

		for (size_t i = 0; i < nders; ++i)
		{
			auto right = eteq::ETensor<double>(targets[i]) - pgrad_norm_nodes[i];
			group3_left.push_back(targets[i]);
			group3_right.push_back(right);
			to_track.emplace(right);
		}
	}

	auto session = eigen::get_session();
	session.track(to_track);

	{
		teq::TensSetT rights;
		for (auto r : ders)
		{
			rights.emplace(r.get());
		}
		session.update_target(rights);

		for (size_t i = 0; i < nders; ++i)
		{
			auto left = ders[i];
			double* ptr = (double*) left->device().data();
			ASSERT_NE(nullptr, ptr);
			for (size_t j = 0, n = left->shape().n_elems(); j < n; ++j)
			{
				EXPECT_DOUBLE_EQ(expect_der1[i][j], ptr[j]);
			}
		}
	}

	// group 1
	{
		teq::TensSetT rights;
		for (auto r : group1_right)
		{
			rights.emplace(r.get());
		}
		session.update_target(rights);
		for (size_t i = 0; i < nders; ++i)
		{
			group1_left[i]->assign(*group1_right[i],
				eteq::global_context()->registry_);
		}

		for (size_t i = 0; i < nders; ++i)
		{
			auto left = group1_left[i];
			double* ptr = (double*) left->device().data();
			ASSERT_NE(nullptr, ptr);
			for (size_t j = 0, n = left->shape().n_elems(); j < n; ++j)
			{
				EXPECT_DOUBLE_EQ(expect_g1[i][j], ptr[j]);
			}
		}
	}

	{
		teq::TensSetT rights;
		for (auto r : ders)
		{
			rights.emplace(r.get());
		}
		session.update_target(rights);

		for (size_t i = 0; i < nders; ++i)
		{
			auto left = ders[i];
			double* ptr = (double*) left->device().data();
			ASSERT_NE(nullptr, ptr);
			for (size_t j = 0, n = left->shape().n_elems(); j < n; ++j)
			{
				EXPECT_DOUBLE_EQ(expect_der2[i][j], ptr[j]);
			}
		}
	}

	// group 2
	{
		teq::TensSetT rights;
		for (auto r : group2_right)
		{
			rights.emplace(r.get());
		}
		session.update_target(rights);
		for (size_t i = 0; i < nders; ++i)
		{
			group2_left[i]->assign(*group2_right[i],
				eteq::global_context()->registry_);
		}

		for (size_t i = 0; i < nders; ++i)
		{
			auto left = group2_left[i];
			double* ptr = (double*) left->device().data();
			ASSERT_NE(nullptr, ptr);
			for (size_t j = 0, n = left->shape().n_elems(); j < n; ++j)
			{
				EXPECT_DOUBLE_EQ(expect_g2[i][j], ptr[j]);
			}
		}
	}

	// group 3
	{
		teq::TensSetT rights;
		for (auto r : group3_right)
		{
			rights.emplace(r.get());
		}
		session.update_target(rights);
		for (size_t i = 0; i < group3_left.size(); ++i)
		{
			group3_left[i]->assign(*group3_right[i],
				eteq::global_context()->registry_);
		}

		for (size_t i = 0, ng3 = group3_left.size(); i < ng3; ++i)
		{
			auto left = group3_left[i];
			double* ptr = (double*) left->device().data();
			ASSERT_NE(nullptr, ptr);
			for (size_t j = 0, n = left->shape().n_elems(); j < n; ++j)
			{
				EXPECT_DOUBLE_EQ(expect_g3[i][j], ptr[j]);
			}
		}
	}
}


#endif // DISABLE_API_TEST
