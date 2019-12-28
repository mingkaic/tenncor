
#ifndef DISABLE_API_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "layr/api.hpp"


TEST(DENSE, Connection)
{
	teq::Shape shape({6});
	teq::Shape shape2({7});
	auto biased_dense = layr::dense<float>(shape, {5}, layr::unif_xavier_init<float>(2), layr::unif_xavier_init<float>(4));
	auto dense = layr::dense<float>(shape2, {6}, layr::unif_xavier_init<float>(3), layr::InitF<float>());

	auto x = eteq::make_variable_scalar<float>(
		0, teq::Shape({6, 2}), "x");
	auto x2 = eteq::make_variable_scalar<float>(
		0, teq::Shape({7, 2}), "x2");
	auto biasedy = biased_dense.connect(eteq::ETensor<float>(x));
	auto y = dense.connect(eteq::ETensor<float>(x2));

	EXPECT_GRAPHEQ(
		"(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:bias[5\\1\\1\\1\\1\\1\\1\\1])", biasedy);

	EXPECT_GRAPHEQ(
		"(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:x2[7\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])", y);
}


TEST(CONV, Connection)
{
	auto conv = layr::conv<float>({6, 5}, 4, 3);

	auto x = eteq::make_variable_scalar<float>(
		0, teq::Shape({4, 10, 9, 2}), "x");
	auto y = conv.connect(eteq::ETensor<float>(x));

	EXPECT_GRAPHEQ(
		"(ADD[3\\6\\4\\2\\1\\1\\1\\1])\n"
		" `--(PERMUTE[3\\6\\4\\2\\1\\1\\1\\1])\n"
		" |   `--(CONV[1\\6\\4\\2\\3\\1\\1\\1])\n"
		" |       `--(PAD[4\\10\\9\\2\\5\\1\\1\\1])\n"
		" |       |   `--(variable:x[4\\10\\9\\2\\1\\1\\1\\1])\n"
		" |       `--(REVERSE[3\\4\\5\\6\\1\\1\\1\\1])\n"
		" |           `--(variable:weight[3\\4\\5\\6\\1\\1\\1\\1])\n"
		" `--(EXTEND[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"     `--(variable:bias[3\\1\\1\\1\\1\\1\\1\\1])", y);
}


TEST(RBM, Connection)
{
	auto rrbm = layr::rbm<float>(6, 5, layr::unif_xavier_init<float>(2), layr::unif_xavier_init<float>(4));
	auto nobias = layr::rbm<float>(7, 6, layr::unif_xavier_init<float>(3), layr::InitF<float>());

	auto x = eteq::make_variable_scalar<float>(0, teq::Shape({6, 2}), "x");
	auto x2 = eteq::make_variable_scalar<float>(0, teq::Shape({7, 2}), "x2");
	auto biasedy = rrbm.fwd_.connect(eteq::ETensor<float>(x));
	auto y = nobias.fwd_.connect(eteq::ETensor<float>(x2));

	EXPECT_GRAPHEQ(
		"(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:hbias[5\\1\\1\\1\\1\\1\\1\\1])", biasedy);

	EXPECT_GRAPHEQ(
		"(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:x2[7\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])", y);
}


TEST(RBM, BackwardConnection)
{
	auto rrbm = layr::rbm<float>(6, 5, layr::unif_xavier_init<float>(2), layr::unif_xavier_init<float>(4));
	auto nobias = layr::rbm<float>(7, 6, layr::unif_xavier_init<float>(3), layr::InitF<float>());

	auto y = eteq::make_variable_scalar<float>(0, teq::Shape({5, 2}), "y");
	auto y2 = eteq::make_variable_scalar<float>(0, teq::Shape({6, 2}), "y2");
	auto biasedx = rrbm.bwd_.connect(eteq::ETensor<float>(y));
	auto x = nobias.bwd_.connect(eteq::ETensor<float>(y2));

	EXPECT_GRAPHEQ(
		"(ADD[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:y[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(PERMUTE[6\\5\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:vbias[6\\1\\1\\1\\1\\1\\1\\1])", biasedx);

	EXPECT_GRAPHEQ(
		"(MATMUL[7\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:y2[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(PERMUTE[7\\6\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])", x);
}


TEST(BIND, Sigmoid)
{
	auto sgm = layr::bind<float>(tenncor::sigmoid<float>);

	auto x = eteq::make_variable_scalar<float>(0, teq::Shape({6, 2}), "x");
	auto s = sgm.connect(eteq::ETensor<float>(x));

	EXPECT_GRAPHEQ(
		"(SIGMOID[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])", s);
}


TEST(BIND, Softmax)
{
	auto sft0 = layr::bind<float>(
		[](eteq::ETensor<float> e)
		{
			return tenncor::softmax(e, 0, 1);
		});

	auto sft1 = layr::bind<float>(
		[](eteq::ETensor<float> e)
		{
			return tenncor::softmax(e, 1, 1);
		});

	auto x = eteq::make_variable_scalar<float>(0, teq::Shape({6, 2}), "x");
	auto s0 = sft0.connect(eteq::ETensor<float>(x));
	auto s1 = sft1.connect(eteq::ETensor<float>(x));

	std::string eps_str = fmts::to_string(std::numeric_limits<float>::epsilon());
	auto expect_str0 = fmts::sprintf(
		"(DIV[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXP[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(SUB[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |           `--(REDUCE_MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |               `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(ADD[1\\2\\1\\1\\1\\1\\1\\1])\n"
		"         `--(REDUCE_SUM[1\\2\\1\\1\\1\\1\\1\\1])\n"
		"         |   `--(EXP[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         |       `--(SUB[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         |           `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         |           `--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         |               `--(REDUCE_MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |                   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         `--(EXTEND[1\\2\\1\\1\\1\\1\\1\\1])\n"
		"             `--(constant:%s[1\\1\\1\\1\\1\\1\\1\\1])", eps_str.c_str());
	EXPECT_GRAPHEQ(expect_str0.c_str(), s0);

	auto expect_str1 = fmts::sprintf(
		"(DIV[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXP[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(SUB[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |           `--(REDUCE_MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |               `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(ADD[6\\1\\1\\1\\1\\1\\1\\1])\n"
		"         `--(REDUCE_SUM[6\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |   `--(EXP[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         |       `--(SUB[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         |           `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         |           `--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         |               `--(REDUCE_MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |                   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         `--(EXTEND[6\\1\\1\\1\\1\\1\\1\\1])\n"
		"             `--(constant:%s[1\\1\\1\\1\\1\\1\\1\\1])", eps_str.c_str());
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

	auto layer = layr::rnn<double>(indim, hidden_dim, tenncor::tanh<double>, nseq,
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(weight_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(bias_data.data(), bshape, label);
		}, seq_dim);

	auto output = layer.connect(in);

	auto err = tenncor::pow(out - output, 2.);
	auto contents = layer.get_storage();
	auto istate = contents[0];
	auto weight = contents[1];
	auto bias = contents[2];

	auto dw = eteq::derive(err, eteq::ETensor<double>(weight));
	auto db = eteq::derive(err, eteq::ETensor<double>(bias));
	auto dstate = eteq::derive(err, eteq::ETensor<double>(istate));

	teq::TensptrsT roots = {dw, db, dstate};
	teq::Session session;
	session.track(roots);
	session.update();

	teq::Shape weight_shape({hidden_dim, (teq::DimT) (indim + hidden_dim)});
	teq::Shape bias_shape({hidden_dim});
	teq::Shape state_shape({hidden_dim});

	{
		auto gotshape = dw->shape();
		ASSERT_ARREQ(weight_shape, gotshape);
	}
	double* gwptr = (double*) dw->data();
	for (size_t i = 0, n = weight_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw[i], gwptr[i]);
	}

	{
		auto gotshape = db->shape();
		ASSERT_ARREQ(bias_shape, gotshape);
	}
	double* gbptr = (double*) db->data();
	for (size_t i = 0, n = bias_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb[i], gbptr[i]);
	}

	{
		auto gotshape = dstate->shape();
		ASSERT_ARREQ(state_shape, gotshape);
	}
	double* gstateptr = (double*) dstate->data();
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

	auto indense = layr::dense<double>(teq::Shape({indim}), {hidden_dim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w0_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b0_data.data(), bshape, label);
		});
	auto rnn = layr::rnn<double>(hidden_dim, hidden_dim, tenncor::tanh<double>, nseq,
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w1_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b1_data.data(), bshape, label);
		}, seq_dim);

	auto layer = layr::link<double>({indense, rnn});

	auto output = layer.connect(in);

	auto err = tenncor::pow(out - output, 2.);
	auto contents = layer.get_storage();
	auto istate = contents[0];
	auto weight1 = contents[1];
	auto bias1 = contents[2];
	auto weight0 = contents[3];
	auto bias0 = contents[4];

	auto dw1 = eteq::derive(err, eteq::ETensor<double>(weight1));
	auto db1 = eteq::derive(err, eteq::ETensor<double>(bias1));
	auto dstate = eteq::derive(err, eteq::ETensor<double>(istate));
	auto dw0 = eteq::derive(err, eteq::ETensor<double>(weight0));
	auto db0 = eteq::derive(err, eteq::ETensor<double>(bias0));

	teq::TensptrsT roots = {dw1, db1, dstate, dw0, db0};
	teq::Session session;
	session.track(roots);
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
		EXPECT_DOUBLE_EQ(expect_gb[i], gb1ptr[i]);
	}

	{
		auto gotshape = dstate->shape();
		ASSERT_ARREQ(state_shape, gotshape);
	}
	double* gstateptr = (double*) dstate->data();
	for (size_t i = 0, n = state_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gstate[i], gstateptr[i]);
	}

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

	auto indense = layr::dense<double>(teq::Shape({indim}), {hidden_dim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w0_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b0_data.data(), bshape, label);
		});
	auto rnn = layr::rnn<double>(hidden_dim, hidden_dim, tenncor::tanh<double>, nseq,
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w1_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b1_data.data(), bshape, label);
		}, seq_dim);
	auto outdense = layr::dense<double>(teq::Shape({hidden_dim}), {outdim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w2_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b2_data.data(), bshape, label);
		});

	auto layer = layr::link<double>({
		indense, rnn, outdense,
		layr::bind<double>(tenncor::sigmoid<double>),
	});

	auto output = layer.connect(in);

	auto err = tenncor::pow(out - output, 2.);
	auto contents = layer.get_storage();
	auto weight2 = contents[0];
	auto bias2 = contents[1];
	auto istate = contents[2];
	auto weight1 = contents[3];
	auto bias1 = contents[4];
	auto weight0 = contents[5];
	auto bias0 = contents[6];

	auto dw0 = eteq::derive(err, eteq::ETensor<double>(weight0));
	auto db0 = eteq::derive(err, eteq::ETensor<double>(bias0));

	auto dstate = eteq::derive(err, eteq::ETensor<double>(istate));
	auto dw1 = eteq::derive(err, eteq::ETensor<double>(weight1));
	auto db1 = eteq::derive(err, eteq::ETensor<double>(bias1));

	auto dw2 = eteq::derive(err, eteq::ETensor<double>(weight2));
	auto db2 = eteq::derive(err, eteq::ETensor<double>(bias2));

	teq::TensptrsT roots = {
		dw0, db0,
		dw1, db1, dstate,
		dw2, db2,
	};
	teq::Session session;
	session.track(roots);
	session.update();

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

	{
		auto gotshape = dstate->shape();
		ASSERT_ARREQ(state_shape, gotshape);
	}
	double* gstateptr = (double*) dstate->data();
	for (size_t i = 0, n = state_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gstate[i], gstateptr[i]);
	}

	{
		auto gotshape = dw2->shape();
		ASSERT_ARREQ(w2_shape, gotshape);
	}
	double* gw2ptr = (double*) dw2->data();
	for (size_t i = 0, n = w2_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw2[i], gw2ptr[i]);
	}

	{
		auto gotshape = db2->shape();
		ASSERT_ARREQ(b2_shape, gotshape);
	}
	double* gb2ptr = (double*) db2->data();
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

	auto indense = layr::dense<double>(teq::Shape({indim}), {hidden_dim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w0_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b0_data.data(), bshape, label);
		});
	auto rnn = layr::rnn<double>(hidden_dim, hidden_dim, tenncor::tanh<double>, nseq,
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w1_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b1_data.data(), bshape, label);
		}, seq_dim);
	auto outdense = layr::dense<double>(teq::Shape({hidden_dim}), {outdim},
		[&](teq::Shape wshape, std::string label)
		{
			return eteq::make_variable<double>(w2_data.data(), wshape, label);
		},
		[&](teq::Shape bshape, std::string label)
		{
			return eteq::make_variable<double>(b2_data.data(), bshape, label);
		});

	auto layer = layr::link<double>({
		indense, rnn, outdense,
		layr::bind<double>(tenncor::sigmoid<double>),
	});

	auto output = layer.connect(in);

	double epsilon = 1e-5;
    auto common = output + epsilon;
	auto err = tenncor::reduce_mean(-(out * tenncor::log(common) + (1. - out) * tenncor::log(1. - common)));

	auto contents = layer.get_storage();
	auto weight2 = contents[0];
	auto bias2 = contents[1];
	auto istate = contents[2];
	auto weight1 = contents[3];
	auto bias1 = contents[4];
	auto weight0 = contents[5];
	auto bias0 = contents[6];

	auto dw0 = eteq::derive(err, eteq::ETensor<double>(weight0));
	auto db0 = eteq::derive(err, eteq::ETensor<double>(bias0));

	auto dstate = eteq::derive(err, eteq::ETensor<double>(istate));
	auto dw1 = eteq::derive(err, eteq::ETensor<double>(weight1));
	auto db1 = eteq::derive(err, eteq::ETensor<double>(bias1));

	auto dw2 = eteq::derive(err, eteq::ETensor<double>(weight2));
	auto db2 = eteq::derive(err, eteq::ETensor<double>(bias2));

	teq::TensptrsT roots = {
		dw0, db0,
		dw1, db1, dstate,
		dw2, db2,
	};
	teq::Session session;
	session.track(roots);
	session.update();

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

	{
		auto gotshape = dstate->shape();
		ASSERT_ARREQ(state_shape, gotshape);
	}
	double* gstateptr = (double*) dstate->data();
	for (size_t i = 0, n = state_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gstate[i], gstateptr[i]);
	}

	{
		auto gotshape = dw2->shape();
		ASSERT_ARREQ(w2_shape, gotshape);
	}
	double* gw2ptr = (double*) dw2->data();
	for (size_t i = 0, n = w2_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gw2[i], gw2ptr[i]);
	}

	{
		auto gotshape = db2->shape();
		ASSERT_ARREQ(b2_shape, gotshape);
	}
	double* gb2ptr = (double*) db2->data();
	for (size_t i = 0, n = b2_shape.n_elems(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_gb2[i], gb2ptr[i]);
	}
}


#endif // DISABLE_API_TEST
