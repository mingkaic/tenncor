
#ifndef DISABLE_OPT_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

// #include "dbg/print/search.hpp"

#include "eteq/derive.hpp"
#include "eteq/optimize.hpp"

#include "generated/api.hpp"

const std::string testdir = "models/test";


TEST(OPTIMIZE, Depends)
{
	// tensor operation
	std::vector<teq::DimT> slist = {2, 3, 4};
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};
	std::vector<double> data2 = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};
	teq::Shape shape(slist);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);
	assert(data2.size() == n);

	eteq::EVariable<double> target = eteq::make_variable<double>(data.data(), shape);
	eteq::ETensor<double> a = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> b = eteq::make_constant<double>(data2.data(), shape);
	auto c = a + b;
	eteq::ETensor<double> d = eteq::make_constant_scalar<double>(4, shape);

	auto ass = tenncor<double>().depends(tenncor<double>().assign(target, b * d), {c});

	eteq::ETensorsT<double> roots = {ass};

	std::ifstream rulefile("cfg/optimizations.json");
	eteq::optimize(roots, rulefile);

	EXPECT_GRAPHEQ(
		"(DEPEND[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(ASSIGN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_|___`--(variable:[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:[88\\60\\296\\152\\244\\...][2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(constant:[81\\25\\102\\48\\128\\...][2\\3\\4\\1\\1\\1\\1\\1])\n", roots[0]);
}


// ensure optimizing does not merge depend node and its arguments to confuse it for nnary operators
TEST(OPTIMIZE, DependsNnary)
{
	// tensor operation
	std::vector<teq::DimT> slist = {2, 3, 4};
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};
	std::vector<double> data2 = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};
	teq::Shape shape(slist);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);
	assert(data2.size() == n);

	eteq::EVariable<double> target = eteq::make_variable<double>(data.data(), shape);
	eteq::ETensor<double> a = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> b = eteq::make_constant<double>(data2.data(), shape);
	auto c = a + b;
	eteq::ETensor<double> d = eteq::make_constant_scalar<double>(4, shape);

	auto add = tenncor<double>().depends(tenncor<double>().add(target, b * d), {c});

	teq::Session sess = eigen::get_session();
	sess.track({teq::TensptrT(add)});
	sess.update_target({add.get()});
	teq::Shape exshape = add->shape();
	double* expect_data = (double*) add->device().data();
	std::vector<double> evdata(expect_data, expect_data + exshape.n_elems());

	eteq::ETensorsT<double> roots = {add};

	std::ifstream rulefile("cfg/optimizations.json");
	eteq::optimize(roots, rulefile);

	EXPECT_GRAPHEQ(
		"(DEPEND[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(ADD[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_|___`--(variable:[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:[88\\60\\296\\152\\244\\...][2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(constant:[81\\25\\102\\48\\128\\...][2\\3\\4\\1\\1\\1\\1\\1])\n", roots[0]);

	sess.track({teq::TensptrT(roots[0])});
	sess.update_target({roots[0].get()});
	teq::Shape gotshape = roots[0]->shape();
	double* got_data = (double*) roots[0]->device().data();
	std::vector<double> gvdata(got_data, got_data + gotshape.n_elems());

	ASSERT_ARREQ(exshape, gotshape);
	EXPECT_VECEQ(evdata, gvdata);
}


TEST(OPTIMIZE, RNNLayer)
{
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
		eteq::make_variable<double>(in_data.data(), in_shape, "in");
	eteq::EVariable<double> weight =
		eteq::make_variable<double>(weight_data.data(), weight_shape, "weight");
	eteq::EVariable<double> bias =
		eteq::make_variable<double>(bias_data.data(), bias_shape, "bias");
	eteq::EVariable<double> istate =
		eteq::make_variable<double>(state_data.data(), state_shape, "state");
	eteq::EVariable<double> out =
		eteq::make_variable<double>(out_data.data(), out_shape, "out");

	teq::RankT seq_dim = 1;
	eteq::ETensor<double> cell_in(eteq::make_variable_scalar<double>(0, teq::Shape({10})));
	auto cell = tenncor<double>().nn.dense(cell_in, weight, bias);

	auto state = tenncor<double>().extend_like(istate,
		tenncor<double>().slice(in, 0, 1, seq_dim));

	auto output = tenncor<double>().nn.rnn(in, state, cell,
		[](const eteq::ETensor<double>& x)
		{
			return tenncor<double>().tanh(x);
		}, seq_dim);

	auto err = tenncor<double>().pow(out - output, 2.);

	auto ders = eteq::derive(err, {weight, bias, istate});
	eteq::ETensorsT<double> roots = {ders[0], ders[1], ders[2], err};

	std::ifstream rulefile("cfg/optimizations.json");
	eteq::optimize(roots, rulefile);

	{
		std::string expect_pbfile = testdir + "/opt0.txt";
		std::ifstream expect_ifs(expect_pbfile);
		std::string expect(
			(std::istreambuf_iterator<char>(expect_ifs)),
			(std::istreambuf_iterator<char>()));
		EXPECT_GRAPHEQ(expect, roots[0]);
	}
	{
		std::string expect_pbfile = testdir + "/opt1.txt";
		std::ifstream expect_ifs(expect_pbfile);
		std::string expect(
			(std::istreambuf_iterator<char>(expect_ifs)),
			(std::istreambuf_iterator<char>()));
		EXPECT_GRAPHEQ(expect, roots[1]);
	}
	{
		std::string expect_pbfile = testdir + "/opt2.txt";
		std::ifstream expect_ifs(expect_pbfile);
		std::string expect(
			(std::istreambuf_iterator<char>(expect_ifs)),
			(std::istreambuf_iterator<char>()));
		EXPECT_GRAPHEQ(expect, roots[2]);
	}
	{
		std::string expect_pbfile = testdir + "/opt3.txt";
		std::ifstream expect_ifs(expect_pbfile);
		std::string expect(
			(std::istreambuf_iterator<char>(expect_ifs)),
			(std::istreambuf_iterator<char>()));
		EXPECT_GRAPHEQ(expect, roots[3]);
	}
}


TEST(OPTIMIZE, CNNLayer)
{
	teq::Shape in_shape({2, 4, 4});
	teq::Shape out_shape({2, 4, 4});

	// invar has 32 elements, outvar has 32 elements
	// conv weight has 54 elements, bias has 2 elements
	// dense weight has 24 elements, bias has 3 elements
	std::vector<double> in_data = {
		0.3916215715, 0.8931230937, 0.1004809072, 0.6106936032,
		0.1609866205, 0.5359489463, 0.9634307603, 0.2478257704,

		0.1131220231, 0.4383789791, 0.0301699064, 0.3556333890,
		0.0795634409, 0.4742393984, 0.8927304780, 0.0425374477,

		0.6454238082, 0.6158521193, 0.8372817240, 0.0632589826,
		0.8192038024, 0.4463440314, 0.4563414313, 0.0836714963,

		0.2945361165, 0.2556635326, 0.1268742524, 0.2480380053,
		0.8182844250, 0.4024040436, 0.9260259954, 0.5273402129,
	};
	std::vector<double> out_data = {
		0.0486572783, 0.2315281440,
		0.4123123876, 0.4495204814,
		0.0607173969, 0.2353327942,
		0.8020420684, 0.0349825945,

		0.2586609385, 0.0877771944,
		0.9793879792, 0.2052353283,
		0.8427712275, 0.3320400669,
		0.2011303283, 0.9303916739,

		0.1586272183, 0.9857192229,
		0.0378285752, 0.6744445943,
		0.0417076290, 0.5292058108,
		0.7405830646, 0.8138606324,

		0.1809403598, 0.1407543910,
		0.9964283066, 0.0307449753,
		0.4892965036, 0.0682599624,
		0.4739679432, 0.0640243035,
	};
	std::vector<double> cweight_data = {
		0.6108596848, 0.0956259063, 0.6491984452,
		0.3726083910, 0.5912457068, 0.9649860713,
		0.5676557932, 0.8560985490, 0.7104686176,

		0.8212629984, 0.7445572667, 0.2392157299,
		0.6922254848, 0.1974129231, 0.1444881939,
		0.5144365413, 0.9551529858, 0.3786297296,

		0.7821505938, 0.3357298454, 0.3313290519,
		0.2713678980, 0.5092105264, 0.0672732383,
		0.7695201538, 0.2726840520, 0.3161799709,

		0.7017531611, 0.7219056805, 0.8948515926,
		0.0569561413, 0.6600032360, 0.5564323337,
		0.0013410820, 0.4314001600, 0.8535746740,

		0.8290206508, 0.0293501654, 0.4915898917,
		0.2890500076, 0.7259008624, 0.7892775434,
		0.8895895730, 0.4935260523, 0.5615521486,

		0.2289719211, 0.4937565211, 0.8541293040,
		0.3559381633, 0.7543374263, 0.7611097303,
		0.8421407822, 0.0207011400, 0.1428302497,
	};
	std::vector<double> cbias_data = {
		0.8801580962, 0.5008790402,
	};

	eteq::EVariable<double> invar = eteq::make_variable<double>(
		in_data.data(), in_shape, "invar");
	eteq::EVariable<double> outvar = eteq::make_variable<double>(
		out_data.data(), out_shape, "outvar");

	// construct CNN
	auto model = tenncor<double>().layer.link({ // input [2\4\4]
		tenncor<double>().layer.conv({3, 3}, 2, 2,
			[&](teq::Shape shape, std::string label) -> eteq::EVariable<double>
			{
				return eteq::make_variable<double>(cweight_data.data(), shape, label);
			},
			[&](teq::Shape shape, std::string label) -> eteq::EVariable<double>
			{
				return eteq::make_variable<double>(cbias_data.data(), shape, label);
			},
			{{1, 1}, {1, 1}}), // outputs [2\4\4]
		tenncor<double>().layer.bind(
			[](const eteq::ETensor<double>& x)
			{
				return tenncor<double>().relu(x);
			}),
	}, invar);

	double learning_rate = 0.01;
	double l2_decay = 0.0001;

	auto normalized = invar / 255. - 0.5;
	eteq::ETensor<double> train_out = eteq::connect(model, normalized);
	auto error = -tenncor<double>().reduce_sum(outvar *
		tenncor<double>().log(train_out + std::numeric_limits<double>::epsilon()));

	eteq::VarptrsT<double> vars = eteq::get_storage(model);
	auto updates = tenncor<double>().approx.adadelta(
		error, eteq::EVariablesT<double>(vars.begin(), vars.end()),
		learning_rate, l2_decay);
	teq::TensMapT<teq::TensptrT> umap;
	eteq::ETensorsT<double> deps;
	deps.reserve(updates.size());
	for (auto& update : updates)
	{
		umap.emplace(update.first.get(), update.second);
		deps.push_back(update.second);
	}
	auto err = tenncor<double>().identity(tenncor<double>().depends(eteq::trail(error, umap), deps));

	eteq::ETensorsT<double> roots = {err};
	std::ifstream rulefile("cfg/optimizations.json");
	eteq::optimize(roots, rulefile);
	err = roots[0];

	std::string expect_pbfile = testdir + "/cnn_opt.txt";
	std::ifstream expect_ifs(expect_pbfile);
	std::string expect(
		(std::istreambuf_iterator<char>(expect_ifs)),
		(std::istreambuf_iterator<char>()));
	EXPECT_GRAPHEQ(expect.c_str(), err);

	teq::Session sess = eigen::get_session();
	sess.track({err});
	sess.update_target({err.get()});

	teq::Shape exshape;
	double evdata = 451.94709417496551;

	teq::Shape gotshape = err->shape();
	double* got_data = (double*) err->device().data();
	std::vector<double> gvdata(got_data, got_data + gotshape.n_elems());

	ASSERT_ARREQ(exshape, gotshape);
	EXPECT_DOUBLE_EQ(evdata, gvdata[0]);
}


#endif // DISABLE_OPT_TEST
