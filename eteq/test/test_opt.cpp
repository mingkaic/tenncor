
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

	auto ass = tenncor::depends(tenncor::assign(target, b * d), {c});

	eteq::ETensorsT<double> roots = {ass};

	std::ifstream rulefile("cfg/optimizations.json");
	eteq::optimize(roots, rulefile);

	EXPECT_GRAPHEQ(
		"(DEPEND[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(ASSIGN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(constant:[88\\60\\296\\152\\244\\...][2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(constant:[81\\25\\102\\48\\128\\...][2\\3\\4\\1\\1\\1\\1\\1])\n", roots[0]);
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

	auto add = tenncor::depends(tenncor::add(target, b * d), {c});

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
		" `--(ADD[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(constant:[88\\60\\296\\152\\244\\...][2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(constant:[81\\25\\102\\48\\128\\...][2\\3\\4\\1\\1\\1\\1\\1])\n", roots[0]);

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
	auto cell = tenncor::nn::dense(cell_in, weight, bias);

	auto state = tenncor::extend_like(istate,
		tenncor::slice(in, 0, 1, seq_dim));

	auto output = tenncor::nn::rnn(in, state, cell,
		layr::UnaryF<double>(tenncor::tanh<double>), seq_dim);

	auto err = tenncor::pow(out - output, 2.);

	auto dw = eteq::derive(err, weight);
	auto db = eteq::derive(err, bias);
	auto dstate = eteq::derive(err, istate);

	eteq::ETensorsT<double> roots = {dw, db, dstate, err};

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


#endif // DISABLE_OPT_TEST
