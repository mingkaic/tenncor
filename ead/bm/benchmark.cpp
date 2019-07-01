#include <random>

#include "benchmark/benchmark.h"

#include "ead/ead.hpp"

#include "ead/parse.hpp"


static std::random_device rnd_device;
static std::mt19937 mersenne_engine(rnd_device());


ade::Shape rand_shape (int n)
{
	std::vector<ade::DimT> slist;
	uint8_t cap = (uint8_t) std::min(255, n);
	for (uint8_t i = 0; i < ade::rank_cap && cap > 1;
		++i, cap = (uint8_t) std::min(255, n))
	{
		std::uniform_int_distribution<> dist(1, cap);
		uint8_t c = dist(mersenne_engine);
		n /= c;
		slist.push_back(c);
	}
	return ade::Shape(slist);
}


static std::vector<double> random_data (size_t n, double lower, double upper)
{
	std::vector<double> out(n);
	std::uniform_real_distribution<double> dist(lower, upper);
	std::generate(out.begin(), out.end(),
		[&dist]() { return dist(mersenne_engine); });
	return out;
}


#define DEFN_BENCHMARK(NAME, FUNC, DEFN)\
DEFN(NAME, FUNC)\
BENCHMARK_TEMPLATE(NAME, double)->Range(64, 2048)\
	->Complexity(benchmark::oN);\
BENCHMARK_TEMPLATE(NAME, float)->Range(64, 2048)\
	->Complexity(benchmark::oN);\
BENCHMARK_TEMPLATE(NAME, int32_t)->Range(64, 2048)\
	->Complexity(benchmark::oN);\
BENCHMARK_TEMPLATE(NAME, int64_t)->Range(64, 2048)\
	->Complexity(benchmark::oN);


#define DEFN_UNARY(NAME, FUNC)\
template <typename T>\
static void NAME(benchmark::State& state)\
{\
	size_t n = state.range(0);\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		ade::Shape shape = rand_shape(n);\
		std::vector<double> data = random_data(shape.n_elems(), -35, 35);\
		std::vector<T> convdata(data.begin(), data.end());\
		ead::VarptrT<T> var = ead::make_variable<T>(convdata.data(), shape, "var");\
		ead::NodeptrT<T> out = FUNC(ead::NodeptrT<T>(var));\
		ead::Session session;\
		session.track({out->get_tensor()});\
		state.ResumeTiming();\
		session.update();\
	}\
	state.SetComplexityN(state.range(0));\
}


#define DEFN_UNARY_POS(NAME, FUNC)\
template <typename T>\
static void NAME(benchmark::State& state)\
{\
	size_t n = state.range(0);\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		ade::Shape shape = rand_shape(n);\
		std::vector<double> data = random_data(shape.n_elems(), 0, 35);\
		std::vector<T> convdata(data.begin(), data.end());\
		ead::VarptrT<T> var = ead::make_variable<T>(convdata.data(), shape, "var");\
		ead::NodeptrT<T> out = FUNC(ead::NodeptrT<T>(var));\
		ead::Session session;\
		session.track({out->get_tensor()});\
		state.ResumeTiming();\
		session.update();\
	}\
	state.SetComplexityN(state.range(0));\
}


DEFN_BENCHMARK(BM_Abs, tenncor::abs, DEFN_UNARY)


DEFN_BENCHMARK(BM_Neg, tenncor::neg, DEFN_UNARY)


DEFN_BENCHMARK(BM_Sin, tenncor::sin, DEFN_UNARY)


DEFN_BENCHMARK(BM_Cos, tenncor::cos, DEFN_UNARY)


DEFN_BENCHMARK(BM_Tan, tenncor::tan, DEFN_UNARY)


DEFN_BENCHMARK(BM_Exp, tenncor::exp, DEFN_UNARY)


DEFN_BENCHMARK(BM_Log, tenncor::log, DEFN_UNARY_POS)


DEFN_BENCHMARK(BM_Sqrt, tenncor::sqrt, DEFN_UNARY_POS)


DEFN_BENCHMARK(BM_Round, tenncor::round, DEFN_UNARY)


#define DEFN_BINARY(NAME, FUNC)\
template <typename T>\
static void NAME(benchmark::State& state)\
{\
	size_t n = state.range(0);\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		ade::Shape shape = rand_shape(n);\
		std::vector<double> data = random_data(shape.n_elems(), 1, 4);\
		std::vector<double> data2 = random_data(shape.n_elems(), 1, 4);\
		std::vector<T> convdata(data.begin(), data.end());\
		std::vector<T> convdata2(data2.begin(), data2.end());\
		ead::VarptrT<T> var = ead::make_variable<T>(convdata.data(), shape, "var");\
		ead::VarptrT<T> var2 = ead::make_variable<T>(convdata2.data(), shape, "var2");\
		ead::NodeptrT<T> out = FUNC(ead::NodeptrT<T>(var), ead::NodeptrT<T>(var2));\
		ead::Session session;\
		session.track({out->get_tensor()});\
		state.ResumeTiming();\
		session.update();\
	}\
	state.SetComplexityN(state.range(0));\
}


DEFN_BENCHMARK(BM_Pow, tenncor::pow, DEFN_BINARY)


DEFN_BENCHMARK(BM_Add, tenncor::add, DEFN_BINARY)


DEFN_BENCHMARK(BM_Sub, tenncor::sub, DEFN_BINARY)


DEFN_BENCHMARK(BM_Mul, tenncor::mul, DEFN_BINARY)


DEFN_BENCHMARK(BM_Div, tenncor::div, DEFN_BINARY)


DEFN_BENCHMARK(BM_Eq, tenncor::eq, DEFN_BINARY)


DEFN_BENCHMARK(BM_Ne, tenncor::neq, DEFN_BINARY)


DEFN_BENCHMARK(BM_Lt, tenncor::lt, DEFN_BINARY)


DEFN_BENCHMARK(BM_Gt, tenncor::gt, DEFN_BINARY)


template <typename T>
static void BM_Matmul(benchmark::State& state)
{
	size_t n = state.range(0);
	for (auto _ : state)
	{
		state.PauseTiming();
		std::uniform_int_distribution<ade::DimT> distc(9, std::min(255ul, n - 1));
		ade::DimT common_dim = distc(mersenne_engine);
		int remaining = (double) n / common_dim;
		std::uniform_int_distribution<> distsides(1, std::min(255, remaining));
		ade::DimT left_dim = distsides(mersenne_engine);
		ade::DimT right_dim = distsides(mersenne_engine);
		ade::Shape leftshape({common_dim, left_dim});
		ade::Shape rightshape({right_dim, common_dim});
		std::vector<double> data = random_data(leftshape.n_elems(), -35, 35);
		std::vector<double> data2 = random_data(rightshape.n_elems(), -35, 35);
		std::vector<T> convdata(data.begin(), data.end());
		std::vector<T> convdata2(data2.begin(), data2.end());
		ead::VarptrT<T> var = ead::make_variable<T>(convdata.data(), leftshape, "var");
		ead::VarptrT<T> var2 = ead::make_variable<T>(convdata2.data(), rightshape, "var2");
		ead::NodeptrT<T> out = tenncor::matmul(
			ead::NodeptrT<T>(var), ead::NodeptrT<T>(var2));
		ead::Session session;
		session.track({out->get_tensor()});
		state.ResumeTiming();
		session.update();
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Matmul, double)
	->Range(64, 2048)
	->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Matmul, float)
	->Range(64, 2048)
	->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Matmul, int32_t)
	->Range(64, 2048)
	->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Matmul, int64_t)
	->Range(64, 2048)
	->Complexity(benchmark::oN);


static void BM_MatmulComplex(benchmark::State& state)
{
	std::vector<ade::DimT> alist = {3, 2};
	std::vector<ade::DimT> blist = {4, 3};
	std::vector<ade::DimT> clist = {2, 4};
	ade::Shape ashape(alist);
	ade::Shape bshape(blist);
	ade::Shape cshape(clist);

	ead::VarptrT<int32_t> a = ead::make_variable<int32_t>(ashape);
	ead::VarptrT<int32_t> b = ead::make_variable<int32_t>(bshape);
	ead::VarptrT<int32_t> c = ead::make_variable<int32_t>(cshape);

	ead::NodeptrT<int32_t> atens(a);
	ead::NodeptrT<int32_t> btens(b);
	ead::NodeptrT<int32_t> ctens(c);

	auto d = tenncor::matmul(atens, btens);
	auto e = tenncor::matmul(ctens, d);
	auto f = tenncor::matmul(tenncor::transpose(d), tenncor::transpose(ctens));
	auto dest = tenncor::matmul(e, f);

	ead::NodeptrT<int32_t> da = ead::derive(dest, atens);
	ead::NodeptrT<int32_t> db = ead::derive(dest, btens);
	ead::NodeptrT<int32_t> dc = ead::derive(dest, ctens);
	ead::Session session;
	session.track({
		da->get_tensor(),
		db->get_tensor(),
		dc->get_tensor(),
	});

	for (auto _ : state)
	{
		state.PauseTiming();
		std::vector<double> ddata = random_data(ashape.n_elems(), 1, 100);
		std::vector<double> ddata2 = random_data(bshape.n_elems(), 1, 100);
		std::vector<double> ddata3 = random_data(cshape.n_elems(), 1, 100);
		std::vector<int32_t> data(ddata.begin(), ddata.end());
		std::vector<int32_t> data2(ddata2.begin(), ddata2.end());
		std::vector<int32_t> data3(ddata3.begin(), ddata3.end());
		state.ResumeTiming();
		a->assign(data.data(), a->shape());
		b->assign(data2.data(), b->shape());
		c->assign(data3.data(), c->shape());
		session.update({
			a->get_tensor().get(),
			b->get_tensor().get(),
			c->get_tensor().get(),
		});
	}
}

BENCHMARK(BM_MatmulComplex);


static void BM_SigmoidMLP(benchmark::State& state)
{
	ade::Shape in_shape({10, 3});
	ade::Shape weight0_shape({9, 10});
	ade::Shape bias0_shape({9});
	ade::Shape weight1_shape({5, 9});
	ade::Shape bias1_shape({5});
	ade::Shape out_shape({5,3});

	ead::VarptrT<double> in = ead::make_variable<double>(in_shape);
	ead::VarptrT<double> weight0 = ead::make_variable<double>(weight0_shape);
	ead::VarptrT<double> bias0 = ead::make_variable<double>(bias0_shape);
	ead::VarptrT<double> weight1 = ead::make_variable<double>(weight1_shape);
	ead::VarptrT<double> bias1 = ead::make_variable<double>(bias1_shape);
	ead::VarptrT<double> out = ead::make_variable<double>(out_shape);

	ead::NodeptrT<double> intens(in);
	ead::NodeptrT<double> weight0tens(weight0);
	ead::NodeptrT<double> bias0tens(bias0);
	ead::NodeptrT<double> weight1tens(weight1);
	ead::NodeptrT<double> bias1tens(bias1);
	ead::NodeptrT<double> outtens(out);

	auto layer0 = tenncor::add(
		tenncor::matmul(intens, weight0tens),
		tenncor::extend(bias0tens, 1, {3}));
	auto sig0 = tenncor::div(
		ead::make_constant_scalar<double>(1, ade::Shape({9, 3})),
		tenncor::add(ead::make_constant_scalar<double>(1, ade::Shape({9, 3})),
			tenncor::exp(tenncor::neg(layer0))));

	auto layer1 = tenncor::add(
		tenncor::matmul(sig0, weight1tens),
		tenncor::extend(bias1tens, 1, {3}));
	auto sig1 = tenncor::div(ead::make_constant_scalar<double>(1, ade::Shape({5, 3})),
		tenncor::add(ead::make_constant_scalar<double>(1, ade::Shape({5, 3})),
			tenncor::exp(tenncor::neg(layer1))));

	auto err = tenncor::pow(tenncor::sub(outtens, sig1),
		ead::make_constant_scalar<double>(2, out_shape));

	auto dw0 = ead::derive(err, weight0tens);
	auto db0 = ead::derive(err, bias0tens);
	auto dw1 = ead::derive(err, weight1tens);
	auto db1 = ead::derive(err, bias1tens);
	ead::Session session;
	session.track({
		dw0->get_tensor(),
		db0->get_tensor(),
		dw1->get_tensor(),
		db1->get_tensor(),
	});

	for (auto _ : state)
	{
		state.PauseTiming();
		std::vector<double> in_data = random_data(in_shape.n_elems(), 0, 1);
		std::vector<double> w0_data = random_data(weight0_shape.n_elems(), 0, 1);
		std::vector<double> b0_data = random_data(bias0_shape.n_elems(), 0, 1);
		std::vector<double> w1_data = random_data(weight1_shape.n_elems(), 0, 1);
		std::vector<double> b1_data = random_data(bias1_shape.n_elems(), 0, 1);
		std::vector<double> out_data = random_data(out_shape.n_elems(), 0, 1);
		state.ResumeTiming();
		in->assign(in_data.data(), in->shape());
		out->assign(out_data.data(), out->shape());
		weight0->assign(w0_data.data(), weight0->shape());
		bias0->assign(b0_data.data(), bias0->shape());
		weight1->assign(w1_data.data(), weight1->shape());
		bias1->assign(b1_data.data(), bias1->shape());
		session.update({
			in->get_tensor().get(),
			out->get_tensor().get(),
			weight0->get_tensor().get(),
			bias0->get_tensor().get(),
			weight1->get_tensor().get(),
			bias1->get_tensor().get(),
		});
	}
}

BENCHMARK(BM_SigmoidMLP);


static void BM_OptimizedSigmoidMLP(benchmark::State& state)
{
	ade::Shape in_shape({10, 3});
	ade::Shape weight0_shape({9, 10});
	ade::Shape bias0_shape({9});
	ade::Shape weight1_shape({5, 9});
	ade::Shape bias1_shape({5});
	ade::Shape out_shape({5,3});

	ead::VarptrT<double> in = ead::make_variable<double>(in_shape);
	ead::VarptrT<double> weight0 = ead::make_variable<double>(weight0_shape);
	ead::VarptrT<double> bias0 = ead::make_variable<double>(bias0_shape);
	ead::VarptrT<double> weight1 = ead::make_variable<double>(weight1_shape);
	ead::VarptrT<double> bias1 = ead::make_variable<double>(bias1_shape);
	ead::VarptrT<double> out = ead::make_variable<double>(out_shape);

	ead::NodeptrT<double> intens(in);
	ead::NodeptrT<double> weight0tens(weight0);
	ead::NodeptrT<double> bias0tens(bias0);
	ead::NodeptrT<double> weight1tens(weight1);
	ead::NodeptrT<double> bias1tens(bias1);
	ead::NodeptrT<double> outtens(out);

	auto layer0 = tenncor::add(
		tenncor::matmul(intens, weight0tens),
		tenncor::extend(bias0tens, 1, {3}));
	auto sig0 = tenncor::sigmoid(layer0);

	auto layer1 = tenncor::add(
		tenncor::matmul(sig0, weight1tens),
		tenncor::extend(bias1tens, 1, {3}));
	auto sig1 = tenncor::sigmoid(layer1);

	auto err = tenncor::pow(tenncor::sub(outtens, sig1),
		ead::make_constant_scalar<double>(2, out_shape));

	auto dw0 = ead::derive(err, weight0tens);
	auto db0 = ead::derive(err, bias0tens);
	auto dw1 = ead::derive(err, weight1tens);
	auto db1 = ead::derive(err, bias1tens);

	// optimize
	auto rules = ead::parse_file<double>("cfg/optimizations.rules");
	ade::TensT roots = {
		dw0->get_tensor(),
		db0->get_tensor(),
		dw1->get_tensor(),
		db1->get_tensor(),
	};
	opt::optimize(roots, rules);

	ead::Session session;
	session.track({
		dw0->get_tensor(),
		db0->get_tensor(),
		dw1->get_tensor(),
		db1->get_tensor(),
	});

	for (auto _ : state)
	{
		state.PauseTiming();
		std::vector<double> in_data = random_data(in_shape.n_elems(), 0, 1);
		std::vector<double> w0_data = random_data(weight0_shape.n_elems(), 0, 1);
		std::vector<double> b0_data = random_data(bias0_shape.n_elems(), 0, 1);
		std::vector<double> w1_data = random_data(weight1_shape.n_elems(), 0, 1);
		std::vector<double> b1_data = random_data(bias1_shape.n_elems(), 0, 1);
		std::vector<double> out_data = random_data(out_shape.n_elems(), 0, 1);
		state.ResumeTiming();
		in->assign(in_data.data(), in->shape());
		out->assign(out_data.data(), out->shape());
		weight0->assign(w0_data.data(), weight0->shape());
		bias0->assign(b0_data.data(), bias0->shape());
		weight1->assign(w1_data.data(), weight1->shape());
		bias1->assign(b1_data.data(), bias1->shape());
		session.update({
			in->get_tensor().get(),
			out->get_tensor().get(),
			weight0->get_tensor().get(),
			bias0->get_tensor().get(),
			weight1->get_tensor().get(),
			bias1->get_tensor().get(),
		});
	}
}

BENCHMARK(BM_OptimizedSigmoidMLP);


BENCHMARK_MAIN();
