#include <random>

#include "benchmark/benchmark.h"

#include "tenncor/tenncor.hpp"


static std::random_device rnd_device;
static std::mt19937 mersenne_engine(rnd_device());


teq::Shape rand_shape (int n)
{
	teq::DimsT slist;
	teq::RankT cap = (teq::RankT) std::min(255, n);
	for (teq::RankT i = 0; i < teq::rank_cap && cap > 1;
		++i, cap = (teq::RankT) std::min(255, n))
	{
		std::uniform_int_distribution<teq::RankT> dist(1, cap);
		teq::RankT c = dist(mersenne_engine);
		n /= c;
		slist.push_back(c);
	}
	return teq::Shape(slist);
}


static std::vector<double> random_data (size_t n, double lower, double upper)
{
	std::vector<double> out(n);
	std::uniform_real_distribution<double> dist(lower, upper);
	std::generate(out.begin(), out.end(),
		[&dist] { return dist(mersenne_engine); });
	return out;
}


#define DEFN_BENCHMARK(NAME, FUNC, DEFN)\
DEFN(NAME, FUNC)\
BENCHMARK_TEMPLATE(NAME, double)->Range(64, 2048)\
	->Complexity(benchmark::oN);\
BENCHMARK_TEMPLATE(NAME, float)->Range(64, 2048)\
	->Complexity(benchmark::oN);\
BENCHMARK_TEMPLATE(NAME, int32_t)->Range(64, 2048)\
	->Complexity(benchmark::oN);


#define DEFN_UNARY(NAME, FUNC)\
template <typename T>\
static void NAME(benchmark::State& state)\
{\
	size_t n = state.range(0);\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		teq::Shape shape = rand_shape(n);\
		std::vector<double> data = random_data(shape.n_elems(), -35, 35);\
		std::vector<T> convdata(data.begin(), data.end());\
		eteq::EVariable<T> var = eteq::make_variable<T>(convdata.data(), shape, "var");\
		eteq::ETensor out = FUNC(var);\
		state.ResumeTiming();\
		out.template calc<T>();\
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
		teq::Shape shape = rand_shape(n);\
		std::vector<double> data = random_data(shape.n_elems(), 0, 35);\
		std::vector<T> convdata(data.begin(), data.end());\
		eteq::EVariable<T> var = eteq::make_variable<T>(convdata.data(), shape, "var");\
		eteq::ETensor out = FUNC(var);\
		state.ResumeTiming();\
		out.template calc<T>();\
	}\
	state.SetComplexityN(state.range(0));\
}


DEFN_BENCHMARK(BM_Abs, tenncor().abs, DEFN_UNARY)


DEFN_BENCHMARK(BM_Neg, tenncor().neg, DEFN_UNARY)


DEFN_BENCHMARK(BM_Sin, tenncor().sin, DEFN_UNARY)


DEFN_BENCHMARK(BM_Cos, tenncor().cos, DEFN_UNARY)


DEFN_BENCHMARK(BM_Tan, tenncor().tan, DEFN_UNARY)


DEFN_BENCHMARK(BM_Exp, tenncor().exp, DEFN_UNARY)


DEFN_BENCHMARK(BM_Log, tenncor().log, DEFN_UNARY_POS)


DEFN_BENCHMARK(BM_Sqrt, tenncor().sqrt, DEFN_UNARY_POS)


DEFN_BENCHMARK(BM_Round, tenncor().round, DEFN_UNARY)


#define DEFN_BINARY(NAME, FUNC)\
template <typename T>\
static void NAME(benchmark::State& state)\
{\
	size_t n = state.range(0);\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		teq::Shape shape = rand_shape(n);\
		std::vector<double> data = random_data(shape.n_elems(), 1, 4);\
		std::vector<double> data2 = random_data(shape.n_elems(), 1, 4);\
		std::vector<T> convdata(data.begin(), data.end());\
		std::vector<T> convdata2(data2.begin(), data2.end());\
		eteq::EVariable<T> var = eteq::make_variable<T>(convdata.data(), shape, "var");\
		eteq::EVariable<T> var2 = eteq::make_variable<T>(convdata2.data(), shape, "var2");\
		eteq::ETensor out = FUNC(var, var2);\
		state.ResumeTiming();\
		out.template calc<T>();\
	}\
	state.SetComplexityN(state.range(0));\
}


DEFN_BENCHMARK(BM_Pow, tenncor().pow, DEFN_BINARY)


DEFN_BENCHMARK(BM_Add, tenncor().add, DEFN_BINARY)


DEFN_BENCHMARK(BM_Sub, tenncor().sub, DEFN_BINARY)


DEFN_BENCHMARK(BM_Mul, tenncor().mul, DEFN_BINARY)


DEFN_BENCHMARK(BM_Div, tenncor().div, DEFN_BINARY)


DEFN_BENCHMARK(BM_Eq, tenncor().eq, DEFN_BINARY)


DEFN_BENCHMARK(BM_Ne, tenncor().neq, DEFN_BINARY)


DEFN_BENCHMARK(BM_Lt, tenncor().lt, DEFN_BINARY)


DEFN_BENCHMARK(BM_Gt, tenncor().gt, DEFN_BINARY)


template <typename T>
static void BM_Matmul(benchmark::State& state)
{
	size_t n = state.range(0);
	for (auto _ : state)
	{
		state.PauseTiming();
		std::uniform_int_distribution<teq::DimT> distc(9, std::min(255ul, n - 1));
		teq::DimT common_dim = distc(mersenne_engine);
		int remaining = (double) n / common_dim;
		std::uniform_int_distribution<> distsides(1, std::min(255, remaining));
		teq::DimT left_dim = distsides(mersenne_engine);
		teq::DimT right_dim = distsides(mersenne_engine);
		teq::Shape leftshape({common_dim, left_dim});
		teq::Shape rightshape({right_dim, common_dim});
		std::vector<double> data = random_data(leftshape.n_elems(), -35, 35);
		std::vector<double> data2 = random_data(rightshape.n_elems(), -35, 35);
		std::vector<T> convdata(data.begin(), data.end());
		std::vector<T> convdata2(data2.begin(), data2.end());
		eteq::EVariable<T> var = eteq::make_variable<T>(convdata.data(), leftshape, "var");
		eteq::EVariable<T> var2 = eteq::make_variable<T>(convdata2.data(), rightshape, "var2");
		eteq::ETensor out = tenncor().matmul(var, var2);
		state.ResumeTiming();
		out.template calc<T>();
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


static void BM_MatmulComplex(benchmark::State& state)
{
	teq::DimsT alist = {3, 2};
	teq::DimsT blist = {4, 3};
	teq::DimsT clist = {2, 4};
	teq::Shape ashape(alist);
	teq::Shape bshape(blist);
	teq::Shape cshape(clist);

	eteq::EVariable<int32_t> a = eteq::make_variable<int32_t>(ashape);
	eteq::EVariable<int32_t> b = eteq::make_variable<int32_t>(bshape);
	eteq::EVariable<int32_t> c = eteq::make_variable<int32_t>(cshape);

	auto d = tenncor().matmul(a, b);
	auto e = tenncor().matmul(c, d);
	auto f = tenncor().matmul(tenncor().transpose(d), tenncor().transpose(c));
	auto dest = tenncor().matmul(e, f);

	eteq::ETensorsT ders = tcr::derive(dest, {a, b, c});
	auto da = ders[0];
	auto db = ders[1];
	auto dc = ders[2];

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
		da.template calc<int32_t>();
		db.template calc<int32_t>();
		dc.template calc<int32_t>();
	}
}

BENCHMARK(BM_MatmulComplex);


static void BM_SigmoidMLP(benchmark::State& state)
{
	teq::Shape in_shape({10, 3});
	teq::Shape weight0_shape({9, 10});
	teq::Shape bias0_shape({9});
	teq::Shape weight1_shape({5, 9});
	teq::Shape bias1_shape({5});
	teq::Shape out_shape({5,3});

	eteq::EVariable<double> in = eteq::make_variable<double>(in_shape);
	eteq::EVariable<double> weight0 = eteq::make_variable<double>(weight0_shape);
	eteq::EVariable<double> bias0 = eteq::make_variable<double>(bias0_shape);
	eteq::EVariable<double> weight1 = eteq::make_variable<double>(weight1_shape);
	eteq::EVariable<double> bias1 = eteq::make_variable<double>(bias1_shape);
	eteq::EVariable<double> out = eteq::make_variable<double>(out_shape);

	auto layer0 =
		tenncor().matmul(in, weight0) +
		tenncor().extend(bias0, 1, {3});
	auto sig0 = 1. / (1. + tenncor().exp(-layer0));

	auto layer1 =
		tenncor().matmul(sig0, weight1) +
		tenncor().extend(bias1, 1, {3});
	auto sig1 = 1. / (1. + tenncor().exp(-layer1));

	auto err = tenncor().pow(out - sig1, 2.);

	auto ders = tcr::derive(err, {weight0, bias0, weight1, bias1});
	auto dw0 = ders[0];
	auto db0 = ders[1];
	auto dw1 = ders[2];
	auto db1 = ders[3];

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
		dw0.template calc<double>();
		db0.template calc<double>();
		dw1.template calc<double>();
		db1.template calc<double>();
	}
}

BENCHMARK(BM_SigmoidMLP);


#ifdef ENABLE_OPT


static void BM_OptimizedSigmoidMLP(benchmark::State& state)
{
	teq::Shape in_shape({10, 3});
	teq::Shape weight0_shape({9, 10});
	teq::Shape bias0_shape({9});
	teq::Shape weight1_shape({5, 9});
	teq::Shape bias1_shape({5});
	teq::Shape out_shape({5,3});

	eteq::EVariable<double> in = eteq::make_variable<double>(in_shape);
	eteq::EVariable<double> weight0 = eteq::make_variable<double>(weight0_shape);
	eteq::EVariable<double> bias0 = eteq::make_variable<double>(bias0_shape);
	eteq::EVariable<double> weight1 = eteq::make_variable<double>(weight1_shape);
	eteq::EVariable<double> bias1 = eteq::make_variable<double>(bias1_shape);
	eteq::EVariable<double> out = eteq::make_variable<double>(out_shape);

	auto layer0 =
		tenncor().matmul(in, weight0) +
		tenncor().extend(bias0, 1, {3});
	auto sig0 = tenncor().sigmoid(layer0);

	auto layer1 =
		tenncor().matmul(sig0, weight1) +
		tenncor().extend(bias1, 1, {3});
	auto sig1 = tenncor().sigmoid(layer1);

	auto err = tenncor().pow(out - sig1, 2.);

	auto dw0 = tcr::derive(err, {weight0});
	auto db0 = tcr::derive(err, {bias0});
	auto dw1 = tcr::derive(err, {weight1});
	auto db1 = tcr::derive(err, {bias1});

	// optimize
	hone::optimize("cfg/optimizations.json");

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
		dw0.template calc<double>();
		db0.template calc<double>();
		dw1.template calc<double>();
		db1.template calc<double>();
	}
}

BENCHMARK(BM_OptimizedSigmoidMLP);


#endif // ENABLE_OPT


BENCHMARK_MAIN();
