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
	teq::Shape shape = rand_shape(n);\
	eteq::EVariable var = eteq::make_variable_scalar<T>(0, shape, "var");\
	eteq::ETensor out = FUNC(var);\
	auto ctx = out.get_context();\
	auto tens = out.get();\
	eigen::Device device(std::numeric_limits<size_t>::max());\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		std::vector<double> data = random_data(shape.n_elems(), -35, 35);\
		std::vector<T> convdata(data.begin(), data.end());\
		var->assign(convdata.data(), shape);\
		state.ResumeTiming();\
		teq::get_eval(ctx).evaluate(device, {tens});\
	}\
	state.SetComplexityN(state.range(0));\
}


#define DEFN_UNARY_POS(NAME, FUNC)\
template <typename T>\
static void NAME(benchmark::State& state)\
{\
	size_t n = state.range(0);\
	teq::Shape shape = rand_shape(n);\
	eteq::EVariable var = eteq::make_variable_scalar<T>(0, shape, "var");\
	eteq::ETensor out = FUNC(var);\
	auto ctx = out.get_context();\
	auto tens = out.get();\
	eigen::Device device(std::numeric_limits<size_t>::max());\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		std::vector<double> data = random_data(shape.n_elems(), 0, 35);\
		std::vector<T> convdata(data.begin(), data.end());\
		var->assign(convdata.data(), shape);\
		state.ResumeTiming();\
		teq::get_eval(ctx).evaluate(device, {tens});\
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
	teq::Shape shape = rand_shape(n);\
	std::vector<double> data = random_data(shape.n_elems(), 1, 4);\
	std::vector<double> data2 = random_data(shape.n_elems(), 1, 4);\
	std::vector<T> convdata(data.begin(), data.end());\
	std::vector<T> convdata2(data2.begin(), data2.end());\
	eteq::EVariable var = eteq::make_variable_scalar<T>(0, shape, "var");\
	eteq::EVariable var2 = eteq::make_variable_scalar<T>(0, shape, "var2");\
	eteq::ETensor out = FUNC(var, var2);\
	auto ctx = out.get_context();\
	auto tens = out.get();\
	eigen::Device device(std::numeric_limits<size_t>::max());\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		std::vector<double> data = random_data(shape.n_elems(), 1, 4);\
		std::vector<double> data2 = random_data(shape.n_elems(), 1, 4);\
		std::vector<T> convdata(data.begin(), data.end());\
		std::vector<T> convdata2(data2.begin(), data2.end());\
		var->assign(convdata.data(), shape);\
		var2->assign(convdata2.data(), shape);\
		state.ResumeTiming();\
		teq::get_eval(ctx).evaluate(device, {tens});\
	}\
	state.SetComplexityN(state.range(0));\
}


DEFN_BENCHMARK(BM_Assign, tenncor().assign, DEFN_BINARY)


DEFN_BENCHMARK(BM_AssignAdd, tenncor().assign_add, DEFN_BINARY)


DEFN_BENCHMARK(BM_AssignSub, tenncor().assign_sub, DEFN_BINARY)


DEFN_BENCHMARK(BM_AssignMul, tenncor().assign_mul, DEFN_BINARY)


DEFN_BENCHMARK(BM_AssignDiv, tenncor().assign_div, DEFN_BINARY)


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
	teq::DimT common_dim = state.range(0);
	std::uniform_int_distribution<> distsides(1, 255);
	teq::DimT vari = distsides(mersenne_engine);
	teq::DimT left_dim = common_dim + vari;
	teq::DimT right_dim = common_dim + (255 - vari);
	teq::Shape leftshape({common_dim, left_dim});
	teq::Shape rightshape({right_dim, common_dim});
	eteq::EVariable var = eteq::make_variable_scalar<T>(0, leftshape, "var");
	eteq::EVariable var2 = eteq::make_variable_scalar<T>(0, rightshape, "var2");
	eteq::ETensor out = tenncor().matmul(var, var2);
	auto ctx = out.get_context();
	auto tens = out.get();
	eigen::Device device(std::numeric_limits<size_t>::max());
	for (auto _ : state)
	{
		state.PauseTiming();
		std::vector<double> data = random_data(leftshape.n_elems(), -35, 35);
		std::vector<double> data2 = random_data(rightshape.n_elems(), -35, 35);
		std::vector<T> convdata(data.begin(), data.end());
		std::vector<T> convdata2(data2.begin(), data2.end());
		var->assign(convdata.data(), leftshape);
		var2->assign(convdata2.data(), rightshape);
		state.ResumeTiming();
		teq::get_eval(ctx).evaluate(device, {tens});
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


template <typename T>
static void BM_Batch_Matmul(benchmark::State& state)
{
	teq::DimT common_dim = state.range(0);
	std::uniform_int_distribution<> distsides(1, 255);
	teq::DimT vari = distsides(mersenne_engine);
	teq::DimT left_dim = common_dim + vari;
	teq::DimT right_dim = common_dim + (255 - vari);
	teq::DimT batch_dim = 2;
	teq::Shape leftshape({common_dim, left_dim, batch_dim});
	teq::Shape rightshape({right_dim, common_dim, batch_dim});
	eteq::EVariable var = eteq::make_variable_scalar<T>(0, leftshape, "var");
	eteq::EVariable var2 = eteq::make_variable_scalar<T>(0, rightshape, "var2");
	eteq::ETensor out = tenncor().matmul(var, var2);
	auto ctx = out.get_context();
	eigen::Device device(std::numeric_limits<size_t>::max());
	auto tens = out.get();
	for (auto _ : state)
	{
		state.PauseTiming();
		std::vector<double> data = random_data(leftshape.n_elems(), -35, 35);
		std::vector<double> data2 = random_data(rightshape.n_elems(), -35, 35);
		std::vector<T> convdata(data.begin(), data.end());
		std::vector<T> convdata2(data2.begin(), data2.end());
		var->assign(convdata.data(), leftshape);
		var2->assign(convdata2.data(), rightshape);
		state.ResumeTiming();
		teq::get_eval(ctx).evaluate(device, {tens});
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Batch_Matmul, double)
	->Range(64, 2048)
	->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Batch_Matmul, float)
	->Range(64, 2048)
	->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Batch_Matmul, int32_t)
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

	eteq::EVariable a = eteq::make_variable<int32_t>(ashape);
	eteq::EVariable b = eteq::make_variable<int32_t>(bshape);
	eteq::EVariable c = eteq::make_variable<int32_t>(cshape);

	auto d = tenncor().matmul(a, b);
	auto e = tenncor().matmul(c, d);
	auto f = tenncor().matmul(tenncor().transpose(d), tenncor().transpose(c));
	auto dest = tenncor().matmul(e, f);

	eteq::ETensorsT ders = tcr::derive(dest, {a, b, c});
	auto da = ders[0];
	auto db = ders[1];
	auto dc = ders[2];
	auto ctx = da.get_context();
	eigen::Device device(std::numeric_limits<size_t>::max());
	auto datens = da.get();
	auto dbtens = db.get();
	auto dctens = dc.get();

	for (auto _ : state)
	{
		state.PauseTiming();
		std::vector<double> ddata = random_data(ashape.n_elems(), 1, 100);
		std::vector<double> ddata2 = random_data(bshape.n_elems(), 1, 100);
		std::vector<double> ddata3 = random_data(cshape.n_elems(), 1, 100);
		std::vector<int32_t> data(ddata.begin(), ddata.end());
		std::vector<int32_t> data2(ddata2.begin(), ddata2.end());
		std::vector<int32_t> data3(ddata3.begin(), ddata3.end());
		a->assign(data.data(), a->shape());
		b->assign(data2.data(), b->shape());
		c->assign(data3.data(), c->shape());
		state.ResumeTiming();
		teq::get_eval(ctx).evaluate(device, {datens, dbtens, dctens});
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

	eteq::EVariable in = eteq::make_variable<double>(in_shape);
	eteq::EVariable weight0 = eteq::make_variable<double>(weight0_shape);
	eteq::EVariable bias0 = eteq::make_variable<double>(bias0_shape);
	eteq::EVariable weight1 = eteq::make_variable<double>(weight1_shape);
	eteq::EVariable bias1 = eteq::make_variable<double>(bias1_shape);
	eteq::EVariable out = eteq::make_variable<double>(out_shape);

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

	auto ctx = dw0.get_context();
	eigen::Device device(std::numeric_limits<size_t>::max());
	auto dw0tens = dw0.get();
	auto db0tens = db0.get();
	auto dw1tens = dw1.get();
	auto db1tens = db1.get();

	for (auto _ : state)
	{
		state.PauseTiming();
		std::vector<double> in_data = random_data(in_shape.n_elems(), 0, 1);
		std::vector<double> w0_data = random_data(weight0_shape.n_elems(), 0, 1);
		std::vector<double> b0_data = random_data(bias0_shape.n_elems(), 0, 1);
		std::vector<double> w1_data = random_data(weight1_shape.n_elems(), 0, 1);
		std::vector<double> b1_data = random_data(bias1_shape.n_elems(), 0, 1);
		std::vector<double> out_data = random_data(out_shape.n_elems(), 0, 1);
		in->assign(in_data.data(), in->shape());
		out->assign(out_data.data(), out->shape());
		weight0->assign(w0_data.data(), weight0->shape());
		bias0->assign(b0_data.data(), bias0->shape());
		weight1->assign(w1_data.data(), weight1->shape());
		bias1->assign(b1_data.data(), bias1->shape());
		state.ResumeTiming();
		teq::get_eval(ctx).evaluate(device, {dw0tens, db0tens, dw1tens, db1tens});
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

	eteq::EVariable in = eteq::make_variable<double>(in_shape);
	eteq::EVariable weight0 = eteq::make_variable<double>(weight0_shape);
	eteq::EVariable bias0 = eteq::make_variable<double>(bias0_shape);
	eteq::EVariable weight1 = eteq::make_variable<double>(weight1_shape);
	eteq::EVariable bias1 = eteq::make_variable<double>(bias1_shape);
	eteq::EVariable out = eteq::make_variable<double>(out_shape);

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

	auto ctx = dw0.get_context();
	eigen::Device device(std::numeric_limits<size_t>::max());
	auto dw0tens = dw0.get();
	auto db0tens = db0.get();
	auto dw1tens = dw1.get();
	auto db1tens = db1.get();

	for (auto _ : state)
	{
		state.PauseTiming();
		std::vector<double> in_data = random_data(in_shape.n_elems(), 0, 1);
		std::vector<double> w0_data = random_data(weight0_shape.n_elems(), 0, 1);
		std::vector<double> b0_data = random_data(bias0_shape.n_elems(), 0, 1);
		std::vector<double> w1_data = random_data(weight1_shape.n_elems(), 0, 1);
		std::vector<double> b1_data = random_data(bias1_shape.n_elems(), 0, 1);
		std::vector<double> out_data = random_data(out_shape.n_elems(), 0, 1);
		in->assign(in_data.data(), in->shape());
		out->assign(out_data.data(), out->shape());
		weight0->assign(w0_data.data(), weight0->shape());
		bias0->assign(b0_data.data(), bias0->shape());
		weight1->assign(w1_data.data(), weight1->shape());
		bias1->assign(b1_data.data(), bias1->shape());
		state.ResumeTiming();
		teq::get_eval(ctx).evaluate(device, {dw0tens, db0tens, dw1tens, db1tens});
	}
}

BENCHMARK(BM_OptimizedSigmoidMLP);


#endif // ENABLE_OPT


BENCHMARK_MAIN();
