#include <random>

#include "benchmark/benchmark.h"

#include "ead/ead.hpp"

#include "ead/compiler/compile.hpp"


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

	auto d = age::matmul(atens, btens);
	auto e = age::matmul(ctens, d);
	auto f = age::matmul(age::transpose(d), age::transpose(ctens));
	auto dest = age::matmul(e, f);

	ead::NodeptrT<int32_t> da = ead::derive(dest, atens);
	ead::NodeptrT<int32_t> db = ead::derive(dest, btens);
	ead::NodeptrT<int32_t> dc = ead::derive(dest, ctens);
	ead::NodesT<int32_t> to_compile = {da, db, dc};
	auto compiled = ead::compiler::compile_roots(to_compile);

	ead::UpdatersT<int32_t> ordered_update = ead::order_updates(compiled);

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
		for (auto& updater : ordered_update)
		{
			updater->assign();
		}
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

	auto layer0 = age::add(age::matmul(intens, weight0tens), age::extend(bias0tens, 1, {3}));
	auto sig0 = age::div(ead::make_constant_scalar<double>(1, ade::Shape({9, 3})),
		age::add(ead::make_constant_scalar<double>(1, ade::Shape({9, 3})),
			age::exp(age::neg(layer0))));

	auto layer1 = age::add(age::matmul(sig0, weight1tens), age::extend(bias1tens, 1, {3}));
	auto sig1 = age::div(ead::make_constant_scalar<double>(1, ade::Shape({5, 3})),
		age::add(ead::make_constant_scalar<double>(1, ade::Shape({5, 3})),
			age::exp(age::neg(layer1))));

	auto err = age::pow(age::sub(outtens, sig1),
		ead::make_constant_scalar<double>(2, out_shape));

	auto dw0 = ead::derive(err, weight0tens);
	auto db0 = ead::derive(err, bias0tens);
	auto dw1 = ead::derive(err, weight1tens);
	auto db1 = ead::derive(err, bias1tens);

	ead::NodesT<double> to_compile = {dw0, db0, dw1, db1};
	auto compiled = ead::compiler::compile_roots(to_compile);

	ead::UpdatersT<double> ordered_update = ead::order_updates(compiled);

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
		for (auto& updater : ordered_update)
		{
			updater->assign();
		}
	}
}

BENCHMARK(BM_SigmoidMLP);


BENCHMARK_MAIN();
