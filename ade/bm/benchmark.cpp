#include <random>

#include "benchmark/benchmark.h"

#include "ade/coord.hpp"


static std::random_device rnd_device;
static std::mt19937 mersenne_engine(rnd_device());


template <size_t N>
static std::vector<ade::DimT> random_vector (
	ade::DimT lower, ade::DimT upper)
{
	std::vector<ade::DimT> out(N);
	std::uniform_int_distribution<ade::DimT> dist(lower, upper);
	std::generate(out.begin(), out.end(),
		[&dist]() { return dist(mersenne_engine); });
	return out;
}


static ade::NElemT random_bignum (ade::NElemT lower, ade::NElemT upper)
{
	std::uniform_int_distribution<ade::DimT> dist(lower, upper);
	return dist(mersenne_engine);
}


static void BM_MakeReduce(benchmark::State& state)
{
	std::vector<ade::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<ade::rank_cap>(1, 255);
		ade::RankT rank = random_bignum(0, ade::rank_cap - 1);
		state.ResumeTiming();
		ade::reduce(rank,
			std::vector<ade::DimT>(slist.begin() + rank, slist.end()));
	}
}

BENCHMARK(BM_MakeReduce);


static void BM_CoordFromIndex(benchmark::State& state)
{
	std::vector<ade::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<ade::rank_cap>(1, 255);
		ade::Shape shape(slist);
		ade::NElemT index = random_bignum(0, shape.n_elems());
		state.ResumeTiming();
		ade::coordinate(shape, index);
	}
}

BENCHMARK(BM_CoordFromIndex);


static void BM_IndexFromCoord(benchmark::State& state)
{
	ade::CoordT coord;
	std::vector<ade::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<ade::rank_cap>(1, 255);
		ade::Shape shape(slist);
		ade::NElemT index = random_bignum(0, shape.n_elems());
		coord = ade::coordinate(shape, index);
		state.ResumeTiming();
		ade::index(shape, coord);
	}
}

BENCHMARK(BM_IndexFromCoord);


static void BM_CoordReduce(benchmark::State& state)
{
	ade::CoordT outcoord, coord;
	std::vector<ade::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<ade::rank_cap>(1, 255);
		ade::Shape shape(slist);
		ade::NElemT index = random_bignum(0, shape.n_elems());
		coord = ade::coordinate(shape, index);
		ade::RankT rank = random_bignum(0, ade::rank_cap - 1);
		auto reducer = ade::reduce(rank,
			std::vector<ade::DimT>(slist.begin() + rank, slist.end()));
		state.ResumeTiming();
		reducer->forward(outcoord.begin(), coord.begin());
	}
}

BENCHMARK(BM_CoordReduce);


static void BM_ReduceReverse(benchmark::State& state)
{
	std::vector<ade::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<ade::rank_cap>(1, 255);
		ade::RankT rank = random_bignum(0, ade::rank_cap - 1);
		auto reducer = ade::reduce(rank,
			std::vector<ade::DimT>(slist.begin() + rank, slist.end()));
		state.ResumeTiming();
		delete reducer->reverse();
	}
}

BENCHMARK(BM_ReduceReverse);


static void BM_RedPermConnect(benchmark::State& state)
{
	std::vector<ade::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<ade::rank_cap>(1, 255);
		ade::RankT rank = random_bignum(0, ade::rank_cap - 1);
		std::vector<ade::RankT> indices(ade::rank_cap);
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), mersenne_engine);
		auto permuter = ade::permute(indices);
		auto reducer = ade::reduce(rank,
			std::vector<ade::DimT>(slist.begin() + rank, slist.end()));
		state.ResumeTiming();
		delete reducer->connect(*permuter);
	}
}

BENCHMARK(BM_RedPermConnect);


BENCHMARK_MAIN();
