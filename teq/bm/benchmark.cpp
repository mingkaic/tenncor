#include <random>

#include "benchmark/benchmark.h"

#include "teq/coord.hpp"


static std::random_device rnd_device;
static std::mt19937 mersenne_engine(rnd_device());


template <size_t N>
static std::vector<teq::DimT> random_vector (
	teq::DimT lower, teq::DimT upper)
{
	std::vector<teq::DimT> out(N);
	std::uniform_int_distribution<teq::DimT> dist(lower, upper);
	std::generate(out.begin(), out.end(),
		[&dist]() { return dist(mersenne_engine); });
	return out;
}


static teq::NElemT random_bignum (teq::NElemT lower, teq::NElemT upper)
{
	std::uniform_int_distribution<teq::DimT> dist(lower, upper);
	return dist(mersenne_engine);
}


static void BM_MakeReduce(benchmark::State& state)
{
	std::vector<teq::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<teq::rank_cap>(1, 255);
		teq::RankT rank = random_bignum(0, teq::rank_cap - 1);
		state.ResumeTiming();
		teq::reduce(rank,
			std::vector<teq::DimT>(slist.begin() + rank, slist.end()));
	}
}

BENCHMARK(BM_MakeReduce);


static void BM_CoordFromIndex(benchmark::State& state)
{
	std::vector<teq::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<teq::rank_cap>(1, 255);
		teq::Shape shape(slist);
		teq::NElemT index = random_bignum(0, shape.n_elems());
		state.ResumeTiming();
		teq::coordinate(shape, index);
	}
}

BENCHMARK(BM_CoordFromIndex);


static void BM_IndexFromCoord(benchmark::State& state)
{
	teq::CoordT coord;
	std::vector<teq::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<teq::rank_cap>(1, 255);
		teq::Shape shape(slist);
		teq::NElemT index = random_bignum(0, shape.n_elems());
		coord = teq::coordinate(shape, index);
		state.ResumeTiming();
		teq::index(shape, coord);
	}
}

BENCHMARK(BM_IndexFromCoord);


static void BM_CoordReduce(benchmark::State& state)
{
	teq::CoordT outcoord, coord;
	std::vector<teq::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<teq::rank_cap>(1, 255);
		teq::Shape shape(slist);
		teq::NElemT index = random_bignum(0, shape.n_elems());
		coord = teq::coordinate(shape, index);
		teq::RankT rank = random_bignum(0, teq::rank_cap - 1);
		auto reducer = teq::reduce(rank,
			std::vector<teq::DimT>(slist.begin() + rank, slist.end()));
		state.ResumeTiming();
		reducer->forward(outcoord.begin(), coord.begin());
	}
}

BENCHMARK(BM_CoordReduce);


static void BM_ReduceReverse(benchmark::State& state)
{
	std::vector<teq::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<teq::rank_cap>(1, 255);
		teq::RankT rank = random_bignum(0, teq::rank_cap - 1);
		auto reducer = teq::reduce(rank,
			std::vector<teq::DimT>(slist.begin() + rank, slist.end()));
		state.ResumeTiming();
		delete reducer->reverse();
	}
}

BENCHMARK(BM_ReduceReverse);


static void BM_RedPermConnect(benchmark::State& state)
{
	std::vector<teq::DimT> slist;
	for (auto _ : state)
	{
		state.PauseTiming();
		slist = random_vector<teq::rank_cap>(1, 255);
		teq::RankT rank = random_bignum(0, teq::rank_cap - 1);
		std::vector<teq::RankT> indices(teq::rank_cap);
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), mersenne_engine);
		auto permuter = teq::permute(indices);
		auto reducer = teq::reduce(rank,
			std::vector<teq::DimT>(slist.begin() + rank, slist.end()));
		state.ResumeTiming();
		delete reducer->connect(*permuter);
	}
}

BENCHMARK(BM_RedPermConnect);


struct SilentLogger final : public logs::iLogger
{
	void log (size_t msg_level, std::string msg) const override {}

	size_t get_log_level (void) const override { return 0; }

	void set_log_level (size_t log_level) override {}

	void warn (std::string msg) const override {}

	void error (std::string msg) const override {}

	void fatal (std::string msg) const override
	{
		throw std::runtime_error(msg);
	}
};


int main(int argc, char** argv)
{
	std::shared_ptr<logs::iLogger> logger = std::make_shared<SilentLogger>();
	set_logger(logger);

	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
	return 0;
}
