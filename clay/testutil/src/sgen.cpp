#include <algorithm>

#include "testify/fuzz/irng.hpp"

#include "clay/testutil/sgen.hpp"

#ifdef TESTUTIL_SGEN_HPP

namespace testutil
{

// generate shapes

std::vector<size_t> make_partial (testify::fuzz_test* fuzzer, std::vector<size_t> shape)
{
	size_t rank = shape.size();
	size_t nzeros = 1;
	if (rank > 2)
	{
		nzeros = fuzzer->get_int(1, "nzeros", {1, rank - 1})[0];
	}
	else if (rank == 1)
	{
		shape.push_back(0);
		return shape;
	}
	std::vector<size_t> zeros = fuzzer->get_int(nzeros, "zeros", {0, rank-1});
	for (size_t zidx : zeros)
	{
		shape[zidx] = 0;
	}
	return shape;
}

void make_incom_partials (testify::fuzz_test* fuzzer, std::vector<size_t> cshape,
	std::vector<size_t>& partial, std::vector<size_t>& incomp)
{
	incomp = partial = make_partial(fuzzer, cshape);
	for (size_t i = 0; i < partial.size(); ++i)
	{
		if (partial[i] != 0)
		{
			incomp[i]++;
		}
	}
}

std::vector<size_t> make_incompatible (std::vector<size_t> shape)
{
	for (size_t i = 0; i < shape.size(); i++)
	{
		shape[i]++;
	}
	return shape;
}

std::vector<size_t> random_shape (testify::fuzz_test* fuzzer, range<size_t> ranks)
{
	size_t rank = ranks.min_;
	if (ranks.min_ != ranks.max_)
	{
		rank = fuzzer->get_int(1, "rank", {ranks.min_, ranks.max_})[0];
	}
	if (rank == 0)
	{
		return std::vector<size_t>{};
	}
	if (rank == 1)
	{
		return fuzzer->get_int(1, "shape", {0, 21});
	}
	std::vector<size_t> shape = fuzzer->get_int(rank, "shape", {0, 21});
	std::string s = ioutil::Stream() << shape;
	fuzzer->ss_ << "shape<" << s << ">" << std::endl;
	return shape;
}

std::vector<size_t> random_def_shape (testify::fuzz_test* fuzzer, range<size_t> ranks, range<size_t> n)
{
	size_t rank = ranks.min_;
	if (rank == 0)
	{
		rank = 1;
	}
	if (ranks.min_ != ranks.max_)
	{
		rank = fuzzer->get_int(1, "rank", {ranks.min_, ranks.max_})[0];
	}
	if (rank < 2)
	{
		return fuzzer->get_int(1, "shape", {n.min_, n.max_});
	}
	// invariant: rank > 1
	size_t maxvalue = 0;
	size_t minvalue = 0;
	size_t lowercut = 0;
	if (rank > 5) lowercut = 1;
	for (size_t i = n.max_; i > lowercut; i /= rank)
	{
		maxvalue++;
	}
	for (size_t i = n.min_; i > lowercut; i /= rank)
	{
		minvalue++;
	}

	// we don't care if minvalue overapproximates
	size_t ncorrection = 0;
	size_t realn = std::pow((double)maxvalue, (double)rank);
	if (realn > n.max_)
	{
		for (size_t error = realn - n.max_; error > 0; error /= maxvalue)
		{
			ncorrection++;
		}
	}

	std::vector<size_t> shape;
	if (ncorrection == rank)
	// we would need too many corrections, make a shape
	// one dimension at a time (sacrificing rank if necessary)
	{
		if (minvalue < n.min_)
		// we don't want our output too small either (avoid bad inputs in tests)
		{
			maxvalue += n.min_ - minvalue;
			minvalue = n.min_;
		}
		for (size_t i = 0; i < rank; i++)
		{
			size_t shapei = maxvalue;
			if (minvalue != maxvalue)
			{
				shapei = fuzzer->get_int(1, ioutil::Stream() << "shape i=" << i, 
					{minvalue, maxvalue})[0];
			}
			shape.push_back(shapei);
			n.max_ /= shapei;
			if (n.max_ < maxvalue)
			{
				break; // stop early
			}
		}
	}
	else
	{
		std::vector<size_t> shape2;
		if (rank-ncorrection)
		{
			shape = fuzzer->get_int(rank-ncorrection, "shapepart1", {minvalue, maxvalue});
		}
		if (ncorrection)
		{
			shape2 = fuzzer->get_int(ncorrection, "shapepart2", {minvalue, maxvalue-1});
		}
		shape.insert(shape.end(), shape2.begin(), shape2.end());
	}
	size_t nelems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
	if (nelems > 10000)
	{
		// warn
	}
	std::string s = ioutil::Stream() << shape;
	fuzzer->ss_ << "shape<" << s << ">" << std::endl;
	return shape;
}

std::vector<size_t> random_undef_shape (testify::fuzz_test* fuzzer, range<size_t> ranks)
{
	std::vector<size_t> rlist = random_def_shape(fuzzer, ranks);
	size_t nzeros = fuzzer->get_int(1, "nzeros", {1, 5})[0];
	for (size_t i = 0; i < nzeros; i++)
	{
		size_t zidx = fuzzer->get_int(1, "zidx", {0, rlist.size()})[0];
		rlist.insert(rlist.begin()+zidx, 0);
	}
	return rlist;
}

void random_shapes (testify::fuzz_test* fuzzer, std::vector<size_t>& partial, std::vector<size_t>& complete)
{
	complete = random_def_shape(fuzzer);
	partial = make_partial(fuzzer, complete);
}

}

#endif
