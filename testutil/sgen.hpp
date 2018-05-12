#include "testify/fuzz/fuzz.hpp"

#include "ioutil/stream.hpp"

#ifndef TESTUTIL_SGEN_HPP
#define TESTUTIL_SGEN_HPP

namespace testutil
{

template <typename T>
struct range
{
	range (T ep1, T ep2) : 
		min_(std::min(ep1, ep2)), 
		max_(std::max(ep1, ep2)) {}

	T min_, max_;
};

// generate shapes as lists

std::vector<size_t> make_partial (testify::fuzz_test* fuzzer, std::vector<size_t> shape);

void make_incom_partials (testify::fuzz_test* fuzzer, std::vector<size_t> cshape,
	std::vector<size_t>& partial, std::vector<size_t>& incomp);

std::vector<size_t> make_incompatible (std::vector<size_t> shape);

std::vector<size_t> random_shape (testify::fuzz_test* fuzzer, range<size_t> ranks = {0, 12});

std::vector<size_t> random_def_shape (testify::fuzz_test* fuzzer, range<size_t> ranks = {2, 12}, range<size_t> n = {17, 7341});

std::vector<size_t> random_undef_shape (testify::fuzz_test* fuzzer, range<size_t> ranks = {2, 12});

void random_shapes (testify::fuzz_test* fuzzer, std::vector<size_t>& partial, std::vector<size_t>& complete);

}

#endif /* TESTUTIL_SGEN_HPP */
