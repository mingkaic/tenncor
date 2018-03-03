//
// Created by Mingkai Chen on 2017-03-10.
//

#include "fuzz/fuzz.hpp"

#include "utils/utils.hpp"
#include "tensor/tensorshape.hpp"

#ifndef TTEST_SGEN_HPP
#define TTEST_SGEN_HPP

namespace testutils
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

std::vector<size_t> make_incompatible (std::vector<size_t> shape);

std::vector<size_t> random_shape (testify::fuzz_test* fuzzer, range<size_t> ranks = {0, 12});

std::vector<size_t> random_def_shape (testify::fuzz_test* fuzzer, range<size_t> ranks = {2, 12}, range<size_t> n = {17, 7341});

std::vector<size_t> random_undef_shape (testify::fuzz_test* fuzzer, range<size_t> ranks = {2, 12});

void random_shapes (testify::fuzz_test* fuzzer, std::vector<size_t>& partial, std::vector<size_t>& complete);


nnet::tensorshape padd(std::vector<size_t> shapelist, size_t nfront, size_t nback);

}

#endif /* TTEST_SGEN_HPP */
