//
// Created by Mingkai Chen on 2017-03-09.
//

#include <vector>
#include <iostream>

#include "include/tensor/tensorshape.hpp"
#include "include/tensor/tensor_handler.hpp"

#include "tests/include/mocks/mock_actor.h"
#include "tests/include/utils/fuzz.h"


#ifndef UTIL_TEST_H
#define UTIL_TEST_H


using namespace nnet;


bool tensorshape_equal (
	const tensorshape& ts1,
	const tensorshape& ts2);


bool tensorshape_equal (
	const tensorshape& ts1,
	std::vector<size_t>& ts2);


void print (std::vector<double> raw);


tensorshape make_partial (FUZZ::fuzz_test* fuzzer, std::vector<size_t> shapelist);


tensorshape make_incompatible (std::vector<size_t> shapelist);


// make partial full, but incompatible to comp
tensorshape make_full_incomp (std::vector<size_t> partial, std::vector<size_t> complete);


tensorshape padd(std::vector<size_t> shapelist, size_t nfront, size_t nback);


std::vector<std::vector<double> > doubleDArr(std::vector<double> v, std::vector<size_t> dimensions);


tensorshape random_shape (FUZZ::fuzz_test* fuzzer);


tensorshape random_def_shape (FUZZ::fuzz_test* fuzzer, int lowerrank = 2, int upperrank = 11, size_t minn = 17, size_t maxn = 7341);


itens_actor* adder (out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs, nnet::TENS_TYPE type);


#endif /* UTIL_TEST_H */
