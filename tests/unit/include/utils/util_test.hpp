//
// Created by Mingkai Chen on 2017-03-09.
//

#include <vector>
#include <iostream>

#include "include/tensor/tensorshape.hpp"

#include "tests/unit/include/utils/fuzz.h"


#ifndef UTIL_TEST_H
#define UTIL_TEST_H


using namespace nnet;


std::vector<std::vector<double> > doubleDArr(std::vector<double> v, std::vector<size_t> dimensions);


itens_actor* adder (out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs, nnet::TENS_TYPE type);


#endif /* UTIL_TEST_H */

