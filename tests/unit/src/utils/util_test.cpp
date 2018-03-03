//
// Created by Mingkai Chen on 2017-03-10.
//

#include "tests/unit/include/utils/util_test.hpp"
#include "tests/unit/include/utils/fuzz.h"


#ifndef UTIL_TTEST_HPP
#define UTIL_TTEST_HPP



tensorshape padd(std::vector<size_t> shapelist, size_t nfront, size_t nback)
{
	std::vector<size_t> out(nfront, 1);
	out.insert(out.end(), shapelist.begin(), shapelist.end());
	out.insert(out.end(), nback, 1);
	return tensorshape(out);
}

std::vector<std::vector<double> > doubleDArr(std::vector<double> v, std::vector<size_t> dimensions)
{
	assert(dimensions.size() == 2);
	size_t cols = dimensions[0];
	size_t rows = dimensions[1];
	std::vector<std::vector<double> > mat(rows);
	auto it = v.begin();
	for (size_t i = 0; i < rows; i++)
	{
		mat[i].insert(mat[i].end(), it, it+cols);
		it+=cols;
	}
	return mat;
}


itens_actor* adder (out_wrapper<void>& dest,
	std::vector<in_wrapper<void> >& srcs, nnet::TENS_TYPE type);


#endif /* UTIL_TTEST_HPP */
