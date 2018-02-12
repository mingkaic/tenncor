/*!
 *
 *  data_op.hpp
 *  cnnet
 *
 *  Purpose:
 *  performs numerical data on a void array
 *
 *  Created by Mingkai Chen on 2018-1-14.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <vector>
#include <unordered_map>

#include "include/tensor/type.hpp"
#include "include/tensor/tensorshape.hpp"

#pragma once
#ifndef TENNCOR_DATA_OP_HPP
#define TENNCOR_DATA_OP_HPP

namespace nnet
{

using VARR = std::pair<void*,tensorshape>;

using VFUNC = std::function<void(VARR,std::vector<VARR>)>;

std::unordered_set<std::string> all_ops (void);

void operate (std::string opname, TENS_TYPE type, VARR dest, std::vector<VARR> src);

#ifndef TENNCOR_D_UNARY_HPP
#define TENNCOR_D_UNARY_HPP

template <typename T>
void abs (VARR dest, std::vector<VARR> srcs);

template <typename T>
void neg (VARR dest, std::vector<VARR> srcs);

template <typename T>
void sin (VARR dest, std::vector<VARR> srcs);

template <typename T>
void cos (VARR dest, std::vector<VARR> srcs);

template <typename T>
void tan (VARR dest, std::vector<VARR> srcs);

template <typename T>
void csc (VARR dest, std::vector<VARR> srcs);

template <typename T>
void sec (VARR dest, std::vector<VARR> srcs);

template <typename T>
void cot (VARR dest, std::vector<VARR> srcs);

template <typename T>
void exp (VARR dest, std::vector<VARR> srcs);

template <typename T>
void ln (VARR dest, std::vector<VARR> srcs);

template <typename T>
void sqrt (VARR dest, std::vector<VARR> srcs);

template <typename T>
void round (VARR dest, std::vector<VARR> srcs);

// aggregation

template <typename T>
void argmax (VARR dest, std::vector<VARR> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.n_elems() == 1);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = srcshape.n_elems();
    auto it = std::max_element(s, s + n);
    d[0] = std::distance(s, it);
}

template <typename T>
void max (VARR dest, std::vector<VARR> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.n_elems() == 1);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = srcshape.n_elems();
    d[0] = *(std::max_element(s, s + n));
}

template <typename T>
void sum (VARR dest, std::vector<VARR> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.n_elems() == 1);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = srcshape.n_elems();
    d[0] = std::accumulate(s, s + n, 0);
}

#endif /* TENNCOR_D_UNARY_HPP */

#ifndef TENNCOR_D_NNARY_HPP
#define TENNCOR_D_NNARY_HPP

template <typename T>
void pow (VARR dest, std::vector<VARR> srcs);

template <typename T>
void add (VARR dest, std::vector<VARR> srcs);

template <typename T>
void sub (VARR dest, std::vector<VARR> srcs);

template <typename T>
void mul (VARR dest, std::vector<VARR> srcs);

template <typename T>
void div (VARR dest, std::vector<VARR> srcs);

template <typename T>
void eq (VARR dest, std::vector<VARR> srcs);

template <typename T>
void neq (VARR dest, std::vector<VARR> srcs);

template <typename T>
void lt (VARR dest, std::vector<VARR> srcs);

template <typename T>
void gt (VARR dest, std::vector<VARR> srcs);

template <typename T>
void rand_binom (VARR dest, std::vector<VARR> srcs);

template <typename T>
void rand_uniform (VARR dest, std::vector<VARR> srcs);

template <>
void rand_uniform<float> (VARR dest, std::vector<VARR> srcs);

template <>
void rand_uniform<double> (VARR dest, std::vector<VARR> srcs);

template <typename T>
void rand_normal (VARR dest, std::vector<VARR> srcs);

#endif /* TENNCOR_D_NNARY_HPP */

}

#include "src/operations/d_unary.ipp"

#include "src/operations/d_nnary.ipp"

#endif /* TENNCOR_DATA_OP_HPP */
