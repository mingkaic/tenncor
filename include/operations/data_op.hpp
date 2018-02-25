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
#include <functional> // for std::bad_function_call();

#include "include/tensor/type.hpp"
#include "include/tensor/tensorshape.hpp"

#pragma once
#ifndef TENNCOR_DATA_OP_HPP
#define TENNCOR_DATA_OP_HPP

namespace nnet
{

using VARR_T = std::pair<void*,tensorshape>;

using CVAR_T = std::pair<const void*,tensorshape>;

using VFUNC_F = std::function<void(VARR_T,std::vector<CVAR_T>)>;

std::unordered_set<std::string> all_ops (void);

void operate (std::string opname, TENS_TYPE type, VARR_T dest, std::vector<CVAR_T> src);

#ifndef TENNCOR_D_UNARY_HPP
#define TENNCOR_D_UNARY_HPP

template <typename T>
void abs (VARR_T dest, std::vector<CVAR_T> srcs);

template <>
void abs<uint8_t> (VARR_T dest, std::vector<CVAR_T> srcs);

template <>
void abs<uint16_t> (VARR_T dest, std::vector<CVAR_T> srcs);

template <>
void abs<uint32_t> (VARR_T dest, std::vector<CVAR_T> srcs);

template <>
void abs<uint64_t> (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void neg (VARR_T dest, std::vector<CVAR_T> srcs);

template <>
void neg<uint8_t> (VARR_T, std::vector<CVAR_T>);

template <>
void neg<uint16_t> (VARR_T, std::vector<CVAR_T>);

template <>
void neg<uint32_t> (VARR_T, std::vector<CVAR_T>);

template <>
void neg<uint64_t> (VARR_T, std::vector<CVAR_T>);

template <typename T>
void logic_not (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape));
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = srcshape.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = (T) !s[src_mul * i];
	}
}

template <typename T>
void sin (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void cos (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void tan (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void csc (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void sec (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void cot (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void exp (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void log (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void sqrt (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void round (VARR_T dest, std::vector<CVAR_T> srcs);

// aggregation

template <typename T>
void argmax (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.n_elems() == 1);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = srcshape.n_elems();
	auto it = std::max_element(s, s + n);
	d[0] = std::distance(s, it);
}

template <typename T>
void max (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.n_elems() == 1);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = srcshape.n_elems();
	d[0] = *(std::max_element(s, s + n));
}

template <typename T>
void sum (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.n_elems() == 1);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = srcshape.n_elems();
	d[0] = std::accumulate(s, s + n, (T) 0);
}

#endif /* TENNCOR_D_UNARY_HPP */

#ifndef TENNCOR_D_NNARY_HPP
#define TENNCOR_D_NNARY_HPP

template <typename T>
void pow (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void add (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void sub (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void mul (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void div (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void eq (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void neq (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void lt (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void gt (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void rand_binom (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void rand_uniform (VARR_T dest, std::vector<CVAR_T> srcs);

template <>
void rand_uniform<float> (VARR_T dest, std::vector<CVAR_T> srcs);

template <>
void rand_uniform<double> (VARR_T dest, std::vector<CVAR_T> srcs);

template <typename T>
void rand_normal (VARR_T dest, std::vector<CVAR_T> srcs);

#endif /* TENNCOR_D_NNARY_HPP */

#ifndef TENNCOR_D_MATMUL_HPP
#define TENNCOR_D_MATMUL_HPP

template <typename T>
void matmul (VARR_T dest, std::vector<CVAR_T> srcs);

#endif /* TENNCOR_D_MATMUL_HPP */

}

#include "src/operations/d_unary.ipp"

#include "src/operations/d_nnary.ipp"

#include "src/operations/d_matmul.ipp"

#endif /* TENNCOR_DATA_OP_HPP */
