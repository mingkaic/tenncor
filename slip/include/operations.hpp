/*!
 *
 *  operations.hpp
 *  slip
 *
 *  Purpose:
 *  generic operation definition
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <cstring>
#include <cassert>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>

#include "clay/state.hpp"

#include "mold/state_range.hpp"

#include "slip/rand.hpp"

#pragma once
#ifndef SLIP_OPERATIONS_HPP
#define SLIP_OPERATIONS_HPP

namespace slip
{

#ifndef SLIP_CAST_HPP
#define SLIP_CAST_HPP

template <typename T>
void cast (clay::State& dest, std::vector<mold::StateRange> srcs);

#endif /* SLIP_CAST_HPP */

#ifndef SLIP_UNARY_HPP
#define SLIP_UNARY_HPP

template <typename T>
T* safe_get (clay::State& state);

template <typename T>
T* safe_get (mold::StateRange& state);

template <typename T>
void copyover (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void abs (clay::State& dest, std::vector<mold::StateRange> srcs);

// abs with unsigned is optimized
template <>
void abs<uint8_t> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <>
void abs<uint16_t> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <>
void abs<uint32_t> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <>
void abs<uint64_t> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void neg (clay::State& dest, std::vector<mold::StateRange> srcs);

// neg with unsigned is not acceptable
template <>
void neg<uint8_t> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <>
void neg<uint16_t> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <>
void neg<uint32_t> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <>
void neg<uint64_t> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void logic_not (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void sin (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void cos (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void tan (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void exp (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void log (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void sqrt (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void round (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void transpose (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void flip (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void arg_max (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void is_max (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void rmax (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void rsum (clay::State& dest, std::vector<mold::StateRange> srcs);

#endif /* SLIP_UNARY_HPP */

#ifndef SLIP_BINARY_HPP
#define SLIP_BINARY_HPP

template <typename T>
void binary (clay::State& dest, std::vector<mold::StateRange> srcs,
	std::function<T(const T&,const T&)> f);

template <typename T>
void pow (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void add (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void sub (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void mul (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void div (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void eq (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void neq (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void lt (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void gt (clay::State& dest, std::vector<mold::StateRange> srcs);

// binomial distribution with decimal types is not acceptable
template <typename T>
void rand_binom (clay::State& dest, std::vector<mold::StateRange> srcs);

template <>
void rand_binom<float> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <>
void rand_binom<double> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void rand_uniform (clay::State& dest, std::vector<mold::StateRange> srcs);

// uniform distribution for decimal types
template <>
void rand_uniform<float> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <>
void rand_uniform<double> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void rand_normal (clay::State& dest, std::vector<mold::StateRange> srcs);

// random normal for decimal types only
template <>
void rand_normal<float> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <>
void rand_normal<double> (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void expand (clay::State& dest, std::vector<mold::StateRange> srcs);

void n_elems (clay::State& dest, std::vector<mold::StateRange> srcs);

void n_dims (clay::State& dest, std::vector<mold::StateRange> srcs);

#endif /* SLIP_BINARY_HPP */

#ifndef SLIP_MATMUL_HPP
#define SLIP_MATMUL_HPP

template <typename T>
void matmul (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void jacobian (clay::State& dest, std::vector<mold::StateRange> srcs);

template <typename T>
void trace_expand (clay::State& dest, std::vector<mold::StateRange> srcs);

#endif /* SLIP_MATMUL_HPP */

}

#include "slip/include/cast.ipp"
#include "slip/include/unary.ipp"
#include "slip/include/binary.ipp"
#include "slip/include/matmul.ipp"

#endif /* SLIP_OPERATIONS_HPP */
