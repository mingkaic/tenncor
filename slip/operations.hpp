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

#include "clay/state.hpp"

#pragma once
#ifndef SLIP_OPERATIONS_HPP
#define SLIP_OPERATIONS_HPP

namespace slip
{

template <typename T>
T* safe_get (std::weak_ptr<char> ptr)
{
    if (ptr.expired())
    {
        throw std::exception();
    }
    return (T*) ptr.lock().get();
} 

template <typename T>
void abs (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::abs(s[src_mul * i]);
	}
}

// abs with unsigned is optimized
template <>
void abs<uint8_t> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void abs<uint16_t> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void abs<uint32_t> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void abs<uint64_t> (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void neg (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = -s[src_mul * i];
	}
}

// neg with unsigned is not acceptable
template <>
void neg<uint8_t> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void neg<uint16_t> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void neg<uint32_t> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void neg<uint64_t> (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void logic_not (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = (T) !s[src_mul * i];
	}
}

template <typename T>
void sin (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::sin(s[src_mul * i]);
	}
}

template <typename T>
void cos (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::cos(s[src_mul * i]);
	}
}

template <typename T>
void tan (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::tan(s[src_mul * i]);
	}
}

template <typename T>
void exp (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::exp(s[src_mul * i]);
	}
}

template <typename T>
void log (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::log(s[src_mul * i]);
	}
}

template <typename T>
void sqrt (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::sqrt(s[src_mul * i]);
	}
}

template <typename T>
void round (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::round(s[src_mul * i]);
	}
}

template <typename T>
void pow (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void add (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void sub (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void mul (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void div (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void eq (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void neq (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void lt (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void gt (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void rand_binom (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void rand_uniform (clay::State& dest, std::vector<clay::State> srcs);

template <>
void rand_uniform<float> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void rand_uniform<double> (clay::State& dest, std::vector<clay::State> srcs);

// binomial distribution with decimal types is not acceptable
template <>
void rand_binom<float> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void rand_binom<double> (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void rand_normal (clay::State& dest, std::vector<clay::State> srcs);

template <>
void rand_normal<float> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void rand_normal<double> (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void matmul (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void argmax (size_t i, void* accum, const void* arr)
{
	T* tarr = (T*) arr;
	T* data = (T*) accum;
	size_t prev = *data;
	if (tarr[prev] < tarr[i])
	{
		*data = i;
	}
}

template <typename T>
void max (size_t i, void* accum, const void* arr)
{
	T* tarr = (T*) arr;
	T* data = (T*) accum;
	if (*data < tarr[i])
	{
		*data = tarr[i];
	}
}

template <typename T>
void sum (size_t i, void* accum, const void* arr)
{
	T* tarr = (T*) arr;
	T* data = (T*) accum;
	*data += tarr[i];
}

}

#endif /* SLIP_OPERATIONS_HPP */
