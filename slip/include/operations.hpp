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

#include "slip/rand.hpp"

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

#ifndef SLIP_CAST_HPP
#define SLIP_CAST_HPP

template <typename T>
void cast (clay::State& dest, std::vector<clay::State> srcs);

#endif /* SLIP_CAST_HPP */

template <typename T>
void unary (clay::State& dest, std::vector<clay::State> srcs,
	std::function<T(const T&)> f)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest.data_);
	const T* s = safe_get<const T>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = f(s[src_mul * i]);
	}
}

template <typename T>
void abs (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::abs(src); });
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
	unary<T>(dest, srcs, [](const T& src) { return -src; });
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
	unary<T>(dest, srcs, [](const T& src) { return !src; });
}

template <typename T>
void sin (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::sin(src); });
}

template <typename T>
void cos (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::cos(src); });
}

template <typename T>
void tan (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::tan(src); });
}

template <typename T>
void exp (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::exp(src); });
}

template <typename T>
void log (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::log(src); });
}

template <typename T>
void sqrt (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::sqrt(src); });
}

template <typename T>
void round (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::round(src); });
}

template <typename T>
void binary (clay::State& dest, std::vector<clay::State> srcs,
	std::function<T(const T&,const T&)> f)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape& srcshape0 = srcs.front().shape_;
	clay::Shape& srcshape1 = srcs.back().shape_;
	T* d = safe_get<T>(dest.data_);
	const T* a = safe_get<const T>(srcs.front().data_);
	const T* b = safe_get<const T>(srcs.back().data_);
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = f(a[i * left_mul], b[i * right_mul]);
	}
}

template <typename T>
void pow (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& b, const T& x) { return std::pow(b, x); });
}

template <typename T>
void add (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a + b; });
}

template <typename T>
void sub (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a - b; });
}

template <typename T>
void mul (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a * b; });
}

template <typename T>
void div (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a / b; });
}

template <typename T>
void eq (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a == b; });
}

template <typename T>
void neq (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a != b; });
}

template <typename T>
void lt (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a < b; });
}

template <typename T>
void gt (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a > b; });
}

template <typename T>
void rand_binom (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape& srcshape0 = srcs.front().shape_;
	clay::Shape& srcshape1 = srcs.back().shape_;
	T* d = safe_get<T>(dest.data_);
	const T* sn = safe_get<const T>(srcs.front().data_);
	const double* sp = safe_get<const double>(srcs.back().data_);
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		std::binomial_distribution<T> dist(sn[i * left_mul], sp[i * right_mul]);
		d[i] = dist(slip::get_generator());
	}
}

// binomial distribution with decimal types is not acceptable
template <>
void rand_binom<float> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void rand_binom<double> (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void rand_uniform (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs,
	[](const T& a, const T& b)
	{
		std::uniform_int_distribution<T> dist(a, b);
		return dist(slip::get_generator());
	});
}

// uniform distribution for decimal types
template <>
void rand_uniform<float> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void rand_uniform<double> (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void rand_normal (clay::State& dest, std::vector<clay::State> srcs)
{
	throw std::bad_function_call();
}

// random normal for decimal types only
template <>
void rand_normal<float> (clay::State& dest, std::vector<clay::State> srcs);

template <>
void rand_normal<double> (clay::State& dest, std::vector<clay::State> srcs);

template <typename T>
void transpose (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest.data_);
	const T* s = safe_get<const T>(srcs.front().data_);
	std::vector<uint64_t> perm;
	if (srcs.size() > 1)
	{
		clay::State& pstate = srcs[1];
		if (pstate.dtype_ != clay::UINT64)
		{
			throw std::exception();
		}
		uint64_t* ptr = safe_get<uint64_t>(pstate.data_);
		perm = std::vector<uint64_t>(ptr, ptr + pstate.shape_.n_elems());
	}
	else
	{
		perm = std::vector<uint64_t>(srcshape.rank());
		std::iota(perm.rbegin(), perm.rend(), 0);
	}
	std::vector<size_t> tmp_coord;
	std::vector<size_t> coord;
	for (size_t i = 0, n = destshape.n_elems();
		i < n; ++i)
	{
		coord = tmp_coord = clay::coordinate(destshape, i);
		for (size_t j = 0, permsize = perm.size();
			j < permsize; ++j)
		{
			coord[perm[j]] = tmp_coord[j];
		}
		d[i] = s[clay::index(srcshape, coord)];
	}
}

template <typename T>
void flip (clay::State& dest, std::vector<clay::State> srcs)
{
	if (srcs.size() != 2)
	{
		throw std::exception();
	}
	clay::Shape& shape = dest.shape_;
	T* d = safe_get<T>(dest.data_);
	const T* s = safe_get<const T>(srcs.front().data_);
	clay::State& dstate = srcs[1];
	if (dstate.dtype_ != clay::UINT64)
	{
		throw std::exception();
	}
	size_t ndims = dstate.shape_.n_elems();
	uint64_t* dims = safe_get<uint64_t>(dstate.data_);
	std::vector<size_t> slist = shape.as_list();
	std::vector<size_t> coord;
	for (size_t i = 0, n = shape.n_elems();
		i < n; ++i)
	{
		coord = clay::coordinate(shape, i);
		for (size_t j = 0; j < ndims; ++j)
		{
			coord[dims[j]] = slist[dims[j]] - coord[dims[j]] - 1;
		}
		d[i] = s[clay::index(shape, coord)];
	}
}

template <typename T>
void argmax (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest.data_);
	const T* s = safe_get<const T>(srcs.front().data_);
	size_t rank = srcshape.rank();
	if (srcs.size() > 1 && rank > 1)
	{
		uint64_t dim = *(safe_get<uint64_t>(srcs[1].data_));
		assert(rank > dim);
		std::vector<size_t> slist = srcshape.as_list();
		slist[dim] = 1;
		clay::Shape nilshape = slist;
		std::vector<size_t> coord;
		size_t n = nilshape.n_elems();
		size_t nd = srcshape[dim];
		for (size_t i = 0; i < n; ++i)
		{
			coord = coordinate(nilshape, i);
			d[i] = index(srcshape, coord);
			for (size_t j = 1; j < nd; ++j)
			{
				coord[dim] = j;
				size_t srcidx = index(srcshape, coord);
				if (s[(size_t) d[i]] < s[srcidx])
				{
					d[i] = srcidx;
				}
			}
		}
	}
	else
	{
		size_t n = srcshape.n_elems();
		*d = std::distance(s, std::max_element(s, s + n));
	}
}

template <typename T>
void max (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest.data_);
	const T* s = safe_get<const T>(srcs.front().data_);
	size_t rank = srcshape.rank();
	if (srcs.size() > 1 && rank > 1)
	{
		uint64_t dim = *(safe_get<uint64_t>(srcs[1].data_));
		assert(rank > dim);
		std::vector<size_t> slist = srcshape.as_list();
		slist[dim] = 1;
		clay::Shape nilshape = slist;
		std::vector<size_t> coord;
		size_t n = nilshape.n_elems();
		size_t nd = srcshape[dim];
		for (size_t i = 0; i < n; ++i)
		{
			coord = coordinate(nilshape, i);
			d[i] = s[index(srcshape, coord)];
			for (size_t j = 1; j < nd; ++j)
			{
				coord[dim] = j;
				size_t srcidx = index(srcshape, coord);
				if (d[i] < s[srcidx])
				{
					d[i] = s[srcidx];
				}
			}
		}
	}
	else
	{
		size_t n = srcshape.n_elems();
		*d = *(std::max_element(s, s + n));
	}
}

template <typename T>
void sum (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest.data_);
	const T* s = safe_get<const T>(srcs.front().data_);
	size_t rank = srcshape.rank();
	if (srcs.size() > 1 && rank > 1)
	{
		uint64_t dim = *(safe_get<uint64_t>(srcs[1].data_));
		assert(rank > dim);
		std::vector<size_t> slist = srcshape.as_list();
		slist[dim] = 1;
		clay::Shape nilshape = slist;
		std::vector<size_t> coord;
		size_t n = nilshape.n_elems();
		size_t nd = srcshape[dim];
		for (size_t i = 0; i < n; ++i)
		{
			coord = coordinate(nilshape, i);
			d[i] = s[index(srcshape, coord)];
			for (size_t j = 1; j < nd; ++j)
			{
				coord[dim] = j;
				size_t srcidx = index(srcshape, coord);
				d[i] += s[srcidx];
			}
		}
	}
	else
	{
		size_t n = srcshape.n_elems();
		*d = std::accumulate(s, s + n, (T) 0);
	}
}

template <typename T>
void expand (clay::State& dest, std::vector<clay::State> srcs)
{
	if (srcs.size() != 3)
	{
		throw std::exception();
	}
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest.data_);
	const T* s = safe_get<const T>(srcs.front().data_);
	clay::State& nstate = srcs[1];
	clay::State& dstate = srcs[2];
	if (nstate.dtype_ != clay::UINT64 ||
		dstate.dtype_ != clay::UINT64)
	{
		throw std::exception();
	}
	if (1 != nstate.shape_.n_elems() ||
		1 != dstate.shape_.n_elems())
	{
		throw std::exception();
	}
	uint64_t mul = *(safe_get<uint64_t>(nstate.data_));
	uint64_t dim = *(safe_get<uint64_t>(dstate.data_));
	std::vector<size_t> slist = srcshape.as_list();
	auto it = slist.begin();
	size_t innern = std::accumulate(it, it + dim, 1, std::multiplies<size_t>());
	size_t outern = srcshape.n_elems();
	size_t repeats = outern / innern;
	size_t nexpansion = innern * mul;
	for (size_t j = 0; j < repeats; ++j)
	{
		for (size_t i = 0; i < mul; ++i)
		{
			size_t outidx = j * nexpansion + i * innern;
			size_t inidx = j * innern;
			std::memcpy(d + outidx, s + inidx, innern * sizeof(T));
		}
	}
}

void n_elems (clay::State& dest, std::vector<clay::State> srcs);

void n_dims (clay::State& dest, std::vector<clay::State> srcs);

#ifndef SLIP_MATMUL_HPP
#define SLIP_MATMUL_HPP

template <typename T>
void matmul (clay::State& dest, std::vector<clay::State> srcs);

#endif /* SLIP_MATMUL_HPP */

}

#include "slip/include/cast.ipp"
#include "slip/include/matmul.ipp"

#endif /* SLIP_OPERATIONS_HPP */
