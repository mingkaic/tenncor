/*!
 *
 *  operator.hpp
 *  llo
 *
 *  Purpose:
 *  define low level operators on continguous data
 *
 */

#include <cstring>
#include <cmath>
#include <functional>

#include "util/rand.hpp"

#include "ade/shape.hpp"

#ifndef LLO_OPERATOR_HPP
#define LLO_OPERATOR_HPP

namespace llo
{

/*! Generic unary operation */
template <typename T>
void unary (T* out, const T* in, size_t n,
	std::function<T(const T&)> f)
{
	for (size_t i = 0; i < n; ++i)
	{
		out[i] = f(in[i]);
	}
}

/*! Absolute */
template <typename T>
void abs (T* out, const T* in, size_t n)
{
	unary<T>(out, in, n, [](const T& src) { return std::abs(src); });
}

template <>
void abs<uint8_t> (uint8_t* out, const uint8_t* in, size_t n);

template <>
void abs<uint16_t> (uint16_t* out, const uint16_t* in, size_t n);

template <>
void abs<uint32_t> (uint32_t* out, const uint32_t* in, size_t n);

template <>
void abs<uint64_t> (uint64_t* out, const uint64_t* in, size_t n);

/*! Negative */
template <typename T>
void neg (T* out, const T* in, size_t n)
{
	unary<T>(out, in, n, [](const T& src) { return -src; });
}

template <>
void neg<uint8_t> (uint8_t* out, const uint8_t* in, size_t n);

template <>
void neg<uint16_t> (uint16_t* out, const uint16_t* in, size_t n);

template <>
void neg<uint32_t> (uint32_t* out, const uint32_t* in, size_t n);

template <>
void neg<uint64_t> (uint64_t* out, const uint64_t* in, size_t n);

/*! Bitwise Not */
template <typename T>
void bit_not (T* out, const T* in, size_t n)
{
	unary<T>(out, in, n, [](const T& src) { return !src; });
}

/*! Sine */
template <typename T>
void sin (T* out, const T* in, size_t n)
{
	unary<T>(out, in, n, [](const T& src) { return std::sin(src); });
}

/*! Cosine */
template <typename T>
void cos (T* out, const T* in, size_t n)
{
	unary<T>(out, in, n, [](const T& src) { return std::cos(src); });
}

/*! Tangent */
template <typename T>
void tan (T* out, const T* in, size_t n)
{
	unary<T>(out, in, n, [](const T& src) { return std::tan(src); });
}

/*! Exponent */
template <typename T>
void exp (T* out, const T* in, size_t n)
{
	unary<T>(out, in, n, [](const T& src) { return std::exp(src); });
}

/*! Natural log */
template <typename T>
void log (T* out, const T* in, size_t n)
{
	unary<T>(out, in, n, [](const T& src) { return std::log(src); });
}

/*! Square root */
template <typename T>
void sqrt (T* out, const T* in, size_t n)
{
	unary<T>(out, in, n, [](const T& src) { return std::sqrt(src); });
}

/*! Round */
template <typename T>
void round (T* out, const T* in, size_t n)
{
	unary<T>(out, in, n, [](const T& src) { return std::round(src); });
}

/*! Flip */
template <typename T>
void flip (T* out, const T* in, ade::Shape shape, uint8_t dim)
{
	size_t n = shape.n_elems();
	std::vector<ade::DimT> slist = shape.as_list();
	uint8_t rank = slist.size();
	if (dim >= rank)
	{
		util::handle_error("attempting to flip a dimension beyond shape rank",
			util::ErrArg<size_t>("dim", dim),
			util::ErrArg<size_t>("rank", rank));
	}
	std::vector<ade::DimT> coord;
	ade::DimT dlimit = slist[dim] - 1;
	for (size_t i = 0; i < n; ++i)
	{
		coord = coordinate(shape, i);
		coord[dim] = dlimit - coord[dim];
		out[i] = in[index(shape, coord)];
	}
}

/*! Permute coordinates according to order */
template <typename T>
void permute (T* out, const T* in, ade::Shape outshape, ade::Shape shape,
	std::vector<uint8_t> order)
{
	size_t n = shape.n_elems();
	uint8_t orig_rank = shape.n_rank();
	bool visited[ade::rank_cap];
	std::memset(visited, false, sizeof(ade::DimT) * ade::rank_cap);
	for (size_t i = 0, n = order.size(); i < n; ++i)
	{
		visited[order[i]] = true;
	}
	for (size_t i = 0, n = orig_rank; i < n; ++i)
	{
		if (false == visited[i])
		{
			order.push_back(i);
		}
	}

	uint8_t norder = order.size();
	std::vector<ade::DimT> coords(orig_rank);
	std::vector<ade::DimT> converted(norder);
	for (ade::NElemT srci = 0; srci < n; ++srci)
	{
		coords = coordinate(shape, srci);

		for (uint8_t i = 0; i < norder; ++i)
		{
			converted[i] = coords[order[i]];
		}

		ade::NElemT desti = index(outshape, converted);
		out[desti] = in[srci];
	}
}

/*! Shape's n_elems */
template <typename T>
void n_elems (T& out, const ade::Shape& in)
{
	out = in.n_elems();
}

/*! Shape's dimension value at dim */
template <typename T>
void n_dims (T& out, const ade::Shape& in, uint8_t dim)
{
	out = in.at(dim);
}

/*! Get first flat index of the max value */
template <typename T>
void arg_max (T& out, const T* in, size_t n)
{
	size_t temp = 0;
	for (size_t i = 1; i < n; ++i)
	{
		if (in[temp] < in[i])
		{
			temp = i;
		}
	}
	out = temp;
}

/*! Get the max value */
template <typename T>
void reduce_max (T& out, const T* in, size_t n)
{
	out = in[0];
	for (size_t i = 1; i < n; ++i)
	{
		if (out < in[i])
		{
			out = in[i];
		}
	}
}

/*! Get the sum of all value */
template <typename T>
void reduce_sum (T& out, const T* in, size_t n)
{
	out = in[0];
	for (size_t i = 1; i < n; ++i)
	{
		out += in[i];
	}
}

/*! Generic binary operation */
template <typename T>
void binary (T* out, const T* a, size_t an, const T* b, size_t bn,
	std::function<T(const T&,const T&)> f)
{
	size_t n = std::max(an, bn);
	for (size_t i = 0; i < n; ++i)
	{
		out[i] = f(a[i % an], b[i % bn]);
	}
}

/*! Pow */
template <typename T>
void pow (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	binary<T>(out, a, an, b, bn,
		[](const T& b, const T& x) { return std::pow(b, x); });
}

/*! Add */
template <typename T>
void add (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	binary<T>(out, a, an, b, bn,
		[](const T& a, const T& b) { return a + b; });
}

/*! Sub */
template <typename T>
void sub (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	binary<T>(out, a, an, b, bn,
		[](const T& a, const T& b) { return a - b; });
}

/*! Mul */
template <typename T>
void mul (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	binary<T>(out, a, an, b, bn,
		[](const T& a, const T& b) { return a * b; });
}

/*! Div */
template <typename T>
void div (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	binary<T>(out, a, an, b, bn,
		[](const T& a, const T& b) { return a / b; });
}

/*! Equality */
template <typename T>
void eq (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	binary<T>(out, a, an, b, bn,
		[](const T& a, const T& b) { return a == b; });
}

/*! Non-equality */
template <typename T>
void neq (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	binary<T>(out, a, an, b, bn,
		[](const T& a, const T& b) { return a != b; });
}

/*! Less than */
template <typename T>
void lt (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	binary<T>(out, a, an, b, bn,
		[](const T& a, const T& b) { return a < b; });
}

/*! Greater than */
template <typename T>
void gt (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	binary<T>(out, a, an, b, bn,
		[](const T& a, const T& b) { return a > b; });
}

/*! Randomly generated values according to binomial distribution */
template <typename T>
void rand_binom (T* out, const T* a, size_t an, const double* b, size_t bn)
{
	size_t n = std::max(an, bn);
	for (size_t i = 0; i < n; ++i)
	{
		std::binomial_distribution<T> dist(a[i % an], b[i % bn]);
		out[i] = dist(util::get_engine());
	}
}

template <>
void rand_binom<double> (double* out,
	const double* a, size_t an, const double* b, size_t bn);

template <>
void rand_binom<float> (float* out,
	const float* a, size_t an, const double* b, size_t bn);

/*! Randomly generated values according to uniform distribution */
template <typename T>
void rand_uniform (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	binary<T>(out, a, an, b, bn,
	[](const T& a, const T& b)
	{
		std::uniform_int_distribution<T> dist(a, b);
		return dist(util::get_engine());
	});
}

template <>
void rand_uniform<double> (double* out,
	const double* a, size_t an, const double* b, size_t bn);

template <>
void rand_uniform<float> (float* out,
	const float* a, size_t an, const float* b, size_t bn);

/*! Randomly generated values according to normal distribution */
template <typename T>
void rand_normal (T* out, const T* a, size_t an, const T* b, size_t bn)
{
	throw std::bad_function_call();
}

template <>
void rand_normal<float> (float* out,
	const float* a, size_t an, const float* b, size_t bn);

template <>
void rand_normal<double> (double* out,
	const double* a, size_t an, const double* b, size_t bn);

/*! Matrix multiplication */
template <typename T>
void matmul (T* out, const T* a, const T* b,
	const ade::Shape& ashape, const ade::Shape& bshape,
	uint8_t agroup_idx, uint8_t bgroup_idx)
{
	auto ita = ashape.begin();
	auto itb = bshape.begin();
	ade::NElemT dim_x = std::accumulate(itb, itb + bgroup_idx,
		1, std::multiplies<ade::NElemT>());
	ade::NElemT dim_y = std::accumulate(
		ita + agroup_idx, ita + ashape.n_rank(),
		1, std::multiplies<ade::NElemT>());
	ade::NElemT dim_z = std::accumulate(ita, ita + agroup_idx,
		1, std::multiplies<ade::NElemT>());

	for (size_t y = 0; y < dim_y; y++)
	{
		for (size_t x = 0; x < dim_x; x++)
		{
			size_t outidx = x + y * dim_x;
			out[outidx] = 0;
			for (size_t z = 0; z < dim_z; z++)
			{
				size_t aidx = dim_z * y + z;
				size_t bidx = x + dim_x * z;
				out[outidx] += a[aidx] * b[bidx];
			}
		}
	}
}

/*! Copy over data from nin to out repeating nin to fit when necessary. */
template <typename T>
void copyover (T* out, size_t nout, const T* in, size_t nin)
{
	size_t mult = nout / nin;
	for (size_t i = 0; i < mult; ++i)
	{
		std::memcpy(out + i * nin, in, sizeof(T) * nin);
	}
	if (size_t leftover = nout % nin)
	{
		std::memcpy(out + mult * nin, in, sizeof(T) * leftover);
	}
}

}

#endif /* LLO_OPERATOR_HPP */
