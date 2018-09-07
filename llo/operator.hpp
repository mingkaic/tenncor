#include <cstring>
#include <cmath>
#include <assert>

#include "ade/shape.hpp"

#ifndef LLO_OPERATOR_HPP
#define LLO_OPERATOR_HPP

template <typename OUTTYPE, typename INTYPE>
void typecast (std::vector<OUTTYPE>& out, const std::vector<INTYPE>& in)
{
	size_t n = out.size();
	assert(n == in.size());
	std::vector<OUTTYPE> temp(in.begin(), in.end());
	std::memcpy(&out[0], &temp[0], sizeof(OUTTYPE) * n);
}

template <typename T>
void unary (std::vector<T>& out, const std::vector<T>& in,
	std::function<T(const T&)> f)
{
	size_t n = out.size();
	assert(n == in.size());
	for (size_t i = 0; i < n; ++i)
	{
		out[i] = f(in[i]);
	}
}

template <typename T>
void abs (std::vector<T>& out, const std::vector<T>& in)
{
	unary<T>(out, in, [](const T& src) { return std::abs(src); });
}

template <>
void abs<uint8_t> (std::vector<T>& out, const std::vector<T>& in);

template <>
void abs<uint16_t> (std::vector<T>& out, const std::vector<T>& in);

template <>
void abs<uint32_t> (std::vector<T>& out, const std::vector<T>& in);

template <>
void abs<uint64_t> (std::vector<T>& out, const std::vector<T>& in);

template <typename T>
void neg (std::vector<T>& out, const std::vector<T>& in)
{
	unary<T>(out, in, [](const T& src) { return -src; });
}

template <>
void neg<uint8_t> (std::vector<T>& out, const std::vector<T>& in);

template <>
void neg<uint16_t> (std::vector<T>& out, const std::vector<T>& in);

template <>
void neg<uint32_t> (std::vector<T>& out, const std::vector<T>& in);

template <>
void neg<uint64_t> (std::vector<T>& out, const std::vector<T>& in);

template <typename T>
void logic_not (std::vector<T>& out, const std::vector<T>& in)
{
	unary<T>(out, in, [](const T& src) { return !src; });
}

template <typename T>
void sin (std::vector<T>& out, const std::vector<T>& in)
{
	unary<T>(out, in, [](const T& src) { return std::sin(src); });
}

template <typename T>
void cos (std::vector<T>& out, const std::vector<T>& in)
{
	unary<T>(out, in, [](const T& src) { return std::cos(src); });
}

template <typename T>
void tan (std::vector<T>& out, const std::vector<T>& in)
{
	unary<T>(out, in, [](const T& src) { return std::tan(src); });
}

template <typename T>
void exp (std::vector<T>& out, const std::vector<T>& in)
{
	unary<T>(out, in, [](const T& src) { return std::exp(src); });
}

template <typename T>
void log (std::vector<T>& out, const std::vector<T>& in)
{
	unary<T>(out, in, [](const T& src) { return std::log(src); });
}

template <typename T>
void sqrt (std::vector<T>& out, const std::vector<T>& in)
{
	unary<T>(out, in, [](const T& src) { return std::sqrt(src); });
}

template <typename T>
void round (std::vector<T>& out, const std::vector<T>& in)
{
	unary<T>(out, in, [](const T& src) { return std::round(src); });
}

template <typename T>
void flip (std::vector<T>& out, const std::vector<T>& in, ade::Shape shape, uint8_t dim)
{
	size_t n = out.size();
	assert(n == in.size());
	assert(n == shape.n_elems());
	std::vector<DimT> slist = shape.as_list();
	std::vector<DimT> coord;
	for (size_t i = 0; i < n; ++i)
	{
		coord = coordinate(shape, i);
		coord[dim] = slist[dim] - coord[dim] - 1;
		out[i] = in[index(shape, coord)];
	}
}

template <typename T>
void permute (std::vector<T>& out, const std::vector<T>& in,
	ade::Shape outshape, ade::Shape shape, std::vector<uint8_t> order)
{
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
	std::vector<DimT> coords(orig_rank);
	std::vector<DimT> converted(norder);
	for (NElemT srci = 0; srci < n; ++srci)
	{
		coords = coordinate(shape, srci);

		for (uint8_t i = 0; i < norder; ++i)
		{
			converted[i] = coords[order[i]];
		}

		NElemT desti = index(outshape, coords);
		out[desti] = in[srci];
	}
}

template <typename T>
void n_elems (std::vector<T>& out, const ade::Shape& in)
{
	throw std::bad_function_call();
}

template <>
void n_elems<uint64_t> (std::vector<T>& out, const ade::Shape& in);

template <typename T>
void n_dims (std::vector<T>& out, const ade::Shape& in, uint8_t dim)
{
	throw std::bad_function_call();
}

template <>
void n_dims<uint8_t> (std::vector<T>& out, const ade::Shape& in, uint8_t dim);

template <typename T>
void arg_max (std::vector<T>& out, const std::vector<T>& in)
{
	assert(out.size() == 1);
	size_t temp = 0;
	for (size_t i = 1, n = in.size(); i < n; ++i)
	{
		if (in[temp] < s[i])
		{
			temp = i;
		}
	}
	out[0] = temp;
}

template <typename T>
void reduce_max (std::vector<T>& out, const std::vector<T>& in)
{
	assert(out.size() == 1);
	T temp = in[0];
	for (size_t i = 1, n = in.size(); i < n; ++i)
	{
		if (temp < s[i])
		{
			temp = s[i];
		}
	}
	out[0] = temp;
}

template <typename T>
void reduce_sum (std::vector<T>& out, const std::vector<T>& in)
{
	assert(out.size() == 1);
	T temp = in[0];
	for (size_t i = 1, n = in.size(); i < n; ++i)
	{
		temp += in[i];
	}
	out[0] = temp;
}

template <typename T>
void binary (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b,
	std::function<T(const T&,const T&)> f)
{
	size_t n = out.size();
	size_t an = a.size();
	size_t bn = b.size();
	assert(n == std::max(an, bn));

	for (size_t i = 0; i < n; ++i)
	{
		out[i] = f(a[i % an], b[i % bn]);
	}
}

template <typename T>
void pow (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	binary<T>(out, a, b, [](const T& b, const T& x) { return std::pow(b, x); });
}

template <typename T>
void add (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a + b; });
}

template <typename T>
void sub (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a - b; });
}

template <typename T>
void mul (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a * b; });
}

template <typename T>
void div (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a / b; });
}

template <typename T>
void eq (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a == b; });
}

template <typename T>
void neq (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a != b; });
}

template <typename T>
void lt (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a < b; });
}

template <typename T>
void gt (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a > b; });
}

template <typename T>
void rand_binom (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<double>& b)
{
	size_t an = a.size();
	size_t bn = b.size();
	size_t n = out.size();
	assert(n == std::max(an, bn));

	for (size_t i = 0; i < n; ++i)
	{
		std::binomial_distribution<T> dist(a[i % an], b[i % bn]);
		out[i] = dist(get_engine());
	}
}

template <typename T>
void rand_uniform (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	binary<T>(out, a, b,
	[](const T& a, const T& b)
	{
		std::uniform_int_distribution<T> dist(a, b);
		return dist(get_engine());
	});
}

template <typename T>
void rand_normal (std::vector<T>& out,
	const std::vector<T>& a, const std::vector<T>& b)
{
	throw std::bad_function_call();
}

template <>
void rand_normal<float> (std::vector<float>& out,
	const std::vector<float>& a, const std::vector<float>& b);

template <>
void rand_normal<double> (std::vector<double>& out,
	const std::vector<double>& a, const std::vector<double>& b);

template <typename T>
void matmul (std::vector<T>& out,
	const std::vector<float>& a, const std::vector<float>& b,
	const ade::Shape& ashape, const ade::Shape& bshape,
	uint8_t agroup_idx, uint8_t bgroup_idx)
{
	auto ita = ashape.begin();
	auto itb = bshape.begin();
	NElemT dim_x = std::accumulate(itb, itb + bgroup_idx,
		1, std::multiplies<NElemT>());
	NElemT dim_y = std::accumulate(ita + agroup_idx, ita + ashape.n_rank(),
		1, std::multiplies<NElemT>());
	NElemT dim_z = std::accumulate(ita, ita + agroup_idx,
		1, std::multiplies<NElemT>());

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

#endif /* LLO_OPERATOR_HPP */
