#include <cmath>

#include "sand/operator.hpp"

#include "util/error.hpp"
#include "util/rand.hpp"

#ifndef SAND_BINARY_HPP
#define SAND_BINARY_HPP

namespace sand
{

template <typename T>
void binary (NodeInfo dest, std::vector<NodeInfo>& srcs,
	std::function<T(const T&,const T&)> f)
{
	if (2 != srcs.size())
	{
		handle_error("binary requires 2 arguments",
			ErrArg<size_t>{"num_args", srcs.size()});
	}

	T* a = (T*) srcs[0].data_;
	T* b = (T*) srcs[1].data_;
	T* c = (T*) dest.data_;

	NElemT an = srcs[0].shape_.n_elems();
	NElemT bn = srcs[1].shape_.n_elems();
	NElemT n = dest.shape_.n_elems();

	for (NElemT i = 0; i < n; ++i)
	{
		c[i] = f(a[i % an], b[i % bn]);
	}
}

template <typename T>
void pow (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<T>(dest, srcs, [](const T& b, const T& x) { return std::pow(b, x); });
}

template <typename T>
void add (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a + b; });
}

template <typename T>
void sub (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a - b; });
}

template <typename T>
void mul (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a * b; });
}

template <typename T>
void div (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a / b; });
}

template <typename T>
void eq (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a == b; });
}

template <typename T>
void neq (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a != b; });
}

template <typename T>
void lt (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a < b; });
}

template <typename T>
void gt (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a > b; });
}

template <typename T>
void rand_binom (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	T* a = (T*) srcs[0].data_;
	double* b = (double*) srcs[1].data_;
	T* c = (T*) dest.data_;

	NElemT an = srcs[0].shape_.n_elems();
	NElemT bn = srcs[1].shape_.n_elems();
	NElemT n = dest.shape_.n_elems();

	for (NElemT i = 0; i < n; ++i)
	{
		std::binomial_distribution<T> dist(a[i % an], b[i % bn]);
		c[i] = dist(get_engine());
	}
}

template <typename T>
void rand_uniform (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<T>(dest, srcs,
	[](const T& a, const T& b)
	{
		std::uniform_int_distribution<T> dist(a, b);
		return dist(get_engine());
	});
}

template <typename T>
void rand_normal (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	throw std::bad_function_call();
}

template <>
void rand_normal<float> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData);

template <>
void rand_normal<double> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData);

}

#endif /* SAND_BINARY_HPP */
