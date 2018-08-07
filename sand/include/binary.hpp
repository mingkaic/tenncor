#include "sand/operator.hpp"

#include "util/error.hpp"

#ifndef SAND_BINARY_HPP
#define SAND_BINARY_HPP

template <typename T>
void add (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	if (2 != srcs.size())
	{
		handle_error("add requires 2 arguments",
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
		c[i] = a[i % an] + b[i % bn];
	}
}

template <typename T>
void mul (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	if (2 != srcs.size())
	{
		handle_error("mul requires 2 arguments",
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
		c[i] = a[i % an] * b[i % bn];
	}
}

#endif /* SAND_BINARY_HPP */
