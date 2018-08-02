#include <functional>

#include "sand/shape.hpp"
#include "sand/type.hpp"
#include "sand/opcode.hpp"
#include "util/error.hpp"

#ifndef OPERATOR_HPP
#define OPERATOR_HPP

struct NodeInfo
{
	char* data_;
	Shape shape_;
};

using Operation = std::function<void(NodeInfo&,std::vector<NodeInfo>&)>;

template <typename T>
void add (NodeInfo dest, std::vector<NodeInfo>& srcs)
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
void mul (NodeInfo dest, std::vector<NodeInfo>& srcs)
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

template <typename T>
void transpose (NodeInfo dest, std::vector<NodeInfo>& srcs)
{
	if (1 != srcs.size())
	{
		handle_error("transpose requires 1 arguments",
			ErrArg<size_t>{"num_args", srcs.size()});
	}

	NodeInfo& src = srcs[0];
	NElemT an = src.shape_.n_elems();
	NElemT n = dest.shape_.n_elems();

	if (an != n)
	{
		handle_error("transposing src to destination of incompatible size",
			ErrArg<size_t>{"ndest", n},
			ErrArg<size_t>{"nsrc", an});
	}

	T* destdata = (T*) dest.data_;
	T* srcdata = (T*) src.data_;

	NElemT srcx = src.shape_.group(0).n_elems();
	NElemT srcy = src.shape_.group(1).n_elems();

	// apply transformation
	for (NElemT srci = 0; srci < n; ++srci)
	{
		NElemT row = srci / srcx;
		NElemT col = srci % srcx;
		NElemT desti = row + col * srcy;
		destdata[desti] = srcdata[srci];
	}
}

Operation get_op (OPCODE opcode, DTYPE type);

#endif /* OPERATOR_HPP */
