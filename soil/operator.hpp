#include <functional>

#include "soil/inode.hpp"
#include "soil/functor.hpp"

#ifndef OPERATOR_HPP
#define OPERATOR_HPP

struct OpArg
{
	DataSource data_;
	Shape shape_;
};

using Operation = std::function<void(OpArg&,std::vector<OpArg>)>;

template <typename T>
void add (OpArg& dest, std::vector<OpArg> srcs)
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
void mul (OpArg& dest, std::vector<OpArg> srcs)
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
void transpose (OpArg& dest, std::vector<OpArg> srcs)
{
	if (1 != srcs.size())
	{
		handle_error("transpose requires 1 arguments",
			ErrArg<size_t>{"num_args", srcs.size()});
	}

	OpArg& src = srcs[0];
	Shape srcshape = src.shape_;
	NElemT an = srcshape.n_elems();
	NElemT n = dest.shape_.n_elems();

	if (an != n)
	{
		handle_error("transposing src to destination of incompatible size",
			ErrArg<size_t>{"ndest", n},
			ErrArg<size_t>{"nsrc", an});
	}

	T* destdata = dest.data_;
	T* srcdata = src.data_;

	NElemT srcx = srcshape.group(0).n_elems();
	NElemT srcy = srcshape.group(1).n_elems();

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
