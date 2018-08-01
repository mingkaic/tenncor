#include <functional>

#include "soil/inode.hpp"
#include "soil/functor.hpp"

#ifndef OPERATOR_HPP
#define OPERATOR_HPP

using Operation = std::function<void(char*,Shape&,std::vector<Nodeptr>&)>;

template <typename T>
void add (char* dest, Shape& destshape, std::vector<Nodeptr>& srcs)
{
	if (2 != srcs.size())
	{
		handle_error("add requires 2 arguments",
			ErrArg<size_t>{"num_args", srcs.size()});
	}

	std::shared_ptr<char> aptr = srcs[0]->calculate();
	std::shared_ptr<char> bptr = srcs[1]->calculate();
	T* a = (T*) aptr.get();
	T* b = (T*) bptr.get();
	T* c = (T*) dest;

	NElemT an = srcs[0]->shape().n_elems();
	NElemT bn = srcs[1]->shape().n_elems();
	NElemT n = destshape.n_elems();

	for (NElemT i = 0; i < n; ++i)
	{
		c[i] = a[i % an] + b[i % bn];
	}
}

template <typename T>
void mul (char* dest, Shape& destshape, std::vector<Nodeptr>& srcs)
{
	if (2 != srcs.size())
	{
		handle_error("mul requires 2 arguments",
			ErrArg<size_t>{"num_args", srcs.size()});
	}

	std::shared_ptr<char> aptr = srcs[0]->calculate();
	std::shared_ptr<char> bptr = srcs[1]->calculate();
	T* a = (T*) aptr.get();
	T* b = (T*) bptr.get();
	T* c = (T*) dest;

	NElemT an = srcs[0]->shape().n_elems();
	NElemT bn = srcs[1]->shape().n_elems();
	NElemT n = destshape.n_elems();

	for (NElemT i = 0; i < n; ++i)
	{
		c[i] = a[i % an] * b[i % bn];
	}
}

template <typename T>
void transpose (char* dest, Shape& destshape, std::vector<Nodeptr>& srcs)
{
	if (1 != srcs.size())
	{
		handle_error("transpose requires 1 arguments",
			ErrArg<size_t>{"num_args", srcs.size()});
	}

	Nodeptr& src = srcs[0];
	Shape srcshape = src->shape();
	NElemT an = srcshape.n_elems();
	NElemT n = destshape.n_elems();

	if (an != n)
	{
		handle_error("transposing src to destination of incompatible size",
			ErrArg<size_t>{"ndest", n},
			ErrArg<size_t>{"nsrc", an});
	}

	T* destdata = (T*) dest;
	std::shared_ptr<char> sptr = src->calculate();
	T* srcdata = (T*) sptr.get();

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
