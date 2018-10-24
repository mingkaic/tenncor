///
///	functor.hpp
///	ade
///
///	Purpose:
///	Define functor nodes of an equation graph
///

#include <algorithm>

#include "ade/string.hpp"
#include "ade/tensor.hpp"
#include "ade/opcode.hpp"
#include "ade/coord.hpp"

#ifndef ADE_FUNCTOR_HPP
#define ADE_FUNCTOR_HPP

namespace ade
{

using ArgsT = std::vector<std::pair<CoordPtrT,Tensorptr>>;

/// Interface of OPCODE-defined operation node
struct iFunctor : public iTensor
{
	virtual ~iFunctor (void) = default;

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Return OPCODE mapping to forward and gradient operators
	virtual OPCODE get_code (void) const = 0;

	/// Return children nodes as a vector of raw pointers
	virtual ArgsT get_children (void) const = 0;
};

/// Functor of the graph mapping to operators specified in template argument OP
template <OPCODE OP> // todo: make OP non-template argument
struct Functor final : public iFunctor
{
	/// Return a Functor with with input tensor and meta arguments
	static Functor<OP>* get (ArgsT args)
	{
		std::string oname = opname(OP);
		const char* label = oname.c_str();
		if (0 == args.size())
		{
			fatalf("cannot %s with no arguments", label);
		}

		Shape shape;
		args[0].first->forward(shape.begin(), args[0].second->shape().begin());
		for (size_t i = 1, n = args.size(); i < n; ++i)
		{
			Shape ishape;
			args[i].first->forward(shape.begin(),
				args[i].second->shape().begin());
			if (false == ishape.compatible_after(shape, 0))
			{
				fatalf("cannot %s with incompatible shapes %s and %s", label,
					shape.to_string().c_str(), ishape.to_string().c_str());
			}
		}
		return new Functor<OP>(shape, args);
	}

	/// Implementation of iTensor
	const Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	Tensorptr gradient (Tensorptr& wrt) const override
	{
		if (wrt.get() == this)
		{
			return shaped_one(shape_);
		}
		return grader<OP>(this, args_, wrt);
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opname(OP);
	}

	/// Implementation of iFunctor
	OPCODE get_code (void) const override
	{
		return OP;
	}

	/// Implementation of iFunctor
	ArgsT get_children (void) const override
	{
		return args;
	}

private:
	Functor (Shape shape, ArgsT args) :
		shape_(shape), args_(args) {}

	/// Shape info built at construction time according to arguments
	Shape shape_;

	/// Tensor arguments (and children)
	ArgsT args_;
};

}

#endif // ADE_FUNCTOR_HPP
