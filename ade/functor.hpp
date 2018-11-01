///
///	functor.hpp
///	ade
///
///	Purpose:
///	Define functor nodes of an equation graph
///

#include <algorithm>
#include <cassert>
#include <list>
#include <unordered_map>

#include "ade/log/string.hpp"

#include "ade/grader.hpp"
#include "ade/traveler.hpp"

#ifndef ADE_FUNCTOR_HPP
#define ADE_FUNCTOR_HPP

namespace ade
{

/// Functor of the graph mapping to operators specified by opcode argument
struct Functor final : public iFunctor
{
	/// Return a Functor with with input tensor and meta arguments
	static Functor* get (OPCODE opcode, ArgsT args)
	{
		std::string oname = opname(opcode);
		const char* label = oname.c_str();
		if (0 == args.size())
		{
			fatalf("cannot %s with no arguments", label);
		}

		Shape shape = args[0].shape();
		for (size_t i = 1, n = args.size(); i < n; ++i)
		{
			Shape ishape = args[i].shape();
			if (false == ishape.compatible_after(shape, 0))
			{
				fatalf("cannot %s with incompatible shapes %s and %s", label,
					shape.to_string().c_str(), ishape.to_string().c_str());
			}
		}
		return new Functor(opcode, shape, args);
	}

	/// Implementation of iTensor
	const Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	Tensorptr gradient (const iTensor* wrt) override
	{
		if (this == wrt)
		{
			return shaped_one(shape_);
		}

		// define traversal path from this to wrt
		PathFinder finder(wrt);
		accept(finder);
		// no path to wrt
		if (finder.parents_.empty())
		{
			return shaped_zero(wrt->shape());
		}
		// else there exists a path to wrt
		// using pathfinder, breadth first traverse to wrt
		std::list<std::pair<iTensor*,Tensorptr>> tmaps = {{this, shaped_one(shape_)}};
		std::vector<Tensorptr> finalgrad;
		while (false == tmaps.empty())
		{
			auto fpair = tmaps.front();
			tmaps.pop_front();
			iTensor* fwd = fpair.first;
			auto bwd = fpair.second;
			if (wrt == fwd)
			{
				finalgrad.push_back(bwd);
				continue;
			}
			iFunctor* func = static_cast<iFunctor*>(fwd);
			OPCODE opcode = func->get_code();
			auto& paint = finder.parents_[func];
			ArgsT children = func->get_children();
			ArgsT grad_children;
			std::transform(children.begin(), children.end(),
				std::back_inserter(grad_children),
				[](MappedTensor& child)
				{ return MappedTensor{
					CoordPtrT(child.mapper_->reverse()),
					shaped_zero(child.tensor_->shape())}; });
			// for each painted child, calculate dThis/dChild
			for (size_t i = 0, n = children.size(); i < n; ++i)
			{
				if (paint[i])
				{
					Tensorptr& child = children[i].tensor_;
					iTensor* tens = child.get();
					auto zero = grad_children[i].tensor_;
					grad_children[i].tensor_ = bwd;
					// pass down forward-gradient pair
					tmaps.push_back({tens,
						gradmap(opcode, children, grad_children)});
					grad_children[i].tensor_ = zero;
				}
			}
		}

		assert(finalgrad.size() > 0);
		if (finalgrad.size() == 1)
		{
			return finalgrad[0];
		}
		ArgsT finalargs;
		std::transform(finalgrad.begin(), finalgrad.end(),
			std::back_inserter(finalargs),
			[](Tensorptr& tens) -> MappedTensor
			{
				return {identity, tens};
			});
		return Functor::get(ADD, finalargs);
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opname(opcode_);
	}

	/// Implementation of iFunctor
	OPCODE get_code (void) const override
	{
		return opcode_;
	}

	/// Implementation of iFunctor
	const ArgsT& get_children (void) const override
	{
		return args_;
	}

private:
	Functor (OPCODE opcode, Shape shape, ArgsT args) :
		opcode_(opcode), shape_(shape), args_(args) {}

	/// OPCODE represented by functor
	OPCODE opcode_;

	/// Shape info built at construction time according to arguments
	Shape shape_;

	/// Tensor arguments (and children)
	ArgsT args_;
};

}

#endif // ADE_FUNCTOR_HPP
