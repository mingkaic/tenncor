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
			return Tensor::SYMBOLIC_ONE;
		}

		// define traversal path from this to wrt
		PathFinder finder(wrt);
		accept(finder);
		// no path to wrt
		if (finder.parents_.empty())
		{
			return Tensor::SYMBOLIC_ZERO;
		}
		// else there exists a path to wrt
		// using pathfinder, breadth first traverse to wrt
		std::list<std::pair<iTensor*,Tensorptr>> tmaps = {
			{this, shaped_one(shape_)}};
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
			// for each painted child, calculate dThis/dChild
			for (size_t i = 0, n = children.size(); i < n; ++i)
			{
				if (paint[i])
				{
					ArgsT args;
					CoordPtrT mapper(children[i].mapper_->reverse());
					for (size_t j = 0; j < n; ++j)
					{
						if (j == i)
						{
							args.push_back(MappedTensor{
								identity, children[j].tensor_});
						}
						else
						{
							CoordPtrT toshape(
								children[j].mapper_->forward(*mapper));
							Tensorptr& tens = children[j].tensor_;
							args.push_back(MappedTensor{toshape, tens});
						}
					}
					// pass down forward-gradient pair
					Tensorptr grad = gradmap(opcode, args, i);
					tmaps.push_back({children[i].tensor_.get(),
						Functor::get(MUL, {
							{identity, Functor::get(COPY, {{mapper, bwd}})},
							{identity, grad},
						})});
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
