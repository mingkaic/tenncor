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

#ifndef ADE_FUNCTOR_HPP
#define ADE_FUNCTOR_HPP

namespace ade
{

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
	virtual const ArgsT& get_children (void) const = 0;
};

struct PathFinder final : public iTraveler
{
	using ParentMapT = std::unordered_map<iTensor*,std::vector<bool>>;

	PathFinder (const iTensor* target) : target_(target) {}

	/// Implementation of iTraveler
	void visit (Tensor* leaf) override {}

	/// Implementation of iTraveler
	void visit (iFunctor* func) override
	{
		if (parents_.end() == parents_.find(func))
		{
			auto& children = func->get_children();
			size_t n = children.size();
			bool has_path = false;
			std::vector<bool> path(n, false);
			for (size_t i = 0; i < n; ++i)
			{
				Tensorptr tens = children[i].second;
				if (tens.get() == target_)
				{
					path[i] = has_path = true;
				}
				else
				{
					tens->accept(*this);
					if (parents_.end() != parents_.find(tens.get()))
					{
						path[i] = has_path = true;
					}
				}
			}
			if (has_path)
			{
				parents_[func] = path;
			}
		}
	}

	const iTensor* target_;

	ParentMapT parents_;
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

		Shape shape = map_shape(args[0].first, args[0].second->shape());
		for (size_t i = 1, n = args.size(); i < n; ++i)
		{
			Shape ishape = map_shape(args[i].first, args[i].second->shape());
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
	Tensorptr gradient (const iTensor* wrt) override
	{
		if (this == wrt)
		{
			return shaped_one(shape_);
		}

		// define traversal path from this to wrt
		PathFinder finder(wrt);
		accept(finder);

		if (finder.parents_.empty())
		{
			return shaped_zero(shape_);
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
				[](std::pair<CoordPtrT,Tensorptr>& child)
				{ return std::pair<CoordPtrT,Tensorptr>{
					child.first->reverse(),
					shaped_zero(child.second->shape())}; });
			// for each painted child, calculate dThis/dChild
			for (size_t i = 0, n = children.size(); i < n; ++i)
			{
				if (paint[i])
				{
					Tensorptr& child = children[i].second;
					iTensor* tens = child.get();
					auto zero = grad_children[i].second;
					grad_children[i].second = bwd;
					// pass down forward-gradient pair
					tmaps.push_back({tens,
						gradmap(opcode, children, grad_children)});
					grad_children[i].second = zero;
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
			[](Tensorptr& tens) -> std::pair<CoordPtrT,Tensorptr>
			{
				return {identity, tens};
			});
		return Functor<ADD>::get(finalargs);
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
	const ArgsT& get_children (void) const override
	{
		return args_;
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
