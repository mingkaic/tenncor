///
///	functor.hpp
///	ade
///
///	Purpose:
///	Define functor nodes of an equation graph
///

#include <algorithm>
#include <unordered_map>
#include <list>

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

		Shape shape;
		args[0].first->forward(shape.begin(), args[0].second->shape().begin());
		for (size_t i = 1, n = args.size(); i < n; ++i)
		{
			Shape ishape;
			args[i].first->forward(ishape.begin(),
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
	Tensorptr gradient (const iTensor* wrt) override
	{
		if (this == wrt)
		{
			return shaped_one(shape_);
		}

		// define traversal path from this to wrt
		PathFinder finder(wrt);
		accept(finder);

		auto zero = shaped_zero(shape_);
		if (finder.parents_.empty())
		{
			return zero;
		}
		// else there exists a path to wrt
		// using pathfinder, breadth first traverse to wrt
		std::list<std::pair<iFunctor*,Tensorptr>> funcs = {{this, shaped_one(shape_)}};
		std::vector<Tensorptr> finalgrad;
		while (false == funcs.empty())
		{
			auto& fpair = funcs.front();
			iFunctor* f = fpair.first;
			auto grad = fpair.second;
			funcs.pop_front();
			auto& paint = finder.parents_[f];
			ArgsT children = f->get_children();
			// prep children for downward traversal
			for (auto& child : children)
			{
				child.first = CoordPtrT(child.first->reverse());
			}
			// for each painted child, calculate dThis/dChild
			for (size_t i = 0, n = children.size(); i < n; ++i)
			{
				if (paint[i])
				{
					Tensorptr& fwd = children[i].second;
					iTensor* child = fwd.get();
					std::vector<Tensorptr> grads(n, zero);
					grads[i] = grad;
					auto g = gradmap(f->get_code(), fwd, children, grads);
					if (wrt == child)
					{
						// grad should be compatible with wrt
						finalgrad.push_back(g);
					}
					else
					{
						// calculate grad using chain rule then pass down forward-gradient pair
						funcs.push_back({static_cast<iFunctor*>(child), g});
					}
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
