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

#include "err/string.hpp"

#include "ade/tensor.hpp"
#include "ade/traveler.hpp"
#include "ade/coord.hpp"

#ifndef ADE_FUNCTOR_HPP
#define ADE_FUNCTOR_HPP

namespace ade
{

/// Functor of the graph mapping to operators specified by opcode argument
struct Functor final : public iFunctor
{
	/// Return a Functor with with input tensor and meta arguments
	static Functor* get (OpPtrT opcode, ArgsT args)
	{
		std::string oname = opcode->to_string();
		const char* label = oname.c_str();
		if (0 == args.size())
		{
			err::fatalf("cannot %s with no arguments", label);
		}

		Shape shape = args[0].shape();
		for (size_t i = 1, n = args.size(); i < n; ++i)
		{
			Shape ishape = args[i].shape();
			if (false == ishape.compatible_after(shape, 0))
			{
				err::fatalf("cannot %s with incompatible shapes %s and %s", label,
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
		auto& pathmap = finder.parents_;
		// no path to wrt
		if (pathmap.empty())
		{
			return Tensor::SYMBOLIC_ZERO;
		}
		// else there exists a path to wrt
		// using pathfinder, breadth first traverse from this to wrt
		GraphStat stat;
		accept(stat);

		std::list<iFunctor*> parents;
		std::transform(pathmap.begin(), pathmap.end(),
			std::back_inserter(parents),
			[](std::pair<iTensor*,std::unordered_set<size_t>> parent)
			{
				return static_cast<ade::iFunctor*>(parent.first);
			});
		parents.sort(
			[&](iFunctor* a, iFunctor* b)
			{
				return stat.graphsize_[a] > stat.graphsize_[b];
			});

		std::unordered_map<const iTensor*,ArgsT> grads = {{this,
			{{extend(0, std::vector<DimT>(shape_.begin(), shape_.end())),
				Tensor::SYMBOLIC_ONE}},
		}};
		for (iFunctor* parent : parents)
		{
			const iOperation& opcode = parent->get_code();
			ArgsT& gradargs = grads[parent];
			MappedTensor bwd = gradargs[0];
			if (gradargs.size() > 1)
			{
				bwd = {identity, opcode.add_grads(gradargs)};
			}

			auto& grad_indices = pathmap[parent];
			ArgsT children = parent->get_children();
			size_t nchildren = children.size();
			// for each painted child, calculate dThis/dChild
			for (size_t i : grad_indices)
			{
				ArgsT args;
				MappedTensor& child = children[i];
				CoordPtrT mapper(child.mapper_->reverse());
				for (size_t j = 0; j < nchildren; ++j)
				{
					Tensorptr& tens = children[j].tensor_;
					if (j == i)
					{
						args.push_back({identity, tens});
					}
					else
					{
						CoordPtrT toshape(
							children[j].mapper_->forward(*mapper));
						args.push_back({toshape, tens});
					}
				}
				// pass down forward-gradient pair
				Tensorptr grad = opcode.gradient(args, i);
				CoordPtrT bwd_mapper(bwd.mapper_->forward(*mapper));
				grads[child.tensor_.get()].push_back({
					identity, opcode.chain_grad(grad,
						{bwd_mapper, bwd.tensor_})
				});
			}
		}

		return opcode_->add_grads(grads[wrt]);
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opcode_->to_string();
	}

	/// Implementation of iFunctor
	const iOperation& get_code (void) const override
	{
		return *opcode_;
	}

	/// Implementation of iFunctor
	const ArgsT& get_children (void) const override
	{
		return args_;
	}

private:
	Functor (OpPtrT& opcode, Shape shape, ArgsT args) :
		opcode_(opcode), shape_(shape), args_(args) {}

	/// OPCODE represented by functor
	OpPtrT opcode_;

	/// Shape info built at construction time according to arguments
	Shape shape_;

	/// Tensor arguments (and children)
	ArgsT args_;
};

}

#endif // ADE_FUNCTOR_HPP
