///
///	functor.hpp
///	ade
///
///	Purpose:
///	Define functor nodes of an equation graph
///

#include <list>
#include <unordered_set>
#include <unordered_map>
#include <iostream>

#include "ade/ifunctor.hpp"

#ifndef ADE_TRAVELER_HPP
#define ADE_TRAVELER_HPP

namespace ade
{

/// Traveler that maps each tensor to its subtree's maximum depth
struct GraphStat final : public iTraveler
{
	/// Implementation of iTraveler
	void visit (Tensor* leaf) override
	{
		graphsize_.emplace(leaf, 0);
	}

	/// Implementation of iTraveler
	void visit (iFunctor* func) override
	{
		if (graphsize_.end() == graphsize_.find(func))
		{
			ArgsT children = func->get_children();
			size_t ngraph = 0;
			for (auto& child : children)
			{
				iTensor* tens = child.tensor_.get();
				tens->accept(*this);
				auto childinfo = graphsize_.find(tens);
				if (graphsize_.end() != childinfo &&
					childinfo->second > ngraph)
				{
					ngraph = childinfo->second;
				}
			}
			graphsize_[func] = ngraph + 1;
		}
	}

	// Maximum depth of the subtree of mapped tensors
	std::unordered_map<iTensor*,size_t> graphsize_;
};

/// Traveler that paints paths to a target tensor
/// All nodes in the path are added as keys to the parents_ map with the values
/// being a boolean vector denoting nodes leading to target
/// For a boolean value x at index i in mapped vector,
/// x is true if the ith child leads to target
struct PathFinder final : public iTraveler
{
	/// Type for mapping function nodes in path to boolean vector
	using ParentMapT = std::unordered_map<iTensor*,std::unordered_set<size_t>>;

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
			std::unordered_set<size_t> path;
			for (size_t i = 0; i < n; ++i)
			{
				Tensorptr tens = children[i].tensor_;
				if (tens.get() == target_)
				{
					path.emplace(i);
				}
				else
				{
					tens->accept(*this);
					if (parents_.end() != parents_.find(tens.get()))
					{
						path.emplace(i);
					}
				}
			}
			if (false == path.empty())
			{
				parents_[func] = path;
			}
		}
	}

	/// Target of tensor all paths are travelling to
	const iTensor* target_;

	/// Map of parent nodes in path
	ParentMapT parents_;
};

struct iGrader : public iTraveler
{
	iGrader (const iTensor* target) : target_(target) {}

	virtual ~iGrader (void) = default;

	/// Implementation of iTraveler
	void visit (Tensor* leaf) override
	{
		if (leaf == target_)
		{
			set_scalar(leaf, 1);
		}
		else
		{
			set_scalar(leaf, 0);
		}
	}

	/// Implementation of iTraveler
	void visit (iFunctor* func) override
	{
		if (func == target_)
		{
			set_scalar(func, 1);
			return;
		}

		PathFinder finder(target_);
		func->accept(finder);

		auto& pathmap = finder.parents_;
		// no path to wrt
		if (pathmap.empty())
		{
			set_scalar(target_, 0);
			return;
		}
		// else there exists a path to wrt
		// using pathfinder, breadth first traverse from this to wrt
		GraphStat stat;
		func->accept(stat);

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

		std::unordered_map<const iTensor*,ArgsT> grads = {{func,
			{{identity, get_scalar(func->shape(), 1)}},
		}};
		for (iFunctor* parent : parents)
		{
			Opcode opcode = parent->get_opcode();
			ArgsT& gradargs = grads[parent];
			MappedTensor bwd = gradargs[0];
			if (gradargs.size() > 1)
			{
				bwd = {identity, add_grads(gradargs)};
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
				Tensorptr grad = get_grad(opcode, args, i);
				CoordPtrT bwd_mapper(bwd.mapper_->forward(*mapper));
				grads[child.tensor_.get()].push_back({
					identity, chain_grad(grad,
						{bwd_mapper, bwd.tensor_})
				});
			}
		}
		set_grad(func, add_grads(grads[target_]));
	}

	virtual Tensorptr chain_grad (Tensorptr& wrt_child,
		MappedTensor wrt_me) const = 0;

	virtual Tensorptr add_grads (ArgsT& args) const = 0;

	virtual Tensorptr get_grad (Opcode opcode, ArgsT args, size_t gradidx) const = 0;

	virtual Tensorptr get_scalar (const Shape& shape, size_t scalar) const = 0;

	/// Set derivative key to target have value get_scalar(key->shape(), scalar);
	virtual void set_scalar (const iTensor* key, size_t scalar) = 0;

	/// Set derivative key to target be the value
	virtual void set_grad (const iTensor* key, Tensorptr value) = 0;

	/// Target of tensor all paths are travelling to
	const iTensor* target_;
};

}

#endif // ADE_TRAVELER_HPP
