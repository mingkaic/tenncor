///
///	traveler.hpp
///	llo
///
///	Purpose:
///	Define tensor iTraveler implementations given llo context
///

#include "llo/node.hpp"

#ifndef LLO_TRAVELER_HPP
#define LLO_TRAVELER_HPP

namespace llo
{

/// Traveler implementation that given a vector of root DataNodes:
/// - gathers a list of all functors under the roots sorted by
///   each functor's subtree's maximum depth in ascending order
/// - a map of each functor's subtree's maximum depth
/// - gathers the context of all roots
/// - ensures all tensor leaves visited have a source
struct GraphStat final : public ade::iTraveler
{
	GraphStat (std::vector<llo::DataNode> roots) :
		global_ctx_([&roots]() -> llo::EvalCtx
		{
			std::vector<const llo::EvalCtx*> contexas(roots.size());
			std::transform(roots.begin(), roots.end(), contexas.begin(),
			[](llo::DataNode& tptr)
			{
				return &tptr.ctx_;
			});
			return llo::EvalCtx(contexas);
		}())
	{
		for (llo::DataNode& tptr : roots)
		{
			tptr.tensor_->accept(*this);
		}
		// sort functions from the root with the smallest subtree to the largest
		// this ensures every children of a node appears before the parent,
		// as is the order of node creations
		funcs_.sort(
		[this](ade::iTensor* a, ade::iTensor* b)
		{
			return this->graphsize_[a] < this->graphsize_[b];
		});
	}

	/// Implemenation of iTraveler
	void visit (ade::Tensor* leaf) override
	{
		if (graphsize_.end() == graphsize_.find(leaf))
		{
			auto srcinfo = global_ctx_.srcs_.find(leaf);
			if (global_ctx_.srcs_.end() != srcinfo)
			{
				leaves_.push_back(srcinfo->second.get());
			}
			else if (
				leaf != ade::Tensor::SYMBOLIC_ONE.get() &&
				leaf != ade::Tensor::SYMBOLIC_ZERO.get())
			{
				ade::fatal("cannot serialize tensor leaf without source");
			}
			graphsize_.emplace(leaf, 0);
		}
	}

	/// Implemenation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (graphsize_.end() == graphsize_.find(func))
		{
			funcs_.push_back(func);
			ade::ArgsT children = func->get_children();
			size_t ngraph = 0;
			for (auto& child : children)
			{
				ade::iTensor* tens = child.second.get();
				if (graphsize_.end() == graphsize_.find(tens))
				{
					child.second->accept(*this);
				}
				auto childinfo = graphsize_.find(tens);
				if (graphsize_.end() != childinfo &&
					childinfo->second > ngraph)
				{
					ngraph = childinfo->second;
				}
				// else child is leaf
			}
			graphsize_[func] = ngraph + 1;
		}
	}

	/// Collected context of all root DataNodes
	llo::EvalCtx global_ctx_;

	/// Vector of leaves visited to maintain order
	std::vector<llo::iSource*> leaves_;

	// List of functions visited (by depth-first) then sorted by graphsize_ in
	// ascending order
	std::list<ade::iFunctor*> funcs_;

	// Maximum depth of the subtree of mapped tensors
	std::unordered_map<ade::iTensor*,size_t> graphsize_;
};

}

#endif // LLO_TRAVELER_HPP
