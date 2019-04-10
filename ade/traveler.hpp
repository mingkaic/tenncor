///
/// traveler.hpp
/// ade
///
/// Purpose:
/// Define common traveler implementations
///

#include <unordered_set>
#include <unordered_map>

#include "ade/ileaf.hpp"
#include "ade/ifunctor.hpp"

#ifndef ADE_TRAVELER_HPP
#define ADE_TRAVELER_HPP

namespace ade
{

/// Traveler that maps each tensor to its subtree's maximum depth
struct GraphStat final : public iTraveler
{
	/// Implementation of iTraveler
	void visit (iLeaf* leaf) override
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
				iTensor* tens = child.get_tensor().get();
				tens->accept(*this);
				auto childinfo = graphsize_.find(tens);
				if (graphsize_.end() != childinfo &&
					childinfo->second > ngraph)
				{
					ngraph = childinfo->second;
				}
			}
			graphsize_.emplace(func, ngraph + 1);
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
	void visit (iLeaf* leaf) override {}

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
				TensptrT tens = children[i].get_tensor();
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

/// Travelers will lose smart pointer references,
// this utility function will grab reference maps of root's subtree
std::unordered_map<iTensor*,TensrefT> track_owners (TensptrT root);

}

#endif // ADE_TRAVELER_HPP
