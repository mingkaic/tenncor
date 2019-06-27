///
/// traveler.hpp
/// ade
///
/// Purpose:
/// Define common traveler implementations
///

#include "stdutil/searchable.hpp"

#include "ade/ileaf.hpp"
#include "ade/ifunctor.hpp"

#ifndef ADE_TRAVELER_HPP
#define ADE_TRAVELER_HPP

namespace ade
{

// todo: move to cppkg
template <typename T, typename = typename std::enable_if<
	std::is_arithmetic<T>::value, T>::type>
struct NumRange final
{
	NumRange (void) : lower_(0), upper_(0) {}

	NumRange (T bound1, T bound2) :
		lower_(std::min(bound1, bound2)),
		upper_(std::max(bound1, bound2)) {}

	T lower_;

	T upper_;
};

/// Traveler that maps each tensor to its subtree's maximum depth
struct GraphStat final : public iTraveler
{
	/// Implementation of iTraveler
	void visit (iLeaf* leaf) override
	{
		graphsize_.emplace(leaf, NumRange<size_t>(0, 0));
	}

	/// Implementation of iTraveler
	void visit (iFunctor* func) override
	{
		if (false == util::has(graphsize_, func))
		{
			ArgsT children = func->get_children();
			size_t nchildren = children.size();
			std::vector<size_t> max_heights;
			std::vector<size_t> min_heights;
			max_heights.reserve(nchildren);
			min_heights.reserve(nchildren);
			for (auto& child : children)
			{
				iTensor* tens = child.get_tensor().get();
				tens->accept(*this);
				NumRange<size_t> range;
				if (false == util::get(range, graphsize_, tens))
				{
					logs::debugf(
						"GraphStat failed to visit child `%s` of functor `%s`",
						tens->to_string().c_str(), func->to_string().c_str());
				}
				max_heights.push_back(range.upper_);
				min_heights.push_back(range.lower_);
			}
			size_t max_height = 1;
			size_t min_height = 1;
			auto max_it = std::max_element(
				max_heights.begin(), max_heights.end());
			auto min_it = std::min_element(
				min_heights.begin(), min_heights.end());
			if (max_heights.end() != max_it)
			{
				max_height += *max_it;
			}
			if (min_heights.end() != max_it)
			{
				min_height += *min_it;
			}
			graphsize_.emplace(func, NumRange<size_t>(min_height, max_height));
		}
	}

	// Maximum depth of the subtree of mapped tensors
	std::unordered_map<iTensor*,NumRange<size_t>> graphsize_;
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
		if (false == util::has(parents_, func))
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
					if (util::has(parents_, tens.get()))
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

/// Traveler that for each child tracks the relationship to all parents
struct ParentFinder final : public ade::iTraveler
{
	using ParentMapT = std::unordered_map<ade::iTensor*,std::vector<size_t>>;

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		parents_.emplace(leaf, ParentMapT());
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (false == util::has(parents_, func))
		{
			auto& children = func->get_children();
			for (size_t i = 0, n = children.size(); i < n; ++i)
			{
				auto& child = children[i];
				auto tens = child.get_tensor();
				tens->accept(*this);
				parents_[tens.get()][func].push_back(i);
			}
			parents_.emplace(func, ParentMapT());
		}
	}

	/// Tracks child to parents relationship
	std::unordered_map<ade::iTensor*,ParentMapT> parents_;
};

/// Map between tensor and its corresponding smart pointer
using OwnerMapT = std::unordered_map<iTensor*,TensrefT>;

/// Travelers will lose smart pointer references,
/// This utility function will grab reference maps of root's subtree
OwnerMapT track_owners (TensT roots);

}

#endif // ADE_TRAVELER_HPP
