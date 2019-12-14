///
/// traveler.hpp
/// teq
///
/// Purpose:
/// Define common traveler implementations
///

#include "estd/estd.hpp"
#include "estd/range.hpp"

#include "teq/ileaf.hpp"
#include "teq/ifunctor.hpp"
#include "teq/placeholder.hpp"

#ifndef TEQ_TRAVELER_HPP
#define TEQ_TRAVELER_HPP

namespace teq
{

/// Extremely generic traveler that visits every node in the graph once
struct OnceTraveler : public iTraveler
{
	virtual ~OnceTraveler (void) = default;

	/// Implementation of iTraveler
	void visit (iLeaf& leaf) override
	{
		if (false == estd::has(visited_, &leaf))
		{
			visited_.emplace(&leaf);
			visit_leaf(leaf);
		}
	}

	/// Implementation of iTraveler
	void visit (iFunctor& func) override
	{
		if (false == estd::has(visited_, &func))
		{
			visited_.emplace(&func);
			visit_func(func);
		}
	}

	/// Implementation of iTraveler
	void visit (Placeholder& placeholder) override
	{
		if (false == estd::has(visited_, &placeholder))
		{
			visited_.emplace(&placeholder);
			visit_place(placeholder);
		}
	}

	/// Do something during unique visit to leaf
	virtual void visit_leaf (iLeaf& leaf) = 0;

	/// Do something during unique visit to functor
	virtual void visit_func (iFunctor& func) = 0;

	/// Do something during unique visit to placeholder
	virtual void visit_place (Placeholder& placeholder) = 0;

	virtual void clear (void)
	{
		visited_.clear();
	}

	/// Set of tensors visited
	TensSetT visited_;
};

/// Traveler that maps each tensor to its subtree's maximum depth
struct GraphStat final : public iTraveler
{
	/// Implementation of iTraveler
	void visit (iLeaf& leaf) override
	{
		graphsize_.emplace(&leaf, estd::NumRange<size_t>());
	}

	/// Implementation of iTraveler
	void visit (iFunctor& func) override
	{
		if (false == estd::has(graphsize_, &func))
		{
			auto children = func.get_children();
			size_t nchildren = children.size();
			std::vector<size_t> max_heights;
			std::vector<size_t> min_heights;
			max_heights.reserve(nchildren);
			min_heights.reserve(nchildren);
			for (TensptrT child : children)
			{
				child->accept(*this);
				estd::NumRange<size_t> range = estd::must_getf(graphsize_, child.get(),
					"GraphStat failed to visit child `%s` of functor `%s`",
						child->to_string().c_str(), func.to_string().c_str());
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
			graphsize_.emplace(&func,
				estd::NumRange<size_t>(min_height, max_height));
		}
	}

	/// Implementation of iTraveler
	void visit (Placeholder& placeholder) override
	{
		graphsize_.emplace(&placeholder, estd::NumRange<size_t>());
	}

	// Maximum depth of the subtree of mapped tensors
	std::unordered_map<iTensor*,estd::NumRange<size_t>> graphsize_;
};

/// Map tensors to indices of children
using ParentMapT = std::unordered_map<iTensor*,std::vector<size_t>>;

/// Traveler that paints paths to a target tensor
/// All nodes in the path are added as keys to the parents_ map with the values
/// being a boolean vector denoting nodes leading to target
/// For a boolean value x at index i in mapped vector,
/// x is true if the ith child leads to target
struct PathFinder final : public OnceTraveler
{
	PathFinder (const iTensor* target) : target_(target) {}

	/// Implementation of OnceTraveler
	void visit_leaf (iLeaf& leaf) override {}

	/// Implementation of OnceTraveler
	void visit_func (iFunctor& func) override
	{
		auto children = func.get_children();
		size_t n = children.size();
		std::unordered_set<size_t> path;
		for (size_t i = 0; i < n; ++i)
		{
			TensptrT tens = children[i];
			if (tens.get() == target_)
			{
				path.emplace(i);
			}
			else
			{
				tens->accept(*this);
				if (estd::has(parents_, tens.get()))
				{
					path.emplace(i);
				}
			}
		}
		if (false == path.empty())
		{
			parents_[&func] = std::vector<size_t>(path.begin(), path.end());
		}
	}

	/// Implementation of OnceTraveler
	void visit_place (Placeholder& placeholder) override
	{
		//
	}

	void clear (void) override
	{
		OnceTraveler::clear();
		parents_.clear();
	}

	/// Target of tensor all paths are travelling to
	const iTensor* target_;

	/// Map of parent to child indices that lead to target tensor
	ParentMapT parents_;
};

/// Traveler that for each child tracks the relationship to all parents
struct ParentFinder final : public iTraveler
{
	/// Implementation of iTraveler
	void visit (iLeaf& leaf) override
	{
		parents_.emplace(&leaf, ParentMapT());
	}

	/// Implementation of iTraveler
	void visit (iFunctor& func) override
	{
		if (false == estd::has(parents_, &func))
		{
			auto children = func.get_children();
			for (size_t i = 0, n = children.size(); i < n; ++i)
			{
				auto tens = children[i];
				tens->accept(*this);
				parents_[tens.get()][&func].push_back(i);
			}
			parents_.emplace(&func, ParentMapT());
		}
	}

	/// Implementation of iTraveler
	void visit (Placeholder& placeholder) override
	{
		parents_.emplace(&placeholder, ParentMapT());
	}

	/// Tracks child to parents relationship
	/// Maps child tensor to parent mapping to indices that point to child
	std::unordered_map<iTensor*,ParentMapT> parents_;
};

/// Map between tensor and its corresponding smart pointer
using OwnerMapT = std::unordered_map<iTensor*,TensrefT>;

/// Travelers will lose smart pointer references,
/// This utility function will grab reference maps of root's subtree
OwnerMapT track_owners (TensptrsT roots);

struct Copier final : public OnceTraveler
{
	Copier (TensSetT ignores = {}) : ignores_(ignores) {}

	std::unordered_map<teq::iTensor*,teq::TensptrT> clones_;

	TensSetT ignores_;

private:
	/// Implementation of OnceTraveler
	void visit_leaf (iLeaf& leaf) override
	{
		if (estd::has(ignores_, &leaf))
		{
			return;
		}
		clones_.emplace(&leaf, leaf.clone());
	}

	/// Implementation of OnceTraveler
	void visit_func (iFunctor& func) override
	{
		if (estd::has(ignores_, &func))
		{
			return;
		}
		auto children = func.get_children();
		auto fcpy = func.clone();
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			TensptrT tens = children[i];
			tens->accept(*this);
			if (estd::get(tens, clones_, tens.get()))
			{
				fcpy->update_child(tens, i);
			}
		}
		clones_.emplace(&func, fcpy);
	}

	/// Implementation of OnceTraveler
	void visit_place (Placeholder& place) override
	{
		if (estd::has(ignores_, &place))
		{
			return;
		}
		clones_.emplace(&place, place.clone());
	}
};

}

#endif // TEQ_TRAVELER_HPP
