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

#ifndef TEQ_TRAVELER_HPP
#define TEQ_TRAVELER_HPP

namespace teq
{

/// Extremely generic traveler that visits every node in the graph once
struct OnceTraveler : public iTraveler
{
	virtual ~OnceTraveler (void) = default;

	/// Implementation of iTraveler
	void visit (iLeaf* leaf) override
	{
		if (false == estd::has(visited_, leaf))
		{
			visited_.emplace(leaf);
			visit_leaf(leaf);
		}
	}

	/// Implementation of iTraveler
	void visit (iFunctor* func) override
	{
		if (false == estd::has(visited_, func))
		{
			visited_.emplace(func);
			visit_func(func);
		}
	}

	virtual void visit_leaf (iLeaf* leaf) {} // do nothing

	virtual void visit_func (iFunctor* func)
	{
		auto& children = func->get_children();
		for (auto child : children)
		{
			child.get_tensor()->accept(*this);
		}
	}

	std::unordered_set<iTensor*> visited_;
};

/// Traveler that maps each tensor to its subtree's maximum depth
struct GraphStat final : public iTraveler
{
	/// Implementation of iTraveler
	void visit (iLeaf* leaf) override
	{
		graphsize_.emplace(leaf, estd::NumRange<size_t>());
	}

	/// Implementation of iTraveler
	void visit (iFunctor* func) override
	{
		if (false == estd::has(graphsize_, func))
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
				estd::NumRange<size_t> range = estd::must_getf(graphsize_, tens,
					"GraphStat failed to visit child `%s` of functor `%s`",
						tens->to_string().c_str(), func->to_string().c_str());
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
			graphsize_.emplace(func, estd::NumRange<size_t>(min_height, max_height));
		}
	}

	// Maximum depth of the subtree of mapped tensors
	std::unordered_map<iTensor*,estd::NumRange<size_t>> graphsize_;
};

using ParentMapT = std::unordered_map<iTensor*,std::vector<size_t>>;

/// Traveler that paints paths to a target tensor
/// All nodes in the path are added as keys to the parents_ map with the values
/// being a boolean vector denoting nodes leading to target
/// For a boolean value x at index i in mapped vector,
/// x is true if the ith child leads to target
struct PathFinder final : public iTraveler
{
	PathFinder (const iTensor* target) : target_(target) {}

	/// Implementation of iTraveler
	void visit (iLeaf* leaf) override {}

	/// Implementation of iTraveler
	void visit (iFunctor* func) override
	{
		if (false == estd::has(parents_, func))
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
					if (estd::has(parents_, tens.get()))
					{
						path.emplace(i);
					}
				}
			}
			if (false == path.empty())
			{
				parents_[func] = std::vector<size_t>(path.begin(), path.end());
			}
		}
	}

	/// Target of tensor all paths are travelling to
	const iTensor* target_;

	/// Map of parent nodes in path
	ParentMapT parents_;
};

/// Traveler that for each child tracks the relationship to all parents
struct ParentFinder final : public iTraveler
{
	/// Implementation of iTraveler
	void visit (iLeaf* leaf) override
	{
		parents_.emplace(leaf, ParentMapT());
	}

	/// Implementation of iTraveler
	void visit (iFunctor* func) override
	{
		if (false == estd::has(parents_, func))
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
	std::unordered_map<iTensor*,ParentMapT> parents_;
};

/// Map between tensor and its corresponding smart pointer
using OwnerMapT = std::unordered_map<iTensor*,TensrefT>;

/// Travelers will lose smart pointer references,
/// This utility function will grab reference maps of root's subtree
OwnerMapT track_owners (TensT roots);

struct HeightMatrix
{
	HeightMatrix (const TensT& roots)
	{
		GraphStat stat;
		for (TensptrT root : roots)
		{
			root->accept(stat);
		}

		std::vector<size_t> root_heights;
		root_heights.reserve(roots.size());
		std::transform(roots.begin(), roots.end(),
			std::back_inserter(root_heights),
			[&stat](const TensptrT& root)
			{
				return stat.graphsize_[root.get()].upper_;
			});
		// max of the maxheight of roots should be the maxheight of the whole graph
		size_t maxheight = *std::max_element(
			root_heights.begin(), root_heights.end());
		funcs_ = std::vector<std::unordered_set<iFunctor*>>(maxheight);

		for (auto& gpair : stat.graphsize_)
		{
			auto tens = gpair.first;
			size_t height = gpair.second.upper_;
			if (0 == height)
			{
				leaves_.emplace(static_cast<iLeaf*>(tens));
			}
			else
			{
				funcs_[height - 1].emplace(static_cast<iFunctor*>(tens));
			}
		}
	}

	std::unordered_set<iLeaf*> leaves_;

	std::vector<std::unordered_set<iFunctor*>> funcs_;
};

}

#endif // TEQ_TRAVELER_HPP
