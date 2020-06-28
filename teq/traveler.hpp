///
/// traveler.hpp
/// teq
///
/// Purpose:
/// Define common traveler implementations
///

#include "estd/range.hpp"

#include "teq/ileaf.hpp"
#include "teq/ifunctor.hpp"

#ifndef TEQ_TRAVELER_HPP
#define TEQ_TRAVELER_HPP

namespace teq
{

/// Extremely generic traveler that visits every node in the graph once
struct iOnceTraveler : public iTraveler
{
	virtual ~iOnceTraveler (void) = default;

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

	virtual void clear (void)
	{
		visited_.clear();
	}

	/// Set of tensors visited
	TensSetT visited_;

protected:
	/// Do something during unique visit to leaf
	virtual void visit_leaf (iLeaf& leaf) = 0;

	/// Do something during unique visit to functor
	virtual void visit_func (iFunctor& func) = 0;
};

/// Traveler that maps each tensor to its positional information represented
/// by the longest and shortest distance from leaves (NumRange)
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
		if (estd::has(graphsize_, &func))
		{
			return;
		}
		auto children = func.get_dependencies();
		size_t nchildren = children.size();
		std::vector<size_t> max_heights;
		std::vector<size_t> min_heights;
		max_heights.reserve(nchildren);
		min_heights.reserve(nchildren);
		for (TensptrT child : children)
		{
			child->accept(*this);
			estd::NumRange<size_t> range =
				estd::must_getf(graphsize_, child.get(),
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

	const estd::NumRange<size_t>& at (teq::iTensor* tens) const
	{
		return estd::must_getf(graphsize_, tens,
			"failed to find range for %s",
			tens->to_string().c_str());
	}

	// Maximum depth of the subtree of mapped tensors
	TensMapT<estd::NumRange<size_t>> graphsize_;
};

struct GraphIndex final : public iTraveler
{
	/// Implementation of iTraveler
	void visit (iLeaf& leaf) override
	{
		indices_.emplace(&leaf, indices_.size());
	}

	/// Implementation of iTraveler
	void visit (iFunctor& func) override
	{
		if (estd::has(indices_, &func))
		{
			return;
		}
		auto children = func.get_dependencies();
		for (auto child : children)
		{
			child->accept(*this);
		}
		indices_.emplace(&func, indices_.size());
	}

	const size_t& at (teq::iTensor* tens) const
	{
		return estd::must_getf(indices_, tens,
			"failed to find index for %s",
			tens->to_string().c_str());
	}

	TensMapT<size_t> indices_;
};

struct PathDirection
{
	std::vector<size_t> children_;
	std::vector<std::string> attrs_;
};

using PathNodeT = std::unordered_map<std::string,PathDirection>;

using TensPathsT = TensMapT<PathNodeT>;

/// Traveler that paints paths to a target tensor
/// All nodes in the path are added as keys to the roadmap_ map with the values
/// being a boolean vector denoting nodes leading to target
/// For a boolean value x at index i in mapped vector,
/// x is true if the ith child leads to target
struct PathFinder final : public iOnceTraveler
{
	PathFinder (iTensor* target, std::string label = "target") :
		targets_({{target, label}}) {}

	/// For multiple targets, the first target found overshadows target nodes under the first subgraph (todo: label roads)
	PathFinder (TensMapT<std::string> targets) : targets_(targets) {}

	void clear (void) override
	{
		iOnceTraveler::clear();
		roadmap_.clear();
	}

	const PathNodeT& at (teq::iTensor* tens) const
	{
		return estd::must_getf(roadmap_, tens,
			"failed to find road node for %s",
			tens->to_string().c_str());
	}

	const TensMapT<std::string>& get_targets (void) const
	{
		return targets_;
	}

	/// Map of parent to child indices that lead to target tensor
	TensPathsT roadmap_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (iLeaf& leaf) override {}

	/// Implementation of iOnceTraveler
	void visit_func (iFunctor& func) override
	{
		auto children = func.get_dependencies();
		size_t n = children.size();
		PathNodeT nexts;
		for (size_t i = 0; i < n; ++i)
		{
			TensptrT tens = children[i];
			std::string label;
			if (estd::get(label, targets_, tens.get()))
			{
				nexts[label].children_.push_back(i);
			}
			else
			{
				tens->accept(*this);
				if (estd::has(roadmap_, tens.get()))
				{
					auto& subnode = at(tens.get());
					for (auto& spair : subnode)
					{
						nexts[spair.first].children_.push_back(i);
					}
				}
			}
		}
		auto attrs = func.ls_attrs();
		for (auto attr : attrs)
		{
			if (auto tens_attr = dynamic_cast<const TensorRef*>(
				func.get_attr(attr)))
			{
				auto tens = tens_attr->get_tensor();
				std::string label;
				if (estd::get(label, targets_, tens.get()))
				{
					nexts[label].attrs_.push_back(attr);
				}
				else
				{
					tens->accept(*this);
					if (estd::has(roadmap_, tens.get()))
					{
						auto& subnode = at(tens.get());
						for (auto& spair : subnode)
						{
							nexts[spair.first].attrs_.push_back(attr);
						}
					}
				}
			}
		}
		if (false == nexts.empty())
		{
			roadmap_.emplace(&func, nexts);
		}
	}

	/// Target of tensor all paths are travelling to
	TensMapT<std::string> targets_;
};

/// Map tensors to indices of children
using ParentMapT = TensMapT<std::vector<size_t>>;

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
		if (estd::has(parents_, &func))
		{
			return;
		}
		auto children = func.get_dependencies();
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			auto tens = children[i];
			tens->accept(*this);
			parents_[tens.get()][&func].push_back(i);
		}
		parents_.emplace(&func, ParentMapT());
	}

	const ParentMapT& at (teq::iTensor* tens) const
	{
		return estd::must_getf(parents_, tens,
			"failed to find parents for %s",
			tens->to_string().c_str());
	}

	/// Tracks child to parents relationship
	/// Maps child tensor to parent mapping to indices that point to child
	TensMapT<ParentMapT> parents_;
};

/// Map between tensor and its corresponding smart pointer
using OwnerMapT = TensMapT<TensrefT>;

/// Travelers will lose smart pointer references,
/// This utility function will grab reference maps of root's subtree
OwnerMapT track_owners (TensptrsT roots);

struct Copier final : public iOnceTraveler
{
	Copier (TensSetT ignores = {}) : ignores_(ignores) {}

	TensMapT<TensptrT> clones_;

	TensSetT ignores_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (iLeaf& leaf) override
	{
		if (estd::has(ignores_, &leaf))
		{
			return;
		}
		clones_.emplace(&leaf, teq::TensptrT(leaf.clone()));
	}

	void visit_func (iFunctor& func) override
	{
		if (estd::has(ignores_, &func))
		{
			return;
		}
		auto children = func.get_dependencies();
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
		auto attrs = fcpy->ls_attrs();
		for (auto attr : attrs)
		{
			if (auto refattr = dynamic_cast<
				const TensorRef*>(fcpy->get_attr(attr)))
			{
				auto reftens = refattr->get_tensor();
				TensptrT newref;
				if (estd::get(newref, clones_, reftens.get()))
				{
					auto alt = refattr->copynreplace(newref);
					fcpy->rm_attr(attr);
					fcpy->add_attr(attr, marsh::ObjptrT(alt));
				}
			}
		}
		clones_.emplace(&func, teq::TensptrT(fcpy));
	}
};

}

#endif // TEQ_TRAVELER_HPP
