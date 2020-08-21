///
/// traveler.hpp
/// teq
///
/// Purpose:
/// Define common traveler implementations
///

#include "estd/range.hpp"

#include "internal/teq/ileaf.hpp"
#include "internal/teq/ifunctor.hpp"

#ifndef TEQ_TRAVELER_HPP
#define TEQ_TRAVELER_HPP

namespace teq
{

using GetDepsF = std::function<TensptrsT(iFunctor&)>;

/// Map between tensor and its corresponding shared pointer
using OwnMapT = TensMapT<TensptrT>;

/// Map between tensor and its corresponding weak pointer
using RefMapT = TensMapT<TensrefT>;

TensptrsT get_alldeps (iFunctor& func);

TensptrsT get_deps (iFunctor& func);

TensptrsT get_args (iFunctor& func);

TensptrsT get_attrs (iFunctor& func);

struct LambdaVisit final : public iTraveler
{
	LambdaVisit (
		std::function<void(iLeaf&)> lvisit,
		std::function<void(iTraveler&,iFunctor&)> fvisit) :
		lvisit_(lvisit), fvisit_(fvisit) {}

	/// Implementation of iTraveler
	void visit (iLeaf& leaf) override
	{
		lvisit_(leaf);
	}

	/// Implementation of iTraveler
	void visit (iFunctor& func) override
	{
		fvisit_(*this, func);
	}

	std::function<void(iLeaf&)> lvisit_;

	std::function<void(iTraveler&,iFunctor&)> fvisit_;
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
		auto deps = func.get_dependencies();
		size_t ndeps = deps.size();
		std::vector<size_t> max_heights;
		std::vector<size_t> min_heights;
		max_heights.reserve(ndeps);
		min_heights.reserve(ndeps);
		multi_visit(*this, deps);
		for (TensptrT child : deps)
		{
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

	const estd::NumRange<size_t>& at (iTensor* tens) const
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
		auto deps = func.get_dependencies();
		multi_visit(*this, deps);
		indices_.emplace(&func, indices_.size());
	}

	const size_t& at (iTensor* tens) const
	{
		return estd::must_getf(indices_, tens,
			"failed to find index for %s",
			tens->to_string().c_str());
	}

	TensMapT<size_t> indices_;
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
		auto deps = func.get_dependencies();
		multi_visit(*this, deps);
		for (size_t i = 0, n = deps.size(); i < n; ++i)
		{
			parents_[deps[i].get()][&func].push_back(i);
		}
		parents_.emplace(&func, ParentMapT());
	}

	const ParentMapT& at (iTensor* tens) const
	{
		return estd::must_getf(parents_, tens,
			"failed to find parents for %s",
			tens->to_string().c_str());
	}

	/// Tracks child to parents relationship
	/// Maps child tensor to parent mapping to indices that point to child
	TensMapT<ParentMapT> parents_;
};

/// Generic traveler that visits every node in the graph once
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
	/// For multiple targets, the first target found overshadows target nodes under the first subgraph (todo: label roads)
	PathFinder (TensMapT<std::string> targets,
		GetDepsF get_fdep = get_alldeps) :
		targets_(targets), get_fdep_(get_fdep) {}

	void clear (void) override
	{
		iOnceTraveler::clear();
		roadmap_.clear();
	}

	const PathNodeT& at (iTensor* tens) const
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
		auto deps = func.get_dependencies();
		auto attrs = func.ls_attrs();

		TensptrsT to_visit = get_fdep_(func);
		for (auto tens : to_visit)
		{
			if (false == estd::has(targets_, tens.get()))
			{
				tens->accept(*this);
			}
		}
		size_t n = deps.size();
		PathNodeT nexts;
		for (size_t i = 0; i < n; ++i)
		{
			TensptrT tens = deps[i];
			std::string label;
			if (estd::get(label, targets_, tens.get()))
			{
				nexts[label].children_.push_back(i);
			}
			else if (estd::has(roadmap_, tens.get()))
			{
				auto& subnode = at(tens.get());
				for (auto& spair : subnode)
				{
					nexts[spair.first].children_.push_back(i);
				}
			}
		}
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
				else if (estd::has(roadmap_, tens.get()))
				{
					auto& subnode = at(tens.get());
					for (auto& spair : subnode)
					{
						nexts[spair.first].attrs_.push_back(attr);
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

	GetDepsF get_fdep_;
};

struct Copier final : public iOnceTraveler
{
	Copier (TensSetT ignores = {}) : ignores_(ignores) {}

	OwnMapT clones_;

	TensSetT ignores_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (iLeaf& leaf) override
	{
		if (estd::has(ignores_, &leaf))
		{
			return;
		}
		clones_.emplace(&leaf, TensptrT(leaf.clone()));
	}

	void visit_func (iFunctor& func) override
	{
		if (estd::has(ignores_, &func))
		{
			return;
		}
		auto deps = func.get_dependencies();
		auto fcpy = func.clone();
		multi_visit(*this, deps);
		for (size_t i = 0, n = deps.size(); i < n; ++i)
		{
			TensptrT tens = deps[i];
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
		clones_.emplace(&func, TensptrT(fcpy));
	}
};

struct OwnerTracker final : public iOnceTraveler
{
	OwnMapT owners_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (iLeaf& leaf) override {}

	/// Implementation of iOnceTraveler
	void visit_func (iFunctor& func) override
	{
		auto deps = func.get_dependencies();
		multi_visit(*this, deps);
		for (const TensptrT& dep : deps)
		{
			owners_.emplace(dep.get(), dep);
		}
	}
};

/// This utility function will grab pointer maps of root's subtree
template <typename TS> // todo: use concept tensptr_ranges
OwnMapT track_ownptrs (const TS& roots)
{
	OwnerTracker tracker;
	multi_visit(tracker, roots);
	for (auto root : roots)
	{
		tracker.owners_.emplace(root.get(), root);
	}
	return tracker.owners_;
}

/// This utility function will grab reference maps of root's subtree
template <typename TS> // todo: use concept tensptr_ranges
RefMapT track_ownrefs (const TS& roots)
{
	OwnerTracker tracker;
	multi_visit(tracker, roots);
	for (auto root : roots)
	{
		tracker.owners_.emplace(root.get(), root);
	}
	RefMapT refs;
	std::transform(tracker.owners_.begin(), tracker.owners_.end(),
		std::inserter(refs, refs.begin()),
		[](std::pair<iTensor*,TensptrT> tp) -> std::pair<iTensor*,TensrefT>
		{
			return {tp.first, tp.second};
		});
	return refs;
}

OwnMapT convert_ownmap (const RefMapT& refs);

}

#endif // TEQ_TRAVELER_HPP
