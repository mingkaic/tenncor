#include <cassert>
#include <list>

#include "teq/derive.hpp"

#ifdef TEQ_DERIVE_HPP

namespace teq
{

static const std::string target_label = "target";

TensptrT derive (TensptrT root, TensptrT target, iDerivativeFuncs& funcs)
{
	if (target == nullptr)
	{
		return funcs.get_const_zero(Shape());
	}

	if (root == nullptr)
	{
		return funcs.get_const_zero(target->shape());
	}

	if (root == target)
	{
		return funcs.get_const_one(target->shape());
	}

	PathFinder finder(target.get(), target_label);
	root->accept(finder);

	auto& roadmap = finder.roadmap_;
	// no path to wrt
	if (roadmap.empty())
	{
		return funcs.get_const_zero(target->shape());
	}
	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	OwnerMapT owners = track_owners({root});
	GraphStat stat;
	GraphIndex indexer;
	root->accept(stat);
	root->accept(indexer);

	std::list<iFunctor*> parents; // todo: make parent order not dependent on sorting algorithm by sorting by order visited
	std::transform(roadmap.begin(), roadmap.end(),
		std::back_inserter(parents),
		[](std::pair<iTensor*,PathNodeT> parent)
		{
			return static_cast<iFunctor*>(parent.first);
		});
	parents.sort(
		[&](iFunctor* a, iFunctor* b)
		{
			size_t aheight = stat.graphsize_[a].upper_;
			size_t bheight = stat.graphsize_[b].upper_;
			if (aheight == bheight) // make traversal more deterministic
			{
				std::string astr = a->to_string();
				std::string bstr = b->to_string();
				if (astr == bstr)
				{
					return indexer.at(a) > indexer.at(b);
				}
				return astr > bstr;
			}
			return aheight > bheight;
		});

	// map functor to its respective super composite derivative
	// let L = root, F = key functor, value of F in grads is dL/dF
	TensMapT<TensptrsT> grads = {
		{root.get(), {funcs.get_const_one(root->shape())}}
	};
	for (iFunctor* parent : parents)
	{
		TensptrsT prevs;
		// sometimes parent might be reachable through attribute only (todo: fix PathFinder to iterate through children)
		if (estd::get(prevs, grads, parent))
		{
			assert(prevs.size() > 0);
			TensptrT bwd = prevs.size() > 1 ? funcs.add(prevs) : prevs.front();
			auto& nexts = roadmap[parent].at(target_label).children_;
			auto parent_ptr = std::static_pointer_cast<iFunctor>(
				owners[parent].lock());
			TensptrsT children = parent->get_children();
			size_t nchildren = children.size();
			for (size_t i : nexts)
			{
				assert(i < nchildren);
				grads[children[i].get()].push_back(
					funcs.lderive(parent_ptr, bwd, i));
			}
		}
	}

	TensptrsT tgrads = estd::must_getf(grads, target.get(),
		"failed to find derivative with respect to %s",
		target->to_string().c_str());
	assert(tgrads.size() > 0);
	return tgrads.size() == 1 ? tgrads.front() : funcs.add(tgrads);
}

}

#endif
