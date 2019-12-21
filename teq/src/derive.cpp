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
	root->accept(stat);

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
				return a->to_string() > b->to_string();
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
		TensptrsT prevs = estd::must_getf(grads, parent,
			"failed to find derivative with respect to %s",
			parent->to_string().c_str());
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
			auto local = funcs.local_derivative(parent_ptr, i);
			auto grad_step = funcs.chain_rule(parent_ptr, local, bwd, i);
			grads[children[i].get()].push_back(grad_step);
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
