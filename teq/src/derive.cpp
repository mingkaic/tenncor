#include <cassert>
#include <list>

#include "teq/derive.hpp"

#ifdef TEQ_DERIVE_HPP

namespace teq
{

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

	PathFinder finder(target.get());
	root->accept(finder);

	auto& pathmap = finder.roadmap_;
	// no path to wrt
	if (pathmap.empty())
	{
		return funcs.get_const_zero(target->shape());
	}
	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	GraphStat stat;
	root->accept(stat);
	auto owners = track_owners({root});

	std::list<iFunctor*> parents;
	std::transform(pathmap.begin(), pathmap.end(),
		std::back_inserter(parents),
		[](std::pair<iTensor*,std::vector<size_t>> parent)
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
	std::unordered_map<const iTensor*,TensptrsT> grads = {
		{root.get(), {funcs.get_const_one(root->shape())}}
	};
	for (iFunctor* parent : parents)
	{
		TensptrsT prevs = estd::must_getf(grads, parent,
			"failed to find derivative with respect to %s",
			parent->to_string().c_str());
		assert(prevs.size() > 0);
		TensptrT bwd = prevs.size() > 1 ? funcs.add(prevs) : prevs.front();

		auto& nexts = pathmap[parent];
		TensptrsT children = parent->get_children();
		// for each painted child, calculate dThis/dChild
		// go through grads in order
		for (size_t i : nexts)
		{
			auto parent_ptr = std::static_pointer_cast<iFunctor>(
				owners[parent].lock());
			auto local = funcs.local_derivative(parent_ptr, i);
			auto grad_step = funcs.chain_rule(parent_ptr, local, bwd, i);
			grads[children[i].get()].push_back(grad_step);
		}
	}

	TensptrsT& outargs = grads[target.get()];
	return outargs.size() > 1 ? funcs.add(outargs) : outargs.front();
}

}

#endif
