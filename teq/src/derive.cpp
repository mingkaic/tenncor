#include <cassert>
#include <list>

#include "teq/derive.hpp"

#ifdef TEQ_DERIVE_HPP

namespace teq
{

static const std::string target_label = "target";

TensptrsT derive (
	TensptrT root,
	const TensptrsT& targets,
	const iDerivativeFuncs& funcs)
{
	size_t n = targets.size();
	if (root == nullptr)
	{
		TensptrsT out;
		out.reserve(n);
		std::transform(targets.begin(), targets.end(),
			std::back_inserter(out),
			[&](const TensptrT& target)
			{
				return funcs.get_const_zero(target->shape());
			});
		return out;
	}

	TensptrsT res(n, nullptr);
	TensptrSetT targs;
	for (size_t i = 0; i < n; ++i)
	{
		if (nullptr == targets[i])
		{
			res[i] = funcs.get_const_zero(Shape());
		}
		else if (root == targets[i])
		{
			res[i] = funcs.get_const_one(targets[i]->shape());
		}
		else
		{
			targs.emplace(targets[i]);
		}
	}

	TensMapT<TensptrsT> grads = {
		{root.get(), {funcs.get_const_one(root->shape())}}
	};
	partial_derive(grads, {root}, targs, funcs);

	for (size_t i = 0; i < n; ++i)
	{
		if (nullptr == res[i])
		{
			auto target = targets[i];
			TensptrsT tgrads;
			if (estd::get(tgrads, grads, target.get()) && tgrads.size() > 0)
			{
				res[i] = tgrads.size() == 1 ?
					tgrads.front() : funcs.add(tgrads);
			}
			else
			{
				res[i] = funcs.get_const_zero(target->shape());
			}
		}
	}
	return res;
}

void partial_derive (TensMapT<TensptrsT>& grads,
	const TensptrSetT& parents,
	const TensptrSetT& targets,
	const iDerivativeFuncs& funcs)
{
	TensMapT<std::string> tids;
	std::transform(targets.begin(), targets.end(),
		std::inserter(tids, tids.end()),
		[](TensptrT target)
		{
			return std::pair<iTensor*,std::string>{
				target.get(), target_label};
		});
	if (tids.empty())
	{
		return;
	}

	PathFinder pfinder(tids, get_args);
	for (auto parent : parents)
	{
		if (nullptr != parent)
		{
			parent->accept(pfinder);
		}
	}
	if (pfinder.roadmap_.empty())
	{
		return;
	}

	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	OwnerMapT owners = track_owners(parents);
	GraphStat stat;
	GraphIndex indexer;
	multi_visit(stat, parents);
	multi_visit(indexer, parents);

	std::list<iFunctor*> tovisits; // todo: make parent order not dependent on sorting algorithm by sorting by order visited
	std::transform(pfinder.roadmap_.begin(), pfinder.roadmap_.end(),
		std::back_inserter(tovisits),
		[](std::pair<iTensor*,PathNodeT> parent)
		{
			return static_cast<iFunctor*>(parent.first);
		});
	tovisits.sort(
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
	for (iFunctor* tens : tovisits)
	{
		TensptrsT prevs = estd::must_getf(grads, tens,
			"failed to find existing grads for %s", tens->to_string().c_str());
		assert(prevs.size() > 0);
		TensptrT bwd = prevs.size() > 1 ? funcs.add(prevs) : prevs.front();
		// bwd = derive root wrt tens
		auto& nexts = pfinder.at(tens).at(target_label).children_;
		auto visitable_ptr = std::static_pointer_cast<iFunctor>(
			owners[tens].lock());
		TensptrsT children = tens->get_args();
		size_t nchildren = children.size();
		// for each i-th child leading to a target,
		// associate child with derive root wrt child
		for (size_t i : nexts)
		{
			if (i < nchildren)
			{
				grads[children[i].get()].push_back(
					funcs.lderive(visitable_ptr, bwd, i));
			}
		}
	}
}

}

#endif
