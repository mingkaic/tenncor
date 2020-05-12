#include <cassert>
#include <list>

#include "teq/derive.hpp"

#ifdef TEQ_DERIVE_HPP

namespace teq
{

static const std::string target_label = "target";

void derive (TensMapT<TensptrT>& outders,
	TensptrT root, const PathFinder& pfinder, const iDerivativeFuncs& funcs)
{
	auto& roadmap = pfinder.roadmap_;

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
		// sometimes parent might be reachable through attribute only
		// (todo: fix PathFinder to iterate through children)
		if (estd::get(prevs, grads, parent))
		{
			assert(prevs.size() > 0);
			TensptrT bwd = prevs.size() > 1 ? funcs.add(prevs) : prevs.front();
			// bwd = derive root wrt parent
			auto& nexts = roadmap.at(parent).at(target_label).children_;
			auto parent_ptr = std::static_pointer_cast<iFunctor>(
				owners[parent].lock());
			TensptrsT children = parent->get_children();
			size_t nchildren = children.size();
			// for each i-th child leading to a target,
			// associate child with derive root wrt child
			for (size_t i : nexts)
			{
				assert(i < nchildren);
				grads[children[i].get()].push_back(
					funcs.lderive(parent_ptr, bwd, i));
			}
		}
	}

	auto& targets = pfinder.get_targets();
	for (auto& target : targets)
	{
		TensptrsT tgrads;
		if (estd::get(tgrads, grads, target.first) &&
			tgrads.size() > 0)
		{
			outders.emplace(target.first, tgrads.size() == 1 ?
				tgrads.front() : funcs.add(tgrads));
		}
	}
}

TensptrsT derive (TensptrT root, const TensptrsT& targets,
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
	TensMapT<std::string> tids;
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
		else if (false == estd::has(tids, targets[i].get()))
		{
			tids.emplace(targets[i].get(), target_label);
		}
	}

	TensMapT<TensptrT> ders;
	if (tids.size() > 0)
	{
		PathFinder finder(tids);
		root->accept(finder);
		if (finder.roadmap_.size() > 0)
		{
			derive(ders, root, finder, funcs);
		}
	}

	for (size_t i = 0; i < n; ++i)
	{
		if (nullptr == res[i])
		{
			auto& target = targets[i];
			if (estd::has(ders, target.get()))
			{
				res[i] = ders[target.get()];
			}
			else
			{
				res[i] = funcs.get_const_zero(target->shape());
			}
		}
	}
	return res;
}

}

#endif
