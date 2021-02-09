#include <cassert>
#include <list>

#include "internal/teq/derive.hpp"

#ifdef TEQ_DERIVE_HPP

namespace teq
{

static const std::string target_label = "target";

TensptrsT backprop (
	TensptrT root,
	const TensptrsT& targets,
	const iBackpropFuncs& funcs)
{
	size_t n = targets.size();
	TensptrsT outs;
	outs.reserve(n);

	if (targets.empty())
	{
		return outs;
	}

	// map X -> J(X,T) for each T in targets
	TensMapT<TensMapT<TensptrsT>> jacobians;
	for (auto targ : targets)
	{
		auto targptr = targ.get();
		jacobians.emplace(targptr, TensMapT<TensptrsT>{
			std::pair<iTensor*,TensptrsT>{targptr,
				TensptrsT{funcs.get_const_eye(
					targ->shape().n_elems(),
					targ->get_meta().type_code())}
			}
		});
	}

	TensMapT<std::string> tids;
	for (auto& target : targets)
	{
		if (nullptr != target && root != target)
		{
			tids.emplace(target.get(), target_label);
		}
	}

	PathFinder pfinder(tids, [](iFunctor& f){ return f.get_args(); });
	if (root != nullptr)
	{
		root->accept(pfinder);
	}
	if (pfinder.roadmap_.empty())
	{
		std::transform(targets.begin(), targets.end(),
		std::back_inserter(outs),
		[&](const TensptrT& target)
		{
			return funcs.get_const_zero(*target);
		});
		return outs;
	}

	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	RefMapT owners = track_ownrefs(TensptrsT{root});
	GraphStat stat;
	GraphIndex indexer;
	root->accept(stat);
	root->accept(indexer);

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
				return indexer.at(a) < indexer.at(b);
			}
			return astr < bstr;
		}
		return aheight < bheight;
	});

	for (iFunctor* tens : tovisits)
	{
		auto& prevs = pfinder.at(tens).at(target_label).args_;
		auto op = std::static_pointer_cast<iFunctor>(owners[tens].lock());
		TensptrsT children = tens->get_args();
		size_t nchildren = children.size();
		TensMapT<TensptrsT> chain_map;
		for (size_t i : prevs)
		{
			if (i < nchildren && estd::get(
				chain_map, jacobians, children[i].get()))
			{
				for (auto& chain_info : chain_map)
				{
					auto& targ = chain_info.first;
					auto& prev_chains = chain_info.second;
					assert(prev_chains.size() > 0);
					TensptrT prev_chain = prev_chains.size() > 1 ?
						funcs.add(prev_chains) : prev_chains.front();
					jacobians[tens][targ].push_back(
						funcs.jacobian_chain(op, prev_chain, i));
				}
			}
		}
	}

	TensMapT<TensptrsT> root_chain = estd::must_getf(
		jacobians, root.get(),
		"failed to find existing jacobian for %s",
		root->to_string().c_str());
	for (auto target : targets)
	{
		TensptrT der;
		TensptrsT chains;
		if (estd::get(chains, root_chain, target.get()) && chains.size() > 0)
		{
			auto jacobian = chains.size() == 1 ? chains.front() : funcs.add(chains);
			// given NxM matrix and target tensor of M cardinality
			// accept derivative of root wrt target by mapped by jacobian
			der = funcs.dejacobianize(jacobian, target);
		}
		else
		{
			der = funcs.get_const_zero(*target);
		}
		outs.push_back(der);
	}
	return outs;
}

TensptrsT derive (
	TensptrT root,
	const TensptrsT& targets,
	const iDerivativeFuncs& funcs)
{
	size_t n = targets.size();
	TensptrsT out;
	out.reserve(n);
	if (root == nullptr)
	{
		std::transform(targets.begin(), targets.end(),
			std::back_inserter(out),
			[&](const TensptrT& target)
			{
				return funcs.get_const_zero(*target);
			});
		return out;
	}

	TensMapT<TensptrsT> grads = {
		{root.get(), {funcs.get_const_one(*root)}}
	};
	TensSetT targset;
	std::transform(targets.begin(), targets.end(),
		std::inserter(targset, targset.end()),
		[](TensptrT target) { return target.get(); });
	partial_derive(grads, {root}, targset, funcs);

	for (auto target : targets)
	{
		TensptrT tens;
		TensptrsT tgrads;
		if (nullptr == target)
		{
			tens = funcs.get_const_zero(*root);
		}
		else if (estd::get(tgrads, grads, target.get()) && tgrads.size() > 0)
		{
			tens = tgrads.size() == 1 ? tgrads.front() : funcs.add(tgrads);
		}
		else
		{
			tens = funcs.get_const_zero(*target);
		}
		out.push_back(tens);
	}
	return out;
}

void partial_derive (TensMapT<TensptrsT>& grads,
	const TensptrSetT& parents,
	const TensSetT& targets,
	const iDerivativeFuncs& funcs)
{
	if (targets.empty())
	{
		return;
	}

	TensSetT parset;
	std::transform(parents.begin(), parents.end(),
		std::inserter(parset, parset.end()),
		[](TensptrT tens) { return tens.get(); });
	TensMapT<std::string> tids;
	for (auto& target : targets)
	{
		if (nullptr != target)
		{
			if (estd::has(parset, target))
			{
				assert(estd::has(grads, target));
			}
			else
			{
				tids.emplace(target, target_label);
			}
		}
	}
	if (tids.empty())
	{
		return;
	}

	PathFinder pfinder(tids, [](iFunctor& f){ return f.get_args(); });
	multi_visit(pfinder, parents);
	if (pfinder.roadmap_.empty())
	{
		return;
	}

	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	RefMapT owners = track_ownrefs(parents);
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
		auto& nexts = pfinder.at(tens).at(target_label).args_;
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
