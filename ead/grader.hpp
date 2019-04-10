///
/// grader.hpp
/// bwd
///
/// Purpose:
/// Define grader traveler to build partial derivative equations
///

#include <list>

#include "ead/generated/api.hpp"
#include "ead/generated/grader.hpp"

#include "ead/constant.hpp"
#include "ead/edge.hpp"

#ifndef EAD_GRADER_HPP
#define EAD_GRADER_HPP

namespace ead
{

/// Derive root with respect to target and optimized
template <typename T>
NodeptrT<T> derive (NodeptrT<T> root, NodeptrT<T> target)
{
	EdgesT edges;
	return derive_with_edges(edges, root, target);
}

/// Derive root with respect to target and optimized
template <typename T>
NodeptrT<T> derive_with_edges (EdgesT& edges, NodeptrT<T> root, NodeptrT<T> target)
{
	if (root->get_tensor() == target->get_tensor())
	{
		return make_constant_scalar((T) 1, target->shape());
	}

	ade::iTensor* target_tens = target->get_tensor().get();
	ade::PathFinder finder(target_tens);
	root->get_tensor()->accept(finder);

	auto& pathmap = finder.parents_;
	// no path to wrt
	if (pathmap.empty())
	{
		return make_constant_scalar((T) 0, target->shape());
	}
	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	std::string source_str = root->get_tensor()->to_string();
	ade::GraphStat stat;
	root->get_tensor()->accept(stat);

	std::list<ade::iFunctor*> parents;
	std::transform(pathmap.begin(), pathmap.end(),
		std::back_inserter(parents),
		[](std::pair<ade::iTensor*,std::unordered_set<size_t>> parent)
		{
			return static_cast<ade::iFunctor*>(parent.first);
		});
	parents.sort(
		[&](ade::iFunctor* a, ade::iFunctor* b)
		{
			return stat.graphsize_[a] > stat.graphsize_[b];
		});

	auto root_grad = make_constant_scalar((T) 1, root->shape());
	std::unordered_map<const ade::iTensor*,NodesT<T>> grads = {
		{root->get_tensor().get(), {root_grad}}
	};
	edges.push_back(Edge{
		root->get_tensor(),
		root_grad->get_tensor(),
		ade::Opcode{
			fmts::sprintf("GRADIENT_%s_%s",
				source_str.c_str(), source_str.c_str()),
			GRADIENT
		}
	});
	for (ade::iFunctor* parent : parents)
	{
		NodesT<T>& gargs = grads[parent];
		NodeptrT<T> bwd = gargs[0];
		for (size_t i = 1, n = gargs.size(); i < n; ++i)
		{
			bwd = age::add(bwd, gargs[i]);
		}

		auto& grad_indices = pathmap[parent];
		ade::ArgsT children = parent->get_children();
		size_t nchildren = children.size();
		// assert: all nnary-children use identity mapping,
		// so no children-arg is direct mapping
		ade::TensT args(nchildren);
		std::transform(children.begin(), children.end(), args.begin(),
			[](ade::FuncArg& arg)
			{
				return arg.get_tensor();
			});
		// for each painted child, calculate dThis/dChild
		// go through grads in order
		std::list<size_t> ordered(grad_indices.begin(), grad_indices.end());
		ordered.sort();
		for (size_t i : ordered)
		{
			auto grad_step = age::chain_rule<T>(parent, bwd, args, i);
			grads[args[i].get()].push_back(grad_step);
			edges.push_back(Edge{
				args[i],
				grad_step->get_tensor(),
				ade::Opcode{
					fmts::sprintf("GRADIENT_%s_%s",
						source_str.c_str(),
						args[i]->to_string().c_str()),
					GRADIENT
				}
			});
		}
	}
	NodesT<T>& outargs = grads[target_tens];
	NodeptrT<T> out = outargs[0];
	for (size_t i = 1, n = outargs.size(); i < n; ++i)
	{
		out = age::add(out, outargs[i]);
	}
	return out;
}

}

#endif // EAD_GRADER_HPP
