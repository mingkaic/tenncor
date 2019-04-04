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

// derive bottom-up
template <typename T>
NodeptrT<T> derive_bu (EdgesT& edges,
	ade::TensptrT root, ade::TensptrT wrt) // todo: replace derive
{
	ade::PathFinder finder(wrt);
	root->accept(finder);

	auto& pathmap = finder.parents_;
	// no path to wrt
	if (pathmap.empty())
	{
		return make_constant_scalar((T) 0, wrt->shape());
	}
	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	ade::GraphStat stat;
	ade::OwnerTracker tracker;
	root->accept(stat);
	root->accept(tracker);

	std::list<ade::iFunctor*> parents;
	std::transform(pathmap.begin(), pathmap.end(),
		std::back_inserter(parents),
		[](std::pair<ade::iTensor*,std::unordered_set<size_t>> parent)
		{
			return static_cast<ade::iFunctor*>(parent.first);
		});
	// parents go from wrt to root
	parents.sort(
		[&](ade::iFunctor* a, ade::iFunctor* b)
		{
			return stat.graphsize_[a] < stat.graphsize_[b];
		});

	// todo: reuse this
	auto wrt_grad = make_constant_scalar((T) 1, wrt->shape());
	std::unordered_map<const ade::iTensor*,NodeptrT<T>> jacobians =
		{{wrt.get(), wrt_grad}};
	edges.push_back(Edge{wrt, wrt_grad, ade::Opcode{"GRADIENT", GRADIENT}});

	for (ade::iFunctor* parent : parents)
	{
		auto& child_indices = pathmap[parent];
		// todo: implement
	}

	return jacobians[wrt.get()]; // root.get()];
}

/// Traveler to obtain derivative of accepted node with respect to target
template <typename T>
struct Grader final : public ade::iTraveler
{
	Grader (const ade::iTensor* target) :
		target_(target)
	{
		if (target_ == nullptr)
		{
			logs::fatal("cannot derive with respect to null");
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		derivatives_.emplace(leaf,
			make_constant_scalar((T) (leaf == target_), target_->shape()));
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (func == target_)
		{
			derivatives_.emplace(func, make_constant_scalar((T) 1, target_->shape()));
			return;
		}

		ade::PathFinder finder(target_);
		func->accept(finder);

		auto& pathmap = finder.parents_;
		// no path to wrt
		if (pathmap.empty())
		{
			derivatives_.emplace(func, make_constant_scalar((T) 0, target_->shape()));
			return;
		}
		// else there exists a path to wrt
		// using pathfinder, breadth first traverse from this to wrt
		ade::GraphStat stat;
		func->accept(stat);

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

		std::unordered_map<const ade::iTensor*,NodesT<T>> grads = {
			{func, {make_constant_scalar((T) 1, func->shape())}},
		};
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
				edges_.push_back(Edge{
					args[i],
					grad_step->get_tensor(),
					ade::Opcode{
						"GRADIENT",
						GRADIENT
					}
				});
			}
		}
		NodesT<T>& finalgargs = grads[target_];
		NodeptrT<T> finalgarg = finalgargs[0];
		for (size_t i = 1, n = finalgargs.size(); i < n; ++i)
		{
			finalgarg = age::add(finalgarg, finalgargs[i]);
		}
		derivatives_.emplace(func, finalgarg);
	}

	EdgesT edges_;

	/// Target of tensor all visited nodes are derived with respect to
	const ade::iTensor* target_;

	/// Map forward root node to derivative root
	std::unordered_map<const ade::iTensor*,NodeptrT<T>> derivatives_;
};

/// Derive root with respect to target and optimized
template <typename T>
NodeptrT<T> derive (NodeptrT<T> root, NodeptrT<T> target)
{
	auto rooten = root->get_tensor();
	Grader<T> grader(target->get_tensor().get());
	rooten->accept(grader);
	auto it = grader.derivatives_.find(rooten.get());
	assert(grader.derivatives_.end() != it);
	return it->second;
}

/// Derive root with respect to target and optimized
template <typename T>
NodeptrT<T> derive (EdgesT& edges, NodeptrT<T> root, NodeptrT<T> target)
{
	auto rooten = root->get_tensor();
	Grader<T> grader(target->get_tensor().get());
	rooten->accept(grader);
	auto it = grader.derivatives_.find(rooten.get());
	assert(grader.derivatives_.end() != it);
	edges.insert(edges.end(),
		grader.edges_.begin(), grader.edges_.end());
	return it->second;
}

}

#endif // EAD_GRADER_HPP
