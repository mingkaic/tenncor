///
/// grad_def.hpp
/// teq
///
/// Purpose:
/// Define gradient builder interface for building derivatives
///

#include <list>

#include "teq/traveler.hpp"

#ifndef TEQ_GRAD_DEF_HPP
#define TEQ_GRAD_DEF_HPP

namespace teq
{

/// Define manditory definitions required for tensor differentiation
/// For some graph F(G(x)), chain rule for calculating dF/dx is
/// defined in the following order:
/// 1. calcualte dF/dG => F local derivative and
/// derivative of super composition (supcomp_grad for G)
/// 2. calculate dG/dx => G local derivative
/// 3. chain dF/dG (supcomp_grad) and dG/dx (local_der)
/// This top-down approach updates tensor shape information such
/// that output derivative dF/dx has the shape of x
struct iGradientBuilder
{
	virtual ~iGradientBuilder (void) = default;

	/// Let op be functor F with arguments args
	/// Return derivative of F wrt args[arg_idx]
	virtual TensptrT local_derivative (FuncptrT op, size_t arg_idx) const = 0;

	/// Let op be functor F with arguments args, and
	/// local_der is derivative of F wrt one of args (say x)
	/// Let supcomp_grad be defined as dG/dF
	/// where G is some super-functor using F
	/// Return derivative G wrt to arg x by applying chain rule
	virtual TensptrT chain_rule (FuncptrT op, const TensptrT& local_der,
		TensptrT supcomp_grad, size_t arg_idx) const = 0;

	/// Return tensor representing 1 constant
	virtual TensptrT get_const_one (Shape shape) const = 0;

	/// Return tensor representing 0 constant
	virtual TensptrT get_const_zero (Shape shape) const = 0;

	/// Return functor representing lhs + rhs
	virtual TensptrT add (TensptrT& lhs, TensptrT& rhs) const = 0;

	/// Return derivative of root with respect to target
	TensptrT derive (TensptrT root, TensptrT target) const
	{
		if (target == nullptr)
		{
			return get_const_zero(teq::Shape());
		}

		if (root == nullptr)
		{
			return get_const_zero(target->shape());
		}

		if (root == target)
		{
			return get_const_one(target->shape());
		}

		PathFinder finder(target.get());
		root->accept(finder);

		auto& pathmap = finder.parents_;
		// no path to wrt
		if (pathmap.empty())
		{
			return get_const_zero(target->shape());
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
			{root.get(), {get_const_one(root->shape())}}
		};
		for (iFunctor* parent : parents)
		{
			TensptrsT& gargs = grads[parent];
			TensptrT bwd = gargs[0];
			for (size_t i = 1, n = gargs.size(); i < n; ++i)
			{
				bwd = add(bwd, gargs[i]);
			}

			auto& grad_indices = pathmap[parent];
			CEdgesT children = parent->get_children();
			size_t nchildren = children.size();
			// assert: all nnary-children use identity mapping,
			// so no children-arg is direct mapping
			TensptrsT args;
			args.reserve(nchildren);
			std::transform(children.begin(), children.end(),
				std::back_inserter(args),
				[](const iEdge& arg)
				{
					return arg.get_tensor();
				});
			// for each painted child, calculate dThis/dChild
			// go through grads in order
			std::list<size_t> ordered(grad_indices.begin(), grad_indices.end());
			ordered.sort();
			for (size_t i : ordered)
			{
				auto parent_ptr = std::static_pointer_cast<iFunctor>(
					owners[parent].lock());
				auto local = local_derivative(parent_ptr, i);
				auto grad_step = chain_rule(parent_ptr, local, bwd, i);
				grads[args[i].get()].push_back(grad_step);
			}
		}
		TensptrsT& outargs = grads[target.get()];
		TensptrT out = outargs[0];
		for (size_t i = 1, n = outargs.size(); i < n; ++i)
		{
			out = add(out, outargs[i]);
		}
		return out;
	}
};

}

#endif // TEQ_GRAD_DEF_HPP
