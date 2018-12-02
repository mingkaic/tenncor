#include "bwd/grader.hpp"

#ifdef BWD_GRADER_HPP

namespace age
{

void Grader::visit (ade::iFunctor* func)
{
	if (func == target_)
	{
		derivatives_.emplace(func, rules_->data(1, target_->shape()));
		return;
	}

	ade::PathFinder finder(target_);
	func->accept(finder);

	auto& pathmap = finder.parents_;
	// no path to wrt
	if (pathmap.empty())
	{
		derivatives_.emplace(func, rules_->data(0, target_->shape()));
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

	std::unordered_map<const ade::iTensor*,ade::TensT> grads = {
		{func, {rules_->data(1, func->shape())}},
	};
	for (ade::iFunctor* parent : parents)
	{
		ade::Opcode opcode = parent->get_opcode();
		ade::TensT& gargs = grads[parent];
		ade::TensptrT bwd(gargs.size() > 1 ? gargs[0] :
			ade::TensptrT(ade::Functor::get(rules_->sum_opcode(), to_args(gargs))));

		auto& grad_indices = pathmap[parent];
		ade::ArgsT children = parent->get_children();
		size_t nchildren = children.size();
		// for each painted child, calculate dThis/dChild
		// go through grads in order
		std::list<size_t> ordered(grad_indices.begin(), grad_indices.end());
		ordered.sort();
		for (size_t i : ordered)
		{
			ade::TensT args;
			ade::MappedTensor& child = children[i];
			ade::CoordPtrT mapper(child.mapper_->reverse());
			for (size_t j = 0; j < nchildren; ++j)
			{
				ade::TensptrT& tens = children[j].tensor_;
				if (j == i)
				{
					args.push_back(tens);
				}
				else
				{
					ade::CoordPtrT toshape(
						children[j].mapper_->forward(*mapper));
					args.push_back(ade::TensptrT(ade::Functor::get(rules_->sum_opcode(),
						{{toshape, tens}})));
				}
			}
			// pass down forward-gradient pair
			ade::TensptrT grad(rules_->grad_rule(opcode.code_, args, i));

			// apply chain rule
			grads[child.tensor_.get()].push_back(ade::TensptrT(ade::Functor::get(
				rules_->prod_opcode(), {
					{ade::identity, grad},
					{ade::identity, ade::TensptrT(ade::Functor::get(rules_->sum_opcode(),
						{{mapper, bwd}}))},
				})));
		}
	}
	derivatives_.emplace(func, ade::TensptrT(
		ade::Functor::get(rules_->sum_opcode(),
			to_args(grads[target_]))));
}

ade::ArgsT to_args (ade::TensT tens)
{
	ade::ArgsT args;
	std::transform(tens.begin(), tens.end(), std::back_inserter(args),
		[](ade::TensptrT& ten)
		{
			return ade::MappedTensor{ade::identity, ten};
		});
	return args;
}

}

#endif
