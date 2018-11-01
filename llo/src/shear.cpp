#include "llo/shear.hpp"

#ifdef LLO_SHEAR_HPP

namespace llo
{

// todo: move somewhere else
static ade::Tensorptr prune0 (bool& is_zero, ade::iFunctor* func,
	std::vector<bool> zeros, ade::ArgsT args)
{
	is_zero = false;
	ade::OPCODE opcode = func->get_code();
	switch (opcode)
	{
		case ade::COPY:
		case ade::ABS:
		case ade::NEG:
		case ade::SIN:
		case ade::TAN:
		case ade::SQRT:
		case ade::ROUND:
			if (zeros[0])
			{
				is_zero = true;
				return ade::shaped_zero(func->shape());
			}
			break;
		case ade::COS:
		case ade::EXP:
			if (zeros[0])
			{
				return ade::shaped_one(func->shape());
			}
			break;
		case ade::LOG:
			if (zeros[0])
			{
				ade::fatal("cannot LOG by zero");
			}
			break;
		case ade::POW:
			if (zeros[0])
			{
				is_zero = true;
				return ade::shaped_zero(func->shape());
			}
			else if (zeros[1])
			{
				return ade::shaped_one(func->shape());
			}
			break;
		case ade::ADD:
		{
			ade::ArgsT filtered;
			for (size_t i = 0, n = args.size(); i < n; ++i)
			{
				if (false == zeros[i])
				{
					filtered.push_back(args[i]);
				}
			}
			if (filtered.empty())
			{
				is_zero = true;
				return ade::shaped_zero(func->shape());
			}
			return ade::Functor::get(ade::ADD, filtered);
		}
		case ade::MUL:
			if (std::any_of(zeros.begin(), zeros.end(),
				[](bool b) { return b; }))
			{
				is_zero = true;
				return ade::shaped_zero(func->shape());
			}
			break;
		case ade::SUB:
			if (zeros[0] && zeros[1])
			{
				is_zero = true;
				return ade::shaped_zero(func->shape());
			}
			else if (zeros[0])
			{
				return ade::Functor::get(ade::NEG, {args[1]});
			}
			else if (zeros[1])
			{
				return args[0].tensor_;
			}
			break;
		case ade::DIV:
			if (zeros[1])
			{
				ade::fatal("cannot DIV by zero");
			}
			else if (zeros[0])
			{
				is_zero = true;
				return ade::shaped_zero(func->shape());
			}
		case ade::MIN:
		case ade::MAX:
		case ade::EQ:
		case ade::NE:
		case ade::LT:
		case ade::GT:
		case ade::RAND_BINO:
		case ade::RAND_UNIF:
		case ade::RAND_NORM:
			break;
		default:
			return ade::Tensorptr(nullptr);
	}
	return ade::Functor::get(opcode, args);
}

DataNode zero_prune (DataNode root)
{
	// assert that context will be unaffected by prune,
	// since source will never be touched
	ade::PathFinder finder(ade::Tensor::SYMBOLIC_ZERO.get());
	root.tensor_->accept(finder);
	if (finder.parents_.empty()) // not path to zero or root is not a parent
	{
		return root;
	}
	GraphStat stat({root});
	// grab the intersection of stat.funcs_ and finder.parents_
	std::list<ade::iFunctor*> parents;
	std::copy_if(stat.funcs_.begin(), stat.funcs_.end(), std::back_inserter(parents),
		[&](ade::iFunctor* func)
		{
			return finder.parents_.end() != finder.parents_.find(func);
		});
	// each proceeding node in parents list is closer to SYMBOLIC_ZERO
	// start pruning according to each parent node in order
	std::unordered_map<ade::iTensor*,ade::Tensorptr> mapping;
	std::unordered_map<ade::iTensor*,bool> zeromap;
	for (ade::iFunctor* func : parents)
	{
		ade::ArgsT children = func->get_children();
		std::vector<bool> paints = finder.parents_[func];
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			ade::iTensor* tens = children[i].tensor_.get();
			auto zit = zeromap.find(tens);
			if (zeromap.end() != zit)
			{
				paints[i] = paints[i] && zit->second;
			}
			auto mit = mapping.find(tens);
			if (mapping.end() != mit)
			{
				children[i].tensor_ = mit->second;
			}
		}
		bool is_zero = false;
		mapping.emplace(func, prune0(is_zero, func, paints, children));
		zeromap.emplace(func, is_zero);
	}
	auto it = mapping.find(root.tensor_.get());
	if (mapping.end() == it)
	{
		ade::fatal("something went wrong"); // todo: probably add context?
	}

	return DataNode{stat.global_ctx_, it->second};
}

}

#endif
