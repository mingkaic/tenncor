#include "age/grader.hpp"

#include "llo/shear.hpp"

#ifdef LLO_SHEAR_HPP

namespace llo
{

// todo: move somewhere else
static ade::Tensorptr prune0 (bool& is_zero, ade::iFunctor* func,
	std::unordered_set<size_t> zeros, ade::ArgsT args)
{
	is_zero = false;
	ade::OPCODE opcode = (ade::OPCODE) func->get_code().opnum();
	if (false == zeros.empty())
	{
		switch (opcode)
		{
			case ade::COPY:
			case ade::ABS:
			case ade::NEG:
			case ade::SIN:
			case ade::TAN:
			case ade::SQRT:
			case ade::ROUND:
			case ade::MUL:
				is_zero = true;
				return ade::shaped_zero(func->shape());
			case ade::COS:
			case ade::EXP:
				return ade::shaped_one(func->shape());
			case ade::LOG:
				ade::fatal("cannot LOG by zero");
			case ade::POW:
				if (zeros.end() != zeros.find(0))
				{
					is_zero = true;
					return ade::shaped_zero(func->shape());
				}
				// else if zeros.end() != zeros.find(1)
				return ade::shaped_one(func->shape());
			case ade::ADD:
			{
				ade::ArgsT filtered;
				for (size_t i = 0, n = args.size(); i < n; ++i)
				{
					if (zeros.end() == zeros.find(i))
					{
						filtered.push_back(args[i]);
					}
				}
				if (filtered.empty())
				{
					is_zero = true;
					return ade::shaped_zero(func->shape());
				}
				return ade::Functor::get(MAKE_CODE(ade::ADD), filtered);
			}
			case ade::SUB:
				if (2 == zeros.size())
				{
					is_zero = true;
					return ade::shaped_zero(func->shape());
				}
				else if (zeros.end() != zeros.find(0))
				{
					return ade::Functor::get(MAKE_CODE(ade::NEG), {args[1]});
				}
				// else if zeros.end() != zeros.find(1)
				return args[0].tensor_;
			case ade::DIV:
				if (zeros.end() != zeros.find(1))
				{
					ade::fatal("cannot DIV by zero");
				}
				// else if 0 == zeros.front()
				is_zero = true;
				return ade::shaped_zero(func->shape());
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
				ade::fatal("cannot prune unknown opcode");
		}
	}
	return ade::Functor::get(ade::make_code(opcode), args);
}

DataNode zero_prune (DataNode root)
{
	// assert that context will be unaffected by prune,
	// since source will never be touched
	ade::PathFinder finder(ade::Tensor::SYMBOLIC_ZERO.get());
	root.tensor_->accept(finder);
	auto& pathmap = finder.parents_;
	if (pathmap.empty()) // not path to zero or root is not a parent
	{
		return root;
	}
	ade::GraphStat stat;
	root.tensor_->accept(stat);
	// grab the intersection of stat.funcs_ and pathmap
	std::list<ade::iFunctor*> parents;
	std::transform(pathmap.begin(), pathmap.end(),
		std::back_inserter(parents),
		[](std::pair<ade::iTensor*,std::unordered_set<size_t>> parent)
		{
			return static_cast<ade::iFunctor*>(parent.first);
		});
	parents.sort(
		[&](ade::iTensor* a, ade::iTensor* b)
		{
			return stat.graphsize_[a] < stat.graphsize_[b];
		});

	// each proceeding node in parents list is closer to SYMBOLIC_ZERO
	// start pruning according to each parent node in order
	std::unordered_map<ade::iTensor*,ade::Tensorptr> mapping;
	std::unordered_map<ade::iTensor*,bool> zeromap;
	for (ade::iFunctor* func : parents)
	{
		ade::ArgsT children = func->get_children();
		std::unordered_set<size_t> indices = pathmap[func];
		for (auto it = indices.begin(), et = indices.end(); it != et;)
		{
			ade::MappedTensor& child = children[*it];
			ade::iTensor* tens = child.tensor_.get();
			auto zit = zeromap.find(tens);
			assert(zeromap.end() != zit );
			if (false == zit->second)
			{
				it = indices.erase(it);
			}
			else
			{
				++it;
			}
			auto mit = mapping.find(tens);
			assert(mapping.end() != mit);
			child.tensor_ = mit->second;
		}
		bool is_zero = false;
		mapping.emplace(func, prune0(is_zero, func, indices, children));
		zeromap.emplace(func, is_zero);
	}
	auto it = mapping.find(root.tensor_.get());
	if (mapping.end() == it)
	{
		ade::fatal("something went wrong"); // todo: probably add context?
	}

	return DataNode{root.ctx_, it->second};
}

}

#endif
