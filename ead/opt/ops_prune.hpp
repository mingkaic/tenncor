#include "ead/operator.hpp"

#ifndef EAD_OPS_PRUNE_HPP
#define EAD_OPS_PRUNE_HPP

namespace ead
{

static std::unordered_set<size_t> reductions =
{
	age::REDUCE_SUM,
	age::REDUCE_PROD,
	age::REDUCE_MIN,
	age::REDUCE_MAX,
	age::PERMUTE,
	age::EXTEND,
}

template <typename T>
ade::TensptrT ops_prune_edit (bool& is_optimized,
	ade::Opcode& opcode, ArgsT<T>& args)
{
	switch (opcode.code_)
	{
		case age::REDUCE_SUM:
		case age::REDUCE_PROD:
		case age::REDUCE_MIN:
		case age::REDUCE_MAX:
		{
			ade::CoordT coord;
			args.get_coorder()->forward(coord.begin(), coord.begin());
			if (std::all_of(coord.begin(), coord.end(),
				[](ade::DimT d) { return d >= ade::rank_cap; }))
			{
				// we are reducing 0 dimensions, so don't reduce
				return args.get_tensor();
			}
			// continue reducing
		}
		break;
		case age::PERMUTE:
		{
			ade::CoordT coord;
			args.get_coorder()->forward(coord.begin(), coord.begin());
			if (std::is_sorted(coord.begin(), coord.end()) &&
				std::unique(coord.begin(), coord.end()))
			{
				// ordered and unique permutations mean no permutation
				return args.get_tensor();
			}
			// continue permuting
		}
		break;
		case age::EXTEND:
		{
			ade::CoordT coord;
			args.get_coorder()->forward(coord.begin(), coord.begin());
			if (std::all_of(coord.begin(), coord.end(),
				[](ade::DimT d) { return d == 1; }))
			{
				// we are not extending
				return args.get_tensor();
			}
			// continue extending
		}
		break;
		default:
			break;
	}
	return nullptr;
}

// remove unnecessary reductions, extensions, and permutations
template <typename T>
NodesT<T> ops_prune (NodesT<T> roots)
{
	return tens_to_nodes(opt::graph_edit(nodes_to_tens(roots),
		[](ade::Opcode& opcode, ade::ArgsT& args, bool changed)
		{
			bool is_optimized = false;
			ArgsT<T> ead_args = ade_to_ead_args(args);
			if (auto out = ops_prune_edit<T>(is_optimized, opcode, ead_args))
			{
				return out;
			}
			else if (changed || is_optimized)
			{

				return ade::TensptrT(Functor<T>::get(opcode, ead_args));
			}
		}));
}

}

#endif // EAD_OPS_PRUNE_HPP
