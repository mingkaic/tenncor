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
};

inline bool is_identity (ade::CoordptrT coorder)
{
	if (ade::identity == coorder || nullptr == coorder)
	{
		return true;
	}
	bool id = false;
	coorder->access([&id](const ade::MatrixT& m)
	{
		id = true;
		for (uint8_t i = 0; id && i < ade::mat_dim; ++i)
		{
			for (uint8_t j = 0; id && j < ade::mat_dim; ++j)
			{
				id = id && m[i][j] == (i == j);
			}
		}
	});
	return id;
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
			if (is_identity(args[0].get_shaper()))
			{
				// we are reducing 0 dimensions, so don't reduce
				return args[0].get_tensor();
			}
			// continue reducing
		}
		break;
		case age::PERMUTE:
		{
			ade::CoordT coord;
			args[0].get_coorder()->forward(coord.begin(), coord.begin());
			if (std::is_sorted(coord.begin(), coord.end()) &&
				std::unique(coord.begin(), coord.end()))
			{
				// ordered and unique permutations mean no permutation
				return args[0].get_tensor();
			}
			// continue permuting
		}
		break;
		case age::EXTEND:
		{
			ade::CoordT coord;
			args[0].get_coorder()->forward(coord.begin(), coord.begin());
			if (std::all_of(coord.begin(), coord.end(),
				[](ade::DimT d) { return d == 1; }))
			{
				// we are not extending
				return args[0].get_tensor();
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
	return tens_to_nodes<T>(opt::graph_edit(nodes_to_tens<T>(roots),
		[](ade::Opcode& opcode, ade::ArgsT& args, bool changed) -> ade::TensptrT
		{
			bool is_optimized = false;
			ArgsT<T> ead_args = ade_to_ead_args<T>(args);
			if (auto out = ops_prune_edit<T>(is_optimized, opcode, ead_args))
			{
				return out;
			}
			if (changed || is_optimized)
			{

				return ade::TensptrT(Functor<T>::get(opcode, ead_args));
			}
			return nullptr;
		}));
}

}

#endif // EAD_OPS_PRUNE_HPP
