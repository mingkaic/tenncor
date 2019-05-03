#include "ead/operator.hpp"

#ifndef EAD_OPS_PRUNE_HPP
#define EAD_OPS_PRUNE_HPP

namespace ead
{

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
			if (ade::is_identity(args[0].get_shaper().get()))
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
		// todo: move this MUL/DIV optimization over to a 'subgraph recognition' optimization technique
		case age::MUL:
		{
			auto a = args[0].get_tensor().get();
			auto b = args[1].get_tensor().get();
			if (auto af = dynamic_cast<ade::iFunctor*>(a))
			{
				if (af->get_opcode().code_ == age::DIV)
				{
					auto achildren = af->get_children();
					if (achildren[1].get_tensor().get() == b)
					{
						return achildren[0].get_tensor();
					}
				}
			}
			if (auto bf = dynamic_cast<ade::iFunctor*>(b))
			{
				if (bf->get_opcode().code_ == age::DIV)
				{
					auto bchildren = bf->get_children();
					if (bchildren[1].get_tensor().get() == a)
					{
						return bchildren[0].get_tensor();
					}
				}
			}
			// continue
			if (auto ac = dynamic_cast<Constant<T>*>(a))
			{
				double* ptr = (double*) ac->data();
				if (std::all_of(ptr, ptr + ac->shape().n_elems(),
					[](double d) { return -1 == d; }))
				{
					opcode = ade::Opcode{"NEG", age::NEG};
					args = {args[1]};
					is_optimized = true;
				}
			}
			if (auto bc = dynamic_cast<Constant<T>*>(b))
			{
				double* ptr = (double*) bc->data();
				if (std::all_of(ptr, ptr + bc->shape().n_elems(),
					[](double d) { return -1 == d; }))
				{
					opcode = ade::Opcode{"NEG", age::NEG};
					args = {args[0]};
					is_optimized = true;
				}
			}
		}
		break;
		case age::DIV:
		{
			auto num = args[0].get_tensor().get();
			auto numerator = dynamic_cast<ade::iFunctor*>(num);
			if (nullptr != numerator &&
				numerator->get_opcode().code_ == age::MUL)
			{
				auto denom = args[1].get_tensor().get();
				auto numchildren = numerator->get_children();
				if (numchildren[0].get_tensor().get() == denom)
				{
					return numchildren[1].get_tensor();
				}
				if (numchildren[1].get_tensor().get() == denom)
				{
					return numchildren[0].get_tensor();
				}
			}
			// continue
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
	return tens_to_nodes<T>(::opt::graph_edit(nodes_to_tens<T>(roots),
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
