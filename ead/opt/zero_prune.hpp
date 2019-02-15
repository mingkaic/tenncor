///
///	zero_prune.hpp
///	ead
///
///	Purpose:
///	Define ead zero pruning functions
///

#include "opt/graph_edit.hpp"

#include "ead/variable.hpp"

#include "ead/opt/nodes.hpp"

#ifndef EAD_ZERO_PRUNE_HPP
#define EAD_ZERO_PRUNE_HPP

namespace ead
{

template <typename T>
static bool const_is_zero (Constant<T>* cst)
{
	double* ptr = cst->get_tensmap()->data();
	return std::all_of(ptr, ptr + cst->shape().n_elems(),
		[](double d) { return 0 == d; });
}

template <typename T>
ade::TensptrT zero_prune_edit (bool& is_optimized,
	ade::Opcode& opcode, ArgsT<T>& args)
{
	size_t n = args.size();
	bool has_zero = false;
	std::vector<bool> is_zero(n, false);
	for (size_t i = 0; i < n; ++i)
	{
		auto cst = dynamic_cast<Constant<T>*>(args[i].get_tensor().get());
		is_zero[i] = nullptr != cst && cst->is_const() && const_is_zero(cst);
		has_zero = has_zero || is_zero[i];
	}
	if (has_zero)
	{
		switch (opcode.code_)
		{
			case age::ABS:
			case age::NEG:
			case age::SIN:
			case age::TAN:
			case age::SQRT:
			case age::ROUND:
			case age::MUL:
				return ade::TensptrT(Constant<T>::get(0, args[0].shape()));
			case age::COS:
			case age::EXP:
				return ade::TensptrT(Constant<T>::get(1, args[0].shape()));
			case age::LOG:
				logs::fatal("cannot LOG by zero");
			case age::POW:
				if (is_zero[0])
				{
					return ade::TensptrT(Constant<T>::get(0, args[0].shape()));
				}
				// else if is_zero[1]
				return ade::TensptrT(Constant<T>::get(1, args[1].shape()));
			case age::ADD:
				if (is_zero[0])
				{
					return args[1].get_tensor();
				}
				// else if is_zero[1]
				return args[0].get_tensor();
			case age::SUB:
				if (is_zero[0] && is_zero[1])
				{
					return ade::TensptrT(Constant<T>::get(0, args[0].shape()));
				}
				else if (is_zero[0])
				{
					is_optimized = true;
					opcode = ade::Opcode{"NEG", age::NEG};
					args = {args[1]};
					return nullptr;
				}
				// else if is_zero[1]
				return args[0].get_tensor();
			case age::DIV:
				if (is_zero[1])
				{
					logs::fatal("cannot DIV by zero");
				}
				// else if is_zero[0]
				return ade::TensptrT(Constant<T>::get(0, args[0].shape()));
			case age::MIN:
			case age::MAX:
			case age::EQ:
			case age::NEQ:
			case age::LT:
			case age::GT:
			case age::RAND_UNIF:
			case age::MATMUL:
				break;
			default:
				logs::fatalf("cannot zero prune unknown opcode \"%s\"",
					opcode.name_.c_str());
		}
	}
	return nullptr;
}

/// Return tree that prunes zero branches in input according to OPCODE
/// For example, add(x, 0) is converted to simply x, while mul(x, 0) is 0
template <typename T>
NodesT<T> zero_prune (NodesT<T> roots)
{
	return tens_to_nodes(opt::graph_edit(nodes_to_tens(roots),
		[](ade::Opcode& opcode, ade::ArgsT& args, bool changed)
		{
			bool is_optimized = false;
			ArgsT<T> ead_args = ade_to_ead_args(args);
			if (auto out = zero_prune_edit<T>(is_optimized, opcode, ead_args))
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

#endif // EAD_ZERO_PRUNE_HPP
