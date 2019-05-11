///
///	one_prune.hpp
///	ead
///
///	Purpose:
///	Define ead one pruning functions
///

#include "opt/graph_edit.hpp"

#include "ead/constant.hpp"

#include "ead/opt/nodes.hpp"

#ifndef EAD_ONE_PRUNE_HPP
#define EAD_ONE_PRUNE_HPP

namespace ead
{

template <typename T>
static bool const_is_one (Constant<T>* cst)
{
	T* ptr = (T*) cst->data();
	return std::all_of(ptr, ptr + cst->shape().n_elems(),
		[](T d) { return 1 == d; });
}

template <typename T>
ade::TensptrT one_prune_edit (bool& is_optimized,
	ade::Opcode& opcode, ArgsT<T>& args)
{
	size_t n = args.size();
	bool has_one = false;
	std::vector<bool> is_one(n, false);
	for (size_t i = 0; i < n; ++i)
	{
		auto cst = dynamic_cast<Constant<T>*>(
			args[i].get_tensor().get());
		is_one[i] = nullptr != cst && cst->is_const() && const_is_one(cst);
		has_one = has_one || is_one[i];
	}
	if (has_one)
	{
		switch (opcode.code_)
		{
			case age::ABS:
			case age::SQRT:
			case age::ROUND:
				return ade::TensptrT(Constant<T>::get_scalar(1, args[0].shape()));
			case age::LOG:
				return ade::TensptrT(Constant<T>::get_scalar((T) 0, args[0].shape()));
			case age::POW:
				if (is_one[0])
				{
					return ade::TensptrT(Constant<T>::get_scalar(1, args[0].shape()));
				}
				// else if is_one[1]
				return args[0].get_tensor();
			case age::MUL:
				if (is_one[0])
				{
					return args[1].get_tensor();
				}
				// else if is_one[1]
				return args[0].get_tensor();
			case age::DIV:
				if (is_one[1])
				{
					return args[0].get_tensor();
				}
				// else if is_one[0]
				break;
			case age::NEG:
			case age::SIN:
			case age::COS:
			case age::TAN:
			case age::EXP:
			case age::ADD:
			case age::SUB:
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
				logs::fatalf("cannot one prune unknown opcode \"%s\"",
					opcode.name_.c_str());
		}
	}
	return nullptr;
}

/// Return tree that prunes one branches in input according to OPCODE
/// For example, mul(x, 1) is converted to simply x, while abs(1) is 1
template <typename T>
NodesT<T> one_prune (NodesT<T> roots)
{
	return tens_to_nodes<T>(::opt::graph_edit(nodes_to_tens<T>(roots),
		[](ade::Opcode& opcode, ade::ArgsT& args, bool changed) -> ade::TensptrT
		{
			bool is_optimized = false;
			ArgsT<T> ead_args = ade_to_ead_args<T>(args);
			if (auto out = one_prune_edit<T>(is_optimized, opcode, ead_args))
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

#endif // EAD_ONE_PRUNE_HPP