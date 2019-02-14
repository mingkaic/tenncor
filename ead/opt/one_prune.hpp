///
///	one_prune.hpp
///	ead
///
///	Purpose:
///	Define ead one pruning functions
///

#include "opt/graph_edit.hpp"

#include "ead/variable.hpp"

#ifndef EAD_ONE_PRUNE_HPP
#define EAD_ONE_PRUNE_HPP

namespace ead
{

template <typename T>
static bool const_is_one (Constant<T>* cst)
{
	double* ptr = cst->get_tensmap()->data();
	return std::all_of(ptr, ptr + cst->shape().n_elems(),
		[](double d) { return 1 == d; });
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
		auto cst = dynamic_cast<Constant<T>*>(args[i].get_tensor().get());
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
				return ade::TensptrT(Constant<T>::get(1, args[0].shape()));
			case age::LOG:
				return ade::TensptrT(Constant<T>::get(0, args[0].shape()));
			case age::POW:
				if (is_one[0])
				{
					return ade::TensptrT(Constant<T>::get(1, args[0].shape()));
				}
				// else if is_one[1]
				if (ade::identity == args[0].get_coorder())
				{
					return args[0].get_tensor();
				}
				is_optimized = true;
				opcode = ade::Opcode{"SUM", age::SUM};
				args = {args[0]};
				return nullptr;
			case age::PROD:
			{
				ade::ArgsT filtered;
				for (size_t i = 0, n = args.size(); i < n; ++i)
				{
					if (false == is_one[i])
					{
						filtered.push_back(args[i]);
					}
				}
				if (filtered.empty())
				{
					return ade::TensptrT(Constant<T>::get(1, args[0].shape()));
				}
				is_optimized = true;
				opcode = ade::Opcode{"PROD", age::PROD};
				args = filtered;
				return nullptr;
			}
			case age::DIV:
				if (is_one[1])
				{
					if (ade::identity == args[0].get_coorder())
					{
						return args[0].get_tensor();
					}
					is_optimized = true;
					opcode = ade::Opcode{"SUM", age::SUM};
					args = {args[0]};
					return nullptr;
				}
				// else if is_one[0]
				break;
			case age::NEG:
			case age::SIN:
			case age::COS:
			case age::TAN:
			case age::EXP:
			case age::SUM:
			case age::SUB:
			case age::MIN:
			case age::MAX:
			case age::EQ:
			case age::NEQ:
			case age::LT:
			case age::GT:
			case age::RAND_UNIF:
			case age::RAND_NORM:
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
	return tens_to_nodes(opt::graph_edit(nodes_to_tens(roots),
		[](ade::Opcode& opcode, ade::ArgsT& args, bool changed)
		{
			bool is_optimized = false;
			ArgsT<T> ead_args = ade_to_ead_args(args);
			if (auto out = one_prune_edit<T>(is_optimized, opcode, ead_args))
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

#endif // EAD_ONE_PRUNE_HPP
