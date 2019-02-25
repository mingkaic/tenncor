#include "opt/graph_edit.hpp"

#include "ead/generated/codes.hpp"

#include "ead/opt/nodes.hpp"

#ifndef EAD_CONST_MERGE_HPP
#define EAD_CONST_MERGE_HPP

namespace ead
{

template <typename T>
ade::TensptrT const_merge_edit (bool& is_optimized,
	ade::Opcode& opcode, ArgsT<T>& args)
{
	ArgsT<T> cargs;
	std::copy_if(args.begin(), args.end(), std::back_inserter(cargs),
		[](FuncArg<T>& arg)
		{
			return nullptr != dynamic_cast<Constant<T>*>(
				arg.get_tensor().get());
		});
	if (cargs.size() == args.size())
	{
		auto temp = Functor<T>::get(opcode, cargs);
		temp->update();
		ade::TensptrT out(Constant<T>::get(temp->data(), temp->shape()));
		delete temp;
		return out;
	}
	return nullptr;
}

template <typename T>
NodesT<T> const_merge (NodesT<T> roots)
{
	return tens_to_nodes<T>(opt::graph_edit(nodes_to_tens<T>(roots),
		[](ade::Opcode& opcode, ade::ArgsT& args, bool changed) -> ade::TensptrT
		{
			bool is_optimized = false;
			ArgsT<T> ead_args = ade_to_ead_args<T>(args);
			if (auto out = const_merge_edit<T>(is_optimized, opcode, ead_args))
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

#endif // EAD_CONST_MERGE_HPP
