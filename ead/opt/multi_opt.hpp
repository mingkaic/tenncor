#include "ead/opt/const_merge.hpp"
#include "ead/opt/ops_merge.hpp"
#include "ead/opt/one_prune.hpp"
#include "ead/opt/zero_prune.hpp"
#include "ead/opt/nodes.hpp"

#ifndef EAD_MULTI_OPT_HPP
#define EAD_MULTI_OPT_HPP

namespace ead
{

template <typename T>
using EditFuncT = std::function<ade::TensptrT(bool&,ade::Opcode&,ArgsT<T>&)>;

template <typename T>
NodesT<T> multi_optimize (NodesT<T> roots,
	std::vector<EditFuncT<T>> edits = {
		const_merge_edit<T>,
		zero_prune_edit<T>,
		one_prune_edit<T>,
		ops_merge_edit<T>,
	})
{
	return tens_to_nodes(opt::graph_edit(nodes_to_tens(roots),
		[&edits](ade::Opcode& opcode, ade::ArgsT& args, bool changed)
		{
			bool is_optimized = false;
			ArgsT<T> ead_args = ade_to_ead_args(args);
			for (auto edit : edits)
			{
				if (auto out = edit(is_optimized, opcode, ead_args))
				{
					return out;
				}
			}
			else if (changed || is_optimized)
			{

				return ade::TensptrT(Functor<T>::get(opcode, ead_args));
			}
		}));
}

}

#endif // EAD_MULTI_OPT_HPP
