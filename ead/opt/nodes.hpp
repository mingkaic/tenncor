#include "ead/grader.hpp"

#ifndef EAD_NODES_HPP
#define EAD_NODES_HPP

namespace ead
{

template <typename T>
inline ade::TensT nodes_to_tens (NodesT<T> nodes)
{
	ade::TensT tens(nodes.size());
	std::transform(nodes.begin(), nodes.end(), tens.begin(),
		[](NodeptrT< T>& node)
		{
			return node->get_tensor();
		});
	return tens;
}

template <typename T>
inline NodesT<T> tens_to_nodes (ade::TensT tens)
{
	std::vector<NodeptrT< T>> nodes(tens.size());
	std::transform(tens.begin(), tens.end(), nodes.begin(),
		[](ade::TensptrT& tens)
		{
			return ead::to_node<T>(tens);
		});
	return nodes;
}

template <typename T>
inline ArgsT<T> ade_to_ead_args (ade::ArgsT args)
{
	ArgsT<T> ead_args;
	std::transform(args.begin(), args.end(), std::back_inserter(ead_args),
		[](ade::FuncArg& arg)
		{
			return FuncArg<T>{
				to_node<T>(arg.get_tensor()),
				arg.get_shaper(),
				arg.map_io(),
				arg.get_coorder(),
			};
		});
	return ead_args;
}

}

#endif // EAD_NODES_HPP
