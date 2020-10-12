#include "tenncor/serial/oxsvc/util.hpp"

#ifdef DISTR_OX_UTIL_HPP

namespace distr
{

namespace ox
{

void merge_graph_proto (onnx::GraphProto& outgraph,
	const onnx::GraphProto& ingraph,
	const std::unordered_set<MERGE_COMMAND>& ignore)
{
	if (false == estd::has(ignore, NODE))
	{
		const auto& innodes = ingraph.node();
		for (const auto& innode : innodes)
		{
			outgraph.add_node()->MergeFrom(innode);
		}
	}

	if (false == estd::has(ignore, INIT))
	{
		const auto& ininits = ingraph.initializer();
		for (const auto& ininit : ininits)
		{
			outgraph.add_initializer()->MergeFrom(ininit);
		}
	}

	if (false == estd::has(ignore, SPARSE_INIT))
	{
		const auto& sininits = ingraph.sparse_initializer();
		for (const auto& sininit : sininits)
		{
			outgraph.add_node()->MergeFrom(sininit);
		}
	}

	if (false == estd::has(ignore, INPUT))
	{
		const auto& inins = ingraph.input();
		for (const auto& inin : inins)
		{
			outgraph.add_input()->MergeFrom(inin);
		}
	}

	if (false == estd::has(ignore, VALUE_INFO))
	{
		const auto& invis = ingraph.value_info();
		for (const auto& invi : invis)
		{
			outgraph.add_value_info()->MergeFrom(invi);
		}
	}

	if (false == estd::has(ignore, QUANT_ANNOT))
	{
		const auto& inqas = ingraph.quantization_annotation();
		for (const auto& inqa : inqas)
		{
			outgraph.add_quantization_annotation()->MergeFrom(inqa);
		}
	}
}

void merge_topograph (TopographyT& outopo,
	const google::protobuf::Map<std::string,std::string>& intopo)
{
	for (const auto& in : intopo)
	{
		outopo.emplace(in.first, in.second);
	}
}

}

}

#endif
