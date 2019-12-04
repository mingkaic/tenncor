#include "pbm/load.hpp"

#ifdef PBM_LOAD_HPP

namespace pbm
{

void load_graph (teq::TensptrSetT& roots,
	const tenncor::Graph& pb_graph, tag::TagRegistry& registry,
	LeafUnmarshF unmarshal_leaf, FuncUnmarshF unmarshal_func)
{
	auto pb_nodes = pb_graph.nodes();
	teq::TensptrsT generated_tens;
	for (const tenncor::Node& pb_node : pb_nodes)
	{
		teq::TensptrT tens;
		if (pb_node.has_source())
		{
			const tenncor::Source& pb_leaf = pb_node.source();
			tens = unmarshal_leaf(pb_leaf, pb_node.label());
		}
		else
		{
			const tenncor::Functor& pb_func = pb_node.functor();
			const auto& pb_edges = pb_func.args();
			std::string opname = pb_func.opname();
			EdgesT args;
			for (auto& pb_edge : pb_edges)
			{
				teq::TensptrT ctens = generated_tens[pb_edge.idx()];
				args.emplace(args.end(), std::pair<teq::TensptrT,marsh::Maps>{
					ctens, marsh::Maps()});

				auto& attrs = args.back().second;
				const auto& pb_attrs = pb_edge.attrs();
				unmarshal_attrs(attrs, pb_attrs);

				roots.erase(ctens);
			}
			marsh::Maps fattrs;
			const auto& pb_fattrs = pb_func.attrs();
			unmarshal_attrs(fattrs, pb_fattrs);
			tens = unmarshal_func(opname, args, std::move(fattrs));
		}
		auto& pb_tags = pb_node.tags();
		for (auto& tagpair : pb_tags)
		{
			const std::string& tagkey = tagpair.first;
			auto& taglabels = tagpair.second.labels();
			auto tagr = registry.tagr_by_key(tagkey);
			for (std::string taglabel : taglabels)
			{
				tagr(tens, taglabel);
			}
		}
		generated_tens.push_back(tens);
		roots.emplace(tens);
	}
}

}

#endif
