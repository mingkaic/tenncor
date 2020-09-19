
#include "tenncor/serial/oxsvc/service.hpp"

#ifdef DISTRIB_OX_SERVICE_HPP

namespace distr
{

namespace ox
{

void merge_graph_proto (onnx::GraphProto& outgraph,
	const onnx::GraphProto& ingraph)
{
	const auto& innodes = ingraph.node();
	for (const auto& innode : innodes)
	{
		outgraph.add_node()->MergeFrom(innode);
	}

	const auto& ininits = ingraph.initializer();
	for (const auto& ininit : ininits)
	{
		outgraph.add_initializer()->MergeFrom(ininit);
	}

	const auto& sininits = ingraph.sparse_initializer();
	for (const auto& sininit : sininits)
	{
		outgraph.add_node()->MergeFrom(sininit);
	}

	const auto& inins = ingraph.input();
	for (const auto& inin : inins)
	{
		outgraph.add_input()->MergeFrom(inin);
	}

	const auto& invis = ingraph.value_info();
	for (const auto& invi : invis)
	{
		outgraph.add_value_info()->MergeFrom(invi);
	}

	const auto& inqas = ingraph.quantization_annotation();
	for (const auto& inqa : inqas)
	{
		outgraph.add_quantization_annotation()->MergeFrom(inqa);
	}
}

}

error::ErrptrT register_oxsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<io::DistrIOService*>(svcs.get_obj(io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("opsvc requires iosvc already registered");
	}
	svcs.add_entry<ox::DistrSerializeService>(ox::oxsvc_key,
		[&](){ return new ox::DistrSerializeService(cfg, iosvc); });
	return nullptr;
}

ox::DistrSerializeService& get_oxsvc (iDistrManager& manager)
{
	auto svc = manager.get_service(ox::oxsvc_key);
	if (nullptr == svc)
	{
		global::fatalf("%s service not found in %s",
			ox::oxsvc_key.c_str(), manager.get_id().c_str());
	}
	return static_cast<ox::DistrSerializeService&>(*svc);
}

}

#endif
