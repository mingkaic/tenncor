
#include "tenncor/serial/oxsvc/service.hpp"

#ifdef DISTRIB_OX_SERVICE_HPP

namespace distr
{

namespace ox
{

void merge_graph_proto (onnx::GraphProto& outgraph,
	const onnx::GraphProto& ingraph)
{
	auto& innodes = ingraph.node();
	auto outnodes = outgraph.mutable_node();
	outnodes->MergeFrom(innodes);

	auto& ininits = ingraph.initializer();
	auto outinits = outgraph.mutable_initializer();
	outinits->MergeFrom(ininits);

	auto& sininits = ingraph.sparse_initializer();
	auto soutinits = outgraph.mutable_sparse_initializer();
	soutinits->MergeFrom(sininits);

	auto& inins = ingraph.input();
	auto outins = outgraph.mutable_input();
	outins->MergeFrom(inins);

	auto& invis = ingraph.value_info();
	auto outvis = outgraph.mutable_value_info();
	outvis->MergeFrom(invis);

	auto& inqas = ingraph.quantization_annotation();
	auto outqas = outgraph.mutable_quantization_annotation();
	outqas->MergeFrom(inqas);
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
