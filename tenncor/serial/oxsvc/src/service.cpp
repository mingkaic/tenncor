
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
		auto outnode = outgraph.add_node();
		outnode->set_name(innode.name());
		outnode->set_op_type(innode.op_type());
		outnode->set_domain(innode.domain());
		outnode->set_doc_string(innode.doc_string());
		outnode->mutable_input()->MergeFrom(innode.input());
		outnode->mutable_output()->MergeFrom(innode.output());
		outnode->mutable_attribute()->MergeFrom(innode.attribute());
	}

	const auto& ininits = ingraph.initializer();
	for (const auto& ininit : ininits)
	{
		auto outinit = outgraph.add_initializer();
		outinit->MergeFrom(ininit);
	}

	const auto& sininits = ingraph.sparse_initializer();
	auto soutinits = outgraph.mutable_sparse_initializer();
	soutinits->MergeFrom(sininits);

	const auto& inins = ingraph.input();
	auto outins = outgraph.mutable_input();
	outins->MergeFrom(inins);

	const auto& invis = ingraph.value_info();
	auto outvis = outgraph.mutable_value_info();
	outvis->MergeFrom(invis);

	const auto& inqas = ingraph.quantization_annotation();
	for (const auto& inqa : inqas)
	{
		auto outqa = outgraph.add_quantization_annotation();
		outqa->set_tensor_name(inqa.tensor_name());
		outqa->mutable_quant_parameter_tensor_names()->MergeFrom(
			inqa.quant_parameter_tensor_names());
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
