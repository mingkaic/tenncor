
#include "tenncor/eteq/opsvc/service.hpp"

#ifdef DISTR_OP_SERVICE_HPP

namespace distr
{

namespace op
{

bool process_get_data (
	const GetDataRequest& req,
	DataStatesT::iterator& it,
	NodeData& reply)
{
	auto id = it->first;
	auto tens = it->second;

	auto& meta = tens->get_meta();
	size_t latest = meta.state_version();

	reply.set_uuid(id);
	reply.set_version(latest);

	void* raw = tens->device().data();
	if (nullptr == raw)
	{
		global::errorf("cannot process tensor %s with null data",
			tens->to_string().c_str());
		return false;
	}

	auto dtype = (egen::_GENERATED_DTYPE) meta.type_code();
	size_t nelems = tens->shape().n_elems();
	google::protobuf::RepeatedField<double> field;
	field.Resize(nelems, 0);
	egen::type_convert(field.mutable_data(), raw, dtype, nelems);
	reply.mutable_data()->Swap(&field);
	return true;
}

}

error::ErrptrT register_opsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<io::DistrIOService*>(svcs.get_obj(io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("opsvc requires iosvc already registered");
	}
	svcs.add_entry<op::DistrOpService>(op::opsvc_key,
		[&]{ return new op::DistrOpService(
			std::make_unique<eigen::Device>(std::numeric_limits<size_t>::max()),
			std::make_unique<eteq::DerivativeFuncs>(), cfg, iosvc); });
	return nullptr;
}

op::DistrOpService& get_opsvc (iDistrManager& manager)
{
	auto svc = manager.get_service(op::opsvc_key);
	if (nullptr == svc)
	{
		global::fatalf("%s service not found in %s",
			op::opsvc_key.c_str(), manager.get_id().c_str());
	}
	return static_cast<op::DistrOpService&>(*svc);
}

}

#endif
