
#include "distrib/services/op/service.hpp"

#ifdef DISTRIB_OP_SERVICE_HPP

namespace distr
{

bool process_get_data (
	const op::GetDataRequest& req,
	DataStatesT::iterator& it,
	op::NodeData& reply)
{
	auto id = it->first;
	auto tens = it->second;

	auto& meta = tens->get_meta();
	size_t latest = meta.state_version();

	reply.set_uuid(id);
	reply.set_version(latest);

	void* raw = tens->device().data();
	auto dtype = (egen::_GENERATED_DTYPE) meta.type_code();

	size_t nelems = tens->shape().n_elems();
	google::protobuf::RepeatedField<double> field;
	field.Resize(nelems, 0);
	egen::type_convert(field.mutable_data(), raw, dtype, nelems);
	reply.mutable_data()->Swap(&field);
	return true;
}

DistrOpService& get_opsvc (iDistrManager& manager)
{
	auto svc = manager.get_service(opsvc_key);
	if (nullptr == svc)
	{
		global::fatalf("%s service not found in %s",
			opsvc_key.c_str(), manager.get_id().c_str());
	}
	return static_cast<DistrOpService&>(*svc);
}

}

#endif
