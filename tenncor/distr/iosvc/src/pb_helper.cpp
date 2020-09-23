#include "tenncor/distr/iosvc/pb_helper.hpp"

#ifdef DISTR_IO_PB_HELPER_HPP

namespace distr
{

namespace io
{

DRefptrT node_meta_to_ref (const NodeMeta& meta)
{
	auto tens_id = meta.uuid();
	auto& slist = meta.shape();
	teq::DimsT sdims(slist.begin(), slist.end());
	return std::make_shared<DistrRef>(
		egen::get_type(meta.dtype()),
		teq::Shape(sdims),
		meta.instance(), tens_id);
}

void tens_to_node_meta (NodeMeta& out, const std::string cid,
	const std::string& uuid, const teq::TensptrT& tens)
{
	auto& meta = tens->get_meta();
	auto shape = tens->shape();
	out.set_uuid(uuid);
	out.set_dtype(meta.type_label());
	for (auto it = shape.begin(), et = shape.end(); it != et; ++it)
	{
		out.add_shape(*it);
	}
	if (auto ref = dynamic_cast<iDistrRef*>(tens.get()))
	{
		out.set_instance(ref->cluster_id());
	}
	else
	{
		out.set_instance(cid);
	}
}

}

}

#endif
