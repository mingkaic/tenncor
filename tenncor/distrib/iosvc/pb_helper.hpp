
#ifndef DISTRIB_IO_PB_HELPER_HPP
#define DISTRIB_IO_PB_HELPER_HPP

#include "tenncor/distrib/reference.hpp"
#include "tenncor/distrib/iosvc/distr.io.grpc.pb.h"

namespace distr
{

namespace io
{

DRefptrT node_meta_to_ref (const NodeMeta& meta);

void tens_to_node_meta (NodeMeta& out, const std::string cid,
	const std::string& uuid, const teq::TensptrT& tens);

}

}

#endif // DISTRIB_IO_PB_HELPER_HPP
