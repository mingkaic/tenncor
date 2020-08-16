
#include "distrib/reference.hpp"

#include "tenncor/distrib/services/io/distr.io.grpc.pb.h"

#ifndef DISTRIB_IO_PB_HELPER_HPP
#define DISTRIB_IO_PB_HELPER_HPP

namespace distr
{

DRefptrT node_meta_to_ref (const io::NodeMeta& meta);

void tens_to_node_meta (io::NodeMeta& out, const std::string cid,
	const std::string& uuid, const teq::TensptrT& tens);

}

#endif // DISTRIB_IO_PB_HELPER_HPP