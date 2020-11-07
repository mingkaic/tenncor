
#ifndef DISTR_OX_UTIL_HPP
#define DISTR_OX_UTIL_HPP

#include "estd/estd.hpp"

#include "tenncor/serial/oxsvc/distr.ox.grpc.pb.h"

namespace distr
{

namespace ox
{

// Topographic map of node id to peer id where node should be deploy to
using TopographyT = types::StrUMapT<std::string>;

enum MERGE_COMMAND
{
	NODE,
	INIT,
	SPARSE_INIT,
	INPUT,
	VALUE_INFO,
	QUANT_ANNOT,
};

// copy everything from ingraph inserting into outgraph
// except for output
void merge_graph_proto (onnx::GraphProto& outgraph,
	const onnx::GraphProto& ingraph,
	const std::unordered_set<MERGE_COMMAND>& ignore = {});

void merge_topograph (TopographyT& outopo,
	const google::protobuf::Map<std::string,std::string>& intopo);

}

}

#endif // DISTR_OX_UTIL_HPP
