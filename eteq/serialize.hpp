///
/// serialize.hpp
/// eteq
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "eteq/make.hpp"

#include "onnx/save.hpp"
#include "onnx/load.hpp"

#ifndef ETEQ_SERIALIZE_HPP
#define ETEQ_SERIALIZE_HPP

namespace eteq
{

void save_graph (onnx::GraphProto& pb_graph, teq::TensptrsT roots,
	const onnx::TensIdT& identified = {});

teq::TensptrsT load_graph (onnx::TensptrIdT& identified_tens,
	const onnx::GraphProto& pb_graph);

}

#endif // ETEQ_SERIALIZE_HPP
