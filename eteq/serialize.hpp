///
/// serialize.hpp
/// eteq
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "eteq/make.hpp"

#include "pbm/save.hpp"
#include "pbm/load.hpp"

#include "onnx/save.hpp"
#include "onnx/load.hpp"

#ifndef ETEQ_SERIALIZE_HPP
#define ETEQ_SERIALIZE_HPP

namespace eteq
{

void save_graph (onnx::GraphProto& pb_graph, teq::TensptrsT roots);

void load_graph (teq::TensptrsT& roots, const onnx::GraphProto& pb_graph);

pbm::TensMapIndicesT save_graph (
	tenncor::Graph& out, teq::TensptrsT roots,
	tag::TagRegistry& registry = tag::get_reg());

void load_graph (teq::TensptrSetT& roots,
	const tenncor::Graph& pb_graph,
	tag::TagRegistry& registry = tag::get_reg());

}

#endif // ETEQ_SERIALIZE_HPP
