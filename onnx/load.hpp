///
/// load.hpp
/// onnx
///
/// Purpose:
/// Define functions for loading teq graph
///

#include "teq/traveler.hpp"
#include "teq/ifunctor.hpp"

#include "onnx/marshal.hpp"

#ifndef ONNX_LOAD_HPP
#define ONNX_LOAD_HPP

namespace onnx
{

using LeafUnmarshF = std::function<teq::TensptrT(const TensorProto&,bool)>;

using EdgesT = std::vector<std::pair<teq::TensptrT,teq::Shape>>;

using FuncUnmarshF = std::function<teq::TensptrT(
	std::string,const EdgesT&,marsh::Maps&&)>;

/// Return graph info through out available from in graph
void load_graph (teq::TensptrSetT& roots, const GraphProto& pb_graph,
	LeafUnmarshF unmarshal_leaf, FuncUnmarshF unmarshal_func);

}

#endif // ONNX_LOAD_HPP
