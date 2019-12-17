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

using LeafUnmarshF = std::function<teq::TensptrT(
	const TensorProto&,teq::Usage,std::string)>;

using FuncUnmarshF = std::function<teq::TensptrT(
	std::string,const teq::TensptrsT&,marsh::Maps&&)>;

/// Return graph info through out available from in graph
void load_graph (teq::TensptrsT& roots, const GraphProto& pb_graph,
	LeafUnmarshF unmarshal_leaf, FuncUnmarshF unmarshal_func);

// void load_model (teq::LayerptrT layer, const ModelProto& pb_model,
// 	LeafUnmarshF unmarshal_leaf, FuncUnmarshF unmarshal_func);

}

#endif // ONNX_LOAD_HPP
