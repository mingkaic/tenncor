///
/// save.hpp
/// onnx
///
/// Purpose:
/// Define functions for saving teq graph
///

#include <list>

#include "teq/traveler.hpp"
#include "teq/ifunctor.hpp"

#include "onnx/marshal.hpp"

#ifndef ONNX_SAVE_HPP
#define ONNX_SAVE_HPP

namespace onnx
{

using LeafMarshF = std::function<std::string(TensorProto*,const teq::iLeaf*)>;

void save_graph (GraphProto& out,
	teq::TensptrsT roots, LeafMarshF marshal_leaf);

}

#endif // ONNX_SAVE_HPP
