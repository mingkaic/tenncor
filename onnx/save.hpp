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

namespace layr
{

using SignsT = std::vector<teq::ShapeSignature>;

// struct iLayer
// {
// 	virtual ~iLayer (void) = default;

// 	/// Return deep copy of this layer with prefixed label
// 	iLayer* clone (std::string label_prefix = "") const
// 	{
// 		return this->clone_impl(label_prefix);
// 	}

// 	/// Return input value of the expected input (first dimension)
// 	virtual SignsT get_inputs (void) const = 0;

// 	/// Return output value of the expected output (first dimension)
// 	virtual SignsT get_outputs (void) const = 0;

// 	/// Return all internal tensors representing the layer
// 	virtual teq::TensptrsT get_contents (void) const = 0;

// 	/// Return the root of the graph that connects input with internal tensors
// 	virtual teq::TensptrT connect (teq::TensptrT input) const = 0;

// protected:
// 	virtual iLayer* clone_impl (std::string label_prefix) const = 0;
// };

}

namespace onnx
{

using LeafMarshF = std::function<void(TensorProto&,const teq::iLeaf&)>;

void save_graph (GraphProto& pb_graph,
	teq::TensptrsT roots, LeafMarshF marshal_leaf);

// void save_layer (GraphProto& pb_graph, layr::iLayer layer);

// void save_model (ModelProto& pb_model, layr::iLayer layer);

}

#endif // ONNX_SAVE_HPP
