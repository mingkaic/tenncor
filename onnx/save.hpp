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

struct iMarshFuncs
{
	virtual ~iMarshFuncs (void) = default;

	virtual size_t get_typecode (const teq::iTensor& tens) const = 0;

	virtual void marsh_leaf (
		TensorProto& pb_tens, const teq::iLeaf& leaf) const = 0;
};

void save_graph (GraphProto& pb_graph, teq::TensptrsT roots,
	const iMarshFuncs& marshaler, const TensIdT& identified = {});

}

#endif // ONNX_SAVE_HPP
