///
/// save.hpp
/// onnx
///
/// Purpose:
/// Define functions for saving teq graph
///

#ifndef ONNX_SAVE_HPP
#define ONNX_SAVE_HPP

#include "onnx/marshal.hpp"

namespace onnx
{

struct iMarshFuncs
{
	virtual ~iMarshFuncs (void) = default;

	virtual size_t get_typecode (const teq::iTensor& tens) const = 0;

	virtual void marsh_leaf (
		TensorProto& pb_tens, const teq::iLeaf& leaf) const = 0;
};

void save_graph (GraphProto& pb_graph, teq::TensT roots,
	const iMarshFuncs& marshaler, const TensIdT& identified = {});

void save_graph (GraphProto& pb_graph, teq::TensptrsT roots,
	const iMarshFuncs& marshaler, const TensIdT& identified = {});

}

#endif // ONNX_SAVE_HPP
