///
/// load.hpp
/// onnx
///
/// Purpose:
/// Define functions for loading teq graph
///

#ifndef ONNX_LOAD_HPP
#define ONNX_LOAD_HPP

#include "internal/onnx/marshal.hpp"

namespace onnx
{

struct iUnmarshFuncs
{
	virtual ~iUnmarshFuncs (void) = default;

	virtual teq::TensptrT unmarsh_leaf (const TensorProto& pb_tens,
		teq::Usage usage, std::string name) const = 0;

	virtual teq::TensptrT unmarsh_leaf (const SparseTensorProto& pb_stens,
		teq::Usage usage, std::string name) const = 0;

	virtual teq::TensptrT unmarsh_func (std::string opname,
		const teq::TensptrsT& children, marsh::Maps&& attrs) const = 0;

	virtual teq::TensptrT unmarsh_layr (std::string opname,
		const teq::TensptrT& root, const teq::TensptrT& child,
		marsh::Maps&& attrs) const = 0;
};

/// Return graph roots mapped to their ids through out available from in graph
teq::TensptrsT load_graph (TensptrIdT& identified_tens,
	const GraphProto& pb_graph, const iUnmarshFuncs& unmarshaler);

}

#endif // ONNX_LOAD_HPP
