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

struct iUnmarshFuncs
{
	virtual ~iUnmarshFuncs (void) = default;

	virtual teq::TensptrT unmarsh_leaf (const TensorProto& pb_tens,
		teq::Usage usage, std::string name) = 0;

	virtual teq::TensptrT unmarsh_func (std::string opname,
		const teq::TensptrsT& children, marsh::Maps&& attrs) = 0;

	virtual teq::TensptrT unmarsh_layr (std::string opname,
		const teq::TensptrsT& roots, const teq::TensptrsT& children,
		marsh::Maps&& attrs) = 0;
};

/// Return graph info through out available from in graph
void load_graph (teq::TensptrsT& roots,
	const GraphProto& pb_graph, iUnmarshFuncs& unmarshaler,
	std::unordered_map<std::string,teq::TensptrT> created_tens = {});

}

#endif // ONNX_LOAD_HPP
