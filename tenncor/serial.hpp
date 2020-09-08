///
/// serial.hpp
/// tenncor
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#ifndef TENNCOR_SERIAL_HPP
#define TENNCOR_SERIAL_HPP

#include "tenncor/serial/serial.hpp"

#include "tenncor/distr.hpp"

namespace tcr
{

void save_model (onnx::ModelProto& pb_model,
	const eteq::ETensorsT& roots,
	const onnx::TensIdT& identified = {});

eteq::ETensorsT load_model (onnx::TensptrIdT& identified_tens,
	const onnx::ModelProto& pb_model,
	const global::CfgMapptrT& ctx = global::context());

}

#endif // TENNCOR_SERIAL_HPP
