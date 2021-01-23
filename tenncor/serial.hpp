///
/// serial.hpp
/// tenncor
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#ifndef TENNCOR_SERIAL_HPP
#define TENNCOR_SERIAL_HPP

#include "tenncor/distr.hpp"
#include "tenncor/serial/oxsvc/oxsvc.hpp"

namespace tcr
{

distr::ox::TopographyT save_model (
	onnx::ModelProto& pb_model,
	const eteq::ETensorsT& roots,
	const onnx::TensptrIdT& identified = {});

eteq::ETensorsT load_model (
	onnx::TensptrIdT& identified_tens,
	const onnx::ModelProto& pb_model,
	const global::CfgMapptrT& ctx = global::context(),
	const distr::ox::TopographyT& topography =
		distr::ox::TopographyT{});

distr::ox::TopographyT save_model (
	onnx::ModelProto& pb_model,
	const global::CfgMapptrT& ctx = global::context());

eteq::ETensorsT load_model (
	global::CfgMapptrT& ctx,
	const onnx::ModelProto& pb_model,
	const distr::ox::TopographyT& topography =
		distr::ox::TopographyT{});

}

#endif // TENNCOR_SERIAL_HPP
