///
/// serialize.hpp
/// eteq
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "eteq/make.hpp"

#include "onnx/onnx.hpp"

#ifndef ETEQ_SERIALIZE_HPP
#define ETEQ_SERIALIZE_HPP

namespace eteq
{

const std::string app_name = "tenncor";
const std::string app_version = "1.0.0";
const std::string eteq_dom = "com.tenncor.eteq";

void save_model (onnx::ModelProto& pb_model, teq::TensT roots,
	const onnx::TensIdT& identified = {});

void save_model (onnx::ModelProto& pb_model, teq::TensptrsT roots,
	const onnx::TensIdT& identified = {});

teq::TensptrsT load_model (onnx::TensptrIdT& identified_tens,
	const onnx::ModelProto& pb_model);

}

#endif // ETEQ_SERIALIZE_HPP
