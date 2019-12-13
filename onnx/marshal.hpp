#include "teq/ileaf.hpp"
#include "teq/ifunctor.hpp"

#include "marsh/objs.hpp"

#ifndef ONNX_MARSHAL_HPP
#define ONNX_MARSHAL_HPP

#include "onnx/onnx.pb.h"

namespace onnx
{

using PbAttrsT = ::google::protobuf::RepeatedPtrField<AttributeProto>;

using TensorT = std::pair<int32_t,teq::Shape>;

using AnnotationsT = std::unordered_map<std::string,std::string>;

const std::string leafname_key = "TENSOR_NAME";

const std::string leafconst_key = "IS_IMMUTABLE";

void marshal_attrs (PbAttrsT& out, const teq::iFunctor* func);

void marshal_tensorshape (TensorShapeProto& out,
	const teq::ShapeSignature& shape);

void marshal_io (ValueInfoProto& out, const teq::ShapeSignature& shape);

void marshal_annotation (TensorAnnotation& out, const teq::iLeaf& leaf);

void unmarshal_attrs (marsh::Maps& out, const PbAttrsT& pb_attrs);

teq::Shape unmarshal_shape (const TensorShapeProto& shape);

teq::Shape unmarshal_shape (const TensorProto& tens);

std::unordered_map<std::string,TensorT> unmarshal_io (
	const google::protobuf::RepeatedPtrField<ValueInfoProto>& values);

std::unordered_map<std::string,AnnotationsT> unmarshal_annotation (
	const google::protobuf::RepeatedPtrField<TensorAnnotation>& as);

}

#endif // ONNX_MARSHAL_HPP
